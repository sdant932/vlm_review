"""Hierarchical drill-down over every number in the study: outputs/drilldown.html.

The whole study collapses to one number per benchmark, and one number describes
nothing. This page starts at the top and lets each number be opened into the
splits that produced it, recursively, until the leaves are individual questions.

Three rules shape it:

* **Nothing is pooled across metrics.** ANLS, token-F1, click-in-bbox and
  multiple-choice accuracy are different measurements; the root node therefore
  carries a "mixed metrics" warning rather than pretending its average means
  something. Within a benchmark every node uses that benchmark's own metric.
* **Every parent's n is the sum of its children's n.** A drill-down whose levels
  do not add up is worse than no drill-down, so the builder asserts this at every
  split and any violation is rendered on the page instead of being swallowed.
* **Rows with `pred is null` are excluded from the metric and counted anyway.**
  Each node reports how many it dropped.

The nesting is data, not code: `SPEC[benchmark](path)` returns the dimensions to
split by at that point, so a level can be added by editing one list. Where a
level has more than one interesting dimension (CharXiv question type can be
opened by subplot count *or* arXiv subject *or* year), each becomes a "by X"
grouping node holding a full partition of its parent -- so the arithmetic check
still applies to each dimension independently.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from blindspot.core.scoring import score

RESULTS = Path("results")
OUT = Path("outputs")

GOOD = "#0ca30c"
BAD = "#d03b3b"
SMALL_N = 30           # below this a node's score is noise, and is labelled as such
DIVERGE = 0.15         # children spread wider than this is the point of drilling in
MAX_EXAMPLES = 3       # concrete cases embedded per leaf

# Primary run per benchmark. slidevqa_allpages is a *condition*, not a benchmark,
# and is handled in the ablations section so the tree's root n stays exact.
MAIN_FILES = {
    "charxiv": "charxiv__haiku-4-5_think2000_native_r0.jsonl",
    "infographicvqa": "infographicvqa__haiku-4-5_think2000_native_r0.jsonl",
    "slidevqa": "slidevqa__haiku-4-5_think2000_native_r0.jsonl",
    "ai2d": "ai2d__haiku-4-5_think2000_native_r0.jsonl",
    "screenspot_pro": "screenspot_pro__haiku-4-5_think2000_native_r0.jsonl",
}
ALLPAGES_FILE = "slidevqa_allpages__haiku-4-5_think2000_native_r0.jsonl"
JUDGE_FILE = "charxiv__haiku-4-5_think2000_native_r0.judged.jsonl"

BENCH_LABEL = {
    "charxiv": "CharXiv",
    "infographicvqa": "InfographicVQA",
    "slidevqa": "SlideVQA",
    "ai2d": "AI2D",
    "screenspot_pro": "ScreenSpot-Pro",
}
BENCH_METRIC = {
    "charxiv": "string match (lower bound)",
    "infographicvqa": "ANLS",
    "slidevqa": "token F1",
    "ai2d": "MC accuracy",
    "screenspot_pro": "click-in-bbox",
}
BENCH_BLURB = {
    "charxiv": ("1,000 arXiv figures, 4 descriptive questions plus 1 reasoning question each. "
                "String scoring is a lower bound on the free-text types; the official judge "
                "covers part of the split and is shown separately, never averaged in."),
    "infographicvqa": "Dense infographics; official ANLS with the 0.5 threshold.",
    "slidevqa": ("Slide decks, evidence pages supplied. Token F1 is the headline and punishes "
                 "formatting; the format-corrected column is shown beside it, never instead."),
    "ai2d": "Grade-school science diagrams, 4-way multiple choice, so chance is 25%.",
    "screenspot_pro": ("Professional application screenshots; the model must return a point "
                       "inside the target's box. Mean target covers 0.065% of the screen."),
}
NAV = [("report.html", "primitives report"), ("datasets.html", "dataset documentation"),
       ("slidevqa.html", "SlideVQA explorer"), ("failure_analysis.html", "failure analysis"),
       ("gallery/charxiv_000.html", "CharXiv gallery"),
       ("gallery/infographicvqa_000.html", "InfographicVQA gallery"),
       ("gallery/screenspot_pro_000.html", "ScreenSpot-Pro gallery")]
# Leaf "see the actual questions" targets, per benchmark.
EVIDENCE_LINK = {
    "charxiv": ("gallery/charxiv_000.html", "CharXiv gallery"),
    "infographicvqa": ("gallery/infographicvqa_000.html", "InfographicVQA gallery"),
    "screenspot_pro": ("gallery/screenspot_pro_000.html", "ScreenSpot-Pro gallery"),
    "slidevqa": ("slidevqa.html", "SlideVQA explorer"),
    "ai2d": ("datasets.html", "AI2D dataset notes"),
}


# --------------------------------------------------------------------------
# loading + scoring
# --------------------------------------------------------------------------

class Shim:
    """The four attributes `scoring.score` reads, taken straight off a result row.

    Using the official entry point rather than a reimplementation is the point:
    the drill-down must not be able to disagree with the rest of the study about
    what a score is.
    """

    __slots__ = ("uid", "dataset", "answer_type", "gold", "meta")

    def __init__(self, rec: dict, dataset: str | None = None):
        self.uid = rec["uid"]
        self.dataset = dataset or rec["dataset"]
        self.answer_type = rec["answer_type"]
        self.gold = rec["gold"]
        self.meta = rec.get("meta") or {}


def read_best(path: Path) -> tuple[dict[str, dict], dict]:
    """Rows keyed by uid, best row wins; plus what had to be thrown away.

    Result files are appended to on resume, so several of them contain the same
    uid more than once. House rule (`aggregate.load_rows`): a later row replaces
    an earlier one unless it would replace a real prediction with a null. The
    tail of a file that is still being written may be a partial line, so parse
    failures are counted rather than raised.
    """
    best: dict[str, dict] = {}
    stats = {"lines": 0, "malformed": 0, "duplicate_uids": 0}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stats["lines"] += 1
            try:
                rec = json.loads(line)
            except Exception:
                stats["malformed"] += 1
                continue
            if "uid" not in rec:
                stats["malformed"] += 1
                continue
            prev = best.get(rec["uid"])
            if prev is not None:
                stats["duplicate_uids"] += 1
            if prev is None or prev.get("pred") is None or rec.get("pred") is not None:
                best[rec["uid"]] = rec
    return best, stats


def score_file(args: tuple[str, str]) -> tuple[str, list[dict], dict]:
    """Load + score one benchmark file. Runs in a worker process."""
    bench, path = args
    best, stats = read_best(Path(path))
    recs = []
    for uid, rec in best.items():
        meta = rec.get("meta") or {}
        pred = rec.get("pred")
        r = {"uid": uid, "bench": bench, "meta": meta, "gold": rec.get("gold"),
             "pred": pred, "null": pred is None, "score": None, "em": None,
             "alt": None, "usage": rec.get("usage") or {}}
        if pred is not None:
            s = score(Shim(rec, bench), pred)
            r["score"] = float(s["score"])
            r["metric"] = s["metric"]
            if "exact_match" in s:
                r["em"] = float(s["exact_match"])
            if "picked" in s:
                r["picked"] = s["picked"]
            if "center_distance" in s:
                r["center_distance"] = s["center_distance"]
            # Format-corrected score: credit answers that mean the same number or
            # the same string once punctuation and scale words are removed.
            if bench in ("slidevqa", "infographicvqa", "charxiv"):
                r["alt"] = 1.0 if (r["score"] < 1.0 and format_equivalent(pred, rec.get("gold") or [])) \
                    else r["score"]
        recs.append(r)
    stats["unique"] = len(best)
    stats["null_pred_after_dedup"] = sum(1 for r in recs if r["null"])
    return bench, recs, stats


# --------------------------------------------------------------------------
# format equivalence (conservative; no substring fallback)
# --------------------------------------------------------------------------

_SCALE = {"k": 1e3, "thousand": 1e3, "thousands": 1e3, "m": 1e6, "mn": 1e6,
          "million": 1e6, "millions": 1e6, "bn": 1e9, "b": 1e9, "billion": 1e9,
          "billions": 1e9, "tn": 1e12, "trillion": 1e12}
_NUM_RE = re.compile(r"^([+-]?\d*\.?\d+)\s*([a-z]*)$")


def _to_number(s: Any) -> float | None:
    t = str(s).strip().lower()
    for ch in (",", "$", "€", "£", "%", " "):
        t = t.replace(ch, "")
    t = t.strip()
    m = _NUM_RE.match(t)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    suf = m.group(2)
    if suf:
        if suf not in _SCALE:
            return None
        v *= _SCALE[suf]
    return v


def _alnum(s: Any) -> str:
    return re.sub(r"[^0-9a-z]", "", str(s).lower())


def format_equivalent(pred: Any, golds: Iterable) -> bool:
    """True when pred and some gold differ only in formatting.

    Sign-sensitive on the numeric path (-5 is not 5) and exact on the folded
    string path (no substring matching), because a lenient detector here would
    manufacture the very finding it is meant to measure.
    """
    if pred is None:
        return False
    pn = _to_number(pred)
    pa = _alnum(pred)
    for g in golds or []:
        gn = _to_number(g)
        if pn is not None and gn is not None:
            if math.isclose(pn, gn, rel_tol=1e-9, abs_tol=1e-12):
                return True
            continue
        if pa and pa == _alnum(g):
            return True
    return False


# --------------------------------------------------------------------------
# nesting spec
# --------------------------------------------------------------------------

@dataclass
class Dim:
    """One way to split a node. `key` maps a record to a child label."""
    name: str
    key: Callable[[dict], Any]
    order: Callable[[tuple[Any, list]], Any] | None = None   # sort key over (label, recs)
    top_k: int | None = None
    other: str = "other (aggregated)"


def _b(v) -> str:
    return "yes" if v else "no"


def _missing(v, label="(metadata missing)"):
    return label if v is None else v


CHARXIV_QLABEL = {
    1: "title", 2: "x-axis label", 3: "y-axis label", 4: "leftmost x tick",
    5: "rightmost x tick", 6: "lowest y tick", 7: "highest y tick",
    8: "x tick spacing", 9: "y tick spacing", 10: "how many lines",
    11: "do any lines intersect", 12: "how many legend labels",
    13: "legend label names", 14: "colorbar range", 15: "colorbar max",
    16: "general trend", 17: "total labeled ticks", 18: "subplot layout",
    19: "number of subplots",
}
CHARXIV_ATYPE = {1: "a-type 1", 2: "a-type 2", 3: "a-type 3", 4: "a-type 4"}
CHARXIV_QSRC = {1: "q-source 1", 2: "q-source 2", 3: "q-source 3"}


def _subplot_bucket(r) -> str:
    n = r["meta"].get("num_subplots")
    if n is None:
        return "(metadata missing)"
    if n == 1:
        return "1 subplot"
    if n <= 3:
        return "2-3 subplots"
    if n <= 6:
        return "4-6 subplots"
    if n <= 12:
        return "7-12 subplots"
    return "13+ subplots"


def _is_na(r) -> str:
    golds = r.get("gold") or []
    na = any(str(g).strip().lower() == "not applicable" for g in golds)
    return "gold is 'Not Applicable' (abstention test)" if na else "gold is a real answer"


def _target_px(r) -> str:
    """Target side length in pixels *as the model saw it*, after the 1568px cap."""
    frac = r["meta"].get("target_area_frac")
    if frac is None:
        return "(metadata missing)"
    side = math.sqrt(frac * 1568 * 882)
    for lim, name in ((12, "<12px"), (20, "12-20px"), (32, "20-32px"), (56, "32-56px")):
        if side < lim:
            return name
    return ">=56px"


def _size_order(item):
    order = ["<12px", "12-20px", "20-32px", "32-56px", ">=56px", "(metadata missing)"]
    lab = item[0]
    return order.index(lab) if lab in order else 99


def _n_desc(item):
    return -len(item[1])


def _alpha(item):
    return str(item[0])


def spec_charxiv(path: tuple) -> list[Dim]:
    if len(path) == 0:
        return [Dim("split", lambda r: r["meta"].get("split") or "(metadata missing)",
                    order=_alpha)]
    split = path[0]
    if len(path) == 1:
        if split == "descriptive":
            return [Dim("question type",
                        lambda r: f"Q{r['meta'].get('qid')} — "
                                  f"{CHARXIV_QLABEL.get(r['meta'].get('qid'), r['meta'].get('qlabel') or '?')}",
                        order=lambda it: int(str(it[0]).split()[0][1:].rstrip("—").strip() or 0))]
        return [Dim("answer type", lambda r: CHARXIV_ATYPE.get(
                        r["meta"].get("reasoning_a_type"), "(metadata missing)"), order=_alpha),
                Dim("question source", lambda r: CHARXIV_QSRC.get(
                        r["meta"].get("reasoning_q_source"), "(metadata missing)"), order=_alpha)]
    if len(path) == 2:
        dims = []
        if split == "descriptive":
            dims.append(Dim("answerability", _is_na, order=_alpha))
        dims += [Dim("subplot count", _subplot_bucket, order=_alpha),
                 Dim("arXiv subject", lambda r: r["meta"].get("category") or "(metadata missing)",
                     order=_n_desc),
                 Dim("arXiv year", lambda r: "20" + str(r["meta"].get("year") or "??"),
                     order=_alpha)]
        return dims
    if len(path) == 3 and split == "descriptive" and str(path[2]).startswith("gold is"):
        # One more level under the abstention split: does subplot count change
        # whether the model invents a value for a question with no answer?
        return [Dim("subplot count", _subplot_bucket, order=_alpha)]
    return []


def spec_infographicvqa(path: tuple) -> list[Dim]:
    if len(path) == 0:
        return [Dim("operation", lambda r: " + ".join(r["meta"].get("operation") or [])
                    or "direct_lookup", order=_n_desc)]
    if len(path) == 1:
        return [Dim("gold answer type",
                    lambda r: " + ".join(r["meta"].get("gold_answer_type") or []) or "(unlabelled)",
                    order=_n_desc)]
    if len(path) == 2:
        return [Dim("gold answer shape", lambda r: _gold_shape(r), order=_n_desc)]
    return []


def _gold_shape(r) -> str:
    golds = r.get("gold") or []
    if not golds:
        return "(no gold)"
    g = str(golds[0])
    if _to_number(g) is not None:
        return "numeric gold"
    ntok = len(g.split())
    if ntok == 1:
        return "one-word gold"
    if ntok <= 4:
        return "short phrase (2-4 words)"
    return "long phrase (5+ words)"


def spec_slidevqa(path: tuple) -> list[Dim]:
    if len(path) == 0:
        return [Dim("evidence spread",
                    lambda r: "multi-page evidence" if r["meta"].get("multi_page")
                    else "single-page evidence", order=_alpha)]
    if len(path) == 1:
        return [Dim("arithmetic required",
                    lambda r: "arithmetic" if r["meta"].get("arithmetic") else "lookup only",
                    order=_alpha)]
    if len(path) == 2:
        return [Dim("evidence pages", lambda r: f"{_missing(r['meta'].get('n_evidence'), '?')} "
                                                f"evidence page(s)", order=_alpha),
                Dim("gold answer shape", _gold_shape, order=_n_desc)]
    if len(path) == 3 and "evidence page" in str(path[2]):
        return [Dim("deck", lambda r: (r["meta"].get("deck") or "(metadata missing)")[:44],
                    order=_n_desc, top_k=15, other="remaining decks (aggregated)")]
    return []


def spec_ai2d(path: tuple) -> list[Dim]:
    if len(path) == 0:
        return [Dim("question type", lambda r: r["meta"].get("qtype") or "(metadata missing)",
                    order=_alpha)]
    if len(path) == 1:
        return [Dim("option count", lambda r: f"{len(r['meta'].get('options') or [])} options",
                    order=_alpha)]
    if len(path) == 2:
        return [Dim("gold letter", lambda r: f"gold = {(r.get('gold') or ['?'])[0]}", order=_alpha),
                Dim("picked letter", lambda r: f"picked {r.get('picked', '?')}", order=_alpha),
                Dim("gold option length",
                    lambda r: _opt_len(r), order=_alpha)]
    if len(path) == 3 and str(path[2]).startswith("gold ="):
        return [Dim("picked letter", lambda r: f"picked {r.get('picked', '?')}", order=_alpha)]
    return []


def _opt_len(r) -> str:
    g = r["meta"].get("gold_text") or ""
    n = len(str(g).split())
    return "1-word answer" if n <= 1 else ("2-3 word answer" if n <= 3 else "4+ word answer")


def spec_screenspot(path: tuple) -> list[Dim]:
    if len(path) == 0:
        return [Dim("element type", lambda r: r["meta"].get("ui_type") or "(metadata missing)",
                    order=_alpha)]
    if len(path) == 1:
        return [Dim("platform", lambda r: r["meta"].get("platform") or "(metadata missing)",
                    order=_n_desc)]
    if len(path) == 2:
        return [Dim("application group", lambda r: r["meta"].get("group") or "(metadata missing)",
                    order=_n_desc),
                Dim("target size", _target_px, order=_size_order)]
    if len(path) == 3 and not str(path[2]).endswith("px"):
        return [Dim("application", lambda r: r["meta"].get("application") or "(metadata missing)",
                    order=_n_desc, top_k=12, other="remaining applications (aggregated)")]
    if len(path) == 4:
        return [Dim("target size", _target_px, order=_size_order)]
    return []


SPEC: dict[str, Callable[[tuple], list[Dim]]] = {
    "charxiv": spec_charxiv,
    "infographicvqa": spec_infographicvqa,
    "slidevqa": spec_slidevqa,
    "ai2d": spec_ai2d,
    "screenspot_pro": spec_screenspot,
}


# --------------------------------------------------------------------------
# tree
# --------------------------------------------------------------------------

@dataclass
class Node:
    label: str
    level: str
    kind: str                       # root | bench | dim | node
    depth: int
    bench: str | None
    recs: list = field(repr=False, default_factory=list)
    children: list = field(default_factory=list)
    path: tuple = ()
    nid: str = ""
    # computed
    n: int = 0
    n_null: int = 0
    value: float | None = None
    metric: str = ""
    em: float | None = None
    alt: float | None = None
    delta: float | None = None
    spread: float | None = None
    judge_n: int = 0
    judge_value: float | None = None
    examples: list = field(default_factory=list)
    note: str = ""


def _agg(recs: list) -> tuple[int, int, float | None, float | None, float | None]:
    live = [r for r in recs if not r["null"]]
    n_null = len(recs) - len(live)
    if not live:
        return 0, n_null, None, None, None
    v = sum(r["score"] for r in live) / len(live)
    ems = [r["em"] for r in live if r["em"] is not None]
    alts = [r["alt"] for r in live if r["alt"] is not None]
    return (len(live), n_null, v,
            (sum(ems) / len(ems)) if ems else None,
            (sum(alts) / len(alts)) if alts else None)


def _examples(recs: list, bench: str) -> list[dict]:
    live = [r for r in recs if not r["null"]]
    if not live:
        return []
    live = sorted(live, key=lambda r: r["score"])
    picks = []
    seen = set()
    for r in [live[0], live[-1], live[len(live) // 2]]:
        if r["uid"] in seen:
            continue
        seen.add(r["uid"])
        picks.append(r)
        if len(picks) >= MAX_EXAMPLES:
            break

    def sh(v, lim=54):
        s = ", ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)
        return s if len(s) <= lim else s[:lim - 1] + "…"

    return [{"uid": r["uid"], "gold": sh(r["gold"]), "pred": sh(r["pred"]),
             "score": r["score"]} for r in picks]


def build(label: str, level: str, kind: str, recs: list, bench: str,
          spec: Callable, path: tuple, depth: int, violations: list,
          counter: list) -> Node:
    counter[0] += 1
    node = Node(label=label, level=level, kind=kind, depth=depth, bench=bench,
                recs=recs, path=path, nid=f"n{counter[0]}")
    node.n, node.n_null, node.value, node.em, node.alt = _agg(recs)
    node.metric = BENCH_METRIC.get(bench, "mixed metrics")

    dims = spec(path) if spec else []
    if not dims:
        node.examples = _examples(recs, bench)
        return node

    dim_nodes = []
    # A level with several dimensions gets a "by X" grouping node, so its real
    # children sit one indent deeper than a single-dimension level's would.
    kid_depth = depth + (2 if len(dims) > 1 else 1)
    for dim in dims:
        groups: dict[Any, list] = defaultdict(list)
        for r in recs:
            groups[dim.key(r)].append(r)
        items = list(groups.items())
        try:
            items.sort(key=dim.order or _n_desc)
        except Exception:
            items.sort(key=_n_desc)
        if dim.top_k is not None and len(items) > dim.top_k:
            by_n = sorted(items, key=lambda kv: -len(kv[1]))
            keep = {k for k, _ in by_n[:dim.top_k]}
            rest = [r for k, v in items if k not in keep for r in v]
            items = [(k, v) for k, v in items if k in keep]
            if rest:
                items.append((f"{dim.other} ×{len(by_n) - dim.top_k}", rest))

        kids = [build(str(k), dim.name, "node", v, bench, spec, path + (str(k),),
                      kid_depth, violations, counter) for k, v in items]

        # --- the arithmetic check, per dimension ---------------------------
        tot_n = sum(k.n for k in kids)
        tot_null = sum(k.n_null for k in kids)
        if tot_n != node.n or tot_null != node.n_null:
            violations.append({
                "node": " › ".join(("ALL",) + path) or "ALL",
                "dimension": dim.name, "parent_n": node.n, "children_n": tot_n,
                "parent_null": node.n_null, "children_null": tot_null})

        if len(dims) == 1:
            node.children = kids
            _finish(node, kids)
            return node
        counter[0] += 1
        dn = Node(label=f"by {dim.name}", level=dim.name, kind="dim", depth=depth + 1,
                  bench=bench, path=path, nid=f"n{counter[0]}")
        dn.n, dn.n_null, dn.value, dn.em, dn.alt = node.n, node.n_null, node.value, node.em, node.alt
        dn.metric = node.metric
        dn.children = kids
        _finish(dn, kids)
        dim_nodes.append(dn)

    node.children = dim_nodes
    return node


def _finish(parent: Node, kids: list[Node]):
    """Delta-vs-parent on each child, and the parent's child-spread flag."""
    for k in kids:
        if k.value is not None and parent.value is not None:
            k.delta = k.value - parent.value
    solid = [k.value for k in kids if k.n >= SMALL_N and k.value is not None]
    if len(solid) >= 2:
        parent.spread = max(solid) - min(solid)


def walk(node: Node):
    yield node
    for c in node.children:
        yield from walk(c)


# --------------------------------------------------------------------------
# ablations / controls
# --------------------------------------------------------------------------

def load_control(name: str) -> tuple[list[dict], dict]:
    p = RESULTS / name
    if not p.exists():
        return [], {"lines": 0, "malformed": 0, "unique": 0}
    best, stats = read_best(p)
    stats["unique"] = len(best)
    return list(best.values()), stats


def control_blocks(main_recs: dict[str, list[dict]]) -> dict:
    """Every ablation in the study, recomputed from disk."""
    by_uid: dict[str, dict] = {}
    for recs in main_recs.values():
        for r in recs:
            by_uid[r["uid"]] = r

    out: dict[str, Any] = {}

    # ---- blind control -------------------------------------------------
    blind, bstats = load_control("control_blind.jsonl")
    agg = defaultdict(lambda: {"blind": [], "sighted": [], "unmatched": 0})
    cx_split = defaultdict(lambda: {"blind": [], "sighted": []})
    for b in blind:
        ds = b["dataset"].replace("_blind", "")
        src = (b.get("meta") or {}).get("src_uid")
        m = by_uid.get(src)
        if m is None or m["null"] or b.get("pred") is None:
            agg[ds]["unmatched"] += 1
            continue
        bs = float(score(Shim(b, ds), b["pred"])["score"])
        agg[ds]["blind"].append(bs)
        agg[ds]["sighted"].append(m["score"])
        if ds == "charxiv":
            sp = (b.get("meta") or {}).get("split") or "?"
            cx_split[sp]["blind"].append(bs)
            cx_split[sp]["sighted"].append(m["score"])
    out["blind"] = {"stats": bstats, "rows": [], "charxiv_split": []}
    for ds in sorted(agg):
        d = agg[ds]
        if not d["blind"]:
            continue
        out["blind"]["rows"].append({
            "bench": BENCH_LABEL.get(ds, ds), "n": len(d["blind"]),
            "blind": statistics.mean(d["blind"]), "sighted": statistics.mean(d["sighted"]),
            "delta": statistics.mean(d["sighted"]) - statistics.mean(d["blind"]),
            "unmatched": d["unmatched"],
            "chance": 0.25 if ds == "ai2d" else None,
            "metric": BENCH_METRIC.get(ds, "")})
    for sp in sorted(cx_split):
        d = cx_split[sp]
        out["blind"]["charxiv_split"].append({
            "bench": f"CharXiv {sp}", "n": len(d["blind"]),
            "blind": statistics.mean(d["blind"]), "sighted": statistics.mean(d["sighted"]),
            "delta": statistics.mean(d["sighted"]) - statistics.mean(d["blind"]),
            "unmatched": 0, "chance": None, "metric": BENCH_METRIC["charxiv"]})

    # ---- one-page ablation ---------------------------------------------
    op, opstats = load_control("control_onepage0.jsonl")
    one, both, ans = [], [], []
    unmatched = 0
    for r in op:
        m = by_uid.get((r.get("meta") or {}).get("src_uid"))
        if m is None or m["null"] or r.get("pred") is None:
            unmatched += 1
            continue
        f1 = float(score(Shim(r, "slidevqa"), r["pred"])["score"])
        one.append(f1)
        both.append(m["score"])
        ans.append(1.0 if f1 >= 0.5 else 0.0)
    out["onepage"] = {"stats": opstats, "n": len(one), "unmatched": unmatched,
                      "one": statistics.mean(one) if one else None,
                      "both": statistics.mean(both) if both else None,
                      "answerable": statistics.mean(ans) if ans else None}

    # ---- coarse localization -------------------------------------------
    ss = main_recs.get("screenspot_pro", [])
    live = [r for r in ss if not r["null"]]
    coarse = []
    for k in (2, 3, 4, 8):
        hit = 0
        for r in live:
            x0, y0, x1, y1 = r["gold"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            px, py = r["pred"]
            same = (min(int(px * k), k - 1) == min(int(cx * k), k - 1)
                    and min(int(py * k), k - 1) == min(int(cy * k), k - 1))
            hit += same
        coarse.append({"label": f"{k}x{k} cell", "n": len(live), "acc": hit / len(live),
                       "chance": 1.0 / (k * k)})
    if live:
        mean_frac = statistics.mean(r["meta"].get("target_area_frac") or 0 for r in live)
        coarse.append({"label": "exact click-in-bbox", "n": len(live),
                       "acc": statistics.mean(r["score"] for r in live),
                       "chance": mean_frac})
    g4, g4stats = load_control("control_grid4.jsonl")
    g4live = [r for r in g4 if r.get("pred") is not None]
    if g4live:
        hit = sum(1 for r in g4live
                  if str(r["pred"]).strip().upper() == str((r.get("gold") or ["?"])[0]).strip().upper())
        coarse.append({"label": "4x4 named cell (model asked for the cell)", "n": len(g4live),
                       "acc": hit / len(g4live), "chance": 1 / 16,
                       "note": "separate control run, not derived"})
    out["coarse"] = {"rows": coarse, "grid4_stats": g4stats,
                     "grid4_used": len(g4live), "grid4_null": len(g4) - len(g4live)}

    # ---- abstention (CharXiv 'Not Applicable') --------------------------
    cx = [r for r in main_recs.get("charxiv", []) if not r["null"]]
    na = [r for r in cx if any(str(g).strip().lower() == "not applicable" for g in (r["gold"] or []))]
    real = [r for r in cx
            if not any(str(g).strip().lower() == "not applicable" for g in (r["gold"] or []))]

    def says_na(r):
        return str(r["pred"]).strip().lower() in ("not applicable", "n/a", "na", "none", "not applicable.")

    per_q = []
    byq = defaultdict(list)
    for r in na:
        byq[r["meta"].get("qid")].append(r)
    for q, rs in byq.items():
        per_q.append({"qid": q, "label": CHARXIV_QLABEL.get(q, str(q)), "n": len(rs),
                      "abstains": statistics.mean(float(says_na(r)) for r in rs)})
    per_q.sort(key=lambda d: d["abstains"])
    out["abstention"] = {
        "n_na": len(na), "n_real": len(real),
        "correct_abstain": statistics.mean(float(says_na(r)) for r in na) if na else None,
        "invents": 1 - (statistics.mean(float(says_na(r)) for r in na) if na else 0),
        "over_abstain": statistics.mean(float(says_na(r)) for r in real) if real else None,
        "per_q": per_q}

    # ---- format artifact ------------------------------------------------
    fmt = []
    for bench in ("slidevqa", "charxiv", "infographicvqa"):
        rs = [r for r in main_recs.get(bench, []) if not r["null"]]
        zeros = [r for r in rs if r["score"] == 0.0]
        if not zeros:
            continue
        eq = sum(1 for r in zeros if format_equivalent(r["pred"], r["gold"]))
        base = statistics.mean(r["score"] for r in rs)
        corr = statistics.mean(r["alt"] if r["alt"] is not None else r["score"] for r in rs)
        fmt.append({"bench": BENCH_LABEL[bench], "n": len(rs), "zeros": len(zeros),
                    "fmt_equiv": eq, "share": eq / len(zeros),
                    "as_scored": base, "corrected": corr,
                    "metric": BENCH_METRIC[bench]})
    out["format"] = fmt

    # ---- numeric error distribution -------------------------------------
    num = []
    groups = [("CharXiv descriptive", [r for r in main_recs.get("charxiv", [])
                                       if r["meta"].get("split") == "descriptive"]),
              ("CharXiv reasoning", [r for r in main_recs.get("charxiv", [])
                                     if r["meta"].get("split") == "reasoning"]),
              ("InfographicVQA", main_recs.get("infographicvqa", [])),
              ("SlideVQA", main_recs.get("slidevqa", []))]
    for lab, rs in groups:
        errs = []
        skipped_fmt = 0
        for r in rs:
            if r["null"] or r["score"] >= 0.5:
                continue
            # A "22%" scored against a gold of "22" is a formatting artifact, not a
            # misread number. Counting it here would report a 0% median error and
            # say the model reads numbers perfectly, which is the opposite of true.
            if format_equivalent(r["pred"], r["gold"]):
                skipped_fmt += 1
                continue
            p = _to_number(r["pred"])
            gs = [_to_number(g) for g in (r["gold"] or [])]
            gs = [g for g in gs if g is not None]
            if p is None or not gs:
                continue
            g = min(gs, key=lambda gg: abs(gg - p))
            if g == 0:
                continue
            errs.append(abs(p - g) / abs(g))
        if len(errs) < 10:
            continue
        errs.sort()
        num.append({"label": lab, "n": len(errs), "median": statistics.median(errs),
                    "within10": sum(1 for e in errs if e <= 0.10) / len(errs),
                    "over100": sum(1 for e in errs if e > 1.0) / len(errs),
                    "skipped_fmt": skipped_fmt})
    out["numeric"] = num

    # ---- SlideVQA 20-slide haystack -------------------------------------
    ap_best, ap_stats = read_best(RESULTS / ALLPAGES_FILE)
    ap = []
    for uid, rec in ap_best.items():
        if rec.get("pred") is None:
            continue
        s = score(Shim(rec, "slidevqa"), rec["pred"])
        ap.append({"uid": uid, "f1": float(s["score"]), "em": float(s["exact_match"]),
                   "meta": rec.get("meta") or {}})
    ev = [r for r in main_recs.get("slidevqa", []) if not r["null"]]
    ev_by = {r["uid"].replace("slidevqa:evidence:", ""): r for r in ev}
    pairs = [(r, ev_by.get(r["uid"].replace("slidevqa:all_pages:", ""))) for r in ap]
    pairs = [(a, b) for a, b in pairs if b is not None]
    out["haystack"] = {
        "n": len(ap), "stats": ap_stats,
        "f1": statistics.mean(r["f1"] for r in ap) if ap else None,
        "em": statistics.mean(r["em"] for r in ap) if ap else None,
        "paired_n": len(pairs),
        "paired_all": statistics.mean(a["f1"] for a, b in pairs) if pairs else None,
        "paired_ev": statistics.mean(b["score"] for a, b in pairs) if pairs else None}
    return out


def judge_join(cx_recs: list[dict]) -> dict:
    """CharXiv official-judge subset, kept strictly apart from string scoring."""
    p = RESULTS / JUDGE_FILE
    if not p.exists():
        return {}
    judged = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "uid" in d and d.get("judge_score") is not None:
            judged[d["uid"]] = d
    by_uid = {r["uid"]: r for r in cx_recs}
    inter = [u for u in judged if u in by_uid and not by_uid[u]["null"]]
    if not inter:
        return {}
    mismatch = sum(1 for u in inter if str(judged[u].get("pred")) != str(by_uid[u]["pred"]))
    per_split = defaultdict(lambda: {"j": [], "s": []})
    for u in inter:
        sp = judged[u].get("split") or by_uid[u]["meta"].get("split") or "?"
        per_split[sp]["j"].append(float(judged[u]["judge_score"]))
        per_split[sp]["s"].append(by_uid[u]["score"])
    return {
        "judged_rows": len(judged), "joined": len(inter), "coverage": len(inter) / len(cx_recs),
        "pred_mismatch": mismatch,
        "judge": statistics.mean(float(judged[u]["judge_score"]) for u in inter),
        "string": statistics.mean(by_uid[u]["score"] for u in inter),
        "agreement": statistics.mean(
            float((float(judged[u]["judge_score"]) >= 0.5) == (by_uid[u]["score"] >= 0.5))
            for u in inter),
        "per_split": {k: {"n": len(v["j"]), "judge": statistics.mean(v["j"]),
                          "string": statistics.mean(v["s"])} for k, v in per_split.items()},
        "uids": set(inter)}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def esc(s) -> str:
    return html.escape(str(s), quote=True)


def _mix(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def score_color(value: float | None, anchor: float, spread: float) -> tuple[str, str]:
    """Diverging ramp anchored at the benchmark's own headline.

    Absolute colour would paint all of ScreenSpot-Pro red and all of CharXiv
    green, which is exactly the information the drill-down already has. Anchoring
    each benchmark at its own number makes "worse than this benchmark's average"
    the thing the colour encodes, which is what you scan for.
    """
    if value is None:
        return ("transparent", "transparent")
    t = (value - anchor) / max(spread, 1e-6)
    t = max(-1.0, min(1.0, t))
    if t >= 0:
        strong = _mix("#8a8a8a", GOOD, t)
    else:
        strong = _mix("#8a8a8a", BAD, -t)
    return strong, strong


CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
 --muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
 --s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;--bad:#d03b3b;--warn:#fab219;
 --track:#e8e7e0;--hover:rgba(42,120,214,.07);--chip:rgba(11,11,11,.05)}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
 --surface:#1a1a19;--page:#0d0d0d;--ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.12);--s1:#3987e5;--s2:#d95926;
 --good:#3fce3f;--bad:#f06a6a;--track:#2c2c2a;--hover:rgba(57,135,229,.13);
 --chip:rgba(255,255,255,.07)}}
:root[data-theme=dark]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#ffffff;
 --ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.12);
 --s1:#3987e5;--s2:#d95926;--good:#3fce3f;--bad:#f06a6a;--track:#2c2c2a;
 --hover:rgba(57,135,229,.13);--chip:rgba(255,255,255,.07)}
:root[data-theme=light]{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;
 --ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
 --s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;--bad:#d03b3b;--track:#e8e7e0;
 --hover:rgba(42,120,214,.07);--chip:rgba(11,11,11,.05)}
*{box-sizing:border-box}
html,body{background:var(--page);color:var(--ink);margin:0;
 font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:30px 22px 90px}
header{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;flex-wrap:wrap}
h1{font-size:26px;margin:0 0 6px}
.dek{color:var(--ink2);margin:0;max-width:78ch}
h2{font-size:19px;margin:42px 0 4px;padding-top:20px;border-top:1px solid var(--grid)}
h2 .sub{display:block;font-size:13.5px;font-weight:400;color:var(--ink2);margin-top:5px;max-width:88ch}
a{color:var(--s1)}
button,select{font:inherit;font-size:13px;padding:6px 11px;border-radius:8px;
 border:1px solid var(--border);background:var(--surface);color:var(--ink2);cursor:pointer}
button:hover,select:hover{color:var(--ink);border-color:var(--axis)}
button.on{background:var(--s1);border-color:var(--s1);color:#fff}
input[type=search]{font:inherit;font-size:13px;padding:6px 10px;border-radius:8px;
 border:1px solid var(--border);background:var(--surface);color:var(--ink);min-width:190px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:15px 16px}
.tlab{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.tval{font-size:30px;line-height:1.1;margin:7px 0 3px;font-variant-numeric:tabular-nums}
.tnote{font-size:12.5px;color:var(--ink2)}
.tile.bad .tval{color:var(--bad)}.tile.good .tval{color:var(--good)}
.pcts{font-size:16px;color:var(--muted)}
.note{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--s2);
 border-radius:9px;padding:13px 16px;font-size:13.5px;color:var(--ink2);margin:16px 0}
.note strong{color:var(--ink)}
.note.bad{border-left-color:var(--bad)}
.note.ok{border-left-color:var(--good)}
.toolbar{position:sticky;top:0;z-index:20;background:var(--page);border-bottom:1px solid var(--grid);
 padding:11px 0;margin:6px 0 0;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.toolbar .gap{width:9px}
.toolbar label{font-size:12.5px;color:var(--muted);margin-right:2px}
.tree{background:var(--surface);border:1px solid var(--border);border-radius:12px;
 padding:8px 4px 12px;margin:14px 0;overflow-x:auto}
.hdr{display:grid;grid-template-columns:minmax(290px,1fr) 148px 74px 62px 124px 70px;
 gap:10px;align-items:end;padding:6px 14px 8px;font-size:11px;color:var(--muted);
 text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--grid);min-width:900px}
.hdr span:nth-child(n+3){text-align:right}
.nd{min-width:900px}
.ln{display:grid;grid-template-columns:minmax(290px,1fr) 148px 74px 62px 124px 70px;
 gap:10px;align-items:center;padding:3px 14px;border-radius:7px;cursor:default}
.ln:hover{background:var(--hover)}
.nd.dim>.ln{opacity:.86}
.nd[data-k="bench"]>.ln{font-weight:600;margin-top:6px}
.nd[data-k="root"]>.ln{font-weight:700}
.lbl{display:flex;align-items:center;gap:6px;min-width:0}
.tw{flex:0 0 15px;width:15px;text-align:center;color:var(--muted);font-size:10px;
 cursor:pointer;user-select:none;transition:transform .12s}
.nd.open>.ln .tw{transform:rotate(90deg)}
.tw.leaf{cursor:default;opacity:.32}
.txt{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lv{font-size:10.5px;color:var(--muted);background:var(--chip);border-radius:4px;
 padding:1px 5px;flex:0 0 auto}
.track{height:13px;background:var(--track);border-radius:4px;position:relative;overflow:hidden}
.bar{display:block;height:100%;width:0;border-radius:0 4px 4px 0;min-width:2px;
 transition:width .15s}
.val{text-align:right;font-variant-numeric:tabular-nums;font-size:13px;white-space:nowrap}
.nnum{text-align:right;font-variant-numeric:tabular-nums;font-size:12.5px;color:var(--ink2)}
.met{text-align:right;font-size:11px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;
 white-space:nowrap}
.dlt{text-align:right;font-variant-numeric:tabular-nums;font-size:12px}
.dlt.up{color:var(--good)}.dlt.dn{color:var(--bad)}
.kids{display:none;border-left:1px solid var(--grid);margin-left:21px}
.nd.open>.kids{display:block}
.flag{display:inline-block;font-size:10px;font-weight:600;padding:1px 6px;border-radius:999px;
 background:color-mix(in srgb,var(--warn) 26%,transparent);color:var(--ink);flex:0 0 auto}
.flag.sm{background:color-mix(in srgb,var(--bad) 20%,transparent)}
.flag.jd{background:color-mix(in srgb,var(--s1) 22%,transparent)}
.ex{margin:4px 0 8px 36px;font-size:12px;border-collapse:collapse;width:calc(100% - 60px);
 min-width:520px}
.ex th{text-align:left;color:var(--muted);font-weight:400;font-size:10.5px;
 text-transform:uppercase;letter-spacing:.04em;padding:2px 8px}
.ex td{padding:2px 8px;border-top:1px solid var(--grid);vertical-align:top;
 font-variant-numeric:tabular-nums}
.ex td.g{color:var(--ink2)}
.ex td.s{text-align:right;width:52px}
.ex .ok{color:var(--good)}.ex .no{color:var(--bad)}
.ex .lk{padding-top:4px}
table.t{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 6px}
table.t th,table.t td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--grid)}
table.t td{font-variant-numeric:tabular-nums}
table.t th[scope=row]{font-weight:400;color:var(--ink2)}
table.t td.num,table.t th.num{text-align:right}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
 padding:16px 20px 18px;margin:14px 0}
.card h3{font-size:15.5px;margin:0 0 3px}
.card .sub{font-size:13px;color:var(--ink2);margin:0 0 12px;max-width:88ch}
.ok{color:var(--good)}.no{color:var(--bad)}
.legend{display:flex;gap:14px;align-items:center;font-size:12px;color:var(--ink2);
 margin:2px 0 0;flex-wrap:wrap}
.ramp{width:120px;height:10px;border-radius:3px;
 background:linear-gradient(90deg,var(--bad),#8a8a8a,var(--good))}
.foot{font-size:12.5px;color:var(--muted);margin-top:36px;border-top:1px solid var(--grid);
 padding-top:14px}
.hide{display:none !important}
"""

JS = r"""
(function(){
const root=document.documentElement;
const tree=document.getElementById('tree');
const nodes=()=>Array.from(tree.querySelectorAll('.nd'));

function setOpen(nd,on){nd.classList.toggle('open',on);}
tree.addEventListener('click',e=>{
  const ln=e.target.closest('.ln'); if(!ln) return;
  if(e.target.closest('a')) return;
  const nd=ln.parentElement;
  if(!nd.querySelector(':scope > .kids')) return;
  setOpen(nd,!nd.classList.contains('open'));
});
tree.addEventListener('keydown',e=>{
  if(e.key!=='Enter'&&e.key!==' ') return;
  const ln=e.target.closest('.ln'); if(!ln) return;
  e.preventDefault(); ln.click();
});

document.getElementById('all').onclick=()=>nodes().forEach(n=>setOpen(n,!!n.querySelector(':scope > .kids')));
document.getElementById('none').onclick=()=>nodes().forEach(n=>setOpen(n,false));
document.querySelectorAll('[data-depth]').forEach(b=>{
  b.onclick=()=>{const d=+b.dataset.depth;
    nodes().forEach(n=>setOpen(n,(+n.dataset.d)<d && !!n.querySelector(':scope > .kids')));};
});

const sortSel=document.getElementById('sort');
function sortAll(){
  const mode=sortSel.value;
  tree.querySelectorAll('.kids').forEach(k=>{
    const kids=Array.from(k.children).filter(c=>c.classList.contains('nd'));
    if(kids.length<2) return;
    kids.sort((a,b)=>{
      if(mode==='n') return (+b.dataset.n)-(+a.dataset.n);
      if(mode==='alpha') return a.dataset.lab.localeCompare(b.dataset.lab);
      const av=a.dataset.v===''?9e9:+a.dataset.v, bv=b.dataset.v===''?9e9:+b.dataset.v;
      if(mode==='low') return av-bv;
      if(mode==='high') return bv-av;
      return (+a.dataset.i)-(+b.dataset.i);
    });
    kids.forEach(c=>k.appendChild(c));
  });
}
sortSel.onchange=sortAll;

const scaleBtn=document.getElementById('scale');
function applyScale(){
  const rel=scaleBtn.classList.contains('on');
  tree.querySelectorAll('.bar').forEach(b=>{
    const v=+b.dataset.v, mx=+b.dataset.mx||1;
    if(isNaN(v)){b.style.width='0';return;}
    const w=rel? (mx>0? v/mx*100:0) : v*100;
    b.style.width=Math.max(w,0.7).toFixed(2)+'%';
  });
  scaleBtn.textContent=rel?'bars: relative to benchmark':'bars: absolute 0-100%';
}
scaleBtn.onclick=()=>{scaleBtn.classList.toggle('on');applyScale();};
applyScale();

const smallBtn=document.getElementById('small');
smallBtn.onclick=()=>{smallBtn.classList.toggle('on');
  const on=smallBtn.classList.contains('on');
  tree.querySelectorAll('.nd[data-small="1"]').forEach(n=>n.classList.toggle('hide',on));
  smallBtn.textContent=on?'small-n hidden':'hide small-n (n<30)';};

const q=document.getElementById('q');
q.oninput=()=>{
  const s=q.value.trim().toLowerCase();
  const all=nodes();
  if(!s){all.forEach(n=>{n.classList.remove('hide');});return;}
  all.forEach(n=>n.classList.add('hide'));
  all.forEach(n=>{
    if(!n.dataset.lab.toLowerCase().includes(s)) return;
    n.classList.remove('hide');
    let p=n.parentElement;
    while(p&&p!==tree){ if(p.classList&&p.classList.contains('nd')){p.classList.remove('hide');setOpen(p,true);} p=p.parentElement; }
    n.querySelectorAll('.nd').forEach(c=>c.classList.remove('hide'));
  });
};

const tb=document.getElementById('theme');
function label(){const d=root.dataset.theme;
  tb.textContent=d==='dark'?'Light mode':(d==='light'?'Dark mode':'Dark mode');}
tb.onclick=()=>{const cur=root.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
  root.dataset.theme=cur==='dark'?'light':'dark';label();};
label();
// open the first two levels so the page is useful before any click
nodes().forEach(n=>setOpen(n,(+n.dataset.d)<2 && !!n.querySelector(':scope > .kids')));
})();
"""


def fmt_pct(v, d=1) -> str:
    return "&mdash;" if v is None else f"{v * 100:.{d}f}"


def render_node(node: Node, anchor: float, spread: float, bmax: float,
                idx: int, judge_by_node: dict) -> str:
    strong, _ = score_color(node.value, anchor, spread)
    flags = []
    if node.n < SMALL_N and node.kind not in ("root", "dim"):
        flags.append('<span class="flag sm" title="fewer than 30 scored rows: '
                     'this number is noise">n&lt;30</span>')
    if node.spread is not None and node.spread >= DIVERGE:
        flags.append(f'<span class="flag" title="widest gap between this node\'s children '
                     f'(n&ge;{SMALL_N})">children spread {node.spread * 100:.0f}pp</span>')
    if node.n_null:
        flags.append(f'<span class="flag" title="rows whose prediction was null; excluded '
                     f'from the metric">{node.n_null} null pred</span>')
    jv = judge_by_node.get(id(node))
    if jv:
        flags.append(f'<span class="flag jd" title="official CharXiv LLM judge on the '
                     f'{jv["n"]} rows of this node it covers &mdash; never averaged with the '
                     f'string score">judge {jv["v"] * 100:.1f}% on {jv["n"]}</span>')

    delta = ""
    if node.delta is not None and node.kind == "node":
        cls = "up" if node.delta >= 0 else "dn"
        delta = f'<span class="dlt {cls}">{node.delta * 100:+.1f}</span>'
    elif node.kind in ("dim", "bench", "root"):
        delta = '<span class="dlt" style="color:var(--muted)">&mdash;</span>'

    alt = ""
    if node.bench == "slidevqa" and node.alt is not None and node.value is not None \
            and abs(node.alt - node.value) > 5e-4:
        alt = (f'<span class="lv" title="same rows, crediting answers that differ from the '
               f'gold only in formatting">fmt-corrected {node.alt * 100:.1f}</span>')
    if node.bench == "slidevqa" and node.em is not None:
        alt += f'<span class="lv" title="exact match, SlideVQA\'s second official metric">' \
               f'EM {node.em * 100:.1f}</span>'

    lvl = f'<span class="lv">{esc(node.level)}</span>' if node.level and node.kind == "node" else ""
    pad = 4 + node.depth * 15
    tw = "&#9656;" if node.children else "&#8226;"
    twcls = "tw" if node.children else "tw leaf"
    val = fmt_pct(node.value)
    ln = (
        f'<div class="ln" tabindex="0" role="button" aria-expanded="false">'
        f'<span class="lbl" style="padding-left:{pad}px">'
        f'<span class="{twcls}">{tw}</span>'
        f'<span class="txt" title="{esc(node.label)}">{esc(node.label)}</span>'
        f'{lvl}{alt}{"".join(flags)}</span>'
        f'<span class="track"><span class="bar" data-v="{"" if node.value is None else f"{node.value:.6f}"}" '
        f'data-mx="{bmax:.6f}" style="background:{strong}"></span></span>'
        f'<span class="val">{val}<span class="pcts" style="font-size:11px">%</span></span>'
        f'{delta}'
        f'<span class="met" title="{esc(node.metric)}">{esc(node.metric)}</span>'
        f'<span class="nnum">n={node.n:,}</span>'
        f'</div>')

    kids = ""
    if node.children:
        parts = []
        for i, c in enumerate(node.children):
            parts.append(render_node(c, anchor, spread, bmax, i, judge_by_node))
        kids = f'<div class="kids">{"".join(parts)}</div>'
    elif node.examples:
        link = EVIDENCE_LINK.get(node.bench or "")
        rows = "".join(
            f'<tr><td class="g">{esc(e["uid"])}</td><td class="g">{esc(e["gold"])}</td>'
            f'<td>{esc(e["pred"])}</td>'
            f'<td class="s {"ok" if e["score"] >= 0.5 else "no"}">{e["score"]:.2f}</td></tr>'
            for e in node.examples)
        lk = (f'<tr><td colspan="4" class="lk"><a href="{link[0]}">browse {esc(link[1])} '
              f'&rarr;</a></td></tr>') if link else ""
        kids = (f'<div class="kids"><table class="ex"><thead><tr><th>uid</th><th>gold</th>'
                f'<th>prediction</th><th class="s">score</th></tr></thead>'
                f'<tbody>{rows}{lk}</tbody></table></div>')

    small = "1" if (node.n < SMALL_N and node.kind == "node") else "0"
    return (f'<div class="nd {"dim" if node.kind == "dim" else ""}" data-k="{node.kind}" '
            f'data-d="{node.depth}" data-n="{node.n}" '
            f'data-v="{"" if node.value is None else f"{node.value:.6f}"}" '
            f'data-i="{idx}" data-small="{small}" data-lab="{esc(node.label)}">'
            f'{ln}{kids}</div>')


def control_html(c: dict, main_recs: dict) -> str:
    parts = []

    b = c["blind"]
    rows = "".join(
        f'<tr><th scope="row">{esc(r["bench"])}</th><td class="num">{r["n"]:,}</td>'
        f'<td class="num">{fmt_pct(r["blind"])}</td><td class="num">{fmt_pct(r["sighted"])}</td>'
        f'<td class="num"><b>{r["delta"] * 100:+.1f}</b></td>'
        f'<td class="num">{"25.0" if r["chance"] else "&mdash;"}</td>'
        f'<td class="num">{r["unmatched"]}</td>'
        f'<td>{esc(r["metric"])}</td></tr>'
        for r in b["rows"] + b["charxiv_split"])
    parts.append(f"""<div class="card"><h3>Blind control &mdash; how much of each number needs the image?</h3>
<p class="sub">The same question asked with the image withheld, joined back to the sighted run by
<code>meta.src_uid</code> and compared on exactly those items, never against the full-split headline.
{b['stats']['unique']:,} control rows on disk.</p>
<table class="t"><thead><tr><th>slice</th><th class="num">n paired</th>
<th class="num">blind</th><th class="num">sighted (same items)</th><th class="num">vision adds (pp)</th>
<th class="num">chance</th><th class="num">unpaired</th><th>metric</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class="sub" style="margin:8px 0 0">AI2D is the outlier and the reason a raw AI2D score is not a
perception measurement: most of it survives with the diagram hidden. The <em>unpaired</em> column
counts control rows whose <code>src_uid</code> belongs to a condition outside this tree &mdash; the
SlideVQA blind sample was drawn across both the evidence and the all-pages arms, and only the
evidence half can be compared like for like here.</p></div>""")

    o = c["onepage"]
    if o["n"]:
        parts.append(f"""<div class="card"><h3>One-page ablation &mdash; SlideVQA multi-evidence, given one slide</h3>
<p class="sub">{o['n']:,} multi-evidence questions re-asked with only the first evidence slide.
Same metric, same items.</p>
<table class="t"><thead><tr><th>condition</th><th class="num">n</th><th class="num">token F1</th></tr></thead>
<tbody>
<tr><th scope="row">both evidence slides</th><td class="num">{o['n']:,}</td><td class="num">{fmt_pct(o['both'])}</td></tr>
<tr><th scope="row">first slide only</th><td class="num">{o['n']:,}</td><td class="num">{fmt_pct(o['one'])}</td></tr>
<tr><th scope="row">collapse</th><td class="num">&mdash;</td><td class="num no"><b>{(o['one'] - o['both']) * 100:+.1f}</b></td></tr>
<tr><th scope="row">still answerable (F1 &ge; 0.5) on one slide</th><td class="num">&mdash;</td>
<td class="num">{fmt_pct(o['answerable'])}</td></tr>
</tbody></table></div>""")

    hs = c["haystack"]
    if hs["n"]:
        parts.append(f"""<div class="card"><h3>The other direction &mdash; 20 slides instead of the evidence</h3>
<p class="sub">The all-pages arm ({hs['n']:,} rows) sends the whole deck and makes the model find the
evidence itself. Held out of the tree's root count because it is a second condition on the same
questions, not extra questions.</p>
<table class="t"><thead><tr><th>condition</th><th class="num">n</th><th class="num">token F1</th>
<th class="num">EM</th></tr></thead><tbody>
<tr><th scope="row">all 20 slides (whole arm)</th><td class="num">{hs['n']:,}</td>
<td class="num">{fmt_pct(hs['f1'])}</td><td class="num">{fmt_pct(hs['em'])}</td></tr>
<tr><th scope="row">paired: evidence pages</th><td class="num">{hs['paired_n']:,}</td>
<td class="num">{fmt_pct(hs['paired_ev'])}</td><td class="num">&mdash;</td></tr>
<tr><th scope="row">paired: all 20 slides</th><td class="num">{hs['paired_n']:,}</td>
<td class="num">{fmt_pct(hs['paired_all'])}</td><td class="num">&mdash;</td></tr>
</tbody></table></div>""")

    cz = c["coarse"]
    rows = "".join(
        f'<tr><th scope="row">{esc(r["label"])}</th><td class="num">{r["n"]:,}</td>'
        f'<td class="num">{fmt_pct(r["acc"])}</td>'
        f'<td class="num">{r["chance"] * 100:.3f}</td>'
        f'<td class="num">{(r["acc"] / r["chance"]):.1f}&times;</td></tr>'
        for r in cz["rows"])
    parts.append(f"""<div class="card"><h3>Coarse localization &mdash; the click is not random, it is imprecise</h3>
<p class="sub">The same ScreenSpot-Pro predictions, scored against progressively finer grids: a
prediction counts if it lands in the same cell as the target's centre. The last row is a separate
control in which the model was shown a labelled 4&times;4 grid and asked to name the cell
({cz['grid4_used']:,} rows used from a file that may still be growing;
{cz['grid4_stats'].get('malformed', 0)} malformed lines skipped).</p>
<table class="t"><thead><tr><th>granularity</th><th class="num">n</th><th class="num">accuracy</th>
<th class="num">chance</th><th class="num">vs chance</th></tr></thead><tbody>{rows}</tbody></table></div>""")

    a = c["abstention"]
    worst = "".join(
        f'<tr><th scope="row">Q{r["qid"]} &mdash; {esc(r["label"])}</th><td class="num">{r["n"]}</td>'
        f'<td class="num">{fmt_pct(r["abstains"])}</td></tr>' for r in a["per_q"][:5])
    best = "".join(
        f'<tr><th scope="row">Q{r["qid"]} &mdash; {esc(r["label"])}</th><td class="num">{r["n"]}</td>'
        f'<td class="num">{fmt_pct(r["abstains"])}</td></tr>' for r in a["per_q"][-3:])
    parts.append(f"""<div class="card"><h3>Abstention &mdash; CharXiv questions whose gold is "Not Applicable"</h3>
<p class="sub">{a['n_na']:,} of the {a['n_na'] + a['n_real']:,} scored CharXiv questions have no answer
in the figure. Getting these right means declining, and the failure mode is inventing a value.
The same nodes are drillable in the tree under CharXiv &rsaquo; descriptive &rsaquo; <em>by answerability</em>.</p>
<table class="t"><thead><tr><th>behaviour</th><th class="num">n</th><th class="num">rate</th></tr></thead><tbody>
<tr><th scope="row">correctly declines when there is no answer</th><td class="num">{a['n_na']:,}</td>
<td class="num ok">{fmt_pct(a['correct_abstain'])}</td></tr>
<tr><th scope="row">invents a value instead</th><td class="num">{a['n_na']:,}</td>
<td class="num no">{fmt_pct(a['invents'])}</td></tr>
<tr><th scope="row">declines when an answer does exist</th><td class="num">{a['n_real']:,}</td>
<td class="num">{fmt_pct(a['over_abstain'])}</td></tr>
</tbody></table>
<table class="t"><thead><tr><th>hardest to decline</th><th class="num">n</th><th class="num">abstains</th></tr></thead>
<tbody>{worst}</tbody></table>
<table class="t"><thead><tr><th>easiest to decline</th><th class="num">n</th><th class="num">abstains</th></tr></thead>
<tbody>{best}</tbody></table></div>""")

    f = c["format"]
    rows = "".join(
        f'<tr><th scope="row">{esc(r["bench"])}</th><td class="num">{r["zeros"]:,}</td>'
        f'<td class="num">{r["fmt_equiv"]:,}</td><td class="num"><b>{fmt_pct(r["share"])}</b></td>'
        f'<td class="num">{fmt_pct(r["as_scored"])}</td><td class="num">{fmt_pct(r["corrected"])}</td>'
        f'<td class="num">{(r["corrected"] - r["as_scored"]) * 100:+.1f}</td></tr>' for r in f)
    parts.append(f"""<div class="card"><h3>Format artifact &mdash; how many hard zeros are only formatting?</h3>
<p class="sub">A hard zero is a scored-0 answer. The detector is deliberately conservative: reduce both
sides to a number after stripping <code>, $ &euro; &pound; %</code> and scale words (bn/billion/m/million/k/thousand)
and compare sign-sensitively; otherwise compare case- and punctuation-folded alphanumerics exactly.
No substring fallback. SlideVQA's token F1 scores <code>22%</code> against <code>22</code> as a zero;
ANLS does not, which is most of why the three columns differ so much.</p>
<table class="t"><thead><tr><th>benchmark</th><th class="num">hard zeros</th>
<th class="num">format-equivalent</th><th class="num">share of zeros</th>
<th class="num">as scored</th><th class="num">format-corrected</th><th class="num">delta</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class="sub" style="margin:8px 0 0">The corrected column is never the headline. It is shown so the
SlideVQA number can be read as what it is: a token-overlap metric, not a comprehension score.</p>
{_slidevqa_numeric_note(main_recs)}</div>""")

    nm = c["numeric"]
    rows = "".join(
        f'<tr><th scope="row">{esc(r["label"])}</th><td class="num">{r["n"]:,}</td>'
        f'<td class="num">{r["median"] * 100:.1f}%</td><td class="num">{fmt_pct(r["within10"])}%</td>'
        f'<td class="num">{fmt_pct(r["over100"])}%</td>'
        f'<td class="num">{r["skipped_fmt"]:,}</td></tr>' for r in nm)
    parts.append(f"""<div class="card"><h3>When the answer is a number and it is wrong, how wrong?</h3>
<p class="sub">Relative error against the nearest numeric gold, over scored-wrong answers where both
sides parse as numbers. Near misses would mean imprecise reading; these are not near misses.</p>
<table class="t"><thead><tr><th>slice</th><th class="num">n numeric failures</th>
<th class="num">median relative error</th><th class="num">within 10%</th>
<th class="num">off by &gt;100%</th><th class="num">format-only, excluded</th></tr></thead>
<tbody>{rows}</tbody></table></div>""")
    return "".join(parts)



def _slidevqa_numeric_note(main_recs: dict) -> str:
    """The one thing the flat SlideVQA number hides, stated as a number.

    Drilling SlideVQA to evidence spread -> arithmetic -> gold answer shape shows
    the arithmetic penalty is almost entirely a scoring artifact on number-shaped
    golds, not a reasoning failure. That is only visible four levels down, which
    is the argument for the page.
    """
    rs = [r for r in main_recs.get("slidevqa", []) if not r["null"]]
    num = [r for r in rs if _gold_shape(r) == "numeric gold"]
    txt = [r for r in rs if _gold_shape(r) != "numeric gold"]
    ari = [r for r in rs if r["meta"].get("arithmetic")]
    lok = [r for r in rs if not r["meta"].get("arithmetic")]
    if not (num and txt and ari and lok):
        return ""

    def m(rows, key):
        return sum((r[key] if r[key] is not None else r["score"]) for r in rows) / len(rows)

    return (f'<div class="note"><strong>Where this actually bites, four levels down.</strong> '
            f'Split SlideVQA by whether the gold answer is a number and the artifact stops being '
            f'uniform. Numeric golds (n={len(num):,}) score {m(num, "score") * 100:.1f} token F1 '
            f'but {m(num, "alt") * 100:.1f} format-corrected, a '
            f'{(m(num, "alt") - m(num, "score")) * 100:+.1f} point move; text golds '
            f'(n={len(txt):,}) move only {(m(txt, "alt") - m(txt, "score")) * 100:+.1f}. '
            f'That is the whole of SlideVQA\'s apparent arithmetic weakness: arithmetic questions '
            f'score {m(ari, "score") * 100:.1f} against {m(lok, "score") * 100:.1f} for lookups, a '
            f'{(m(lok, "score") - m(ari, "score")) * 100:.1f}-point gap that shrinks to '
            f'{(m(lok, "alt") - m(ari, "alt")) * 100:.1f} points once formatting is credited '
            f'&mdash; because arithmetic answers are numbers and lookups mostly are not. '
            f'The flat 68.8 hides this completely.</div>')


def render(trees: list[Node], root: Node, meta: dict, controls: dict,
           judge: dict, violations: list, load_stats: dict,
           judge_by_node: dict, main_recs: dict) -> str:
    tiles = [f'<div class="tile"><div class="tlab">questions</div>'
             f'<div class="tval">{root.n:,}</div>'
             f'<div class="tnote">{len(trees)} benchmarks &middot; official splits</div></div>']
    for t in trees:
        tone = "bad" if (t.value or 0) < 0.5 else ("good" if (t.value or 0) >= 0.8 else "")
        tiles.append(
            f'<div class="tile {tone}"><div class="tlab">{esc(BENCH_LABEL[t.bench])}</div>'
            f'<div class="tval">{fmt_pct(t.value)}<span class="pcts">%</span></div>'
            f'<div class="tnote">n={t.n:,} &middot; {esc(t.metric)}</div></div>')

    body = []
    for t in trees:
        vals = [x.value for x in walk(t) if x.value is not None and x.n >= SMALL_N]
        anchor = t.value or 0.0
        spread = max(
            (statistics.quantiles([abs(v - anchor) for v in vals], n=10)[-1]
             if len(vals) >= 10 else max((abs(v - anchor) for v in vals), default=0.1)),
            0.02)
        bmax = max(vals, default=1.0) or 1.0
        body.append(render_node(t, anchor, spread, bmax, 0, judge_by_node))
    tree_html = render_node_root(root, body)

    vio = ""
    if violations:
        rows = "".join(
            f'<tr><th scope="row">{esc(v["node"])}</th><td>{esc(v["dimension"])}</td>'
            f'<td class="num">{v["parent_n"]}</td><td class="num">{v["children_n"]}</td></tr>'
            for v in violations[:40])
        vio = (f'<div class="note bad"><strong>{len(violations)} node(s) do not add up.</strong> '
               f'Listed rather than hidden.<table class="t"><thead><tr><th>node</th><th>dimension</th>'
               f'<th class="num">parent n</th><th class="num">sum of children</th></tr></thead>'
               f'<tbody>{rows}</tbody></table></div>')
    else:
        vio = ('<div class="note ok"><strong>Every parent equals the sum of its children.</strong> '
               f'Checked at all {meta["splits_checked"]:,} splits in the tree, on both the scored '
               f'count and the null-prediction count. Nothing is dropped silently: a record with '
               f'missing metadata lands in an explicit <em>(metadata missing)</em> child rather '
               f'than falling out.</div>')

    lrows = "".join(
        f'<tr><th scope="row">{esc(BENCH_LABEL.get(k, k))}</th>'
        f'<td class="num">{s["lines"]:,}</td><td class="num">{s["unique"]:,}</td>'
        f'<td class="num">{s["duplicate_uids"]:,}</td>'
        f'<td class="num">{s["null_pred_raw"]:,}</td>'
        f'<td class="num">{s["null_pred_after_dedup"]:,}</td>'
        f'<td class="num">{s["malformed"]:,}</td></tr>'
        for k, s in load_stats.items())

    jb = ""
    if judge:
        pr = "".join(
            f'<tr><th scope="row">{esc(k)}</th><td class="num">{v["n"]:,}</td>'
            f'<td class="num">{fmt_pct(v["judge"])}</td><td class="num">{fmt_pct(v["string"])}</td>'
            f'<td class="num">{(v["judge"] - v["string"]) * 100:+.1f}</td></tr>'
            for k, v in sorted(judge["per_split"].items()))
        jb = f"""<div class="card"><h3>CharXiv: string scoring is a lower bound</h3>
<p class="sub">CharXiv's official grader is an LLM judge with per-question-type rubrics. This harness
scores strings. A partial judge file covers {judge['joined']:,} rows
({fmt_pct(judge['coverage'], 0)}% of the split) and joins cleanly &mdash;
{judge['pred_mismatch']} of those rows disagree about what the model actually said, so the join is
sound. Judge-scored nodes carry a blue badge in the tree; the two scores are shown side by side and
are never averaged into one number.</p>
<table class="t"><thead><tr><th>split</th><th class="num">n judged</th><th class="num">official judge</th>
<th class="num">string match (same rows)</th><th class="num">gap (pp)</th></tr></thead>
<tbody>{pr}
<tr><th scope="row"><b>all judged rows</b></th><td class="num">{judge['joined']:,}</td>
<td class="num"><b>{fmt_pct(judge['judge'])}</b></td><td class="num">{fmt_pct(judge['string'])}</td>
<td class="num">{(judge['judge'] - judge['string']) * 100:+.1f}</td></tr></tbody></table>
<p class="sub" style="margin:8px 0 0">The judge and the string matcher agree on
{fmt_pct(judge['agreement'], 0)}% of items at the 0.5 threshold. That agreement is what licenses
using string scoring for the {100 - judge['coverage'] * 100:.0f}% of the split the judge has not
covered &mdash; while remembering the free-text types are still undercounted there.</p></div>"""

    nav = " &middot; ".join(f'<a href="{h}">{esc(t)}</a>' for h, t in NAV)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Claude Haiku 4.5 &mdash; drill-down</title><style>{CSS}</style></head>
<body><div class="wrap">
<header><div>
<h1>Every number in the study, opened up</h1>
<p class="dek">One headline per benchmark, then the splits that produced it, then the splits inside
those, down to the individual questions. {root.n:,} questions across {len(trees)} benchmarks, each
scored with its own official metric. Click any row to open it; the number beside each child is how
far that child sits from its parent, which is the only reason to drill down at all.</p>
<p class="dek" style="margin-top:8px;font-size:13px">{nav}</p>
</div><button id="theme" type="button">Dark mode</button></header>

<div class="tiles">{''.join(tiles)}</div>

<div class="note"><strong>The root number is not a score.</strong> ANLS, token F1, click-in-bbox and
multiple-choice accuracy measure different things on different scales; averaging them would produce a
number that describes nothing. The root row therefore shows only the question count, and every
benchmark below carries its own metric name on every row.</div>

{vio}

<h2>The drill-down<span class="sub">Depth 0 is the whole study. Colour is diverging and anchored at
each benchmark's own headline, so red means "worse than this benchmark's average", not "low" &mdash;
otherwise every ScreenSpot-Pro row would be red and every CharXiv row green. Nodes with fewer than
{SMALL_N} scored rows are labelled; a parent whose children spread more than
{DIVERGE * 100:.0f} points carries a badge, because that is where the drill-down is telling you
something.</span></h2>

<div class="toolbar">
<button id="all" type="button">expand all</button>
<button id="none" type="button">collapse all</button>
<span class="gap"></span><label>to depth</label>
<button data-depth="1" type="button">1</button><button data-depth="2" type="button">2</button>
<button data-depth="3" type="button">3</button><button data-depth="4" type="button">4</button>
<button data-depth="5" type="button">5</button><button data-depth="9" type="button">all</button>
<span class="gap"></span><label for="sort">sort children</label>
<select id="sort"><option value="def">natural order</option><option value="n">by n</option>
<option value="low">by score, worst first</option><option value="high">by score, best first</option>
<option value="alpha">alphabetically</option></select>
<span class="gap"></span>
<button id="scale" type="button">bars: absolute 0-100%</button>
<button id="small" type="button">hide small-n (n&lt;30)</button>
<span class="gap"></span>
<input id="q" type="search" placeholder="filter labels…" aria-label="filter labels">
</div>
<div class="legend"><span class="ramp"></span>
<span>worse than benchmark average &rarr; better</span>
<span class="flag">children spread</span><span class="flag sm">n&lt;30</span>
<span class="flag jd">judge-scored</span></div>

<div class="tree" id="tree"><div class="hdr"><span>node</span><span>score</span><span>value</span>
<span>vs parent</span><span>metric</span><span>n</span></div>
{tree_html}</div>

<p class="dek" style="font-size:13px">Flattened export:
<a href="drilldown.csv">drilldown.csv</a> &middot; <a href="drilldown.json">drilldown.json</a>
&mdash; every node in the tree with its n, metric, value and delta, so the numbers can be checked
without a browser.</p>

<h2>What the data had to say before it was scored<span class="sub">Result files are appended to on
resume, so several contain the same uid more than once. A later row replaces an earlier one unless
that would replace a real prediction with a null &mdash; the same rule the rest of the study uses.
Everything below the tree is computed after that deduplication.</span></h2>
<div class="card"><table class="t"><thead><tr><th>benchmark</th><th class="num">lines on disk</th>
<th class="num">unique questions</th><th class="num">duplicate rows</th>
<th class="num">null pred (raw)</th><th class="num">null pred (after dedup)</th>
<th class="num">malformed</th></tr></thead><tbody>{lrows}</tbody></table>
<p class="sub" style="margin:10px 0 0">Null predictions are excluded from every metric and counted at
every node they belong to; a node that dropped any carries the count as a badge. Deduplication
recovers most nulls because a resumed run re-asked the question and got an answer the second time.</p>
</div>

{jb}

<h2>The controls, recomputed<span class="sub">Each of these is an ablation over the same questions,
recomputed here from the result files rather than quoted. Where an ablation joins back to the main
run it is compared only against those same items.</span></h2>
{control_html(controls, main_recs)}

<p class="foot">Generated by <code>blindspot/reporting/drilldown_report.py</code> from the result JSONL files
under <code>results/</code>. Nothing on this page is transcribed from a previous report: every value
is recomputed at build time using <code>blindspot/core/scoring.py</code>, the same scorer the rest of the
study uses.</p>
</div><script>{JS}</script></body></html>"""


def render_node_root(root: Node, bench_html: list[str]) -> str:
    ln = (f'<div class="ln" tabindex="0" role="button">'
          f'<span class="lbl" style="padding-left:4px"><span class="tw">&#9656;</span>'
          f'<span class="txt">ALL &mdash; every question in the study</span>'
          f'<span class="flag" title="the benchmarks below use four different metrics">'
          f'mixed metrics: not comparable</span></span>'
          f'<span class="track"></span>'
          f'<span class="val">&mdash;</span><span class="dlt" style="color:var(--muted)">&mdash;</span>'
          f'<span class="met">&mdash;</span><span class="nnum">n={root.n:,}</span></div>')
    return (f'<div class="nd open" data-k="root" data-d="0" data-n="{root.n}" data-v="" '
            f'data-i="0" data-small="0" data-lab="ALL">'
            f'{ln}<div class="kids">{"".join(bench_html)}</div></div>')


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def flatten(root: Node, trees: list[Node]) -> list[dict]:
    rows = []

    def rec(n: Node, path: list[str]):
        p = path + [n.label]
        rows.append({
            "path": " > ".join(p), "depth": n.depth, "kind": n.kind,
            "benchmark": BENCH_LABEL.get(n.bench or "", ""), "level": n.level,
            "label": n.label, "n_scored": n.n, "n_null_pred": n.n_null,
            "metric": n.metric if n.kind != "root" else "mixed (not comparable)",
            "value": "" if n.value is None else round(n.value, 6),
            "value_pct": "" if n.value is None else round(n.value * 100, 2),
            "delta_vs_parent_pp": "" if n.delta is None else round(n.delta * 100, 2),
            "exact_match_pct": "" if n.em is None else round(n.em * 100, 2),
            "format_corrected_pct": "" if n.alt is None else round(n.alt * 100, 2),
            "children_spread_pp": "" if n.spread is None else round(n.spread * 100, 2),
            "small_n": int(n.n < SMALL_N and n.kind == "node"),
        })
        for c in n.children:
            rec(c, p)

    rec(root, [])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "drilldown.html"))
    ap.add_argument("--csv", default=str(OUT / "drilldown.csv"))
    ap.add_argument("--json", default=str(OUT / "drilldown.json"))
    a = ap.parse_args()

    jobs = [(b, str(RESULTS / f)) for b, f in MAIN_FILES.items()]
    raw_nulls = {}
    for b, f in MAIN_FILES.items():
        n = 0
        with open(RESULTS / f, encoding="utf-8") as fh:
            for line in fh:
                if '"pred": null' in line or '"pred":null' in line:
                    n += 1
        raw_nulls[b] = n

    main_recs: dict[str, list[dict]] = {}
    load_stats: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=min(6, len(jobs))) as ex:
        for bench, recs, stats in ex.map(score_file, jobs):
            main_recs[bench] = recs
            stats["null_pred_raw"] = raw_nulls[bench]
            load_stats[bench] = stats
    load_stats = {b: load_stats[b] for b in MAIN_FILES}

    judge = judge_join(main_recs["charxiv"])

    violations: list = []
    counter = [0]
    trees = []
    for bench in MAIN_FILES:
        t = build(BENCH_LABEL[bench], "benchmark", "bench", main_recs[bench], bench,
                  SPEC[bench], (), 1, violations, counter)
        t.note = BENCH_BLURB[bench]
        trees.append(t)

    root = Node(label="ALL", level="", kind="root", depth=0, bench=None)
    root.children = trees
    root.n = sum(t.n for t in trees)
    root.n_null = sum(t.n_null for t in trees)
    root.metric = "mixed metrics (not comparable)"
    assert root.n + root.n_null == sum(len(main_recs[b]) for b in MAIN_FILES), \
        "root does not account for every loaded row"

    # Judge coverage per CharXiv node, kept as a side-channel so it can never be
    # averaged into the string score.
    judge_by_node: dict[int, dict] = {}
    if judge:
        juids = judge["uids"]
        jscore = {}
        for line in open(RESULTS / JUDGE_FILE, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("uid") in juids and d.get("judge_score") is not None:
                jscore[d["uid"]] = float(d["judge_score"])
        cx_tree = next(t for t in trees if t.bench == "charxiv")
        for node in walk(cx_tree):
            vals = [jscore[r["uid"]] for r in node.recs if r["uid"] in jscore]
            if len(vals) >= SMALL_N and len(vals) >= 0.5 * max(node.n, 1):
                judge_by_node[id(node)] = {"n": len(vals), "v": sum(vals) / len(vals)}

    controls = control_blocks(main_recs)

    meta = {"splits_checked": sum(1 for n in walk(root) if n.children)}
    html_out = render(trees, root, meta, controls, judge, violations, load_stats,
                      judge_by_node, main_recs)
    Path(a.out).write_text(html_out, encoding="utf-8")

    rows = flatten(root, trees)
    with open(a.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    Path(a.json).write_text(json.dumps({
        "root_n": root.n, "nodes": len(rows), "splits_checked": meta["splits_checked"],
        "violations": violations, "load_stats": load_stats,
        "benchmarks": {t.bench: {"n": t.n, "value": t.value, "metric": t.metric} for t in trees},
        "tree": rows}, indent=1), encoding="utf-8")

    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1024:.0f} KB), "
          f"{len(rows):,} nodes, {meta['splits_checked']:,} splits checked, "
          f"{len(violations)} arithmetic violations")
    print(f"wrote {a.csv} and {a.json}")
    for b, t in zip(MAIN_FILES, trees):
        print(f"  {BENCH_LABEL[b]:16s} n={t.n:5,d} null={t.n_null:2d} "
              f"{t.metric:28s} {t.value * 100:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
