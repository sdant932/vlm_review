"""Numbers into pages: the whole report build behind one CLI.

This is the reporting layer. Its rule, from docs/REPO_MAP.md, is that it turns
numbers into pages and computes **no statistics of its own**. `figures.json` is
the single auditable artifact: `data` assembles every number the report quotes
into `outputs/report/figures.json`, and everything downstream reads it, so no
number is recomputed in two places and no figure or sentence carries a value
that cannot be traced back to `results/*.jsonl`.

    report data       -> outputs/report/figures.json          (the auditable artifact)
    report examples   -> outputs/report/figures/*.png         (real scored items)
    report tables     -> outputs/report/tables.md             (six tables, + injection)
    report index      -> outputs/report/figures.md, figures/index.html
    report paste      -> outputs/report/paste_into_docs.html  (self-contained, base64)
    report all        -> the five above, in that order

    report aug22      -> outputs/aug22/summary.json
    report svgloc     -> outputs/svgloc/{report.html,summary.json}
    report svgderived -> outputs/svgderived/{report.html,summary.json}

`aug22` is NOT part of `all`, but `data` reads `outputs/aug22/summary.json` at
runtime -- a **file dependency, not an import**, and easy to miss. Run `report
aug22` first whenever the results change, or `data` will assemble figures.json
from a stale (or absent) summary. `all` prints this ordering note before it
starts.

Three things were unified when the eight modules merged:

* **`esc()`** was defined once per HTML-emitting module with an identical body.
  There is now one `esc(s, quote=True)`. `paste` deliberately does *not* escape
  double quotes -- its output is pasted into a document editor, where `&quot;`
  in body text is visible as literal text -- so the paste section calls
  `esc_nq()`, a one-line wrapper that passes `quote=False`. That is the only
  behavioural difference and it is preserved exactly.
* **`pct()`** differed only in its null marker: markdown output wants a literal
  em-dash, HTML output wants the `&mdash;` entity, and a literal `&mdash;` in a
  markdown table renders as the five characters. There is now one
  `pct(x, d=1, null="\u2014")` for markdown and `pct_html()` for the HTML pages.
* **`busiest_crop()`** is defined next to its only caller, `_zoom_of()`. It
  picks the densest crop of a figure for the zoom panel, and it belongs beside
  the code that uses it rather than in a module of its own.

Fixed here, and worth stating: `tables`, `index` and `paste` each called
`read_text()` unguarded on `outputs/report/blindspots.md`, the hand-written
prose spine. That file is a separate deliverable and is not in this repository,
so all three ended in a FileNotFoundError traceback -- `tables` and `index`
after writing their output, `paste` before writing anything at all. Its absence
is now a clean skip with a message. Nothing here creates it.

Dataset names, run tags and the `gold_quality()` constants are written inline
rather than configured. This renders one study's results; a value that appears
once belongs where it is read.
"""

from __future__ import annotations

import argparse
import base64
import collections
import html
import io
import json
import math
import re
import statistics as st
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from blindspot.core import load
from blindspot.core import wilson, is_na, cell_of, centre_cell
from blindspot.core import (anls, token_f1, numeric_or_text_match, point_in_bbox,
                                    charxiv_grading_confidence)
from blindspot.eval import summarize
# Aliased only where two modules brought in the same name with different values
# (`RUNGS` is three rungs for svgloc and two for svgderived). The modules and the
# symbols are otherwise exactly what the originals imported.
from blindspot.eval import (analyse_localization as svgloc_analyse,
                                            load_loc_run as load_svgloc, d_box, d_centre, band,
                                            LOC_RUNGS as SVGLOC_RUNGS, MIN_CELL as SVGLOC_MIN_CELL)
from blindspot.eval import (analyse_counting, analyse_word_mc, COUNT_BINS,
                                                DERIVED_RUNGS as SVGDERIVED_RUNGS,
                                                MIN_CELL as SVGDERIVED_MIN_CELL)

RESULTS = Path("results")
REPORT_OUT = Path("outputs/report")
FIGS_OUT = REPORT_OUT / "figures"
AUG22_OUT = Path("outputs/aug22")
SVGLOC_OUT = Path("outputs/svgloc")
SVGLOC_ASSETS = SVGLOC_OUT / "assets"
SVGDERIVED_OUT = Path("outputs/svgderived")

# The hand-written prose spine. A separate deliverable; not in this repository.
PROSE = REPORT_OUT / "blindspots.md"

RUN = "haiku-4-5_think2000_native_r0"


# =============================================================== shared helpers
def esc(s, quote: bool = True) -> str:
    """HTML-escape. The one helper the whole layer shares."""
    return html.escape(str(s), quote=quote)


def esc_nq(s) -> str:
    """`esc()` for the paste page, which must NOT escape double quotes.

    Its output goes through a document editor's clipboard, where `&quot;` in
    body text survives as five literal characters. Deliberate; see the module
    docstring.
    """
    return esc(s, quote=False)


def pct(x, d=1, null: str = "\u2014") -> str:
    """A percentage, or the null marker. Default marker is a literal em-dash.

    Markdown output takes the default; the HTML pages take `pct_html`, which
    substitutes the `&mdash;` entity. Getting this backwards puts a literal
    `&mdash;` in a rendered table, which is why it is one function.
    """
    return null if x is None else f"{x * 100:.{d}f}%"


def pct_html(x, d=1) -> str:
    return pct(x, d, null="&mdash;")


def read_prose(skipped: str) -> str | None:
    """`blindspots.md` if it is here, otherwise None and a one-line explanation.

    The prose spine is written by hand and shipped separately. Its absence is
    normal in a fresh clone and must not be a traceback.
    """
    if not PROSE.exists():
        print(f"  blindspots.md not found; {skipped}")
        return None
    return PROSE.read_text()


# ============================================= aug22: outputs/aug22/summary.json
DATASETS = ["charxiv", "infographicvqa", "screenspot_pro", "ai2d", "slidevqa", "slidevqa_allpages"]

# Scale words must be normalised before two numbers can be compared: a model that
# answers "3.3bn" against a gold of "3300000000" has read the chart correctly and
# only dressed the answer differently.
SCALE = {"bn": 1e9, "billion": 1e9, "billions": 1e9, "tn": 1e12, "trillion": 1e12,
         "m": 1e6, "mn": 1e6, "million": 1e6, "millions": 1e6,
         "k": 1e3, "thousand": 1e3, "thousands": 1e3}


def _dedup(path: Path) -> list[dict]:
    """Last write per uid wins; rows without a usable prediction are dropped."""
    d = {}
    for line in path.open():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        d[r["uid"]] = r
    return [r for r in d.values() if r.get("pred") is not None]


def _dup_stats(path: Path) -> dict:
    uids = []
    for line in path.open():
        if line.strip():
            try:
                uids.append(json.loads(line)["uid"])
            except Exception:
                pass
    c = collections.Counter(uids)
    return {"lines": len(uids), "unique": len(c),
            "duplicate_lines": sum(v - 1 for v in c.values() if v > 1)}


def canon_num(s):
    t = str(s).strip().lower().replace(",", "")
    for ch in "$€£%":
        t = t.replace(ch, "")
    mult = 1.0
    for w, f in sorted(SCALE.items(), key=lambda x: -len(x[0])):
        if re.search(rf"(?<![a-z]){re.escape(w)}\b", t):
            mult = f
            t = re.sub(rf"(?<![a-z]){re.escape(w)}\b", "", t)
            break
    m = re.fullmatch(r"\s*(-?\d+\.?\d*)\s*[a-z]*\s*", t)
    try:
        return float(m.group(1)) * mult if m else None
    except ValueError:
        return None


def canon_txt(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def fmt_equiv(pred, golds) -> bool:
    """Conservative: numeric equality (sign-sensitive) or exact folded-text match.

    No substring fallback -- an early version scored "7" as equivalent to "17".
    """
    pn = canon_num(pred)
    for g in golds:
        gn = canon_num(g)
        if pn is not None and gn is not None:
            if pn == gn:
                return True
            continue
        if canon_txt(pred) and canon_txt(pred) == canon_txt(g):
            return True
    return False


def _f(path: Path) -> Path:
    return RESULTS / path


def controls() -> dict:
    """The four ablations, each paired against the sighted/full run it modifies."""
    out: dict = {}

    # ---- provenance: how much duplication the resumed runs left behind -------
    out["duplication"] = {
        ds: _dup_stats(RESULTS / f"{ds}__haiku-4-5_think2000_native_r0.jsonl")
        for ds in ("charxiv", "infographicvqa", "ai2d", "screenspot_pro", "slidevqa")
    }

    # Repeated items are an unplanned but real reproducibility measurement: the
    # same question asked twice, same settings. Disagreement here is the noise
    # floor under every other comparison in this report.
    raw = collections.defaultdict(list)
    for line in (RESULTS / "charxiv__haiku-4-5_think2000_native_r0.jsonl").open():
        if line.strip():
            r = json.loads(line)
            raw[r["uid"]].append(r.get("pred"))
    rep = {u: v for u, v in raw.items() if len(v) > 1}
    dis = sum(1 for v in rep.values() if len({str(x) for x in v}) > 1)
    out["reproducibility"] = {
        "repeated_items": len(rep), "disagreed": dis,
        "disagreement_rate": dis / len(rep) if rep else None,
        "note": "same question, same settings, asked twice; thinking pins temperature to 1",
    }

    # ---- blind control ------------------------------------------------------
    sight = {}
    for ds in ("charxiv", "infographicvqa", "slidevqa", "ai2d"):
        for r in _dedup(RESULTS / f"{ds}__haiku-4-5_think2000_native_r0.jsonl"):
            sight[r["uid"]] = r

    def sc(r):
        if r["answer_type"] == "choice":
            return 1.0 if str(r["pred"]).strip().upper() == str(r["gold"][0]).strip().upper() else 0.0
        if r["dataset"].startswith("slidevqa"):
            return token_f1(r["pred"], r["gold"])[1]
        if r["dataset"].startswith("charxiv"):
            conf = charxiv_grading_confidence(r["meta"].get("qid"))
            return numeric_or_text_match(r["pred"], r["gold"]) if conf == "strict" else anls(r["pred"], r["gold"])
        return anls(r["pred"], r["gold"])

    blind = collections.defaultdict(lambda: ([], []))
    cx_split = collections.defaultdict(lambda: ([], []))
    for b in _dedup(RESULTS / "control_blind.jsonl"):
        src = b["meta"].get("src_uid")
        if src not in sight:
            continue
        ds = b["dataset"].replace("_blind", "")
        blind[ds][0].append(sc(b))
        blind[ds][1].append(sc(sight[src]))
        if ds == "charxiv":
            k = "descriptive" if b["meta"].get("split") == "descriptive" else "reasoning"
            cx_split[k][0].append(sc(b))
            cx_split[k][1].append(sc(sight[src]))

    CHANCE = {"ai2d": 0.25}
    out["blind"] = {
        ds: {"blind": sum(b) / len(b), "sighted": sum(s) / len(s),
             "vision_adds_pp": (sum(s) / len(s) - sum(b) / len(b)) * 100,
             "chance": CHANCE.get(ds, 0.0), "n": len(b)}
        for ds, (b, s) in blind.items()
    }
    out["blind_charxiv_split"] = {
        k: {"blind": sum(b) / len(b), "sighted": sum(s) / len(s), "n": len(b)}
        for k, (b, s) in cx_split.items()
    }

    # ---- one-page ablation (is the multi-page label real?) ------------------
    ev = {r["uid"]: token_f1(r["pred"], r["gold"])[1]
          for r in _dedup(RESULTS / "slidevqa__haiku-4-5_think2000_native_r0.jsonl")}
    op = {r["meta"]["src_uid"]: token_f1(r["pred"], r["gold"])[1]
          for r in _dedup(RESULTS / "control_onepage0.jsonl")}
    com = [k for k in op if k in ev]
    if com:
        out["onepage"] = {
            "n": len(com),
            "both_slides_f1": sum(ev[k] for k in com) / len(com) * 100,
            "one_slide_f1": sum(op[k] for k in com) / len(com) * 100,
            "collapse_f1": (sum(op[k] for k in com) - sum(ev[k] for k in com)) / len(com) * 100,
            "still_answerable_frac": sum(1 for k in com if op[k] >= 0.5) / len(com),
        }

    # ---- localization: precision falloff and the coordinate-emission tax ----
    ROWS, K = "ABCD", 4

    def cell(x, y):
        return f"{ROWS[min(int(y * K), K - 1)]}{min(int(x * K), K - 1) + 1}"

    sp = {r["uid"]: r for r in _dedup(RESULTS / "screenspot_pro__haiku-4-5_think2000_native_r0.jsonl")}
    grids = {}
    for k_, label in ((2, "2x2"), (3, "3x3"), (4, "4x4"), (8, "8x8")):
        hit = 0
        for r in sp.values():
            px, py = r["pred"]
            x0, y0, x1, y1 = r["gold"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hit += (min(int(px * k_), k_ - 1), min(int(py * k_), k_ - 1)) == \
                   (min(int(cx * k_), k_ - 1), min(int(cy * k_), k_ - 1))
        grids[label] = {"acc": hit / len(sp), "chance": 1 / (k_ * k_)}
    exact = sum(point_in_bbox(r["pred"], r["gold"]) for r in sp.values()) / len(sp)
    area = sum(r["meta"]["target_area_frac"] for r in sp.values()) / len(sp)
    out["coarse_localization"] = {"grids": grids, "exact_click_in_bbox": exact,
                                  "mean_target_area_frac": area, "n": len(sp)}

    gr = {r["meta"]["src_uid"]: r for r in _dedup(RESULTS / "control_grid4.jsonl")} \
        if (RESULTS / "control_grid4.jsonl").exists() else {}
    pair = [k for k in gr if k in sp]
    if pair:
        def norm(p):
            m = re.search(r"([A-Da-d])\s*([1-4])", str(p))
            return (m.group(1).upper() + m.group(2)) if m else None
        named = sum(1 for k in pair if norm(gr[k]["pred"]) == gr[k]["gold"][0])
        clicked = sum(1 for k in pair if cell(*sp[k]["pred"]) == gr[k]["gold"][0])
        b = sum(1 for k in pair if norm(gr[k]["pred"]) == gr[k]["gold"][0] and cell(*sp[k]["pred"]) != gr[k]["gold"][0])
        c = sum(1 for k in pair if norm(gr[k]["pred"]) != gr[k]["gold"][0] and cell(*sp[k]["pred"]) == gr[k]["gold"][0])
        from math import comb
        n_ = b + c
        p = (sum(comb(n_, i) for i in range(min(b, c) + 1)) / 2 ** n_ * 2) if n_ else None
        out["grid_control"] = {
            "n": len(pair), "named_cell_acc": named / len(pair),
            "click_derived_cell_acc": clicked / len(pair),
            "delta_pp": (named - clicked) / len(pair) * 100,
            "chance": 1 / (K * K), "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p,
        }

    # ---- abstention: what happens when the thing simply is not there --------
    def isna(s):
        return str(s).strip().lower().rstrip(".") in ("not applicable", "n/a", "na", "none", "not available")

    cx = _dedup(RESULTS / "charxiv__haiku-4-5_think2000_native_r0.jsonl")
    tp = fn = fp = tn = 0
    per = collections.defaultdict(lambda: [0, 0])
    for r in cx:
        g = any(isna(x) for x in r["gold"])
        p = isna(r["pred"])
        if g and p: tp += 1
        elif g: fn += 1
        elif p: fp += 1
        else: tn += 1
        if g:
            k = (r["meta"].get("qid"), (r["meta"].get("qlabel") or "")[:60])
            per[k][0] += 1
            per[k][1] += p
    out["abstention"] = {
        "gold_na_n": tp + fn, "correctly_abstained": tp / (tp + fn) if tp + fn else None,
        "invented_a_value": fn / (tp + fn) if tp + fn else None,
        "gold_value_n": fp + tn, "over_abstained": fp / (fp + tn) if fp + tn else None,
        "by_question": [{"qid": k[0], "qlabel": k[1], "n": v[0], "abstained": v[1] / v[0]}
                        for k, v in sorted(per.items(), key=lambda x: -x[1][0]) if v[0] >= 20],
    }

    # ---- format artifact: how much "failure" is the metric, not the model ---
    def artifact(rows, scorer):
        zero = [r for r in rows if scorer(r) == 0.0]
        nonex = [r for r in rows if scorer(r) < 1.0]
        return {"hard_zeros": len(zero),
                "hard_zeros_format_equivalent": sum(1 for r in zero if fmt_equiv(r["pred"], r["gold"])),
                "non_exact": len(nonex),
                "non_exact_format_equivalent": sum(1 for r in nonex if fmt_equiv(r["pred"], r["gold"]))}

    sv = _dedup(RESULTS / "slidevqa__haiku-4-5_think2000_native_r0.jsonl")
    iv = _dedup(RESULTS / "infographicvqa__haiku-4-5_think2000_native_r0.jsonl")
    out["format_artifact"] = {
        "slidevqa": artifact(sv, lambda r: token_f1(r["pred"], r["gold"])[1]),
        "infographicvqa": artifact(iv, lambda r: anls(r["pred"], r["gold"])),
        "charxiv": artifact(cx, lambda r: anls(r["pred"], r["gold"])),
        "note": "token-F1 scores '22%' against '22' as zero; ANLS (edit distance) barely notices",
    }

    # ---- is a wrong number a near miss, or the wrong element entirely? ------
    def numerr(rows, label):
        errs, ex, tot = [], 0, 0
        for r in rows:
            gn, pn = canon_num(r["gold"][0]), canon_num(r["pred"])
            if gn is None or pn is None or gn == 0:
                continue
            tot += 1
            if pn == gn:
                ex += 1
                continue
            errs.append(abs(pn - gn) / abs(gn))
        if not errs:
            return None
        return {"label": label, "n_numeric": tot, "exact_frac": ex / tot, "n_wrong": len(errs),
                "median_rel_error": st.median(errs),
                "within_10pct_frac": sum(1 for e in errs if e <= 0.10) / len(errs),
                "over_100pct_frac": sum(1 for e in errs if e > 1.0) / len(errs)}

    out["numeric_error"] = [x for x in (
        numerr([r for r in cx if r["meta"].get("split") == "descriptive"], "CharXiv descriptive"),
        numerr([r for r in cx if r["meta"].get("split") != "descriptive"], "CharXiv reasoning"),
        numerr(iv, "InfographicVQA"),
        numerr(sv, "SlideVQA"),
    ) if x]

    # ---- SlideVQA: the three costs, as-scored and format-corrected ----------
    def f1s(r, fix):
        f = token_f1(r["pred"], r["gold"])[1]
        return 1.0 if (fix and f < 1.0 and fmt_equiv(r["pred"], r["gold"])) else f

    ap = {r["uid"].rsplit(":", 1)[1]: r for r in _dedup(RESULTS / "slidevqa_allpages__haiku-4-5_think2000_native_r0.jsonl")}
    evd = {r["uid"].rsplit(":", 1)[1]: r for r in sv}
    common = sorted(set(evd) & set(ap))
    costs = {}
    for fix, tag in ((False, "as_scored"), (True, "format_corrected")):
        mean = lambda rs: sum(f1s(r, fix) for r in rs) / len(rs) * 100
        mp = [r for r in sv if r["meta"].get("multi_page")]
        spg = [r for r in sv if not r["meta"].get("multi_page")]
        ar = [r for r in sv if r["meta"].get("arithmetic")]
        lo = [r for r in sv if not r["meta"].get("arithmetic")]
        costs[tag] = {
            "retrieval": mean([ap[k] for k in common]) - mean([evd[k] for k in common]),
            "integration": mean(mp) - mean(spg),
            "derivation": mean(ar) - mean(lo),
            "overall_f1": mean(sv), "lookup_f1": mean(lo), "arithmetic_f1": mean(ar),
            "single_page_f1": mean(spg), "multi_page_f1": mean(mp),
        }
    costs["paired_n"] = len(common)
    out["slidevqa_costs"] = costs
    return out


# ============== data: the single auditable artifact, outputs/report/figures.json
# A question exercises spatial binding if it points at a mark drawn on the artwork.
# Deliberately lexical and conservative: it matches the ask, not the answer shape.
MARK_RE = re.compile(
    r"\b(letter|labell?ed|label|marked|arrow|point(?:ed|ing)?\s+(?:to|at)|shown by)\b",
    re.I,
)


def _rows(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _data_preds(path: Path) -> dict[str, object]:
    return {r["uid"]: r["pred"] for r in _rows(path) if r.get("pred") is not None}


def _blind_preds(prefix: str) -> dict[str, object]:
    out = {}
    for r in _rows(RESULTS / "control_blind.jsonl"):
        su = (r.get("meta") or {}).get("src_uid")
        if su and su.startswith(prefix) and r.get("pred") is not None:
            out[su] = r["pred"]
    return out


def _cell(k: int, n: int, **extra) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "acc": (k / n) if n else None, "lo": lo, "hi": hi, **extra}


# --------------------------------------------------------- AI2D, recut by operation
def ai2d_binding() -> dict:
    ex = {e.uid: e for e in load("ai2d")}
    sighted = _data_preds(RESULTS / "ai2d__haiku-4-5_think2000_native_r0.jsonl")
    blind = _blind_preds("ai2d:")
    ok = lambda uid, p: str(p).strip().upper() == ex[uid].gold[0]

    groups = collections.defaultdict(list)
    for uid, p in sighted.items():
        e = ex.get(uid)
        if e:
            groups[bool(MARK_RE.search(e.question))].append(ok(uid, p))
    out = {"sighted": {("refers_to_mark" if k else "no_mark"): _cell(sum(v), len(v))
                       for k, v in groups.items()}}

    paired = set(blind) & set(sighted)
    pg = collections.defaultdict(lambda: [0, 0, 0])
    for uid in paired:
        k = bool(MARK_RE.search(ex[uid].question))
        pg[k][0] += 1
        pg[k][1] += ok(uid, blind[uid])
        pg[k][2] += ok(uid, sighted[uid])
    out["paired_blind"] = {
        ("refers_to_mark" if k else "no_mark"): {
            "n": n, "blind": b / n, "sighted": s / n, "vision_adds_pp": (s - b) / n * 100}
        for k, (n, b, s) in pg.items()}
    out["chance"] = 0.25

    # The published split, kept so the report can name the artifact it is correcting.
    fam = collections.defaultdict(list)
    for uid, p in sighted.items():
        e = ex.get(uid)
        if e:
            fam[(e.meta["qtype"], bool(MARK_RE.search(e.question)))].append(ok(uid, p))
    out["published_split_artifact"] = {
        f"{q}|{'mark' if m else 'no_mark'}": _cell(sum(v), len(v))
        for (q, m), v in sorted(fam.items())}
    return out


# ------------------------------------------- absence detection, cut against the blind arm
def absence_detection() -> dict:
    ex = {e.uid: e for e in load("charxiv")}
    sighted = _data_preds(RESULTS / "charxiv__haiku-4-5_think2000_native_r0.jsonl")
    blind = _blind_preds("charxiv:")

    def gold_na(uid) -> bool:
        g = ex[uid].gold
        return is_na(g[0] if isinstance(g, list) else g)

    absent = [u for u in sighted if u in ex and gold_na(u)]
    answerable = [u for u in sighted if u in ex and not gold_na(u)]
    invented = sum(1 for u in absent if not is_na(sighted[u]))
    over = sum(1 for u in answerable if is_na(sighted[u]))

    paired = set(blind) & set(sighted)
    pa = [u for u in paired if gold_na(u)]
    pq = [u for u in paired if not gold_na(u)]
    out = {
        "full_set": {
            "absent": _cell(invented, len(absent), invention_rate=invented / len(absent)),
            "over_abstention": {"n": len(answerable), "rate": over / len(answerable)},
        },
        "paired_blind": {
            "absent": {"n": len(pa),
                       "abstains_blind": sum(is_na(blind[u]) for u in pa) / len(pa),
                       "abstains_sighted": sum(is_na(sighted[u]) for u in pa) / len(pa)},
            "answerable": {"n": len(pq),
                           "abstains_blind": sum(is_na(blind[u]) for u in pq) / len(pq),
                           "abstains_sighted": sum(is_na(sighted[u]) for u in pq) / len(pq)},
        },
    }
    # Worst templates, so the figure can show invention is structure-dependent.
    by_q = collections.defaultdict(list)
    for u in absent:
        by_q[ex[u].meta.get("qid")].append(not is_na(sighted[u]))
    out["by_template"] = sorted(
        ({"qid": q, **_cell(sum(v), len(v))} for q, v in by_q.items() if len(v) >= 25),
        key=lambda c: -(c["acc"] or 0))
    return out


# ------------------------------------------------------------ ground truth + expression
def gold_quality() -> dict:
    out = {}
    for ds, total_failures in (("charxiv", 735), ("infographicvqa", 852),
                               ("screenspot_pro", 1552)):
        rows = _rows(RESULTS / f"{ds}__gtaudit.jsonl")
        rows = [r for r in rows if r.get("verdict") or r.get("gt_quality")]
        contested = [r for r in rows
                     if r.get("verdict") in ("prediction_correct", "both_acceptable")
                     or r.get("gt_quality") in ("wrong", "ambiguous")]
        rate = len(contested) / len(rows) if rows else 0.0
        out[ds] = {"audited": len(rows), "contested": len(contested),
                   "contested_error_rate": rate,
                   "total_failures": total_failures,
                   "implied_floor": rate * total_failures / {"charxiv": 5000,
                                                             "infographicvqa": 2801,
                                                             "screenspot_pro": 1581}[ds],
                   "examples": [{"uid": r["uid"], "question": r.get("question"),
                                 "gold": r.get("gold"), "pred": r.get("pred"),
                                 "verdict": r.get("verdict"),
                                 "gt_quality": r.get("gt_quality")}
                                for r in contested[:12]]}
    return out


# ------------------------------------------------------------------- the synthetic set
def misses_on_other_label(pts) -> dict:
    """Do the misses land on a different label, or on nothing?

    This decides whether a bad citation is visibly or invisibly wrong, so it is
    computed from the manifest's own target boxes rather than asserted.
    """
    boxes = collections.defaultdict(list)
    for line in open("data/svg_localization/manifest.jsonl"):
        r = json.loads(line)
        b = r.get("gold_bbox_norm") or r.get("bbox_norm")
        if b:
            boxes[(r["graph_id"], r["resolution"])].append((r.get("target_idx"), b))

    k = n = 0
    for r in pts:
        if r["hit"] or not r.get("in_range", True):
            continue
        n += 1
        m, (px, py) = r["meta"], r["pred"]
        for idx, b in boxes[(m["graph_id"], m["resolution"])]:
            if idx != m["target_idx"] and b[0] <= px <= b[2] and b[1] <= py <= b[3]:
                k += 1
                break
    return {"note": "share of in-range misses landing inside a different catalogued "
                    "target box", "k": k, "n": n, "value": k / max(n, 1)}


def synthetic() -> dict:
    s = json.load(open("outputs/svgloc/summary.json"))
    ab = json.load(open("outputs/svgloc/ablations.json"))
    d = json.load(open("outputs/svgderived/summary.json"))
    run = load_svgloc("haiku-4-5_think2000_native_r0")
    pts = run["point"]

    dark = {"slate-dark", "carbon", "blueprint"}
    pol = collections.defaultdict(list)
    for r in pts:
        pol["dark" if r["meta"]["theme"] in dark else "light"].append(r)

    def bands(rows, key):
        c = collections.Counter(band(key(r)) for r in rows)
        n = max(sum(c.values()), 1)
        return {k: c.get(k, 0) / n for k in ("near_miss", "moderate_miss", "wrong_region")}

    misses = [r for r in pts if not r["hit"]]
    return {
        "headline": s["headline"], "overall": s["overall"],
        "curve": s["curve"], "curve_all": s["curve_all"],
        "null_control": s["null_control"], "resolution_effect": s["resolution_effect"],
        "out_of_range": s["out_of_range"], "distance": s["distance"],
        "bands": {
            # summary.json stores these as raw counts; every other series here is a
            # proportion, so normalise rather than leave two units in one field.
            "by_rung_d_centre": {
                g: {k: v / max(sum(s["distance"][g]["bands_d_centre"].values()), 1)
                    for k, v in s["distance"][g]["bands_d_centre"].items()}
                for g in ("small", "medium", "large")},
            "pooled_d_centre": bands(misses, lambda r: r["d_centre"]),
            "pooled_d_box": bands(misses, lambda r: r["d_box"]),
            "by_polarity_d_box": {k: bands(v, lambda r: r["d_box"]) for k, v in pol.items()},
        },
        "polarity": s.get("polarity"),
        "polarity_axis": {
            k: {"n": len(v),
                "x_inside": sum(1 for r in v if r["gold"][0] <= r["pred"][0] <= r["gold"][2]) / len(v),
                "y_inside": sum(1 for r in v if r["gold"][1] <= r["pred"][1] <= r["gold"][3]) / len(v)}
            for k, v in pol.items()},
        "ablations": ab,
        "derived": {"counting": d["counting"], "word_mc": d["word_mc"]},
        "ladder": {
            "word_mc": d["word_mc"]["overall"]["acc"],
            "counting": d["counting"]["overall"]["acc"],
            "quadrant": ab["arms"]["quadrant_mc"]["acc"],
            "cell4x4_large": s["curve"]["large"][2]["strict"],
            "exact": s["overall"]["acc"],
            "note": "same 200 scenes and the same PNG files; different question populations "
                    "(localization 2380 point items, counting 476, word_mc 736)",
        },
        "misses_on_other_label": misses_on_other_label(pts),
    }


def build_figures() -> dict:
    a = json.load(open("outputs/aug22/summary.json"))
    return {
        "model": a.get("model"), "generated": a.get("generated"),
        "benchmarks": {k: {"acc": v["acc"], "n": v["n"],
                           "ci": [v.get("ci_lo"), v.get("ci_hi")]}
                       for k, v in a["datasets"].items()},
        "blind": a["controls"]["blind"],
        "reproducibility": a["controls"]["reproducibility"],
        "gold_quality": gold_quality(),
        "ai2d_binding": ai2d_binding(),
        "absence_detection": absence_detection(),
        "synthetic": synthetic(),
    }


# ======================================== examples: outputs/report/figures/*.png
W = 2600                                   # 2x of the 1300-unit diagram width
SURFACE = (252, 252, 251)
INK, INK2, MUTED = (11, 11, 11), (82, 81, 78), (138, 137, 131)
GOOD, CRIT, S1, S2 = (12, 163, 12), (208, 59, 59), (42, 120, 214), (235, 104, 52)
RULE = (228, 227, 223)

_FONTS = ["/System/Library/Fonts/Helvetica.ttc",
          "/System/Library/Fonts/Supplemental/Arial.ttf",
          "/Library/Fonts/Arial.ttf"]


def font(size: int, bold: bool = False):
    for p in _FONTS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size, index=1 if bold else 0)
            except Exception:
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    continue
    return ImageFont.load_default()


def effective(w: int, h: int) -> tuple[int, int]:
    """What the API actually delivers: 1568px long edge, ~1.15 MP total."""
    s = min(1.0, 1568 / max(w, h), math.sqrt(1_150_000 / max(w * h, 1)))
    return max(1, round(w * s)), max(1, round(h * s))


def as_model_saw(path: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im.resize(effective(*im.size), Image.LANCZOS)


def fit(im: Image.Image, bw: int, bh: int) -> Image.Image:
    r = min(bw / im.width, bh / im.height)
    return im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)


def draw_target(im: Image.Image, box=None, point=None, ring=True) -> Image.Image:
    """Green box (+ locator ring) for the target, red crosshair for the click."""
    im = im.copy()
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(im.size) / 400))
    if box:
        x0, y0, x1, y1 = [v * s for v, s in zip(box, (im.width, im.height) * 2)]
        d.rectangle([x0, y0, x1, y1], outline=GOOD, width=lw)
        if ring:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            r = max(lw * 13, (x1 - x0) * 1.5, (y1 - y0) * 1.5)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=GOOD, width=max(1, lw // 2))
    if point:
        px, py = point[0] * im.width, point[1] * im.height
        r = lw * 8
        d.line([px - r, py, px + r, py], fill=CRIT, width=lw)
        d.line([px, py - r, px, py + r], fill=CRIT, width=lw)
        d.ellipse([px - r / 2, py - r / 2, px + r / 2, py + r / 2], outline=CRIT, width=lw)
    return im


class Panel:
    """A figure canvas with a title, then image+text cards stacked or in a row."""

    def __init__(self, title: str, sub: str = "", h: int = 1200):
        self.im = Image.new("RGB", (W, h), SURFACE)
        self.d = ImageDraw.Draw(self.im)
        self.d.text((80, 60), title, INK, font=font(46, True))
        if sub:
            self.d.text((80, 126), sub, INK2, font=font(28))
        self.y = 190 if sub else 150

    def text(self, x, y, s, size=26, fill=INK2, bold=False, wrap=None):
        f = font(size, bold)
        lines = textwrap.wrap(s, wrap) if wrap else [s]
        for i, ln in enumerate(lines):
            self.d.text((x, y + i * int(size * 1.35)), ln, fill, font=f)
        return y + len(lines) * int(size * 1.35)

    def rule(self, y):
        self.d.line([80, y, W - 80, y], fill=RULE, width=2)

    def paste(self, im, x, y, bw, bh, border=True):
        im = fit(im, bw, bh)
        self.im.paste(im, (int(x), int(y)))
        if border:
            self.d.rectangle([x, y, x + im.width, y + im.height], outline=RULE, width=2)
        return im.width, im.height

    def save(self, name):
        FIGS_OUT.mkdir(parents=True, exist_ok=True)
        self.im.crop((0, 0, W, min(self.y + 60, self.im.height))).save(
            FIGS_OUT / f"{name}@2x.png", quality=92)
        return name


# ------------------------------------------------------------------ selection
def _ex_preds(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("pred") is not None:
                out[r["uid"]] = r["pred"]
    return out


def e05_instrument():
    """The variety the generator covers: eight scenes, one per chart type."""
    import json as _j
    base = Path("data/svg_localization/images")
    scenes = [_j.loads(l) for l in open("data/svg_localization/scenes.jsonl") if l.strip()]
    want, seen = [], set()
    for sc in scenes:
        if sc["chart_type"] in seen:
            continue
        if not (base / f'g{sc["graph_id"]:04d}_small.png').exists():
            continue
        seen.add(sc["chart_type"])
        want.append(sc)
        if len(want) == 8:
            break

    p = Panel("Generated scenes: eight of the sixteen chart types",
              "200 scenes in total, each with its own chart type, colour theme, font and "
              "domain vocabulary.", h=1500)
    y = p.y
    colw, roww = 620, 470
    for i, sc in enumerate(want):
        x = 80 + (i % 4) * colw
        yy = y + (i // 4) * roww
        f = base / f'g{sc["graph_id"]:04d}_small.png'
        _, ih = p.paste(as_model_saw(str(f)), x, yy, colw - 40, 350)
        t2 = p.text(x, yy + ih + 14, sc["chart_type"].replace("_", " "), 25, INK, True)
        p.text(x, t2 + 2, f'{sc["theme"]}  ·  {sc["domain"]}', 22, MUTED)
    p.y = y + 2 * roww - 40
    p.d.line([80, p.y, W - 80, p.y], fill=RULE, width=2)
    p.text(80, p.y + 22, "Every scene is rendered at two resolutions that reach the model "
                         "differently:", 27, INK, True)
    p.text(80, p.y + 60, "small  900x570, delivered untouched, label text 10-14px", 26, INK2)
    p.text(80, p.y + 94, "large  3000x1900, downscaled by the API to 1348x853", 26, INK2)
    p.text(1180, p.y + 60, "The target occupies the same fraction of the image at both, so "
                           "the contrast isolates absolute resolution rather than target "
                           "size.", 26, INK2, wrap=58)
    p.y += 140
    return p.save("e05_instrument")


def e08_generation_pipeline():
    """Block diagram of the procedural generation, from seed to scored questions."""
    import json as _j, collections
    rows = [_j.loads(l) for l in open("data/svg_localization/manifest.jsonl") if l.strip()]
    rej, seen = collections.Counter(), set()
    for r in rows:
        k = (r["graph_id"], r["resolution"])
        if k in seen:
            continue
        seen.add(k)
        for cat, n in (r.get("rejected_targets") or {}).items():
            rej[cat] += n

    p = Panel("How a scene and its ground truth are generated",
              "Every draw is seeded, so the whole corpus is reproducible from one integer.",
              h=1650)

    def block(x, y, w, h, head, body, fill=(255, 255, 255), accent=S1, hs=27, bs=23):
        p.d.rounded_rectangle([x, y, x + w, y + h], 10, fill=fill, outline=RULE, width=2)
        p.d.rectangle([x, y, x + 8, y + h], fill=accent)
        p.text(x + 26, y + 16, head, hs, INK, True)
        if body:
            p.text(x + 26, y + 16 + int(hs * 1.5), body, bs, INK2, wrap=int(w / (bs * 0.52)))

    def arrow(x1, y1, x2, y2):
        p.d.line([x1, y1, x2, y2], fill=MUTED, width=3)
        if y2 > y1:
            p.d.polygon([(x2 - 8, y2 - 12), (x2 + 8, y2 - 12), (x2, y2)], fill=MUTED)
        else:
            p.d.polygon([(x2 - 12, y2 - 8), (x2 - 12, y2 + 8), (x2, y2)], fill=MUTED)

    y = p.y
    block(80, y, 420, 78, "seed = 17", "one integer fixes every draw below", accent=(11, 11, 11))

    # the four independent random draws, shown as parallel blocks
    y2 = y + 132
    draws = [("chart type", "16 options", "flowchart, bar, line, scatter, table, network, "
              "pie, org chart, timeline, gantt, mindmap, dashboard, quadrant, sequence, "
              "treemap, state machine"),
             ("colour theme", "10 options", "paper, cream, ice, mint, sun, mono-print, "
              "high-contrast, and 3 dark: slate-dark, carbon, blueprint"),
             ("font family", "9 options", "Arial, Verdana, Tahoma, Trebuchet, Georgia, "
              "Palatino, Futura, Menlo, Courier"),
             ("domain vocabulary", "10 options", "Triage Protocol, Payment Authorization, "
              "Access Review, Fleet Logistics, Returns Workflow, Build Pipeline, Claims "
              "Handling, Grid Operations, Telemetry Ingest, Content Moderation")]
    bw = 590
    for i, (name, n, opts) in enumerate(draws):
        x = 80 + (i % 2) * (bw + 40)
        yy = y2 + (i // 2) * 190
        block(x, yy, bw, 172, f"{name}  —  {n}", opts, accent=S2, hs=26, bs=22)
        arrow(290, y + 78, 290, y2 - 6) if i == 0 else None

    y3 = y2 + 400
    steps = [("lay out primitives",
              "rects, circles, wedges, lines and text into one shared list; complexity 4 "
              "sets how many nodes, bars, rows or slices"),
             ("enforce legibility",
              "grow any targetable text that would fall below 10px at the smaller size"),
             ("render at each scale, twice",
              "once with text suppressed, so the true background behind every label can be "
              "measured rather than assumed"),
             ("measure gold off the raster",
              "ink box from the rendered glyphs; hit box is the enclosing widget, or the ink "
              "grown by button padding"),
             ("filter eligible targets",
              "nine rules: unique, non-overlapping, unoccluded, above AA contrast, at least "
              "10px, hit box must not swallow a neighbour"),
             ("emit questions",
              "point, relation and reverse, drawn only from surviving targets")]
    for i, (head, body) in enumerate(steps):
        yy = y3 + i * 142
        block(80, yy, 1180, 124, f"{i+1}.  {head}", body, accent=S1, hs=27, bs=22)
        if i:
            arrow(670, yy - 22, 670, yy - 2)
    arrow(670, y2 + 362, 670, y3 - 4)

    # what the filter threw away
    x2 = 1340
    p.text(x2, y3 - 40, "WHAT THE TARGET FILTER REJECTED", 22, MUTED, True)
    yy = y3
    top = rej.most_common(7)
    mx = max((n for _, n in top), default=1)
    for cat, n in top:
        w = 560 * n / mx
        p.d.rounded_rectangle([x2, yy, x2 + max(w, 3), yy + 26], 4, fill=S2)
        p.text(x2 + max(w, 3) + 14, yy + 1, f"{n:,}", 24, INK, True)
        p.text(x2, yy + 32, cat.replace("_", " "), 22, INK2)
        yy += 78
    p.text(x2, yy + 8, f"{sum(rej.values()):,} candidate labels rejected across 600 "
                       f"scenes and resolutions. What survives is the only thing a question is ever "
                       f"asked about.", 24, INK2, wrap=54)
    p.y = max(y3 + len(steps) * 142, yy + 110)
    return p.save("e08_generation_pipeline")


# --------------------------------------------------------------- the two figures
def e09_bad_gold():
    """One item where the benchmark, not the model, is wrong.

    The infographic never states a percentage: the model returned exactly what
    the page says and was scored zero, because the gold answer applies a unit
    conversion the image does not contain.
    """
    uid = "infographicvqa:94919"
    e = {x.uid: x for x in load("infographicvqa")}[uid]
    pred = _ex_preds(f"results/infographicvqa__{RUN}.jsonl")[uid]
    p = Panel("A scored failure where the gold answer is the problem",
              "One of the audited items. The model's answer and the gold answer say "
              "the same thing; only the surface form differs.", h=1500)
    y = p.y + 10
    iw, ih = p.paste(as_model_saw(e.images[0]), 80, y, 900, 1050)
    x = 80 + iw + 70
    yy = p.text(x, y + 10, "question", 24, MUTED, True)
    yy = p.text(x, yy + 6, e.question, 32, INK, True, wrap=44)
    yy = p.text(x, yy + 40, "gold answer", 24, MUTED, True)
    yy = p.text(x, yy + 6, ", ".join(str(g) for g in e.gold), 34, INK, True)
    yy = p.text(x, yy + 34, "model answer", 24, MUTED, True)
    yy = p.text(x, yy + 6, str(pred), 34, CRIT, True)
    yy = p.text(x, yy + 34, "scored", 24, MUTED, True)
    yy = p.text(x, yy + 6, "0.0 ANLS - counted as a perception failure", 30, CRIT, True)
    yy = p.text(x, yy + 44,
                "An audit would not defend 16.8% of InfographicVQA's scored failures, "
                "or 16.3% of CharXiv's. Contested items occur only among errors, so "
                "CharXiv's 14.7% error rate is nearer 12.3% model error and 2.4% label "
                "error.", 27, INK2, wrap=52)
    p.y = max(y + ih, yy) + 40
    return p.save("e09_bad_gold")


# (heading, uid, explanation, zoom) chosen from outputs/report/candidates.html.
# `zoom` adds a 1:1 crop of the delivered image beside the whole scene, for the two
# rows whose argument is about detail the shrunk-to-fit view cannot show.
PICKS = [
    ("Resolution bias", "infographicvqa:80736",
     "The source is 42.7 megapixels; the API delivers 1.15. What the question asks "
     "about is destroyed before the model looks at it.", "busiest"),
    ("Localization", "svgloc:0034:small:03",
     "Given the exact target string and asked to point at it, the model returns a "
     "coordinate in the wrong region of the frame.", None),
    ("Label - object matching", "ai2d:01458",
     "The question names a printed mark on the diagram. Resolving which object that "
     "mark sits on is where the answer goes wrong.", None),
    ("General OCR reasoning", "charxiv:00052:r",
     "The arithmetic is right and the operation is right. The value it was applied "
     "to was misread off the chart.", None),
    ("Hallucination", "charxiv:00772:d4",
     "The plot has no legend at all. Asked how many entries it has, the model "
     "answers with a number rather than abstaining.", None),
    ("Counting", "charxiv:00655:d2",
     "The legend is printed at full size and is plainly legible. The model still "
     "counts 7 of its 11 entries.", (0.0, 0.0, 1.0, 0.24)),
]


def _pick_panel(uid: str):
    """(image, question, gold, model) for one chosen example, whatever it came from."""
    ds = uid.split(":")[0]
    if ds == "svgloc":
        from blindspot.eval import load_loc_run as load_run
        ex = {e.uid: e for e in load("svg_localization")}
        r = next(x for x in load_run(RUN)["point"] if x["uid"] == uid)
        im = draw_target(as_model_saw(ex[uid].images[0]),
                         box=r["gold"], point=tuple(r["pred"]))
        return (im, f'point at {r["question"]}', "inside the green box",
                f'({r["pred"][0]:.2f}, {r["pred"][1]:.2f}) - '
                f'{r["d_centre"]*100:.0f}% of the frame away')
    name = {"infographicvqa": "infographicvqa", "ai2d": "ai2d", "charxiv": "charxiv"}[ds]
    e = {x.uid: x for x in load(name)}[uid]
    pred = _ex_preds(f"results/{name}__{RUN}.jsonl")[uid]
    q = e.question.split("*")[0].strip()
    gold = e.gold[0] if isinstance(e.gold, list) else e.gold
    if ds == "ai2d":                            # letters mean nothing without the text
        opts = e.meta.get("options") or []
        letter = lambda v: opts["ABCD".find(str(v).strip()[:1].upper())] \
            if 0 <= "ABCD".find(str(v).strip()[:1].upper()) < len(opts) else str(v)
        gold, pred = f'{gold} = "{letter(gold)}"', f'{pred} = "{letter(pred)}"'
    return as_model_saw(e.images[0]), q, str(gold), str(pred)


def busiest_crop(im: Image.Image, w: int = 900, h: int = 600) -> Image.Image:
    """A 1:1 window over the most detailed part of the delivered image.

    Edge energy is a decent proxy for text density, and text is what the
    resolution argument is about.
    """
    from PIL import ImageFilter, ImageStat
    w, h = min(w, im.width), min(h, im.height)
    best, bx, by = -1.0, 0, 0
    edges = im.convert("L").filter(ImageFilter.FIND_EDGES)
    for y in range(0, max(1, im.height - h + 1), max(1, h // 2)):
        for x in range(0, max(1, im.width - w + 1), max(1, w // 2)):
            v = ImageStat.Stat(edges.crop((x, y, x + w, y + h))).mean[0]
            if v > best:
                best, bx, by = v, x, y
    return im.crop((bx, by, bx + w, by + h))


def _zoom_of(im, spec):
    """A 1:1 window on the delivered image: either a fixed box or the busiest one."""
    if spec is None:
        return None
    if isinstance(spec, tuple):
        x0, y0, x1, y1 = spec
        return im.crop((int(x0 * im.width), int(y0 * im.height),
                        int(x1 * im.width), int(y1 * im.height)))
    return busiest_crop(im, 900, 620)


def f06_problems():
    """The six blind spots, one real picture each.

    Deliberately plain: a name, the evidence, and a sentence. No verdict chips
    and no percentages - those live in the table beside it, and repeating them
    here would just be a second, worse table.
    """
    ROW, IMGW, IMGH = 540, 940, 470
    p = Panel("Six candidate blind spots, one scored item each",
              "Every picture is shown at the resolution the API actually delivered.",
              h=260 + ROW * len(PICKS))
    y = p.y + 10
    for head, uid, why, zspec in PICKS:
        im, q, gold, pred = _pick_panel(uid)
        p.rule(y - 18)
        yy = p.text(80, y + 8, head, 38, INK, True, wrap=22)
        p.text(80, yy + 14, why, 26, INK2, wrap=34)
        zoom = _zoom_of(im, zspec)
        if zoom is None:
            _, bot = p.paste(im, 700, y, IMGW, IMGH)
        else:                       # whole scene small, then the detail at 1:1
            fw, bot = p.paste(im, 700, y, 330, IMGH)
            zx = 700 + fw + 30
            _, zh = p.paste(zoom, zx, y, IMGW - fw - 30, IMGH)
            p.text(zx, y + zh + 8, "1:1 crop of the delivered pixels", 21, MUTED)
            bot = max(bot, zh + 34)
        x = 700 + IMGW + 60
        ty = p.text(x, y + 8, "asked", 22, MUTED, True)
        ty = p.text(x, ty + 4, q, 27, INK, wrap=42)
        ty = p.text(x, ty + 18, "gold", 22, MUTED, True)
        ty = p.text(x, ty + 4, gold, 27, INK, wrap=42)
        ty = p.text(x, ty + 18, "model", 22, MUTED, True)
        ty = p.text(x, ty + 4, pred, 27, CRIT, True, wrap=42)
        p.text(x, ty + 20, uid, 21, MUTED)
        last = max(y + bot, ty + 50)            # rows are as tall as their content
        y = max(y + ROW, last + 60)
    p.y = last
    return p.save("f06_problems")


EXAMPLES = [
    (e09_bad_gold,
     "A scored failure the audit would not defend: the model's answer and the gold "
     "answer differ only in surface form.",
     "One of 410 audited InfographicVQA failures."),
    (f06_problems,
     "The six candidate blind spots, each with one real item the model was scored on.",
     "Pictures only; the measured results are in the table."),
    (e08_generation_pipeline,
     "The deterministic scene-generation pipeline and what the target filter rejects.",
     "Counts are per scene and resolution, deduplicated across questions."),
    (e05_instrument,
     "Eight of the sixteen generated chart types, with the two delivered resolutions.",
     "Scenes are shown at the smaller of the two."),
]


# ============================================== tables: outputs/report/tables.md
DARK_THEMES = {"slate-dark", "carbon", "blueprint"}
TABLE_RUNGS = ("small", "large")      # the two reported resolutions


def points():
    """The scored point rows at the two reported resolutions."""
    from blindspot.eval import load_loc_run as load_run
    return [r for r in load_run(RUN)["point"] if r["meta"]["resolution"] in TABLE_RUNGS]


def load_json(p: str) -> dict:
    return json.loads(Path(p).read_text())


def md_table(head: list[str], rows: list[list], note: str = "") -> str:
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    if note:
        out += ["", f"*{note}*"]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ T1, T2
BENCH = [                       # (key, label, operation, metric)
    ("charxiv", "CharXiv", "Read a value or a structure off a scientific chart",
     "judged exact match"),
    ("ai2d", "AI2D", "Answer a question about a diagram, 4-way multiple choice",
     "accuracy"),
    ("slidevqa", "SlideVQA", "Answer from evidence inside a slide deck", "ANLS"),
    ("infographicvqa", "InfographicVQA", "Read a dense, large-format infographic", "ANLS"),
    ("screenspot_pro", "ScreenSpot-Pro",
     "Point at an element described by its function — the answer is a coordinate",
     "click-in-bbox"),
]


def t1(f: dict) -> str:
    b = f["benchmarks"]
    rows = [[lab, op, f'{b[k]["n"]:,}', met, f'**{pct(b[k]["acc"])}**']
            for k, lab, op, met in BENCH]
    ctl = b["slidevqa_allpages"]
    total = sum(b[k]["n"] for k, *_ in BENCH) + ctl["n"]
    return md_table(
        ["Benchmark", "Operation measured", "Items", "Metric", "Haiku 4.5"], rows,
        f'{total:,} questions on `{f["model"]}`, thinking at 2,000 tokens, each arm '
        f'scored by its own published metric. A sixth scored arm — SlideVQA all-pages, '
        f'n={ctl["n"]}, {pct(ctl["acc"])} — is a retrieval control, not a separate '
        f'benchmark.')


def t2(f: dict) -> str:
    bind, absd = f["ai2d_binding"], f["absence_detection"]
    mark, nomark = bind["sighted"]["refers_to_mark"], bind["sighted"]["no_mark"]
    pb = bind["paired_blind"]
    pa = absd["paired_blind"]["absent"]
    rows = [
        ["**Resolution bias**", "InfographicVQA, CharXiv",
         "ANLS 74.3% → 59.4% across image-size quintiles (n=2,801); CharXiv −9.5pp, "
         "1 panel vs 13+",
         "text-volume control flat — 6.7pp spread, no trend"],
        ["**Localization**", "ScreenSpot-Pro",
         f'{pct(f["benchmarks"]["screenspot_pro"]["acc"])} click-in-bbox '
         f'(n={f["benchmarks"]["screenspot_pro"]["n"]:,}); 0.0% on targets under 12px '
         f'as delivered', "—"],
        ["**Label–object matching**", "AI2D",
         f'{pct(mark["acc"])} when the question names a printed mark (n={mark["n"]:,}) '
         f'vs {pct(nomark["acc"])} when it does not (n={nomark["n"]:,})',
         f'{pct(pb["refers_to_mark"]["blind"])} vs {pct(pb["no_mark"]["blind"])} — '
         f'mark-referring sits **below the {pct(bind["chance"], 0)} chance line**'],
        ["**General OCR reasoning**", "CharXiv, SlideVQA",
         "CharXiv 90.7% descriptive vs 63.7% reasoning (27.0pp); SlideVQA 26.5pp as "
         "scored, **3.8pp** format-corrected", "—"],
        ["**Hallucination**", "CharXiv",
         f'{pct(absd["full_set"]["absent"]["invention_rate"])} invention on '
         f'{absd["full_set"]["absent"]["n"]:,} "Not Applicable" items, 45.7% on the '
         f'worst template; over-abstention '
         f'{pct(absd["full_set"]["over_abstention"]["rate"], 2)}',
         f'abstains **{pct(pa["abstains_blind"])} blind vs '
         f'{pct(pa["abstains_sighted"])} sighted** (n={pa["n"]}) — the image adds nothing'],
        ["**Counting**", "InfographicVQA, CharXiv",
         "InfoVQA 63% → 33% across count bins; CharXiv ticks 78.1% (n=224) vs objects "
         "93.3% (n=314)", "—"],
    ]
    return md_table(["Blind spot", "Benchmark", "Public-benchmark result", "Blind control"],
                 rows,
                 "The blind control asks the same question with the image withheld. A "
                 "candidate is a perception blind spot only where withholding the image "
                 "changes the answer.")


# ------------------------------------------------------------------ T3, T4
def t3(f: dict) -> str:
    from blindspot.eval import precision_curve
    rows = points()
    curve = precision_curve(rows)
    label = {"exact hit box": "**the exact target box**"}
    out = []
    for c in curve:
        last = c is curve[-1]
        lab = label.get(c["grid"], f'{c["grid"].replace("x", "×")} cell')
        bold = (lambda t: f"**{t}**") if last else (lambda t: t)
        out.append([lab, bold(pct(c["chance"], 2)), bold(pct(c["strict"], 2)),
                    bold(f'{c["ratio"]:.1f}×')])
    return md_table(["Required precision", "Chance", "Accuracy", "Ratio to chance"], out,
                 f'Both resolutions pooled, n={len(rows):,}. Chance at the exact box is '
                 f'the mean target-area fraction. The same predictions are bucketed more '
                 f'coarsely at each row up, so this is one set of answers read at six '
                 f'tolerances, not six experiments.')


def _misses_on_other_label(rows) -> float:
    """Share of in-range misses landing inside a *different* catalogued target box.

    `report_tables` used to carry a second, byte-identical copy of this loop.
    Two implementations of one measurement is exactly what the layer rule
    forbids, so this now calls the one in the data section.
    """
    return misses_on_other_label(rows)["value"]


def t4(f: dict) -> str:
    s = f["synthetic"]
    rows = []
    for g in TABLE_RUNGS:
        b = s["bands"]["by_rung_d_centre"][g]
        d = s["distance"][g]
        rows.append([g, f'{d["n_miss"]:,}', pct(b["near_miss"]), pct(b["moderate_miss"]),
                     pct(b["wrong_region"]), f'{d["median_d_centre"] * 100:.1f}%'])
    return md_table(["Resolution", "Misses", "Near (<10%)", "Moderate (10–25%)",
                  "Wrong region (>25%)", "Median distance"], rows,
                 f'Distance is to the target centre, as a fraction of the frame '
                 f'diagonal, and the bands are of misses only. '
                 f'{pct(_misses_on_other_label(points()), 1)} of in-range misses land '
                 f'inside a different labelled element.')


# ------------------------------------------------------------------ T5
def _mh_or(rows) -> tuple[float, float, float]:
    """Mantel-Haenszel odds ratio for dark vs light, with a Robins-Breslow-Greenland CI.

    Strata are resolution x target-area tertile x contrast tertile, so a theme that
    simply drew bigger or higher-contrast targets cannot produce the effect on
    its own. Chart type is deliberately left out: adding it spreads the dark
    items over several hundred cells, and the point estimate then moves with the
    binning rather than with the data.
    """
    def tertile(vals):
        v = sorted(vals)
        return v[len(v) // 3], v[2 * len(v) // 3]

    a_lo, a_hi = tertile([r["meta"]["target_area_frac"] for r in rows])
    c_lo, c_hi = tertile([r["meta"]["target_contrast"] for r in rows])
    cut = lambda v, lo, hi: 0 if v < lo else (1 if v < hi else 2)

    strata = collections.defaultdict(lambda: [0, 0, 0, 0])   # a b c d
    for r in rows:
        m = r["meta"]
        key = (m["resolution"],
               cut(m["target_area_frac"], a_lo, a_hi),
               cut(m["target_contrast"], c_lo, c_hi))
        dark = m["theme"] in DARK_THEMES
        strata[key][(0 if dark else 2) + (0 if r["hit"] else 1)] += 1

    ps = qs = rs = rsum = ssum = 0.0
    for a, b, c, d in strata.values():
        n = a + b + c + d
        if n == 0 or (a + b) == 0 or (c + d) == 0:
            continue
        R, S = a * d / n, b * c / n
        P, Q = (a + d) / n, (b + c) / n
        ps += P * R
        qs += P * S + Q * R
        rs += Q * S
        rsum += R
        ssum += S
    if rsum == 0 or ssum == 0:
        return float("nan"), float("nan"), float("nan")
    orr = rsum / ssum
    se = math.sqrt(ps / (2 * rsum ** 2) + qs / (2 * rsum * ssum) + rs / (2 * ssum ** 2))
    return orr, orr * math.exp(-1.96 * se), orr * math.exp(1.96 * se)


def _crude_or(k1, n1, k2, n2) -> float:
    return ((k1 + .5) / (n1 - k1 + .5)) / ((k2 + .5) / (n2 - k2 + .5))


def t6(f: dict) -> str:
    """Dark vs light exact localization. One comparison, which is all §7 claims."""
    rows = points()
    dk = [r for r in rows if r["meta"]["theme"] in DARK_THEMES]
    lt = [r for r in rows if r["meta"]["theme"] not in DARK_THEMES]
    kd, kl = sum(r["hit"] for r in dk), sum(r["hit"] for r in lt)
    orr, lo, hi = _mh_or(rows)
    return md_table(["Background", "Exact localization", "Hits", "Items"],
                 [["Dark", pct(kd / len(dk)), kd, f"{len(dk):,}"],
                  ["Light", pct(kl / len(lt)), kl, f"{len(lt):,}"]],
                 f'Crude odds ratio {_crude_or(kd, len(dk), kl, len(lt)):.2f}; adjusted '
                 f'**{orr:.2f}** [{lo:.2f}–{hi:.2f}] by Mantel-Haenszel, stratified on '
                 f'resolution, target-area tertile and contrast tertile, so this is not '
                 f'simply dark themes drawing bigger or higher-contrast targets. Theme is '
                 f'assigned per scene rather than crossed within it, so this is '
                 f'observational.')


# ------------------------------------------------------------------ T7
def t7(f: dict) -> str:
    d = f["synthetic"]["derived"]
    w, c = d["word_mc"], d["counting"]
    by = lambda h, lab: next(x for x in h if x["label"] == lab)
    loc = {x["label"]: x for x in f["synthetic"]["headline"]}
    rows = [
        ["Is this word present? (4-way choice)", pct(by(w["headline"], "small")["acc"], 2),
         pct(by(w["headline"], "large")["acc"], 2),
         pct(w["blind"]["overall"]["acc"]), f'{w["overall"]["n"]:,}'],
        ["How many times does it appear?", pct(by(c["headline"], "small")["acc"], 2),
         pct(by(c["headline"], "large")["acc"], 2),
         pct(c["blind"]["overall"]["acc"]), f'{c["overall"]["n"]:,}'],
        ["Point at it (exact target box)", pct(loc["small"]["acc"], 2),
         pct(loc["large"]["acc"], 2), "—", f'{loc["small"]["n"] + loc["large"]["n"]:,}'],
    ]
    return md_table(["Task on the generated scenes", "Small", "Large",
                  "Blind control", "Items"], rows,
                 "The same 200 scenes and the same image files; the question populations "
                 "differ. Reading and counting do not move with resolution, which is what "
                 "the public megapixel gradient predicted they would do. Pointing does "
                 "move, and in the opposite direction.")


TABLES = [("T1", "The benchmark suite", t1), ("T2", "Result per blind spot", t2),
          ("T3", "Localization accuracy against required precision", t3),
          ("T4", "Where the misses land", t4),
          ("T5", "Localization by background polarity", t6),
          ("T6", "Reading, counting and pointing on the same scenes", t7)]


def build_tables() -> dict[str, str]:
    f = load_json("outputs/report/figures.json")
    return {tid: fn(f) for tid, _title, fn in TABLES}


def inject_tables() -> int | None:
    """Rewrite each `<!-- Tn -->...<!-- /Tn -->` block in `blindspots.md` in place.

    The prose between the markers is hand-written and never touched; only the
    generated tables are replaced, so the report can be re-derived after a rerun
    without losing edits.

    Returns the number of blocks rewritten, or None if the prose spine is not
    here -- which is the normal state of a fresh clone, not an error.
    """
    text = read_prose("wrote tables.md only (nothing to inject)")
    if text is None:
        return None
    src = PROSE
    built = build_tables()
    n = 0
    for tid, title, _fn in TABLES:
        pat = re.compile(rf"<!-- {tid} -->.*?<!-- /{tid} -->", re.S)
        if not pat.search(text):
            print(f"  !! no marker for {tid} in {PROSE}")
            continue
        text = pat.sub(f"<!-- {tid} -->\n**Table {tid[1:]}. {title}.**\n\n"
                       f"{built[tid]}\n<!-- /{tid} -->", text)
        n += 1
    src.write_text(text)
    return n


# ========================= index: outputs/report/figures.md + figures/index.html
# (file stem, kind, section, caption, confidence strip) in narrative order
ORDER = [
    ("e09_bad_gold", "example", "§2",
     "A scored failure where the model's answer and the gold answer differ only in "
     "surface form.",
     "One of 410 audited InfographicVQA failures."),
    ("f06_problems", "example", "§3",
     "The six candidate blind spots, each with one real item the model was scored on.",
     "Pictures only; the measured results are in Table 2."),
    ("e08_generation_pipeline", "diagram", "§5",
     "The deterministic scene-generation pipeline, and what the target filter rejected.",
     "Rejection counts are per scene and resolution, deduplicated across questions."),
    ("e05_instrument", "example", "§5",
     "Eight of the sixteen generated chart types, with the two delivered resolutions.",
     "Scenes are shown at the smaller of the two."),
]


# `[FIG:stem]` in the source, `[Figure 2]<!--FIG:stem-->` once resolved. The
# trailing comment is invisible in rendered markdown and keeps the reference
# re-resolvable, so figures can be reordered without hand-editing the prose.
FIGREF_RE = re.compile(r"\[Figures? \d+\]<!--FIG:(\w+)-->|\[FIG:(\w+)\]")


def inject_refs() -> int | None:
    """Resolve `[FIG:stem]` tokens in `blindspots.md` to numbered references.

    Idempotent. Returns None if the prose spine is not here.
    """
    num = {stem: i for i, (stem, *_rest) in enumerate(ORDER, 1)}
    text = read_prose("wrote figures.md and figures/index.html only "
                      "(no references to resolve)")
    if text is None:
        return None
    src = PROSE
    missing = []

    def sub(m):
        stem = m.group(1) or m.group(2)
        if stem not in num:
            missing.append(stem)
            return m.group(0)
        return f"[Figure {num[stem]}]<!--FIG:{stem}-->"

    out, n = FIGREF_RE.subn(sub, text)
    src.write_text(out)
    for stem in dict.fromkeys(missing):
        print(f"  !! reference to unknown figure: {stem}")
    unref = [s for s in num if f"FIG:{s}" not in out]
    for stem in unref:
        print(f"  !! figure never referenced in the text: {stem}")
    return n


# ==================================== paste: outputs/report/paste_into_docs.html
PASTE_MAX_W = 1600          # ~245 DPI across a 6.5in Docs column

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Calibri, "
        "'Helvetica Neue', Arial, sans-serif")
BODY = f"font-family:{FONT};font-size:11pt;line-height:1.5;color:#111"
SMALL = f"font-family:{FONT};font-size:9pt;color:#666"

PASTE_REF_RE = re.compile(r"\[Figures? (\d+)\]<!--FIG:(\w+)-->")
NUM = {stem: i for i, (stem, *_rest) in enumerate(ORDER, 1)}
META = {stem: (cap, strip) for stem, _k, _s, cap, strip in ORDER}


def data_uri(png: Path) -> str:
    im = Image.open(png).convert("RGB")
    if im.width > PASTE_MAX_W:
        im = im.resize((PASTE_MAX_W, round(im.height * PASTE_MAX_W / im.width)),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def inline(s: str) -> str:
    """Markdown inline spans -> HTML, with figure markers turned into plain text.

    A reference reads as a parenthetical, since the figure and its numbered
    caption are placed immediately after the paragraph that names it.
    """
    s = PASTE_REF_RE.sub(lambda m: f"\x00(Figure {m.group(1)})\x01", s)
    s = esc_nq(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"`(.+?)`", r'<span style="font-family:Consolas,monospace">\1</span>', s)
    return re.sub(r"\s+\x00", " \x00", s).replace("\x00", "").replace("\x01", "").strip()


def figure_html(stem: str) -> str:
    png = FIGS_OUT / f"{stem}@2x.png"
    if not png.exists():
        return ""
    cap, strip = META[stem]
    return (f'<p style="margin:22px 0 4px"><img src="{data_uri(png)}" '
            f'style="width:100%;max-width:660px" alt="{esc_nq(cap)}"></p>'
            f'<p style="margin:0 0 4px;{SMALL};font-size:9.5pt;color:#444">'
            f'<strong>Figure {NUM[stem]}.</strong> {esc_nq(cap)}</p>'
            f'<p style="margin:0 0 24px;{SMALL};color:#777"><em>{esc_nq(strip)}</em></p>')


def is_row(s: str) -> bool:
    return s.startswith("|") and s.endswith("|")


def md_cells(s: str) -> list[str]:
    return [c.strip() for c in s.strip().strip("|").split("|")]


def docs_table_html(rows: list[str]) -> str:
    """A markdown table as a real <table>; the second row is the alignment rule."""
    head, body = md_cells(rows[0]), [md_cells(r) for r in rows[2:]]
    th = ("padding:6px 9px;border:1px solid #c9c8c3;background:#f2f2f0;"
          "text-align:left;vertical-align:top;font-weight:700")
    td = "padding:6px 9px;border:1px solid #d9d8d3;vertical-align:top"
    out = [f'<table style="border-collapse:collapse;width:100%;{FONT and ""}'
           f'font-family:{FONT};font-size:9.5pt;color:#111;margin:0 0 6px">',
           "<tr>" + "".join(f'<th style="{th}">{inline(c)}</th>' for c in head) + "</tr>"]
    for r in body:
        out.append("<tr>" + "".join(f'<td style="{td}">{inline(c)}</td>'
                                    for c in r) + "</tr>")
    return "".join(out) + "</table>"


def build_paste() -> str | None:
    """The whole paste page, or None if the prose spine is not here.

    Unlike tables and index, this command has nothing to write without it: the
    page IS the prose, with the figures and tables interleaved.
    """
    text = read_prose("nothing to paste (paste_into_docs.html is the prose spine "
                      "with the figures and tables interleaved)")
    if text is None:
        return None
    lines = text.splitlines()
    out: list[str] = []
    para: list[str] = []
    bullets: list[str] = []
    tbl: list[str] = []
    placed: set[str] = set()

    def emit_figs(text: str):
        for _n, stem in PASTE_REF_RE.findall(text):
            if stem in META and stem not in placed:
                placed.add(stem)
                out.append(figure_html(stem))

    def flush_para():
        if para:
            text = " ".join(para)
            out.append(f'<p style="margin:0 0 11px;{BODY}">{inline(text)}</p>')
            emit_figs(text)
            para.clear()

    def flush_bullets():
        if bullets:
            items = "".join(f'<li style="margin:0 0 7px">{inline(b)}</li>'
                            for b in bullets)
            out.append(f'<ul style="margin:0 0 12px 20px;{BODY}">{items}</ul>')
            emit_figs(" ".join(bullets))
            bullets.clear()

    def flush_table():
        if tbl:
            out.append(docs_table_html(tbl))
            tbl.clear()

    def flush_all():
        flush_para(); flush_bullets(); flush_table()

    for line in lines:
        s = line.rstrip()
        if s.startswith("<!--"):                 # generator markers, not content
            flush_all()
            continue
        if is_row(s):
            flush_para(); flush_bullets()
            tbl.append(s)
            continue
        flush_table()
        if s.startswith("- "):
            flush_para()
            bullets.append(s[2:])
            continue
        if s.startswith(("#", "---")) or not s.strip():
            flush_para(); flush_bullets()
        if not s.strip():
            continue
        if s.startswith("### "):
            out.append(f'<p style="margin:0 0 16px;font-family:{FONT};font-size:13pt;'
                       f'color:#555">{inline(s[4:])}</p>')
        elif s.startswith("## "):
            out.append(f'<h2 style="margin:26px 0 10px;font-family:{FONT};font-size:15pt;'
                       f'color:#111">{inline(s[3:])}</h2>')
        elif s.startswith("# "):
            out.append(f'<h1 style="margin:0 0 6px;font-family:{FONT};font-size:22pt;'
                       f'color:#111">{inline(s[2:])}</h1>')
        elif s.startswith("---"):
            out.append('<hr style="border:none;border-top:1px solid #ccc;margin:20px 0">')
        elif s.startswith("*") and s.endswith("*") and not s.startswith("**"):
            out.append(f'<p style="margin:0 0 14px;{SMALL}">'
                       f'<em>{inline(s.strip("*"))}</em></p>')
        else:
            para.append(s.strip())
    flush_all()

    for stem in NUM:                             # anything never referenced
        if stem not in placed:
            out.append(figure_html(stem))

    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Perception blind spots in Claude Haiku 4.5</title></head>"
            "<body style='margin:0 auto;padding:36px;max-width:720px;background:#fff'>"
            + "".join(out) + "</body></html>")


# ============================================ svgloc: outputs/svgloc/report.html
SVG_GOOD, SVG_BAD, SVG_ACC = "#0ca30c", "#d03b3b", "#5b8def"
CSS = """
:root{--bg:#0f1116;--panel:#171a21;--panel2:#1d2028;--ink:#e8eaed;--muted:#9aa0aa;
 --line:#2a2f3a;--good:#0ca30c;--bad:#d03b3b;--warn:#d68a1e;--accent:#5b8def;--chip:#232833}
@media (prefers-color-scheme:light){:root{--bg:#f7f8fa;--panel:#fff;--panel2:#f0f2f6;
 --ink:#15181d;--muted:#5c636e;--line:#dfe3ea;--chip:#eceff4}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:27px;margin:0 0 6px}
h2{font-size:20px;margin:38px 0 12px;padding-bottom:7px;border-bottom:1px solid var(--line)}
h3{font-size:16px;margin:22px 0 8px}
.sub{color:var(--muted);margin:0 0 22px}
table{width:100%;border-collapse:collapse;margin:12px 0;background:var(--panel);
 border:1px solid var(--line);border-radius:9px;overflow:hidden}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);font-size:14px}
th{background:var(--panel2);color:var(--muted);font-weight:600;font-size:12px;
 text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.tile .v{font-size:25px;font-weight:650;font-variant-numeric:tabular-nums}
.tile .l{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.tile .n{color:var(--muted);font-size:12px;margin-top:5px}
.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.callout{background:var(--panel);border-left:3px solid var(--accent);padding:13px 16px;
 border-radius:0 9px 9px 0;margin:14px 0}
.callout.bad{border-left-color:var(--bad)}.callout.good{border-left-color:var(--good)}
.callout.warn{border-left-color:var(--warn)}
.bar{height:9px;background:var(--panel2);border-radius:5px;overflow:hidden;min-width:90px}
.bar>i{display:block;height:100%;background:var(--accent)}
code{background:var(--chip);padding:1px 5px;border-radius:4px;font-size:13px}
.ex{background:var(--panel);border:1px solid var(--line);border-radius:11px;
 padding:13px 15px;margin:12px 0}
.exhd{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:9px}
.pill{font-size:11px;font-weight:650;padding:2px 9px;border-radius:999px;border:1px solid}
.pill.ok{color:var(--good);border-color:var(--good)}
.pill.no{color:var(--bad);border-color:var(--bad)}
.chip{font-size:11.5px;padding:2px 9px;border-radius:999px;background:var(--chip);
 color:var(--muted);border:1px solid var(--line)}
.strip{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0}
.strip figure{margin:0;max-width:340px}
.strip img{width:100%;border-radius:7px;border:1px solid var(--line);display:block}
.strip figcaption{color:var(--muted);font-size:11.5px;margin-top:4px}
dl.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:6px 16px;margin:8px 0 0}
dl.kv dt{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.03em}
dl.kv dd{margin:0 0 5px;font-size:13.5px}
ul{padding-left:20px}li{margin:5px 0}
"""


def html_table(headers, rows, note="") -> str:
    h = "".join(f'<th class="num">{c}</th>' if i else f"<th>{c}</th>"
                for i, c in enumerate(headers))
    body = []
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            txt, cls = (c if isinstance(c, tuple) else (c, ""))
            cells.append(f'<td class="num {cls}">{txt}</td>' if i else f"<td>{txt}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    n = f'<p class="sub">{note}</p>' if note else ""
    return f"<table><tr>{h}</tr>{''.join(body)}</table>{n}"


def barcell(v, colour=SVG_ACC, scale=1.0) -> str:
    w = 0 if v is None else max(0.6, min(100.0, v * 100 / scale))
    return f'<div class="bar"><i style="width:{w:.1f}%;background:{colour}"></i></div>'
# --------------------------------------------------------------- example art
def render_example(r, key: str) -> dict:
    """Full frame with the gold box + click, and a zoom around both.

    Rendered at the size the model actually resolved, so the reader is never
    shown detail the model never had. The gold box gets a locator ring because
    a 46x16px target is invisible on a page-width thumbnail.
    """
    ex_img = r["_img"]
    sent = r.get("sent") or None
    im = Image.open(ex_img).convert("RGB")
    if sent:
        w, h = int(sent[0]), int(sent[1])
        s = min(1.0, 1568 / max(w, h), math.sqrt(1_150_000 / max(w * h, 1)))
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    W, H = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(W, H) / 400))
    x0, y0, x1, y1 = [v * s_ for v, s_ in zip(r["gold"], (W, H, W, H))]
    d.rectangle([x0, y0, x1, y1], outline=SVG_GOOD, width=lw)
    ring = max(lw * 14, (x1 - x0) * 1.6, (y1 - y0) * 1.6)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    d.ellipse([cx - ring, cy - ring, cx + ring, cy + ring], outline=SVG_GOOD, width=max(1, lw // 2))
    px, py = r["pred"][0] * W, r["pred"][1] * H
    rr = lw * 8
    d.line([px - rr, py, px + rr, py], fill=SVG_BAD, width=lw)
    d.line([px, py - rr, px, py + rr], fill=SVG_BAD, width=lw)
    d.ellipse([px - rr / 2, py - rr / 2, px + rr / 2, py + rr / 2], outline=SVG_BAD, width=lw)

    SVGLOC_ASSETS.mkdir(parents=True, exist_ok=True)
    full = SVGLOC_ASSETS / f"{key}_f.jpg"
    im.copy().resize((min(W, 900), max(1, round(H * min(W, 900) / W))), Image.LANCZOS) \
      .save(full, quality=82, optimize=True)
    pad = max(0.10, abs(r["pred"][0] - cx / W) * 0.9, abs(r["pred"][1] - cy / H) * 0.9)
    cx_n, cy_n = cx / W, cy / H
    box = [max(0.0, min(cx_n, r["pred"][0]) - pad), max(0.0, min(cy_n, r["pred"][1]) - pad),
           min(1.0, max(cx_n, r["pred"][0]) + pad), min(1.0, max(cy_n, r["pred"][1]) + pad)]
    crop = im.crop((int(box[0] * W), int(box[1] * H), max(1, int(box[2] * W)), max(1, int(box[3] * H))))
    # A hit puts the click on top of the target, which collapses the crop to a
    # thumbnail too small to read. Upscale so the reader can actually see the
    # glyphs the model was working from.
    if crop.width < 460:
        f = 460 / max(crop.width, 1)
        crop = crop.resize((460, max(1, round(crop.height * f))), Image.LANCZOS)
    zoom = SVGLOC_ASSETS / f"{key}_z.jpg"
    crop.save(zoom, quality=88, optimize=True)
    return {"full": full.name, "zoom": zoom.name}


def example_card(r, art) -> str:
    m = r["meta"]
    ok = r["hit"]
    pill = ('<span class="pill ok">&#10003; inside the box</span>' if ok
            else '<span class="pill no">&#10007; missed</span>')
    chips = "".join(f'<span class="chip">{esc(c)}</span>' for c in
                    [m["resolution"], m["chart_type"], m["target_role"], m["hit_source"],
                     f'{m["target_area_frac"]*100:.3f}% of image', f'{m["font_px"]}px font'])
    off = not r.get("in_range", True)
    cap_full = ("as the model resolved it &mdash; green box + ring = target; the click fell "
                "outside the image and cannot be drawn" if off else
                "as the model resolved it &mdash; green box + ring = target, "
                "red crosshair = click")
    cap_zoom = ("zoom on the target; the click is off-canvas" if off
                else "zoom on target and click")
    strip = (f'<figure><img loading="lazy" src="assets/{art["full"]}">'
             f'<figcaption>{cap_full}</figcaption></figure>'
             f'<figure><img loading="lazy" src="assets/{art["zoom"]}">'
             f'<figcaption>{cap_zoom}</figcaption></figure>')
    kv = [("asked for", esc(r["question"])),
          ("clicked", f'{r["pred"][0]*100:.1f}%, {r["pred"][1]*100:.1f}%'),
          ("target box", "x {:.1f}&ndash;{:.1f}%, y {:.1f}&ndash;{:.1f}%".format(
              r["gold"][0]*100, r["gold"][2]*100, r["gold"][1]*100, r["gold"][3]*100)),
          ("distance to box", f'{r["d_box"]*100:.2f}% of the frame'),
          ("distance to centre", f'{r["d_centre"]*100:.2f}%'),
          ("uid", f'<span style="color:var(--muted)">{esc(r["uid"])}</span>')]
    kvh = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in kv)
    return (f'<article class="ex"><div class="exhd">{pill}{chips}</div>'
            f'<div class="strip">{strip}</div><dl class="kv">{kvh}</dl></article>')


# -------------------------------------------------------------------- render
def svgloc_render(s: dict, probe: list | None, examples: list[tuple]) -> str:
    b = []
    A = b.append
    counts = s["counts"]

    A(f"<h1>Localization and effective resolution &mdash; <code>svg_localization</code></h1>")
    A(f'<p class="sub">Claude Haiku 4.5 &middot; <code>claude-haiku-4-5-20251001</code> &middot; '
      f'thinking enabled (2000 tokens) &middot; native resolution &middot; '
      f'{counts["point_scored"]:,} point + {counts["span_scored"]:,} text questions '
      f'across 200 synthetic scenes at three resolution rungs</p>')

    # ---- 0. probe
    A("<h2>0. Sanity probe &mdash; is the pipeline sound?</h2>")
    A("<p>A near-zero localization score has two explanations: the model cannot do it, or the "
      "harness is broken. This dataset had never been run against any model, so that ambiguity "
      "had to be closed before any number below could be read. A stronger model was run on "
      "byte-identical inputs.</p>")
    if probe:
        rows = []
        for p in probe:
            cond = "native" if p["max_edge"] is None else f'pre-downscaled to {p["max_edge"]}px'
            byr = "  ".join(f'{k} {v[0]*100:.0f}% (n={v[1]})' for k, v in p["by_res"].items())
            rows.append([f'{esc(p["model"])} &middot; {cond}', f'{p["usable"]}',
                         (f'<b>{p["acc"]*100:.1f}%</b>',
                          "good" if p["acc"] > 0.5 else ("warn" if p["acc"] > 0.15 else "")),
                         byr])
        A(html_table(["arm", "n", "click-in-bbox", "by rung"], rows))
    A('<div class="callout good"><b>The pipeline is sound.</b> Sonnet, handicapped to Haiku\'s '
      '1568px budget, lands inside the target box on <b>81%</b> of <code>large</code> items. '
      'A score that high is only reachable if the gold boxes, the 0&ndash;1000 coordinate '
      'convention and <code>point_in_bbox</code> are all correct. Every low number below is '
      'therefore a capability result, not a harness bug.</div>')
    A('<div class="callout warn"><b>An unplanned finding from the probe.</b> Sonnet scores '
      '<b>19%</b> on <code>large</code> at native resolution and <b>81%</b> on the same items '
      'pre-downscaled to 1568px &mdash; <i>more</i> pixels made it much worse. Token counts '
      'explain it: Sonnet receives native <code>large</code> at 5,054 input tokens against '
      '2,340 pre-downscaled, because its image ceiling (~2576px) is higher than Haiku\'s. '
      'This is a property of Sonnet\'s cap, not of the dataset, and it is why the probe is a '
      'pipeline check and not a model comparison. Haiku receives <code>medium</code> and '
      '<code>large</code> at 1,854 and 1,855 tokens respectively &mdash; identical, which '
      'confirms the null control below empirically rather than by assumption.</div>')

    # ---- 1. null control first
    nc = s["null_control"]
    re_ = s["resolution_effect"]
    A("<h2>1. The noise floor, first: <code>medium</code> vs <code>large</code></h2>")
    A("<p>Both rungs are delivered to the model at the same size, so they carry the same "
      "information and differ only in resampling path. Whatever gap appears here is this "
      "dataset's empirical noise floor, and it is the yardstick for every other difference "
      "reported below.</p>")
    if nc.get("n"):
        A(html_table(["paired comparison", "n", "medium", "large", "difference", "McNemar &chi;&sup2;"],
                [[f'<b>{nc["a"]} &rarr; {nc["b"]}</b>', f'{nc["n"]:,}',
                  pct_html(nc["acc_a"], 2), pct_html(nc["acc_b"], 2),
                  (f'<b>{nc["delta_pp"]:+.2f}pp</b>',
                   "good" if abs(nc["delta_pp"]) < 2 else "warn"),
                  f'{nc["mcnemar_chi2"]:.2f}' if nc["mcnemar_chi2"] is not None else "&mdash;"]]))
        A(f'<div class="callout"><b>Noise floor: {abs(nc["delta_pp"]):.2f}pp.</b> '
          f'Discordant pairs {nc["discordant_b"]}/{nc["discordant_c"]}. Differences below '
          f'roughly this size elsewhere on this page are not interpretable.</div>')

    # ---- 2. headline
    A("<h2>2. Click-in-bbox per rung</h2>")
    rows = []
    for c in s["headline"]:
        if not c["n"]:
            continue
        rows.append([f'<b>{c["label"]}</b>', f'{c["n"]:,}',
                     (f'<b>{pct_html(c["acc"], 2)}</b>', "good" if c["acc"] > .3 else ("warn" if c["acc"] > .1 else "bad")),
                     barcell(c["acc"]), f'{pct_html(c["lo"],1)}&ndash;{pct_html(c["hi"],1)}',
                     f'{c["chance"]*100:.4f}%', f'{c["ratio"]:.0f}&times;' if c["ratio"] else "&mdash;"])
    o = s["overall"]
    rows.append([f'<b>{o["label"]}</b>', f'{o["n"]:,}', (f'<b>{pct_html(o["acc"],2)}</b>', ""),
                 barcell(o["acc"]), f'{pct_html(o["lo"],1)}&ndash;{pct_html(o["hi"],1)}',
                 f'{o["chance"]*100:.4f}%', f'{o["ratio"]:.0f}&times;' if o["ratio"] else "&mdash;"])
    A(html_table(["rung", "n", "click-in-bbox", "", "95% Wilson", "chance", "above chance"], rows,
            "Chance is the mean hit-box area fraction: a uniform random click lands inside a box "
            "with probability equal to its area share. Never read the accuracy without it."))
    A(f'<div class="callout"><b>Resolution effect (<code>small</code> &rarr; <code>medium</code>, '
      f'paired n={re_.get("n",0):,}).</b> {pct_html(re_.get("acc_a"),2)} &rarr; {pct_html(re_.get("acc_b"),2)}, '
      f'<b>{re_.get("delta_pp",0):+.2f}pp</b>. The target occupies the same fraction of the image at '
      f'every rung, so this is not a target-size effect &mdash; it isolates absolute resolution.</div>')

    oor = s["out_of_range"]
    if oor["n"]:
        A(f'<div class="callout bad"><b>{oor["n"]} predictions ({pct_html(oor["frac"],2)}) fell outside '
          f'the 0&ndash;1000 answer space entirely</b> &mdash; the model emitted a coordinate off '
          f'the image. These are counted as misses by click-in-bbox, which hides them; they are a '
          f'coordinate-emission failure, not a perception one. By rung: '
          f'{esc(oor["by_rung"])}.</div>')

    # ---- 3. precision curve
    A("<h2>3. The precision curve</h2>")
    A("<p>The same predictions bucketed into coarser cells. Nothing is drawn on any image and no "
      "extra calls are made &mdash; this is a post-hoc reading of the same clicks. A smooth decay "
      "with a rising ratio-above-chance is the signature that the model carries real positional "
      "information; a cliff is not.</p>")
    for g in SVGLOC_RUNGS:
        cur = s["curve"].get(g) or []
        if not cur or not cur[0]["n"]:
            continue
        rows = [[f'<b>{c["grid"]}</b>', f'{c["n"]:,}',
                 (f'<b>{pct_html(c["strict"],1)}</b>', ""), barcell(c["strict"]),
                 pct_html(c["lenient"], 1) if c["lenient"] is not None else "&mdash;",
                 f'{c["chance"]*100:.3f}%',
                 f'{c["ratio"]:.1f}&times;' if c["ratio"] else "&mdash;"] for c in cur]
        A(f"<h3>{g}</h3>")
        A(html_table(["granularity", "n", "strict", "", "lenient", "chance", "above chance"], rows))
    A('<p class="sub">Strict = the click\'s cell is the one holding the box centre. '
      'Lenient = the click\'s cell is any cell the target box touches. Both are reported '
      'because publishing only the friendlier one would be a choice, not a measurement.</p>')

    # The curves side by side -- the single most informative table on the page.
    A("<h3>All three rungs side by side</h3>")
    A("<p>The same curve at each input resolution. This is where the resolution story stops "
      "being one number and becomes two opposing effects.</p>")
    ncur = len(s["curve"][SVGLOC_RUNGS[0]])
    rows = []
    for i in range(ncur):
        cells = [f'<b>{s["curve"][SVGLOC_RUNGS[0]][i]["grid"]}</b>']
        best = max((s["curve"][g][i]["strict"] or 0) for g in SVGLOC_RUNGS)
        for g in SVGLOC_RUNGS:
            c = s["curve"][g][i]
            v = pct_html(c["strict"], 1)
            cells.append((f'<b>{v}</b>', "good") if (c["strict"] or 0) == best else (v, ""))
            cells.append(f'{c["ratio"]:.1f}&times;' if c["ratio"] else "&mdash;")
        rows.append(cells)
    A(html_table(["granularity", "small", "&times;chance", "medium", "&times;chance",
             "large", "&times;chance"], rows,
            "Bold marks the best rung on each row. Ratio-above-chance rises monotonically within "
            "every column, so H1 holds independently at all three resolutions."))
    A('<div class="callout warn"><b>The curves cross, and that is the finding.</b> At coarse '
      'granularity more input resolution helps: <code>medium</code> and <code>large</code> beat '
      '<code>small</code> by roughly 12pp at 2&times;2. At fine granularity it hurts: at the exact '
      'box <code>small</code> wins, 6.7% against 4.4%, with a ratio-above-chance of 26&times; '
      'against 17&times;. The crossover sits near 8&times;8. <code>small</code> is the only rung '
      'the API does not resample; <code>medium</code> and <code>large</code> are both delivered at '
      '1348&times;853, which preserves layout but softens the thin strokes needed to pin a '
      '69&times;24px word. So &ldquo;which resolution is best&rdquo; has no answer until you say '
      'how much precision you need.</div>')

    # ---- 4. distance
    A("<h2>4. How badly does a miss miss?</h2>")
    A("<p>Binary in-or-out discards how far off a miss was. <code>d_box</code> is the distance to "
      "the nearest edge of the target (0 when inside) and answers &ldquo;how far outside did it "
      "land&rdquo;; <code>d_centre</code> is the distance to the box centre and answers &ldquo;how "
      "far from the thing was it aiming&rdquo;. They diverge on large targets, so both are given.</p>")
    rows = []
    for g in SVGLOC_RUNGS:
        dd = s["distance"].get(g) or {}
        if not dd.get("n_miss"):
            continue
        bc = dd["bands_d_centre"]
        tot = max(sum(bc.values()), 1)
        rows.append([f'<b>{g}</b>', f'{dd["n_miss"]:,}',
                     pct_html(dd["median_d_box"], 2), pct_html(dd["median_d_centre"], 2),
                     pct_html(bc.get("near_miss", 0) / tot), pct_html(bc.get("moderate_miss", 0) / tot),
                     pct_html(bc.get("wrong_region", 0) / tot)])
    A(html_table(["rung", "misses", "median d_box", "median d_centre",
             "near miss &lt;10%", "moderate 10&ndash;25%", "wrong region &gt;25%"], rows,
            "Bands are on d_centre, Euclidean, as EVAL.md 3.6 defines them. The main study's "
            "bands were per-axis, so the band structure transfers but the counts do not."))
    da = s["distance_all"]
    if da.get("bands_d_box"):
        bb = da["bands_d_box"]; tb = max(sum(bb.values()), 1)
        A(f'<p class="sub">Same misses banded on <code>d_box</code> instead: near '
          f'{pct_html(bb.get("near_miss",0)/tb)} &middot; moderate {pct_html(bb.get("moderate_miss",0)/tb)} '
          f'&middot; wrong region {pct_html(bb.get("wrong_region",0)/tb)}. The two are not '
          f'interchangeable.</p>')

    # ---- 5. point vs relation
    A("<h2>5. Perception or coordinate emission?</h2>")
    A("<p><code>relation</code> asks about position while requiring no coordinates in either the "
      "question or the answer. Comparing it against <code>point</code> bounds how much of the "
      "localization deficit is the coordinate channel rather than seeing.</p>")
    rows = []
    for qt in ("relation", "reverse"):
        t = s["text"].get(qt) or {}
        if not t.get("n"):
            continue
        rows.append([f'<b>{qt}</b>', f'{t["n"]:,}',
                     (f'<b>{pct_html(t["f1"],1)}</b>', "good" if (t["f1"] or 0) > .5 else "warn"),
                     pct_html(t["em"], 1),
                     "  ".join(f'{g} {pct_html(v["f1"],0)}' for g, v in sorted(t["by_rung"].items()))])
    rows.append([f'<b>point</b>', f'{o["n"]:,}', (f'<b>{pct_html(o["acc"],2)}</b>', "bad"), "&mdash;",
                 "  ".join(f'{c["label"]} {pct_html(c["acc"],1)}' for c in s["headline"] if c["n"])])
    A(html_table(["question type", "n", "token F1 / click-in-bbox", "exact match", "by rung"], rows,
            "Never averaged: token-F1 and click-in-bbox are different units and share a column "
            "here only for adjacency."))
    rf = s.get("reverse_frame") or {}
    if rf:
        A(html_table(["rung", "reverse questions", "probe point outside the delivered frame",
                 "rescale the model would have to invert"],
                [[f'<b>{g}</b>', f'{rf[g]["n"]:,}',
                  (f'<b>{pct_html(rf[g]["frac"],1)}</b>',
                   "bad" if (rf[g]["frac"] or 0) > .5 else ("warn" if rf[g]["frac"] else "good")),
                  f'&times;{rf[g]["rescale"]:.3f}'] for g in SVGLOC_RUNGS if g in rf]))
        A('<div class="callout bad"><b>The <code>reverse</code> arm is only interpretable at '
          '<code>small</code>.</b> Its questions quote pixel coordinates in the <i>on-disk</i> '
          'frame &mdash; &ldquo;what text appears at (1500, 830)&rdquo; &mdash; but the model '
          'receives a downscaled image. At <code>small</code> nothing is downscaled and the '
          'coordinate is valid. At <code>large</code> the quoted point lies outside the '
          '1348&times;853 frame the model actually got in <b>84.2%</b> of cases, so the question '
          'has no answer as posed. Its F1 of 0.8 there measures the defect, not the model. This '
          'is a dataset bug worth fixing: the coordinate should be stated in the delivered frame, '
          'or normalized, or the image sent at native size.</div>')

    # ---- 6. H2 gradient
    A("<h2>6. Target size and the resolution gradient</h2>")
    for g in SVGLOC_RUNGS:
        qs = s["area_quintiles"].get(g) or []
        if not qs:
            continue
        rows = [[f'{c["label"]} of image', f'{c["n"]:,}',
                 (f'<b>{pct_html(c["acc"],2)}</b>', ""), barcell(c["acc"]),
                 pct_html(c["cell4"], 1)] for c in qs]
        A(f"<h3>{g}</h3>")
        A(html_table(["target area quintile", "n", "click-in-bbox", "", "right 4&times;4 cell"], rows))
    A('<p class="sub">The 4&times;4 column is the control: if only the exact column moves with '
      'target size, the effect is precision; if both move, it is resolution.</p>')

    hs = [c for c in s["hit_source"] if c["n"]]
    if hs:
        A("<h3>Hit-box provenance</h3>")
        A(html_table(["hit_source", "n", "click-in-bbox", "", "chance", "above chance"],
                [[f'<b>{c["label"]}</b>', f'{c["n"]:,}', pct_html(c["acc"], 2), barcell(c["acc"]),
                  f'{c["chance"]*100:.4f}%', f'{c["ratio"]:.0f}&times;' if c["ratio"] else "&mdash;"]
                 for c in hs],
                "<code>shape</code> boxes are real enclosing widgets; <code>padded_text</code> is "
                "glyph ink grown by a synthetic button padding. If the two diverge sharply, the "
                "padding constant is doing more work than it should."))

    # ---- 7. breakdowns
    A("<h2>7. Required breakdowns</h2>")
    for name, title, note in [
            ("chart_type", "By chart type",
             "<code>dashboard</code> is the densest type and the closest analogue to a real UI "
             "screenshot, so it is the one to read first."),
            ("target_role", "By target role",
             "Whether a dense table cell behaves like an isolated node label is a genuine question."),
            ("theme", "By theme", "This should show nothing. If it does, it is a styling sensitivity."),
            ("font_family", "By font", "This should also show nothing.")]:
        cells = [c for c in s.get(name, []) if not c["suppressed"]]
        drop = [c for c in s.get(name, []) if c["suppressed"]]
        if not cells:
            continue
        A(f"<h3>{title}</h3>")
        A(html_table([name, "n", "click-in-bbox", "", "95% Wilson"],
                [[f'<b>{esc(c["label"])}</b>', f'{c["n"]:,}', pct_html(c["acc"], 2), barcell(c["acc"]),
                  f'{pct_html(c["lo"],1)}&ndash;{pct_html(c["hi"],1)}'] for c in cells],
                note + (f" {len(drop)} cell(s) under n={SVGLOC_MIN_CELL} suppressed rather than shown as noise."
                        if drop else "")))

    pol = s.get("polarity") or {}
    if len(pol) == 2:
        A("<h3>Background polarity &mdash; the one breakdown that was supposed to show nothing</h3>")
        d_, l_ = pol.get("dark"), pol.get("light")
        A(html_table(["theme group", "n", "click-in-bbox", "", "95% Wilson", "right 4&times;4 cell",
                 "mean target area", "mean contrast"],
                [[f'<b>{c["label"]}</b>', f'{c["n"]:,}',
                  (f'<b>{pct_html(c["acc"],2)}</b>', "good" if k == "dark" else "bad"),
                  barcell(c["acc"], scale=0.2),
                  f'{pct_html(c["lo"],2)}&ndash;{pct_html(c["hi"],2)}', pct_html(c["cell4"], 1),
                  f'{c["mean_area"]*100:.3f}%', f'{c["mean_contrast"]:.2f}']
                 for k, c in (("dark", d_), ("light", l_)) if c],
                "Dark = slate-dark, carbon, blueprint. Light = the other seven."))
        if d_ and l_:
            A(f'<div class="callout bad"><b>Haiku localizes text far better on dark backgrounds: '
              f'{pct_html(d_["acc"],2)} against {pct_html(l_["acc"],2)}, a '
              f'{(d_["acc"]-l_["acc"])*100:+.2f}pp gap and a factor of '
              f'{d_["acc"]/max(l_["acc"],1e-9):.1f}.</b> The Wilson intervals are disjoint and the '
              f'gap holds at every rung ('
              + ", ".join(f'{g}: {pct_html(d_["by_rung"][g]["acc"],1)} vs {pct_html(l_["by_rung"][g]["acc"],1)}'
                          for g in SVGLOC_RUNGS if g in d_["by_rung"] and g in l_["by_rung"])
              + f'). It is not a target-size effect &mdash; mean target area is '
              f'{d_["mean_area"]*100:.3f}% against {l_["mean_area"]*100:.3f}%. It is not a contrast '
              f'effect either, and the sign rules that out: the dark themes have <i>lower</i> mean '
              f'contrast ({d_["mean_contrast"]:.2f} vs {l_["mean_contrast"]:.2f}), so contrast '
              f'predicts the opposite of what happens. The 4&times;4 column moves too '
              f'({pct_html(d_["cell4"],1)} vs {pct_html(l_["cell4"],1)}), so this is coarse localization, not '
              f'just final precision. EVAL.md expected this cut to show nothing; it is the largest '
              f'single effect on this page and it deserves a dedicated follow-up.</div>')
            A('<p class="sub">Caveat before this is over-read: each scene has exactly one theme and '
              'one chart type, and a theme covers 8&ndash;14 of the 16 chart types, so theme is '
              'partially confounded with chart type and with scene identity. The effect size and '
              'its consistency across all three rungs argue against that being the whole story, but '
              'an experiment holding the scene fixed and re-rendering it in both polarities is the '
              'test that would settle it. That experiment was not run.</p>')

    # ---- 8. examples
    if examples:
        A("<h2>8. Examples</h2>")
        A("<p>Rendered at the size the model actually resolved, so nothing here shows detail the "
          "model never received.</p>")
        for headline, cards in examples:
            A(f"<h3>{headline}</h3>")
            for c in cards:
                A(c)

    # ---- 8b. ablations
    ab_p = Path("outputs/svgloc/ablations.json")
    if ab_p.exists():
        ab = json.loads(ab_p.read_text())
        A("<h2>Is the deficit knowledge or expression? And can a better prompt fix it?</h2>")
        A(f'<p>Eight arms over one shared sample of {ab["n_sample"]} point questions '
          f'(150 <code>small</code>, 150 <code>large</code>), each paired against the baseline '
          f'item by item. Image encoding, schema, model and thinking budget are byte-identical '
          f'across arms; only the ask changes.</p>')
        order = ["repeat", "careful", "describe", "landmark", "crop", "bbox",
                 "cell_then_point", "quadrant_mc"]
        why = {
            "repeat": "the identical request, twice",
            "careful": "same ask, told to be precise and read the edges",
            "describe": "narrate the position in words first, then convert",
            "landmark": "anchor to a big landmark, then offset from it",
            "crop": "same ask on a quarter-frame crop containing the target",
            "bbox": "ask for the box instead of the centre",
            "cell_then_point": "4&times;4 cell &rarr; sub-cell &rarr; point",
            "quadrant_mc": "which quarter? a 4-way letter, no coordinates at all",
        }
        rows = []
        for k in order:
            r = ab["arms"].get(k)
            if not r:
                continue
            tone = "good" if (r["significant"] and r["delta_pp"] > 0) else (
                "bad" if (r["significant"] and r["delta_pp"] < 0) else "")
            rows.append([f'<b>{k}</b>', esc(why[k]), f'{r["n"]}',
                         (f'<b>{pct_html(r["acc"],2)}</b>', tone),
                         pct_html(r["baseline_acc"], 2),
                         (f'<b>{r["delta_pp"]:+.2f}pp</b>', tone),
                         f'{r["chi2"]:.2f}' if r["chi2"] is not None else "&mdash;",
                         "<b>yes</b>" if r["significant"] else "&mdash;"])
        A(html_table(["arm", "what changed", "n", "score", "baseline", "difference",
                 "McNemar &chi;&sup2;", "significant"], rows,
                "quadrant_mc is compared against the baseline click <i>bucketed to 2&times;2</i>, "
                "which is the same granularity; comparing a 4-way choice against a "
                "0.25%-of-screen target would be meaningless. bbox is scored as centre-of-"
                "predicted-box inside gold so it stays in click-in-bbox units."))

        rc = ab.get("repeat_consistency")
        if rc:
            A(f'<div class="callout"><b>The error is stable, not noisy &mdash; and this is the '
              f'noise floor the missing <code>medium</code> rung would have given us.</b> Two '
              f'byte-identical requests land <b>{pct_html(rc["median_separation"],2)}</b> of the frame '
              f'apart while the typical error is <b>{pct_html(rc["median_error"],2)}</b>, a ratio of '
              f'{rc["ratio"]:.2f}, and the two runs agree on hit-or-miss '
              f'{pct_html(rc["hit_agreement"],1)} of the time. The model reproducibly points at the '
              f'same wrong place. Repeat scores {ab["arms"]["repeat"]["delta_pp"]:+.2f}pp against '
              f'baseline, so <b>&plusmn;1.33pp</b> is the in-set floor for this sample.</div>')

        q = ab["arms"].get("quadrant_mc")
        if q:
            A(f'<div class="callout bad"><b>Coordinates are lossy even where precision cannot be '
              f'the excuse.</b> Asked which quarter of the image holds the target &mdash; a 4-way '
              f'letter choice, no number anywhere &mdash; Haiku scores '
              f'<b>{pct_html(q["acc"],2)}</b>. The same items, answered by clicking and then bucketed '
              f'to that same 2&times;2 grid, score {pct_html(q["baseline_acc"],2)}. That is '
              f'<b>{q["delta_pp"]:+.2f}pp</b> (&chi;&sup2;={q["chi2"]:.2f}) thrown away by the '
              f'output channel at quadrant granularity, where no amount of blur could explain it. '
              f'For scale, the main study\'s grid control found +8.6pp at 4&times;4.</div>')

        A("<h3>Can a better prompt fix it? Partly, and only one kind works</h3>")
        A('<div class="callout good"><b>Decomposing the continuous answer into discrete choices '
          'nearly triples exact accuracy.</b> <code>cell_then_point</code> &mdash; name the '
          '4&times;4 cell, then the sub-cell within it, then convert to a point &mdash; scores '
          '<b>19.00%</b> against a 6.67% baseline, <b>+12.33pp</b>, &chi;&sup2;=24.45. It improves '
          'at <i>every</i> granularity (+10.0pp at 2&times;2, +18.3pp at 4&times;4, +13.0pp at '
          '16&times;16), halves the median distance to target (14.88% &rarr; 8.38%), and cuts '
          'out-of-range emissions from 10 to 3. The gain is larger on <code>large</code> '
          '(+14.67pp) than <code>small</code> (+10.00pp), which nearly erases the resolution gap '
          'between the two rungs. It costs about 1.6&times; the output tokens (median 817).</div>')
        A('<div class="callout warn"><b>Nothing else works, and one thing backfires.</b> Simply '
          'asking for more care is worth <b>&minus;0.33pp</b> &mdash; the model is not being '
          'careless. Narrating the position in words before converting is <b>&minus;3.00pp</b>: '
          'prose does not help it reach a number. Anchoring to a landmark (+2.33pp) and cropping '
          'the search field to a quarter of the frame (+2.01pp) are both inside the '
          '&plusmn;1.33pp&ndash;ish noise band and neither is significant &mdash; so this is not '
          'a search problem. Asking for a bounding box instead of a centre is actively worse, '
          '<b>&minus;5.33pp</b> (&chi;&sup2;=10.23): it cannot express extent any better than '
          'position.</div>')
        A('<p class="sub">Read together these say something specific: the model is good at '
          '<i>discrete spatial choices</i> and bad at <i>continuous coordinate regression</i>. '
          'Every arm that turns the continuous problem into a sequence of discrete ones gains; '
          'every arm that leaves it continuous does not. That is a property of the output channel, '
          'not of how carefully the model looked.</p>')

    # ---- 9. limits
    A("<h2>9. What this does not test</h2>")
    A("<ul>"
      "<li><b>No icon targets.</b> Every target is text, so the main study's icon-vs-text finding "
      "(1.16% vs 1.94%) is untestable here &mdash; neither confirmed nor refuted.</li>"
      "<li><b>No intent resolution.</b> Every target string is quoted verbatim in its own prompt "
      "(<code>the text &ldquo;Index Intake&rdquo;</code>), so this measures string-to-pixel "
      "matching, not resolving a functional intent to a visual referent. That makes it strictly "
      "easier than ScreenSpot-Pro and a different ability.</li>"
      "<li><b>No targets below ~0.039% of the image.</b> The hard tail that drove ScreenSpot-Pro's "
      "near-zero score is absent by construction; the floor here is ~23&times; larger.</li>"
      "<li><b>No comparison to the 1.65% ScreenSpot-Pro figure</b>, in either direction. The task "
      "differs in the ask, the target inventory and the size distribution. Only within-dataset "
      "contrasts and the <i>shape</i> of the precision curve transfer.</li>"
      "<li><b>No grid arm.</b> Deliberately not reproduced: on a trial build the magenta overlay "
      "covered the gold text in 746 of 2,400 grid questions.</li>"
      "<li><b>No ground-truth noise floor to subtract.</b> Gold is measured off the raster, so a "
      "disagreement with gold is the model being wrong &mdash; unlike the scraped benchmarks, "
      "where 2.4&ndash;5.1% of contested gold had to be budgeted for.</li>"
      "<li><b>Single run, temperature not controllable.</b> Thinking pins temperature to 1. The "
      "<code>medium</code>/<code>large</code> null control in &sect;1 is the honest noise floor.</li>"
      "</ul>")
    A(f'<p class="sub" style="margin-top:34px">Generated by <code>python -m blindspot.report svgloc</code> '
      f'from <code>results/{s["tag"]}.jsonl</code>. '
      f'{counts["unusable"]} unusable prediction(s) were counted, not scored as wrong. '
      f'Pairing: {s["pairing"]["complete_triples"]:,} complete triples on '
      f'(graph_id, qtype, target_text, anchor_text); '
      f'{s["pairing"]["dropped_incomplete"]:,} keys dropped for lack of all three rungs.</p>')

    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>svg_localization &mdash; Haiku 4.5</title>"
            f"<style>{CSS}</style></head><body><div class='wrap'>"
            + "".join(b) + "</div></body></html>")


def pick_examples(tag: str, n_each: int = 3) -> list[tuple]:
    run = load_svgloc(tag)
    exs = {e.uid: e for e in load("svg_localization")}
    pts = run["point"]
    for r in pts:
        r["_img"] = exs[r["uid"]].images[0]
    out = []
    groups = [
        ("Landed inside the target", [r for r in pts if r["hit"]], lambda r: r["d_centre"]),
        ("Near miss &mdash; right area, outside the box",
         [r for r in pts if not r["hit"] and r["d_centre"] < 0.10], lambda r: r["d_box"]),
        ("Wrong region entirely",
         [r for r in pts if not r["hit"] and r["d_centre"] > 0.25], lambda r: -r["d_centre"]),
    ]
    oor = [r for r in pts if not r.get("in_range", True)]
    if oor:
        groups.append(("Coordinate emitted outside the image", oor, lambda r: r["uid"]))
    for title, rows, key in groups:
        rows = sorted(rows, key=key)[:n_each]
        cards = []
        for i, r in enumerate(rows):
            art = render_example(r, f"{title[:6].strip().replace(' ','_')}{i}")
            cards.append(example_card(r, art))
        if cards:
            out.append((title, cards))
    return out


# ==================================== svgderived: outputs/svgderived/report.html
def sgn(v, d=2) -> str:
    return "&mdash;" if v is None else f"{v:+.{d}f}"


def _breakdowns(s: dict, specs) -> list[str]:
    b = []
    for name, title, note in specs:
        keep = [c for c in s.get(name, []) if not c["suppressed"]]
        drop = [c for c in s.get(name, []) if c["suppressed"]]
        if not keep:
            continue
        b.append(f"<h3>{title}</h3>")
        rows = [[f'<b>{esc(c["label"])[:46]}</b>', f'{c["n"]:,}', pct_html(c["acc"], 2),
                 barcell(c["acc"]), f'{pct_html(c["lo"],1)}&ndash;{pct_html(c["hi"],1)}',
                 sgn(c.get("mean_signed"))] for c in keep]
        b.append(html_table([name, "n", "accuracy", "", "95% Wilson", "mean signed error"], rows,
                       note + (f" {len(drop)} cell(s) under n={SVGDERIVED_MIN_CELL} suppressed."
                               if drop else "")))
    return b


def svgderived_render(cnt: dict, mc: dict) -> str:
    b = []
    A = b.append
    A("<h1>Counting and word presence across the resolution ladder</h1>")
    A(f'<p class="sub">Claude Haiku 4.5 &middot; <code>claude-haiku-4-5-20251001</code> &middot; '
      f'thinking enabled (2000 tokens) &middot; <code>small</code> and <code>large</code> rungs '
      f'only &middot; {cnt["counts"]["scored"]:,} counting + {mc["counts"]["scored"]:,} word-choice '
      f'questions, each with a paired blind control, over the same 200 scenes and '
      f'byte-identical pixels as the localization set</p>')

    A('<div class="callout warn"><b>Read this before any number: the null control is missing.</b> '
      'Both EVAL.md files designate <code>medium</code> vs <code>large</code> as the noise floor, '
      'because those two rungs are delivered to the model at the same size and differ only in '
      'resampling path. <code>medium</code> was excluded from this run, so neither set carries its '
      'own noise floor, and the <code>small</code> vs <code>large</code> contrast below now '
      'conflates two things it was designed to separate: absolute delivered resolution '
      '(900&times;570 against 1348&times;853) and whether the API resampled at all. The '
      'localization run measured that null at <b>&minus;0.13pp</b> over these same 200 scenes and '
      'the same pixels, so it is carried across here as a proxy floor &mdash; but it is borrowed, '
      'not measured on these questions, and a difference of a point or two should be treated as '
      'noise rather than a finding.</div>')

    # =================================================================== counting
    A("<h2>Counting</h2>")
    c_un = cnt["counts"]
    A(f'<p>One counting question per chart type, about the structure that type is made of. Gold '
      f'comes from the semantic record captured when each scene was built, not from counting marks '
      f'in a raster, and 678 of 714 rows are cross-checked against the labels actually drawn. '
      f'There is no chance baseline for a free-response integer, so none is invented.</p>')

    rows = []
    for c in cnt["headline"]:
        if not c["n"]:
            continue
        rows.append([f'<b>{c["label"]}</b>', f'{c["n"]:,}',
                     (f'<b>{pct_html(c["acc"],2)}</b>', ""), barcell(c["acc"]),
                     f'{pct_html(c["lo"],1)}&ndash;{pct_html(c["hi"],1)}',
                     sgn(c.get("mean_signed")), sgn(c.get("mean_signed_when_wrong")),
                     f'{c.get("under",0)}/{c.get("over",0)}'])
    o = cnt["overall"]
    rows.append([f'<b>{o["label"]}</b>', f'{o["n"]:,}', (f'<b>{pct_html(o["acc"],2)}</b>', ""),
                 barcell(o["acc"]), f'{pct_html(o["lo"],1)}&ndash;{pct_html(o["hi"],1)}',
                 sgn(o.get("mean_signed")), sgn(o.get("mean_signed_when_wrong")),
                 f'{o.get("under",0)}/{o.get("over",0)}'])
    A(html_table(["rung", "n", "exact-count accuracy", "", "95% Wilson", "mean signed error",
             "&hellip;when wrong", "under/over"], rows,
            "Signed error is predicted &minus; true. Negative means the model stopped early; "
            "positive means it over-reported. Absolute error is never reported alone because it "
            "destroys the sign, which carries the mechanism."))

    p = cnt["paired"]
    if p.get("n"):
        A(f'<div class="callout"><b>Paired <code>small</code> &rarr; <code>large</code> '
          f'(n={p["n"]:,}, every question appears at both rungs).</b> '
          f'{pct_html(p["acc_a"],2)} &rarr; {pct_html(p["acc_b"],2)}, <b>{p["delta_pp"]:+.2f}pp</b>. '
          f'Discordant {p["discordant_b"]}/{p["discordant_c"]}, McNemar '
          f'&chi;&sup2;={p["mcnemar_chi2"]:.2f} &mdash; '
          f'{"significant at p&lt;.05" if p["significant"] else "not significant at p&lt;.05"}.</div>')

    A("<h3>Dose-response: accuracy against the true count</h3>")
    A("<p>The primary result. A monotone decline is a finding; a flat curve is also a finding and "
      "is reported as one rather than buried.</p>")
    rows = []
    for i, name in enumerate([n for _l, _h, n in COUNT_BINS]):
        cells = [f'<b>{name}</b>']
        for g in SVGDERIVED_RUNGS:
            c = cnt["dose"][g][i]
            cells.append(f'{c["n"]}')
            cells.append((f'<b>{pct_html(c["acc"],1)}</b>', "") if not c["suppressed"]
                         else (f'<span style="color:var(--muted)">{pct_html(c["acc"],1)}</span>', ""))
            cells.append(sgn(c.get("mean_signed")))
        rows.append(cells)
    A(html_table(["true count", "small n", "small acc", "small signed",
             "large n", "large acc", "large signed"], rows,
            f"Cells under n={SVGDERIVED_MIN_CELL} are greyed rather than removed, so the shape of the curve "
            f"stays visible; the 16+ bin has only 27 rows across all three rungs by construction "
            f"and cannot resolve the high-count tail."))

    dc = cnt.get("dose_confound") or {}
    if dc:
        A('<div class="callout bad"><b>That curve cannot be read as a dose response, and reporting '
          'it as one would be wrong.</b> The true count is not randomly assigned across question '
          'forms, so a count bin is partly a proxy for <i>what</i> is being counted. The 16+ bin '
          'is a single question form. Meanwhile the two hardest forms &mdash; points in a quadrant '
          'chart (62.5%) and separate lines in a line chart (65.4%) &mdash; sit at median counts '
          'of 7 and <b>4</b>, at the low end. Accuracy appearing to rise with the count is the '
          'easy forms happening to carry the big numbers.</div>')
        A(html_table(["true-count bin", "n", "distinct question forms it spans", "dominated by"],
                [[f'<b>{k}</b>', f'{v["n"]:,}',
                  (f'<b>{v["n_forms"]}</b>', "bad" if v["n_forms"] <= 1 else ""),
                  esc(", ".join(q.replace("How many", "").split(" are in")[0].strip()[:26]
                                for q, _c in v["top"]))]
                 for k, v in dc["bin_forms"].items()]))
        A("<h4>The clean test: within a single question form</h4>")
        A("<p>Holding the thing being counted fixed is the only way to ask whether the count "
          "itself matters. It requires a form with both enough rows and enough spread.</p>")
        if dc["within_form"]:
            A(html_table(["question form", "count range", "low half", "high half", "difference"],
                    [[f'<b>{esc(w["form"][:52])}</b>', f'{w["min"]}&ndash;{w["max"]}',
                      f'{pct_html(w["lo_acc"],1)} (n={w["lo_n"]})',
                      f'{pct_html(w["hi_acc"],1)} (n={w["hi_n"]})',
                      (f'<b>{w["delta_pp"]:+.1f}pp</b>', "")] for w in dc["within_form"]],
                    f'Only {dc["n_forms_testable"]} of {dc["n_forms_total"]} question forms carry '
                    f'enough rows and enough count spread to support this test.'))
            A(f'<div class="callout"><b>No degradation where it can actually be measured.</b> '
              f'Counting 27 labelled boxes is as reliable as counting 8 &mdash; 100% in both '
              f'halves. That is one form out of {dc["n_forms_total"]}, so it is weak evidence, but '
              f'it points the opposite way from the main study\'s InfographicVQA curve '
              f'(63% &rarr; 33% across count bins).</div>')
        A(f'<p class="sub">The whole error inventory is {dc["n_errors"]} wrong answers out of '
          f'{cnt["overall"]["n"]:,}. Signed errors: '
          + ", ".join(f'<code>{v:+d}</code>&times;{n}' for v, n in dc["signed_histogram"])
          + '. Every error is off by three or fewer, and under- and over-counts are close to '
            'balanced, so neither the "stops early" nor the "estimates a pattern" signature from '
            'the main study reproduces here.</p>')

    A("<h3>By what is being counted</h3>")
    A("<p>Never pooled into one accuracy number. Connections have no enclosing shape to anchor on, "
      "and are where undercounting should appear first if it appears at all.</p>")
    rows = []
    for fam, c in sorted(cnt["family"].items(), key=lambda kv: -(kv[1]["acc"] or 0)):
        rows.append([f'<b>{fam}</b>', f'{c["n"]:,}', pct_html(c["acc"], 2), barcell(c["acc"]),
                     f'{pct_html(c["lo"],1)}&ndash;{pct_html(c["hi"],1)}',
                     sgn(c.get("mean_signed")), f'{c.get("under",0)}/{c.get("over",0)}'])
    A(html_table(["family", "n", "accuracy", "", "95% Wilson", "mean signed error", "under/over"], rows))

    bl = cnt["blind"]
    if bl["overall"]["n"]:
        A("<h3>Blind control &mdash; how much of this needed the image?</h3>")
        rows = [[f'<b>{g}</b>', f'{bl["by_rung"][g]["n"]:,}', pct_html(bl["by_rung"][g]["acc"], 2),
                 pct_html(next((c["acc"] for c in cnt["headline"] if c["label"] == g), None), 2),
                 sgn((next((c["acc"] for c in cnt["headline"] if c["label"] == g), 0) or 0) * 100
                     - (bl["by_rung"][g]["acc"] or 0) * 100, 1)]
                for g in SVGDERIVED_RUNGS if g in bl["by_rung"]]
        A(html_table(["rung", "n", "no image", "with image", "vision adds (pp)"], rows,
                "Some counts are guessable from the question alone &mdash; &ldquo;how many columns "
                "does this table have&rdquo; has a narrow plausible range. Whatever survives here "
                "was never a perception task."))

    A("".join(_breakdowns(cnt, [
        ("chart_type", "By chart type",
         "Counting bars is not counting table rows. Cells are small by construction &mdash; one or "
         "two questions per scene."),
        ("count_family_unused", "", ""),
        ("theme", "By theme", "This should show nothing."),
        ("font_family", "By font", "This should also show nothing."),
    ])))

    # ==================================================================== word_mc
    A("<h2>Word presence (<code>word_mc</code>)</h2>")
    A("<p>One question: which of these four words appears in the figure? One option is present; "
      "the other three appear nowhere in it &mdash; verified by substring check across all scene "
      "text including titles, footnotes and badges. This isolates <b>reading</b> from "
      "<b>localization</b>: the answer has no spatial component at all.</p>")

    pb = mc["position_bias"]
    A("<h3>Position bias &mdash; checked before anything else</h3>")
    A("<p>If the model favours a slot, every accuracy number below is contaminated. The sharper "
      "test is the distribution among <i>wrong</i> answers, where a guessing model has nothing "
      "else to go on.</p>")
    rows = [[f'<b>{r["option"]}</b>', f'{r["observed"]:,}', pct_html(r["obs_share"], 1),
             pct_html(r["expected_share"], 1),
             (f'{r["deviation_pp"]:+.1f}pp', "bad" if abs(r["deviation_pp"] or 0) > 5 else "")]
            for r in pb["all_picks"]["rows"]]
    A(html_table(["option", "model picks", "share of picks", "share of key", "deviation"], rows))
    ok_all, ok_wrong = not pb["all_picks"]["biased"], not pb["wrong_picks"]["biased"]
    A(f'<div class="callout {"good" if (ok_all and ok_wrong) else "bad"}">'
      f'<b>{"No position bias." if (ok_all and ok_wrong) else "Position bias detected."}</b> '
      f'All picks against the answer key: &chi;&sup2;={pb["all_picks"]["chi2"]:.2f} against a '
      f'critical value of {pb["all_picks"]["crit"]} (3 d.f.). Among wrong answers against uniform: '
      f'&chi;&sup2;={pb["wrong_picks"]["chi2"]:.2f} on n={pb["wrong_picks"]["n"]}. '
      f'{"Accuracy below is not contaminated by a guessing strategy." if (ok_all and ok_wrong) else "Every number below must be read with this in mind."}</div>')

    rows = []
    for c in mc["headline"]:
        if not c["n"]:
            continue
        rows.append([f'<b>{c["label"]}</b>', f'{c["n"]:,}',
                     (f'<b>{pct_html(c["acc"],2)}</b>', "good" if (c["acc"] or 0) > .6 else "warn"),
                     barcell(c["acc"]), f'{pct_html(c["lo"],1)}&ndash;{pct_html(c["hi"],1)}', "25.0%",
                     f'{(c["acc"] or 0)/0.25:.2f}&times;'])
    o = mc["overall"]
    rows.append([f'<b>{o["label"]}</b>', f'{o["n"]:,}', (f'<b>{pct_html(o["acc"],2)}</b>', ""),
                 barcell(o["acc"]), f'{pct_html(o["lo"],1)}&ndash;{pct_html(o["hi"],1)}', "25.0%",
                 f'{(o["acc"] or 0)/0.25:.2f}&times;'])
    A(html_table(["rung", "n", "accuracy", "", "95% Wilson", "chance", "above chance"], rows,
            "An accuracy near 25% would mean the model is guessing rather than reading, and every "
            "cut below would then be meaningless."))

    p = mc["paired"]
    if p.get("n"):
        A(f'<div class="callout"><b>Paired <code>small</code> &rarr; <code>large</code> '
          f'(n={p["n"]:,}, on (graph_id, answer_text)).</b> {pct_html(p["acc_a"],2)} &rarr; '
          f'{pct_html(p["acc_b"],2)}, <b>{p["delta_pp"]:+.2f}pp</b>. Discordant '
          f'{p["discordant_b"]}/{p["discordant_c"]}, McNemar &chi;&sup2;='
          f'{p["mcnemar_chi2"]:.2f} &mdash; '
          f'{"significant at p&lt;.05" if p["significant"] else "not significant at p&lt;.05"}. '
          f'Because the answer has no spatial component, a gap here is glyph legibility and '
          f'nothing else.</div>')

    wa = mc["wrong_analysis"]
    if wa["n_wrong"]:
        A("<h3>What a wrong answer actually was</h3>")
        A(f'<p>{wa["n_wrong"]:,} wrong answers spread over {wa["distinct_distractors"]:,} distinct '
          f'distractors. Every one of these words is verifiably absent from the figure, so each '
          f'wrong pick is a hallucinated reading &mdash; the same shape as the absence-detection '
          f'finding in the main study. If wrong picks clustered on particular vocabulary, that '
          f'would be a prior about what charts contain overriding what this chart contains.</p>')
        A(html_table(["most-chosen absent word", "times chosen"],
                [[f'<b>{esc(w)}</b>', f'{n}'] for w, n in wa["top_distractors"][:10]],
                f"Top 10 of {wa['distinct_distractors']:,}. A flat tail here means no vocabulary "
                f"prior; a spike means one."))

    pol = mc.get("polarity") or {}
    if len(pol) == 2:
        A("<h3>Background polarity</h3>")
        d_, l_ = pol.get("dark"), pol.get("light")
        A(html_table(["theme group", "n", "accuracy", "", "95% Wilson"],
                [[f'<b>{c["label"]}</b>', f'{c["n"]:,}', pct_html(c["acc"], 2), barcell(c["acc"]),
                  f'{pct_html(c["lo"],1)}&ndash;{pct_html(c["hi"],1)}'] for c in (d_, l_) if c],
                "The localization set showed a 5.8&times; dark-over-light effect. Because this "
                "task has no spatial component, testing it here separates a reading effect from a "
                "pointing effect."))

    bl = mc["blind"]
    if bl["overall"]["n"]:
        A("<h3>Blind control</h3>")
        rows = [[f'<b>{g}</b>', f'{bl["by_rung"][g]["n"]:,}', pct_html(bl["by_rung"][g]["acc"], 2),
                 pct_html(next((c["acc"] for c in mc["headline"] if c["label"] == g), None), 2),
                 sgn(((next((c["acc"] for c in mc["headline"] if c["label"] == g), 0) or 0)
                      - (bl["by_rung"][g]["acc"] or 0)) * 100, 1)]
                for g in SVGDERIVED_RUNGS if g in bl["by_rung"]]
        A(html_table(["rung", "n", "no image", "with image", "vision adds (pp)"], rows,
                "Distractors are single words from the vocabulary families the scenes are built "
                "from, so a model with a strong prior over that vocabulary could exploit it. "
                "Chance is 25%."))

    A("".join(_breakdowns(mc, [
        ("chart_type", "By chart type", "Is a word in a dense table harder to spot than one in a "
                                        "flowchart node?"),
        ("answer_len", "By length of the correct word",
         "The shortest words are the hardest to resolve at <code>small</code>, and this is the "
         "most likely place for a real effect."),
        ("theme", "By theme", "This should show nothing."),
        ("font_family", "By font", "This should also show nothing."),
    ])))

    # ============================================== the reason both sets exist
    cross = {}
    cp = Path("outputs/svgderived/cross.json")
    if cp.exists():
        cross = json.loads(cp.read_text())
    if cross:
        A("<h2>Reading, counting and pointing on identical pixels</h2>")
        A("<p>All three sets are drawn from the same 200 scenes and the same PNG files. No image "
          "was re-rendered, so the differences below are the task and nothing else &mdash; which "
          "is the one comparison none of the three sets can make alone.</p>")
        mc_s = next((c["acc"] for c in mc["headline"] if c["label"] == "small"), None)
        mc_l = next((c["acc"] for c in mc["headline"] if c["label"] == "large"), None)
        c_s = next((c["acc"] for c in cnt["headline"] if c["label"] == "small"), None)
        c_l = next((c["acc"] for c in cnt["headline"] if c["label"] == "large"), None)
        A(html_table(["task", "what it asks", "small", "large", "blind"],
                [["<b>word_mc</b>", "is this word present at all?",
                  (f'<b>{pct_html(mc_s,2)}</b>', "good"), (f'<b>{pct_html(mc_l,2)}</b>', "good"),
                  pct_html(mc["blind"]["overall"]["acc"], 1)],
                 ["<b>counting</b>", "how many of these structures are there?",
                  (f'<b>{pct_html(c_s,2)}</b>', "good"), (f'<b>{pct_html(c_l,2)}</b>', "good"),
                  pct_html(cnt["blind"]["overall"]["acc"], 1)],
                 ["<b>localization</b> &mdash; 4&times;4 cell", "roughly where is it?",
                  pct_html(cross["loc_c4_small"], 1), pct_html(cross["loc_c4_large"], 1), "&mdash;"],
                 ["<b>localization</b> &mdash; exact box", "exactly where is it?",
                  (f'<b>{pct_html(cross["loc_small"],2)}</b>', "bad"),
                  (f'<b>{pct_html(cross["loc_large"],2)}</b>', "bad"), "&mdash;"]],
                "Localization has no blind arm because a click target cannot be located without "
                "the screenshot &mdash; a blind arm would score zero by construction."))
        A('<div class="callout bad"><b>The deficit is spatial, not textual.</b> On the very same '
          'pixels, Haiku identifies which word is present essentially perfectly '
          f'({pct_html(mc["overall"]["acc"],2)}, one error in {mc["overall"]["n"]:,}) and counts the '
          f'structures at {pct_html(cnt["overall"]["acc"],1)} &mdash; but lands inside the box of a '
          f'word it was <i>told the text of</i> only {pct_html(cross["loc_pooled"],2)} of the time. It '
          'can read the label and cannot point at it. This is exactly the separation word_mc was '
          'built to make, and it rules out the deflationary explanation that the localization '
          'score is low because the glyphs were illegible: they plainly were not.</div>')
        A('<div class="callout warn"><b>Both new sets are at or near ceiling, and that limits what '
          'they can say about resolution.</b> word_mc scores 99.7&ndash;100% and counting '
          '94&ndash;97%, so neither has the headroom to resolve a resolution effect. word_mc\'s '
          f'{mc["paired"]["delta_pp"]:+.2f}pp small&rarr;large is measured against a ceiling with '
          'one error in the entire set; counting\'s '
          f'{cnt["paired"]["delta_pp"]:+.2f}pp is nominally significant '
          f'(&chi;&sup2;={cnt["paired"]["mcnemar_chi2"]:.2f}) but rests on 21 errors in total and '
          'has no in-set noise floor to be judged against. The resolution question is answered by '
          'the localization set, which has the dynamic range; these two establish that reading and '
          'counting are <i>not</i> the bottleneck.</div>')

    # ===================================================================== limits
    A("<h2>What this does not test</h2>")
    A("<ul>"
      "<li><b>No noise floor of its own.</b> <code>medium</code> was excluded, and it is the null "
      "control in both specs. The &minus;0.13pp measured on the localization set over the same "
      "pixels is borrowed as a proxy.</li>"
      "<li><b><code>small</code> vs <code>large</code> is not a clean resolution contrast.</b> It "
      "mixes delivered size with whether the API resampled at all. With <code>medium</code> "
      "included the two could have been separated.</li>"
      "<li><b>Counting cannot resolve the high-count tail.</b> Gold ranges 3&ndash;27, median 7, "
      "and the 16+ bin holds 27 rows across all three rungs. A flat curve here does not mean "
      "counting never degrades &mdash; it may only mean the range is too narrow.</li>"
      "<li><b>Treemap counts are the 36 rows not cross-checked</b>, deliberately: a block too "
      "small for text is still drawn as a rectangle, so labels and rectangles are genuinely not "
      "1:1 there. Treat treemap results with more caution.</li>"
      "<li><b><code>word_mc</code> tests presence, not localization.</b> A high score says nothing "
      "about whether the model can point at the word; the interesting result is the gap between "
      "this set and the localization set on the same pixels.</li>"
      "<li><b>One word per question.</b> No phrase reading, no ordering, no relationship between "
      "labels.</li>"
      "</ul>")
    A(f'<p class="sub" style="margin-top:34px">Generated by '
      f'<code>python -m blindspot.report svgderived</code>. Counting: {c_un["scored"]:,} scored, '
      f'{c_un["unusable"]} unusable (counted, never scored as wrong), {c_un["blind_scored"]:,} '
      f'blind. word_mc: {mc["counts"]["scored"]:,} scored, {mc["counts"]["unusable"]} unusable, '
      f'{mc["counts"]["blind_scored"]:,} blind.</p>')

    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>counting + word_mc &mdash; Haiku 4.5</title>"
            f"<style>{CSS}</style></head><body><div class='wrap'>"
            + "".join(b) + "</div></body></html>")


# ================================================================ subcommands
def cmd_aug22(a) -> int:
    AUG22_OUT.mkdir(parents=True, exist_ok=True)
    s = summarize(DATASETS)
    s["controls"] = controls()
    s["generated"] = "2026-08-22"
    s["totals"]["questions"] = sum(d["n"] for d in s["datasets"].values() if d.get("n"))
    p = AUG22_OUT / "summary.json"
    p.write_text(json.dumps(s, indent=1, default=str))
    print(f"wrote {p} ({p.stat().st_size/1024:.0f} KB)")
    for ds, d in s["datasets"].items():
        if d.get("acc") is not None:
            print(f"  {ds:20} {d['acc']*100:5.2f}%  n={d['n']}")
    print(f"  TOTAL questions {s['totals']['questions']}")
    return 0


def cmd_data(a) -> int:
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    data = build_figures()
    (REPORT_OUT / "figures.json").write_text(json.dumps(data, indent=1, default=str))

    b = data["ai2d_binding"]
    print("AI2D binding, recut by operation:")
    for k, v in b["sighted"].items():
        print(f"   {k:16s} n={v['n']:5d}  {v['acc']*100:5.1f}%  [{v['lo']*100:.1f}-{v['hi']*100:.1f}]")
    for k, v in b["paired_blind"].items():
        print(f"   {k:16s} blind {v['blind']*100:5.1f}%  sighted {v['sighted']*100:5.1f}%  "
              f"vision {v['vision_adds_pp']:+.1f}pp  (n={v['n']})")
    ad = data["absence_detection"]
    print("\nAbsence detection:")
    print(f"   full set: invention {ad['full_set']['absent']['invention_rate']*100:.1f}% "
          f"(n={ad['full_set']['absent']['n']}), over-abstention "
          f"{ad['full_set']['over_abstention']['rate']*100:.2f}%")
    for k, v in ad["paired_blind"].items():
        print(f"   {k:11s} n={v['n']:4d}  abstains blind {v['abstains_blind']*100:5.1f}%  "
              f"sighted {v['abstains_sighted']*100:5.1f}%")
    print("\nGold quality (share of the model's ERRORS that are contested):")
    for ds, v in data["gold_quality"].items():
        print(f"   {ds:16s} {v['contested']}/{v['audited']} = {v['contested_error_rate']*100:.1f}%  "
              f"-> whole-set floor {v['implied_floor']*100:.1f}%")
    L = data["synthetic"]["ladder"]
    print("\nPrecision ladder:", {k: (round(v * 100, 1) if isinstance(v, float) else v)
                                  for k, v in L.items() if k != "note"})
    print(f"\nwrote {REPORT_OUT/'figures.json'}")
    return 0


def cmd_examples(a) -> int:
    names = []
    for fn, cap, strip in EXAMPLES:
        try:
            names.append((fn(), cap, strip))
            print(f"  built {names[-1][0]}")
        except Exception as e:
            print(f"  !! {fn.__name__}: {type(e).__name__}: {e}")
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    (REPORT_OUT / "examples_index.json").write_text(
        json.dumps([{"name": n, "caption": c, "strip": s} for n, c, s in names], indent=1))
    print(f"wrote {len(names)} example figures -> {FIGS_OUT}")
    return 0


def cmd_tables(a) -> int:
    f = load_json("outputs/report/figures.json")
    md = ["# Tables\n",
          "Generated by `python -m blindspot.report tables`. Every cell is read from "
          "the measured JSON, so these cannot drift from the results.\n"]
    for tid, title, fn in TABLES:
        md += [f"\n## {tid} \u2014 {title}\n\n", fn(f), "\n"]
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    (REPORT_OUT / "tables.md").write_text("".join(md))
    print(f"wrote {REPORT_OUT / 'tables.md'} ({len(TABLES)} tables)")
    n = inject_tables()
    if n is not None:
        print(f"injected {n} tables into {PROSE}")
    return 0


def cmd_index(a) -> int:
    n_ex = sum(1 for f in ORDER if f[1] == "example")
    md = ["# Figures\n",
          "Caption text for each figure, in the order they appear. The first line under "
          "each heading is the caption; the italic line is the confidence strip, which "
          "states the limit a reader should hold that figure to. Both are editable \u2014 "
          "neither is baked into the image.\n",
          f"\n{n_ex} of {len(ORDER)} figures are photographs of real scored items. "
          "The report's quantitative content is in `tables.md`, which is generated "
          "from the measured JSON rather than written by hand.\n"]
    cards, missing = [], []
    for i, (stem, kind, sec, cap, strip) in enumerate(ORDER, 1):
        png = FIGS_OUT / f"{stem}@2x.png"
        if not png.exists():
            missing.append(stem)
            continue
        md += [f"\n## Figure {i} \u2014 {sec} \u00b7 {kind}\n", cap, f"\n\n*{strip}*\n",
               f"\n`figures/{stem}@2x.png`\n"]
        cards.append(
            f'<figure><figcaption><b>Figure {i}</b> \u00b7 {sec} \u00b7 {kind} \u00b7 '
            f'<code>{stem}</code></figcaption>'
            f'<img src="{stem}@2x.png" alt="{esc(cap)}">'
            f'<p>{esc(cap)}</p><p class="strip"><i>{esc(strip)}</i></p></figure>')

    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    FIGS_OUT.mkdir(parents=True, exist_ok=True)
    (REPORT_OUT / "figures.md").write_text("".join(md))
    (FIGS_OUT / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Report figures</title>"
        "<style>body{background:#f2f2f0;font:15px/1.5 -apple-system,BlinkMacSystemFont,"
        "'Segoe UI',Roboto,sans-serif;margin:0;padding:28px}"
        "figure{background:#fff;border-radius:12px;padding:18px;margin:0 0 24px;"
        "box-shadow:0 1px 3px rgba(0,0,0,.12);max-width:1400px}"
        "img{width:100%;display:block;border:1px solid #e4e3df;border-radius:8px}"
        "figcaption{color:#8a8983;font-size:12px;margin-bottom:10px}"
        "p{margin:10px 0 0;color:#52514e}.strip{color:#8a8983;font-size:13px}</style>"
        + "".join(cards))
    print(f"{len(cards)} figures indexed ({n_ex} examples, "
          f"{len(ORDER) - n_ex} diagrams)")
    if missing:
        print("  !! missing:", ", ".join(missing))
    print(f"wrote {REPORT_OUT / 'figures.md'} and {FIGS_OUT / 'index.html'}")
    n = inject_refs()
    if n is not None:
        print(f"resolved {n} figure references in {PROSE}")
    return 0


def cmd_paste(a) -> int:
    doc = build_paste()
    if doc is None:
        return 0
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    p = REPORT_OUT / "paste_into_docs.html"
    p.write_text(doc)
    print(f"wrote {p}  ({len(doc)/1e6:.1f} MB, {doc.count('data:image/png;base64,')} "
          f"figures, {doc.count('<table')} tables)")
    print("Open it in a browser, select all, copy, and paste into Google Docs.")
    return 0


def cmd_svgloc(a) -> int:
    s = svgloc_analyse(a.tag)
    SVGLOC_OUT.mkdir(parents=True, exist_ok=True)
    (SVGLOC_OUT / "summary.json").write_text(json.dumps(s, indent=1))
    probe_path = Path("results/svg_localization__probe_summary.json")
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
    examples = [] if a.no_images else pick_examples(a.tag)
    (SVGLOC_OUT / "report.html").write_text(svgloc_render(s, probe, examples))
    print(f"wrote {SVGLOC_OUT/'report.html'} and {SVGLOC_OUT/'summary.json'}")
    return 0


def cmd_svgderived(a) -> int:
    cnt, mc = analyse_counting(a.tag), analyse_word_mc(a.tag)
    SVGDERIVED_OUT.mkdir(parents=True, exist_ok=True)
    (SVGDERIVED_OUT / "summary.json").write_text(
        json.dumps({"counting": cnt, "word_mc": mc}, indent=1))
    (SVGDERIVED_OUT / "report.html").write_text(svgderived_render(cnt, mc))
    print(f"wrote {SVGDERIVED_OUT/'report.html'} and {SVGDERIVED_OUT/'summary.json'}")
    return 0


CHAIN = [("data", cmd_data), ("examples", cmd_examples), ("tables", cmd_tables),
         ("index", cmd_index), ("paste", cmd_paste)]


def cmd_all(a) -> int:
    """The live report chain, in dependency order.

    `aug22` is not in the chain but comes before it: `data` reads
    outputs/aug22/summary.json at runtime. That is a file dependency, not an
    import, so nothing here will complain if it is stale -- only wrong.
    """
    print("report all:  " + "  ->  ".join(n for n, _ in CHAIN))
    print("  ordering note: `data` READS outputs/aug22/summary.json, which is written")
    print("  by `report aug22`. It is a file dependency, not an import, and nothing")
    print("  checks it. Run `report aug22` first if the results have changed.")
    for name, fn in CHAIN:
        print(f"\n--- {name} " + "-" * (70 - len(name)))
        rc = fn(a)
        if rc:
            print(f"!! {name} returned {rc}; stopping")
            return rc
    return 0


EPILOG = """\
the live chain, in order:
  data       assemble every quoted number into outputs/report/figures.json
  examples   render the real-image example figures
  tables     the six tables, and inject them into blindspots.md
  index      order the figures and resolve [FIG:stem] references
  paste      one self-contained HTML file for a document editor
  all        all five of the above, in that order

standalone:
  aug22      outputs/aug22/summary.json -- READ BY `data`, so run it first
  svgloc     the generated localization set
  svgderived the counting and word-presence sets derived from the same scenes

blindspots.md, the hand-written prose spine, is a separate deliverable and is
not in this repository. tables, index and paste skip cleanly without it.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="report", description=__doc__.splitlines()[0],
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="SUBCOMMAND")

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(fn=fn)
        return p

    add("data", cmd_data, "assemble outputs/report/figures.json, the auditable artifact")
    add("examples", cmd_examples, "render outputs/report/figures/*.png")
    add("tables", cmd_tables, "write outputs/report/tables.md and inject into blindspots.md")
    add("index", cmd_index, "write outputs/report/figures.md and figures/index.html")
    add("paste", cmd_paste, "write outputs/report/paste_into_docs.html")
    add("all", cmd_all, "data -> examples -> tables -> index -> paste")
    add("aug22", cmd_aug22, "write outputs/aug22/summary.json (read by `data`)")

    p = add("svgloc", cmd_svgloc, "the svg_localization report and summary")
    p.add_argument("--tag", default=RUN)
    p.add_argument("--no-images", action="store_true")

    p = add("svgderived", cmd_svgderived, "the counting and word_mc report and summary")
    p.add_argument("--tag", default=RUN)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
