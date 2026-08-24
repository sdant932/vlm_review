#!/usr/bin/env python
"""Runs into numbers -- the whole analysis layer behind one command.

    python -m blindspot.eval aggregate       -> outputs/summary.json
    python -m blindspot.eval localization    -> outputs/svgloc/summary.json
    python -m blindspot.eval derived         -> outputs/svgderived/summary.json
    python -m blindspot.eval ablations       -> outputs/svgloc/ablations.json
    python -m blindspot.eval annotate        -> outputs/annotations/ (+ galleries)
    python -m blindspot.eval tiling          -> results/<ds>__tiled3x3_r0.jsonl

THE LAYER'S RULE (from REPO_MAP.md): this layer reads `results/*.jsonl` and
writes JSON. It does not render the report. Keeping rendering out is not
tidiness -- it is what makes every number independently checkable, and what
lets a report rebuild in seconds without re-scoring thousands of rows. A
reviewer diffs two runs by diffing JSON instead of parsing a web page.

Two subcommands sit outside that rule, and it is said here rather than left to
be discovered:

* `annotate` also emits browsable HTML galleries. The artifact is still the
  per-question sidecar JSON; the galleries are a browsing aid over it, not a
  report, and nothing downstream reads numbers out of them.
* `tiling` is a one-off experiment that *calls the API*. It writes into
  `results/` like a runner rather than reading from it, and it is the only
  subcommand here that costs money.

Six subcommands:
aggregate, svgloc_eval, svgderived_eval, svgloc_ablation_eval, annotate,
tiling. Imports from `blindspot.core.*` are deliberately left exactly as the
originals had them.
"""

from __future__ import annotations

import argparse
import base64
import collections
import glob
import html
import io
import json
import math
import os
import random
import statistics as st
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from blindspot.core import load
from blindspot.core import classify as classify_failure, classify_point
from blindspot.core import Budget, MODEL, RESULTS as RUNNER_RESULTS
from blindspot.core import point_in_bbox, score, token_f1
from blindspot.core import wilson, cell_of, centre_cell, bbox_cells, quantiles
from blindspot.core import LABELS, is_not_applicable, primitive_for

# `svgderived_eval` imported core.scoring.score under this name to distinguish
# the benchmark's own scorer from the local hit/miss derived from it.
official_score = score

RESULTS = Path("results")
OUT = Path("outputs")

LOC_RUNGS = ("small", "medium", "large")     # svg_localization ran all three
DERIVED_RUNGS = ("small", "large")           # counting / word_mc ran two
GRIDS = (2, 3, 4, 8, 16)
MIN_CELL = 30          # EVAL.md 5: suppress cells under n=30 rather than show noise
DS = "svg_localization"
DARK_THEMES = {"slate-dark", "carbon", "blueprint"}


# ===========================================================================
#  Shared helpers
# ===========================================================================

def read_jsonl(path: Path) -> list[dict]:
    """Every parseable JSON object in a .jsonl, or [] if the file is absent.

    Deduplicated from three near-identical copies (`svgloc_eval.read_jsonl`,
    `svgderived_eval.read_jsonl`, `svgloc_ablation_eval._rows`). The only
    difference between them was that two used a bare `open()` whose handle was
    left to the garbage collector; this keeps the `with` form, which closes
    deterministically. Malformed lines are dropped silently in all three
    originals and are dropped silently here -- the behaviour on parseable
    content is identical.
    """
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def by(rows, keyfn) -> dict:
    """Group rows by a key function. Identical in svgloc_eval and svgderived_eval."""
    g = collections.defaultdict(list)
    for r in rows:
        g[keyfn(r)].append(r)
    return dict(g)


# ===========================================================================
#  aggregate -- results/*.jsonl -> outputs/summary.json
# ===========================================================================

def load_rows(dataset: str, with_judge: bool = True) -> list[dict]:
    """All usable rows for a dataset, unioned across tag schemes.

    ScreenSpot-Pro results ended up split across two differently-tagged files
    mid-project; unioning here (best row per uid wins) is what stops the report
    silently counting half a dataset.
    """
    examples = {e.uid: e for e in load(dataset)}
    best: dict[str, dict] = {}
    for f in sorted(glob.glob(f"results/{dataset}__*.jsonl")):
        if ".judged" in f or "_excluded" in f or "__gtaudit" in f or "__equiv" in f:
            continue
        for line in open(f):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            prev = best.get(rec["uid"])
            if prev is None or prev.get("pred") is None or rec.get("pred") is not None:
                best[rec["uid"]] = rec

    judge = judged_scores(dataset) if with_judge else {}
    equiv = equiv_verdicts(dataset)
    rows = []
    for uid, rec in best.items():
        ex = examples.get(uid)
        if ex is None or rec.get("pred") is None:
            continue
        prim, prov = primitive_for(ex)
        row = dict(rec)
        row.update(score(ex, rec["pred"]))
        row.update({
            "_ex": ex, "question": ex.question, "primitive": prim,
            "provenance": prov, "not_applicable": is_not_applicable(ex),
        })
        # CharXiv's official grader is authoritative where it ran; keep the
        # string-match score alongside so the two remain comparable per item.
        if uid in judge:
            row["judge_score"] = judge[uid]
            row["string_score"] = row["score"]
            row["score"] = judge[uid]
            row["metric"] = "charxiv_official_judge"
        # Why it failed, not just that it did. Deterministic where the answer is
        # list-shaped; the LLM pass resolves the rest.
        if (row.get("score") or 0) < 0.5:
            if ex.answer_type == "point":
                mode = classify_point(ex.gold, rec.get("pred"))
            elif ex.answer_type == "choice":
                mode = "wrong_option"
            else:
                mode = classify_failure(ex.gold, rec.get("pred"))
            v = equiv.get(uid)
            if v:
                if v.get("equivalent"):
                    mode = "format_only"
                    row["meaning_equivalent"] = True
                elif mode == "unclassified":
                    mode = v.get("failure_mode", "unclassified")
                row["gold_looks_wrong"] = bool(v.get("gold_looks_wrong"))
            row["failure_mode"] = mode
        rows.append(row)
    return rows


def equiv_verdicts(dataset: str) -> dict[str, dict]:
    """uid -> meaning-equivalence + failure-mode verdict, when that pass has run."""
    out = {}
    p = RESULTS / f"{dataset}__equiv.jsonl"
    if p.exists():
        for line in open(p):
            try:
                v = json.loads(line)
            except Exception:
                continue
            if "error" not in v:
                out[v["uid"]] = v
    return out


def judged_scores(dataset: str) -> dict[str, float]:
    """uid -> official LLM-judge score, when a judge pass exists."""
    out = {}
    for f in glob.glob(f"results/{dataset}__*.judged.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("judge_score") is not None:
                out[r["uid"]] = float(r["judge_score"])
    return out


def wilson_of_values(vals: list[float]) -> tuple[float, float] | None:
    """Wilson interval over a list of per-item scores. NOT `core.stats.wilson`.

    Kept distinct on purpose. `core.stats.wilson(k, n)` takes a success count
    and a trial count, returns `(0.0, 1.0)` when n == 0, and does not clamp.
    This one takes the scores themselves (which for CharXiv can be fractional,
    not just 0/1), returns None when the list is empty so the caller can emit a
    null rather than a fake full-width interval, and clamps the endpoints into
    [0, 1]. Substituting either for the other changes published numbers.
    """
    n = len(vals)
    if not n:
        return None
    p, z = sum(vals) / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0)) / d
    return max(0.0, c - m), min(1.0, c + m)


def agg_cell(rows: list[dict], key: str = "score") -> dict | None:
    """summary.json's cell shape: acc / n / ci_lo / ci_hi, or None when empty."""
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    ci = wilson_of_values(vals)
    return {"acc": sum(vals) / len(vals), "n": len(vals),
            "ci_lo": ci[0] if ci else None, "ci_hi": ci[1] if ci else None}


def slice_by(rows: list[dict], keyfn: Callable[[dict], Any], min_n: int = 1) -> list[dict]:
    g = defaultdict(list)
    for r in rows:
        g[keyfn(r)].append(r)
    out = []
    for k, v in g.items():
        c = agg_cell(v)
        if c and c["n"] >= min_n:
            out.append({"label": str(k), **c})
    return sorted(out, key=lambda d: d["acc"])


DATASETS = ["charxiv", "infographicvqa", "screenspot_pro", "ai2d"]


def summarize(datasets: list[str] | None = None) -> dict:
    datasets = datasets or DATASETS
    all_rows: dict[str, list[dict]] = {ds: load_rows(ds) for ds in datasets}
    judged = {ds: judged_scores(ds) for ds in datasets}

    summary: dict[str, Any] = {
        "model": "claude-haiku-4-5-20251001",
        "datasets": {},
        "primitives": {},
        "charxiv": {},
        "localization": {},
        "totals": {},
    }

    # ---- per dataset -----------------------------------------------------
    for ds, rows in all_rows.items():
        c = agg_cell(rows) or {}
        fails = [r for r in rows if (r.get("score") or 0) < 0.5]
        modes = collections.Counter(r.get("failure_mode", "unclassified") for r in fails)
        n_equiv = sum(1 for r in fails if r.get("meaning_equivalent"))
        summary["datasets"][ds] = {
            **c,
            "errors": 0,
            "judged_n": len(judged.get(ds, {})),
            "failures": len(fails),
            "failure_modes": dict(modes),
            "meaning_equivalent": n_equiv,
            # Official metric stays the headline; this is the adjusted twin.
            "acc_meaning_adjusted": ((c.get("acc", 0) * c.get("n", 0) + n_equiv) / c["n"]
                                     if c.get("n") else None),
            "gold_looks_wrong": sum(1 for r in fails if r.get("gold_looks_wrong")),
        }

    # ---- primitive x dataset matrix (the headline) -----------------------
    # N/A-heavy cells are scored on the answerable subset as well, because a
    # pooled number there measures "can you tell this doesn't apply".
    for prim in LABELS:
        per_ds = {}
        for ds, rows in all_rows.items():
            sel = [r for r in rows if r["primitive"] == prim]
            if not sel:
                continue
            answerable = [r for r in sel if not r["not_applicable"]]
            entry = {"all": agg_cell(sel), "answerable": agg_cell(answerable),
                     "na_rate": 1 - len(answerable) / len(sel),
                     "provenance": sel[0]["provenance"]}
            per_ds[ds] = entry
        if per_ds:
            pooled = [r for rows in all_rows.values() for r in rows if r["primitive"] == prim]
            pooled_ans = [r for r in pooled if not r["not_applicable"]]
            summary["primitives"][prim] = {
                "label": LABELS[prim],
                "sources": per_ds,
                "pooled": agg_cell(pooled),
                "pooled_answerable": agg_cell(pooled_ans),
                "n_sources": len(per_ds),
            }

    # ---- CharXiv: descriptive vs reasoning, and judge vs string match -----
    cx = all_rows.get("charxiv", [])
    if cx:
        desc = [r for r in cx if r["_ex"].meta.get("split") == "descriptive"]
        reas = [r for r in cx if r["_ex"].meta.get("split") == "reasoning"]
        summary["charxiv"]["descriptive"] = agg_cell(desc)
        summary["charxiv"]["reasoning"] = agg_cell(reas)
        summary["charxiv"]["by_qlabel"] = slice_by(
            desc, lambda r: r["_ex"].meta.get("qlabel") or "?", min_n=5)
        summary["charxiv"]["by_qlabel_answerable"] = slice_by(
            [r for r in desc if not r["not_applicable"]],
            lambda r: r["_ex"].meta.get("qlabel") or "?", min_n=5)
        j = judged.get("charxiv", {})
        if j:
            # Compare against string_score: `score` now holds the judge verdict,
            # so using it here would compare the judge with itself (100% by
            # construction) and silently destroy the check.
            paired_scores = [(r.get("string_score", r["score"]), j[r["uid"]])
                             for r in cx if r["uid"] in j]
            ours = [1.0 if s >= 0.5 else 0.0 for s, _ in paired_scores]
            off = [jj for _, jj in paired_scores]
            summary["charxiv"]["grader_comparison"] = {
                "n": len(paired_scores),
                "string_match": sum(ours) / len(ours) if ours else None,
                "official_judge": sum(off) / len(off) if off else None,
                "agreement": (sum(1 for a, b in zip(ours, off) if a == b) / len(paired_scores)
                              if paired_scores else None),
            }

    # ---- localization: the resolution story -------------------------------
    sp = all_rows.get("screenspot_pro", [])
    if sp:
        def bucket(r):
            frac = r["_ex"].meta.get("target_area_frac", 0) or 0
            side = math.sqrt(frac * 1568 * 882)
            for lim, name in ((12, "<12px"), (20, "12-20px"), (32, "20-32px"), (56, "32-56px")):
                if side < lim:
                    return name
            return ">=56px"
        summary["localization"] = {
            "by_target_size": slice_by(sp, bucket, min_n=5),
            "by_ui_type": slice_by(sp, lambda r: r["_ex"].meta.get("ui_type") or "?", min_n=5),
            "by_application": slice_by(sp, lambda r: r["_ex"].meta.get("group") or "?", min_n=10),
        }

    # ---- benchmark quality: reported as a caveat, never as an adjustment ----
    # Scores stay on the official metric over the full official split. These
    # numbers tell a reader how much annotation noise sits underneath, which is
    # a different thing from correcting for it: filtering to the cells a model
    # does well on would be indistinguishable from cherry-picking.
    quality = {}
    for ds, rows in all_rows.items():
        p_aud = RESULTS / f"{ds}__gtaudit.jsonl"
        if not p_aud.exists():
            continue
        aud = []
        for line in open(p_aud):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" not in r:
                aud.append(r)
        if not aud:
            continue
        bad = [r for r in aud if r.get("gt_quality") != "unambiguous"]
        fails = [r for r in rows if (r.get("score") or 0) < 0.5]
        rate = len(bad) / len(aud)
        quality[ds] = {
            "audited": len(aud),
            "failures": len(fails),
            "total": len(rows),
            "bad_gt_in_failures": rate,
            # The number that actually characterises the benchmark: the audit
            # sampled failures, where bad GT is enriched because it causes them.
            "bad_gt_whole_set": rate * len(fails) / max(len(rows), 1),
            "headline_if_all_credited": ((sum(r.get("score") or 0 for r in rows)
                                          + rate * len(fails)) / len(rows)) if rows else None,
        }
    summary["benchmark_quality"] = quality

    # ---- AI2D blind control: how much of the score needs the image at all ----
    p_blind = RESULTS / "ai2d_blind_control.json"
    if p_blind.exists():
        blind = json.loads(p_blind.read_text())
        acc_by = collections.defaultdict(list)
        for r in blind:
            acc_by[r["qtype"]].append(r["pred"] == r["gold"])
        seen = {}
        for r in all_rows.get("ai2d", []):
            seen.setdefault(r["_ex"].meta.get("qtype"), []).append(r.get("score") or 0)
        summary["ai2d_blind_control"] = {
            k: {"blind": sum(v) / len(v), "blind_n": len(v),
                "with_image": (sum(seen[k]) / len(seen[k])) if seen.get(k) else None,
                "n": len(seen.get(k, []))}
            for k, v in acc_by.items()}

    n_all = sum(len(v) for v in all_rows.values())
    summary["totals"] = {
        "questions": n_all,
        "primitives_measured": len(summary["primitives"]),
        "multi_source_primitives": sum(1 for p in summary["primitives"].values()
                                       if p["n_sources"] > 1),
    }
    return summary


def cmd_aggregate(a: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s = summarize()
    p = OUT / "summary.json"
    p.write_text(json.dumps(s, indent=1, default=str))
    print(f"wrote {p}  ({p.stat().st_size/1024:.0f} KB)")
    print(f"  {s['totals']['questions']} questions | "
          f"{s['totals']['primitives_measured']} primitives | "
          f"{s['totals']['multi_source_primitives']} with >1 source")
    for ds, d in s["datasets"].items():
        if d.get("acc") is not None:
            print(f"  {ds:16} {d['acc']*100:5.1f}%  n={d['n']}")
    return 0


# ===========================================================================
#  localization -- data/svg_localization, following its EVAL.md
# ===========================================================================
#
# Design rules taken straight from EVAL.md and enforced here rather than left
# to the caller:
#
# * score `point` against `gold_bbox_norm` (the widget hit box), never
#   `text_ink_bbox_norm` (the glyph outline, ~3x smaller and a different task);
# * chance is the mean hit-box area fraction, not 1/n of anything, because a
#   uniform random click lands in a box with probability equal to its area
#   share;
# * never score a null or unparseable prediction as wrong -- count it
#   separately;
# * never average across metrics, and never average across resolutions;
# * pair on (graph_id, qtype, target_text, anchor_text), never on the uid
#   index, because the eligible-target pool differs per rung and question `:02`
#   is not the same target at every resolution.

def load_loc_run(tag: str) -> dict:
    """Join a result file to the examples and score it.

    Retries append, so the last usable row per uid wins -- same convention as
    the main study's loader. Rows that never produced a prediction are kept
    aside and counted, never scored as wrong.
    """
    exs = {e.uid: e for e in load(DS)}
    raw = read_jsonl(RESULTS / f"{DS}__{tag}.jsonl")
    best: dict[str, dict] = {}
    for r in raw:
        if r.get("pred") is None:
            best.setdefault(r["uid"], r)
        else:
            best[r["uid"]] = r
    point, span, unusable = [], [], []
    for uid, r in best.items():
        e = exs.get(uid)
        if e is None:
            continue
        if r.get("pred") is None:
            unusable.append({"uid": uid, "reason": r.get("parse_error") or r.get("error") or "null"})
            continue
        row = {"uid": uid, "pred": r["pred"], "meta": e.meta, "gold": e.gold,
               "question": e.question, "thinking": r.get("thinking") or "",
               "sent": (r.get("sent_image_sizes") or [None])[0],
               "usage": r.get("usage") or {}, "latency_s": r.get("latency_s")}
        if e.answer_type == "point":
            x, y = row["pred"]
            row["in_range"] = 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
            row["hit"] = bool(point_in_bbox((x, y), e.gold))
            row["d_box"] = d_box((x, y), e.gold)
            row["d_centre"] = d_centre((x, y), e.gold)
            point.append(row)
        else:
            em, f1 = token_f1(r["pred"], e.gold)
            row["em"], row["f1"] = em, f1
            span.append(row)
    return {"tag": tag, "point": point, "span": span, "unusable": unusable,
            "lines": len(raw), "unique": len(best)}


# ----------------------------------------------------------------- geometry
def d_box(pred, box) -> float:
    """Euclidean distance to the nearest point of the hit box; 0 when inside.

    `point_in_bbox` is exactly `d_box == 0`, which is why this is the natural
    continuous companion to the binary metric rather than a separate story.
    """
    x, y = pred
    x0, y0, x1, y1 = box
    return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))


def d_centre(pred, box) -> float:
    x, y = pred
    x0, y0, x1, y1 = box
    return math.hypot(x - (x0 + x1) / 2, y - (y0 + y1) / 2)


def band(d: float) -> str:
    """EVAL.md 3.6 bands, on a Euclidean distance.

    The main study's `classify_point` used a per-axis (L-infinity) test; EVAL.md
    3.6 defines the bands over a distance, so this is Euclidean. The band
    structure is what transfers between the two studies, not the counts.
    """
    if d < 0.10:
        return "near_miss"
    if d <= 0.25:
        return "moderate_miss"
    return "wrong_region"


# ------------------------------------------------------------------ helpers
def acc(rows) -> float | None:
    return (sum(r["hit"] for r in rows) / len(rows)) if rows else None


def chance_of(rows) -> float:
    return st.mean(r["meta"]["target_area_frac"] for r in rows) if rows else 0.0


def loc_cell(rows, label, extra=None) -> dict:
    """One reportable cell: n, accuracy, Wilson interval, chance, ratio.

    Deliberately NOT merged with `derived_cell` below. It reads
    `meta["target_area_frac"]` through `chance_of`, which the counting and
    word_mc rows do not carry, and its key order is the published order in
    outputs/svgloc/summary.json. Sharing an implementation would either raise
    KeyError on the derived sets or reorder a published artifact.
    """
    n = len(rows)
    k = sum(r["hit"] for r in rows)
    ch = chance_of(rows)
    lo, hi = wilson(k, n)
    out = {"label": label, "n": n, "k": k,
           "acc": (k / n) if n else None, "lo": lo, "hi": hi,
           "chance": ch, "ratio": (k / n / ch) if n and ch else None,
           "suppressed": n < MIN_CELL}
    if extra:
        out.update(extra)
    return out


# ------------------------------------------------------------------ analyses
def precision_curve(rows) -> list[dict]:
    """EVAL.md 3.4 + 3.5: strict and lenient, at every rung, with chance.

    Strict = the click's cell equals the cell holding the box centre.
    Lenient = the click's cell is any cell the box touches.
    """
    n = len(rows)
    out = []
    for g in GRIDS:
        k = sum(1 for r in rows if cell_of(*r["pred"], g) == centre_cell(r["gold"], g))
        anyk = sum(1 for r in rows if cell_of(*r["pred"], g) in bbox_cells(r["gold"], g))
        ch = 1 / (g * g)
        lo, hi = wilson(k, n)
        out.append({"grid": f"{g}x{g}", "n": n, "strict": k / n if n else None,
                    "lenient": anyk / n if n else None, "chance": ch,
                    "ratio": (k / n / ch) if n else None, "lo": lo, "hi": hi})
    k = sum(r["hit"] for r in rows)
    ch = chance_of(rows)
    lo, hi = wilson(k, n)
    out.append({"grid": "exact hit box", "n": n, "strict": k / n if n else None,
                "lenient": None, "chance": ch,
                "ratio": (k / n / ch) if n and ch else None, "lo": lo, "hi": hi})
    return out


def loc_paired(rows_by_rung: dict, a: str, b: str) -> dict:
    """Paired comparison on genuinely-identical targets (EVAL.md 4.1).

    Key is (graph_id, qtype, target_text, anchor_text). Pairing on the uid's
    question index would silently compare different targets on 194 of 1,555
    triples, because the eligible-target pool differs per rung.
    """
    def key(r):
        m = r["meta"]
        return (m["graph_id"], m["qtype"], m.get("target_text"), m.get("anchor_text"))

    ka = {key(r): r for r in rows_by_rung.get(a, [])}
    kb = {key(r): r for r in rows_by_rung.get(b, [])}
    both = sorted(set(ka) & set(kb), key=lambda t: (t[0], str(t[2])))
    if not both:
        return {"a": a, "b": b, "n": 0}
    ha = sum(ka[k]["hit"] for k in both)
    hb = sum(kb[k]["hit"] for k in both)
    # McNemar discordants: b_ = a hit & b missed, c_ = a missed & b hit
    b_ = sum(1 for k in both if ka[k]["hit"] and not kb[k]["hit"])
    c_ = sum(1 for k in both if not ka[k]["hit"] and kb[k]["hit"])
    chi = ((abs(b_ - c_) - 1) ** 2 / (b_ + c_)) if (b_ + c_) else None
    return {"a": a, "b": b, "n": len(both),
            "acc_a": ha / len(both), "acc_b": hb / len(both),
            "delta_pp": (hb - ha) / len(both) * 100,
            "discordant_b": b_, "discordant_c": c_, "mcnemar_chi2": chi,
            "median_dbox_a": st.median([ka[k]["d_box"] for k in both]),
            "median_dbox_b": st.median([kb[k]["d_box"] for k in both])}


def triples(rows_by_rung: dict) -> dict:
    def key(r):
        m = r["meta"]
        return (m["graph_id"], m["qtype"], m.get("target_text"), m.get("anchor_text"))
    keysets = {g: {key(r) for r in rows_by_rung.get(g, [])} for g in LOC_RUNGS}
    complete = set.intersection(*keysets.values()) if all(keysets.values()) else set()
    allk = set().union(*keysets.values()) if any(keysets.values()) else set()
    return {"complete_triples": len(complete), "any_rung": len(allk),
            "dropped_incomplete": len(allk) - len(complete)}


def dist_summary(rows) -> dict:
    misses = [r for r in rows if not r["hit"]]
    def pct(vals, q):
        return st.quantiles(vals, n=100)[q - 1] if len(vals) > 2 else (vals[0] if vals else None)
    db = [r["d_box"] for r in misses]
    dc = [r["d_centre"] for r in misses]
    dc_all = [r["d_centre"] for r in rows]
    return {
        "n_miss": len(misses),
        "median_d_box": st.median(db) if db else None,
        "median_d_centre": st.median(dc) if dc else None,
        "median_d_centre_all": st.median(dc_all) if dc_all else None,
        "p90_d_box": pct(sorted(db), 90) if db else None,
        "p90_d_centre": pct(sorted(dc), 90) if dc else None,
        "bands_d_centre": dict(collections.Counter(band(r["d_centre"]) for r in misses)),
        "bands_d_box": dict(collections.Counter(band(r["d_box"]) for r in misses)),
    }


def analyse_localization(tag: str) -> dict:
    run = load_loc_run(tag)
    pts, spans = run["point"], run["span"]
    out: dict = {"tag": tag,
                 "counts": {"lines": run["lines"], "unique": run["unique"],
                            "point_scored": len(pts), "span_scored": len(spans),
                            "unusable": len(run["unusable"]),
                            "unusable_detail": run["unusable"][:20]}}

    # Coordinate hygiene -- a prediction outside [0,1] is a model error, not a
    # miss, and it is invisible in click-in-bbox because it just never hits.
    oor = [r for r in pts if not r["in_range"]]
    out["out_of_range"] = {"n": len(oor),
                           "frac": len(oor) / len(pts) if pts else None,
                           "by_rung": dict(collections.Counter(
                               r["meta"]["resolution"] for r in oor)),
                           "examples": [{"uid": r["uid"], "pred": r["pred"]} for r in oor[:10]]}

    by_rung = by(pts, lambda r: r["meta"]["resolution"])
    out["headline"] = [loc_cell(by_rung.get(g, []), g) for g in LOC_RUNGS]
    out["overall"] = loc_cell(pts, "all rungs pooled")
    out["pairing"] = triples(by_rung)
    out["null_control"] = loc_paired(by_rung, "medium", "large")   # expect ~0
    out["resolution_effect"] = loc_paired(by_rung, "small", "medium")
    out["curve"] = {g: precision_curve(by_rung.get(g, [])) for g in LOC_RUNGS}
    out["curve_all"] = precision_curve(pts)
    out["distance"] = {g: dist_summary(by_rung.get(g, [])) for g in LOC_RUNGS}
    out["distance_all"] = dist_summary(pts)

    # H2 gradient: area quintiles, computed within rung so the quintile is not
    # just a proxy for which rung the row came from.
    out["area_quintiles"] = {}
    for g in LOC_RUNGS:
        qs = quantiles(by_rung.get(g, []), lambda r: r["meta"]["target_area_frac"], 5)
        out["area_quintiles"][g] = [
            loc_cell(ch, f"{lo*100:.4f}-{hi*100:.4f}%",
                     {"lo_frac": lo, "hi_frac": hi,
                      "cell4": sum(1 for r in ch if cell_of(*r["pred"], 4) == centre_cell(r["gold"], 4)) / len(ch)})
            for lo, hi, ch in qs]

    # EVAL.md 5 predicted theme would show nothing. It shows a great deal, so it
    # gets its own cut, with the two obvious confounds measured beside it: if
    # dark themes simply had bigger or higher-contrast targets, that would
    # explain the gap without any styling sensitivity.
    grp = by(pts, lambda r: "dark" if r["meta"]["theme"] in DARK_THEMES else "light")
    out["polarity"] = {}
    for k, v in grp.items():
        c = loc_cell(v, k)
        c["cell4"] = (sum(1 for r in v if cell_of(*r["pred"], 4) == centre_cell(r["gold"], 4))
                      / len(v)) if v else None
        c["mean_area"] = st.mean(r["meta"]["target_area_frac"] for r in v) if v else None
        c["mean_contrast"] = st.mean(r["meta"]["target_contrast"] for r in v) if v else None
        c["by_rung"] = {g: {"n": len(vv), "acc": acc(vv)}
                        for g, vv in by(v, lambda r: r["meta"]["resolution"]).items()}
        out["polarity"][k] = c

    for name, keyfn in [("hit_source", lambda r: r["meta"]["hit_source"]),
                        ("chart_type", lambda r: r["meta"]["chart_type"]),
                        ("target_role", lambda r: r["meta"]["target_role"]),
                        ("theme", lambda r: r["meta"]["theme"]),
                        ("font_family", lambda r: r["meta"]["font_family"])]:
        out[name] = sorted((loc_cell(v, k) for k, v in by(pts, keyfn).items()),
                           key=lambda c: -(c["acc"] or 0))

    # The `reverse` arm quotes pixel coordinates in the ON-DISK frame, but the
    # model receives a downscaled image. At `large` the quoted point is usually
    # outside the frame the model actually got, so the question is unanswerable
    # as posed and its score is a dataset defect rather than a capability
    # result. Quantified here so the report can say so with a number.
    exs_all = {e.uid: e for e in load(DS)}
    rev_frame = {}
    for g in LOC_RUNGS:
        sub = [e for e in exs_all.values()
               if e.meta["qtype"] == "reverse" and e.meta["resolution"] == g]
        outside = 0
        for e in sub:
            pp = e.meta.get("probe_point_px")
            ew, eh = e.meta["effective_px"]
            if pp and (pp[0] > ew or pp[1] > eh):
                outside += 1
        rev_frame[g] = {"n": len(sub), "outside": outside,
                        "frac": outside / len(sub) if sub else None,
                        "rescale": (sub[0].meta["effective_px"][0] / sub[0].meta["img_size"][0])
                                   if sub else None}
    out["reverse_frame"] = rev_frame

    # H3: point vs relation. relation asks about position while requiring no
    # coordinates in either direction, so the gap bounds the expression cost.
    sp_by = by(spans, lambda r: r["meta"]["qtype"])
    out["text"] = {}
    for qt in ("relation", "reverse"):
        rows = sp_by.get(qt, [])
        out["text"][qt] = {
            "n": len(rows),
            "em": st.mean([r["em"] for r in rows]) if rows else None,
            "f1": st.mean([r["f1"] for r in rows]) if rows else None,
            "by_rung": {g: {"n": len(v),
                            "em": st.mean([r["em"] for r in v]) if v else None,
                            "f1": st.mean([r["f1"] for r in v]) if v else None}
                        for g, v in by(rows, lambda r: r["meta"]["resolution"]).items()},
        }
    return out


def cmd_localization(a: argparse.Namespace) -> int:
    src = RESULTS / f"{DS}__{a.tag}.jsonl"

    # BUG FIX. The original scored whatever it found and wrote the summary
    # unconditionally, exiting 0. A missing or mistyped tag therefore replaced a
    # good outputs/svgloc/summary.json with an n=0 one and reported success --
    # silent data loss, and the report downstream would render a page of zeros
    # without a single complaint. A zero-row scoring run is now an abort.
    if not src.exists():
        print(f"ABORT: no result file at {src} (tag {a.tag!r}). "
              f"Refusing to overwrite {a.out} with an n=0 summary.", file=sys.stderr)
        return 2

    s = analyse_localization(a.tag)
    c = s["counts"]
    if c["point_scored"] + c["span_scored"] == 0:
        print(f"ABORT: {src} has {c['lines']} line(s) but nothing scoreable "
              f"({c['unique']} unique uid(s), {c['unusable']} unusable, 0 point, 0 text). "
              f"Refusing to overwrite {a.out} with an n=0 summary.", file=sys.stderr)
        return 2

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, indent=1))
    h = s["headline"]
    print(f"{a.tag}: {c['point_scored']} point / {c['span_scored']} text "
          f"/ {c['unusable']} unusable")
    for cl in h:
        if cl["n"]:
            print(f"  {cl['label']:7s} n={cl['n']:5d}  click-in-bbox {cl['acc']*100:6.2f}% "
                  f"[{cl['lo']*100:.2f}-{cl['hi']*100:.2f}]  chance {cl['chance']*100:.4f}%  "
                  f"{cl['ratio']:.1f}x")
    print(f"  wrote {p}")
    return 0


# ===========================================================================
#  derived -- the svg_localization-derived sets: `counting` and `word_mc`
# ===========================================================================
#
# Both were run at `small` and `large` only, at the user's direction. That has
# one consequence worth stating before any number is read, and it is stated on
# the page rather than buried here: **the null control is gone**. Both EVAL.md
# files designate `medium` vs `large` as the noise floor, because those two
# rungs deliver at the same size and differ only in resampling path. Without
# `medium` there is no within-set noise floor, so the localization run's
# measured null (-0.13pp over the same 200 scenes and byte-identical pixels) is
# carried across as a proxy and labelled as one.
#
# Rules enforced here rather than left to the caller:
# * counting is scored by exact integer match, and the SIGNED error is reported
#   per bin -- absolute error alone destroys the mechanism (undercount vs
#   overcount);
# * counting families are never pooled into one accuracy number;
# * word_mc position bias is tested before any accuracy number is trusted;
# * a null or unparseable prediction is counted, never scored as wrong.

COUNT_BINS = [(1, 4, "1-4"), (5, 6, "5-6"), (7, 9, "7-9"), (10, 15, "10-15"), (16, 10**6, "16+")]
CHI2_CRIT_DF3 = 7.815          # 95%, 3 degrees of freedom


def load_derived_run(ds: str, tag: str, blind: bool = False) -> tuple[list[dict], list[dict]]:
    """Join a result file to its examples and score it. Returns (scored, unusable)."""
    exs = {e.uid: e for e in load(ds)}
    path = RESULTS / (f"{ds}__blind_{tag}.jsonl" if blind else f"{ds}__{tag}.jsonl")
    best: dict[str, dict] = {}
    for r in read_jsonl(path):
        uid = r["uid"][6:] if blind and r["uid"].startswith("blind:") else r["uid"]
        if r.get("pred") is None:
            best.setdefault(uid, r)
        else:
            best[uid] = r
    scored, unusable = [], []
    for uid, r in best.items():
        e = exs.get(uid)
        if e is None:
            continue
        if r.get("pred") is None:
            unusable.append({"uid": uid, "reason": r.get("parse_error") or r.get("error") or "null"})
            continue
        row = {"uid": uid, "pred": r["pred"], "meta": e.meta, "gold": e.gold,
               "question": e.question, "thinking": r.get("thinking") or ""}
        row.update(official_score(e, r["pred"]))
        row["hit"] = bool(row.get("score", 0) >= 0.5)
        scored.append(row)
    return scored, unusable


def derived_cell(rows, label, extra=None) -> dict:
    """Cell shape for the derived sets: no chance/ratio, because these rows
    carry no `target_area_frac` to compute an area-share chance from."""
    n = len(rows)
    k = sum(r["hit"] for r in rows)
    lo, hi = wilson(k, n)
    out = {"label": label, "n": n, "k": k, "acc": (k / n) if n else None,
           "lo": lo, "hi": hi, "suppressed": n < MIN_CELL}
    if extra:
        out.update(extra)
    return out


def in_derived_rungs(rows) -> list[dict]:
    """BUG FIX helper. `headline` is built from DERIVED_RUNGS but `overall` was
    built from *all* rows and labelled "both rungs" -- so a stray `medium` row
    made a three-rung pool wear a two-rung label, and `overall.n` no longer
    equalled the sum of the headline cells. The pool is now filtered to match
    the label; the count of rows this excludes is reported alongside."""
    return [r for r in rows if r["meta"]["resolution"] in DERIVED_RUNGS]


def count_bin(v: int) -> str:
    for lo, hi, name in COUNT_BINS:
        if lo <= v <= hi:
            return name
    return "?"


def signed_stats(rows) -> dict:
    se = [r["signed_error"] for r in rows if r.get("signed_error") is not None]
    ae = [r["abs_error"] for r in rows if r.get("abs_error") is not None]
    wrong = [r["signed_error"] for r in rows
             if r.get("signed_error") is not None and not r["hit"]]
    return {"mean_signed": st.mean(se) if se else None,
            "mean_signed_when_wrong": st.mean(wrong) if wrong else None,
            "median_abs": st.median(ae) if ae else None,
            "under": sum(1 for v in wrong if v < 0), "over": sum(1 for v in wrong if v > 0)}


def derived_paired(rows_by_rung: dict, a: str, b: str, keyfn) -> dict:
    ka = {keyfn(r): r for r in rows_by_rung.get(a, [])}
    kb = {keyfn(r): r for r in rows_by_rung.get(b, [])}
    both = sorted(set(ka) & set(kb), key=str)
    if not both:
        return {"a": a, "b": b, "n": 0}
    ha = sum(ka[k]["hit"] for k in both)
    hb = sum(kb[k]["hit"] for k in both)
    b_ = sum(1 for k in both if ka[k]["hit"] and not kb[k]["hit"])
    c_ = sum(1 for k in both if not ka[k]["hit"] and kb[k]["hit"])
    chi = ((abs(b_ - c_) - 1) ** 2 / (b_ + c_)) if (b_ + c_) else None
    return {"a": a, "b": b, "n": len(both), "acc_a": ha / len(both), "acc_b": hb / len(both),
            "delta_pp": (hb - ha) / len(both) * 100,
            "discordant_b": b_, "discordant_c": c_, "mcnemar_chi2": chi,
            "significant": bool(chi is not None and chi > 3.841)}


# ------------------------------------------------------------------- counting
def analyse_counting(tag: str) -> dict:
    rows, unusable = load_derived_run("svg_counting", tag)
    blind, blind_bad = load_derived_run("svg_counting", tag, blind=True)
    pooled = in_derived_rungs(rows)
    out = {"dataset": "svg_counting", "tag": tag,
           "counts": {"scored": len(rows), "unusable": len(unusable),
                      "blind_scored": len(blind), "blind_unusable": len(blind_bad),
                      "unusable_detail": unusable[:10],
                      "outside_reported_rungs": len(rows) - len(pooled)}}
    byr = by(rows, lambda r: r["meta"]["resolution"])
    out["headline"] = [derived_cell(byr.get(g, []), g, signed_stats(byr.get(g, []))) for g in DERIVED_RUNGS]
    out["overall"] = derived_cell(pooled, "both rungs", signed_stats(pooled))
    out["paired"] = derived_paired(byr, "small", "large",
                                   lambda r: (r["meta"]["graph_id"], r["question"]))

    # 3.3 dose-response, and the interaction: does the collapse point move with
    # resolution? Neither the ladder nor the counting curve shows that alone.
    out["dose"] = {}
    for g in DERIVED_RUNGS:
        cells = []
        for _lo, _hi, name in COUNT_BINS:
            sub = [r for r in byr.get(g, []) if count_bin(r["meta"]["true_count"]) == name]
            cells.append(derived_cell(sub, name, signed_stats(sub)))
        out["dose"][g] = cells
    out["dose_all"] = [derived_cell([r for r in rows if count_bin(r["meta"]["true_count"]) == name],
                                    name, signed_stats([r for r in rows
                                                        if count_bin(r["meta"]["true_count"]) == name]))
                       for _lo, _hi, name in COUNT_BINS]

    # The dose-response above is not interpretable on its own: the true count is
    # not randomly assigned across question forms, so a count bin is partly a
    # proxy for *what* is being counted. Measure that confound, then run the
    # clean version -- within a single form, where the thing counted is fixed.
    forms = by(rows, lambda r: r["meta"]["question_form"])
    bin_forms = {}
    for _lo, _hi, name in COUNT_BINS:
        sub = [r for r in rows if count_bin(r["meta"]["true_count"]) == name]
        cf = collections.Counter(r["meta"]["question_form"] for r in sub)
        bin_forms[name] = {"n": len(sub), "n_forms": len(cf),
                           "top": [(q, c) for q, c in cf.most_common(3)]}
    within = []
    for q, v in sorted(forms.items()):
        tc = [r["meta"]["true_count"] for r in v]
        if len(v) < 20 or (max(tc) - min(tc)) < 4:
            continue
        med = st.median(tc)
        lo = [r for r in v if r["meta"]["true_count"] <= med]
        hi = [r for r in v if r["meta"]["true_count"] > med]
        if len(lo) < 6 or len(hi) < 6:
            continue
        al = sum(r["hit"] for r in lo) / len(lo)
        ah = sum(r["hit"] for r in hi) / len(hi)
        within.append({"form": q, "min": min(tc), "max": max(tc),
                       "lo_n": len(lo), "lo_acc": al, "hi_n": len(hi), "hi_acc": ah,
                       "delta_pp": (ah - al) * 100})
    errs = [r for r in rows if not r["hit"]]
    out["dose_confound"] = {
        "bin_forms": bin_forms, "within_form": within,
        "n_forms_testable": len(within), "n_forms_total": len(forms),
        "n_errors": len(errs),
        "signed_histogram": sorted(collections.Counter(
            r["signed_error"] for r in errs if r.get("signed_error") is not None).items()),
    }

    # 3.2/5: families are reported separately and never pooled.
    out["family"] = {}
    for fam, v in by(rows, lambda r: r["meta"]["count_family"]).items():
        c = derived_cell(v, fam, signed_stats(v))
        c["by_rung"] = {g: derived_cell(vv, g, signed_stats(vv))
                        for g, vv in by(v, lambda r: r["meta"]["resolution"]).items()}
        out["family"][fam] = c

    for name, keyfn in [("chart_type", lambda r: r["meta"]["chart_type"]),
                        ("question_form", lambda r: r["meta"]["question_form"]),
                        ("theme", lambda r: r["meta"]["theme"]),
                        ("font_family", lambda r: r["meta"]["font_family"])]:
        out[name] = sorted((derived_cell(v, k, signed_stats(v)) for k, v in by(rows, keyfn).items()),
                           key=lambda c: -(c["acc"] or 0))

    bb = by(blind, lambda r: r["meta"]["resolution"])
    out["blind"] = {"overall": derived_cell(blind, "blind"),
                    "by_rung": {g: derived_cell(v, g) for g, v in bb.items()},
                    "by_family": {k: derived_cell(v, k) for k, v in
                                  by(blind, lambda r: r["meta"]["count_family"]).items()}}
    return out


# -------------------------------------------------------------------- word_mc
def chi2_against(observed: dict, expected_share: dict, n: int) -> dict:
    chi = 0.0
    rows = []
    for k in "ABCD":
        o = observed.get(k, 0)
        e = expected_share.get(k, 0.0) * n
        if e > 0:
            chi += (o - e) ** 2 / e
        rows.append({"option": k, "observed": o, "obs_share": o / n if n else None,
                     "expected_share": expected_share.get(k, 0.0),
                     "deviation_pp": ((o / n) - expected_share.get(k, 0.0)) * 100 if n else None})
    return {"chi2": chi, "crit": CHI2_CRIT_DF3, "biased": chi > CHI2_CRIT_DF3, "rows": rows, "n": n}


def analyse_word_mc(tag: str) -> dict:
    rows, unusable = load_derived_run("svg_word_mc", tag)
    blind, blind_bad = load_derived_run("svg_word_mc", tag, blind=True)
    pooled = in_derived_rungs(rows)
    out = {"dataset": "svg_word_mc", "tag": tag, "chance": 0.25,
           "counts": {"scored": len(rows), "unusable": len(unusable),
                      "blind_scored": len(blind), "blind_unusable": len(blind_bad),
                      "unusable_detail": unusable[:10],
                      "outside_reported_rungs": len(rows) - len(pooled)}}

    # 3.1: position bias first. If this fails, nothing below is trustworthy.
    key = collections.Counter(r["gold"][0] for r in rows)
    n = len(rows)
    share = {k: key.get(k, 0) / n for k in "ABCD"}
    picks = collections.Counter(r.get("picked") for r in rows)
    wrong_picks = collections.Counter(r.get("picked") for r in rows if not r["hit"])
    nw = sum(wrong_picks.values())
    out["position_bias"] = {
        "all_picks": chi2_against(picks, share, n),
        # among wrong answers the model has nothing to go on, so a slot
        # preference shows up here first -- the sharper of the two tests.
        "wrong_picks": chi2_against(wrong_picks, {k: 0.25 for k in "ABCD"}, nw),
        "key_share": share,
    }

    byr = by(rows, lambda r: r["meta"]["resolution"])
    out["headline"] = [derived_cell(byr.get(g, []), g) for g in DERIVED_RUNGS]
    out["overall"] = derived_cell(pooled, "both rungs")
    out["paired"] = derived_paired(byr, "small", "large",
                                   lambda r: (r["meta"]["graph_id"], r["meta"]["answer_text"]))

    # 3.2: a wrong answer is either a hallucinated reading (picked an absent
    # distractor) or a failure to spot the present word. Both look identical in
    # the accuracy number.
    miss = [r for r in rows if not r["hit"]]
    chosen = collections.Counter()
    for r in miss:
        opts = r["meta"].get("options") or []
        p = r.get("picked")
        if p and p in "ABCD" and len(opts) == 4:
            chosen[opts["ABCD".index(p)]] += 1
    out["wrong_analysis"] = {"n_wrong": len(miss),
                             "top_distractors": chosen.most_common(15),
                             "distinct_distractors": len(chosen)}

    for name, keyfn in [("chart_type", lambda r: r["meta"]["chart_type"]),
                        ("theme", lambda r: r["meta"]["theme"]),
                        ("font_family", lambda r: r["meta"]["font_family"]),
                        ("answer_len", lambda r: _lenbin(r["meta"].get("answer_len") or 0))]:
        out[name] = sorted((derived_cell(v, k) for k, v in by(rows, keyfn).items()),
                           key=lambda c: -(c["acc"] or 0))

    out["polarity"] = {k: derived_cell(v, k) for k, v in
                       by(rows, lambda r: "dark" if r["meta"]["theme"] in DARK_THEMES else "light").items()}

    bb = by(blind, lambda r: r["meta"]["resolution"])
    out["blind"] = {"overall": derived_cell(blind, "blind"),
                    "by_rung": {g: derived_cell(v, g) for g, v in bb.items()}}
    return out


def _lenbin(n: int) -> str:
    if n <= 4:
        return "<=4 chars"
    if n <= 6:
        return "5-6"
    if n <= 8:
        return "7-8"
    if n <= 10:
        return "9-10"
    return "11+"


def cmd_derived(a: argparse.Namespace) -> int:
    s = {"counting": analyse_counting(a.tag), "word_mc": analyse_word_mc(a.tag)}
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, indent=1))
    for k in ("counting", "word_mc"):
        d = s[k]
        print(f"{k}: {d['counts']['scored']} scored, {d['counts']['unusable']} unusable, "
              f"blind {d['counts']['blind_scored']}")
        skipped = d["counts"]["outside_reported_rungs"]
        if skipped:
            print(f"   note: {skipped} row(s) outside {DERIVED_RUNGS} excluded from "
                  f"'both rungs' (they are still in the per-cut breakdowns)")
        for c in d["headline"]:
            if c["n"]:
                print(f"   {c['label']:7s} n={c['n']:4d}  acc {c['acc']*100:6.2f}% "
                      f"[{c['lo']*100:.1f}-{c['hi']*100:.1f}]"
                      + (f"  mean signed {c['mean_signed']:+.2f}" if c.get("mean_signed") is not None else ""))
    print(f"  wrote {p}")
    return 0


# ===========================================================================
#  ablations -- prompt/answer-channel arms, scored against the baseline
# ===========================================================================
#
# Every arm is compared on the same uids as the baseline (the main native Haiku
# run), so each delta is a within-item contrast rather than two independent
# samples. Arms answer in different units, so the comparable quantity is stated
# per arm rather than pooled:
#
#     repeat, careful, describe, cell_then_point, landmark, crop
#         click-in-bbox, directly comparable to baseline
#     quadrant_mc
#         compared against the BASELINE CLICK BUCKETED TO 2x2, which is the same
#         granularity. Comparing it to exact click-in-bbox would be comparing a
#         4-way choice to a 0.25%-of-screen target and would be meaningless.
#     bbox
#         centre of the predicted box inside gold, so it stays in click-in-bbox
#         units rather than becoming an IoU number that cannot be mixed in.

ABL_TAG = "haiku-4-5_think2000"
POINT_ARMS = ("repeat", "careful", "describe", "cell_then_point", "landmark")


def load_arm(arm: str) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for r in read_jsonl(RESULTS / f"svgloc_abl_{arm}__{ABL_TAG}.jsonl"):
        if r.get("pred") is not None:
            best[r["uid"]] = r
    return best


def mcnemar(pairs) -> dict:
    b = sum(1 for a, c in pairs if a and not c)
    c_ = sum(1 for a, c in pairs if not a and c)
    chi = ((abs(b - c_) - 1) ** 2 / (b + c_)) if (b + c_) else None
    return {"discordant_base": b, "discordant_arm": c_, "chi2": chi,
            "significant": bool(chi is not None and chi > 3.841)}


def analyse_ablations(sample_uids: list[str]) -> dict:
    base_run = load_loc_run("haiku-4-5_think2000_native_r0")
    base = {r["uid"]: r for r in base_run["point"] if r["uid"] in set(sample_uids)}
    exs = {e.uid: e for e in load("svg_localization")}
    out = {"n_sample": len(sample_uids), "n_baseline": len(base), "arms": {}}

    b_hit = sum(r["hit"] for r in base.values())
    out["baseline"] = {"n": len(base), "acc": b_hit / max(len(base), 1),
                       "wilson": wilson(b_hit, len(base))}
    b2 = sum(1 for r in base.values()
             if cell_of(*r["pred"], 2) == centre_cell(r["gold"], 2))
    out["baseline_2x2"] = {"n": len(base), "acc": b2 / max(len(base), 1),
                           "wilson": wilson(b2, len(base))}

    for arm in POINT_ARMS + ("crop", "bbox", "quadrant_mc"):
        rows = load_arm(arm)
        if not rows:
            continue
        uids = [u for u in rows if u in base]
        rec = {"arm": arm, "n": len(uids)}

        if arm == "quadrant_mc":
            k = 0
            pairs = []
            for u in uids:
                e = exs[u]
                gold_q = (0 if (e.gold[1] + e.gold[3]) / 2 < 0.5 else 2) + \
                         (0 if (e.gold[0] + e.gold[2]) / 2 < 0.5 else 1)
                ok = str(rows[u]["pred"]).strip().upper() == "ABCD"[gold_q]
                k += ok
                bq = cell_of(*base[u]["pred"], 2) == centre_cell(base[u]["gold"], 2)
                pairs.append((bq, ok))
            rec.update({"metric": "quadrant letter", "acc": k / len(uids),
                        "wilson": wilson(k, len(uids)),
                        "compare_to": "baseline click bucketed to 2x2",
                        "baseline_acc": out["baseline_2x2"]["acc"],
                        "delta_pp": (k / len(uids) - out["baseline_2x2"]["acc"]) * 100,
                        **mcnemar(pairs)})
            out["arms"][arm] = rec
            continue

        hits, pairs, dbox, dcen = 0, [], [], []
        area_ratio = []
        for u in uids:
            r = rows[u]
            gold = exs[u].gold if arm != "crop" else None
            if arm == "crop":
                # gold was remapped into the crop frame at build time; recompute
                # it here the same way so scoring cannot silently drift.
                b = exs[u].gold
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                x0 = min(max(cx - 0.25, 0.0), 0.5)
                y0 = min(max(cy - 0.25, 0.0), 0.5)
                gold = [(b[0] - x0) / .5, (b[1] - y0) / .5, (b[2] - x0) / .5, (b[3] - y0) / .5]
            p = r["pred"]
            if arm == "bbox":
                x0, y0, x1, y1 = p
                pt = ((x0 + x1) / 2, (y0 + y1) / 2)
                ga = abs((gold[2] - gold[0]) * (gold[3] - gold[1]))
                pa = abs((x1 - x0) * (y1 - y0))
                if ga > 0:
                    area_ratio.append(pa / ga)
            else:
                pt = tuple(p)
            ok = bool(point_in_bbox(pt, gold))
            hits += ok
            pairs.append((base[u]["hit"], ok))
            dbox.append(d_box(pt, gold))
            dcen.append(d_centre(pt, gold))
        # named `arm_acc` rather than `acc` so it does not shadow the module-level
        # `acc()` helper that the localization section defines.
        arm_acc = hits / len(uids)
        bacc = sum(base[u]["hit"] for u in uids) / len(uids)
        rec.update({"metric": "click-in-bbox" if arm != "bbox" else "bbox centre in gold",
                    "acc": arm_acc, "wilson": wilson(hits, len(uids)),
                    "baseline_acc": bacc, "delta_pp": (arm_acc - bacc) * 100,
                    "median_d_box": st.median(dbox), "median_d_centre": st.median(dcen),
                    "baseline_median_d_box": st.median(d_box(tuple(base[u]["pred"]), exs[u].gold)
                                                       for u in uids),
                    **mcnemar(pairs)})
        if area_ratio:
            rec["median_area_ratio"] = st.median(area_ratio)
        out["arms"][arm] = rec

    # Repeat gets an extra read: the spread between two identical requests is
    # the in-set noise floor, and comparing it to the distance-to-gold separates
    # a noisy estimate from a stable but wrong one.
    rep = load_arm("repeat")
    uids = [u for u in rep if u in base]
    if uids:
        sep, err, agree = [], [], 0
        for u in uids:
            p1, p2 = tuple(base[u]["pred"]), tuple(rep[u]["pred"])
            sep.append(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))
            err.append(d_centre(p1, exs[u].gold))
            agree += (base[u]["hit"] == bool(point_in_bbox(p2, exs[u].gold)))
        out["repeat_consistency"] = {
            "n": len(uids),
            "median_separation": st.median(sep),
            "median_error": st.median(err),
            "ratio": st.median(sep) / st.median(err) if st.median(err) else None,
            "hit_agreement": agree / len(uids),
            "identical": sum(1 for s in sep if s < 1e-9) / len(uids),
        }
    return out


def cmd_ablations(a: argparse.Namespace) -> int:
    uids = json.loads(Path("results/svgloc_ablation_uids.json").read_text())
    s = analyse_ablations(uids)
    Path("outputs/svgloc").mkdir(parents=True, exist_ok=True)
    Path("outputs/svgloc/ablations.json").write_text(json.dumps(s, indent=1))
    b = s["baseline"]
    print(f"baseline on the shared sample: n={b['n']}  {b['acc']*100:.2f}% "
          f"[{b['wilson'][0]*100:.1f}-{b['wilson'][1]*100:.1f}]   "
          f"2x2 {s['baseline_2x2']['acc']*100:.1f}%")
    print(f"{'arm':16s} {'n':>4s} {'metric':>22s} {'acc':>7s} {'vs base':>9s} {'chi2':>6s} {'sig':>4s}")
    for arm, r in s["arms"].items():
        print(f"{arm:16s} {r['n']:4d} {r['metric']:>22s} {r['acc']*100:6.2f}% "
              f"{r['delta_pp']:+8.2f}pp {(r['chi2'] if r['chi2'] is not None else float('nan')):6.2f} "
              f"{'YES' if r['significant'] else '-':>4s}")
    rc = s.get("repeat_consistency")
    if rc:
        print(f"\nrepeat: median separation between two identical requests "
              f"{rc['median_separation']*100:.2f}% of frame vs median error "
              f"{rc['median_error']*100:.2f}%  (ratio {rc['ratio']:.2f}); "
              f"hit-agreement {rc['hit_agreement']*100:.1f}%")
    return 0


# ===========================================================================
#  annotate -- per-asset sidecar JSON, plus paginated galleries
# ===========================================================================
#
# Two rules shape this:
#
# * **Render from what the model saw, not from the source file.** ScreenSpot-Pro
#   ships 5120x2880 screenshots but Haiku receives ~1568px. Annotating the
#   original would display detail the model never had -- exactly the confusion
#   this project exists to avoid.
# * **Rendering is CPU-bound and embarrassingly parallel**, so it runs in a
#   process pool. The API stage is I/O-bound and threaded; mixing the two in one
#   pool would let PIL work starve the requests.

ANNOT = OUT / "annotations"
GALLERY = OUT / "gallery"
ASSETS = OUT / "assets"          # full-size annotated images, linked not inlined
FULL_MAX = 1600                  # long edge of the click-through image
GOOD, BAD = "#0ca30c", "#d03b3b"
THUMB_MAX = 560


# ---------------------------------------------------------------- rendering
def _thumb(im: Image.Image, max_w: int = THUMB_MAX, q: int = 72) -> str:
    im = im.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, max(1, round(im.height * max_w / im.width))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


_FONT_PATHS = ("/System/Library/Fonts/Supplemental/Arial.ttf",
               "/System/Library/Fonts/Helvetica.ttc")


def _font(size: int):
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap(draw, text: str, font, width: int) -> list[str]:
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if draw.textlength(t, font=font) <= width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_overlay(im: Image.Image, row: dict, labels: bool = True) -> Image.Image:
    """Gold target and prediction drawn on the image, for point-type answers.

    `labels=False` omits the "ground truth"/"prediction" captions. They are useful
    in the gallery but must be off when the image is sent to a judge: the caption
    is drawn adjacent to the box and can sit on top of the very UI element under
    adjudication, so the judge is asked whether a box contains an element that the
    annotation itself has covered up.
    """
    W, H = im.size
    if row.get("answer_type") != "point" or not row.get("gold"):
        return im
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(W, H) / 400))
    x0, y0, x1, y1 = [c * s for c, s in zip(row["gold"], (W, H, W, H))]
    d.rectangle([x0, y0, x1, y1], outline=GOOD, width=lw)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ring = max(lw * 14, (x1 - x0) * 1.6, (y1 - y0) * 1.6)
    d.ellipse([cx - ring, cy - ring, cx + ring, cy + ring], outline=GOOD, width=max(1, lw // 2))
    if labels:
        d.text((x0, max(0, y0 - lw * 9)), "ground truth", fill=GOOD, font=_font(lw * 8))
    if row.get("pred"):
        px, py = row["pred"][0] * W, row["pred"][1] * H
        r = lw * 8
        d.line([px - r, py, px + r, py], fill=BAD, width=lw)
        d.line([px, py - r, px, py + r], fill=BAD, width=lw)
        d.ellipse([px - r / 2, py - r / 2, px + r / 2, py + r / 2], outline=BAD, width=lw)
        if labels:
            d.text((px + r, py + r), "prediction", fill=BAD, font=_font(lw * 8))
    return im


def render_full(row: dict) -> str | None:
    """Write the click-through image: the asset with both annotations, plus a
    caption band carrying question / ground truth / prediction.

    Written to disk rather than inlined -- a page with 50 full-size data URIs
    would be tens of megabytes, and the whole point of the larger view is that
    it is only fetched when someone actually clicks.
    """
    src = row.get("_image")
    if not src:
        return None
    out_dir = ASSETS / row["dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{row['uid'].replace(':', '_').replace('/', '_')}.jpg"
    dest = out_dir / name

    im = Image.open(src).convert("RGB")
    if max(im.size) > FULL_MAX:
        sc = FULL_MAX / max(im.size)
        im = im.resize((max(1, round(im.width * sc)), max(1, round(im.height * sc))), Image.LANCZOS)
    im = draw_overlay(im, row)

    # Caption band: for span answers this is the only place the annotation can
    # live, since gold and prediction are text rather than a location.
    W = im.width
    f_lab, f_txt = _font(15), _font(17)
    scratch = ImageDraw.Draw(im)
    ok = (row.get("score") or 0) >= 0.5
    def fmt(v, is_box):
        if row.get("answer_type") != "point":
            return str(v)
        try:
            if is_box:
                x0, y0, x1, y1 = v
                return (f"box x {x0*100:.1f}–{x1*100:.1f}%, y {y0*100:.1f}–{y1*100:.1f}% "
                        f"(centre {(x0+x1)/2*100:.1f}%, {(y0+y1)/2*100:.1f}%)")
            return f"point {v[0]*100:.1f}%, {v[1]*100:.1f}%"
        except Exception:
            return str(v)

    blocks = [("question", str(row.get("question", ""))[:400]),
              ("ground truth", fmt(row.get("gold"), True)),
              ("prediction", fmt(row.get("pred"), False))]
    wrapped = [(lab, _wrap(scratch, txt, f_txt, W - 200)) for lab, txt in blocks]
    band_h = 22 + sum(24 + 22 * len(ls) for _, ls in wrapped)

    canvas = Image.new("RGB", (W, im.height + band_h), "#111111")
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    y = im.height + 14
    for lab, lines in wrapped:
        col = GOOD if lab == "ground truth" else (GOOD if (lab == "prediction" and ok) else
                                                  BAD if lab == "prediction" else "#898781")
        d.text((18, y), lab.upper(), fill="#898781", font=f_lab)
        for ln in lines:
            d.text((160, y - 2), ln, fill=col, font=f_txt)
            y += 22
        y += 8
    d.text((W - 120, im.height + 14), "CORRECT" if ok else "WRONG",
           fill=GOOD if ok else BAD, font=f_txt)

    canvas.save(dest, format="JPEG", quality=82)
    return str(Path("..") / "assets" / row["dataset"] / name)


def render_asset(row: dict) -> tuple[str, str | None]:
    """(main thumbnail, optional zoom) with the gold target and prediction drawn."""
    path = row["_image"]
    im = Image.open(path).convert("RGB")

    if row.get("answer_type") != "point":
        return _thumb(im), None

    W, H = im.size
    x0, y0, x1, y1 = [c * s for c, s in zip(row["gold"], (W, H, W, H))]
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(W, H) / 500))
    d.rectangle([x0, y0, x1, y1], outline=GOOD, width=lw)
    if row.get("pred"):
        px, py = row["pred"][0] * W, row["pred"][1] * H
        r = lw * 6
        d.line([px - r, py, px + r, py], fill=BAD, width=lw)
        d.line([px, py - r, px, py + r], fill=BAD, width=lw)
    pad = max(x1 - x0, y1 - y0) * 6 + 80
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    zoom = im.crop((round(max(0, cx - pad)), round(max(0, cy - pad)),
                    round(min(W, cx + pad)), round(min(H, cy + pad))))
    return _thumb(im), _thumb(zoom, max_w=420)


def build_one(row: dict) -> dict:
    """Module-level so the process pool can pickle it. Writes the sidecar JSON."""
    main_img, zoom = render_asset(row)
    full = render_full(row)
    rec = {k: row[k] for k in
           ("uid", "dataset", "answer_type", "question", "gold", "pred",
            "score", "metric", "grading_confidence", "primitive",
            "center_distance", "signed_error", "abs_error", "true_count",
            "polarity", "not_applicable", "judge_score", "string_score") if k in row}
    rec.update({
        "image": row["_image"],
        "sent_image_size": row.get("sent_image_sizes", [None])[0],
        "usage": row.get("usage"),
        "request_id": row.get("request_id"),
        "meta": row.get("meta", {}),
    })
    out = ANNOT / row["dataset"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{row['uid'].replace(':', '_').replace('/', '_')}.json").write_text(
        json.dumps(rec, indent=1, default=str))
    return {**rec, "_thumb": main_img, "_zoom": zoom, "_full": full}


# ------------------------------------------------------------------ gallery
CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;
 --ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--border:rgba(11,11,11,.10);
 --accent:#2a78d6;--good:#0ca30c;--bad:#d03b3b}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
 --surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--border:rgba(255,255,255,.10);--accent:#3987e5}}
:root[data-theme=dark]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;
 --ink2:#c3c2b7;--grid:#2c2c2a;--border:rgba(255,255,255,.10);--accent:#3987e5}
*{box-sizing:border-box}
html,body{background:var(--page);color:var(--ink);margin:0;
 font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 70px}
h1{font-size:23px;margin:0 0 4px}
.dek{color:var(--ink2);margin:0 0 18px}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:0 0 18px;
 padding:12px 14px;background:var(--surface);border:1px solid var(--border);border-radius:10px}
.bar label{font-size:12.5px;color:var(--ink2)}
select,button{font:inherit;font-size:13px;padding:5px 9px;border-radius:7px;
 border:1px solid var(--border);background:var(--page);color:var(--ink)}
.case{background:var(--surface);border:1px solid var(--border);border-radius:11px;
 padding:13px;margin-bottom:12px}
.hd{display:flex;gap:9px;align-items:flex-start;margin-bottom:10px}
.pill{font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;white-space:nowrap}
.ok{background:color-mix(in srgb,var(--good) 15%,transparent);color:var(--good)}
.no{background:color-mix(in srgb,var(--bad) 15%,transparent);color:var(--bad)}
.tag{font-size:11px;color:var(--muted);border:1px solid var(--border);
 padding:2px 7px;border-radius:999px}
.q{font-size:13.5px;line-height:1.45;flex:1}
.imgs{display:grid;grid-template-columns:1fr 300px;gap:10px;align-items:start}
.imgs.one{grid-template-columns:1fr}
@media(max-width:760px){.imgs{grid-template-columns:1fr}}
.imgs img{width:100%;max-height:250px;object-fit:contain;object-position:top left;
 background:var(--grid);border:1px solid var(--border);border-radius:7px;display:block}
dl{margin:11px 0 0;font-size:12.5px;display:grid;gap:4px}
dl>div{display:grid;grid-template-columns:150px 1fr;gap:10px}
dt{color:var(--muted)} dd{margin:0;overflow-wrap:anywhere}
dd.g{color:var(--good)} dd.b{color:var(--bad)}
nav{display:flex;gap:7px;flex-wrap:wrap;margin:20px 0 0}
nav a{font-size:13px;padding:5px 10px;border:1px solid var(--border);
 border-radius:7px;text-decoration:none;color:var(--ink2);background:var(--surface)}
nav a.cur{background:var(--accent);color:#fff;border-color:transparent}
.imgs a{display:block;position:relative}
.imgs a::after{content:"click to enlarge";position:absolute;right:6px;bottom:6px;
 font-size:10.5px;padding:2px 7px;border-radius:999px;background:rgba(0,0,0,.62);color:#fff;
 opacity:0;transition:opacity .12s}
.imgs a:hover::after,.imgs a:focus-visible::after{opacity:1}
.imgs a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
#lb{position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.92);display:none}
#lb.on{display:block}
#lb .stage{position:absolute;inset:0;overflow:hidden;cursor:grab;touch-action:none}
#lb .stage.grabbing{cursor:grabbing}
#lb img{position:absolute;top:0;left:0;transform-origin:0 0;will-change:transform;
 max-width:none;max-height:none;user-select:none;-webkit-user-drag:none}
#lb .ctrl{position:fixed;top:14px;left:50%;transform:translateX(-50%);display:flex;gap:6px;
 align-items:center;z-index:2;background:rgba(20,20,20,.85);padding:6px 8px;border-radius:10px}
#lb .ctrl button{font:inherit;font-size:14px;line-height:1;min-width:34px;padding:7px 9px;
 border-radius:7px;border:1px solid rgba(255,255,255,.18);background:#222;color:#eee;cursor:pointer}
#lb .ctrl button:hover{background:#333}
#lb .ctrl .lvl{color:#c3c2b7;font-size:12.5px;min-width:52px;text-align:center;
 font-variant-numeric:tabular-nums}
#lb .hint{position:fixed;bottom:14px;left:50%;transform:translateX(-50%);color:#898781;
 font-size:12px;z-index:2;background:rgba(20,20,20,.8);padding:5px 11px;border-radius:999px}
"""

LIGHTBOX_JS = """
(function(){
 const lb=document.getElementById('lb'); if(!lb) return;
 const stage=lb.querySelector('.stage'), img=lb.querySelector('img'),
       lvl=lb.querySelector('.lvl');
 let s=1, fit=1, tx=0, ty=0, drag=false, lx=0, ly=0;

 const apply=()=>{img.style.transform=`translate(${tx}px,${ty}px) scale(${s})`;
                  lvl.textContent=Math.round(s/fit*100)+'%';};
 // Fit on open, then let the user magnify well past 1:1 -- a 22px UI target is
 // the whole reason this exists, so the ceiling is deliberately high.
 const fitView=()=>{const r=stage.getBoundingClientRect();
   fit=Math.min(r.width/img.naturalWidth, r.height/img.naturalHeight);
   s=fit; tx=(r.width-img.naturalWidth*s)/2; ty=(r.height-img.naturalHeight*s)/2; apply();};
 const zoomAt=(px,py,f)=>{const ns=Math.min(fit*40, Math.max(fit*0.5, s*f));
   tx=px-(px-tx)*(ns/s); ty=py-(py-ty)*(ns/s); s=ns; apply();};
 const centreZoom=f=>{const r=stage.getBoundingClientRect(); zoomAt(r.width/2,r.height/2,f);};

 img.addEventListener('load', fitView);
 addEventListener('resize', ()=>{ if(lb.classList.contains('on')) fitView(); });

 stage.addEventListener('wheel', e=>{ e.preventDefault();
   const r=stage.getBoundingClientRect();
   zoomAt(e.clientX-r.left, e.clientY-r.top, e.deltaY<0?1.18:1/1.18); }, {passive:false});
 stage.addEventListener('dblclick', e=>{ const r=stage.getBoundingClientRect();
   if(s>fit*1.5) fitView(); else zoomAt(e.clientX-r.left, e.clientY-r.top, 5); });
 stage.addEventListener('mousedown', e=>{ drag=true; lx=e.clientX; ly=e.clientY;
   stage.classList.add('grabbing'); e.preventDefault(); });
 addEventListener('mousemove', e=>{ if(!drag) return;
   tx+=e.clientX-lx; ty+=e.clientY-ly; lx=e.clientX; ly=e.clientY; apply(); });
 addEventListener('mouseup', ()=>{ drag=false; stage.classList.remove('grabbing'); });

 lb.querySelector('.zin').onclick =()=>centreZoom(1.5);
 lb.querySelector('.zout').onclick=()=>centreZoom(1/1.5);
 lb.querySelector('.zfit').onclick=fitView;
 const close=()=>{ lb.classList.remove('on'); img.removeAttribute('src'); };
 lb.querySelector('.zclose').onclick=close;
 addEventListener('keydown', e=>{ if(!lb.classList.contains('on')) return;
   if(e.key==='Escape') close();
   if(e.key==='+'||e.key==='=') centreZoom(1.5);
   if(e.key==='-') centreZoom(1/1.5);
   if(e.key==='0') fitView(); });

 document.querySelectorAll('a.zoom').forEach(a=>a.addEventListener('click', e=>{
   e.preventDefault(); img.src=a.getAttribute('href'); lb.classList.add('on'); }));
})();
"""

LIGHTBOX_HTML = """
<div id="lb">
 <div class="ctrl">
  <button class="zout" title="zoom out (-)">&minus;</button>
  <span class="lvl">100%</span>
  <button class="zin" title="zoom in (+)">+</button>
  <button class="zfit" title="fit to screen (0)">fit</button>
  <button class="zclose" title="close (Esc)">&times;</button>
 </div>
 <div class="stage"><img alt="full size, annotated"></div>
 <span class="hint">scroll or double-click to zoom &middot; drag to pan &middot; Esc to close</span>
</div>
"""


JS = """
const sel=document.getElementById('f-res'),pr=document.getElementById('f-prim');
function apply(){const r=sel.value,p=pr?pr.value:'all';
 document.querySelectorAll('.case').forEach(c=>{
  const okr = r==='all'||c.dataset.res===r;
  const okp = p==='all'||c.dataset.prim===p;
  c.style.display=(okr&&okp)?'':'none';});}
sel.addEventListener('change',apply); if(pr)pr.addEventListener('change',apply);
document.getElementById('theme').addEventListener('click',()=>{
 const d=document.documentElement.dataset.theme==='dark';
 document.documentElement.dataset.theme=d?'light':'dark';});
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def _coord(v, is_box: bool) -> str:
    """Readable form for normalized boxes/points; passthrough for everything else."""
    try:
        if is_box:
            x0, y0, x1, y1 = v
            return (f"centre {(x0+x1)/2*100:.1f}%, {(y0+y1)/2*100:.1f}% "
                    f"(box {x0*100:.1f}–{x1*100:.1f}% x {y0*100:.1f}–{y1*100:.1f}%)")
        return f"{v[0]*100:.1f}%, {v[1]*100:.1f}%"
    except Exception:
        return str(v)


def case_html(a: dict) -> str:
    """One evaluated question. Images link to the full-size annotated copy."""
    ok = (a.get("score") or 0) >= 0.5
    is_point = a.get("answer_type") == "point"
    full = a.get("_full")

    def linked(img: str) -> str:
        if not full:
            return img
        return f'<a class="zoom" href="{full}" title="open full size with both annotations">{img}</a>'

    main_img = linked(f'<img src="{a["_thumb"]}" alt="evaluated asset">')
    zoom = ""
    if a.get("_zoom"):
        zoom = ('<figure>'
                + linked(f'<img src="{a["_zoom"]}" alt="zoom on target">')
                + '<figcaption style="font-size:11px;color:var(--muted)">zoom on target</figcaption>'
                + '</figure>')

    prim = a.get("primitive") or "-"
    extra = ""
    if a.get("metric") == "exact_count":
        extra = (f'<div><dt>signed error</dt><dd>{a.get("signed_error")}</dd></div>'
                 f'<div><dt>true count</dt><dd>{a.get("true_count")}</dd></div>')
    elif a.get("metric") == "click_in_bbox":
        cd = a.get("center_distance")
        extra = ('<div><dt>distance to target centre</dt><dd>'
                 + (f"{cd*100:.1f}% of screen" if cd is not None else "&mdash;")
                 + "</dd></div>")
    conf_note = (' <span class="tag">graded approximately</span>'
                 if a.get("grading_confidence") == "fuzzy" and a.get("judge_score") is None else "")
    if a.get("judge_score") is not None:
        agree = (a["judge_score"] >= .5) == ((a.get("string_score") or 0) >= .5)
        extra += ('<div><dt>CharXiv official judge</dt><dd class="{}">{}</dd></div>'
                  .format("g" if a["judge_score"] >= .5 else "b",
                          ("scored correct" if a["judge_score"] >= .5 else "scored wrong")
                          + ("" if agree else " &mdash; disagrees with string matching")))

    return f"""
<article class="case" data-res="{'ok' if ok else 'no'}" data-prim="{esc(prim)}">
 <div class="hd">
  <span class="pill {'ok' if ok else 'no'}">{'&#10003;' if ok else '&#10007;'}</span>
  <span class="tag">{esc(prim)}</span>
  <span class="q">{esc(a.get('question',''))[:400]}</span>
 </div>
 <div class="imgs{'' if zoom else ' one'}"><figure>{main_img}</figure>{zoom}</div>
 <dl>
  <div><dt>model answered</dt><dd class="{'g' if ok else 'b'}">{esc(_coord(a.get('pred'), False) if is_point else a.get('pred'))}</dd></div>
  <div><dt>gold</dt><dd class="g">{esc(_coord(a.get('gold'), True) if is_point else a.get('gold'))}</dd></div>
  <div><dt>metric</dt><dd>{esc(a.get('metric'))}{conf_note}</dd></div>
  {extra}
  <div><dt>uid</dt><dd style="color:var(--muted)">{esc(a.get('uid'))}</dd></div>
 </dl>
</article>"""


def write_pages(dataset: str, annots: list[dict], per_page: int) -> list[Path]:
    GALLERY.mkdir(parents=True, exist_ok=True)
    prims = sorted({a.get("primitive") or "-" for a in annots})
    pages = [annots[i:i + per_page] for i in range(0, len(annots), per_page)] or [[]]
    paths = []
    for i, chunk in enumerate(pages):
        nav = "".join(
            f'<a href="{dataset}_{j:03d}.html"{" class=cur" if j == i else ""}>{j+1}</a>'
            for j in range(len(pages)))
        opts = "".join(f'<option value="{esc(p)}">{esc(p)}</option>' for p in prims)
        n_ok = sum(1 for a in annots if (a.get("score") or 0) >= 0.5)
        p = GALLERY / f"{dataset}_{i:03d}.html"
        p.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(dataset)} &mdash; annotated assets {i+1}/{len(pages)}</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>{esc(dataset)} &mdash; annotated assets</h1>
<p class="dek">Every evaluated question with its image, gold answer and Haiku 4.5's answer.
{len(annots)} assets &middot; {n_ok} correct ({n_ok/max(len(annots),1)*100:.0f}%) &middot;
page {i+1} of {len(pages)}</p>
<div class="bar">
 <label for="f-res">result</label>
 <select id="f-res"><option value="all">all</option><option value="no">incorrect only</option>
 <option value="ok">correct only</option></select>
 <label for="f-prim">primitive</label>
 <select id="f-prim"><option value="all">all</option>{opts}</select>
 <button id="theme" type="button">toggle theme</button>
</div>
{''.join(case_html(a) for a in chunk)}
<nav>{nav}</nav>
</div>
{LIGHTBOX_HTML}
<script>{JS}
{LIGHTBOX_JS}</script></body></html>""", encoding="utf-8")
        paths.append(p)
    return paths


def rows_for(dataset: str, tag: str | None = None) -> list[dict]:
    """Rows for the gallery, sourced from the aggregate layer.

    This used to re-implement loading and scoring locally, which meant it
    silently missed everything the aggregate layer adds -- including CharXiv's
    official judge verdict, so the galleries were showing string-match results
    while the report showed judged ones.
    """
    rows = []
    for r in load_rows(dataset):
        ex = r["_ex"]
        row = {k: v for k, v in r.items() if k != "_ex"}
        row["_image"] = ex.images[0]
        row["question"] = ex.question
        row["primitive"] = (LABELS.get(r.get("primitive"))
                            or ex.meta.get("qlabel")
                            or (ex.meta.get("operation") or [None])[0]
                            or ex.meta.get("ui_type") or ex.meta.get("split") or "-")
        row.setdefault("answer_type", ex.answer_type)
        row["gold"] = ex.gold
        rows.append(row)
    return rows


def cmd_annotate(a: argparse.Namespace) -> int:
    ANNOT.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets:
        rows = rows_for(ds, a.tag)
        if not rows:
            print(f"{ds}: no results for tag {a.tag}, skipping")
            continue
        if a.limit:
            rows = rows[: a.limit]
        # Incorrect first: the failures are what the report is about.
        rows.sort(key=lambda r: (r.get("score") or 0, r["uid"]))
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            annots = list(pool.map(build_one, rows, chunksize=8))
        pages = write_pages(ds, annots, a.per_page)
        print(f"{ds}: {len(annots)} annotated -> {ANNOT/ds}/  +  {len(pages)} gallery page(s)")
    return 0


# ===========================================================================
#  tiling -- one-off: do native-resolution patches recover grounding accuracy?
# ===========================================================================
#
# The premise is mechanical, not speculative. Haiku 4.5 caps a single image at a
# ~1568px long edge, so a 3840x2160 screenshot loses 93% of its pixels on the
# way in. But the cap is *per image*, and a 3x3 tile of that same screenshot is
# ~1408x792 -- already under the cap, so each patch arrives at native
# resolution. Tiling therefore trades one downscaled view for N full-detail
# views, at N times the image tokens.
#
# The model is shown a labelled full view first (for global context), then each
# patch in reading order, and answers with a patch address plus a position
# inside that patch. Coordinates are mapped back to whole-image space here.
#
# Patches overlap so that a target straddling a seam is still wholly inside at
# least one patch.

# Keep total images per request under 20: above that the API caps each image at
# 2000x2000, and the whole point here is to stay under the per-image ceiling.
GRID = (3, 3)
OVERLAP = 0.12
FULL_VIEW_MAX_EDGE = 1024  # context only; the patches carry the detail

SCHEMA = {
    "type": "object",
    "properties": {
        "patch": {"type": "integer", "description": "index of the patch containing the element"},
        "x": {"type": "integer", "description": "0-1000 horizontal position within that patch"},
        "y": {"type": "integer", "description": "0-1000 vertical position within that patch"},
    },
    "required": ["patch", "x", "y"],
    "additionalProperties": False,
}

INSTRUCTION = (
    "You are locating a UI element in a screenshot.\n"
    "You are given a low-detail view of the whole screen, then {n} overlapping "
    "patches of that same screen at full resolution, numbered 0 to {last} in "
    "reading order (left to right, top to bottom).\n"
    "Find the element, decide which patch shows it most completely, and give its "
    "centre as x and y in a 0-1000 coordinate system *within that patch* "
    "(0 = left/top edge of the patch, 1000 = right/bottom edge of the patch).\n"
    "Always answer, even if uncertain.\n\n"
    "Element: "
)


def _b64(im: Image.Image, max_edge: int | None = None) -> str:
    im = im.convert("RGB")
    if max_edge and max(im.size) > max_edge:
        s = max_edge / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def make_patches(im: Image.Image, grid=GRID, overlap=OVERLAP) -> list[tuple[Image.Image, tuple[float, float, float, float]]]:
    """Split into an overlapping grid. Returns (patch, normalized box) per tile."""
    rows, cols = grid
    W, H = im.size
    pw, ph = W / cols, H / rows
    ox, oy = pw * overlap, ph * overlap
    out = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, c * pw - ox)
            y0 = max(0, r * ph - oy)
            x1 = min(W, (c + 1) * pw + ox)
            y1 = min(H, (r + 1) * ph + oy)
            out.append((im.crop((round(x0), round(y0), round(x1), round(y1))),
                        (x0 / W, y0 / H, x1 / W, y1 / H)))
    return out


def build_content(im: Image.Image, instruction: str) -> tuple[list[dict], list[tuple], list[tuple[int, int]]]:
    patches = make_patches(im)
    content: list[dict] = [
        {"type": "text", "text": INSTRUCTION.format(n=len(patches), last=len(patches) - 1) + instruction},
        {"type": "text", "text": "Low-detail view of the whole screen:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                     "data": _b64(im, FULL_VIEW_MAX_EDGE)}},
    ]
    boxes, sizes = [], []
    for i, (patch, box) in enumerate(patches):
        content.append({"type": "text", "text": f"Patch {i} (full resolution):"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": _b64(patch)}})
        boxes.append(box)
        sizes.append(patch.size)
    return content, boxes, sizes


def to_global(patch_idx: int, x: int, y: int, boxes: list[tuple]) -> tuple[float, float]:
    """Map a within-patch 0-1000 point back to whole-image normalized coordinates."""
    idx = max(0, min(patch_idx, len(boxes) - 1))
    x0, y0, x1, y1 = boxes[idx]
    return x0 + (x / 1000.0) * (x1 - x0), y0 + (y / 1000.0) * (y1 - y0)


def run_one(client, ex, budget: Budget, thinking_budget: int) -> dict:
    im = Image.open(ex.images[0])
    content, boxes, sizes = build_content(im, ex.question)
    rec = {"uid": ex.uid, "dataset": ex.dataset, "gold": ex.gold, "meta": ex.meta,
           "mode": "tiled", "grid": list(GRID), "patch_sizes": sizes,
           "source_size": list(im.size), "thinking_budget": thinking_budget}
    try:
        t0 = time.monotonic()
        resp = client.messages.create(
            model=MODEL,
            max_tokens=thinking_budget + 2048,
            thinking={"type": "enabled", "budget_tokens": thinking_budget},
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": content}],
        )
        budget.add(resp.usage.input_tokens, resp.usage.output_tokens)
        text = next((b.text for b in resp.content if b.type == "text"), None)
        rec.update({"raw": text, "stop_reason": resp.stop_reason,
                    "latency_s": round(time.monotonic() - t0, 2),
                    "usage": {"input_tokens": resp.usage.input_tokens,
                              "output_tokens": resp.usage.output_tokens}})
        obj = json.loads(text)
        rec["patch"] = obj["patch"]
        rec["pred"] = list(to_global(obj["patch"], obj["x"], obj["y"], boxes))
        rec["score"] = point_in_bbox(tuple(rec["pred"]), ex.gold)
    except Exception as e:
        rec.update({"pred": None, "score": None, "error": f"{type(e).__name__}: {e}"})
    return rec


def cmd_tiling(a: argparse.Namespace) -> int:
    # Same seed as the baseline run, so the comparison is on the same questions.
    examples = load(a.dataset)
    examples = random.Random(a.seed).sample(examples, min(200, len(examples)))[: a.limit]

    out = RUNNER_RESULTS / f"{a.dataset}__tiled{GRID[0]}x{GRID[1]}_r0.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            try:
                r = json.loads(line)
                if r.get("pred") is not None:
                    done.add(r["uid"])
            except Exception:
                pass
    todo = [e for e in examples if e.uid not in done]
    print(f"{a.dataset}: {len(examples)} selected, {len(done)} done, {len(todo)} to run -> {out}")

    # Imported here, not at module scope. `tiling` is the only subcommand that
    # calls the model; the other five are offline analyses, and a module-level
    # import would make every one of them pay for an API client they never use.
    # It also makes the step-declaration test honest: a module that imports
    # anthropic is assumed to spend money, so only the arm that does should.
    import anthropic

    client, budget, lock = anthropic.Anthropic(), Budget(a.max_spend), threading.Lock()
    n = 0
    with open(out, "a") as fh, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futs = [pool.submit(run_one, client, e, budget, a.thinking_budget) for e in todo]
        for f in as_completed(futs):
            rec = f.result()
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n += 1
                if n % 10 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} | ${budget.spent:.3f}", flush=True)
            if budget.exhausted():
                print("  !! spend cap reached")
                break
    print(f"done | {budget.calls} calls | ${budget.spent:.3f}")
    return 0


# ===========================================================================
#  CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m blindspot.eval",
        description="Analysis layer: results/*.jsonl -> JSON. One subcommand per artifact.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="SUBCOMMAND")

    p = sub.add_parser("aggregate", help="results -> outputs/summary.json")
    p.set_defaults(fn=cmd_aggregate)

    p = sub.add_parser("localization",
                       help="svg_localization -> outputs/svgloc/summary.json")
    p.add_argument("--tag", default="haiku-4-5_think2000_native_r0")
    p.add_argument("--out", default="outputs/svgloc/summary.json")
    p.set_defaults(fn=cmd_localization)

    p = sub.add_parser("derived",
                       help="svg_counting + svg_word_mc -> outputs/svgderived/summary.json")
    p.add_argument("--tag", default="haiku-4-5_think2000_native_r0")
    p.add_argument("--out", default="outputs/svgderived/summary.json")
    p.set_defaults(fn=cmd_derived)

    p = sub.add_parser("ablations",
                       help="localization ablation arms -> outputs/svgloc/ablations.json")
    p.set_defaults(fn=cmd_ablations)

    p = sub.add_parser("annotate",
                       help="per-asset sidecar JSON + browsable galleries")
    p.add_argument("--datasets", nargs="+",
                   default=["charxiv", "infographicvqa", "screenspot_pro"])
    p.add_argument("--tag", default="haiku-4-5_think2000_native_r0")
    p.add_argument("--per-page", type=int, default=50)
    p.add_argument("--limit", type=int, default=None, help="cap assets per dataset")
    p.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8) - 6))
    p.set_defaults(fn=cmd_annotate)

    p = sub.add_parser("tiling", help="tiled-patch grounding experiment (CALLS THE API)")
    p.add_argument("--dataset", default="screenspot_pro")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thinking-budget", type=int, default=2000)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--max-spend", type=float, default=4.0)
    p.set_defaults(fn=cmd_tiling)

    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
