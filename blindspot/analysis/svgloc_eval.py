"""Analysis for data/svg_localization, following its EVAL.md.

Computation only -- rendering lives in svgloc_report.py, so every number on the
page can be regenerated and diffed without touching HTML.

Design rules taken straight from EVAL.md and enforced here rather than left to
the caller:

* score `point` against `gold_bbox_norm` (the widget hit box), never
  `text_ink_bbox_norm` (the glyph outline, ~3x smaller and a different task);
* chance is the mean hit-box area fraction, not 1/n of anything, because a
  uniform random click lands in a box with probability equal to its area share;
* never score a null or unparseable prediction as wrong -- count it separately;
* never average across metrics, and never average across resolutions;
* pair on (graph_id, qtype, target_text, anchor_text), never on the uid index,
  because the eligible-target pool differs per rung and question `:02` is not
  the same target at every resolution.
"""

from __future__ import annotations

import collections
import json
import math
import statistics as st
from pathlib import Path

from blindspot.core.adapters import load
from blindspot.core.scoring import point_in_bbox, token_f1
from blindspot.core.stats import wilson, cell_of, centre_cell, bbox_cells, quantiles

DS = "svg_localization"
RESULTS = Path("results")
RUNGS = ("small", "medium", "large")
GRIDS = (2, 3, 4, 8, 16)
MIN_CELL = 30          # EVAL.md 5: suppress cells under n=30 rather than show noise


# --------------------------------------------------------------------- load
def read_jsonl(path: Path) -> list[dict]:
    rows = []
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


def load_run(tag: str) -> dict:
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


def cell(rows, label, extra=None) -> dict:
    """One reportable cell: n, accuracy, Wilson interval, chance, ratio."""
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


def by(rows, keyfn) -> dict:
    g = collections.defaultdict(list)
    for r in rows:
        g[keyfn(r)].append(r)
    return dict(g)


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


def paired(rows_by_rung: dict, a: str, b: str) -> dict:
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
    keysets = {g: {key(r) for r in rows_by_rung.get(g, [])} for g in RUNGS}
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


def analyse(tag: str) -> dict:
    run = load_run(tag)
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
    out["headline"] = [cell(by_rung.get(g, []), g) for g in RUNGS]
    out["overall"] = cell(pts, "all rungs pooled")
    out["pairing"] = triples(by_rung)
    out["null_control"] = paired(by_rung, "medium", "large")   # expect ~0
    out["resolution_effect"] = paired(by_rung, "small", "medium")
    out["curve"] = {g: precision_curve(by_rung.get(g, [])) for g in RUNGS}
    out["curve_all"] = precision_curve(pts)
    out["distance"] = {g: dist_summary(by_rung.get(g, [])) for g in RUNGS}
    out["distance_all"] = dist_summary(pts)

    # H2 gradient: area quintiles, computed within rung so the quintile is not
    # just a proxy for which rung the row came from.
    out["area_quintiles"] = {}
    for g in RUNGS:
        qs = quantiles(by_rung.get(g, []), lambda r: r["meta"]["target_area_frac"], 5)
        out["area_quintiles"][g] = [
            cell(ch, f"{lo*100:.4f}-{hi*100:.4f}%",
                 {"lo_frac": lo, "hi_frac": hi,
                  "cell4": sum(1 for r in ch if cell_of(*r["pred"], 4) == centre_cell(r["gold"], 4)) / len(ch)})
            for lo, hi, ch in qs]

    # EVAL.md 5 predicted theme would show nothing. It shows a great deal, so it
    # gets its own cut, with the two obvious confounds measured beside it: if
    # dark themes simply had bigger or higher-contrast targets, that would
    # explain the gap without any styling sensitivity.
    dark = {"slate-dark", "carbon", "blueprint"}
    grp = by(pts, lambda r: "dark" if r["meta"]["theme"] in dark else "light")
    out["polarity"] = {}
    for k, v in grp.items():
        c = cell(v, k)
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
        out[name] = sorted((cell(v, k) for k, v in by(pts, keyfn).items()),
                           key=lambda c: -(c["acc"] or 0))

    # The `reverse` arm quotes pixel coordinates in the ON-DISK frame, but the
    # model receives a downscaled image. At `large` the quoted point is usually
    # outside the frame the model actually got, so the question is unanswerable
    # as posed and its score is a dataset defect rather than a capability
    # result. Quantified here so the report can say so with a number.
    exs_all = {e.uid: e for e in load(DS)}
    rev_frame = {}
    for g in RUNGS:
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


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="haiku-4-5_think2000_native_r0")
    ap.add_argument("--out", default="outputs/svgloc/summary.json")
    a = ap.parse_args()
    s = analyse(a.tag)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, indent=1))
    h = s["headline"]
    print(f"{a.tag}: {s['counts']['point_scored']} point / {s['counts']['span_scored']} text "
          f"/ {s['counts']['unusable']} unusable")
    for c in h:
        if c["n"]:
            print(f"  {c['label']:7s} n={c['n']:5d}  click-in-bbox {c['acc']*100:6.2f}% "
                  f"[{c['lo']*100:.2f}-{c['hi']*100:.2f}]  chance {c['chance']*100:.4f}%  "
                  f"{c['ratio']:.1f}x")
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
