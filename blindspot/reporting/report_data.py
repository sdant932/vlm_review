"""Assemble every number the final report quotes into one auditable artifact.

Written so that no figure and no sentence in `blindspots.md` carries a number that
cannot be traced back to `results/*.jsonl`. Two of these series are recomputed here
rather than read from the existing summaries, because the published cuts are wrong
for what the report needs:

* **AI2D binding** is published split by *answer format* -- `adapters.py` bins a
  question as `label_reference` when all four options are <=2 characters. That puts
  "What is the structure labeled C?" in the reasoning bucket purely because its
  options are wordy, and those items score 57.2%, not 89.4%. The report needs the
  split by *operation*: does the question refer to a mark printed on the artwork.
* **Absence detection** has never been cut against the blind arm. Doing so is what
  demotes it from a perception blind spot to a calibration one.

    python -m blindspot.reporting.report_data
"""

from __future__ import annotations

import collections
import json
import re
import statistics as st
from pathlib import Path

from blindspot.core.adapters import load
from blindspot.core.stats import wilson, is_na, cell_of, centre_cell
from blindspot.analysis.svgloc_eval import load_run as load_svgloc, d_box, d_centre, band

OUT = Path("outputs/report")
RESULTS = Path("results")

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


def _preds(path: Path) -> dict[str, object]:
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
    sighted = _preds(RESULTS / "ai2d__haiku-4-5_think2000_native_r0.jsonl")
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
    sighted = _preds(RESULTS / "charxiv__haiku-4-5_think2000_native_r0.jsonl")
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


def build() -> dict:
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = build()
    (OUT / "figures.json").write_text(json.dumps(data, indent=1, default=str))

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
    print(f"\nwrote {OUT/'figures.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
