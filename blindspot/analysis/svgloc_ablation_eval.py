"""Score the localization ablations against the baseline run, paired per item.

Every arm is compared on the same uids as the baseline (the main native Haiku
run), so each delta is a within-item contrast rather than two independent
samples. Arms answer in different units, so the comparable quantity is stated
per arm rather than pooled:

    repeat, careful, describe, cell_then_point, landmark, crop
        click-in-bbox, directly comparable to baseline
    quadrant_mc
        compared against the BASELINE CLICK BUCKETED TO 2x2, which is the same
        granularity. Comparing it to exact click-in-bbox would be comparing a
        4-way choice to a 0.25%-of-screen target and would be meaningless.
    bbox
        centre of the predicted box inside gold, so it stays in click-in-bbox
        units rather than becoming an IoU number that cannot be mixed in.
"""

from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

from blindspot.core.adapters import load
from blindspot.core.scoring import point_in_bbox
from blindspot.core.stats import wilson, cell_of, centre_cell
from blindspot.analysis.svgloc_eval import load_run, d_box, d_centre

RESULTS = Path("results")
TAG = "haiku-4-5_think2000"
POINT_ARMS = ("repeat", "careful", "describe", "cell_then_point", "landmark")


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


def load_arm(arm: str) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for r in _rows(RESULTS / f"svgloc_abl_{arm}__{TAG}.jsonl"):
        if r.get("pred") is not None:
            best[r["uid"]] = r
    return best


def mcnemar(pairs) -> dict:
    b = sum(1 for a, c in pairs if a and not c)
    c_ = sum(1 for a, c in pairs if not a and c)
    chi = ((abs(b - c_) - 1) ** 2 / (b + c_)) if (b + c_) else None
    return {"discordant_base": b, "discordant_arm": c_, "chi2": chi,
            "significant": bool(chi is not None and chi > 3.841)}


def analyse(sample_uids: list[str]) -> dict:
    base_run = load_run("haiku-4-5_think2000_native_r0")
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
        acc = hits / len(uids)
        bacc = sum(base[u]["hit"] for u in uids) / len(uids)
        rec.update({"metric": "click-in-bbox" if arm != "bbox" else "bbox centre in gold",
                    "acc": acc, "wilson": wilson(hits, len(uids)),
                    "baseline_acc": bacc, "delta_pp": (acc - bacc) * 100,
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


def main() -> int:
    uids = json.loads(Path("results/svgloc_ablation_uids.json").read_text())
    s = analyse(uids)
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


if __name__ == "__main__":
    raise SystemExit(main())
