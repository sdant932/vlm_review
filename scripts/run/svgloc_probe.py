"""Harness sanity probe for data/svg_localization (EVAL.md 3.9).

A near-zero localization score has two explanations: the model cannot do it, or
the harness is broken. On a dataset that has never been run against any model
that ambiguity is sharper than it was on ScreenSpot-Pro, not weaker -- there is
no prior run to sanity-check against.

So: run a stronger model (Sonnet) on byte-identical inputs before reporting
anything. If Sonnet lands in the boxes, the pipeline is sound and a low Haiku
score is a capability result. If both models score near zero, suspect the
dataset or the harness and check verify/index.html before writing a conclusion.

This does NOT reuse scripts/run/coord_probe.py: that one stratifies by which items
Haiku already hit or missed, which presupposes the Haiku run this probe is meant
to precede. Sampling here is from the manifest alone.

Three arms on the SAME uids:
    haiku   native      -- the target of the study
    sonnet  native       -- is the pipeline sound?
    sonnet  edge1568     -- ...or was that just Sonnet's bigger image budget?

Sonnet 5's image ceiling (~2576px) is higher than Haiku's (~1568px), and `large`
is 3000x1900, so the two models would not otherwise receive the same thing.

    python scripts/run/svgloc_probe.py --per-cell 10 --max-spend 4
"""
from __future__ import annotations
import argparse, json, random, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from blindspot.core.adapters import load
from blindspot.core.runner import Budget, run_one, short_name
from blindspot.core.scoring import point_in_bbox

RESULTS = Path("results")
DS = "svg_localization"


def sample_rung(rung: str, n: int, seed: int) -> list:
    """n point questions from one rung, stratified across area tertiles.

    Same deterministic scheme as sample(), restricted to a single resolution so
    a rung can be measured on its own without paying for the other two.
    """
    pts = [e for e in load(DS)
           if e.answer_type == "point" and e.meta["resolution"] == rung]
    rows = sorted(pts, key=lambda e: (e.meta["target_area_frac"], e.uid))
    total, out = len(rows), []
    for t in range(3):
        band = rows[t * total // 3:(t + 1) * total // 3]
        want = n // 3 + (1 if t < n % 3 else 0)
        rng = random.Random(f"{seed}:{rung}:{t}")
        out += rng.sample(band, min(want, len(band)))
    return out


def sample(per_cell: int, seed: int) -> list:
    """Deterministic, stratified by (resolution x target-area tertile).

    Area tertiles are cut *within* a resolution: target_area_frac is a fraction
    of the image, so the same target occupies the same fraction at every rung.
    Cutting globally would therefore just reproduce the resolution split.
    """
    pts = [e for e in load(DS) if e.answer_type == "point"]
    by_res = defaultdict(list)
    for e in pts:
        by_res[e.meta["resolution"]].append(e)
    out = []
    for res in sorted(by_res):
        rows = sorted(by_res[res], key=lambda e: (e.meta["target_area_frac"], e.uid))
        n = len(rows)
        for t in range(3):
            band = rows[t * n // 3:(t + 1) * n // 3]
            rng = random.Random(f"{seed}:{res}:{t}")
            out += rng.sample(band, min(per_cell, len(band)))
    return out


def arm(client, examples, model, max_edge, budget, thinking, concurrency,
        suffix: str = "") -> Path:
    cond = "native" if max_edge is None else f"edge{max_edge}"
    out = RESULTS / f"{DS}__probe{suffix}_{short_name(model)}_think{thinking}_{cond}_r0.jsonl"
    with open(out, "w") as fh, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(run_one, client, e, budget, thinking, max_edge, model)
                for e in examples]
        for f in futs:
            fh.write(json.dumps(f.result()) + "\n")
    return out


def summarise(path: Path, examples: list) -> dict:
    golds = {e.uid: e.gold for e in examples}
    meta = {e.uid: e.meta for e in examples}
    rows = [json.loads(l) for l in open(path) if l.strip()]
    usable = [r for r in rows if r.get("pred") is not None]
    per_res: Counter = Counter()
    per_res_n: Counter = Counter()
    for r in usable:
        ok = point_in_bbox(tuple(r["pred"]), golds[r["uid"]])
        res = meta[r["uid"]]["resolution"]
        per_res[res] += ok
        per_res_n[res] += 1
    hits = sum(point_in_bbox(tuple(r["pred"]), golds[r["uid"]]) for r in usable)
    return {"file": path.name, "rows": len(rows), "usable": len(usable),
            "unusable": len(rows) - len(usable),
            "hits": int(hits), "acc": hits / max(len(usable), 1),
            "by_res": {k: (per_res[k] / per_res_n[k], per_res_n[k]) for k in sorted(per_res_n)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--rung", choices=["small", "medium", "large"], default=None,
                    help="restrict to one resolution rung (use with --n)")
    ap.add_argument("--n", type=int, default=100, help="sample size when --rung is given")
    ap.add_argument("--arms", default="strong-native",
                    help="comma-separated subset of: target-native,strong-native,strong-edge1568")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--strong-model", default="claude-sonnet-5")
    ap.add_argument("--target-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--thinking-budget", type=int, default=2000)
    ap.add_argument("--max-spend", type=float, default=4.0)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()

    exs = sample_rung(a.rung, a.n, a.seed) if a.rung else sample(a.per_cell, a.seed)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{DS}__probe{'_'+a.rung if a.rung else ''}_uids.json").write_text(
        json.dumps([e.uid for e in exs], indent=1))
    print(f"probe sample: {len(exs)} point questions "
          f"({Counter(e.meta['resolution'] for e in exs)})")

    client = anthropic.Anthropic(max_retries=0, timeout=120.0)
    budget = Budget(a.max_spend)
    all_arms = {"target-native": (a.target_model, None),
                "strong-native": (a.strong_model, None),
                "strong-edge1568": (a.strong_model, 1568)}
    plan = [all_arms[k.strip()] for k in a.arms.split(",") if k.strip() in all_arms]
    out = []
    for model, max_edge in plan:
        p = arm(client, exs, model, max_edge, budget, a.thinking_budget, a.concurrency,
                suffix=f"_{a.rung}" if a.rung else "")
        s = summarise(p, exs)
        s["model"], s["max_edge"] = model, max_edge
        out.append(s)
        by = "  ".join(f"{k} {v[0]*100:.0f}% (n={v[1]})" for k, v in s["by_res"].items())
        print(f"  {model:32s} {'native' if max_edge is None else f'edge{max_edge}':9s} "
              f"click-in-bbox {s['acc']*100:5.1f}%  ({s['hits']}/{s['usable']}, "
              f"{s['unusable']} unusable)   {by}   ${budget.spent:.2f}")
    (RESULTS / f"{DS}__probe{'_'+a.rung if a.rung else ''}_summary.json").write_text(
        json.dumps(out, indent=1))
    print(f"total ${budget.spent:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
