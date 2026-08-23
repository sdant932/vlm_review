"""Sonnet control probe: does a stronger model land in the gold boxes?

Purpose is a harness sanity check, not a model comparison. If Sonnet hits boxes
that Haiku misses on byte-identical inputs, the coordinate pipeline is sound and
Haiku's low score is a capability result rather than a bug.

Two conditions, because Sonnet 5's image ceiling (~2576px) is higher than Haiku's
(~1568px): a native win could be resolution rather than model. Running Sonnet
handicapped to Haiku's pixel budget separates those.

    python scripts/run/coord_probe.py --max-spend 1
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import anthropic
from blindspot.core.adapters import load
from blindspot.core.runner import Budget, run_one, short_name
from blindspot.core.scoring import point_in_bbox

HAIKU_TAG = "haiku-4-5_think2000_native_r0"
RESULTS = Path("results")


def haiku_rows(ds: str) -> dict[str, dict]:
    p = RESULTS / f"{ds}__{HAIKU_TAG}.jsonl"
    out = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            out[r["uid"]] = r
    return out


def pick(ds: str, n_hit: int, n_miss: int) -> list[str]:
    """Stratified, deterministic: some Haiku hits, some misses, smallest targets first."""
    exs = {e.uid: e for e in load(ds)}
    hits, misses = [], []
    for uid, r in sorted(haiku_rows(ds).items()):
        e = exs.get(uid)
        if e is None or not r.get("pred"):
            continue
        (hits if point_in_bbox(tuple(r["pred"]), e.gold) else misses).append(uid)
    misses.sort(key=lambda u: exs[u].meta.get("target_area_frac", 1.0))
    return hits[:n_hit] + misses[:n_miss]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--thinking-budget", type=int, default=2000)
    ap.add_argument("--max-spend", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=5)
    a = ap.parse_args()

    plan = {"screenspot": pick("screenspot", 2, 3), "screenspot_pro": pick("screenspot_pro", 1, 4)}
    sel = {ds: {e.uid: e for e in load(ds) if e.uid in set(u)} for ds, u in plan.items()}

    print(f"probe model: {a.model}")
    for ds, uids in plan.items():
        print(f"  {ds}: {len(uids)} examples")
    (RESULTS).mkdir(exist_ok=True)
    Path("results/probe_uids.json").write_text(json.dumps(plan, indent=2))

    client = anthropic.Anthropic()
    budget = Budget(a.max_spend)

    from concurrent.futures import ThreadPoolExecutor
    for cond, max_edge in [("native", None), ("edge1568", 1568)]:
        for ds, uids in plan.items():
            tag = f"{short_name(a.model)}_think{a.thinking_budget}_{cond}"
            out = RESULTS / f"{ds}__{tag}_r0.jsonl"
            todo = [sel[ds][u] for u in uids]
            with open(out, "w") as fh, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
                futs = [pool.submit(run_one, client, e, budget, a.thinking_budget, max_edge, a.model)
                        for e in todo]
                for f in futs:
                    fh.write(json.dumps(f.result()) + "\n")
            print(f"  wrote {out.name}  (${budget.spent:.3f} so far)")

    print(f"\ndone | {budget.calls} calls | ${budget.spent:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
