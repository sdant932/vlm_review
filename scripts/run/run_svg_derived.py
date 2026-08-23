"""Run the svg_localization-derived sets (counting, word_mc) on selected rungs.

A thin wrapper over the production request path rather than a new one: the same
`runner.run_one`, the same schema, the same scorer. The only thing this adds is
rung selection and a paired blind arm, because both EVAL.md files require the
blind control and neither the runner nor controls.py can express "these rungs
only".

    python scripts/run/run_svg_derived.py --datasets svg_counting svg_word_mc \
        --rungs small large --max-spend 12

Resumes: rows already present with a usable prediction are skipped, so a killed
run can be restarted without paying twice.
"""
from __future__ import annotations
import argparse, json, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from blindspot.core.adapters import ADAPTERS, Example
from blindspot.core.runner import Budget, run_one, short_name, MODEL, FatalBillingError

RESULTS = Path("results")

# Identical wording to scripts/run/controls.py so the blind arms of the two studies
# are comparable rather than merely similar.
BLIND_NOTE = (
    "\n\nNo image is provided. Answer from the question alone, using your best "
    "guess. You must commit to a specific answer; do not say that you cannot tell."
)


def blind_of(e: Example) -> Example:
    return Example(uid=f"blind:{e.uid}", dataset=f"{e.dataset}_blind", images=[],
                   question=e.question + BLIND_NOTE, answer_type=e.answer_type,
                   gold=e.gold, meta={**e.meta, "control": "blind", "src_uid": e.uid})


def done_uids(path: Path) -> set[str]:
    out = set()
    if not path.exists():
        return out
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("pred") is not None and not r.get("error") and not r.get("parse_error"):
            out.add(r["uid"])
    return out


def run(client, examples, out: Path, budget: Budget, thinking: int, concurrency: int,
        model: str) -> None:
    have = done_uids(out)
    todo = [e for e in examples if e.uid not in have]
    if not todo:
        print(f"  {out.name}: already complete ({len(have)} rows)")
        return
    lock = threading.Lock()
    written = failed = 0
    with open(out, "a") as fh, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(run_one, client, e, budget, thinking, None, model): e for e in todo}
        for f in as_completed(futs):
            try:
                rec = f.result()
            except FatalBillingError:
                print("  !! billing error -- stopping")
                break
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                written += 1
                failed += rec.get("pred") is None
                if written % 100 == 0:
                    print(f"    {written}/{len(todo)} | ${budget.spent:.3f} | "
                          f"no-pred {failed}", flush=True)
            if budget.exhausted():
                print(f"  !! budget ceiling ${budget.limit} reached")
                break
    print(f"  {out.name}: +{written} rows (${budget.spent:.3f} total, {failed} without a prediction)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["svg_counting", "svg_word_mc"])
    ap.add_argument("--rungs", nargs="+", default=["small", "large"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--thinking-budget", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-spend", type=float, default=12.0)
    ap.add_argument("--skip-blind", action="store_true")
    a = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    client = anthropic.Anthropic(max_retries=0, timeout=120.0)
    budget = Budget(a.max_spend)
    tag = f"{short_name(a.model)}_think{a.thinking_budget}_native_r0"
    rungs = set(a.rungs)

    for ds in a.datasets:
        exs = [e for e in ADAPTERS[ds]() if e.meta["resolution"] in rungs]
        print(f"{ds}: {len(exs)} questions at rungs {sorted(rungs)}")
        run(client, exs, RESULTS / f"{ds}__{tag}.jsonl", budget,
            a.thinking_budget, a.concurrency, a.model)
        if not a.skip_blind:
            # Both EVAL.md files require this: whatever survives with the image
            # withheld was never a perception task.
            run(client, [blind_of(e) for e in exs], RESULTS / f"{ds}__blind_{tag}.jsonl",
                budget, a.thinking_budget, a.concurrency, a.model)
    print(f"done | ${budget.spent:.3f} | {budget.calls} calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
