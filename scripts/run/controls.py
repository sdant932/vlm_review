"""Controls that isolate *why* Haiku fails, run as ablations on the main eval.

Three experiments, all reusing the production request path (`runner.run_one`) so
that the only thing that differs from the headline run is the manipulation
itself -- same model, same thinking budget, same schema, same scorer.

  blind      Ask the identical question with NO image. Anything still answered
             correctly was never a perception task: it was recoverable from the
             question text plus world knowledge. This is the ceiling that must be
             subtracted from every accuracy number before calling it "vision".

  onepage    SlideVQA multi-evidence questions with only the FIRST evidence slide.
             If accuracy holds, the question never genuinely spanned pages and the
             "multi-hop" label is a dataset artifact. If it collapses, the span is
             real and the earlier -4.0 F1 multi-page cost is a true integration cost.

  grid       ScreenSpot-Pro with a labelled 4x4 grid drawn over the screenshot;
             the model names the CELL rather than emitting coordinates. Click
             accuracy conflates "did not see it" with "saw it, cannot say where".
             Grid accuracy >> click accuracy would prove the deficit is expression.

Nothing here overwrites an existing result file; every arm writes its own
results/control_<arm>*.jsonl.
"""
from __future__ import annotations
import argparse, json, random, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from blindspot.core.adapters import ADAPTERS, Example
from blindspot.core.runner import run_one, Budget, MODEL, FatalBillingError

RESULTS = Path("results")

BLIND_NOTE = (
    "\n\nNo image is provided. Answer from the question alone, using your best "
    "guess. You must commit to a specific answer; do not say that you cannot tell."
)


def blind_examples(dataset: str, n: int, seed: int = 0) -> list[Example]:
    """Same question, same schema, image stripped."""
    pool = [e for e in ADAPTERS[dataset]() if e.answer_type != "point"]
    random.Random(seed).shuffle(pool)
    out = []
    for e in pool[:n]:
        out.append(Example(
            uid=f"blind:{e.uid}", dataset=f"{e.dataset}_blind",
            images=[],                      # the manipulation
            question=e.question + BLIND_NOTE,
            answer_type=e.answer_type,
            gold=e.gold, meta={**e.meta, "control": "blind", "src_uid": e.uid},
        ))
    return out


def onepage_examples(which: int = 0) -> list[Example]:
    """SlideVQA multi-evidence questions, given only one of their evidence slides."""
    out = []
    for e in ADAPTERS["slidevqa"]():
        if not e.meta.get("multi_page"):
            continue
        idx = which if which < len(e.images) else len(e.images) - 1
        out.append(Example(
            uid=f"onepage{which}:{e.uid}", dataset="slidevqa_onepage",
            images=[e.images[idx]],         # the manipulation
            question=e.question, answer_type=e.answer_type, gold=e.gold,
            meta={**e.meta, "control": f"onepage{which}", "src_uid": e.uid,
                  "n_pages_sent": 1},
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["blind", "onepage"])
    ap.add_argument("--datasets", nargs="*", default=["charxiv", "infographicvqa", "slidevqa", "ai2d"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--which", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-spend", type=float, default=4.0)
    a = ap.parse_args()

    if a.arm == "blind":
        exs, out = [], RESULTS / "control_blind.jsonl"
        for d in a.datasets:
            exs += blind_examples(d, a.n)
    else:
        exs, out = onepage_examples(a.which), RESULTS / f"control_onepage{a.which}.jsonl"

    done = set()
    if out.exists():
        for line in out.open():
            try: done.add(json.loads(line)["uid"])
            except Exception: pass
    exs = [e for e in exs if e.uid not in done]
    print(f"{a.arm}: {len(exs)} to run ({len(done)} already done) -> {out}", flush=True)
    if not exs:
        return

    budget = Budget(a.max_spend)
    client = anthropic.Anthropic(max_retries=0, timeout=120.0)
    n = 0
    with out.open("a") as f, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futs = {pool.submit(run_one, client, e, budget, 2000, None): e for e in exs}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except FatalBillingError as e:
                print("FATAL BILLING:", e); break
            f.write(json.dumps(rec) + "\n"); n += 1
            if n % 100 == 0:
                f.flush(); print(f"  {n}/{len(exs)} | ${budget.spent:.3f}", flush=True)
    print(f"done | {n} calls | ${budget.spent:.3f}", flush=True)


if __name__ == "__main__":
    main()
