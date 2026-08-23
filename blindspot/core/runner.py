"""API runner: metered, resumable, and safe to kill at any point.

Design constraints that shaped this, all of them learned the hard way earlier
in this project:

* **No budget visibility.** A plain API key cannot read credit balance or spend
  caps (every /v1/organizations/* endpoint returns 401), so the harness meters
  itself and stops at `--max-spend` rather than discovering the ceiling as a
  mid-run 400.
* **Billing failures are fatal, not retryable.** Backing off and retrying a
  credit-balance error just burns minutes going nowhere, so it aborts the run.
* **Every run is resumable.** Results append to JSONL keyed by uid; a rerun
  skips what is already there. A crash costs the in-flight requests, not the run.
* **Non-deterministic by construction.** `temperature` is unavailable in
  anthropic 1.0.0, and thinking pins it to 1 regardless, so `--repeat` exists to
  measure run-to-run variance instead of pretending scores are exact.

Usage:
    python -m blindspot.core.runner --datasets screenspot --limit 20 --max-spend 1
    python -m blindspot.core.runner --datasets screenspot_pro --max-edge 1568   # ablation
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import anthropic

from blindspot.core.adapters import ADAPTERS, load
from blindspot.core.sampling import stratify, cell_key_for, report_cells
from blindspot.core.prompts import build_request, parse_response

MODEL = "claude-haiku-4-5-20251001"

# Per-model pricing (USD per million tokens) and thinking dialect.
#
# The thinking dialect is not cosmetic: `budget_tokens` was REMOVED on the 4.6+
# generation and returns a 400 there, while `adaptive` does not exist on 4.5-era
# models. Sending the wrong one is an immediate hard failure, so the model
# registry owns this rather than the call site.
MODELS = {
    "claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00, "thinking": "budget"},
    "claude-haiku-4-5":          {"in": 1.00, "out": 5.00, "thinking": "budget"},
    "claude-sonnet-4-5":         {"in": 3.00, "out": 15.00, "thinking": "budget"},
    "claude-sonnet-5":           {"in": 3.00, "out": 15.00, "thinking": "adaptive"},
    "claude-opus-5":             {"in": 5.00, "out": 25.00, "thinking": "adaptive"},
}


def model_spec(model: str) -> dict:
    if model not in MODELS:
        raise SystemExit(
            f"unknown model {model!r}; add it to MODELS with its pricing and "
            f"thinking dialect. Known: {', '.join(sorted(MODELS))}"
        )
    return MODELS[model]


def short_name(model: str) -> str:
    """Filename-safe tag so results from different models never collide."""
    return model.replace("claude-", "").replace("-20251001", "")


def thinking_config(model: str, budget: int) -> dict:
    return ({"type": "enabled", "budget_tokens": budget}
            if model_spec(model)["thinking"] == "budget"
            else {"type": "adaptive"})


RESULTS = Path("results")


class Budget:
    """Thread-safe spend tally with a hard ceiling."""

    def __init__(self, limit_usd: float | None):
        self.limit = limit_usd
        self.spent = 0.0
        self.calls = 0
        self._lock = threading.Lock()

    def add(self, in_tok: int, out_tok: int, model: str = MODEL) -> None:
        spec = model_spec(model)
        with self._lock:
            self.spent += in_tok / 1e6 * spec["in"] + out_tok / 1e6 * spec["out"]
            self.calls += 1

    def exhausted(self) -> bool:
        return self.limit is not None and self.spent >= self.limit


class FatalBillingError(RuntimeError):
    """Raised on credit-balance / spend-cap errors, which must not be retried."""


def _is_billing_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in ("credit balance", "billing", "spend limit", "quota"))


def run_one(client, ex, budget: Budget, thinking_budget: int, max_edge: int | None,
            model: str = MODEL, max_retries: int = 3) -> dict:
    content, schema, sizes, preflight_downscaled = build_request(ex, max_edge)
    rec = {
        "uid": ex.uid,
        "dataset": ex.dataset,
        "answer_type": ex.answer_type,
        "gold": ex.gold,
        "meta": ex.meta,
        "sent_image_sizes": sizes,
        "preflight_downscaled": preflight_downscaled,
        "max_edge": max_edge,
        "thinking_budget": thinking_budget,
        "model": model,
    }

    for attempt in range(max_retries):
        try:
            t0 = time.monotonic()
            resp = client.messages.create(
                model=model,
                # Thinking output plus the JSON answer must both fit. 1024 tokens
                # of headroom was not enough -- two pilot rows came back with
                # stop_reason=max_tokens and a truncated answer after thinking ran
                # long, so a failure here silently reads as a model failure when
                # it is really a budgeting bug. 2048 still truncated 4 of the
                # hardest ScreenSpot-Pro localizations, hence 4096.
                max_tokens=thinking_budget + 4096,
                thinking=thinking_config(model, thinking_budget),
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": content}],
            )
            latency = time.monotonic() - t0
            budget.add(resp.usage.input_tokens, resp.usage.output_tokens, model)

            text = next((b.text for b in resp.content if b.type == "text"), None)
            thinking = next((b.thinking for b in resp.content if b.type == "thinking"), "")
            rec.update({
                "raw": text,
                "thinking": thinking,
                "stop_reason": resp.stop_reason,
                "request_id": resp._request_id,
                "latency_s": round(latency, 2),
                "usage": {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
            })
            try:
                rec["pred"] = parse_response(ex, text) if text else None
            except Exception as e:  # schema held but content was unusable
                rec["pred"] = None
                rec["parse_error"] = f"{type(e).__name__}: {e}"
            return rec

        except Exception as e:
            if _is_billing_error(e):
                raise FatalBillingError(str(e)) from e
            if attempt == max_retries - 1:
                rec.update({"pred": None, "error": f"{type(e).__name__}: {e}"})
                return rec
            time.sleep(2**attempt + random.random())

    return rec


def existing_uids(path: Path) -> set[str]:
    """uids that already have a *usable* prediction.

    Anything without one -- an API error, a truncated answer, a parse failure --
    is deliberately excluded so a rerun retries it. Otherwise a transient 5xx or
    a since-fixed budgeting bug gets permanently baked into the results as if it
    were a model failure.
    """
    if not path.exists():
        return set()
    uids = set()
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue  # tolerate a torn final line from a killed run
            if not rec.get("error") and not rec.get("parse_error") and rec.get("pred") is not None:
                uids.add(rec["uid"])
    return uids


def run_dataset(client, dataset: str, args, budget: Budget) -> Path:
    model = getattr(args, "model", MODEL)
    res = "native" if args.max_edge is None else f"edge{args.max_edge}"
    # Model goes in the tag so a Sonnet control run can never overwrite, or be
    # silently pooled with, the Haiku results it exists to be compared against.
    tag = f"{short_name(model)}_think{args.thinking_budget}_{res}"
    out = RESULTS / f"{dataset}__{tag}_r{args.run}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    examples = load(dataset)
    uids = getattr(args, "uids", None)
    if uids:
        want = set(uids)
        examples = [e for e in examples if e.uid in want]
        missing = want - {e.uid for e in examples}
        if missing:
            raise SystemExit(f"{dataset}: uids not found: {sorted(missing)[:5]}")
    elif getattr(args, "per_cell", None):
        # Stratify by the slice we intend to report. Sampling by row instead
        # gave per-cell counts of 3-16 in the pilot -- noise rendered as
        # findings ("count lines: 100%, n=3"). See sampling.py.
        examples, realised = stratify(examples, cell_key_for(dataset),
                                      per_cell=args.per_cell, seed=args.seed)
        report_cells(dataset, realised)
    elif args.limit:
        rng = random.Random(args.seed)
        examples = rng.sample(examples, min(args.limit, len(examples)))

    done = existing_uids(out)
    todo = [e for e in examples if e.uid not in done]
    print(f"\n{dataset}: {len(examples)} selected, {len(done)} already done, {len(todo)} to run -> {out}")
    if not todo:
        return out

    # Record exactly what this run intends to do. Without it, a budget stop
    # leaves uids simply absent from the JSONL -- byte-identical to "never
    # selected" -- so dropped work cannot be told from work never asked for.
    out.with_suffix(".todo.json").write_text(json.dumps({
        "dataset": dataset, "tag": tag, "model": model, "seed": args.seed,
        "per_cell": getattr(args, "per_cell", None), "limit": args.limit,
        "max_edge": args.max_edge, "thinking_budget": args.thinking_budget,
        "uids": [e.uid for e in examples],
    }, indent=1))

    lock = threading.Lock()
    written = failed = 0
    fatal: list[Exception] = []
    stop = threading.Event()
    recent: list[bool] = []

    # Sliding submission window rather than queueing every future up front: at
    # 20k+ todo, f.cancel() cannot stop running work and ThreadPoolExecutor's
    # __exit__ blocks until the queue drains, so Ctrl-C and the budget stop both
    # appear to hang.
    pending = iter(todo)
    window = max(args.concurrency * 4, 16)

    with open(out, "a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        def submit(e):
            return pool.submit(run_one, client, e, budget, args.thinking_budget,
                               args.max_edge, model)

        inflight = {submit(e) for e in itertools.islice(pending, window)}
        while inflight:
            done_set, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done_set:
                try:
                    rec = fut.result()
                except FatalBillingError as e:
                    fatal.append(e); stop.set(); continue
                except Exception as e:
                    rec = {"uid": "?", "error": f"{type(e).__name__}: {e}"}
                with lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    written += 1
                    err = bool(rec.get("error"))
                    failed += err
                    recent.append(err)
                    if len(recent) > 200:
                        recent.pop(0)
                    if written % 100 == 0 or written == len(todo):
                        print(f"  {written}/{len(todo)} | ${budget.spent:.3f} | errors {failed}", flush=True)
                    if written % 500 == 0:
                        os.fsync(fh.fileno())
                # A wrong model id or revoked key would otherwise burn the run.
                if len(recent) >= 200 and sum(recent) / len(recent) > 0.5:
                    print("  !! >50% of the last 200 calls failed -- aborting", flush=True)
                    stop.set()
                if budget.exhausted():
                    print(f"  !! spend cap ${budget.limit} reached -- stopping cleanly", flush=True)
                    stop.set()
            if not stop.is_set():
                inflight |= {submit(e) for e in itertools.islice(pending, len(done_set))}

    if fatal:
        raise fatal[0]

    # Reconcile: anything selected but not usably answered is reported, not hidden.
    missing = [e.uid for e in examples if e.uid not in existing_uids(out)]
    if missing:
        out.with_suffix(".missing.json").write_text(json.dumps(missing, indent=1))
        print(f"  !! {len(missing)} selected uids have no usable answer "
              f"-> {out.with_suffix('.missing.json').name}", flush=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Haiku 4.5 perception blind-spot eval")
    p.add_argument("--datasets", nargs="+", default=["charxiv", "infographicvqa", "screenspot_pro"],
                   choices=sorted(ADAPTERS))
    p.add_argument("--limit", type=int, default=None, help="random sample of N (prefer --per-cell)")
    p.add_argument("--per-cell", type=int, default=None,
                   help="stratified: N per primitive cell (the statistically meaningful option)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (keeps subsets stable across runs)")
    p.add_argument("--thinking-budget", type=int, default=2000)
    p.add_argument("--max-edge", type=int, default=None,
                   help="pre-downscale images to this long edge (ablation; Haiku caps at ~1568)")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-spend", type=float, default=5.0, help="USD hard stop")
    p.add_argument("--run", type=int, default=0, help="run index, for repeat-variance measurement")
    p.add_argument("--model", default=MODEL, choices=sorted(MODELS),
                   help="target model; thinking dialect and pricing follow from it")
    p.add_argument("--uids", nargs="+", default=None,
                   help="run only these example uids (overrides --limit)")
    args = p.parse_args()

    # max_retries=0: the SDK's default of 2 multiplies with the runner's own 3
    # into six hidden attempts, which turns a 529 wave into a 6x wall-time
    # blowout that is invisible in the logs. Retries belong in one place.
    # read=120s: the SDK default is 600s, but p90 latency is ~12s -- a stalled
    # connection would otherwise park a worker for ten minutes with no signal.
    # A plain float, deliberately: anthropic 1.0.0 rejects an httpx.Timeout
    # object with "APIConnectionError: Connection error" before any request
    # leaves the process -- which reads exactly like a network outage.
    client = anthropic.Anthropic(max_retries=0, timeout=120.0)
    budget = Budget(args.max_spend)
    t0 = time.monotonic()

    try:
        for ds in args.datasets:
            if budget.exhausted():
                print(f"spend cap reached; skipping {ds}")
                continue
            run_dataset(client, ds, args, budget)
    except FatalBillingError as e:
        print(f"\nFATAL billing error -- run aborted, results so far are saved.\n  {e}", file=sys.stderr)
        return 2

    print(f"\ndone in {time.monotonic()-t0:.0f}s | {budget.calls} calls | ${budget.spent:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
