"""Meaning-equivalence judge for span answers, plus failure-mode resolution.

Two jobs in one call, because they need the same context:

1. Was the model actually right? ANLS scores "310.5 million" against "310.5" as
   zero, and "1 IN 5 WOMEN" against "20%" as zero. Those are formatting failures,
   not perception failures, and a study about perception should not count them as
   the latter. A 60-item audit put this at ~12% of InfographicVQA's failures,
   with a further ~2% where the shipped gold is itself wrong.

2. If it was wrong, *how*? The deterministic classifier in failure_modes.py
   settles list-shaped cases exactly; this resolves the rest.

The official ANLS score is never overwritten -- it stays as the comparable,
published-protocol number. This adds a second column beside it.

Usage:
    python -m blindspot.judging.equiv_judge --dataset infographicvqa
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from blindspot.analysis.aggregate import load_rows
from blindspot.core.failure_modes import LABELS, classify
from blindspot.core.runner import Budget, RESULTS

SCHEMA = {
    "type": "object",
    "properties": {
        "equivalent": {"type": "boolean"},
        "gold_looks_wrong": {"type": "boolean"},
        "failure_mode": {"type": "string", "enum": [
            "order_only", "extra_items", "missing_items", "partial_overlap",
            "format_only", "wrong_value"]},
        "why": {"type": "string"},
    },
    "required": ["equivalent", "gold_looks_wrong", "failure_mode", "why"],
    "additionalProperties": False,
}

PROMPT = """A vision model answered a question about a document or chart. You are given the
question, the benchmark's reference answer, and the model's answer. You cannot see the image.

Question: {q}
Reference answer: {gold}
Model answer: {pred}

Decide three things.

1. equivalent: does the model's answer mean the SAME THING as the reference, ignoring
   formatting, units, separators, capitalisation, rounding and phrasing?
   Examples of equivalent: "310.5 million" vs "310.5"; "1 in 5" vs "20%";
   "Gabrielle Douglas" vs "douglas"; "9 out of 10" vs "90%".

2. gold_looks_wrong: judging only from the question's wording, does the model's answer look
   better supported than the reference? Be conservative -- you cannot see the image, so say
   true only when the reference is internally inconsistent with the question as written.

3. failure_mode: if not equivalent, which of these best describes the difference?
   order_only      - the same items, listed in a different order
   extra_items     - all the reference items, plus additional ones
   missing_items   - a subset of the reference items
   partial_overlap - some items match, some do not
   format_only     - the same value expressed differently
   wrong_value     - a genuinely different answer
"""


def judge_one(client, r: dict, model: str, budget: Budget) -> dict:
    ex = r["_ex"]
    prompt = PROMPT.format(q=ex.question[:900], gold=ex.gold, pred=r.get("pred"))
    try:
        resp = client.messages.create(
            model=model, max_tokens=600, thinking={"type": "disabled"},
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": prompt}])
        budget.add(resp.usage.input_tokens, resp.usage.output_tokens, model)
        o = json.loads(next(b.text for b in resp.content if b.type == "text"))
    except Exception as e:
        return {"uid": r["uid"], "error": f"{type(e).__name__}: {e}"[:160]}
    return {"uid": r["uid"], "dataset": r["dataset"], "judge_model": model,
            "gold": ex.gold, "pred": r.get("pred"), "anls": r.get("string_score", r.get("score")),
            **o}


def main() -> int:
    ap = argparse.ArgumentParser(description="Meaning-equivalence + failure-mode judging")
    ap.add_argument("--dataset", default="infographicvqa")
    ap.add_argument("--judge-model", default="claude-sonnet-5")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-spend", type=float, default=6.0)
    a = ap.parse_args()

    rows = [r for r in load_rows(a.dataset) if (r.get("score") or 0) < 0.5]

    # Deterministic first: list-shaped failures are decided exactly and for free.
    for r in rows:
        r["failure_mode"] = classify(r["_ex"].gold, r.get("pred"))
    need = [r for r in rows if r["failure_mode"] == "unclassified"]

    out = RESULTS / f"{a.dataset}__equiv.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                pass
    todo = [r for r in need if r["uid"] not in done]

    print(f"{a.dataset}: {len(rows)} failures | "
          f"{len(rows)-len(need)} classified deterministically | {len(todo)} to judge")
    if todo:
        client = anthropic.Anthropic(max_retries=0, timeout=120.0)
        budget, lock, n = Budget(a.max_spend), threading.Lock(), 0
        with open(out, "a") as fh, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            for rec in pool.map(lambda r: judge_one(client, r, a.judge_model, budget), todo):
                with lock:
                    fh.write(json.dumps(rec) + "\n"); fh.flush(); n += 1
                    if n % 100 == 0:
                        print(f"  {n}/{len(todo)} | ${budget.spent:.2f}", flush=True)
                if budget.exhausted():
                    print("  !! spend cap reached"); break
        print(f"  judged {n} | ${budget.spent:.2f}")

    # ---- report ----
    verdicts = {}
    if out.exists():
        for line in open(out):
            try:
                v = json.loads(line)
                if "error" not in v:
                    verdicts[v["uid"]] = v
            except Exception:
                pass
    import collections
    modes, equiv, badgold = collections.Counter(), 0, 0
    for r in rows:
        v = verdicts.get(r["uid"])
        if v:
            if v["equivalent"]:
                equiv += 1
                modes["format_only"] += 1
                continue
            badgold += bool(v.get("gold_looks_wrong"))
            modes[v["failure_mode"]] += 1
        else:
            if r["failure_mode"] == "format_only":
                equiv += 1
            modes[r["failure_mode"]] += 1

    all_rows = load_rows(a.dataset)
    total = len(all_rows)
    # ANLS is continuous, so the official score is the MEAN, not the pass rate.
    official = sum(r.get("score") or 0 for r in all_rows)
    print(f"\n{a.dataset}: {total} questions")
    print(f"  official score           {official/total*100:.1f}%")
    print(f"  + meaning-equivalent     {(official+equiv)/total*100:.1f}%  "
          f"({equiv} answers correct but scored 0 on formatting)")
    print(f"  golds that look wrong    {badgold}")
    print("\n  failure modes:")
    for m, c in modes.most_common():
        print(f"    {LABELS.get(m, m):28} {c:>5}  ({c/max(len(rows),1)*100:.1f}% of failures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
