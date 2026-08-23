"""CharXiv's official LLM-judge grading.

CharXiv does not score with string matching. It batches (question, gold,
response) triplets under a per-question-type rubric and asks a judge model to
extract the final answer and assign a binary score. Our own normalized-match /
ANLS scorers are a lower bound on the free-text types -- a correct answer
phrased differently is marked wrong -- so any CharXiv number we publish next to
theirs has to come from this path.

The prompt, the seven rubrics and the batching shape are vendored verbatim in
`vendor/charxiv_constants.py`. What we choose is the judge model: CharXiv used
GPT-4o; we use a strong Claude model. That is a deviation and it is recorded in
every output row (`judge_model`) rather than glossed.

The judge never sees the image -- only the question, gold and response -- so it
cannot rescue an answer the model got wrong, only recognise one it got right in
different words.

Usage:
    python -m blindspot.judging.judge --dataset charxiv --judge-model claude-opus-5
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from blindspot.core.adapters import load
from blindspot.core.runner import Budget, RESULTS, model_spec
from blindspot.core.vendor.charxiv_constants import (DESCRIPTIVE_GRADING_PREFIX, REASONING_GRADING_PREFIX,
                                                     REASONING_GRADING_INST, rubric_for)

DEFAULT_JUDGE = "claude-opus-5"
BATCH = 5  # CharXiv batches five triplets per judging call


def _fill_prefix(prefix: str, n: int, keys: list[str], question: str) -> str:
    return (prefix
            .replace("<|NUM_TRIPLETS|>", str(n))
            .replace("<|JSON_KEYS|>", ", ".join(keys))
            .replace("<|OVERARCHING_QUESTION|>", question))


def build_descriptive_query(qid: int, question: str, batch: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """(prompt, json keys) for one batch of same-qid triplets. Mirrors CharXiv."""
    keys = []
    for i in range(len(batch)):
        keys += [f"extract_answer_T{i+1}", f"score_T{i+1}"]
    prompt = _fill_prefix(DESCRIPTIVE_GRADING_PREFIX, len(batch), keys, question)
    prompt += rubric_for(qid)
    for i, (resp, gold) in enumerate(batch, 1):
        prompt += (f"\n\nT{i}:\nGround Truth: {gold}\nModel Response: {resp}")
    return prompt, keys


def build_reasoning_query(question: str, gold: str, resp: str, inst_category: int) -> str:
    return (REASONING_GRADING_PREFIX
            + REASONING_GRADING_INST[int(inst_category)]
              .replace("<|question|>", str(question))
              .replace("<|ground_truth|>", str(gold))
              .replace("<|response|>", str(resp)))


def ask(client, prompt: str, model: str, budget: Budget, max_retries: int = 4) -> dict | None:
    for attempt in range(max_retries):
        try:
            kw = {"model": model, "max_tokens": 1024,
                  "messages": [{"role": "user", "content": prompt}]}
            if model_spec(model)["thinking"] == "adaptive":
                kw["thinking"] = {"type": "disabled"}   # grading is extraction, not reasoning
            r = client.messages.create(**kw)
            budget.add(r.usage.input_tokens, r.usage.output_tokens, model)
            txt = next((b.text for b in r.content if b.type == "text"), "") or ""
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0)) if m else None
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(2**attempt)
    return None


def grade(dataset: str, tag: str, judge_model: str, concurrency: int,
          max_spend: float, limit: int | None = None) -> Path:
    """Grade a results file with CharXiv's official protocol; write a sidecar."""
    from blindspot.reporting.report import load_results
    from collections import defaultdict

    examples = {e.uid: e for e in load(dataset)}
    rows = [r for r in load_results(dataset, tag) if r.get("pred") is not None]
    if limit:
        rows = rows[:limit]

    out = RESULTS / f"{dataset}__{tag}.judged.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                pass
    rows = [r for r in rows if r["uid"] not in done]
    if not rows:
        print(f"{dataset}: nothing left to judge -> {out}")
        return out

    # Descriptive triplets batch by qid (the rubric and overarching question are
    # per-type); reasoning is graded one at a time, as CharXiv does.
    desc: dict[int, list] = defaultdict(list)
    reas: list = []
    for r in rows:
        ex = examples.get(r["uid"])
        if ex is None:
            continue
        if ex.meta.get("split") == "reasoning":
            reas.append((r, ex))
        elif ex.meta.get("qid"):
            desc[int(ex.meta["qid"])].append((r, ex))

    jobs = []
    for qid, items in desc.items():
        for i in range(0, len(items), BATCH):
            jobs.append(("desc", qid, items[i:i + BATCH]))
    jobs += [("reas", None, [item]) for item in reas]

    client = anthropic.Anthropic(max_retries=0, timeout=120.0)
    budget = Budget(max_spend)
    lock = threading.Lock()
    n_done = n_fail = 0

    def run(job):
        kind, qid, items = job
        if kind == "desc":
            from blindspot.core.vendor.charxiv_constants import DESCRIPTIVE_GRADING_QMAP
            prompt, keys = build_descriptive_query(
                qid, DESCRIPTIVE_GRADING_QMAP[qid],
                [(str(r.get("pred")), str(ex.gold[0])) for r, ex in items])
            obj = ask(client, prompt, judge_model, budget)
            recs = []
            for i, (r, ex) in enumerate(items, 1):
                sc = None if obj is None else obj.get(f"score_T{i}")
                recs.append({"uid": r["uid"], "qid": qid, "split": "descriptive",
                             "judge_score": None if sc is None else float(sc),
                             "extracted": None if obj is None else obj.get(f"extract_answer_T{i}"),
                             "gold": ex.gold[0], "pred": r.get("pred"),
                             "judge_model": judge_model})
            return recs
        r, ex = items[0]
        obj = ask(client, build_reasoning_query(
            ex.meta.get("raw_question") or ex.question, ex.gold[0], str(r.get("pred")),
            ex.meta.get("reasoning_a_type", 1)), judge_model, budget)
        sc = None if obj is None else (obj.get("score") if "score" in obj else None)
        return [{"uid": r["uid"], "qid": None, "split": "reasoning",
                 "judge_score": None if sc is None else float(sc),
                 "extracted": None if obj is None else obj.get("extract_answer"),
                 "gold": ex.gold[0], "pred": r.get("pred"), "judge_model": judge_model}]

    print(f"{dataset}: judging {len(rows)} responses in {len(jobs)} calls "
          f"with {judge_model} -> {out}")
    with open(out, "a") as fh, ThreadPoolExecutor(max_workers=concurrency) as pool:
        for recs in pool.map(run, jobs):
            with lock:
                for rec in recs:
                    fh.write(json.dumps(rec) + "\n")
                    n_done += 1
                    n_fail += rec["judge_score"] is None
                fh.flush()
                if n_done % 200 < len(recs):
                    print(f"  {n_done}/{len(rows)} | ${budget.spent:.2f}", flush=True)
            if budget.exhausted():
                print("  !! judge spend cap reached", flush=True)
                break
    print(f"{dataset}: judged {n_done} ({n_fail} ungraded) | ${budget.spent:.2f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="CharXiv official LLM-judge grading")
    ap.add_argument("--dataset", default="charxiv")
    ap.add_argument("--tag", default="haiku-4-5_think2000_native_r0")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-spend", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    grade(a.dataset, a.tag, a.judge_model, a.concurrency, a.max_spend, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
