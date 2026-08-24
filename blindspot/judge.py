#!/usr/bin/env python
"""Grading that costs money.

    python -m blindspot.judge charxiv  --dataset charxiv --judge-model claude-opus-5
    python -m blindspot.judge equiv    --dataset infographicvqa
    python -m blindspot.judge gt-audit --dataset charxiv --per-category 5

Separate from `blindspot.core` because these call a model rather than
compare strings: they are non-deterministic, metered, and every one of them takes
a `--max-spend` ceiling.

  charxiv    CharXiv's official LLM-judge protocol. CharXiv does not score with
             string matching. It batches (question, gold, response) triplets under
             a per-question-type rubric and asks a judge model to extract the final
             answer and assign a binary score. Our own normalized-match / ANLS
             scorers are a lower bound on the free-text types -- a correct answer
             phrased differently is marked wrong -- so any CharXiv number published
             next to theirs has to come from this path. The prompt, the seven
             rubrics and the batching shape are vendored verbatim in
             `blindspot/core/vendor/charxiv_prompts.py`. What we choose is the
             judge model: CharXiv used GPT-4o, we use a strong Claude model. That
             is a deviation and it is recorded in every output row (`judge_model`)
             rather than glossed. The judge never sees the image -- only question,
             gold and response -- so it cannot rescue an answer the model got
             wrong, only recognise one it got right in different words.

  equiv      Meaning-equivalence judge for span answers, plus failure-mode
             resolution. Two jobs in one call, because they need the same context.
             (1) Was the model actually right? ANLS scores "310.5 million" against
             "310.5" as zero, and "1 IN 5 WOMEN" against "20%" as zero. Those are
             formatting failures, not perception failures, and a study about
             perception should not count them as the latter. A 60-item audit put
             this at ~12% of InfographicVQA's failures, with a further ~2% where
             the shipped gold is itself wrong. (2) If it was wrong, *how*? The
             deterministic classifier in `core/failure_modes.py` settles
             list-shaped cases exactly; this resolves the rest. The official ANLS
             score is never overwritten -- it stays as the comparable,
             published-protocol number. This adds a second column beside it.

  gt-audit   Adjudicate ground truth against the model, WITH the image. The
             equivalence judge is text-only: it can tell that "310.5 million" means
             "310.5", but not whether the benchmark's answer is supported by the
             figure. Two InfographicVQA golds have already been shown wrong by hand
             (77360 sums in a category the question does not ask about; 80749
             reports 16 where the infographic says "at least 64"), which raises how
             common that is and whether it differs by dataset. This sends the
             image, the question, the shipped gold and the model's answer to a
             stronger vision model and asks which is actually correct. For
             coordinate answers the gold box and the predicted point are drawn onto
             the image first, since a bounding box cannot be adjudicated as text.
             The judge is told it may side with the benchmark, and is asked to
             justify a verdict against it -- an adjudicator that always sides with
             the model would manufacture exactly the conclusion this is meant to
             test. This is how the contested-gold floor was measured.

`load_results` lives here rather than being imported from a report renderer: the
judge is upstream of reporting and must not depend on it.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from PIL import Image

from blindspot.eval import load_rows
from blindspot.core import load
from blindspot.core import FAILURE_MODE_LABELS as LABELS, classify
from blindspot.core import Budget, RESULTS, model_spec
from blindspot.charxiv import (DESCRIPTIVE_GRADING_PREFIX, REASONING_GRADING_PREFIX,
                                                     REASONING_GRADING_INST, rubric_for)


def load_results(dataset: str, tag: str = "haiku-4-5_think2000_native_r0") -> list[dict]:
    """One row per uid, preferring a usable prediction over a later failure.

    A uid can appear more than once: reruns append, and the retry logic keys off
    whether a *usable* row exists rather than the newest row. Plain last-wins
    would let a truncated retry overwrite an earlier good answer and show up as a
    model failure. Rule: any row with a prediction beats one without; among rows
    with predictions, the most recent wins.
    """
    path = RESULTS / f"{dataset}__{tag}.jsonl"
    if not path.exists():
        return []
    by_uid: dict[str, dict] = {}
    for line in open(path):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        prev = by_uid.get(rec["uid"])
        if prev is None or prev.get("pred") is None or rec.get("pred") is not None:
            by_uid[rec["uid"]] = rec
    return list(by_uid.values())


# ================================================================== charxiv
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
            from blindspot.charxiv import DESCRIPTIVE_GRADING_QMAP
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


def cmd_charxiv(a) -> int:
    grade(a.dataset, a.tag, a.judge_model, a.concurrency, a.max_spend, a.limit)
    return 0

# ==================================================================== equiv
EQUIV_SCHEMA = {
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

EQUIV_PROMPT = """A vision model answered a question about a document or chart. You are given the
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
    prompt = EQUIV_PROMPT.format(q=ex.question[:900], gold=ex.gold, pred=r.get("pred"))
    try:
        resp = client.messages.create(
            model=model, max_tokens=600, thinking={"type": "disabled"},
            output_config={"format": {"type": "json_schema", "schema": EQUIV_SCHEMA}},
            messages=[{"role": "user", "content": prompt}])
        budget.add(resp.usage.input_tokens, resp.usage.output_tokens, model)
        o = json.loads(next(b.text for b in resp.content if b.type == "text"))
    except Exception as e:
        return {"uid": r["uid"], "error": f"{type(e).__name__}: {e}"[:160]}
    return {"uid": r["uid"], "dataset": r["dataset"], "judge_model": model,
            "gold": ex.gold, "pred": r.get("pred"), "anls": r.get("string_score", r.get("score")),
            **o}


def cmd_equiv(a) -> int:

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

# ================================================================= gt-audit
GTAUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": [
            "gold_correct",        # benchmark right, model wrong
            "prediction_correct",  # model right, benchmark wrong
            "both_acceptable",     # question admits both readings
            "neither_correct",     # both wrong
        ]},
        "gt_quality": {"type": "string", "enum": ["unambiguous", "ambiguous", "wrong"]},
        "what_the_figure_shows": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "gt_quality", "what_the_figure_shows", "reasoning"],
    "additionalProperties": False,
}

GTAUDIT_PROMPT = """You are auditing a visual-question-answering benchmark, not grading a model.

Below is the image, the question as posed, the benchmark's reference answer, and a model's
answer. The model was scored WRONG. Your job is to determine whether that scoring was right.

Question: {q}
Benchmark reference answer: {gold}
Model answer: {pred}

Read the figure yourself first, then decide:

- verdict:
    gold_correct       the reference is right and the model is wrong
    prediction_correct the model is right and the reference is wrong or misreads the question
    both_acceptable    the question is genuinely ambiguous and both are defensible readings
    neither_correct    both answers are wrong

- gt_quality: is the reference answer `unambiguous` (clearly the single right answer to the
  question as worded), `ambiguous` (the wording admits more than one defensible answer), or
  `wrong` (contradicted by the figure)?

- what_the_figure_shows: state the relevant values or elements you can actually read in the
  image, in one or two sentences. Be concrete.

Default to `gold_correct`. Benchmarks are usually right, and you should only rule against the
reference when the figure plainly supports doing so. Say precisely which part of the figure
justifies it.{extra}"""

POINT_EXTRA = """

This is a UI-localization item. The image has been annotated: the GREEN box with a circle
around it is the benchmark's target element, and the RED crosshair is where the model clicked.
No other text has been added to the screenshot -- everything else you see is the original UI.
The instruction describes an element to click. Judge whether the green box actually contains
the element the instruction describes -- if it does, the reference is correct even though the
model missed it."""


def _image_block(row) -> dict:
    """Base64 image; for point answers, with gold box and prediction drawn on."""
    from blindspot.eval import draw_overlay
    ex = row["_ex"]
    im = Image.open(ex.images[0]).convert("RGB")
    if ex.answer_type == "point":
        # labels=False: the caption would otherwise occlude the element being judged.
        im = draw_overlay(im, {"answer_type": "point", "gold": ex.gold, "pred": row.get("pred")},
                          labels=False)
    if max(im.size) > 1568:
        sc = 1568 / max(im.size)
        im = im.resize((round(im.width * sc), round(im.height * sc)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(buf.getvalue()).decode()}}


def audit_one(client, row, model: str, budget: Budget) -> dict:
    ex = row["_ex"]
    is_point = ex.answer_type == "point"
    gold = ex.gold
    if is_point:
        x0, y0, x1, y1 = ex.gold
        gold = f"the element inside the green box (centre {(x0+x1)/2*100:.1f}%, {(y0+y1)/2*100:.1f}%)"
    pred = row.get("pred")
    if is_point and pred:
        pred = f"clicked at {pred[0]*100:.1f}%, {pred[1]*100:.1f}% (red crosshair)"
    text = GTAUDIT_PROMPT.format(q=ex.question[:900], gold=gold, pred=pred,
                         extra=POINT_EXTRA if is_point else "")
    try:
        resp = client.messages.create(
            model=model, max_tokens=4000, thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": GTAUDIT_SCHEMA}},
            messages=[{"role": "user", "content": [_image_block(row), {"type": "text", "text": text}]}])
        budget.add(resp.usage.input_tokens, resp.usage.output_tokens, model)
        txt = next((b.text for b in resp.content if b.type == "text"), None)
        if txt is None:  # ran out of tokens mid-thinking; no answer block emitted
            return {"uid": row["uid"], "error": f"no text block (stop={resp.stop_reason})"}
        o = json.loads(txt)
    except Exception as e:
        return {"uid": row["uid"], "error": f"{type(e).__name__}: {e}"[:200]}
    return {"uid": row["uid"], "dataset": row["dataset"],
            "failure_mode": row.get("failure_mode"), "judge_model": model,
            "question": ex.question[:400], "gold": str(ex.gold), "pred": str(row.get("pred")),
            "image": ex.images[0], **o}


def cmd_gt_audit(a) -> int:

    import collections, random
    rows = [r for r in load_rows(a.dataset) if (r.get("score") or 0) < 0.5]
    by_mode = collections.defaultdict(list)
    for r in rows:
        by_mode[r.get("failure_mode", "unclassified")].append(r)

    rng = random.Random(a.seed)
    sample = []
    for mode, rs in sorted(by_mode.items()):
        sample += rng.sample(rs, min(a.per_category, len(rs)))
    print(f"{a.dataset}: {len(rows)} failures across {len(by_mode)} modes "
          f"-> auditing {len(sample)} ({a.per_category}/mode) with {a.judge_model}")

    out = RESULTS / f"{a.dataset}__gtaudit.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                pass
    todo = [r for r in sample if r["uid"] not in done]

    client = anthropic.Anthropic(max_retries=0, timeout=180.0)
    budget, lock, n = Budget(a.max_spend), threading.Lock(), 0
    with open(out, "a") as fh, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        for rec in pool.map(lambda r: audit_one(client, r, a.judge_model, budget), todo):
            with lock:
                fh.write(json.dumps(rec) + "\n"); fh.flush(); n += 1
                if n % 10 == 0:
                    print(f"  {n}/{len(todo)} | ${budget.spent:.2f}", flush=True)
            if budget.exhausted():
                print("  !! spend cap reached"); break

    recs = [json.loads(l) for l in open(out)]
    recs = [r for r in recs if "error" not in r]
    v = collections.Counter(r["verdict"] for r in recs)
    q = collections.Counter(r["gt_quality"] for r in recs)
    print(f"\n{a.dataset}: {len(recs)} adjudicated | ${budget.spent:.2f}")
    print("  verdict:")
    for k, c in v.most_common():
        print(f"    {k:20} {c:>4}  ({c/len(recs)*100:.0f}%)")
    print("  ground-truth quality:")
    for k, c in q.most_common():
        print(f"    {k:20} {c:>4}  ({c/len(recs)*100:.0f}%)")
    bad = [r for r in recs if r["gt_quality"] != "unambiguous"]
    print(f"\n  => reference answer NOT unambiguous in {len(bad)}/{len(recs)} "
          f"({len(bad)/len(recs)*100:.0f}%) of audited failures")
    return 0


# ================================================================ dispatch

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m blindspot.judge",
                                description="LLM-judge grading. Every subcommand spends money.")
    sub = p.add_subparsers(dest="cmd", metavar="JUDGE")

    ch = sub.add_parser("charxiv", help="CharXiv official LLM-judge grading")
    ch.add_argument("--dataset", default="charxiv")
    ch.add_argument("--tag", default="haiku-4-5_think2000_native_r0")
    ch.add_argument("--judge-model", default=DEFAULT_JUDGE)
    ch.add_argument("--concurrency", type=int, default=16)
    ch.add_argument("--max-spend", type=float, default=15.0)
    ch.add_argument("--limit", type=int, default=None)
    ch.set_defaults(fn=cmd_charxiv)

    eq = sub.add_parser("equiv", help="meaning-equivalence + failure-mode judging")
    eq.add_argument("--dataset", default="infographicvqa")
    eq.add_argument("--judge-model", default="claude-sonnet-5")
    eq.add_argument("--concurrency", type=int, default=16)
    eq.add_argument("--max-spend", type=float, default=6.0)
    eq.set_defaults(fn=cmd_equiv)

    ga = sub.add_parser("gt-audit", help="vision-based ground-truth audit")
    ga.add_argument("--dataset", required=True)
    ga.add_argument("--per-category", type=int, default=5)
    ga.add_argument("--judge-model", default="claude-sonnet-5")
    ga.add_argument("--concurrency", type=int, default=8)
    ga.add_argument("--max-spend", type=float, default=5.0)
    ga.add_argument("--seed", type=int, default=0)
    ga.set_defaults(fn=cmd_gt_audit)
    return p


def main(argv=None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    if not getattr(a, "fn", None):
        p.print_help()
        return 2
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
