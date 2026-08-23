"""Adjudicate ground truth against the model, WITH the image.

The equivalence judge in equiv_judge.py is text-only: it can tell that
"310.5 million" means "310.5", but it cannot tell whether the benchmark's answer
is actually supported by the figure. Two InfographicVQA golds have already been
shown wrong by hand (77360 sums in a category the question does not ask about;
80749 reports 16 where the infographic says "at least 64"), which raises the
obvious question of how common that is and whether it differs by dataset.

This sends the image, the question, the shipped gold and the model's answer to a
stronger vision model and asks which is actually correct. For coordinate answers
the gold box and the predicted point are drawn onto the image first, since a
bounding box cannot be adjudicated as text.

The judge is told it may side with the benchmark, and is asked to justify a
verdict against it -- an adjudicator that always sides with the model would
manufacture exactly the conclusion this is meant to test.

Usage:
    python -m blindspot.judging.gt_audit --dataset charxiv --per-category 5
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from PIL import Image

from blindspot.analysis.aggregate import load_rows
from blindspot.core.failure_modes import LABELS as FM_LABELS
from blindspot.core.runner import Budget, RESULTS

SCHEMA = {
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

PROMPT = """You are auditing a visual-question-answering benchmark, not grading a model.

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
    from blindspot.analysis.annotate import draw_overlay
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
    text = PROMPT.format(q=ex.question[:900], gold=gold, pred=pred,
                         extra=POINT_EXTRA if is_point else "")
    try:
        resp = client.messages.create(
            model=model, max_tokens=4000, thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Vision-based ground-truth audit")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--per-category", type=int, default=5)
    ap.add_argument("--judge-model", default="claude-sonnet-5")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-spend", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

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


if __name__ == "__main__":
    raise SystemExit(main())
