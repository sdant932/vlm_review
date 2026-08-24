"""Score real model samples, to show what the reward actually does.

Describing a reward is cheap; the useful thing is seeing a group of real answers
scored, because that is where the reward's weaknesses show. This asks the target
model the same question several times, scores every answer by IoU against the
true box, and computes the group statistics GRPO would use.

Nothing here is illustrative. The boxes are the model's, the scores are computed
from them, and a group where every sample scores zero is reported as such rather
than replaced with a tidier one.

Spending
--------
`--prompts P --samples S` is P*S calls, so the arguments alone are not a budget:
`--prompts 100 --samples 100` is ten thousand of them. Every call is therefore
priced into a `core.Budget` and the ceiling is checked BEFORE each one. The
module is serial by construction -- one request in flight at a time -- so the
worst overshoot is a single call. See the comment on the check in `main`.

Usage
-----
    python -m blindspot.report_worked --prompts 3 --samples 8 --max-spend 0.50
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import re
import statistics as st
from pathlib import Path

import anthropic

from blindspot.core import MODEL, Budget

REPO = Path(__file__).resolve().parents[1]

PROMPT = (
    "Return the bounding box of {q}.\n"
    'Answer with JSON only: {{"box": [x0, y0, x1, y1]}}\n'
    "Coordinates are fractions of the image between 0 and 1, with [0,0] at the "
    "top-left corner and [1,1] at the bottom-right. x0 < x1 and y0 < y1."
)
BOX_RE = re.compile(r'"box"\s*:\s*\[([^\]]+)\]')


def iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter)


def parse(text: str):
    m = BOX_RE.search(text)
    if not m:
        return None
    try:
        v = [float(x) for x in m.group(1).split(",")]
    except ValueError:
        return None
    if len(v) != 4 or not (v[0] < v[2] and v[1] < v[3]):
        return None
    return v


def ask(client, budget: Budget, img_b64: str, media: str, q: str) -> str:
    r = client.messages.create(
        model=MODEL,
        max_tokens=2000 + 1024,
        thinking={"type": "enabled", "budget_tokens": 2000},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": media, "data": img_b64}},
            {"type": "text", "text": PROMPT.format(q=q)},
        ]}],
    )
    # Priced from the response's own usage, not an estimate, and charged before
    # the text is handed back so no path can consume an answer without paying
    # for it.
    budget.add(r.usage.input_tokens, r.usage.output_tokens, MODEL)
    return "".join(b.text for b in r.content if b.type == "text")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/svgloc_mr")
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="outputs/finetune/worked_examples.json")
    ap.add_argument("--max-spend", type=float, default=0.50,
                    help="USD ceiling for this run (default 0.50). --prompts x "
                         "--samples is a call count, not a budget; this is the "
                         "budget. Pass 0 to make the run a no-op.")
    a = ap.parse_args()

    root = REPO / a.dataset
    rows = [json.loads(l) for l in (root / "manifest.jsonl").open() if l.strip()]

    # One prompt per target-size band, so the table spans easy to hard.
    rows.sort(key=lambda r: r["box_area_frac"])
    rng = random.Random(a.seed)
    n = len(rows)
    picks = [rows[rng.randrange(i * n // a.prompts, (i + 1) * n // a.prompts)]
             for i in range(a.prompts)]

    budget = Budget(a.max_spend)
    print(f"{a.prompts} prompt(s) x {a.samples} sample(s) = "
          f"{a.prompts * a.samples} call(s), ceiling ${a.max_spend:.4g}")

    client = anthropic.Anthropic(max_retries=2, timeout=180.0)
    out, capped = [], False
    for rec in picks:
        b64 = base64.b64encode((root / rec["image"]).read_bytes()).decode()
        samples = []
        for _ in range(a.samples):
            # Checked BEFORE the call, not after: a call already issued when the
            # ceiling trips still completes and still bills. `core`'s runner
            # needs a sliding submission window for that reason -- a wide window
            # once carried a $0.029 cap 12.8x over. Here there is exactly one
            # request in flight, so this check bounds the overshoot at one call.
            # Do not put these calls in a thread pool without reading the
            # `window = max(args.concurrency, 1)` comment in core.py first.
            if budget.exhausted():
                capped = True
                break
            try:
                box = parse(ask(client, budget, b64, "image/png", rec["question"]))
            except Exception as e:                       # noqa: BLE001
                print(f"    call failed: {e}")
                box = None
            samples.append({"box": box,
                            "iou": iou(box, rec["box_norm"]) if box else 0.0})
        if not samples:                 # the cap hit before this group started
            break
        r = [s["iou"] for s in samples]
        mean, sd = st.mean(r), (st.pstdev(r) or 0.0)
        for s in samples:
            s["advantage"] = (s["iou"] - mean) / sd if sd > 1e-9 else 0.0
        out.append({"uid": rec["uid"], "question": rec["question"],
                    "aspect": rec["aspect"], "image_px": rec["image_px"],
                    "area_frac": rec["box_area_frac"], "true_box": rec["box_norm"],
                    "samples": samples, "mean_iou": mean, "std_iou": sd,
                    "usable_group": sd > 1e-9,
                    # A group cut short by the ceiling is a smaller group, and a
                    # group statistic over 3 of 8 samples is not the one asked
                    # for. Say so in the record rather than let it read as whole.
                    "n_samples": len(samples), "truncated": len(samples) < a.samples})
        print(f"  {rec['uid']}  area {rec['box_area_frac']*100:.3f}%  "
              f"mean IoU {mean:.3f}  std {sd:.3f}  "
              f"{'usable' if sd > 1e-9 else 'DEAD GROUP'}"
              f"{f'  [TRUNCATED {len(samples)}/{a.samples}]' if len(samples) < a.samples else ''}")
        if capped:
            break

    print(f"\n{budget.calls} call(s), ${budget.spent:.4f} of ${a.max_spend:.4g}")

    if not out:
        # An empty run must not truncate the artifact it was going to replace:
        # `[]` on disk is indistinguishable from a run that found nothing.
        print(f"!! no sample was taken -- ${a.max_spend:.4g} left no room for a "
              f"single call. {a.out} left untouched.")
        return 2
    if capped:
        print(f"!! spend cap ${a.max_spend:.4g} reached -- stopped after "
              f"{len(out)}/{len(picks)} prompt(s). The file below is PARTIAL; "
              f"re-run with a higher --max-spend for the whole table.")

    p = REPO / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
