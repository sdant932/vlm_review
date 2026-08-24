"""Score real model samples, to show what the reward actually does.

Describing a reward is cheap; the useful thing is seeing a group of real answers
scored, because that is where the reward's weaknesses show. This asks the target
model the same question several times, scores every answer by IoU against the
true box, and computes the group statistics GRPO would use.

Nothing here is illustrative. The boxes are the model's, the scores are computed
from them, and a group where every sample scores zero is reported as such rather
than replaced with a tidier one.

Usage
-----
    python -m scripts.report.finetune_worked --prompts 3 --samples 8
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

REPO = Path(__file__).resolve().parents[2]
MODEL = "claude-haiku-4-5-20251001"

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


def ask(client, img_b64: str, media: str, q: str) -> str:
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
    return "".join(b.text for b in r.content if b.type == "text")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/svgloc_mr")
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="outputs/finetune/worked_examples.json")
    a = ap.parse_args()

    root = REPO / a.dataset
    rows = [json.loads(l) for l in (root / "manifest.jsonl").open() if l.strip()]

    # One prompt per target-size band, so the table spans easy to hard.
    rows.sort(key=lambda r: r["box_area_frac"])
    rng = random.Random(a.seed)
    n = len(rows)
    picks = [rows[rng.randrange(i * n // a.prompts, (i + 1) * n // a.prompts)]
             for i in range(a.prompts)]

    client = anthropic.Anthropic(max_retries=2, timeout=180.0)
    out = []
    for rec in picks:
        b64 = base64.b64encode((root / rec["image"]).read_bytes()).decode()
        samples = []
        for _ in range(a.samples):
            try:
                box = parse(ask(client, b64, "image/png", rec["question"]))
            except Exception as e:                       # noqa: BLE001
                print(f"    call failed: {e}")
                box = None
            samples.append({"box": box,
                            "iou": iou(box, rec["box_norm"]) if box else 0.0})
        r = [s["iou"] for s in samples]
        mean, sd = st.mean(r), (st.pstdev(r) or 0.0)
        for s in samples:
            s["advantage"] = (s["iou"] - mean) / sd if sd > 1e-9 else 0.0
        out.append({"uid": rec["uid"], "question": rec["question"],
                    "aspect": rec["aspect"], "image_px": rec["image_px"],
                    "area_frac": rec["box_area_frac"], "true_box": rec["box_norm"],
                    "samples": samples, "mean_iou": mean, "std_iou": sd,
                    "usable_group": sd > 1e-9})
        print(f"  {rec['uid']}  area {rec['box_area_frac']*100:.3f}%  "
              f"mean IoU {mean:.3f}  std {sd:.3f}  "
              f"{'usable' if sd > 1e-9 else 'DEAD GROUP'}")

    p = REPO / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
