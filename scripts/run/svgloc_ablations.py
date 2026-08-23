"""Diagnostic and prompt-wording ablations for svg_localization point questions.

Two questions, one shared sample so every arm is paired against every other.

DIAGNOSTIC -- is the deficit knowledge or expression?
  repeat        the identical request a second time. Thinking pins temperature
                to 1, so the spread between two runs is the in-set noise floor
                the localization set otherwise lacks, and it separates "noisy
                estimate" from "stable but wrong".
  quadrant_mc   which quadrant is it in, as a 4-way letter choice. Same
                granularity as the 2x2 rung of the precision curve, but with the
                continuous channel removed entirely. This is the grid-control
                idea WITHOUT drawing on the image -- the dataset forbids an
                overlay, which occluded the gold text in 31% of a trial build.
  crop          the same question on an image cropped to the quadrant holding
                the target. Shrinks the search field without changing precision
                demands, separating "could not find it" from "could not aim".
  bbox          ask for the element's box instead of its centre, scored as
                centre-of-predicted-box inside gold so it stays comparable.

WORDING -- can we just ask better? Each arm keeps the image, the schema, the
model and the thinking budget byte-identical and varies only the text, so any
difference is the wording and nothing else. All four are built on the finding
that coarse spatial ability is intact (~68%) while precision is not, so they
scaffold coarse-to-fine rather than simply asking harder.
"""
from __future__ import annotations
import argparse, json, random, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
import anthropic
from blindspot.core.adapters import load, Example
from blindspot.core.runner import Budget, run_one, short_name, MODEL
from blindspot.core.prompts import POINT_INSTRUCTION

RESULTS = Path("results")
CROPS = Path("cache/svgloc_crops")
RUNGS = ("small", "large")

CAREFUL = (
    "Locate the described UI element in the screenshot and return the point at "
    "its center.\n"
    "Use a normalized coordinate system where x=0 is the left edge, x=1000 the "
    "right edge, y=0 the top edge, and y=1000 the bottom edge.\n"
    "Be precise: the target is small, often under 1% of the image, so an answer "
    "that is merely in the right region will miss it. Read off the element's "
    "left and right edges and take the midpoint, then its top and bottom edges "
    "and take the midpoint.\n"
    "Always return your single best guess, even if you are uncertain.\n\n"
    "Element: ")

DESCRIBE = (
    "Locate the described UI element in the screenshot.\n"
    "First, in your reasoning, describe in words where it sits: which part of "
    "the image, what is immediately above, below, left and right of it, and how "
    "far across and down the frame it falls.\n"
    "Then convert that description into a point at the element's center, using "
    "a normalized coordinate system where x=0 is the left edge, x=1000 the "
    "right edge, y=0 the top edge, and y=1000 the bottom edge.\n"
    "Always return your single best guess, even if you are uncertain.\n\n"
    "Element: ")

CELL_THEN_POINT = (
    "Locate the described UI element in the screenshot.\n"
    "Work coarse to fine. First, in your reasoning, divide the image into a 4x4 "
    "grid of equal cells, columns A-D from the left and rows 1-4 from the top, "
    "and name the single cell that contains the element. Then divide that cell "
    "into a 4x4 grid and name the sub-cell. Then convert the sub-cell to a "
    "point.\n"
    "Report the point at the element's center in a normalized coordinate system "
    "where x=0 is the left edge, x=1000 the right edge, y=0 the top edge, and "
    "y=1000 the bottom edge.\n"
    "Always return your single best guess, even if you are uncertain.\n\n"
    "Element: ")

LANDMARK = (
    "Locate the described UI element in the screenshot.\n"
    "First, in your reasoning, find the largest nearby landmark you are "
    "confident about -- a title, an axis, a panel edge, a big labelled box -- "
    "and estimate its position. Then state where the target sits relative to "
    "that landmark, as a fraction of the image width and height. Then combine "
    "the two into an absolute position.\n"
    "Report the point at the element's center in a normalized coordinate system "
    "where x=0 is the left edge, x=1000 the right edge, y=0 the top edge, and "
    "y=1000 the bottom edge.\n"
    "Always return your single best guess, even if you are uncertain.\n\n"
    "Element: ")

BBOX_INSTRUCTION = (
    "Locate the described UI element in the screenshot and return its bounding "
    "box.\n"
    "Use a normalized coordinate system where x=0 is the left edge, x=1000 the "
    "right edge, y=0 the top edge, and y=1000 the bottom edge. Report x0,y0 as "
    "the top-left corner and x1,y1 as the bottom-right corner of the element.\n"
    "Always return your single best guess, even if you are uncertain.\n\n"
    "Element: ")

QUADRANT_LETTERS = ["top-left", "top-right", "bottom-left", "bottom-right"]


def sample(n: int, seed: int) -> list[Example]:
    """Deterministic, balanced across rung x area tertile."""
    pts = [e for e in load("svg_localization")
           if e.answer_type == "point" and e.meta["resolution"] in RUNGS]
    per_rung = n // len(RUNGS)
    out = []
    for rung in RUNGS:
        rows = sorted((e for e in pts if e.meta["resolution"] == rung),
                      key=lambda e: (e.meta["target_area_frac"], e.uid))
        for t in range(3):
            band = rows[t * len(rows) // 3:(t + 1) * len(rows) // 3]
            want = per_rung // 3 + (1 if t < per_rung % 3 else 0)
            out += random.Random(f"{seed}:{rung}:{t}").sample(band, min(want, len(band)))
    return out


def quadrant_of(box) -> int:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return (0 if cy < 0.5 else 2) + (0 if cx < 0.5 else 1)


def make_crop(e: Example) -> Example | None:
    """Crop to the half-width, half-height quadrant containing the target.

    The search field shrinks 4x; the target keeps its absolute pixel size, so
    precision demands are unchanged in pixels while the field is smaller. Gold
    is remapped into the crop's own normalized frame.
    """
    CROPS.mkdir(parents=True, exist_ok=True)
    src = Path(e.images[0])
    b = e.gold
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    x0 = min(max(cx - 0.25, 0.0), 0.5)
    y0 = min(max(cy - 0.25, 0.0), 0.5)
    x1, y1 = x0 + 0.5, y0 + 0.5
    if not (x0 <= b[0] and b[2] <= x1 and y0 <= b[1] and b[3] <= y1):
        return None                     # target straddles the crop edge; skip
    out = CROPS / f"{e.uid.replace(':', '_')}.png"
    if not out.exists():
        im = Image.open(src)
        W, H = im.size
        im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))).save(out)
    g = [(b[0] - x0) / 0.5, (b[1] - y0) / 0.5, (b[2] - x0) / 0.5, (b[3] - y0) / 0.5]
    return Example(uid=e.uid, dataset=e.dataset, images=[str(out)], question=e.question,
                   answer_type="point", gold=g, meta={**e.meta, "arm": "crop"})


def build(arm: str, exs: list[Example]) -> list[Example]:
    if arm in ("baseline", "repeat"):
        return exs
    if arm == "quadrant_mc":
        out = []
        for e in exs:
            q = quadrant_of(e.gold)
            out.append(Example(uid=e.uid, dataset=e.dataset, images=e.images,
                               question=f"Which quarter of the image contains {e.question}?",
                               answer_type="choice", gold=["ABCD"[q]],
                               meta={**e.meta, "options": QUADRANT_LETTERS, "arm": arm}))
        return out
    if arm == "crop":
        return [c for c in (make_crop(e) for e in exs) if c]
    if arm == "bbox":
        return [Example(uid=e.uid, dataset=e.dataset, images=e.images, question=e.question,
                        answer_type="bbox", gold=e.gold,
                        meta={**e.meta, "arm": arm,
                              "prompt_override": BBOX_INSTRUCTION + e.question})
                for e in exs]
    text = {"careful": CAREFUL, "describe": DESCRIBE,
            "cell_then_point": CELL_THEN_POINT, "landmark": LANDMARK}[arm]
    return [Example(uid=e.uid, dataset=e.dataset, images=e.images, question=e.question,
                    answer_type="point", gold=e.gold,
                    meta={**e.meta, "arm": arm, "prompt_override": text + e.question})
            for e in exs]


ARMS = ["repeat", "quadrant_mc", "crop", "bbox",
        "careful", "describe", "cell_then_point", "landmark"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--thinking-budget", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-spend", type=float, default=16.0)
    a = ap.parse_args()

    exs = sample(a.n, a.seed)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "svgloc_ablation_uids.json").write_text(
        json.dumps([e.uid for e in exs], indent=1))
    print(f"shared sample: {len(exs)} point questions "
          f"({sum(1 for e in exs if e.meta['resolution']=='small')} small / "
          f"{sum(1 for e in exs if e.meta['resolution']=='large')} large)")

    client = anthropic.Anthropic(max_retries=0, timeout=120.0)
    budget = Budget(a.max_spend)
    tag = f"{short_name(a.model)}_think{a.thinking_budget}"
    for arm in a.arms:
        todo = build(arm, exs)
        out = RESULTS / f"svgloc_abl_{arm}__{tag}.jsonl"
        have = set()
        if out.exists():
            for line in open(out):
                if line.strip():
                    r = json.loads(line)
                    if r.get("pred") is not None:
                        have.add(r["uid"])
        todo = [e for e in todo if e.uid not in have]
        if not todo:
            print(f"  {arm:16s} already complete"); continue
        n_ok = 0
        with open(out, "a") as fh, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            futs = [pool.submit(run_one, client, e, budget, a.thinking_budget, None, a.model)
                    for e in todo]
            for f in as_completed(futs):
                rec = f.result()
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                n_ok += rec.get("pred") is not None
        print(f"  {arm:16s} {n_ok}/{len(todo)} usable  (${budget.spent:.2f})")
        if budget.exhausted():
            print("  !! budget ceiling reached"); break
    print(f"done | ${budget.spent:.3f} | {budget.calls} calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
