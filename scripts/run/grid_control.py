"""Does Haiku fail to LOCATE, or fail to SAY where?

Click-in-bbox conflates two very different deficits. This arm removes coordinate
emission entirely: a labelled 4x4 grid is drawn over the screenshot and the model
names the cell (e.g. "B3") containing the target.

The comparison is apples-to-apples because the click predictions from the main run
can be reduced to the same 4x4 cell. Same items, same granularity, only the answer
FORMAT differs:

    click -> derived cell   31.2%   (model emitted x,y; we bucketed it)
    named cell              ????    (model names the cell directly)

If naming beats clicking, coordinate emission is lossy and the perception was
partly intact. If they match, the ceiling is perceptual and no output format saves it.
"""
from __future__ import annotations
import argparse, json, random, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from PIL import Image, ImageDraw, ImageFont
from blindspot.core.adapters import ADAPTERS, Example
from blindspot.core.runner import run_one, Budget, FatalBillingError

K = 4
GRID_DIR = Path("cache/grid"); GRID_DIR.mkdir(parents=True, exist_ok=True)
ROWS = "ABCD"

INSTR = (
    f"The screenshot is divided into a {K}x{K} grid drawn in magenta. Rows are "
    f"labelled A (top) to {ROWS[K-1]} (bottom); columns are numbered 1 (left) to {K} (right). "
    "Each cell is labelled in its top-left corner.\n"
    "Reply with ONLY the label of the single cell that contains the described element, "
    "for example B3. Always commit to one cell even if you are unsure.\n\n"
    "Element: "
)

def gridded(src: str) -> str:
    """Draw the labelled grid once per screenshot; cache by source path."""
    out = GRID_DIR / (re.sub(r"[^A-Za-z0-9]+", "_", src)[-120:] + f".k{K}.jpg")
    if out.exists():
        return str(out)
    with Image.open(src) as im:
        im = im.convert("RGB")
        if max(im.size) > 1568:                    # what the model sees anyway
            s = 1568 / max(im.size)
            im = im.resize((round(im.width*s), round(im.height*s)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        w, h = im.size
        try: font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", max(14, w//55))
        except Exception: font = ImageFont.load_default()
        for i in range(1, K):
            d.line([(w*i/K, 0), (w*i/K, h)], fill=(255, 0, 255), width=2)
            d.line([(0, h*i/K), (w, h*i/K)], fill=(255, 0, 255), width=2)
        for r in range(K):
            for c in range(K):
                lab = f"{ROWS[r]}{c+1}"
                x, y = w*c/K + 5, h*r/K + 4
                d.rectangle([x-3, y-2, x+w//22, y+h//30], fill=(255, 0, 255))
                d.text((x, y), lab, fill=(255, 255, 255), font=font)
        im.save(out, quality=88)
    return str(out)

def cell_of(x, y):
    return f"{ROWS[min(int(y*K), K-1)]}{min(int(x*K), K-1)+1}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=350)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-spend", type=float, default=2.0)
    a = ap.parse_args()

    pool = list(ADAPTERS["screenspot_pro"]())
    random.Random(0).shuffle(pool)
    exs = []
    for e in pool[:a.n]:
        x0, y0, x1, y1 = e.gold
        exs.append(Example(
            uid=f"grid:{e.uid}", dataset="screenspot_pro_grid",
            images=[gridded(e.images[0])],
            question=INSTR + e.question,
            answer_type="span",
            gold=[cell_of((x0+x1)/2, (y0+y1)/2)],
            meta={**e.meta, "control": "grid4", "src_uid": e.uid, "bbox": e.gold},
        ))
    out = Path("results/control_grid4.jsonl")
    done = set()
    if out.exists():
        done = {json.loads(l)["uid"] for l in out.open() if l.strip()}
    exs = [e for e in exs if e.uid not in done]
    print(f"grid: {len(exs)} to run -> {out}", flush=True)
    budget = Budget(a.max_spend)
    client = anthropic.Anthropic(max_retries=0, timeout=120.0)
    n = 0
    with out.open("a") as f, ThreadPoolExecutor(max_workers=a.concurrency) as pl:
        futs = [pl.submit(run_one, client, e, budget, 2000, None) for e in exs]
        for fut in as_completed(futs):
            try: rec = fut.result()
            except FatalBillingError as e: print("FATAL:", e); break
            f.write(json.dumps(rec)+"\n"); n += 1
            if n % 50 == 0: f.flush(); print(f"  {n}/{len(exs)} | ${budget.spent:.3f}", flush=True)
    print(f"done | {n} calls | ${budget.spent:.3f}", flush=True)

if __name__ == "__main__":
    main()
