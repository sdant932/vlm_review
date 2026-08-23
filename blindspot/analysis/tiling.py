"""Experiment: does sending the screenshot as interleaved native-resolution patches
recover the grounding accuracy that downscaling destroys?

The premise is mechanical, not speculative. Haiku 4.5 caps a single image at a
~1568px long edge, so a 3840x2160 screenshot loses 93% of its pixels on the way
in. But the cap is *per image*, and a 3x3 tile of that same screenshot is
~1408x792 -- already under the cap, so each patch arrives at native resolution.
Tiling therefore trades one downscaled view for N full-detail views, at N times
the image tokens.

The model is shown a labelled full view first (for global context), then each
patch in reading order, and answers with a patch address plus a position inside
that patch. Coordinates are mapped back to whole-image space here.

Patches overlap so that a target straddling a seam is still wholly inside at
least one patch.

Usage:
    python -m blindspot.analysis.tiling --dataset screenspot_pro --limit 50
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from PIL import Image

from blindspot.core.adapters import load
from blindspot.core.runner import Budget, MODEL, RESULTS
from blindspot.core.scoring import point_in_bbox

# Keep total images per request under 20: above that the API caps each image at
# 2000x2000, and the whole point here is to stay under the per-image ceiling.
GRID = (3, 3)
OVERLAP = 0.12
FULL_VIEW_MAX_EDGE = 1024  # context only; the patches carry the detail

SCHEMA = {
    "type": "object",
    "properties": {
        "patch": {"type": "integer", "description": "index of the patch containing the element"},
        "x": {"type": "integer", "description": "0-1000 horizontal position within that patch"},
        "y": {"type": "integer", "description": "0-1000 vertical position within that patch"},
    },
    "required": ["patch", "x", "y"],
    "additionalProperties": False,
}

INSTRUCTION = (
    "You are locating a UI element in a screenshot.\n"
    "You are given a low-detail view of the whole screen, then {n} overlapping "
    "patches of that same screen at full resolution, numbered 0 to {last} in "
    "reading order (left to right, top to bottom).\n"
    "Find the element, decide which patch shows it most completely, and give its "
    "centre as x and y in a 0-1000 coordinate system *within that patch* "
    "(0 = left/top edge of the patch, 1000 = right/bottom edge of the patch).\n"
    "Always answer, even if uncertain.\n\n"
    "Element: "
)


def _b64(im: Image.Image, max_edge: int | None = None) -> str:
    im = im.convert("RGB")
    if max_edge and max(im.size) > max_edge:
        s = max_edge / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def make_patches(im: Image.Image, grid=GRID, overlap=OVERLAP) -> list[tuple[Image.Image, tuple[float, float, float, float]]]:
    """Split into an overlapping grid. Returns (patch, normalized box) per tile."""
    rows, cols = grid
    W, H = im.size
    pw, ph = W / cols, H / rows
    ox, oy = pw * overlap, ph * overlap
    out = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, c * pw - ox)
            y0 = max(0, r * ph - oy)
            x1 = min(W, (c + 1) * pw + ox)
            y1 = min(H, (r + 1) * ph + oy)
            out.append((im.crop((round(x0), round(y0), round(x1), round(y1))),
                        (x0 / W, y0 / H, x1 / W, y1 / H)))
    return out


def build_content(im: Image.Image, instruction: str) -> tuple[list[dict], list[tuple], list[tuple[int, int]]]:
    patches = make_patches(im)
    content: list[dict] = [
        {"type": "text", "text": INSTRUCTION.format(n=len(patches), last=len(patches) - 1) + instruction},
        {"type": "text", "text": "Low-detail view of the whole screen:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                     "data": _b64(im, FULL_VIEW_MAX_EDGE)}},
    ]
    boxes, sizes = [], []
    for i, (patch, box) in enumerate(patches):
        content.append({"type": "text", "text": f"Patch {i} (full resolution):"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": _b64(patch)}})
        boxes.append(box)
        sizes.append(patch.size)
    return content, boxes, sizes


def to_global(patch_idx: int, x: int, y: int, boxes: list[tuple]) -> tuple[float, float]:
    """Map a within-patch 0-1000 point back to whole-image normalized coordinates."""
    idx = max(0, min(patch_idx, len(boxes) - 1))
    x0, y0, x1, y1 = boxes[idx]
    return x0 + (x / 1000.0) * (x1 - x0), y0 + (y / 1000.0) * (y1 - y0)


def run_one(client, ex, budget: Budget, thinking_budget: int) -> dict:
    im = Image.open(ex.images[0])
    content, boxes, sizes = build_content(im, ex.question)
    rec = {"uid": ex.uid, "dataset": ex.dataset, "gold": ex.gold, "meta": ex.meta,
           "mode": "tiled", "grid": list(GRID), "patch_sizes": sizes,
           "source_size": list(im.size), "thinking_budget": thinking_budget}
    try:
        t0 = time.monotonic()
        resp = client.messages.create(
            model=MODEL,
            max_tokens=thinking_budget + 2048,
            thinking={"type": "enabled", "budget_tokens": thinking_budget},
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": content}],
        )
        budget.add(resp.usage.input_tokens, resp.usage.output_tokens)
        text = next((b.text for b in resp.content if b.type == "text"), None)
        rec.update({"raw": text, "stop_reason": resp.stop_reason,
                    "latency_s": round(time.monotonic() - t0, 2),
                    "usage": {"input_tokens": resp.usage.input_tokens,
                              "output_tokens": resp.usage.output_tokens}})
        obj = json.loads(text)
        rec["patch"] = obj["patch"]
        rec["pred"] = list(to_global(obj["patch"], obj["x"], obj["y"], boxes))
        rec["score"] = point_in_bbox(tuple(rec["pred"]), ex.gold)
    except Exception as e:
        rec.update({"pred": None, "score": None, "error": f"{type(e).__name__}: {e}"})
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description="Tiled-patch grounding experiment")
    p.add_argument("--dataset", default="screenspot_pro")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thinking-budget", type=int, default=2000)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--max-spend", type=float, default=4.0)
    a = p.parse_args()

    # Same seed as the baseline run, so the comparison is on the same questions.
    examples = load(a.dataset)
    examples = random.Random(a.seed).sample(examples, min(200, len(examples)))[: a.limit]

    out = RESULTS / f"{a.dataset}__tiled{GRID[0]}x{GRID[1]}_r0.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            try:
                r = json.loads(line)
                if r.get("pred") is not None:
                    done.add(r["uid"])
            except Exception:
                pass
    todo = [e for e in examples if e.uid not in done]
    print(f"{a.dataset}: {len(examples)} selected, {len(done)} done, {len(todo)} to run -> {out}")

    client, budget, lock = anthropic.Anthropic(), Budget(a.max_spend), threading.Lock()
    n = 0
    with open(out, "a") as fh, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futs = [pool.submit(run_one, client, e, budget, a.thinking_budget) for e in todo]
        for f in as_completed(futs):
            rec = f.result()
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n += 1
                if n % 10 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} | ${budget.spent:.3f}", flush=True)
            if budget.exhausted():
                print("  !! spend cap reached")
                break
    print(f"done | {budget.calls} calls | ${budget.spent:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
