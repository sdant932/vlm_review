#!/usr/bin/env python
"""Pull the FULL FlowLearn simulated test sets (both variants), in parallel.

The earlier pull captured 150 of 4,000 images and only the `word` variant. That
matters because FlowLearn-sim is the only dataset here with usable ground truth
for arrow-following, which is one of the perceptual primitives the project is
supposed to measure.

Two variants, same 2,000 flowcharts each:
  word  (mermaid_word)      nodes labelled with nonsense words  -- "dihedron cushite"
  char  (mermaid_char_v2)   nodes labelled with random characters -- strictly harder

Neither can be shortcut with world knowledge, which is what makes them clean
probes: the model has to trace the arrow.

Each image carries four ground truths (Arrow_AtoB, Arrow_betweenAB, Num_Nodes,
Num_Arrows) plus the full Mermaid graph, so ~6 scoreable questions per image.

Uses snapshot_download with allow_patterns so the whole image directory comes
down in one parallel sweep rather than 4,000 sequential hf_hub_download calls.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

REPO = "jopan/FlowLearn"
VARIANTS = [("word", "mermaid_word"), ("char", "mermaid_char_v2")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    # HF returns 429 at 16 workers on this repo; 4 is what completes.
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="cap images per variant (default: all)")
    args = ap.parse_args()

    out_dir = Path(args.out) / "flowlearn_sim"
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    # Annotations first -- small, and they tell us which images we actually need.
    anns: dict[str, dict] = {}
    for variant, _ in VARIANTS:
        p = hf_hub_download(REPO, f"SimFlowchart/{variant}/VQA/test.json", repo_type="dataset")
        anns[variant] = json.loads(Path(p).read_text())
        print(f"[{variant}] {len(anns[variant]):,} annotated images")

    # One parallel snapshot per variant, restricted to that variant's jpegs.
    roots = {}
    for variant, subdir in VARIANTS:
        print(f"[{variant}] fetching images (workers={args.workers}) ...", flush=True)
        # snapshot_download resumes from the local cache, so a 429 mid-way costs
        # only the in-flight files; back off and re-enter rather than restarting.
        for attempt in range(args.retries):
            try:
                roots[variant] = Path(snapshot_download(
                    REPO, repo_type="dataset",
                    allow_patterns=[f"SimFlowchart/images/{subdir}/jpeg/*"],
                    max_workers=args.workers,
                ))
                break
            except Exception as e:
                if attempt == args.retries - 1:
                    raise
                wait = min(120, 5 * 2 ** attempt)
                print(f"  [{variant}] {type(e).__name__} -- retry {attempt+1}/{args.retries} in {wait}s", flush=True)
                time.sleep(wait)

    manifest = out_dir / "manifest.jsonl"
    written = missing = 0
    with open(manifest, "w") as f:
        for variant, subdir in VARIANTS:
            items = list(anns[variant].items())
            if args.limit:
                items = items[: args.limit]
            src_dir = roots[variant] / "SimFlowchart" / "images" / subdir / "jpeg"

            def copy_one(item):
                img_file, qa = item
                src = src_dir / img_file
                if not src.exists():
                    return None
                dest = f"{variant}_{img_file}"
                shutil.copy(src, img_out / dest)
                return {"variant": variant, "image": f"images/{dest}", **qa}

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for rec in pool.map(copy_one, items):
                    if rec is None:
                        missing += 1
                        continue
                    f.write(json.dumps(rec) + "\n")
                    written += 1
            print(f"[{variant}] done")

    print(f"\n[flowlearn_sim] wrote {written:,} rows -> {manifest}"
          f"{f'  ({missing} images missing upstream)' if missing else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
