#!/usr/bin/env python
"""Fetch only the FlowLearn simulated images the stratified run actually needs.

`snapshot_download` pulls every file matching the pattern -- 10,000 jpegs -- and
HuggingFace 429s this repo well before that finishes, even at 4 workers. But the
study samples 300 questions per cell, and all six question families share the
same image, so ~350 images per variant saturates every cell. Fetching those
directly is a 6x smaller download and stays under the rate limit.

Already-cached files return instantly, so this resumes whatever the earlier
snapshot attempts managed to pull.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "jopan/FlowLearn"
VARIANTS = [("word", "mermaid_word"), ("char", "mermaid_char_v2")]


def fetch(repo_path: str, retries: int = 5) -> str | None:
    for attempt in range(retries):
        try:
            return hf_hub_download(REPO, repo_path, repo_type="dataset")
        except Exception as e:
            if "429" not in str(e) and "Too Many" not in str(e):
                return None
            time.sleep(min(60, 3 * 2**attempt))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-variant", type=int, default=350)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    out_dir = Path(args.out) / "flowlearn_sim"
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    rows, missing = [], 0
    for variant, subdir in VARIANTS:
        ann = json.loads(Path(fetch(f"SimFlowchart/{variant}/VQA/test.json")).read_text())
        items = list(ann.items())[: args.per_variant]
        print(f"[{variant}] fetching {len(items)} images (workers={args.workers}) ...", flush=True)

        def one(item):
            img_file, qa = item
            local = fetch(f"SimFlowchart/images/{subdir}/jpeg/{img_file}")
            if not local:
                return None
            dest = f"{variant}_{img_file}"
            shutil.copy(local, img_out / dest)
            return {"variant": variant, "image": f"images/{dest}", **qa}

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, rec in enumerate(pool.map(one, items), 1):
                if rec is None:
                    missing += 1
                else:
                    rows.append(rec)
                if i % 100 == 0:
                    print(f"  [{variant}] {i}/{len(items)}", flush=True)

    with open(out_dir / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[flowlearn_sim] {len(rows)} rows written"
          f"{f' ({missing} unavailable)' if missing else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
