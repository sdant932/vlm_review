#!/usr/bin/env python
"""ScreenSpot-Pro isn't in a load_dataset-able format (raw per-app JSON annotations +
a matching images/ folder, not parquet/imagefolder), so it needs its own downloader.
Samples up to --per-app entries from each application's annotation file (for
diversity across the 23 professional apps) and pulls only the referenced images,
rather than the full 1619-file repo.
"""
import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "likaixin/ScreenSpot-Pro"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-app", type=int, default=500, help="max examples per application")
    parser.add_argument("--out", default="data/screenspot_pro")
    args = parser.parse_args()

    out_dir = Path(args.out)
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    annotation_files = sorted(f for f in files if f.startswith("annotations/") and f.endswith(".json"))

    manifest_path = out_dir / "manifest.jsonl"
    count = 0
    with open(manifest_path, "w") as out_f:
        for ann_file in annotation_files:
            local_ann = hf_hub_download(REPO, ann_file, repo_type="dataset")
            entries = json.loads(Path(local_ann).read_text())
            for entry in entries[: args.per_app]:
                img_rel = entry["img_filename"]
                try:
                    local_img = hf_hub_download(REPO, f"images/{img_rel}", repo_type="dataset")
                except Exception as e:
                    print(f"  skip {img_rel}: {type(e).__name__}: {e}")
                    continue
                dest_name = img_rel.replace("/", "_")
                shutil.copy(local_img, img_out / dest_name)
                record = dict(entry)
                record["image"] = f"images/{dest_name}"
                out_f.write(json.dumps(record) + "\n")
                count += 1
            print(f"  {ann_file}: pulled {min(len(entries), args.per_app)} examples")

    print(f"[screenspot_pro] saved {count} examples across {len(annotation_files)} apps -> {manifest_path}")


if __name__ == "__main__":
    main()
