#!/usr/bin/env python
"""Pull capped, stratification-free samples of the datasets in DATASETS.md into data/.

Schema-agnostic: dumps every non-image field as-is into a JSONL manifest, and saves
any PIL.Image field to disk as a JPG, replacing that field with a relative path.
Uses streaming so we never materialize a full dataset on disk.
"""
import argparse
import itertools
import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image

DATASETS = [
    {"name": "infographicvqa", "repo": "lmms-lab/DocVQA", "config": "InfographicVQA", "split": "validation"},
    {"name": "chartqa", "repo": "HuggingFaceM4/ChartQA", "config": None, "split": "test"},
    {"name": "docvqa", "repo": "lmms-lab/DocVQA", "config": "DocVQA", "split": "validation"},
    {"name": "charxiv", "repo": "princeton-nlp/CharXiv", "config": None, "split": "validation"},
    # PlotQA dropped: the achang/plot_qa HF mirror is Donut OCR-training markup, not real
    # Q&A text -- the actual dataset lives on Google Drive with no small official subset
    # (test split alone is ~1.2M+ QA pairs). ChartQA + CharXiv already cover this primitive.
    {"name": "slidevqa", "repo": "NTT-hil-insight/SlideVQA", "config": None, "split": "test"},
    {"name": "screenspot", "repo": "rootsautomation/ScreenSpot", "config": None, "split": "test"},
    {"name": "rico_screenqa", "repo": "bevaya/RICO-ScreenQA", "config": None, "split": "test"},
    {"name": "livexiv", "repo": "LiveXiv/LiveXiv", "config": "v7-VQA", "split": "test"},
    # AI2D: grade-school science diagrams. DATASETS.md originally dropped it as
    # pre-2020, but it is the only remaining source of arrow/flow-following on real
    # diagrams once FlowLearn was excluded -- the primitive the brief names and that
    # CharXiv covers with a single question type.
    {"name": "ai2d", "repo": "lmms-lab/ai2d", "config": None, "split": "test"},
    # NOTE: FlowLearn and ScreenSpot-Pro are NOT load_dataset-able (raw JSON + images,
    # not parquet/imagefolder) -- handled by download_flowlearn.py / download_screenspot_pro.py.
]


def save_example(example, idx, out_dir):
    record = {}
    for key, value in example.items():
        if isinstance(value, Image.Image):
            img_dir = out_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = img_dir / f"{idx:05d}_{key}.jpg"
            value.convert("RGB").save(img_path, quality=90)
            record[key] = str(img_path.relative_to(out_dir))
        else:
            record[key] = value
    return record


def run_one(ds, n, out_root):
    n = min(n, ds["max_n"]) if ds.get("max_n") else n
    out_dir = out_root / ds["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"streaming": True}
    if ds["config"]:
        kwargs["name"] = ds["config"]

    print(f"=== {ds['name']} ({ds['repo']}) ===")
    try:
        stream = load_dataset(ds["repo"], split=ds["split"], **kwargs)
    except Exception as e:
        print(f"  FAILED to open: {type(e).__name__}: {e}")
        return

    manifest_path = out_dir / "manifest.jsonl"
    count = 0
    with open(manifest_path, "w") as f:
        try:
            for idx, example in enumerate(itertools.islice(stream, n)):
                try:
                    record = save_example(example, idx, out_dir)
                    f.write(json.dumps(record, default=str) + "\n")
                    count += 1
                except Exception as e:
                    print(f"  skip idx {idx}: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"  FAILED during iteration after {count} examples: {type(e).__name__}: {e}")

    print(f"  saved {count} examples -> {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200_000, help="max examples per dataset (per-dataset max_n overrides this down)")
    parser.add_argument("--out", default="data")
    parser.add_argument("--only", nargs="*", default=None, help="dataset name(s) to restrict to")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        if args.only and ds["name"] not in args.only:
            continue
        run_one(ds, args.n, out_root)


if __name__ == "__main__":
    main()
