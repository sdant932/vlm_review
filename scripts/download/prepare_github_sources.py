#!/usr/bin/env python
"""Extract data/manifests from the GitHub-only sources cloned into third_party/.

BlindTest (vision-llms-are-blind): images ship WITHOUT ground truth — the authors'
generation notebooks embed the answer in the filename (e.g. "gt_3_image_...png") but
the publicly released images were renamed before shipping, stripping it. There is no
metadata.json or labels file in the repo either. So `ground_truth` is left null here;
recovering it requires either re-running (a port of) the generation notebooks, or
manual annotation. The `commonly_incorrect`-style per-model correct/incorrect folders
in the repo are a weak proxy (do NOT treat as ground truth) and are skipped here.

Ferret-UI (ml-ferret): playground/sample_data/ is a single illustrative example (1
image, 3 JSON files) shipped to document the training data *format* — Apple never
publicly released the actual 14-task eval benchmark from the paper. This script
copies that one example through for reference but it is NOT a usable eval set.
"""
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = REPO_ROOT / "third_party"
DATA = REPO_ROOT / "data"

PROMPTS = {
    "CircledWord": [
        "Which letter is being circled?",
        "Which character is being highlighted with a red oval?",
    ],
    "CountingCircles": [
        "How many {shapes} are in the image? Answer with only the number in numerical format.",
        "Count the {shapes} in the image. Answer with a number in curly brackets e.g. {{3}}.",
    ],
    "CountingRowsAndColumns": [
        "Count the number of rows and columns and answer with numbers in curly brackets. For example, rows={{5}} columns={{6}}",
        "How many rows and columns are in the table? Answer with only the numbers in a pair (row, column), e.g., (5,6)",
    ],
    "LineIntersection": [
        "Count the intersection points where the blue and red lines meet. Put your answer in curly brackets, e.g., {{2}}.",
        "How many times do the blue and red lines touch each other? Answer with a number in curly brackets, e.g., {{5}}.",
    ],
    "NestedSquares": [
        "Count total number of squares in the image. Answer with only the number in numerical format in curly brackets e.g. {{3}}.",
        "How many squares are in the image? Please answer with a number in curly brackets e.g., {{10}}.",
    ],
    "SubwayMap": [
        "How many single-colored paths go from {station1} to {station2}? Answer with a number in curly brackets, e.g., {{3}}",
        "Count the one-colored routes that go from {station1} to {station2}. Answer with a number in curly brackets, e.g., {{3}}",
    ],
    "TouchingCircle": [
        "Are the two circles touching each other? Answer with Yes/No.",
        "Are the two circles overlapping? Answer with Yes/No.",
    ],
}


def prepare_blindtest():
    src_root = THIRD_PARTY / "vision-llms-are-blind" / "src"
    out_dir = DATA / "blindtest"
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    count = 0
    with open(manifest_path, "w") as f:
        for task, prompts in PROMPTS.items():
            img_dir = src_root / task / "images"
            if not img_dir.exists():
                print(f"  [blindtest] skip {task}: no images/ dir")
                continue
            # Only take image files directly in images/, not the per-model
            # correct/incorrect subfolders (those are other models' results, not data).
            image_files = sorted(p for p in img_dir.glob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
            for img_path in image_files:
                dest_name = f"{task}_{img_path.name}"
                shutil.copy(img_path, img_out / dest_name)
                record = {
                    "task": task,
                    "image": f"images/{dest_name}",
                    "prompts": prompts,
                    "ground_truth": None,
                    "note": "ground truth not shipped publicly (see script docstring)",
                }
                f.write(json.dumps(record) + "\n")
                count += 1
    print(f"[blindtest] saved {count} images across {len(PROMPTS)} tasks -> {manifest_path}")
    print("[blindtest] ground_truth is null for all rows -- see script docstring before scoring")


def prepare_ferret_ui():
    src_root = THIRD_PARTY / "ml-ferret" / "ferretui" / "playground"
    out_dir = DATA / "ferret_ui"
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    count = 0
    with open(manifest_path, "w") as f:
        for img_path in (src_root / "images").glob("*"):
            shutil.copy(img_path, img_out / img_path.name)
        for json_path in (src_root / "sample_data").glob("*.json"):
            examples = json.loads(json_path.read_text())
            for ex in examples:
                ex["_source_file"] = json_path.name
                ex["_NOTE"] = "illustrative example only -- NOT the real Ferret-UI eval benchmark"
                f.write(json.dumps(ex) + "\n")
                count += 1
    print(f"[ferret_ui] saved {count} example rows (from {img_out}) -> {manifest_path}")
    print("[ferret_ui] WARNING: this is Apple's single illustrative data-format example,")
    print("[ferret_ui] not the actual 14-task eval benchmark from the paper -- that was never publicly released.")


if __name__ == "__main__":
    prepare_blindtest()
    prepare_ferret_ui()
