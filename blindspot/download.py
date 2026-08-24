#!/usr/bin/env python
"""Fetch a dataset. One entry point per source, because no two arrive alike.

    python -m blindspot.download hf [--only NAME...] [--n N] [--out DIR]
    python -m blindspot.download screenspot-pro [--per-app N] [--out DIR]
    python -m blindspot.download flowlearn [--n N] [--out DIR]
    python -m blindspot.download flowlearn-full [--out DIR] [--workers N] [--retries N] [--limit N]
    python -m blindspot.download flowlearn-subset [--per-variant N] [--workers N] [--out DIR]
    python -m blindspot.download github-sources

`hf` is the generic Hugging Face puller, for the datasets that are actually
`load_dataset`-able. It is schema-agnostic: every non-image field is dumped as-is
into a JSONL manifest, and any PIL.Image field is saved to disk as a JPG with that
field replaced by a relative path. Streaming, so a full dataset is never
materialized on disk.

Everything else is here because it is *not* load_dataset-able:

  screenspot-pro     raw per-app JSON annotations plus a matching images/ folder.
                     Samples up to --per-app entries from each application (for
                     diversity across the 23 professional apps) and pulls only the
                     referenced images rather than the full 1619-file repo.

  flowlearn          two subsets in two different layouts. SciFlowchart (real arxiv
                     figures) has caption/OCR/image_file per entry and no arrow QA;
                     SimFlowchart (procedurally generated) carries the real prize --
                     Arrow_AtoB / Arrow_betweenAB, Num_Nodes, Num_Arrows and the full
                     Flowchart-to-Mermaid graph, keyed by filename, with images under
                     SimFlowchart/images/mermaid_word/jpeg/ (not the naively-guessed
                     SimFlowchart/word/... path).

  flowlearn-full     the FULL simulated test sets, both variants, in parallel. The
                     first pull captured 150 of 4,000 images and only the `word`
                     variant, which matters because FlowLearn-sim is the only dataset
                     here with usable arrow-following ground truth. Two variants over
                     the same 2,000 flowcharts: `word` (mermaid_word) labels nodes with
                     nonsense words, `char` (mermaid_char_v2) with random characters --
                     strictly harder, and neither shortcuttable with world knowledge.
                     Uses snapshot_download with allow_patterns so the image directory
                     comes down in one parallel sweep rather than 4,000 sequential
                     hf_hub_download calls.

  flowlearn-subset   only the simulated images a stratified run actually needs.
                     snapshot_download pulls every file matching the pattern -- 10,000
                     jpegs -- and HuggingFace 429s this repo well before that finishes,
                     even at 4 workers. But the study samples 300 questions per cell and
                     all six question families share the same image, so ~350 images per
                     variant saturates every cell: a 6x smaller download that stays
                     under the rate limit. Cached files return instantly, so this
                     resumes whatever the snapshot attempts managed to pull.

  github-sources     manifests from the GitHub-only repos cloned into third_party/.
                     BlindTest (vision-llms-are-blind): images ship WITHOUT ground truth
                     -- the authors' generation notebooks embed the answer in the
                     filename ("gt_3_image_...png") but the released images were renamed
                     before shipping, stripping it, and there is no metadata.json or
                     labels file either. So `ground_truth` is left null; recovering it
                     needs a port of the generation notebooks or manual annotation. The
                     `commonly_incorrect`-style per-model folders are a weak proxy (do
                     NOT treat as ground truth) and are skipped. Ferret-UI (ml-ferret):
                     playground/sample_data/ is a single illustrative example (1 image,
                     3 JSON files) shipped to document the training data *format* --
                     Apple never released the 14-task eval benchmark from the paper.
                     Copied through for reference; it is NOT a usable eval set.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from PIL import Image

FLOWLEARN_REPO = "jopan/FlowLearn"
FLOWLEARN_VARIANTS = [("word", "mermaid_word"), ("char", "mermaid_char_v2")]


# ====================================================================== hf
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


def cmd_hf(args) -> int:

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        if args.only and ds["name"] not in args.only:
            continue
        run_one(ds, args.n, out_root)
    return 0

# ========================================================= screenspot-pro
SSPRO_REPO = "likaixin/ScreenSpot-Pro"


def cmd_screenspot_pro(args) -> int:

    out_dir = Path(args.out)
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(SSPRO_REPO, repo_type="dataset")
    annotation_files = sorted(f for f in files if f.startswith("annotations/") and f.endswith(".json"))

    manifest_path = out_dir / "manifest.jsonl"
    count = 0
    with open(manifest_path, "w") as out_f:
        for ann_file in annotation_files:
            local_ann = hf_hub_download(SSPRO_REPO, ann_file, repo_type="dataset")
            entries = json.loads(Path(local_ann).read_text())
            for entry in entries[: args.per_app]:
                img_rel = entry["img_filename"]
                try:
                    local_img = hf_hub_download(SSPRO_REPO, f"images/{img_rel}", repo_type="dataset")
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
    return 0

# ============================================================== flowlearn


def prepare_sci(n, out_root):
    out_dir = out_root / "flowlearn_sci"
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    local_ann = hf_hub_download(FLOWLEARN_REPO, "SciFlowchart/all.json", repo_type="dataset")
    entries = json.loads(Path(local_ann).read_text())

    manifest_path = out_dir / "manifest.jsonl"
    count = 0
    with open(manifest_path, "w") as f:
        for entry in entries[:n]:
            img_file = entry.get("image_file")
            if not img_file:
                continue
            try:
                local_img = hf_hub_download(FLOWLEARN_REPO, f"SciFlowchart/images/{img_file}", repo_type="dataset")
            except Exception as e:
                print(f"  [flowlearn_sci] skip {img_file}: {type(e).__name__}: {e}")
                continue
            shutil.copy(local_img, img_out / img_file)
            record = {
                "image": f"images/{img_file}",
                "caption": entry.get("caption"),
                "imageText": entry.get("imageText"),
                "figType": entry.get("figType"),
            }
            f.write(json.dumps(record) + "\n")
            count += 1
    print(f"[flowlearn_sci] saved {count} examples -> {manifest_path}")


def prepare_sim(n, out_root):
    out_dir = out_root / "flowlearn_sim"
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    count = 0
    with open(manifest_path, "w") as f:
        # "char" variant uses random-character node labels instead of words -- a second,
        # harder read on the same arrow-following primitive. Both use their VQA *test*
        # split (eval-relevant); train/support splits are fine-tuning data, skipped here.
        for variant, img_subdir in [("word", "mermaid_word"), ("char", "mermaid_char_v2")]:
            local_ann = hf_hub_download(FLOWLEARN_REPO, f"SimFlowchart/{variant}/VQA/test.json", repo_type="dataset")
            entries = json.loads(Path(local_ann).read_text())
            variant_count = 0
            for img_file, qa in list(entries.items())[:n]:
                try:
                    local_img = hf_hub_download(FLOWLEARN_REPO, f"SimFlowchart/images/{img_subdir}/jpeg/{img_file}", repo_type="dataset")
                except Exception as e:
                    print(f"  [flowlearn_sim/{variant}] skip {img_file}: {type(e).__name__}: {e}")
                    continue
                dest_name = f"{variant}_{img_file}"
                shutil.copy(local_img, img_out / dest_name)
                record = {"variant": variant, "image": f"images/{dest_name}", **qa}
                f.write(json.dumps(record) + "\n")
                count += 1
                variant_count += 1
            print(f"  [flowlearn_sim/{variant}] {variant_count} examples")
    print(f"[flowlearn_sim] saved {count} examples -> {manifest_path}")


def cmd_flowlearn(args) -> int:
    out_root = Path(args.out)
    prepare_sci(args.n, out_root)
    prepare_sim(args.n, out_root)
    return 0

# ========================================================= flowlearn-full


def cmd_flowlearn_full(args) -> int:

    out_dir = Path(args.out) / "flowlearn_sim"
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    # Annotations first -- small, and they tell us which images we actually need.
    anns: dict[str, dict] = {}
    for variant, _ in FLOWLEARN_VARIANTS:
        p = hf_hub_download(FLOWLEARN_REPO, f"SimFlowchart/{variant}/VQA/test.json", repo_type="dataset")
        anns[variant] = json.loads(Path(p).read_text())
        print(f"[{variant}] {len(anns[variant]):,} annotated images")

    # One parallel snapshot per variant, restricted to that variant's jpegs.
    roots = {}
    for variant, subdir in FLOWLEARN_VARIANTS:
        print(f"[{variant}] fetching images (workers={args.workers}) ...", flush=True)
        # snapshot_download resumes from the local cache, so a 429 mid-way costs
        # only the in-flight files; back off and re-enter rather than restarting.
        for attempt in range(args.retries):
            try:
                roots[variant] = Path(snapshot_download(
                    FLOWLEARN_REPO, repo_type="dataset",
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
        for variant, subdir in FLOWLEARN_VARIANTS:
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

# ======================================================= flowlearn-subset


def fetch_with_retry(repo_path: str, retries: int = 5) -> str | None:
    for attempt in range(retries):
        try:
            return hf_hub_download(FLOWLEARN_REPO, repo_path, repo_type="dataset")
        except Exception as e:
            if "429" not in str(e) and "Too Many" not in str(e):
                return None
            time.sleep(min(60, 3 * 2**attempt))
    return None


def cmd_flowlearn_subset(args) -> int:

    out_dir = Path(args.out) / "flowlearn_sim"
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    rows, missing = [], 0
    for variant, subdir in FLOWLEARN_VARIANTS:
        ann = json.loads(Path(fetch_with_retry(f"SimFlowchart/{variant}/VQA/test.json")).read_text())
        items = list(ann.items())[: args.per_variant]
        print(f"[{variant}] fetching {len(items)} images (workers={args.workers}) ...", flush=True)

        def one(item):
            img_file, qa = item
            local = fetch_with_retry(f"SimFlowchart/images/{subdir}/jpeg/{img_file}")
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

# ========================================================= github-sources
REPO_ROOT = Path(__file__).resolve().parents[1]   # blindspot/download.py -> repo root
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


def cmd_github_sources(args) -> int:
    prepare_blindtest()
    prepare_ferret_ui()
    return 0


# ================================================================ dispatch

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m blindspot.download",
                                description="Fetch a dataset. One subcommand per source.")
    sub = p.add_subparsers(dest="cmd", metavar="SOURCE")

    hf = sub.add_parser("hf", help="generic Hugging Face puller (load_dataset-able sets)")
    hf.add_argument("--n", type=int, default=200_000,
                    help="max examples per dataset (per-dataset max_n overrides this down)")
    hf.add_argument("--out", default="data")
    hf.add_argument("--only", nargs="*", default=None, help="dataset name(s) to restrict to")
    hf.set_defaults(fn=cmd_hf)

    ssp = sub.add_parser("screenspot-pro", help="ScreenSpot-Pro: per-app JSON + images")
    ssp.add_argument("--per-app", type=int, default=500, help="max examples per application")
    ssp.add_argument("--out", default="data/screenspot_pro")
    ssp.set_defaults(fn=cmd_screenspot_pro)

    fl = sub.add_parser("flowlearn", help="FlowLearn: SciFlowchart + SimFlowchart samples")
    fl.add_argument("--n", type=int, default=20_000)
    fl.add_argument("--out", default="data")
    fl.set_defaults(fn=cmd_flowlearn)

    flf = sub.add_parser("flowlearn-full",
                         help="the full FlowLearn simulated test sets, both variants, in parallel")
    flf.add_argument("--out", default="data")
    # HF returns 429 at 16 workers on this repo; 4 is what completes.
    flf.add_argument("--workers", type=int, default=4)
    flf.add_argument("--retries", type=int, default=6)
    flf.add_argument("--limit", type=int, default=None, help="cap images per variant (default: all)")
    flf.set_defaults(fn=cmd_flowlearn_full)

    fls = sub.add_parser("flowlearn-subset",
                         help="only the simulated images a stratified run needs")
    fls.add_argument("--per-variant", type=int, default=350)
    fls.add_argument("--workers", type=int, default=3)
    fls.add_argument("--out", default="data")
    fls.set_defaults(fn=cmd_flowlearn_subset)

    gh = sub.add_parser("github-sources",
                        help="manifests from the repos cloned into third_party/ (takes no options)")
    gh.set_defaults(fn=cmd_github_sources)
    return p


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        p.print_help()
        return 2
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
