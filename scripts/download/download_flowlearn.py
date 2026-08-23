#!/usr/bin/env python
"""FlowLearn isn't a load_dataset-able format either. Two subsets, two layouts:

- SciFlowchart (real arxiv figures): SciFlowchart/all.json has caption/OCR/image_file
  per entry; images live at SciFlowchart/images/<image_file>. No Mermaid/arrow QA here.
- SimFlowchart (procedurally generated): SimFlowchart/word/VQA/test.json has the real
  prize -- Arrow_AtoB / Arrow_betweenAB (arrow-direction true/false pairs), Num_Nodes,
  Num_Arrows, and full Flowchart-to-Mermaid graph ground truth, keyed by filename.
  Matching images are at SimFlowchart/images/mermaid_word/jpeg/<filename> (note: not
  the naively-guessed SimFlowchart/word/... or .../images/... path).
"""
import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "jopan/FlowLearn"


def prepare_sci(n, out_root):
    out_dir = out_root / "flowlearn_sci"
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    local_ann = hf_hub_download(REPO, "SciFlowchart/all.json", repo_type="dataset")
    entries = json.loads(Path(local_ann).read_text())

    manifest_path = out_dir / "manifest.jsonl"
    count = 0
    with open(manifest_path, "w") as f:
        for entry in entries[:n]:
            img_file = entry.get("image_file")
            if not img_file:
                continue
            try:
                local_img = hf_hub_download(REPO, f"SciFlowchart/images/{img_file}", repo_type="dataset")
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
            local_ann = hf_hub_download(REPO, f"SimFlowchart/{variant}/VQA/test.json", repo_type="dataset")
            entries = json.loads(Path(local_ann).read_text())
            variant_count = 0
            for img_file, qa in list(entries.items())[:n]:
                try:
                    local_img = hf_hub_download(REPO, f"SimFlowchart/images/{img_subdir}/jpeg/{img_file}", repo_type="dataset")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    out_root = Path(args.out)
    prepare_sci(args.n, out_root)
    prepare_sim(args.n, out_root)
