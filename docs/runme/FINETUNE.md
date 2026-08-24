# RUNME — Part 3: the finetuning recipe

Build the training data that would fix the localization defect Part 2 measured,
and the document that argues for it. This pipeline **constructs datasets**; it
does not train.

**Pipeline:** `finetune_data` — `ladder → build → verify → report`
**Reads:** Part 2's generator · **Writes:** `data/svgloc_mr/`, `data/sft_bbox/`, `outputs/finetune/`, `outputs/part3/`

## Launch

```bash
python -m blindspot.pipelines finetune_data --list
python -m blindspot.pipelines finetune_data --all --offline
python -m blindspot.pipelines finetune_data --stage build verify
python -m blindspot.pipelines finetune_data --from report --max-spend 0.50
```

`--list` prints 4 stages, 8 steps; **1 of them calls the API.**

⚠️ **`report/worked-examples` calls the API.** `blindspot.report_worked` samples
the model — 3 prompts × 8 samples ≈ 24 calls — to compute GRPO group statistics
over real samples. Everything else in this pipeline is offline. The step is
flagged `needs_api=True`, so `--offline` skips it and `--max-spend` is required
whenever it is selected:

```bash
python -m blindspot.pipelines finetune_data --all --offline          # everything except worked-examples
python -m blindspot.pipelines finetune_data --all --max-spend 0.50   # including it
```

`report_worked` takes no `--max-spend` of its own, so the framework gates it but
cannot cap it mid-run. **Its output is model-sampled**, so `--seed` selects
*which* records are asked about but does not make the artifact reproducible — it
is the one output of this pipeline that cannot be regenerated offline, and three
consecutive runs produced three different files.

**Deliverable:** `outputs/part3/part3.md` — the finetuning plan: a
multi-resolution encoder for the resolution defect, and SFT-on-boxes plus GRPO
with an IoU reward for localization.

---

## Why this is separate from the evaluation pipelines

Pipelines 1 and 2 measure the shipped model. This one builds data to **change**
it. Keeping them apart means a change to the training recipe cannot silently move
an evaluation number.

The one permitted edge is to Part 2's generator: `generate_finetune` rebuilds the
same procedural scenes by calling into `blindspot.generate` rather than
re-implementing them, so a change to a builder cannot make the training boxes and
the evaluation boxes disagree.

---

## The two design decisions worth knowing before you run anything

### 1. The supervision target is a **box**, not a point

Part 2 asked for a point and scored click-in-bbox. A point is a poor supervision
target: any point inside the target is equally correct, so the single point in
the manifest is one arbitrary choice out of thousands, and the model is penalised
for picking a different, equally valid one.

A box has one correct answer, and overlap with it is a **graded** signal rather
than a coin flip.

Which box matters. The manifest carries two and they are **not** interchangeable
— the region Part 2 accepted a click in is, for shape-held targets, the whole
enclosing node. That is the answer to "where is the widget", not "where is the
text". Only the supervision target is drawn in the gallery; the wider accepted
region stays in the JSON flagged `answers_the_question`, where it cannot be
mistaken for ground truth.

### 2. The recorded ink box is not the painted pixels

`text_ink_bbox` is PIL's `ImageDraw.textbbox` — the font's **layout** box
(advance widths, ascent, descent), not the pixels the rasteriser paints. Glyphs
overhang their advance width, so the recorded box clips the last glyph and the
bottom of the text.

Measured over 400 items: median overlap with the painted extent **0.899**, with
**18.5% below 0.75**. (`ExactInk.verify` re-measures it on the committed set and
reports 0.908 with a much thinner tail.)

Harmless for Part 2, which scored a click against the much wider hit box. Not
harmless when the ink box becomes a regression target — at these sizes overlap is
savage: a 30×9px box grown one pixel per side falls to **0.77**.

`ExactInk` fixes this **exactly**, by rendering the scene twice — once with
the label, once without — and taking the pixels that changed. An earlier fix
thresholded against the background colour inside a dilated window; that works
most of the time but has no principled stopping rule.

---

## 1. Generate the resolution / aspect ladder

Part 2's set is one canvas shape (1500×950, aspect 1.58) at three scales, of
which **only the smallest reaches the model untouched** — the other two are
downscaled by the API, so they measure the same delivered pixels twice.

Fine for evaluation. As training data it teaches one frame shape and one size —
and the defect being fixed is precisely that the model does not treat coordinates
as independent of the frame it is given.

```bash
python -m blindspot.generate_finetune ladder --out data/svgloc_mr
python -m blindspot.generate_finetune ladder --scenes-per-aspect 4 --out data/svgloc_mr_dev
```

Six aspect ratios; within each, four sizes from small up to the largest that
reaches the model **without** being downscaled. The caps are the ones the
generator already encodes: 1568px on the longest edge and ~1.15 megapixels —
whichever binds first (for 21:9 it is the edge, for most shapes the pixel count).

Current output: `data/svgloc_mr/` — 385 files, 12MB, manifest 1.1MB.

> ⚠️ `data/svgloc_mr` is documented only here — no README beside the data, absent
> from `data/README.md` and `docs/DATASETS.md`.

---

## 2. Build the training records

```bash
python -m blindspot.generate_finetune samples --n 20 --seed 0 --out data/sft_bbox/sft_bbox_20.jsonl
```

Each record is one image + one question + the box the model should return.
`samples` calls `ExactInk` to get the pixel-exact target.

`--out` is **required, with no default**: the committed `sft_bbox_20.jsonl` is a
reference artefact and a defaulted output path would overwrite it on a bare run.

Current output: `data/sft_bbox/sft_bbox_20.jsonl`, 20 records, 28KB.

---

## 3. Verify — make the ground truth checkable by eye

```bash
python -m blindspot.generate_finetune audit --dataset data/svgloc_mr
python -m blindspot.report_finetune gallery
```

Every record is shown twice — the whole image for context, and a zoom around the
target, because at 900×570 a target occupying 0.02% of the frame is a few pixels
and invisible at page scale. The annotation sits beside each pair exactly as
written to the JSONL, so a box that looks wrong on screen traces straight to the
field that produced it.

> The outline is drawn strictly **outside** the box. PIL renders a multi-pixel
> `rectangle` outline inward, which paints over the glyph rows at the box edge and
> makes a correct box look like it clips the text. An earlier version did exactly
> that and the boxes were wrongly blamed.

`audit`'s gallery path now **follows `--dataset`**, defaulting to
`outputs/finetune/gallery_<dataset>.html`, so auditing a throwaway set no longer
overwrites the shipped one. Override with `--out`.

Currently on disk: `outputs/finetune/gallery.html` (1.2MB) from
`report_finetune gallery`, and `outputs/finetune/gallery_multires.html` (1.9MB),
which `generate_finetune audit` produced under its old fixed name. A fresh audit
of `data/svgloc_mr` now writes `gallery_svgloc_mr.html` instead and leaves the
old file alone.

---

## 4. Render the document

```bash
python -m blindspot.report_finetune figures                  # explanatory figures
python -m blindspot.report_finetune examples                 # writes assets, prints markdown
python -m blindspot.report_finetune examples --inject        # also splices into part3.md
python -m blindspot.report_worked                            # CALLS THE API
python -m blindspot.render_markdown --src outputs/part3/part3.md --paste
```

`figures` produces two figures whose prose is hard to follow because the
trick is a difference of two renders — much easier to see than to read:

| figure | shows |
|---|---|
| `fig_box_extraction.png` | four panels: delivered image · same scene with no text · the pixels one label changed · the resulting box |
| `fig_frames.png` | the size/shape ladder — six aspect ratios, four sizes each, drawn to scale from a common corner |

`examples` selects **one record per aspect-ratio/size cell**, so the strip
shows the range of frames rather than ten views of the same shape, and biases to
the small end of each cell because that is where the model fails.

`render_markdown` generates HTML from the markdown and is never hand-edited, so
the two cannot drift. **Editing `part3.md` and re-running this is the whole
workflow.** Its markdown parser is deliberately small — headings, lists, tables,
block quotes, fenced code, rules, inline bold/italic/code. No markdown library is
installed and one is not worth adding.

`--paste` additionally writes `part3_paste.html`, the copy meant for pasting into
a document editor. The pipeline's `render` step passes it; without it you get
`part3.html` only, and the paste copy on disk goes stale.

Current output: `outputs/part3/{part3.md,part3.html,part3_paste.html}` +
`assets/ex01–ex09.png`.

---

## 5. What the document argues

`outputs/part3/part3.md` is hand-written prose with a machine-injected examples
section. In outline:

1. **Lever table** — pretraining a multi-resolution encoder vs post-training
   (SFT / GRPO).
2. **Resolution** — the measured InfoVQA gradient, 74.3 → 59.4.
3. **Localization** — three training-data sources; balancing tables by source,
   target size and background; the measured rule for partitioning SFT vs GRPO;
   the box-prediction task definition; the cross-entropy limitation and the
   soft-label ablation; GRPO with an **IoU reward**; dataset construction.

The size bins it quotes (0.039% – 2.751%) match the shipped manifest.

---

## 6. Known issues in this pipeline

| issue | detail |
|---|---|
| **`worked_examples.json` is not reproducible** | boxes come from the model; three consecutive runs produced three different files. It is also the only step here that spends money, and it carries no internal spend cap — the pipeline gates it, but cannot stop it mid-run. |
| **The ladder is not chained to the build** | `generate_finetune samples` reads `data/svg_localization/manifest.jsonl`, never `data/svgloc_mr`. So SFT records come from Part 2's single 900×570 canvas — the monoculture §1 says the ladder exists to fix. The ladder feeds only `audit`, `report_finetune {figures,examples}` and `report_worked`. Verify the intent before relying on the stage diagram. |
| **`data/svgloc_mr` is undocumented** | no README beside the data, absent from `data/README.md` and `docs/DATASETS.md`. |
| **Split outputs** | training data to `data/{svgloc_mr,sft_bbox}/`, galleries and worked examples to `outputs/finetune/`, figures and prose to `outputs/part3/`. Deliberate — training data is data, pages are output — but it means the pipeline's artifacts are in four places. |

Three issues that used to be listed here are **fixed**: this code now lives in the
one `blindspot` package, so it is inside `tests/test_all.py`'s structural sweep
and its two `sys.path` shims are gone; it is installed with the package; and
`generate_finetune audit` no longer writes to a fixed gallery path regardless of
`--dataset`.
