# RUNME — Part 3: the finetuning recipe

Build the training data that would fix the localization defect Part 2 measured,
and the document that argues for it. This pipeline **constructs datasets**; it
does not train.

**Pipeline:** `finetune_data` — `ladder → build → verify → report`
**Reads:** Part 2's generator, `data/svgloc_mr/` · **Writes:** only where `--out` says

> ⚠️ Everything this pipeline builds is gitignored and none of it regenerates.
> Read [§0](#0-building-is-opt-in) before running it with `--out`.

## Launch

```bash
python -m blindspot.pipelines finetune_data --list                      # the read-only plan
python -m blindspot.pipelines finetune_data --all                       # audits the shipped ladder
python -m blindspot.pipelines finetune_data --all --out /tmp/ft --offline
python -m blindspot.pipelines finetune_data --stage build verify --out /tmp/ft
python -m blindspot.pipelines finetune_data --from report --out /tmp/ft --max-spend 0.50
```

Without `--out`, the plan is **1 step**: the ground-truth audit of the committed
ladder, which writes nothing you cannot rebuild. With `--out DIR` it is 4 stages,
8 steps — **1 of which calls the API** — and every one of them writes inside
`DIR`:

```
DIR/svgloc_mr/                    the ladder
DIR/sft_bbox/sft_bbox_20.jsonl    the SFT records
DIR/gallery_svgloc_mr.html        the audit gallery
DIR/gallery.html                  the record gallery
DIR/assets/                       figures + example strip
DIR/worked_examples.json          the model samples
DIR/part3.html, part3_paste.html  the rendered document
```

⚠️ **`report/worked-examples` calls the API.** `blindspot.report_worked` samples
the model — 3 prompts × 8 samples ≈ 24 calls — to compute GRPO group statistics
over real samples. Everything else in this pipeline is offline. The step is
flagged `needs_api=True`, so `--offline` skips it and `--max-spend` is required
whenever it is selected:

```bash
python -m blindspot.pipelines finetune_data --all --out /tmp/ft --offline          # everything except worked-examples
python -m blindspot.pipelines finetune_data --all --out /tmp/ft --max-spend 0.50   # including it
```

`report_worked` now owns a `--max-spend` of its own (default `$0.50`) and prices
every call against it **before** issuing the next one, so the ceiling stops it
mid-run rather than after the fact; the step declares a `spend`, so the
pipeline's ceiling reaches it too. It is serial by construction — one request in
flight — so the worst overshoot is a single call. A run cut short says so, marks
the short groups `truncated`, and an entirely empty run writes nothing rather
than replacing the file with `[]`.

**Its output is model-sampled**, so `--seed` selects *which* records are asked
about but does not make the artifact reproducible — it is the one output of this
pipeline that cannot be regenerated offline, and three consecutive runs produced
three different files.

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

## 0. Building is opt-in

Six artifacts here are **reference artifacts**: they are what the shipped
`part3.md` quotes and shows, they are gitignored and untracked, and rebuilding
them is not the same as restoring them.

| artifact | why a rerun is not a restore |
|---|---|
| `data/svgloc_mr/` | a fresh `ladder` emits the same 1,513 uids with a different `target_text`/`box_px` for **91%** of them |
| `data/sft_bbox/sft_bbox_20.jsonl` | `samples --n 20 --seed 0` reproduces **5** of the 20 shipped uids |
| `outputs/finetune/gallery.html` | drawn from those records |
| `outputs/finetune/worked_examples.json` | model samples; not reproducible at all |
| `outputs/part3/assets/` | figures and the example strip `part3.md` embeds |
| `outputs/part3/part3.html` | the rendered deliverable |

`python -m blindspot.pipelines finetune_data --all` used to overwrite **every one
of them**, and it is the command this page gave. It no longer can:

* **Writing is opt-in.** Without `--out`, the steps that write a reference
  artifact are not scheduled at all — the plan is the ground-truth audit and
  nothing else. This is the rule `synth_localization_eval` already followed for
  its `generate` stage.
* **An `--out` that lands on one is refused**, by name and with the reason. The
  check is on the paths the steps would be *given*, not on the base alone, so
  `--out outputs/part3` is caught by the `assets` it would write:

```
$ python -m blindspot.pipelines finetune_data --out outputs/part3 --list
refusing --out outputs/part3: the assets it would write lands on

    outputs/part3/assets

which is the figures and example strip part3.md embeds. It is gitignored and
untracked, so there is no copy to restore, and rerunning this pipeline does
not reproduce it. Point --out at a scratch directory instead, e.g. --out
/tmp/finetune_dev.
See docs/runme/FINETUNE.md section 0.                                   [exit 1]
```

The sections below give each step directly. They show a scratch `--out`; that is
not decoration. `generate_finetune samples` has always required `--out` with no
default so that a bare run could not clobber the reference file — and the
pipeline then handed it that exact path back, which is how the protection was
worth nothing. Run these against a scratch directory and compare, rather than
building over the reference set.

---

## 1. Generate the resolution / aspect ladder

Part 2's set is one canvas shape (1500×950, aspect 1.58) at three scales, of
which **only the smallest reaches the model untouched** — the other two are
downscaled by the API, so they measure the same delivered pixels twice.

Fine for evaluation. As training data it teaches one frame shape and one size —
and the defect being fixed is precisely that the model does not treat coordinates
as independent of the frame it is given.

```bash
python -m blindspot.generate_finetune ladder --out /tmp/ft/svgloc_mr
python -m blindspot.generate_finetune ladder --scenes-per-aspect 4 --out /tmp/ft/svgloc_mr_dev
```

`--out data/svgloc_mr` rebuilds the shipped ladder in place — see §0.

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
python -m blindspot.generate_finetune samples --n 20 --seed 0 --out /tmp/ft/sft_bbox_20.jsonl
```

Each record is one image + one question + the box the model should return.
`samples` calls `ExactInk` to get the pixel-exact target.

`--out` is **required, with no default**, so a bare run cannot write anywhere.
That protects the CLI, and only the CLI: the pipeline used to pass
`data/sft_bbox/sft_bbox_20.jsonl` in as the argument, which put the reference
file back in the line of fire on every `--all`. Both halves are fixed now — see
§0 — but the lesson is the general one: a required argument is not a guard if a
caller hardcodes the dangerous value.

Current output: `data/sft_bbox/sft_bbox_20.jsonl`, 20 records, 28KB.

---

## 3. Verify — make the ground truth checkable by eye

```bash
python -m blindspot.generate_finetune audit --dataset data/svgloc_mr          # read-only
python -m blindspot.generate_finetune audit --dataset /tmp/ft/svgloc_mr --out /tmp/ft/gallery.html
python -m blindspot.report_finetune gallery --records /tmp/ft/sft_bbox_20.jsonl \
                                            --out /tmp/ft/gallery.html
```

`report_finetune gallery` still **defaults** `--records` and `--out` to the
shipped pair, so pass both when you are working on a scratch set. The pipeline
always passes both.

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
python -m blindspot.report_finetune figures  --dataset /tmp/ft/svgloc_mr --out-dir /tmp/ft/assets
python -m blindspot.report_finetune examples --dataset /tmp/ft/svgloc_mr --out-dir /tmp/ft/assets
python -m blindspot.report_finetune examples --inject         # splices into part3.md itself
python -m blindspot.report_worked --dataset /tmp/ft/svgloc_mr \
       --out /tmp/ft/worked_examples.json --max-spend 0.50    # CALLS THE API
python -m blindspot.render_markdown --src outputs/part3/part3.md \
       --out /tmp/ft/part3.html --paste
```

Every one of these defaults to a path under `outputs/part3/` or
`outputs/finetune/` — convenient for the shipped build, destructive for a trial
run. Pass the flags.

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
| **`worked_examples.json` is not reproducible** | boxes come from the model; three consecutive runs produced three different files. It is the only step here that spends money. It now carries its own `--max-spend` (default `$0.50`), checked before every call, and declares a `spend` so the pipeline's ceiling reaches it — a run that hits the ceiling stops there and marks the short groups `truncated`. |
| **The ladder is not chained to the build** | `generate_finetune samples` reads `data/svg_localization/manifest.jsonl`, never `data/svgloc_mr`. So SFT records come from Part 2's single 900×570 canvas — the monoculture §1 says the ladder exists to fix. The ladder feeds only `audit`, `report_finetune {figures,examples}` and `report_worked`. Verify the intent before relying on the stage diagram. |
| **`data/svgloc_mr` is undocumented** | no README beside the data, absent from `data/README.md` and `docs/DATASETS.md`. |
| **Split outputs** | training data to `data/{svgloc_mr,sft_bbox}/`, galleries and worked examples to `outputs/finetune/`, figures and prose to `outputs/part3/`. Deliberate — training data is data, pages are output — but it means the pipeline's artifacts are in four places. |

One more is **fixed**: `finetune_data --all` no longer rebuilds this pipeline's
six reference artifacts in place. Writing is opt-in behind `--out`, and an
`--out` that resolves onto one of them is refused by name — [§0](#0-building-is-opt-in).

Three older issues are also **fixed**: this code now lives in the
one `blindspot` package, so it is inside `tests/test_all.py`'s structural sweep
and its two `sys.path` shims are gone; it is installed with the package; and
`generate_finetune audit` no longer writes to a fixed gallery path regardless of
`--dataset`.
