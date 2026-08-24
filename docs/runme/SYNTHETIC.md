# RUNME — Part 2: the generated dataset

Build a procedural chart/diagram dataset where every confound is a knob, ask
three question types about **the same 200 scenes**, and score them.

**Pipeline:** `synth_localization_eval` — `generate → audit → run → eval → report`
**Reads/writes:** `data/svg_localization/` · `results/svg_*` · `outputs/svgloc/`

> ⚠️ `data/svg_localization` is committed and is the source of truth. Read
> [§0](#0-do-not-regenerate-in-place) before running the generator.

## Launch

```bash
python -m blindspot.pipelines synth_localization_eval --list
python -m blindspot.pipelines synth_localization_eval --all --max-spend 12
python -m blindspot.pipelines synth_localization_eval --task counting --stage run eval --max-spend 3
python -m blindspot.pipelines synth_localization_eval --stage generate audit --out /tmp/svgloc_new
python -m blindspot.pipelines synth_localization_eval --from eval --offline
```

`--list` prints 4 stages, 11 steps; 4 of them call the API. The `generate` stage
is **empty unless `--out DIR` is given** — see §0 — so with `--out` it is 5 stages
and 13 steps.

`--task` selects any of the three question types (`localization`,
`text_existence`, `counting`) and prunes every stage to just what that task
needs. `--max-spend` is required when API steps are selected.

**Deliverable:** sections 5–7 of `blindspots.md`. That document is a
separate deliverable and is not in this repository, so `outputs/` is empty
on a fresh clone and the final injection step has nothing to write into.
That is expected. What this pipeline produces — the generated dataset,
where the clicks land, and the separation of localization from recognition.
Sections 1–4 come from [BENCHMARKS.md](BENCHMARKS.md).

---

## Why this dataset exists

ScreenSpot-Pro measures localization but confounds three things at once. This
set separates them, because the layout is generated rather than scraped:

| confound | how this set separates it |
|---|---|
| target size | `target_area_frac` recorded per question |
| image size | the same scene emitted at three sizes |
| perception vs. coordinate emission | three question types over identical pixels |

Ground truth is the text placement the generator just computed — not a human
annotation. There is **no labelling noise floor to subtract**.

### The three question types

| set | asks | answer type | scored by |
|---|---|---|---|
| **localization** | point at this element | `point` | click-in-bbox |
| **word presence** | which of these four words appears? | `choice` | multiple_choice |
| **counting** | how many bars / rows / nodes? | `count` | count_score (signed error) |

The finding is the contrast between them. Same images, same scenes:

| task | small | large | blind control | n |
|---|---:|---:|---:|---:|
| word presence | 99.73% | 100.00% | 22.6% | 736 |
| counting | 94.12% | 97.06% | 12.0% | 476 |
| pointing | 6.68% | 4.41% | — | 1,587 |

The model reads the text and counts its occurrences near-perfectly, then cannot
say where it is. The blind controls confirm the first two rows depend on the
image rather than on a language prior.

---

## 0. Do not regenerate in place

`data/svg_localization` ships committed, and **it is the source of truth** —
every published number was computed against it. The generator has since drifted
from it: the same command with the same seed yields 4,724 questions against the
committed 4,723, changes ground truth on 73% of shared uids, and rebinds some
uids to a different question entirely:

```
svgloc:0015:medium:00   committed 'Lead Referral'    regenerated 'Nordics'
```

`results/*.jsonl` is keyed by uid, so regenerating over a set you have already
scored would join model answers to the wrong questions.

**Rule:** the generator is for building a *new* set into a *new* `--out`
directory. To evaluate the study's dataset, skip §1 entirely — it is already on
disk.

The pipeline enforces this. `--out data/svg_localization` is refused outright:

```
$ python -m blindspot.pipelines synth_localization_eval --out data/svg_localization --list
refusing --out data/svg_localization: that is the committed dataset and the source
of truth for every published number. The generator has drifted from it, so
regenerating in place would rebind uids to different questions.
See docs/runme/SYNTHETIC.md section 0.                                     [exit 1]
```

The bare subcommands do **not** enforce it — `python -m blindspot.generate scenes`
defaults `--out` to `data/svg_localization`, and `make dataset` calls it that way.
Always pass `--out` yourself.

## 1. Generate

The dataset ships committed, so a fresh clone can skip this entirely.

```bash
python -m blindspot.pipelines synth_localization_eval --stage generate --out /tmp/svgloc_new
```

or the steps directly:

```bash
python -m blindspot.generate scenes \
       --count 200 --complexity 4 --seed 17 --out /tmp/svgloc_new
python -m blindspot.generate questions --data /tmp/svgloc_new
```

The generator draws every scene with Pillow and emits a matching SVG **from the
same primitive list**, so the vector source and the raster cannot drift apart.

Verified on a 6-scene slice: 141 questions, 6 chart types, 4 themes, all targets
meeting WCAG AA for their size. Full 200-scene run: ~35s, 16 chart types, 10
themes, 9 font families.

`generate questions` re-renders nothing. It reads `scenes.jsonl` (semantic
content) and `manifest.jsonl`, and writes new questions against the images
already on disk — so all three question sets ask about identical pixels.

Useful `scenes` flags: `--list-types`, `--types bar_chart,flowchart`, `--scales`,
`--no-images`, `--min-contrast`, `--min-font-px`.

---

## 2. Audit — before trusting any score

```bash
python -m blindspot.generate audit            --data /tmp/svgloc_new --open
python -m blindspot.generate examples         --data /tmp/svgloc_new --per-type 4 --open
python -m blindspot.generate examples-derived --data /tmp/svgloc_new --open
```

All three default to `--data data/svg_localization`. `audit` and `examples` also
take `--out`; the pipeline points them at `<data>/verify` and `<data>/examples`.

`audit` positions every overlay from `manifest.jsonl` alone — it reads
`gold_bbox_px` and `image_px`, converts to percentages, and lets the browser
place the box. It deliberately does **not** re-derive geometry from the layout
code: if the overlay were drawn by the same routine that produced the gold box,
a bug would cancel itself out and the audit would show a perfect fit over wrong
coordinates.

What to look for:
- every green box sits tightly around its text, at all three resolutions
- the red crosshair (point questions) sits inside its box
- grid cell labels agree with where the box falls on the 4×4 rule
- `reverse` probe points land on the text they name

Verified on the 6-scene slice: `consistency errors: 0 (all checks passed)`.

`examples` is not the audit — it shows a reader exactly what the model is
asked and exactly what counts as right, with `POINT_INSTRUCTION` prepended as it
would be sent. Sending the manifest's element description bare turns
localization into an unanswerable fragment.

---

## 3. Run the model

```bash
python -m blindspot.pipelines synth_localization_eval --stage run --max-spend 12

# localization only
python -m blindspot.core --datasets svg_localization --max-spend 5

# counting + word presence, with their blind controls
python -m blindspot.run_api derived --datasets svg_counting svg_word_mc --max-spend 3
python -m blindspot.run_api derived --skip-blind --max-spend 2
```

**Probe the harness before believing a low score.**

```bash
python -m blindspot.run_api probe --rung small --n 100 --max-spend 1
```

Ablations — is the failure the prompt, the answer channel, or the perception?

```bash
python -m blindspot.run_api ablations --n 300 --max-spend 8
```

Eight arms: `repeat, careful, describe, cell_then_point, landmark, crop, bbox,
quadrant_mc`.

---

## 4. Evaluate

```bash
python -m blindspot.eval localization    # → outputs/svgloc/summary.json
python -m blindspot.eval derived         # → outputs/svgderived/summary.json
python -m blindspot.eval ablations       # → outputs/svgloc/ablations.json
```

Verified output:

```
svgloc_eval    2380 point / 2340 text / 3 unusable
  small   n=793  click-in-bbox  6.68% [5.15-8.64]  chance 0.2554%  26.2x
  medium  n=793  click-in-bbox  4.67% [3.40-6.36]  chance 0.2533%  18.4x
  large   n=794  click-in-bbox  4.41% [3.19-6.07]  chance 0.2555%  17.3x

svgloc_ablation_eval  baseline n=300  6.67%  2x2 66.3%
  cell_then_point  19.00%  +12.33pp  chi2 24.45  SIG
  quadrant_mc      80.33%  +14.00pp  chi2 20.50  SIG
  bbox              1.33%   -5.33pp  chi2 10.23  SIG
```

### Traps that make this easy to score wrong

Read `data/svg_localization/EVAL.md` first. It is the load-bearing document —
13 Python call sites cite it as the source of a rule.

1. **The pairing trap.** `medium` and `large` are delivered to the model at the
   *same* dimensions after the API downscale, so `medium` vs `large` is a **null
   control**, not a resolution comparison. Comparing the wrong pair silently
   compares different questions.
2. **The precision ladder is one set of predictions read at six tolerances**, not
   six experiments. Do not report the rows as independent.
3. **Counting uses signed error**, not absolute — the direction of the miss is
   the finding.
4. **Do not compare absolute scores to ScreenSpot-Pro's 1.65%.** This set shares
   the method but not the task. Its target floor is ~23× larger.

> **Note:** with `results/` absent, `blindspot.eval localization` reports
> `0 point / 0 text` and **writes an n=0 summary over a good one**, exit 0. Back
> up `outputs/svgloc/summary.json` before running it on an incomplete tree.

---

## 5. Report

```bash
python -m blindspot.report svgloc        # self-contained HTML + summary
python -m blindspot.report svgderived
```

Both take `--tag` to point at a non-default results tag; `svgloc` also takes
`--no-images` for a text-only page.
