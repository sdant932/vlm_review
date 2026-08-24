# Datasets

The dataset plan for this study, updated to record what was actually used. The shortlist
below was drawn up first; **What was actually scored** further down says which of it
survived contact with the data.


## Task

Characterize Claude Haiku 4.5's blind spots on perception tasks relevant to business
applications: reading charts/graphs, diagrams/flowcharts, documents, presentation
slides, and UI screens. Decompose these into underlying perceptual primitives
(localization, arrow/line-following, counting, spatial relations) and test across a
range of datasets to find where Haiku does well vs. poorly.

Target model is accessed via the **Anthropic API** (not run locally) — any agent
picking this up can swap in its own API key/client; the datasets and harness below
are provider-agnostic until the actual scoring call.

## Environment

- Python 3.11+, installed with `./setup.sh`. `environment.yml` builds the same thing
  as a conda env named `blindspot`. Full detail: [../SETUP.md](../SETUP.md).
- Runtime deps are three — `anthropic`, `pillow`, `numpy` — plus `datasets` and
  `huggingface_hub` in the `download` extra, which only `python -m blindspot.download`
  needs. The `openai` / `opencv` / `torch` / `transformers` set named in the original
  plan was never imported and is not installed.
- API keys go in `.env` (copy from `.env.example`): `ANTHROPIC_API_KEY`, and
  `OPENAI_API_KEY` only if you swap the CharXiv judge back to GPT-4o.
- `blindspot.download hf` uses `load_dataset` with `streaming=True` and `.take(N)`
  rather than full downloads (see Practical notes below).

## Machine constraints (checked 2026-08-21)

- Disk: 826GB free of 926GB total — not a constraint even downloading every dataset
  below in full (~85–140GB worst case).
- RAM: 48GB — irrelevant for API-based eval (one image at a time); only matters if
  a local VLM is added for comparison later.

---

## Final dataset shortlist

★ = highest priority

### Charts & Infographics

| Dataset | Date | Size | HF id / source | Goal |
|---|---|---|---|---|
| **InfographicVQA** ★ | 2021 | 5,485 images / 30,035 QA | `lmms-lab/DocVQA` (config `InfographicVQA`) | Real infographics: charts + dense text + icons + layout together — closest single dataset to what a business "reads" in practice |
| ChartQA | 2022 | ~21.9K images / ~32.7K QA | `HuggingFaceM4/ChartQA` | Precise value reading, bar/line comparison on real-world charts |
| CharXiv | 2024 (NeurIPS) | 2,323 charts / 10K+ QA | `princeton-nlp/CharXiv` | Fixes ChartQA's main weakness — real scientific charts (not template-generated), descriptive vs. reasoning question splits |
| LiveXiv | 2024 (ICLR 2025) | ~2K–11K rows per monthly snapshot | `LiveXiv/LiveXiv` | Continuously refreshed from new arXiv papers — use to rule out training-data contamination as an explanation for scores on the static benchmarks above |

### Diagrams & Flowcharts

| Dataset | Date | Size | HF id / source | Goal |
|---|---|---|---|---|
| FlowLearn | 2024 | 3.8K real + 10K simulated flowcharts | `jopan/FlowLearn` | Arrow-following via ground-truth Mermaid graph structure (which box connects to which) |
| BlindTest | 2024 | 7 tasks, small/extensible | GitHub: `anguyen8/vision-llms-are-blind` (not on HF — clone + run generator) | Confound-free line-intersection/tracing primitive; paper already reports a Claude 3.5 Sonnet baseline (74%) to compare against |
| **AI2D** ★ | 2016 | 4,903 diagrams / 15,501 MC questions | `lmms-lab/ai2d` | Added after the initial plan, and pre-2020 in spite of the scope rule. Labelled science diagrams whose questions often name a printed mark (an arrow, a letter, a callout) — the only source that cleanly separates *reading a diagram* from *matching a printed label to the thing it points at*. That binding operation turned out to be one of the study's findings, which is what earned it the exception. |

### Documents & Presentations

| Dataset | Date | Size | HF id / source | Goal |
|---|---|---|---|---|
| DocVQA | 2020 | 12,767 images / 50,000 QA | `lmms-lab/DocVQA` (config `DocVQA`) | Structured document reading — forms, memos, tables |
| SlideVQA | 2023 (AAAI) | 2.6K decks / 52K slide images / 14.5K QA | `NTT-hil-insight/SlideVQA` | Real slide-deck QA — multi-slide evidence retrieval + numerical reasoning |

### UI Screens

| Dataset | Date | Size | HF id / source | Goal |
|---|---|---|---|---|
| ScreenSpot-Pro | 2025 | 1,581 test cases, 23 professional apps | GitHub/HF: `likaixin2000/ScreenSpot-Pro-GUI-Grounding` | Element localization on realistic high-res business software (CAD, IDEs, office apps) |
| ScreenSpot(-v2) | 2024 | ~1,200+ instructions | `rootsautomation/ScreenSpot` | Baseline UI element grounding (mobile/desktop/web) |
| RICO-ScreenQA | 2022 | 35K screens / 86K QA | `rootsautomation/RICO-ScreenQA` | General screen reading/counting |
| Ferret-UI | 2024 | 14 UI tasks, iPhone/Android splits | GitHub: `apple/ml-ferret` (eval data in `playground/sample_data/`, not cleanly on HF) | Most decomposed UI perception test — icon recognition, widget listing, find-text as separate "elementary" tasks, plus referring/grounding/reasoning. **License: CC BY-NC 4.0 (non-commercial only)** |

### Excluded (considered, then dropped)

- **BLINK, CLEVR, TallyQA, VSR, RefCOCO** — natural-photo/template-limited primitive
  benchmarks, out of scope once the task narrowed to document/flowchart/presentation/UI.
- **FigureQA, DVQA** — synthetic, single-chart-type or yes/no-only, superseded by
  ChartQA/CharXiv for this purpose.
- **PlotQA** — dropped. The `achang/plot_qa` HF mirror turned out to be Donut
  OCR-training markup, not real question/answer text. The actual dataset is hosted
  on Google Drive with no small official eval subset (its test split alone pairs
  33,657 images with ~1.2M+ QA pairs) — a proper fix means a custom Drive downloader
  plus self-sampling, which isn't worth it since ChartQA + CharXiv already cover the
  chart-value-reading primitive with real QA and clean HF hosting.

---

## What was actually scored

Five benchmarks, 13,965 questions, all on `claude-haiku-4-5-20251001` with thinking at
2,000 tokens.

| Dataset | Items scored | Metric | Result |
|---|---:|---|---:|
| CharXiv | 5,000 | judged exact match | 84.7% |
| AI2D | 3,086 | accuracy | 81.6% |
| SlideVQA | 1,003 | ANLS | 68.8% |
| InfographicVQA | 2,801 | ANLS | 66.7% |
| ScreenSpot-Pro | 1,581 | click-in-bbox | 1.8% |

Plus a sixth scored arm — SlideVQA all-pages, n=494, 58.5% — which is a retrieval control
rather than a separate benchmark, and the synthetic dataset generated for this study
(`data/svg_localization`: 2,380 localization, 476 counting and 736 word-presence
questions).

Pulled and prepared but **not scored**: ChartQA, DocVQA, RICO-ScreenQA, LiveXiv,
FlowLearn, ScreenSpot-v2, BlindTest, Ferret-UI. These ran out of time budget rather than
being ruled out — their adapters and download scripts work. LiveXiv is the one to pick up
first: it was the intended contamination check and nothing replaced it.

BlindTest has **no usable ground truth** in its public release. The shipped image
filenames had the answer-bearing prefix stripped before release, confirmed by reading its
generator notebooks. It is usable for qualitative probing as-is; scoring it needs either
porting the generation logic to produce fresh labelled images, or annotating the ~30
images by hand.

## Adapters

`blindspot/core.py` registers eleven:

```
charxiv  infographicvqa  ai2d  slidevqa  slidevqa_allpages  screenspot  screenspot_pro
flowlearn_sim  svg_localization  svg_counting  svg_word_mc
```

Adding a twelfth means writing one generator function that yields `Example` records and
registering it in `ADAPTERS`. Gold boxes are normalized to `[0,1]` as `(x0,y0,x1,y1)`
there, whatever encoding the source used.

## Practical notes

- **Prefer the standard eval split over bulk-downloading.** Each dataset's official
  `test`/`validation` split is the "known split" to score against — not the full
  train+val+test union, and not an arbitrary manual cap. Verify true split sizes via
  HF's datasets-server `/size` endpoint before assuming a number from a paper abstract
  (those often quote the full dataset, not the split you'd actually use).
- **Use `streaming=True`** with `datasets.load_dataset` and `.take(N)` to avoid
  materializing full downloads when a dataset's real eval split is still large.
- **Watch API-side image resizing.** The Anthropic API downsizes/tiles large images
  before the model sees them — for localization/counting/arrow-tracing tasks, log
  the as-sent resolution/token count so a "failure" can be distinguished from a
  resizing artifact rather than a genuine perception gap.
- **Contamination check.** CharXiv/LiveXiv/Ferret-UI/FlowLearn/ScreenSpot-Pro are all
  2024+ — plausibly newer than some training cutoffs but not guaranteed to be
  unseen. LiveXiv's rolling monthly snapshots are the cleanest way to sidestep this;
  worth prioritizing if contamination is a concern.
- **Pair primitives with business tasks.** E.g. if a counting/arrow-tracing primitive
  (BlindTest, FlowLearn) shows a specific failure, go confirm it shows up concretely
  in ChartQA/InfographicVQA/ScreenSpot — that's what makes a blind-spot finding land
  as a business-relevant result rather than an abstract CV benchmark score.

## Suggested starting order

1. InfographicVQA (top priority)
2. ChartQA + CharXiv (real charts, two difficulty tiers)
3. FlowLearn + BlindTest (arrow/line-following)
4. DocVQA + SlideVQA (documents/presentations)
5. ScreenSpot(-Pro) + Ferret-UI (UI localization, two granularities)
6. RICO-ScreenQA, LiveXiv as budget/time allows
