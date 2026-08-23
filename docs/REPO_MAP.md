# Repository map

Every module, what it does, and whether it is part of the current pipeline.

Nothing has been deleted. The study ran over about 36 hours and several analyses were
superseded by later ones; those are kept because some report figures still trace back to
them and because they document what was tried. The tags say which is which:

| tag | meaning |
|---|---|
| **live** | part of the current download → run → score → report chain |
| **standalone** | works, produces a real diagnostic, not in the current chain |
| **superseded** | replaced by a later module; kept for provenance |
| **one-off** | a single experiment, run once, answer recorded |

---

## `blindspot/core` — the foundation

Everything else depends on this. Nothing in it imports upward.

| module | loc | | |
|---|---:|---|---|
| `adapters.py` | 607 | **live** | Normalizes each dataset's manifest into a common `Example` record. One `Example` is one scoreable question, which is not the same as one manifest row. Gold boxes are always normalized to `[0,1]` as `(x0,y0,x1,y1)` whatever the source used. Adding a dataset means adding a function here. |
| `prompts.py` | 241 | **live** | Prompt construction and response schemas, keyed by `answer_type`. Answers come back through structured outputs rather than a regex over free text. |
| `runner.py` | 352 | **live** | The API runner: self-metering, resumable, safe to kill. Entry point for most evaluation. |
| `scoring.py` | 246 | **live** | Per-dataset scorers, each following that benchmark's own published metric so numbers stay comparable. |
| `sampling.py` | 83 | **live** | Stratified sampling by the cell you intend to report on. |
| `taxonomy.py` | 118 | **live** | One perceptual-primitive taxonomy across datasets — the spine that lets a chart finding and a UI finding be compared. |
| `failure_modes.py` | 148 | **live** | Classifies *why* an answer scored wrong: wrong value, right value wrong format, right items wrong order, and so on. |
| `stats.py` | 67 | **live** | Wilson intervals, quantile binning, and coarse-grid helpers. Shared by the analysis layer. |
| `mermaid.py` | 156 | **live** | Parses FlowLearn's Mermaid ground truth into a graph. Golds are re-derived from it rather than trusted from the shipped QA fields — see the note in `adapters.py`. Lives in `core` because `adapters` needs it. |
| `vendor/charxiv_constants.py` | — | **live** | CharXiv's own prompts and the seven judge rubrics, vendored verbatim. Do not edit to taste. |

## `blindspot/judging` — grading that costs money

Separate from `core.scoring` because these call a model rather than compare strings, so
they are non-deterministic and metered.

| module | loc | | |
|---|---:|---:|---|
| `judge.py` | 198 | **live** | CharXiv's official LLM-judge protocol. Any CharXiv number quoted next to the published one has to come through here — string matching is a lower bound on the free-text question types. |
| `equiv_judge.py` | 176 | **live** | Meaning-equivalence judge for span answers, plus failure-mode resolution. |
| `gt_audit.py` | 205 | **live** | Shows the judge the image and asks whether the *benchmark* is wrong. This is how the contested-gold floor was measured. |
| `gt_quality.py` | 111 | **live** | Rates how trustworthy each question's ground truth is, so scores can be reported against a known noise floor. |

## `blindspot/analysis` — runs into numbers

Reads `results/*.jsonl`, writes JSON. No HTML: keeping rendering out means the numbers
are independently checkable and a report rebuilds in seconds without re-scoring.

| module | loc | | |
|---|---:|---:|---|
| `aggregate.py` | 335 | **live** | Results → `outputs/summary.json`. |
| `svgloc_eval.py` | 374 | **live** | The generated localization set, following its `EVAL.md`. Produces the precision ladder and the distance bands. |
| `svgderived_eval.py` | 329 | **live** | The counting and word-presence sets derived from the same scenes. |
| `svgloc_ablation_eval.py` | 195 | **live** | Scores the prompt/answer-channel ablations against the baseline, paired per item. |
| `annotate.py` | 536 | standalone | A sidecar JSON per evaluated question, plus the image overlays. Feeds `task_pages`. |
| `tiling.py` | 202 | one-off | Does sending a screenshot as interleaved native-resolution patches restore the accuracy that downscaling destroys? Asked and answered. |

## `blindspot/reporting` — numbers into pages

The current report build is five modules in this order:

```
report_data  ->  report_examples  ->  report_tables  ->  report_index  ->  report_paste
```

| module | loc | | |
|---|---:|---:|---|
| `report_data.py` | 312 | **live** | Assembles every number the report quotes into `outputs/report/figures.json`. Nothing downstream computes its own statistics — this is the single auditable artifact. |
| `report_examples.py` | 445 | **live** | The real-image example figures. |
| `report_tables.py` | 320 | **live** | The report's six tables, straight from the measured JSON. Also injects them into `blindspots.md` between `<!-- Tn -->` markers, leaving hand-written prose untouched. |
| `report_index.py` | 119 | **live** | Orders the figures and resolves `[FIG:stem]` tokens to numbered references. Idempotent. |
| `report_paste.py` | 200 | **live** | One self-contained HTML file: markdown tables become real `<table>` elements and PNGs are inlined as base64, so a document editor pastes them as editable tables. |
| `aug22_summary.py` | 348 | **live** | Produces `outputs/aug22/summary.json`, which `report_data` reads. A file dependency rather than an import, which is easy to miss. |
| `report_candidates.py` | 328 | standalone | Contact sheet of four candidate example images per blind spot, for choosing figure panels. Served its purpose; kept so the choice can be revisited. |
| `cause_pages.py` | 3142 | standalone | Per-cause evidence pages. Nothing imports it any more — its six shared helpers moved to `core/stats.py`. |
| `svgloc_report.py` | 660 | standalone | Self-contained HTML report for the generated localization set. |
| `svgderived_report.py` | 382 | standalone | Same, for the counting and word-presence sets. |
| `drilldown_report.py` | 1651 | standalone | Hierarchical drill-down over every number in the study. |
| `slidevqa_report.py` | 1540 | standalone | The SlideVQA arm in detail. |
| `task_pages.py` | 329 | standalone | One page per perceptual primitive: what it tests, how it scores. |
| `aug22_report.py` | 383 | superseded | The corrected headline report, replaced by the `report_*` chain. |
| `report.py` | 746 | superseded | The first self-contained HTML report. Still holds `load_results`, which `judging/judge.py` imports. |
| `summary_report.py` | 375 | superseded | Rendered `outputs/report.html` from `outputs/summary.json`. |

## `scripts/download`

| script | loc | |
|---|---:|---|
| `download_datasets.py` | 101 | The generic Hugging Face puller, for the datasets that are `load_dataset`-able. |
| `download_screenspot_pro.py` | 57 | ScreenSpot-Pro is not: raw per-app JSON annotations plus images. |
| `download_flowlearn.py` | 93 | FlowLearn is not either — two subsets in two different layouts. |
| `download_flowlearn_full.py` | 113 | The full simulated test sets, both variants, in parallel. |
| `fetch_flowlearn_subset.py` | 84 | Only the images a stratified run actually needs. |
| `prepare_github_sources.py` | 115 | Extracts manifests from the repos cloned into `third_party/` (BlindTest, Ferret-UI). |

## `scripts/generate` — the synthetic dataset

| script | loc | |
|---|---:|---|
| `gen_svg_localization.py` | 1699 | Generates the whole dataset: procedural charts and diagrams with exact text placement as ground truth. Deterministic given `--seed`. Draws with Pillow and emits matching SVG from the same geometry, so raster and vector cannot drift. |
| `gen_svg_derived.py` | 271 | Derives the counting and word-presence question sets from the *existing* scenes, so all three tasks ask about the same images. |
| `verify_svg_localization.py` | 316 | Visual audit: gold boxes drawn on the images. Run this before trusting any score. |
| `examples_svg_localization.py` | 310 | Browsable page of example questions and answers. |
| `examples_svg_derived.py` | 209 | The same, for the derived sets. |

## `scripts/run` — everything that calls the API

| script | loc | |
|---|---:|---|
| `official_eval.py` | 311 | ScreenSpot / ScreenSpot-Pro under the benchmark's own published protocol, so the result is leaderboard-comparable. Deliberately differs from `core/runner.py`; the differences are listed in its docstring. |
| `run_svg_derived.py` | 120 | Runs the counting and word-presence sets. |
| `svgloc_probe.py` | 160 | Harness sanity probe. Run it before believing a low score is a capability result. |
| `svgloc_ablations.py` | 234 | Eight prompt-wording and answer-channel ablations on the point questions. |
| `controls.py` | 119 | Blind, one-page and grid controls that isolate why an arm fails. |
| `grid_control.py` | 108 | Does the model fail to locate, or fail to say where? |
| `coord_probe.py` | 87 | A stronger model on identical inputs, as a harness check rather than a model comparison. |

## `scripts/analyze` — diagnostics

| script | loc | |
|---|---:|---|
| `failure_analysis.py` | 292 | Is a grounding miss a format problem, an instruction problem, or a vision problem? |
| `coord_diagnostics.py` | 430 | Annotated PNGs plus a self-contained explainer for the coordinate-compression finding. |
| `capability_report.py` | 131 | Which axis does UI grounding break along? |
| `analyse_gtaudit.py` | 93 | Ground-truth quality rates by failure mode and question type. |
| `annotate_probe.py` | 250 | Draws ground truth against prediction directly onto the screenshots. |
| `build_datasets_page.py` | 230 | What each dataset is and whether it turned out usable. |

## `tests/`

Offline and deterministic — no API calls, no downloads, no dependency on `results/`.
179 tests, a few seconds on a fresh clone.

| file | what it pins |
|---|---|
| `test_scoring.py` | the scorers, which produce every number in the study: the ANLS threshold boundary, ANLS normalization being deliberately stricter than the general one, token-F1 refusing substring containment, click-in-bbox on realistically tiny targets |
| `test_stats.py` | Wilson intervals, quantile binning and the coarse grid — pinned because these moved packages during the reorganization |
| `test_dataset_invariants.py` | the adapter contract, against the committed dataset: gold boxes normalized to `[0,1]` with `x0<x1`, no degenerate targets, resolvable image paths, the registry intact |
| `test_repo_structure.py` | the class of bug that motivated the suite — `__file__`-relative roots that break when a module changes directory, packages writing into the source tree, the layering direction (`core` never imports upward), `sys.path` shims coming back |

`scripts/verify_install.py` is not a pipeline stage — it checks that every module imports,
every CLI parses and the shipped dataset loads. `setup.sh` and `make verify` both call it.

## Known rough edges

- `judging/judge.py` imports `load_results` from `reporting/report.py`, a superseded
  module. The dependency points the wrong way. It is a function-local import and works;
  moving `load_results` into `core` would be the clean fix.
- Seven scripts under `scripts/analyze/` and `scripts/download/` take no arguments and
  start work on invocation. `verify_install.py` skips them deliberately rather than
  running a full analysis to prove they parse.
- `cause_pages.py` and `drilldown_report.py` are 3,142 and 1,651 lines. Both are single
  HTML renderers that grew; neither is imported by anything now.
