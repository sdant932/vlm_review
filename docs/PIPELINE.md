# Pipeline

Four stages. Each writes files the next one reads, so any stage can be re-run on its own.

```
  download / generate  ->  data/<dataset>/manifest.jsonl
                              |
                              v  scripts/run/  or  blindspot.core.runner
                           results/<dataset>__<tag>.jsonl
                              |
                              v  blindspot.analysis.*
                           outputs/**/summary.json
                              |
                              v  blindspot.reporting.*
                           outputs/report/
```

Every command below is run from the repository root with the environment active.

---

## 1. Get data

### The generated dataset — already committed

`data/svg_localization` ships with this repository. Nothing to do. To rebuild it from
scratch (deterministic given the seed):

```bash
python scripts/generate/gen_svg_localization.py --count 200 --complexity 4 --seed 17
python scripts/generate/gen_svg_derived.py          # counting + word_mc question sets
```

The generator draws every scene with Pillow and emits a matching SVG from the same
geometry, so the vector source and the raster cannot drift. Ground truth is the text
placement it just computed, not an annotation — there is no labelling noise floor.

Audit what it produced before trusting it:

```bash
python scripts/generate/verify_svg_localization.py --open      # gold boxes drawn on the images
python scripts/generate/examples_svg_localization.py --open    # browsable question/answer page
```

### The public benchmarks — not redistributed here

```bash
python scripts/download/download_datasets.py       # the HF-hosted ones
python scripts/download/download_screenspot_pro.py # not load_dataset-able: per-app JSON
python scripts/download/download_flowlearn.py      # not load_dataset-able either
python scripts/download/prepare_github_sources.py  # extracts from clones in third_party/
```

Each writes `data/<name>/images/` plus a `manifest.jsonl` mapping every example to its
ground truth. Sizes and licences: [DATASETS.md](DATASETS.md).

Adapters registered in `blindspot/core/adapters.py`:

```
charxiv  infographicvqa  ai2d  slidevqa  slidevqa_allpages  screenspot  screenspot_pro
flowlearn_sim  svg_localization  svg_counting  svg_word_mc
```

## 2. Run the model

```bash
# the main runner: metered, resumable, stops hard at --max-spend
python -m blindspot.core.runner --datasets svg_localization --max-spend 5

# a small pilot first, always
python -m blindspot.core.runner --datasets charxiv --limit 20 --max-spend 0.50

# stratified rather than random, so per-cell n is something you chose
python -m blindspot.core.runner --datasets charxiv --per-cell 250 --max-spend 20

# resolution ablation: hand the model fewer pixels
python -m blindspot.core.runner --datasets screenspot_pro --max-edge 1568 --max-spend 5
```

Results append to `results/<dataset>__<tag>.jsonl`, keyed by uid. Re-running skips what
is already there, so a killed run resumes and costs only its in-flight requests.

Specialised runners, for arms the main runner does not cover:

| | |
|---|---|
| `scripts/run/official_eval.py` | ScreenSpot / ScreenSpot-Pro under the benchmark's own published protocol, so the number is comparable to the leaderboard |
| `scripts/run/run_svg_derived.py` | the counting and word-presence sets |
| `scripts/run/svgloc_probe.py` | harness sanity probe — run this before trusting a low score |
| `scripts/run/svgloc_ablations.py` | prompt-wording and answer-channel ablations on the point questions |
| `scripts/run/controls.py` | blind / one-page / grid controls that isolate *why* an arm fails |
| `scripts/run/grid_control.py` | does the model fail to locate, or fail to say where? |
| `scripts/run/coord_probe.py` | a stronger model on identical inputs, as a harness check |

**Run a blind control for anything you intend to call a perception failure.** Ask the
same question with the image withheld. If the score barely moves, the image was never
carrying the answer and the finding is about language, not vision.

## 3. Score and aggregate

```bash
python -m blindspot.analysis.aggregate                 # -> outputs/summary.json
python -m blindspot.analysis.svgloc_eval               # -> outputs/svgloc/summary.json
python -m blindspot.analysis.svgderived_eval           # -> outputs/svgderived/summary.json
python -m blindspot.analysis.svgloc_ablation_eval      # -> outputs/svgloc/ablations.json
```

For CharXiv, string matching undercounts free-text answers. The published protocol uses
an LLM judge, so any CharXiv number quoted next to theirs must come through it:

```bash
python -m blindspot.judging.judge --dataset charxiv --judge-model claude-opus-5
python -m blindspot.judging.equiv_judge --dataset infographicvqa
```

Ground-truth quality — how much of the measured error is the benchmark being wrong
rather than the model:

```bash
python -m blindspot.judging.gt_audit --dataset charxiv --per-category 5
python scripts/analyze/analyse_gtaudit.py
```

## 4. Build the report

Four modules, in this order. `report_tables` both writes `tables.md` and injects the
tables into `blindspots.md` in place, between `<!-- Tn -->` markers; prose outside those
markers is hand-written and never touched.

```bash
python -m blindspot.reporting.report_data       # -> outputs/report/figures.json
python -m blindspot.reporting.report_examples   # -> outputs/report/figures/*.png
python -m blindspot.reporting.report_tables     # -> outputs/report/tables.md
python -m blindspot.reporting.report_index      # -> outputs/report/figures.md
python -m blindspot.reporting.report_paste      # -> outputs/report/paste_into_docs.html
```

`figures.json` is the auditable bundle: every number the report quotes, in one file,
each traceable to `results/*.jsonl`. Nothing downstream computes its own statistics.
`paste_into_docs.html` is self-contained — open it, select all, paste into a document.

Other renderers under `blindspot/reporting/` are earlier, superseded pages. They still
run, and some report figures trace back to them, which is why they are kept. See
[REPO_MAP.md](REPO_MAP.md).

## Diagnostics

```bash
python scripts/analyze/failure_analysis.py     # is a grounding miss format, instruction, or vision?
python scripts/analyze/coord_diagnostics.py    # annotated PNGs + the coordinate-compression explainer
python scripts/analyze/capability_report.py    # which axis does UI grounding break along?
python scripts/analyze/build_datasets_page.py  # what each dataset is, and whether it is usable
```

## Cost

The runner meters itself against `--max-spend` and aborts on a billing error rather than
retrying into one. A plain API key cannot read a credit balance — no organization
endpoint exposes it — so this ceiling is the only spend control there is. Set it on every
run. Full reasoning in [METHODOLOGY.md](METHODOLOGY.md).
