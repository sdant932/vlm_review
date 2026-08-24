# Pipeline

Four stages. Each writes files the next one reads, so any stage can be re-run on its own.

```
  download / generate  ->  data/<dataset>/manifest.jsonl
                              |
                              v  blindspot.core  or  blindspot.run_api
                           results/<dataset>__<tag>.jsonl
                              |
                              v  blindspot.eval
                           outputs/**/summary.json
                              |
                              v  blindspot.report
                           outputs/report/
```

Every command below is run from the repository root with the environment active.
All three stages are wrapped by `blindspot.pipelines`, which is the shortest path
if you want the whole thing rather than one step — see
[STRUCTURE.md §3](STRUCTURE.md#3-running-a-pipeline) and the runbooks in
[runme/](runme/).

---

## 1. Get data

### The generated dataset — already committed

`data/svg_localization` ships with this repository. Nothing to do.

> ⚠️ **Do not regenerate it in place.** The generator has drifted from the
> committed set, and `results/*.jsonl` is keyed by uid, so a regeneration would
> bind existing answers to different questions. Build a *new* set into a *new*
> `--out` instead. Full detail:
> [runme/SYNTHETIC.md §0](runme/SYNTHETIC.md#0-do-not-regenerate-in-place).

```bash
python -m blindspot.generate scenes --count 200 --complexity 4 --seed 17 --out /tmp/svgloc_new
python -m blindspot.generate questions --data /tmp/svgloc_new   # counting + word_mc sets
```

The generator draws every scene with Pillow and emits a matching SVG from the same
geometry, so the vector source and the raster cannot drift. Ground truth is the text
placement it just computed, not an annotation — there is no labelling noise floor.

Audit what it produced before trusting it:

```bash
python -m blindspot.generate audit    --data /tmp/svgloc_new --open   # gold boxes on the images
python -m blindspot.generate examples --data /tmp/svgloc_new --open   # question/answer page
```

Both default to `--data data/svg_localization` if you leave the flag off, which is
the right thing when you are auditing the shipped set and the wrong thing when you
are auditing a new one.

### The public benchmarks — not redistributed here

```bash
python -m blindspot.download hf --only charxiv ai2d slidevqa infographicvqa
python -m blindspot.download screenspot-pro    # not load_dataset-able: per-app JSON
python -m blindspot.download flowlearn         # not load_dataset-able either
python -m blindspot.download github-sources    # extracts from clones in third_party/
```

Each writes `data/<name>/images/` plus a `manifest.jsonl` mapping every example to its
ground truth. Sizes and licences: [DATASETS.md](DATASETS.md).

Adapters registered in `blindspot/core.py`:

```
charxiv  infographicvqa  ai2d  slidevqa  slidevqa_allpages  screenspot  screenspot_pro
flowlearn_sim  svg_localization  svg_counting  svg_word_mc
```

## 2. Run the model

```bash
# the main runner: metered, resumable, stops hard at --max-spend
python -m blindspot.core --datasets svg_localization --max-spend 5

# a small pilot first, always
python -m blindspot.core --datasets charxiv --limit 20 --max-spend 0.50

# stratified rather than random, so per-cell n is something you chose
python -m blindspot.core --datasets charxiv --per-cell 250 --max-spend 20

# resolution ablation: hand the model fewer pixels
python -m blindspot.core --datasets screenspot_pro --max-edge 1568 --max-spend 5
```

Results append to `results/<dataset>__<tag>.jsonl`, keyed by uid. Re-running skips what
is already there, so a killed run resumes and costs only its in-flight requests.

Specialised drivers, for arms the main runner does not cover — all of
`blindspot.run_api`:

| | |
|---|---|
| `run_api official` | ScreenSpot / ScreenSpot-Pro under the benchmark's own published protocol, so the number is comparable to the leaderboard. `--rescore` recomputes it from saved responses with no API calls |
| `run_api derived` | the counting and word-presence sets, with their blind controls |
| `run_api probe` | harness sanity probe — run this before trusting a low score |
| `run_api ablations` | prompt-wording and answer-channel ablations on the point questions |
| `run_api controls --arm blind\|onepage` | controls that isolate *why* an arm fails |
| `run_api grid` | does the model fail to locate, or fail to say where? |
| `run_api coord-probe` | a stronger model on identical inputs, as a harness check |

**Run a blind control for anything you intend to call a perception failure.** Ask the
same question with the image withheld. If the score barely moves, the image was never
carrying the answer and the finding is about language, not vision.

## 3. Score and aggregate

```bash
python -m blindspot.eval aggregate         # -> outputs/summary.json
python -m blindspot.eval localization      # -> outputs/svgloc/summary.json
python -m blindspot.eval derived           # -> outputs/svgderived/summary.json
python -m blindspot.eval ablations         # -> outputs/svgloc/ablations.json
```

For CharXiv, string matching undercounts free-text answers. The published protocol uses
an LLM judge, so any CharXiv number quoted next to theirs must come through it:

```bash
python -m blindspot.judge charxiv --dataset charxiv --judge-model claude-opus-5 --max-spend 4
python -m blindspot.judge equiv --dataset infographicvqa --max-spend 2
```

Ground-truth quality — how much of the measured error is the benchmark being wrong
rather than the model:

```bash
python -m blindspot.judge gt-audit --dataset charxiv --per-category 5 --max-spend 2
python -m blindspot.diagnose gt-quality
```

Every `blindspot.judge` subcommand spends money. Set `--max-spend` on each.

## 4. Build the report

Five subcommands, in this order — or `python -m blindspot.report all`, which runs
exactly these five. `tables` both writes `tables.md` and injects the tables into
`blindspots.md` in place, between `<!-- Tn -->` markers; prose outside those markers
is hand-written and never touched.

```bash
python -m blindspot.report summary     # -> outputs/report/summary.json  (read by `data`)
python -m blindspot.report data        # -> outputs/report/figures.json
python -m blindspot.report examples    # -> outputs/report/figures/*.png
python -m blindspot.report tables      # -> outputs/report/tables.md
python -m blindspot.report index       # -> outputs/report/figures.md
python -m blindspot.report paste       # -> outputs/report/paste_into_docs.html
```

`summary` is a **file** dependency of `data`, not an import, which is easy to miss: run
it first or `data` reads a stale `outputs/report/summary.json`. The `report` stage of
`blindspot.pipelines literature_eval` runs it as its first step for that reason.

`figures.json` is the auditable bundle: every number the report quotes, in one file,
each traceable to `results/*.jsonl`. Nothing downstream computes its own statistics.
`paste_into_docs.html` is self-contained — open it, select all, paste into a document.

The two per-dataset pages are standalone rather than part of that chain:

```bash
python -m blindspot.report svgloc       # the generated localization set
python -m blindspot.report svgderived   # the counting and word-presence sets
```

## Diagnostics

```bash
python -m blindspot.diagnose failure-modes    # is a grounding miss format, instruction, or vision?
python -m blindspot.diagnose coordinates      # annotated PNGs + the coordinate-compression explainer
python -m blindspot.diagnose capability       # which axis does UI grounding break along?
python -m blindspot.diagnose annotate-probe   # gold box vs predicted click, on the screenshots
python -m blindspot.diagnose dataset-page     # what each dataset is, and whether it is usable
```

## Cost

The runner meters itself against `--max-spend` and aborts on a billing error rather than
retrying into one. A plain API key cannot read a credit balance — no organization
endpoint exposes it — so this ceiling is the only spend control there is. Set it on every
run. Full reasoning in [METHODOLOGY.md](METHODOLOGY.md).
