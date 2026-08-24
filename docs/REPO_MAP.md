# Repository map

Every module in `blindspot/`, its subcommands, its size, and what it does. One flat
package: there are no subpackages, and everything with a `__main__` is one of the
files below. Line counts are `wc -l`.

The layout rationale is [STRUCTURE.md](STRUCTURE.md); what to run in what order is
[runme/](runme/).

---

## The foundation

| module | loc | CLI | what it is |
|---|---:|---|---|
| `core.py` | 2225 | `python -m blindspot.core --datasets X --max-spend N` | Everything the rest builds on, and it imports none of them. Dataset adapters (one `Example` per scoreable question, gold boxes always normalized to `[0,1]` as `(x0,y0,x1,y1)`), prompt construction and structured-output schemas keyed by `answer_type`, the self-metering resumable API runner, the per-benchmark scorers, stratified sampling, Wilson intervals and grid helpers, the cross-dataset primitive taxonomy, failure-mode classification, and the Mermaid parser FlowLearn's gold needs. Its CLI is the main evaluation entry point. |
| `charxiv.py` | 595 | — (constants only) | CharXiv's published prompts and the seven judge rubrics, reproduced verbatim so our numbers stay comparable to their paper. Do not edit to taste. Previously mis-named `vendor`. |

Adding a dataset means writing one generator function in `core.py` and registering
it in `ADAPTERS`. Eleven are registered:

```
charxiv  infographicvqa  ai2d  slidevqa  slidevqa_allpages  screenspot  screenspot_pro
flowlearn_sim  svg_localization  svg_counting  svg_word_mc
```

## Getting data

| module | loc | subcommands | what it does |
|---|---:|---|---|
| `download.py` | 550 | `hf` · `screenspot-pro` · `flowlearn` · `flowlearn-full` · `flowlearn-subset` · `github-sources` | Six benchmark pullers. `hf` is the generic `load_dataset` path; the rest exist because ScreenSpot-Pro (per-app JSON + images), FlowLearn (two subsets, two layouts) and the GitHub-hosted sets are not `load_dataset`-able. Each writes `data/<name>/images/` plus a `manifest.jsonl`. |
| `generate.py` | 2918 | `scenes` · `questions` · `audit` · `examples` · `examples-derived` | The synthetic dataset. `scenes` procedurally builds charts and diagrams with exact text placement as ground truth, deterministic given `--seed`, drawing with Pillow and emitting matching SVG from the same primitive list so raster and vector cannot drift. `questions` derives the counting and word-presence sets from the *existing* scenes, re-rendering nothing. `audit` draws gold boxes on the images from the manifest alone. `examples` / `examples-derived` are browsable question/answer pages. |
| `generate_finetune.py` | 762 | `ladder` · `samples` · `audit` (+ class `ExactInk`) | Builds the Part 3 training data; never trains, never calls the API. `ladder` renders six aspect ratios × four sizes, every one delivered exactly as rendered. `ExactInk` recovers pixel-exact boxes by rendering each scene twice — with the label and without — and taking the pixels that changed, because the manifest's `text_ink_bbox` is PIL's *layout* box and clips glyph overhang. `samples` writes the SFT records: the supervision target is a box, not a point. |

## Calling the API

| module | loc | subcommands | what it does |
|---|---:|---|---|
| `run_api.py` | 1241 | `official` · `ablations` · `probe` · `derived` · `controls` · `grid` · `coord-probe` | Experiment drivers over `core`'s runner, for arms the plain runner does not cover. `official` runs ScreenSpot / ScreenSpot-Pro under the benchmark's own published protocol so the number is leaderboard-comparable (and `--rescore` recomputes it from saved responses with no API calls). `controls` and `grid` isolate *why* an arm fails. `probe` and `coord-probe` put a stronger model on identical inputs as a harness check, not a model comparison. |
| `judge.py` | 590 | `charxiv` · `equiv` · `gt-audit` | LLM-judge grading — model calls rather than string comparison, so non-deterministic and metered. `charxiv` is CharXiv's official protocol; any CharXiv number quoted next to the published one has to come through it, because string matching is a lower bound on the free-text types. `gt-audit` shows the judge the image and asks whether the *benchmark* is wrong: this is how the contested-gold floor was measured. |

## Runs into numbers

Reads `results/*.jsonl`, writes JSON. No HTML — keeping rendering out is what makes
the numbers independently checkable and lets a report rebuild in seconds without
re-scoring.

| module | loc | subcommands | what it does |
|---|---:|---|---|
| `eval.py` | 2119 | `aggregate` · `localization` · `derived` · `ablations` · `annotate` · `tiling` | One artifact per subcommand: `outputs/summary.json`, `outputs/svgloc/summary.json`, `outputs/svgderived/summary.json`, `outputs/svgloc/ablations.json`. `localization` produces the precision ladder and the distance bands, following `data/svg_localization/EVAL.md`. Two stated exceptions to the layer's rule, both flagged in its `--help`: `annotate` also emits browsing galleries over its sidecar JSON, and `tiling` **calls the API** — a one-off asking whether interleaved native-resolution patches restore what downscaling destroys. |

## Numbers into pages

| module | loc | subcommands | what it does |
|---|---:|---|---|
| `report.py` | 3178 | `data` · `examples` · `tables` · `index` · `paste` · `all` · `summary` · `svgloc` · `svgderived` | The live chain is `data → examples → tables → index → paste`, and `all` runs it. `data` assembles every number the report quotes into `outputs/report/figures.json` — the single auditable artifact; nothing downstream computes its own statistics. `tables` also injects the seven tables into `blindspots.md` between `<!-- Tn -->` markers, leaving hand-written prose untouched. `paste` emits one self-contained HTML file whose tables paste into a document editor as editable tables. `summary` writes `outputs/report/summary.json`, which `data` **reads as a file** — a dependency that is easy to miss, so it is the first step of the `report` stage in `pipelines.py`. `svgloc` and `svgderived` are the standalone per-dataset pages. |
| `report_pages.py` | 7734 | `causes` · `drilldown` · `slidevqa` · `tasks` · `primitives` · `headline` · `candidates` | The standalone evidence pages — everything the report links out to rather than contains. `causes` writes one page per blind spot plus the index that ranks them; `drilldown` emits every number in an openable tree alongside a `.json` and `.csv` of the same; `tasks` writes one page per perceptual primitive; `primitives` and `headline` are the two overview pages, and both read a summary JSON **as a file**, so their writers must have run first. Each page recomputes its own numbers from `results/*.jsonl` rather than reading `figures.json`, which is what lets them disagree with the report — and on ScreenSpot-Pro they did, correctly. The eight legacy renderers merged here kept three separate `format_equivalent` variants because merging them would have changed three published numbers. |
| `report_finetune.py` | 695 | `gallery` · `figures` · `examples` | The Part 3 artefacts. One module because all three share one drawing rule: the outline is stroked strictly *outside* the box, since PIL renders a multi-pixel `rectangle` inward and makes a correct box look like it clips the text. Only the supervision target is ever drawn; the wider `accept_region` stays in the JSON where it cannot be mistaken for ground truth. |
| `report_worked.py` | 185 | no subcommands (`--dataset --prompts --samples --seed --out --max-spend`) | GRPO group statistics over real model samples. **Calls the API** — roughly 24 calls, capped by its own `--max-spend` (default `$0.50`), which is checked before each call; the module is serial, so the worst overshoot is one call. Its output is model-sampled, so `--seed` selects *which* records are asked about but does not make the artifact reproducible. |
| `render_markdown.py` | 382 | no subcommands (`--src --out --paste`) | Markdown → self-contained HTML for any document; `part3.html` is generated from `part3.md` and never hand-edited, so the two cannot drift. Its parser is deliberately small — headings, lists, tables, block quotes, fenced code, rules, inline bold/italic/code. No markdown library is installed and one is not worth adding. |

## Diagnostics

| module | loc | subcommands | what it does |
|---|---:|---|---|
| `diagnose.py` | 1428 | `failure-modes` · `coordinates` · `capability` · `gt-quality` · `annotate-probe` · `dataset-page` | One diagnostic question per subcommand, each writing a self-contained page. `failure-modes`: is a grounding miss a format problem, an instruction problem, or a vision problem? `coordinates`: annotated PNGs plus the coordinate-compression explainer — the one to read first for ScreenSpot-Pro. `capability`: which axis does UI grounding break along? `gt-quality`: ground-truth quality rates by failure mode and question type. `annotate-probe`: gold box against predicted click, drawn on the screenshots. `dataset-page`: what each dataset is and whether it turned out usable. |

## Plumbing

| module | loc | CLI | what it does |
|---|---:|---|---|
| `pipelines.py` | 488 | `python -m blindspot.pipelines <name> --list\|--all\|--stage` | All three pipelines in one file. A pipeline owns no logic — it is an ordered list of steps, each an invocation of one of the modules above with the arguments this effort wants. Keeping them together makes the differences visible on one screen. |
| `flow.py` | 198 | — (library) | The launcher framework behind `pipelines.py`: stage listing, `--dry-run`, `--from` resume, `--offline`, `--max-spend` split across API steps, optional-step tolerance, fail-fast with a resume hint. Steps that reach the API carry `needs_api=True`, which is what makes the offline skip and the spend gate work. |
| `tools.py` | 239 | `verify-install` · `compare` | Not pipeline stages. `verify-install` checks that every module imports, every CLI parses and the shipped dataset loads; `setup.sh` and `make verify` both call it. `compare` structurally diffs two JSON artifacts — the schema, not the values, because two runs over different sample sizes produce the same shape and different numbers. |

## The three pipelines

| pipeline | stages | feeds |
|---|---|---|
| `literature_eval` | download → run → controls → judge → eval → diagnose → report | `blindspots.md` §1–4 |
| `synth_localization_eval` | generate → audit → run → eval → report | `blindspots.md` §5–7 |
| `finetune_data` | ladder → build → verify → report | `part3.md` |

Both pipelines that build data are opt-in behind `--out`. `generate` in pipeline 2
is empty unless `--out DIR` is given: the committed dataset is the source of truth
and the pipeline refuses an `--out` that resolves to it. Pipeline 3 follows the
same rule for all six of its reference artifacts — without `--out` it schedules
only the read-only audit, and an `--out` that would land a step on one of them is
refused by name. See [runme/SYNTHETIC.md §0](runme/SYNTHETIC.md#0-do-not-regenerate-in-place)
and [runme/FINETUNE.md §0](runme/FINETUNE.md#0-building-is-opt-in).

## `tests/`

`tests/` — six files, offline and deterministic. No API calls, no downloads, no
dependency on `results/`. It pins:

| group | what it pins |
|---|---|
| scoring | every number in the study comes through here: the ANLS threshold boundary, ANLS normalization being deliberately stricter than the general one, token-F1 refusing substring containment, click-in-bbox on realistically tiny targets |
| stats | Wilson intervals, quantile binning and the coarse grid — pinned because these moved modules during the consolidation |
| dataset invariants | the adapter contract against the committed dataset: gold boxes normalized to `[0,1]` with `x0<x1`, no degenerate targets, resolvable image paths, the registry intact |
| structure | the class of bug that motivated the suite — `__file__`-relative roots that break when a module changes directory, modules writing into the source tree, the layering direction, `sys.path` shims coming back, and every pipeline step that reaches the API declaring `needs_api=True` |

The structural sweep is shape-based rather than a hand-kept list, because it was a
hand-kept list once and two `sys.path` shims lived undetected in the finetune
package the whole time the anti-shim test was passing.

## `legacy/`

The pre-consolidation modules, frozen. **Nothing imports them, nothing runs them,
they are not packaged, and `tests/test_all.py` excludes them from its sweep** —
they carry `parents[2]` roots and `sys.path` shims that were correct for the nested
layout they were written in, so sweeping them would fail the structural tests for
code that no longer executes.

They are there so a merged module can be traced back: `blindspot/core.py` is nine
files, `report.py` is eight, `generate.py` is five, and no filename survives to say
which one a given function came from. 70 files, 22,142 loc — every merged original,
plus the nine superseded renderers that were not carried forward at all. The seven
modules that moved one-to-one are deliberately absent, since git records those as
renames. File-by-file mapping: [legacy/README.md](../legacy/README.md).

## Known rough edges

- `report_pages.py` is 7,734 lines; `generate.py` and `report.py` are 2,918 and
  3,178. Both of the latter are unions of five
  and eight modules; each subcommand is still the original module's code and
  docstring, but neither file is pleasant to navigate.
- `report_worked.py` is the only Part 3 step that cannot be regenerated offline:
  the boxes come from the model, so no seed makes the artifact reproducible.
- `report.py`'s `gold_quality()` hardcodes measured constants. Fine for a frozen
  study, stale the moment the runs are repeated.
- `download github-sources` takes no arguments. It is in the `--help` sweep, but
  the underlying puller starts work on invocation, so do not "just check its usage"
  against the original in git.
