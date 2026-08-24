# Artifacts: what produces what

Every generated file, the exact command that writes it, and what it needs first.
Nothing here is committed — `outputs/` is gitignored in full, and so is every
`data/` directory except `data/svg_localization`. All of it is reproducible from
`results/` plus the datasets.

Read this when you have an artifact and want to know where it came from, or want
an artifact and don't know which command makes it. For *why* the pipeline is
shaped this way, see [STRUCTURE.md](STRUCTURE.md); for running a whole effort end
to end, see [runme/](runme/).

---

## The dependency spine

```
data/<dataset>/manifest.jsonl        download / generate
        |
        v   blindspot.core | blindspot.run_api          ($ spends)
results/<dataset>__<tag>.jsonl
        |
        v   blindspot.eval                              (JSON; `annotate` also renders galleries)
outputs/**/summary.json
        |
        v   blindspot.report                            (renders, computes nothing)
outputs/report/**
```

Two edges in that picture are **file** dependencies, not imports, and are the
usual cause of a confusing failure:

- `report data` reads `outputs/report/summary.json`. Run `report summary` first.
- `report index` reads `outputs/report/figures/`. Run `report examples` first.

---

## JSON — the numbers

| artifact | command | needs |
|---|---|---|
| `outputs/summary.json` | `python -m blindspot.eval aggregate` | `results/{charxiv,infographicvqa,screenspot_pro,ai2d}__*` |
| `outputs/svgloc/summary.json` | `python -m blindspot.eval localization` | `results/svg_localization__*` |
| `outputs/svgderived/summary.json` | `python -m blindspot.eval derived` | `results/svg_{counting,word_mc}__*` |
| `outputs/svgloc/ablations.json` | `python -m blindspot.eval ablations` | `results/svgloc_abl_*` + `svgloc_ablation_uids.json` |
| `outputs/report/summary.json` | `python -m blindspot.report summary` | the **`slidevqa_allpages`** arm plus the blind/one-page controls. `slidevqa_allpages` is a separate adapter and **is not a step in any pipeline** — run `blindspot.core --datasets slidevqa_allpages` yourself or the whole report chain cannot start. |
| `outputs/report/figures.json` | `python -m blindspot.report data` | all of the above |

`figures.json` is the auditable bundle: every number the report quotes, in one
file, each traceable to a `results/*.jsonl` row. Nothing downstream recomputes a
statistic — so if a number appears in a page but not here, it was typed by hand.

## The report chain

Run in this order, or `python -m blindspot.report all` for the middle five.

| # | command | writes |
|---|---|---|
| 0 | `report summary` | `outputs/report/summary.json` |
| 1 | `report data` | `outputs/report/figures.json` |
| 2 | `report examples` | `outputs/report/figures/*.png` |
| 3 | `report tables` | `outputs/report/tables.md` |
| 4 | `report index` | `outputs/report/figures.md`, `figures/index.html` |
| 5 | `report paste` | `outputs/report/paste_into_docs.html` |

`tables` and `index` also inject into `blindspots.md` between `<!-- Tn -->`
markers. That file is a separate deliverable and is **not in this repository**, and it is
gitignored, so a clone cannot obtain it at any price.

`tables` and `index` skip the injection and still write their own output. **`paste`
does not** — with no prose to paste it prints `blindspots.md not found; nothing to
paste`, exits 0, and writes no file. So `paste_into_docs.html` is unbuildable from a
clean clone. That is a real gap, not an expected skip.

`index` also degrades quietly: run out of order it exits 0 and emits a figure-less
`figures.md` and a 527-byte `index.html` rather than failing.

`paste_into_docs.html` is self-contained — markdown tables become real `<table>`
elements and PNGs are inlined as base64, so a document editor pastes them as
editable tables rather than a screenshot.

## Standalone HTML pages

These are not part of the report chain. Each answers one question and is
self-contained.

| artifact | command | what it is |
|---|---|---|
| `outputs/svgloc/report.html` | `python -m blindspot.report svgloc` | the generated localization set in detail |
| `outputs/svgderived/report.html` | `python -m blindspot.report svgderived` | the counting and word-presence sets |
| `outputs/failure_analysis.html` | `python -m blindspot.diagnose failure-modes` | is a grounding miss format, instruction, or vision? |
| `outputs/coord_diagnostics.html` | `python -m blindspot.diagnose coordinates` | annotated PNGs + the coordinate-compression explainer. Read this first for ScreenSpot-Pro. |
| `outputs/datasets.html` | `python -m blindspot.diagnose dataset-page` | what each dataset is and whether it turned out usable |
| — | `python -m blindspot.diagnose capability` | which axis UI grounding breaks along (stdout) |
| — | `python -m blindspot.diagnose gt-quality` | ground-truth quality by failure mode (stdout) |
| `outputs/probe/*.png` | `python -m blindspot.diagnose annotate-probe` | gold box vs predicted click, drawn on the screenshots |
| `outputs/gallery/<dataset>_NNN.html`, `outputs/annotations/`, `outputs/assets/` | `python -m blindspot.eval annotate` | every scored item as the model saw it, failures first, 50 per page |

`diagnose coordinates` needs `screenspot` **results**, not just its manifest — it
reads `coord_rows(ds, ...)`. Downloading the dataset is not enough; the pipeline never
*runs* `screenspot`, so you must score it yourself first. `diagnose annotate-probe`
additionally needs `results/probe_uids.json`, written only by `run_api probe`, which is
also not a pipeline step. Neither prerequisite is obtainable from `literature_eval`
alone.

## Dataset audit pages

Written beside the dataset they audit, from the manifest alone — deliberately
not from the layout code, because an overlay drawn by the same routine that
produced the gold box would hide a bug in that routine.

| artifact | command |
|---|---|
| `<data>/verify/index.html` | `python -m blindspot.generate audit --data <dir> --out <dir>/verify/index.html` |
| `<data>/examples/index.html` | `python -m blindspot.generate examples --data <dir> --out <dir>/examples/index.html` |
| `<data>/{counting,word_mc}/examples.html` | `python -m blindspot.generate examples-derived --data <dir>` |

Both default to `--data data/svg_localization`, which is right when auditing the
shipped set and wrong when auditing a new one. Pass `--data` explicitly.

## Part 3 artifacts

> ⚠️ **Every command in this table except `samples` writes to a reference artifact
> by default.** The paths below are gitignored and untracked, so there is no copy to
> restore, and a rerun does not reproduce them — the ladder alone changes
> `target_text`/`box_px` for ~91% of uids. `<dir>` in the table means *pass it*, not
> *it is required*: only `samples --out` is `required=True`. This is not
> hypothetical — these defaults have already overwritten `outputs/part3/assets/*`
> and `part3.html` during a routine test run.
>
> | command | flag | silently defaults to |
> |---|---|---|
> | `generate_finetune ladder` | `--out` | `data/svgloc_mr` |
> | `generate_finetune samples` | `--out` | **required — no default** ✅ |
> | `report_finetune gallery` | `--records` / `--out` | `data/sft_bbox/…` / `outputs/finetune/gallery.html` |
> | `report_finetune figures` | `--out-dir` | `outputs/part3/assets` |
> | `report_finetune examples` | `--out-dir` / `--md` | `outputs/part3/assets` / `outputs/part3/part3.md` |
> | `report_worked` | `--out` | `outputs/finetune/worked_examples.json` |
> | `render_markdown` | `--out` | derived from `--src` → `outputs/part3/part3.html` |
>
> Pass every flag explicitly, or drive the whole thing through
> `python -m blindspot.pipelines finetune_data --out DIR`, which refuses to target
> any of these by name. The pipeline is protected; **these bare commands are not.**

| artifact | command | note |
|---|---|---|
| `data/svgloc_mr/` | `generate_finetune ladder --out <dir>` | 6 aspect ratios × 4 sizes |
| `data/sft_bbox/*.jsonl` | `generate_finetune samples --n N --seed N --out <file>` | `--out` is required — no default |
| `outputs/finetune/gallery.html` | `report_finetune gallery` | deterministic; regenerates byte-identically |
| `outputs/finetune/gallery_<dataset>.html` | `generate_finetune audit --dataset <dir> --out <file>` | named for the dataset audited. **No command writes `gallery_multires.html` any more** — a file by that name on disk is stale output from before the path was derived. |
| `outputs/part3/assets/fig_*.png` | `report_finetune figures` | |
| `outputs/part3/assets/ex*.png` | `report_finetune examples` | `--inject` also splices into `part3.md` |
| `outputs/finetune/worked_examples.json` | `python -m blindspot.report_worked --max-spend 0.50` | **spends money**, capped by `--max-spend` (default `$0.50`, checked before every call); model-sampled, so not reproducible |
| `outputs/part3/part3.html` | `render_markdown --src outputs/part3/part3.md` | add `--paste` for `part3_paste.html` |

`part3.md` is hand-written prose with a machine-injected examples section, and
`part3.html` is generated from it — so those two cannot drift from each other.

That guarantee protects the wrong edge. What DOES drift is `part3.md`'s injected
captions against the PNGs in `assets/`: re-running `examples` without `--inject`
rewrites the images and leaves the captions describing the old ones. Measured in
this build, **8 of 9 captions name a different chart, size and target than the
image they label**, and `part3.html` inherits every one as alt text. If you re-run
`examples`, run it with `--inject` or fix the captions by hand.
Separately, only `--paste` refreshes `part3_paste.html`, so it goes stale if you
forget.

## Determinism

| artifact | reproducible? |
|---|---|
| `generate scenes/questions` output | yes, given `--seed`, **but see the drift note below** |
| `gallery.html` | yes — byte-identical, and a test asserts it |
| `part3.html` | yes, modulo whitespace |
| `figures.json` and everything downstream | yes, given the same `results/` |
| `results/*.jsonl` | **no** — `temperature` is unavailable and thinking pins it to 1. Measured item-level disagreement between two identical runs is 10.1%. |
| `worked_examples.json` | **no** — `--seed` picks which records are asked about, not what comes back |

**Drift note.** The scene generator no longer reproduces the committed
`data/svg_localization`: same seed, 73% of shared uids get different ground
truth, and some uids bind to a different question entirely. `results/*.jsonl` is
keyed by uid, so regenerating in place would join existing answers to the wrong
questions. `generate scenes` therefore requires an explicit `--out` — no default — and both
the pipeline and the Makefile refuse to target the committed set. The Part 3
artifacts above are protected the same way: `finetune_data` schedules nothing
that writes them unless `--out DIR` is given. See
[runme/FINETUNE.md §0](runme/FINETUNE.md#0-building-is-opt-in).

## Standalone evidence pages

`blindspot/report_pages.py` — seven builds that sit outside the `blindspots.md`
chain. They read `results/*.jsonl` and `data/<ds>/manifest.jsonl` **directly**, so
they do not need `figures.json` or any `summary.json` and can run without the
report chain. In the dependency spine they branch off raw results beside `eval`,
not after `report`.

| artifact | command | what it is |
|---|---|---|
| `outputs/causes/*.html` + `index.html` | `report_pages causes` | 15 per-cause evidence pages — one claim per page, its quantitative evidence, then examples grouped by benchmark. `absence_detection`, `answer_expression`, `counting`, `label_reference_binding`, `language_prior_override`, `position_bias`, `effective_resolution`, `derivation_vs_reading`, `cross_page_integration`, `list_answer_integrity`, `ground_truth_noise` and the rest. |
| `outputs/assets_causes/` | same | the full-size and thumbnail JPEGs those pages link to (~1,400 files) |
| `outputs/drilldown.{html,json,csv}` | `report_pages drilldown` | hierarchical drill-down from each headline number to individual questions, asserting at every split that a parent's n equals the sum of its children's |
| `outputs/slidevqa.html` + `assets_slidevqa/` | `report_pages slidevqa` | the SlideVQA arm: `evidence` (oracle slides) vs `all_pages` (full deck), which is what isolates retrieval from reading |
| `outputs/tasks/*.html` | `report_pages tasks` | one page per perceptual primitive |
| `outputs/report.html` | `report_pages primitives` | the per-primitive overview |
| `outputs/aug22/report.html` | `report_pages headline` | the corrected headline report — every accuracy printed beside the control that says what it means |
| `outputs/report/candidates.html` + `candidates/` | `report_pages candidates` | contact sheet of four candidate example images per blind spot, for choosing figure panels |

**Fidelity, measured against the original study's output on the full results:**
`causes` produces 16 pages with names identical to the reference and matching
`<img>`/`<table>` counts on every one, each within 6 bytes; `drilldown.json` and
`.csv` are byte-identical; `slidevqa.html` is an exact byte match with
`assets_slidevqa/` at 3,760 files = 3,760; `candidates.html` is an exact match;
`tasks` gives 11 pages with identical names, 7 of them byte-identical.

These generators were quarantined out of the package at one point on the reasoning that
nothing imported them. That was the wrong test: a module whose whole job is to
write files has no importers by construction. The right question is whether
anything else produces the artifact — and nothing did.

`report_pages` runs as the `pages` stage of `literature_eval`
(`pipelines.py:108`), or directly. Note that only `candidates` is optional in
that stage, so a `causes` abort on thin data stops the five pages after it —
pass `--continue-on-error` if you want the rest anyway.
