# Repository structure

How this repository is laid out and why. The consolidation described here is
**done** — everything below describes what is on disk. Per-pipeline runbooks are
in [runme/](runme/); the module-by-module index is [REPO_MAP.md](REPO_MAP.md).

---

## 1. The organising idea

> Reports are not a flow. Each report module is generic in the sense of living
> with the other reusable modules and serving whichever dataset it is pointed at.
> The three pipelines are **aggregations** of what needs to be done.

```
blindspot/<module>.py     the code, grouped by what it does    ← every module
blindspot/pipelines.py    which to call, in what order         ← no logic
```

A pipeline owns **no logic**. It owns a list of steps. All three live in one file
so the differences between them are visible on one screen — they share a scorer, a
runner and a report chain, and only the argument lists differ.

### The three pipelines

| # | Pipeline | Effort | Document it feeds |
|---|---|---|---|
| 1 | `literature_eval` | evaluate on the published benchmarks we download | `blindspots.md` §1–4 |
| 2 | `synth_localization_eval` | generate our dataset and evaluate on it | `blindspots.md` §5–7 |
| 3 | `finetune_data` | build the SFT / GRPO training data | `part3.md` |

The names say what the effort *is*. `benchmarks` and `synthetic` described the
input; `literature_eval` and `synth_localization_eval` describe the work. And
`finetune_data` is explicit that the pipeline builds data — it does not train.

Pipelines 1 and 2 feed **one shared document**, which is why reporting cannot itself
be a pipeline: `blindspots.md` runs from "the benchmarks are unreliable" (pipeline 1)
to "so we built our own set and re-measured" (pipeline 2) inside a single argument.
Split the report by pipeline and the argument breaks in half.

---

## 2. The tree

```
haiku-perception-blindspots/
│
├── blindspot/                     one flat package, sixteen modules
│   ├── core.py           2209     adapters · prompts · runner · scoring · sampling
│   │                              stats · taxonomy · failure_modes · mermaid
│   ├── charxiv.py         595     CharXiv's prompts and judge rubrics, verbatim.
│   │                              Separate because it must never be edited to taste.
│   ├── generate.py       2842     the scene generator, the derived question sets,
│   │                              the audit and example pages
│   ├── generate_finetune.py 762   the resolution ladder, exact-ink recovery, SFT records
│   ├── eval.py           1998     results/*.jsonl -> JSON. One subcommand per artifact
│   ├── report.py         2786     the live report chain + the per-dataset pages
│   ├── report_finetune.py 665     the Part 3 gallery, figures and example strip
│   ├── report_worked.py   132     GRPO group statistics over real samples (CALLS THE API)
│   ├── render_markdown.py 382     markdown -> self-contained HTML, any document
│   ├── run_api.py        1194     official protocol · controls · probes · ablations
│   ├── judge.py           590     CharXiv judge · equivalence · ground-truth audit
│   ├── diagnose.py       1428     six diagnostics, one subcommand each
│   ├── download.py        550     six benchmark pullers
│   ├── tools.py           239     verify-install + compare
│   ├── flow.py            175     the launcher framework — library, no CLI
│   └── pipelines.py       276     all three pipelines
│
├── tests/test_all.py              scorers · stats · dataset invariants · structure
├── legacy/                        the pre-consolidation modules, reference only
│
├── data/                          one directory per dataset
├── docs/                          this directory
├── outputs/ results/ cache/       regenerable, gitignored
└── third_party/                   clones the GitHub-sourced datasets need
```

Sixteen modules, ~16,800 loc, down from 80 files and 25,161.

`core.py` is what everything else imports, and it imports nothing from the rest, so
the harness can be pointed at a new dataset without pulling in any reporting code.
`tests/test_all.py` pins that direction.

### The cost, stated plainly

`generate.py` and `report.py` are 2,842 and 2,786 lines. That is past the size
where a single file is easy to navigate, and it is the real price of sixteen. The
alternative — splitting each into two by sub-role — costs four more files and buys
back navigability. Worth revisiting if these files are ever going to be edited
rather than read.

### `legacy/`

The pre-consolidation modules, frozen for reference and provenance — 70 files,
22,142 loc. Nothing imports them, nothing runs them, they are not packaged, and
`tests/test_all.py` excludes them from its structural sweep on purpose: they carry
`parents[2]` roots and `sys.path` shims that were correct for the nested layout
they were written in.

Every module that was **merged** is preserved there, because merging destroys the
one-to-one mapping that would otherwise tell you where a function came from. The
seven modules that moved one-to-one are not, because git already records those as
renames. [legacy/README.md](../legacy/README.md) has the file-by-file mapping.

---

## 3. Running a pipeline

```bash
python -m blindspot.pipelines                                   # all three, summarised
python -m blindspot.pipelines literature_eval --list            # the plan, run nothing
python -m blindspot.pipelines literature_eval --all --max-spend 40
python -m blindspot.pipelines synth_localization_eval --task counting --stage run eval --max-spend 3
python -m blindspot.pipelines finetune_data --stage build verify
python -m blindspot.pipelines literature_eval --from judge --max-spend 5
```

`blindspot/flow.py` implements once, for all three: stage listing, `--dry-run`,
`--from` resume, `--offline`, `--max-spend` splitting across API steps,
optional-step tolerance, fail-fast with a resume hint.

Two guards come free from putting spend control in the framework:

```
$ python -m blindspot.pipelines synth_localization_eval --stage run
!! 4 step(s) call the API and --max-spend was not set.
   A plain API key cannot read a credit balance, so this ceiling is the
   only spend control there is. Set it, or use --offline.          [exit 2]
```

A missing `ANTHROPIC_API_KEY` fails the same way, before any work starts.

`synth_localization_eval --out DIR` refuses `data/svg_localization` outright — see
§4.

---

## 4. Known limits — documented, not fixed

Left as-is deliberately. Recorded so whoever inherits this is not surprised.

**The generator no longer reproduces the committed dataset.** Same command, same
seed: 4,724 questions against the committed 4,723, ground truth changed on 73% of
shared uids, 98 boxes below IoU 0.80, and some uids bound to a **different
question** (`svgloc:0015:medium:00` is `'Lead Referral'` committed, `'Nordics'`
regenerated). Since `results/*.jsonl` is keyed by uid, regenerating over a scored
set would join answers to the wrong questions.

→ **`data/svg_localization` stays committed and is the source of truth.** Do not
regenerate it in place. `python -m blindspot.generate scenes` is for building a
*new* set into a *new* `--out` directory, and the pipeline refuses an `--out` that
resolves to the committed set. See
[runme/SYNTHETIC.md §0](runme/SYNTHETIC.md#0-do-not-regenerate-in-place).

**Measured constants in source.** `report.py`'s `gold_quality()` hardcodes
`total_failures` (735 / 852 / 1552) and the denominators (5000 / 2801 / 1581).
Fine for a frozen study; they would go stale if the runs were repeated, since
measured run-to-run disagreement is 10.1%.

**The markdown renderer exists twice.** `report.py paste` for `blindspots.md`,
`render_markdown.py` for `part3.md`. Both base64-embed figures and convert markdown
tables. Merging them is optional and was not part of this consolidation.

**`eval localization` writes an n=0 summary** when `results/` is absent, rather than
aborting. Harmless while `outputs/` is empty; a two-line guard if it ever bites.

**The finetune ladder is not chained to the sample builder.**
`generate_finetune samples` reads `data/svg_localization`, not `data/svgloc_mr`, so
the aspect/size ladder does not reach the SFT records. May be intended; the stage
diagram implies otherwise.

**Generated HTML is still committed inside `data/`.**
`data/svg_localization/{verify,examples}/index.html` are audit and browsing pages
that `blindspot.generate audit` and `... examples` rebuild from the manifest. They
are tracked, so a rebuild shows up as a diff.

**`data/svgloc_mr` is documented only in the runbook** — no README beside the data,
and it is absent from `data/README.md` and [DATASETS.md](DATASETS.md).

**The report prose lives elsewhere.** `blindspots.md` is a separate deliverable and
is not in this repository, so `report tables|index|paste` have nothing to write
into on a fresh clone. Expected — `outputs/` is regenerable and gitignored.

**`results/` is not distributed.** Re-deriving any number means re-running the API.
[RESULTS_MANIFEST.md](RESULTS_MANIFEST.md) inventories the original files with row
counts and checksums so a regenerated set can be checked against them.

---

## 5. What the consolidation fixed

| Problem before | Fixed by |
|---|---|
| three top-level packages split by *when written*, not *what they do* | one package, three pipelines split by effort |
| `finetune/` exempt from every structural test — its files were not in the test's file list | it is `generate_finetune.py` + `report_finetune.py` in the one package; the sweep covers them |
| two `sys.path.insert` shims the anti-shim test could not see | one package, no shims; the sweep is shape-based, not a hand-kept list |
| `judging/judge.py` imported upward into `reporting.report` | `judge.py` carries its own `load_results`; the edge is gone |
| 9.5k loc of superseded renderers beside the live chain | frozen in `legacy/`, out of the package and out of the test sweep |
| `gt_quality.py` — 111 loc, zero importers, no entry point | dropped; the original is in `legacy/` |
| no single entry point per effort | `python -m blindspot.pipelines <name>` |

## 6. Design rules that still hold

- **`figures.json` is the single auditable artifact.** `report data` assembles
  every number the report quotes into `outputs/report/figures.json`, and nothing
  downstream computes its own statistics. A reviewer audits one file.
- **The analysis layer writes JSON, not HTML.** `eval` reads `results/*.jsonl` and
  emits JSON; rendering is `report`'s job. That is what makes every number
  independently checkable and lets a report rebuild in seconds without re-scoring.
  Two subcommands are stated exceptions and say so in their help: `eval annotate`
  also emits browsing galleries, and `eval tiling` calls the API.
- **`core.py` never imports upward.** Pinned by a test.
- **Every API step declares itself.** A pipeline step that reaches the API carries
  `needs_api=True`, which is what makes `--offline` and the `--max-spend` gate
  work. A test asserts no step reaches the API without it.
