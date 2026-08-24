# Perception blind spots in Claude Haiku 4.5

Code and data for a study characterizing where Claude Haiku 4.5's visual perception
breaks down on the kinds of images business applications actually process: scientific
charts, diagrams, slide decks, dense infographics, and application screenshots.

Everything here is the measurement apparatus. The written report is a separate
deliverable and is not in this repository.

- **Model under test:** `claude-haiku-4-5-20251001`, thinking enabled at 2,000 tokens
- **Scale:** 13,965 questions across five public benchmarks, plus a synthetic dataset
  of 200 procedurally generated scenes built for this study
- **Generated on:** August 2026

---

## What the study found

Five public benchmarks, each scored by its own published metric.

| Benchmark | Operation measured | Items | Metric | Haiku 4.5 |
|---|---|---|---:|---|
| CharXiv | Read a value or a structure off a scientific chart | 5,000 | judged exact match | 84.7% |
| AI2D | Answer a question about a diagram, 4-way multiple choice | 3,086 | accuracy | 81.6% |
| SlideVQA | Answer from evidence inside a slide deck | 1,003 | ANLS | 68.8% |
| InfographicVQA | Read a dense, large-format infographic | 2,801 | ANLS | 66.7% |
| ScreenSpot-Pro | Point at an element described by its function | 1,581 | click-in-bbox | 1.8% |

A sixth scored arm — SlideVQA all-pages, n=494, 58.5% — is a retrieval control rather
than a separate benchmark.

The gap in that last row is the finding. Reading a chart and pointing at a chart are not
the same operation, and the second one fails almost completely. Because ScreenSpot-Pro
confounds three things at once — target size, image size, and the coordinate-emission
format — the study built a synthetic dataset that separates them
(`data/svg_localization`, 200 scenes, 16 chart types, rendered at three sizes).

Asking the same questions about the same generated scenes:

| Task on the same 200 scenes | Small | Large | Blind control | Items |
|---|---|---|---|---:|
| Is this word present? (4-way choice) | 99.73% | 100.00% | 22.6% | 736 |
| How many times does it appear? | 94.12% | 97.06% | 12.0% | 476 |
| Point at it (exact target box) | 6.68% | 4.41% | — | 1,587 |

The model reads the text and counts its occurrences near-perfectly, then cannot say
where it is. The blind controls confirm the first two rows depend on the image rather
than on a language prior.

Localization does not fail uniformly — it degrades with the precision demanded:

| Required precision | Chance | Accuracy | Ratio to chance |
|---|---:|---:|---:|
| 2×2 cell | 25.00% | 65.66% | 2.6× |
| 3×3 cell | 11.11% | 52.05% | 4.7× |
| 4×4 cell | 6.25% | 40.58% | 6.5× |
| 8×8 cell | 1.56% | 17.96% | 11.5× |
| 16×16 cell | 0.39% | 6.36% | 16.3× |
| the exact target box | 0.26% | 5.55% | 21.7× |

One set of predictions read at six tolerances, not six experiments. The signal is real —
21.7× chance is not guessing — but it is coarse. Coarse position is available; precise
position is not.

Full numbers, the other blind-spot candidates (label–object matching on AI2D,
hallucination on unanswerable questions, counting, resolution sensitivity), their blind
controls, and the limits of each claim are in the report.

---

## Quickstart

```bash
git clone <this repo> && cd haiku-perception-blindspots
./setup.sh                     # installs deps, creates runtime dirs, verifies the install
$EDITOR .env                   # ANTHROPIC_API_KEY=sk-ant-...

# smallest useful thing: 20 localization questions against Haiku 4.5, ~$0.10
python -m blindspot.core --datasets svg_localization --limit 20 --max-spend 0.10
```

`setup.sh --conda` creates a conda environment first if you prefer that to a venv.

```bash
make test      # the offline test suite, no API key needed
make verify    # the above, plus compileall and a CLI sweep
```

The synthetic dataset is committed, so that command works on a fresh clone with no
downloads. The five public benchmarks are third-party data and are **not** redistributed
here — `python -m blindspot.download` fetches them. See
[docs/DATASETS.md](docs/DATASETS.md).

A whole effort at once, rather than one step:

```bash
python -m blindspot.pipelines                                 # the three pipelines
python -m blindspot.pipelines literature_eval --list          # the plan, run nothing
python -m blindspot.pipelines literature_eval --all --max-spend 40
```

---

## Repository layout

```
blindspot/              one flat package -- every module, no subpackages
  core.py               adapters, prompts, the metered runner, scorers, sampling,
                        stats, taxonomy, failure modes, mermaid -- the layer
                        everything else builds on
  charxiv.py            CharXiv's published prompts and judge rubrics, verbatim
  download.py           fetch the public benchmarks into data/
  generate.py           build and audit the synthetic dataset
  generate_finetune.py  the resolution ladder, exact-ink boxes, SFT records
  run_api.py            the experiment drivers: official protocol, controls, probes
  judge.py              LLM-judge grading and ground-truth adjudication
  eval.py               runs -> numbers (JSON, never HTML)
  report.py             numbers -> figures, tables, HTML
  report_finetune.py    the Part 3 gallery, figures and example strip
  report_worked.py      GRPO group statistics over real samples (calls the API)
  render_markdown.py    markdown -> self-contained HTML
  diagnose.py           six diagnostics, one subcommand each
  tools.py              verify-install, artifact compare
  flow.py               the launcher framework (library, no CLI)
  pipelines.py          all three pipelines: which steps, in what order

tests/test_all.py       offline and deterministic
legacy/                 the pre-consolidation modules, reference only

data/svg_localization/  the dataset this study generated (committed, 43MB)
docs/                   everything below
```

`core.py` is what everything else imports, and it imports nothing from the rest, so the
harness can be pointed at a new dataset without pulling in any reporting code. A
pipeline owns no logic — it is a list of steps.

## The three pipelines

| pipeline | effort | feeds |
|---|---|---|
| `literature_eval` | evaluate on the published benchmarks we download | report §1–4 |
| `synth_localization_eval` | generate our dataset and evaluate on it | report §5–7 |
| `finetune_data` | build the SFT / GRPO training data | `outputs/part3/part3.md` |

Pipelines 1 and 2 feed one shared document, which is why the report chain is not itself
a pipeline. Runbooks: [docs/runme/](docs/runme/).

## Documentation

| | |
|---|---|
| [docs/PIPELINE.md](docs/PIPELINE.md) | What to run, in what order, with real commands |
| [docs/STRUCTURE.md](docs/STRUCTURE.md) | How the repository is laid out and why |
| [docs/REPO_MAP.md](docs/REPO_MAP.md) | Every module, its subcommands, one line each |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Why the harness is built this way — the decisions that shaped it |
| [docs/DATASETS.md](docs/DATASETS.md) | Which benchmarks, why, and which were considered and dropped |
| [docs/RESULTS_MANIFEST.md](docs/RESULTS_MANIFEST.md) | Inventory of the raw run files, which are not in git |
| [docs/runme/BENCHMARKS.md](docs/runme/BENCHMARKS.md) | Runbook: the public-benchmark pipeline |
| [docs/runme/SYNTHETIC.md](docs/runme/SYNTHETIC.md) | Runbook: the generated dataset |
| [docs/runme/FINETUNE.md](docs/runme/FINETUNE.md) | Runbook: the finetuning data |
| [legacy/README.md](legacy/README.md) | What is in `legacy/` and what it maps to |
| [data/svg_localization/README.md](data/svg_localization/README.md) | The generated dataset: what it is and what it controls for |
| [data/svg_localization/EVAL.md](data/svg_localization/EVAL.md) | How to score it, and the traps that make it easy to score wrong |
| [SETUP.md](SETUP.md) | Install, verify, tests, troubleshooting |

## Picking this up

If you are continuing this work, read the report first, then
[docs/METHODOLOGY.md](docs/METHODOLOGY.md) and [docs/PIPELINE.md](docs/PIPELINE.md). The
things most likely to trip you up are collected in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md#known-limits) and
[docs/STRUCTURE.md](docs/STRUCTURE.md#4-known-limits--documented-not-fixed) — in
particular that `results/` is not in this repository, so every number above has to be
re-derived by re-running the API before it can be re-checked, and that the generator has
drifted from the committed dataset and must not be run over it in place.
