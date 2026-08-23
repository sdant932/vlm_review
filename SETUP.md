# Setup

## Requirements

- Python 3.11 or newer
- An Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
- ~1GB disk for the repository and the shipped dataset. Downloading the public
  benchmarks needs considerably more — the full set used here was ~16GB.

No GPU. The model under test is API-only, and every image in this project is drawn and
measured with Pillow.

## Install

```bash
./setup.sh
```

That installs the package in editable mode, creates the runtime directories git does not
track (`results/ outputs/ cache/ third_party/`), writes a `.env` from the template if one
is missing, and verifies the install.

If you prefer conda:

```bash
./setup.sh --conda          # creates the env from environment.yml
conda activate blindspot
./setup.sh                  # then the normal path
```

Then put your key in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`.env` is gitignored. `OPENAI_API_KEY` is optional and only needed if you switch the
CharXiv judge back to GPT-4o, which is what the CharXiv authors used; the harness
defaults to a Claude judge.

## Dependencies

Three, plus two more only for downloading benchmarks:

| | |
|---|---|
| `anthropic` | the API client. The study ran on 1.0.0 |
| `pillow` | all rendering, measurement and downscaling |
| `numpy` | regression fits in the coordinate diagnostics |
| `datasets`, `huggingface_hub` | *optional*, only `scripts/download/` needs them |

This project does **not** need torch, transformers, opencv, scikit-learn, pandas or
matplotlib. If you are migrating an older environment that had them, they were never
imported.

## Verify

```bash
python scripts/verify_install.py
```

Checks three things: every module in both packages imports, every script with an argparse
CLI parses its arguments, and the shipped dataset loads through the adapter layer. Expect:

```
==> imports
   ok
==> CLIs
   7 script(s) take no arguments, skipped: ...
   ok
==> dataset
   4723 questions, first uid svgloc:0000:small:00 (point)
   ok
```

`make verify` runs the same thing with a `compileall` pass in front.

## First run

```bash
python -m blindspot.core.runner --datasets svg_localization --limit 20 --max-spend 0.10
```

Twenty localization questions against Haiku 4.5, roughly $0.10. Results append to
`results/svg_localization__haiku-4-5_think2000_native_r0.jsonl`.

The run is resumable — it is keyed by uid, so re-running skips what is already there —
and `--max-spend` is a hard stop, not a warning. Set it on every run: a plain API key
cannot read a credit balance, so this is the only spend control available.

## Troubleshooting

**`ModuleNotFoundError: blindspot`** — `pip install -e .` did not run, or ran into a
different interpreter than the one you are using. Check `which python` matches the
environment `setup.sh` installed into.

**A 400 mentioning `thinking`** — you pointed `--model` at a model whose thinking dialect
differs from the default. `budget_tokens` was removed on the 4.6+ generation; `adaptive`
does not exist on 4.5-era models. Add the model to `MODELS` in `blindspot/core/runner.py`
with its pricing and dialect.

**A run stops immediately citing credit balance** — that is deliberate. Billing errors are
fatal rather than retryable, because retrying one burns minutes going nowhere.

**An analysis or reporting module raises `FileNotFoundError` on something under
`outputs/`** — those modules chain, and each reads what the previous one wrote. The order
is in [docs/PIPELINE.md](docs/PIPELINE.md). Note that `results/` is not distributed with
this repository, so the reporting chain cannot run on a fresh clone until you have
generated results of your own.
