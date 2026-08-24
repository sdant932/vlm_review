# RUNME — Part 1: public benchmark evaluation

Measure Claude Haiku 4.5 on five third-party benchmarks under each one's own
published metric. Nothing here touches the generated dataset — that is
[SYNTHETIC.md](SYNTHETIC.md).

**Pipeline:** `literature_eval` — `download → run → controls → judge → eval → diagnose → report`
**Reads:** `data/<benchmark>/` · **Writes:** `results/*.jsonl`, `outputs/`

## Launch

The whole pipeline is one command. Every step calls a **generic** module in
`blindspot/`; `blindspot/pipelines.py` only says which, in what order.

```bash
python -m blindspot.pipelines literature_eval --list                 # print the plan, run nothing
python -m blindspot.pipelines literature_eval --all --max-spend 40
python -m blindspot.pipelines literature_eval --stage eval report --offline
python -m blindspot.pipelines literature_eval --from judge --max-spend 5
python -m blindspot.pipelines literature_eval --all --dry-run
```

`--list` prints 7 stages, 26 steps; 12 of them call the API.

`--max-spend` is split across the API steps and is **required** whenever any are
selected — a plain API key cannot read a credit balance, so this ceiling is the
only spend control there is. Omitting it exits 2 before any work starts, as does
a missing `ANTHROPIC_API_KEY`.

A failed step aborts with a resume hint (`--from <stage>`); steps marked optional
warn and continue.

**Deliverable:** sections 1–4 of `blindspots.md`. That document is a
separate deliverable and is not in this repository, so `outputs/` is empty
on a fresh clone and the final injection step has nothing to write into.
That is expected. What this pipeline produces — the benchmark suite,
its reliability, the six blind-spot candidates, and which two were taken further.
Sections 5–7 come from [SYNTHETIC.md](SYNTHETIC.md); the two pipelines write one
shared document.

---

## 0. Prerequisites

```bash
./setup.sh                            # editable install, runtime dirs, verify
export ANTHROPIC_API_KEY=sk-ant-...   # or put it in .env
python -m blindspot.tools verify-install
```

`OPENAI_API_KEY` is optional — only if you swap the CharXiv judge back to GPT-4o.
The harness defaults to a Claude judge.

Disk: the full benchmark set is **~16GB**. Per-dataset, resolved:

| dataset | on disk |
|---|---:|
| slidevqa | 4.8 GB |
| screenspot_pro | 3.1 GB |
| docvqa | 2.9 GB |
| infographicvqa | 2.0 GB |
| rico_screenqa | 1.2 GB |
| livexiv | 913 MB |
| flowlearn_sci | 415 MB |
| screenspot | 306 MB |
| ai2d | 172 MB |
| chartqa | 136 MB |
| charxiv | 96 MB |

---

## 1. Download

```bash
python -m blindspot.pipelines literature_eval --stage download    # all three sources
```

or the pullers directly — three sources need their own, because they are not
`load_dataset`-able:

```bash
python -m blindspot.download hf --only charxiv --n 5000
python -m blindspot.download screenspot-pro          # per-app JSON + images
python -m blindspot.download flowlearn               # two subsets, two layouts
python -m blindspot.download github-sources          # from clones in third_party/
```

Each writes `data/<name>/images/` plus a `manifest.jsonl` mapping every example
to its ground truth.

> ⚠️ `download github-sources` takes **no arguments** and its underlying puller
> starts work on invocation. `--help` is safe (argparse intercepts it), but do
> not run it to "just see what it does".

**Verify before running anything expensive:**

```bash
python -m blindspot.tools verify-install
python -m pytest
```

---

## 2. Run the model

Always pilot first. `--max-spend` is a hard stop, not a warning — set it on
every run. A plain API key cannot read a credit balance, so this is the only
spend control that exists.

```bash
# pilot
python -m blindspot.core --datasets charxiv --limit 20 --max-spend 0.50

# stratified — per-cell n is something you chose, not something you got
python -m blindspot.core --datasets charxiv --per-cell 250 --max-spend 20

# full sweep
python -m blindspot.pipelines literature_eval --stage run --max-spend 40

# resolution ablation: hand the model fewer pixels
python -m blindspot.core --datasets screenspot_pro --max-edge 1568 --max-spend 5
```

Results append to `results/<dataset>__<tag>.jsonl`, keyed by uid. Re-running
skips what is already there, so a killed run resumes and costs only its
in-flight requests. Verified: a repeat invocation reported `2 selected, 4720
already done, 0 to run`.

### Leaderboard-comparable numbers

`blindspot.core` uses this study's prompt. For a number you can put next
to a published one, use the benchmark's own protocol:

```bash
python -m blindspot.run_api official --datasets screenspot_pro --full --max-spend 5
python -m blindspot.run_api official --datasets screenspot_pro --rescore   # no API calls
```

`--rescore` recomputes metrics from saved raw responses. Verified output:

```
official : action_acc   0.0%   wrong_format 1579
lenient  : action_acc   1.8%   wrong_format    4   text 2.7%  icon 0.5%
answered in pixels despite being asked for 0-1: 55.9%
```

### Controls — run one for anything you will call a perception failure

```bash
python -m blindspot.run_api controls --arm blind    --datasets charxiv --n 500 --max-spend 2
python -m blindspot.run_api controls --arm onepage  --datasets slidevqa --n 500 --max-spend 2
python -m blindspot.run_api grid --n 350 --max-spend 1
python -m blindspot.run_api coord-probe --model claude-sonnet-5 --max-spend 0.50
```

Ask the same question with the image withheld. **If the score barely moves, the
image was never carrying the answer** and the finding is about language, not
vision. `coord-probe` runs a stronger model on identical inputs — it is a
harness check, not a model comparison.

---

## 3. Judge

String matching is a **lower bound** on free-text answers. Any CharXiv number
quoted next to the published one must come through the official judge.

```bash
python -m blindspot.judge charxiv --dataset charxiv --judge-model claude-opus-5 --max-spend 4
python -m blindspot.judge equiv   --dataset infographicvqa --max-spend 2
```

Ground-truth quality — how much measured error is the *benchmark* being wrong:

```bash
python -m blindspot.judge gt-audit --dataset charxiv --per-category 5 --max-spend 2
python -m blindspot.diagnose gt-quality
```

**Every `blindspot.judge` subcommand calls a model.** They are non-deterministic
and metered; `blindspot.diagnose gt-quality` reads what they wrote and is free.

---

## 4. Aggregate

```bash
python -m blindspot.eval aggregate        # → outputs/summary.json
```

Verified against the committed results:

```
wrote outputs/summary.json  (21 KB)
  12468 questions | 11 primitives | 3 with >1 source
  charxiv           84.7%  n=5000
  infographicvqa    66.7%  n=2801
  screenspot_pro     1.8%  n=1581
  ai2d              81.6%  n=3086
```

---

## 5. Diagnose

Each answers one question and writes a self-contained page.

```bash
python -m blindspot.diagnose failure-modes     # format, instruction, or vision?
python -m blindspot.diagnose coordinates       # → outputs/coord_diagnostics.html + PNGs
python -m blindspot.diagnose capability        # which axis does grounding break along?
python -m blindspot.diagnose dataset-page      # what each dataset is, is it usable
python -m blindspot.diagnose annotate-probe    # gold vs prediction on the screenshots
```

`coordinates` is the one to read first for ScreenSpot-Pro. Verified:

```
screenspot       n= 200  acc= 21.0%  slope=1.023  intercept=+0.008
screenspot_pro   n=1580  acc=  1.6%  slope=0.871  intercept=+0.193
```

> All of these except `capability` take **no arguments and start work on
> invocation**. `capability` takes an optional positional dataset, defaulting to
> `screenspot_pro`.

---

## 6. Report

```bash
python -m blindspot.pipelines literature_eval --stage report
```

which is these five, in this order:

```bash
python -m blindspot.report data        # → outputs/report/figures.json
python -m blindspot.report examples    # → outputs/report/figures/*.png
python -m blindspot.report tables      # → outputs/report/tables.md
python -m blindspot.report index       # → outputs/report/figures.md
python -m blindspot.report paste       # → outputs/report/paste_into_docs.html
```

`python -m blindspot.report all` runs exactly that chain. `data` reads
`outputs/report/summary.json` as a **file**, not an import, so run
`python -m blindspot.report summary` first if that file is stale or absent.

The cross-study document that combines these results with Part 2's is
`blindspots.md`, a separate deliverable that is not in this repository.

The superseded renderers that used to sit beside this chain were deleted in the
consolidation; git history and [../../legacy/README.md](../../legacy/README.md)
say which and why. Do not resurrect them.

---

## 7. Troubleshooting

**`FileNotFoundError: data/charxiv/manifest.jsonl`** — the dataset is not
downloaded. `blindspot.eval aggregate` raises a raw traceback here rather than a
message; §1 first.

**A 400 mentioning `thinking`** — `--model` points at a model whose thinking
dialect differs. `budget_tokens` was removed on the 4.6+ generation; `adaptive`
does not exist on 4.5-era models. Add the model to `MODELS` in
`blindspot/core.py` with its pricing and dialect.

**A run stops immediately citing credit balance** — deliberate. Billing errors
are fatal rather than retryable, because retrying one burns minutes going
nowhere.

**Numbers differ between two identical runs** — expected. `temperature` is
unavailable in anthropic 1.0.0 and thinking pins it to 1. Measured item-level
disagreement between two identical runs is **10.1%** (2,121 items, 214
disagreed). Runs are resumable, not bit-reproducible.
