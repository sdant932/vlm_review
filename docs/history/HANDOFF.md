> **Historical.** Preserved verbatim from the working tree for provenance.
> It describes an earlier state of the project and is **not** current guidance.
> For how things actually work now, read the top-level `README.md` and
> `docs/PIPELINE.md`.

---

# Handoff: Haiku 4.5 Perception Blind-Spot Take-Home

## What this is

Prep work for a take-home task: characterize Claude Haiku 4.5's blind spots on
perception tasks relevant to business applications (charts/graphs, diagrams/
flowcharts, documents, presentation slides, UI screens). See `DATASETS.md` for the
full rationale, the final dataset shortlist with dates/sizes/goals, which ones got
excluded and why, and the metrics/scoring flow.

This machine (`/Users/sdoveh`) already has the full environment and most of the
data set up — nothing needs to move. You just need your own Anthropic API key and
a fresh Claude Code session picking this up.

## Everything already done

- **Conda env `takehome`** (Python 3.11) with `anthropic`, `openai`, `datasets`,
  `pillow`, `opencv-python-headless`, `torch`/`transformers` (only needed if you
  add a local-VLM comparison — Haiku itself is API-only). Activate with:
  ```bash
  conda activate takehome
  cd ~/gitrep/takehome
  ```
- **Datasets downloaded to `data/`** (~15GB), one folder per dataset, each with an
  `images/` dir and a `manifest.jsonl` mapping each example to its ground truth.
  Current status (may still be finishing — check `data/*/manifest.jsonl` line
  counts against the totals in `DATASETS.md` if numbers look low):

  | Dataset | Status |
  |---|---|
  | ChartQA, CharXiv, DocVQA, InfographicVQA, SlideVQA, ScreenSpot, RICO-ScreenQA | done, at their real eval-split size |
  | LiveXiv, FlowLearn (real + simulated), ScreenSpot-Pro | were still downloading as of last check — verify these finished |
  | BlindTest, Ferret-UI | done — these are capped at whatever's publicly available (30 and 8 rows respectively), not a subsample |
  | PlotQA | **dropped** — see "Excluded" in `DATASETS.md`, its HF mirror wasn't real Q&A and the actual dataset lives on ungated Google Drive with no small eval subset |

- **Download scripts** (`scripts/`) if you need to re-run or extend anything:
  `download_datasets.py` (generic HF `load_dataset` puller), `download_flowlearn.py`
  and `download_screenspot_pro.py` (custom — those two aren't in a `load_dataset`-able
  format), `prepare_github_sources.py` (extracts BlindTest + Ferret-UI from the
  cloned repos in `third_party/`).
- **`third_party/`** has the cloned source repos for BlindTest, Ferret-UI,
  ScreenSpot-Pro, FlowLearn, SlideVQA (gitignored, not part of this repo's history).

## What's NOT done yet — this is the actual next step

**The eval harness itself hasn't been written.** `DATASETS.md` documents the planned
flow (adapter layer → prompt construction → API call → answer parsing → scoring →
persistence → aggregation) and which metric fits each dataset (MC accuracy, ANLS,
relaxed accuracy, binary accuracy, count accuracy + MAE, click-in-bbox accuracy). None
of that is implemented — no code has called the Anthropic API yet, so no Haiku
results exist. That's the next thing to build.

Also still open: **BlindTest has no ground truth** in its public release (confirmed
by reading its generator notebooks — the shipped image filenames had the
answer-bearing prefix stripped before release). Usable for qualitative probing now;
scoring it needs either porting the notebooks' generation logic to produce fresh
labeled images, or manual annotation (only ~30 images total, tractable by hand).

## Your API key

```bash
cd ~/gitrep/takehome
cp .env.example .env
```
Then set `ANTHROPIC_API_KEY=<your key>` in `.env` (already gitignored — never gets
committed). Get a key from console.anthropic.com if you don't have one.

## Where to start

1. Read `DATASETS.md` in full — it has the dataset table, metrics, and flow.
2. Verify the in-progress downloads actually finished (`wc -l data/*/manifest.jsonl`
   against the "Size of that split" numbers from the size-audit conversation — ask
   your Claude session to re-verify via HF's datasets-server `/size` endpoint if
   unsure).
3. Build the eval harness per the flow in `DATASETS.md`: adapter functions per
   dataset → prompt templates → Anthropic API runner → per-metric scorers →
   `results/<dataset>.jsonl` → aggregation/report.
4. Pilot on ~20 examples per dataset before scaling to the full pulled set, to catch
   prompt/parser bugs cheaply.
