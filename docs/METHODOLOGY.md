# Methodology

Why the harness is built the way it is. Most of these are decisions that cost something
to learn, so they are written down rather than left in the code.

---

## Measuring perception, not language

A model can answer a question about a chart without looking at the chart. Question
phrasing, world knowledge and answer-option plausibility all leak. So **every candidate
blind spot has a blind control**: the same question, same prompt, image withheld.

If withholding the image barely changes the score, the finding is about language and
does not belong in a perception study. The blind controls are what separate the real
findings here from the plausible ones — on AI2D the blind score is 62.7% against a 25%
chance line, which means a naive reading of the sighted score would have credited
perception with work the question text was doing.

`python -m blindspot.run_api controls` runs these.

## Scoring by each benchmark's own metric

Every arm is scored the way its publishers score it — ANLS for InfographicVQA and
SlideVQA, judged exact match for CharXiv, click-in-bbox for ScreenSpot and
ScreenSpot-Pro, plain accuracy for AI2D. Inventing a uniform metric would make the arms
comparable to each other and incomparable to everything published, which is the wrong
trade.

The consequence is that cross-benchmark comparisons in this study are about *operations*
(reading vs pointing), not about which number is bigger.

### CharXiv needs a judge

CharXiv does not score with string matching. It batches (question, gold, response) under
a per-question-type rubric and asks a judge model to extract the answer and score it
binary. Our normalized-match scorer is a **lower bound** on the free-text types — a
correct answer phrased differently is marked wrong.

`blindspot/core.py` reports which side of that line each question falls on rather
than pooling them, and `python -m blindspot.judge charxiv` runs the real protocol. The one deviation from
CharXiv's setup: they used GPT-4o as judge, we use a Claude model. That is recorded in
every output row as `judge_model` rather than glossed over.

## Ground truth is not free of error

Some of the model's "failures" are the benchmark being wrong. `blindspot.judge gt-audit` shows
an adjudicating model the image and asks whether the gold or the prediction is right.

Measured contested-gold floors on the model's *error set*: CharXiv 16.3%,
InfographicVQA 16.8%, ScreenSpot-Pro 0.0% — which translate to whole-set floors of 2.4%,
5.1% and 0.0%. Any effect smaller than the relevant floor is not reportable.

The synthetic dataset exists partly to escape this: its ground truth is the text
placement the generator just computed, not a human annotation, so it has no labelling
noise floor to subtract.

## Structured outputs, not regex parsing

Answers come back through structured outputs. With thinking enabled the model reasons in
its thinking block and emits only schema-conforming JSON. This removes the entire class
of parser bugs where a correct answer is scored wrong because it arrived in an unexpected
sentence.

The exception is CharXiv's descriptive questions, which carry their own answer-format
instructions (vendored verbatim). Adding our own on top would change the task, so those
go through untouched.

## The image ceiling is the whole story on localization

Haiku 4.5 downsizes anything larger to roughly a 1,568px long edge. A 3840×2160
screenshot therefore loses about 93% of its pixels before the model sees it.

Two consequences shaped the design:

1. `--max-edge` exists as an ablation. If a score is unchanged when the image is
   pre-shrunk to 1568, the failure is not about resolution.
2. On the synthetic dataset the same scene is rendered at three sizes, two of which the
   API delivers at the same dimensions. Any difference between those two is noise, which
   gives a **null control** — a measurable floor for how much apparent effect is just
   run-to-run variance. It came out at 0.13pp.

`blindspot/core.py` logs the as-sent resolution on every row, so a resizing artifact can
always be distinguished from a perception failure after the fact.

## Sampling by the cell you intend to report

An early pilot sampled CharXiv by figure with `random.sample`. Each figure contributes
four of nineteen randomly chosen question types, so a 200-figure sample produced
per-question-type counts between 3 and 16 — numbers with no statistical content that
nevertheless rendered as confident bars ("count lines: 100%, n=3").

`blindspot/core.py`'s `stratify()` works on whatever axis the report will break results down by.
Cells smaller than the target contribute their whole pool, and the realised n is returned
so under-filled cells get reported rather than silently shipped.

## Spend control, because nothing else provides it

A plain API key cannot read a credit balance — every organization endpoint returns 401.
The harness therefore meters itself: it prices each request from a per-model table and
stops at `--max-spend` rather than discovering the ceiling as a mid-run 400.

Billing failures are treated as **fatal, not retryable**. Backing off and retrying a
credit-balance error burns minutes going nowhere.

## Resumability

Results append to `results/<dataset>__<tag>.jsonl` keyed by uid. A rerun loads what is
already there and skips it, so a killed run costs its in-flight requests and nothing
else. This matters at 14,000 questions: the alternative is a run you dare not interrupt.

## Runs are not deterministic

`temperature` is unavailable in anthropic 1.0.0, and thinking pins it to 1 regardless.
Scores are therefore not exact, and `--repeat` exists to measure that rather than pretend
otherwise. Measured: 2,121 items asked twice under identical settings, 214 disagreed —
a **10.1% item-level disagreement rate**. Any single-run difference smaller than that is
not a result.

## Thinking dialects differ by model generation

`thinking={"type": "enabled", "budget_tokens": N}` was removed on the 4.6+ generation and
returns a 400 there; `{"type": "adaptive"}` does not exist on 4.5-era models. `MODELS` in
`blindspot/core.py` carries the dialect per model and `thinking_config()` branches on it.
Adding a model means adding its pricing and its dialect there.

## Control models are harness checks, not comparisons

Where a stronger model is run (`blindspot.run_api coord-probe`), the purpose is to establish
that the harness is sound: if a stronger model lands inside the gold boxes on identical
inputs, a low score is a capability result rather than a plumbing bug. Model-vs-model
claims would need matched conditions this study did not run.

---

## Known limits

Stated plainly, because a reader should be able to tell what this study does and does not
establish.

- **`results/` is not in this repository.** The raw API responses are ~64MB across 42
  files and were left out. `docs/RESULTS_MANIFEST.md` inventories them with row counts
  and checksums, but re-deriving any number here means re-running the API. Everything
  downstream of `results/` is therefore reproducible in method, not in a single command.
- **The report itself is a separate deliverable** and is not in this repository, so
  figures referenced by the report cannot be regenerated from a fresh clone alone.
- **The polarity result is observational.** Theme is assigned per scene rather than
  crossed within it, so dark and light backgrounds differ in more than background. The
  adjusted odds ratio is stratified on resolution, target-area tertile and contrast
  tertile, which handles the obvious confounds; it is sensitive to how those strata are
  binned, and the estimate moves with the binning. It is computed in
  `blindspot/report.py::_mh_or` so it can be audited rather than taken on trust.
- **One CharXiv figure is unresolved.** Descriptive/reasoning reads 90.7/63.7 through the
  judged path and 91.0/59.4 through the raw path. The tables quote the judged pair; the
  discrepancy has not been chased down.
- **Contamination is not ruled out.** CharXiv, ScreenSpot-Pro and FlowLearn are 2024+ but
  not guaranteed unseen. LiveXiv's rolling snapshots were pulled as the clean check and
  not scored. The synthetic dataset is contamination-proof by construction, which is part
  of why the headline localization result rests on it.
- **Single model, single configuration.** Everything is `claude-haiku-4-5-20251001` with
  thinking at 2,000 tokens. Nothing here says how the failures scale with model size or
  thinking budget.
