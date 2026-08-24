# Pipeline 1 — literature_eval

Agent brief: run it from the docs alone, under $0.50. Compare against
`blindspots.md`.

## Headline

**$0.4633, 170 model calls.** The chain is intact from download through
`outputs/summary.json`. It is **broken from there to `outputs/report/`**.

The agent worked in a scratch copy because `flow.py` runs every step with
`cwd=ROOT` — there is no way to point a pipeline at a different `results/`.

## Numbers vs the reference report

| arm | agent | report | read |
|---|---|---|---|
| CharXiv, string match | **87.4%** (n=140) | 84.7% judged (n=5,000) | consistent — and note string match came in *above* the judged number, cutting against the docs' "string matching is a lower bound" |
| CharXiv descriptive | 91.9% (n=113) | 90.7% | consistent |
| CharXiv reasoning | 68.8% (n=27) | 63.7% | the descriptive/reasoning gap reproduces cleanly |
| CharXiv N/A hallucination | 11.1% (3/27) | 10.6% (n=1,000) | consistent |
| AI2D | 87.5% (n=8) | 81.6% | consistent, n useless |
| AI2D blind control | 50.0% (4/8) | 62.7% (chance 25%) | consistent — "partly answerable without the image" survives |
| ScreenSpot-Pro | 0.0% (0/8) | 1.8% | consistent; n=8 can't distinguish |
| answered in pixels when asked for 0–1 | 8/8 = 100% | 55.9% | directionally consistent |

**Nothing contradicts the report.** The two arms with usable n — the CharXiv
descriptive/reasoning split and the N/A hallucination rate — land within 1–5
points of published.

## Structural gaps found

1. **`--max-spend` overshoots 12.8×.** `window = concurrency * 4` with
   `--concurrency 32` puts 128 requests in flight. Measured: cap crossed at
   record 15, run continued to 140. *(Fixed — window bounded and sized to
   remaining budget; simulated overshoot 128 calls → 11.)*
2. **The report stage cannot run**: `pipelines` omits `report aug22`, which
   `report data` reads as a file.
3. **`diagnose coordinates` needs a dataset the pipeline never downloads**
   (`screenspot` v1).
4. **`report aug22` needs `slidevqa_allpages`**, which no pipeline step runs.
5. **`eval aggregate` silently drops SlideVQA** — `DATASETS` lists four of five.
   The published "12,468 questions" is exactly the four non-SlideVQA benchmarks,
   so the omission shipped unnoticed.
6. **`eval aggregate` gives no indication a run was partial** — reports
   `n=140` in the same shape as `n=5000`.
7. `PIPELINE.md`'s dataflow diagram is wrong: `report.py` reads `results/*.jsonl`
   directly, never `outputs/summary.json`.
8. SlideVQA's metric is misdocumented as ANLS in three places; the code computes
   `token_f1`.
9. Question total inconsistent: 13,965 vs 13,471 — the 494-item control is
   double-counted.
10. No documented way to fetch a *small* benchmark sample; `--per-app` appears
    only in a module docstring.

## Breakages

- `download hf` **hangs at interpreter exit** (2 of 3 runs) in
  `arrow::internal::ThreadPool::Shutdown`. Work completes; the process never
  returns, so `--stage download` appears hung.
- Five steps fail with raw `FileNotFoundError` tracebacks rather than messages.
