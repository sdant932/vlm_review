# Pipeline 2 — synth_localization_eval

Agent brief: get it running from the docs alone, under $0.50. Compare against
`blindspots.md`.

## Headline

Works end to end, and the study's central contrast reproduces at 1/20 the sample
size — **but the run cost $3.40 against a $0.50 ceiling**, because `--max-spend`
was not enforced on three of the pipeline's four API steps.

> **Fixed.** `run_api` submitted every call up front, so `break` on the ceiling
> exited the loop while `ThreadPoolExecutor.__exit__` drained the queue and
> billed it. Now bounded: a simulated $0.10 cap went from 1000 calls to 14.

## Spend

| step | cap asked | spent | calls | rows kept |
|---|---:|---:|---:|---:|
| localization (`blindspot.core`) | $0.15 | $0.113 | 30 | 30 |
| word_mc + blind (`run_api derived`) | $0.10 | **$3.241** | 1461 | **44** |
| counting | $0.04 | $0.043 | 15 | 15 |

**1,417 calls were billed and their results discarded.**

## Numbers vs the reference report

| task | report (small/large) | agent | verdict |
|---|---|---|---|
| word presence | 99.73 / 100.00 | 100.00 (n=28) / 100.00 (n=26) | consistent |
| counting | 94.12 / 97.06 | 100.00 (n=9) / 100.00 (n=14) | consistent |
| point at it | 6.68 / 4.41 | 0.00 (n=12) / 7.69 (n=13) | consistent, tiny n |

Pooled: recognition **100% (98/98)** vs pointing **4.26% (2/47), 16.4× chance**.
The precision ladder reproduces its shape — 2×2 72.3%, 4×4 42.6%, 16×16 8.5%,
exact 4.26%, ratio-to-chance rising monotonically.

**The claim survives at 1/20 the sample size.** Nothing contradicts the report.

Could not confirm: the blind controls (structurally unreachable on a partial
budget — the blind arm runs only after the *entire* image arm, with no
`--only-blind`), and the medium-vs-large null control (0 complete triples in a
random sample).

## Documentation defects

1. **`--max-spend` is not a hard stop**, contradicting `SETUP.md:108`,
   `PIPELINE.md:80,182`, `BENCHMARKS.md:106`, `SYNTHETIC.md:28`. *(Fixed.)*
2. `--task` and `--out` are used by the runbook but absent from `--help`.
3. `SYNTHETIC.md:241` documents a bug that was already fixed.
4. No documented way to run the derived sets small — `run_api derived` has no
   `--limit`.
5. `SYNTHETIC.md:7` misstates what the audit stage writes (into
   `data/svg_localization/`, hardcoded).
6. Two incompatible namings of "the three question types" between
   `SYNTHETIC.md` and `EVAL.md`.
7. `blindspots.md:110` says "two sizes"; the data has three.

## Breakages

- `report svgderived` → `TypeError: unsupported format string passed to
  NoneType.__format__` — crashes precisely when accuracy is 100%, i.e. on the
  study's own result. *(Fixed.)*
- `eval ablations` → `FileNotFoundError` aborts the whole pipeline when the
  ablations were never run; the step is not `optional=True`. *(Fixed.)*

## On the rule it was asked to find

> **Rule:** the generator is for building a *new* set into a *new* `--out`.

Found it. Enforcement was **partial**: the pipeline refused correctly, but
`generate.py` defaulted `--out` to the committed dataset with no guard.
*(Fixed — `--out` is now required.)*
