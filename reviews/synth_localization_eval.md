# Pipeline 2 — `synth_localization_eval`, re-run against ARTIFACTS.md

Agent brief: run it end to end from `docs/ARTIFACTS.md` alone, ~100 model-scored
questions, verify three named fixes, test the dataset guard adversarially,
compare against `takehome/outputs/report/blindspots.md` §5–7.

## Verdict

**It runs end to end and the study's central contrast reproduces hard: 99.39%
recognition (164/165) against 3.77% pointing (2/53), Fisher p = 3.5e-46.** All
three named fixes are real. But:

- **`ARTIFACTS.md` never once names this pipeline** (0 hits for
  `synth_localization_eval` in the file). It is organised by command, not by
  pipeline, and it does not cover 4 of the 11 steps at all. It does not stand on
  its own as an instruction source for this task.
- **The pipeline's `--out` guard has a live hole.** From any cwd other than the
  repo root, `--out data/svg_localization` is *not* refused. Only the
  generator's own backstop prevents the loss.
- **The n=0 abort was applied to `eval localization` only.** `eval ablations`
  silently overwrote a good 4,419-byte `ablations.json` with a 1,631-byte n=2
  one during this run. Same bug class, same directory, not fixed.
- **`--max-spend` is still not a hard stop on 6 of 7 `run_api` subcommands**,
  including two of this pipeline's four API steps. A $0.001 cap on `probe` bought
  6 calls and $0.020 — 20× over.
- New crash: `eval ablations` → `ZeroDivisionError` (eval.py:1325).

Cost: **$0.337, 112 billed calls, 84 new scored rows.**

---

## 1. Gate

| gate | expected | got | verdict |
|---|---|---|---|
| `pytest` | 583 passed, 4 skipped, 0 failed | `583 passed, 4 skipped in 38.44s` | **pass** |
| `tools verify-install` | ok | imports ok · 15 CLIs · 2 libs skipped · 4723 questions, first uid `svgloc:0000:small:00` · exit 0 | **pass** |
| `pipelines synth_localization_eval --list` | 4 stages, 11 steps, 4 API | audit 2 + run 4 + eval 3 + report 2 = 11, 4 API · exit 0 | **pass** |

Gate footnote: `pyproject.toml:44` sets `addopts = "-q"`. Typing `pytest -q`
yields `-qq`, which **suppresses the summary line entirely** — you get a wall of
dots and no pass/fail count, exit 0. Run bare `pytest`. Not a defect, but it
will cost the next agent ten minutes.

---

## 2. Every command, in order, with its ARTIFACTS.md line

| # | stage/step | command | ARTIFACTS.md | result |
|---|---|---|---|---|
| 1 | audit/ground-truth | `generate audit --data data/svg_localization --out …/verify/index.html` | **:114** | ok, 4723 audited, `consistency errors: 0` |
| 2 | audit/examples | `generate examples --data … --out …/examples/index.html` | **:115** | ok, 4723 questions |
| 3 | — (not a pipeline step) | `generate examples-derived --data …` | **:116** | ok, word_mc 1104 / counting 714 |
| 4 | run/localization | `blindspot.core --datasets svg_localization --max-spend 0.12` | **spine :20–21 only, no table row** | ok, 45 calls / $0.149 |
| 5 | run/localization:probe | `run_api probe --rung small --n 6 --max-spend 0.001` | **absent** (:102 misattributes) | ok, 6 calls / $0.020 |
| 6 | run/derived | `pipelines … --task text_existence counting --stage run --max-spend 0.06` | **spine :20–21 only** | ok, 53 calls / $0.138, 25 rows |
| 7 | run/localization:ablations | `run_api ablations --n 2 --max-spend 0.02` | **:45** (as input) | ok, 3 of 8 arms, 6 calls / $0.022 |
| 8 | eval/localization | `pipelines … --stage eval` → `eval localization` | **:43** | ok → `outputs/svgloc/summary.json` |
| 9 | eval/derived | same stage → `eval derived` | **:44** | ok → `outputs/svgderived/summary.json` |
| 10 | eval/ablations | same stage → `eval ablations` | **:45** | **CRASH** (§7), passed on retry |
| 11 | report/svgloc | `pipelines … --stage report` → `report svgloc` | **:89** | ok |
| 12 | report/svgderived | same stage → `report svgderived` | **:90** | ok |
| 13 | whole chain, free | `pipelines … --all --offline` | — | exit 0, 4 stages |
| 14 | documented invocation | `pipelines … --from eval --offline` (SYNTHETIC.md:19) | — | exit 0 |

Steps 4, 5 and 7 were run as bare commands rather than through
`--stage run`, and that is forced, not preference — see §4.

---

## 3. Spend and question count, per question type

| arm | cap asked | spent | **billed calls** | **rows kept** | waste |
|---|---:|---:|---:|---:|---:|
| localization (`core`) | $0.12 | $0.149 | 45 | 45 | 0 |
| localization baseline top-up (`core --uids`) | $0.05 | $0.008 | 2 | 2 | 0 |
| word_mc + counting + both blind arms (`run_api derived`, via pipeline) | $0.06 | **$0.138** | 53 | **25** | **28** |
| probe, sonnet-5 sanity arm | **$0.001** | **$0.020** | 6 | 6 | 0 |
| ablations, 3 of 8 arms | $0.02 | $0.022 | 6 | 6 | 0 |
| **total** | | **$0.337** | **112** | **84** | 28 |

New rows by question type (uid diff against a snapshot taken before spending):

| question type | where | new rows this run | scored in the eval (pooled with pre-existing) |
|---|---|---:|---:|
| `point` | main manifest | 25 | 94 |
| `relation` | main manifest | 12 | 40 |
| `reverse` | main manifest | 10 | 43 |
| `word_mc` | `word_mc/` | 22 (+1 blind) | 116 (+2 blind) |
| `counting` | `counting/` | 1 (+1 blind) | 70 (+1 blind) |

**The dataset carries five question types, not three.** `docs/runme/SYNTHETIC.md:50–56`
and `blindspots.md:110` both say three. The main manifest is
`point` 2,380 / `relation` 1,143 / `reverse` 1,200 = 4,723; the derived sets add
`counting` 714 and `word_mc` 1,104. `relation` and `reverse` are scored by
`eval localization` and land in `summary.json["text"]`, but the pipeline's
`--task` flag offers no name for them and `blindspots.md` never reports them.
They come free with the localization call, so I exercised all five.

`reverse` is the most interesting of the two undocumented arms: it asks what text
sits at a given point — the exact inverse of pointing. It scores **23.3% EM
(n=43)** against `relation`'s **57.5% EM (n=40)**. That is a second, independent
probe of the "knows what, not where" claim, from the other direction, and nobody
reports it.

---

## 4. Why steps 4/5/7 could not be run through `--stage run`

`flow.py:73` splits `--max-spend` as `share = min(self.spend, max_spend)` — a
**per-step** ceiling, not a split, despite `flow.py:113` advertising
`"USD ceiling, split across API steps"`. With four API steps selected, `--max-spend X`
authorises up to `4X`.

That would be survivable if the ceiling bound. It does not. `metered()`
(run_api.py:75–110), the sliding-window fix, has **exactly one call site** —
run_api.py:845, inside `derived_run`. Every other subcommand still queues all
futures up front:

| subcommand | line | metered? | in this pipeline? |
|---|---|---|---|
| `derived` | 845 | **yes** | yes |
| `official` | 426 | no | no |
| `ablations` | 640 | **no** | **yes** |
| `probe` | 703 | **no** | **yes** |
| `controls` | 969 | no | no |
| `grid` | 1073 | no | no |
| `coord-probe` | 1143 | no | no |

`core.run_one` (core.py:1963) never checks `budget.exhausted()`, so an unmetered
submission is a billed call regardless of the cap.

Measured, not inferred:

```
$ python -m blindspot.run_api probe --rung small --n 6 --max-spend 0.001
probe sample: 6 point questions (Counter({'small': 6}))
  claude-sonnet-5   native  click-in-bbox 0.0% (0/6, 0 unusable)  small 0% (n=6)  $0.02
total $0.020
```

**6 calls and $0.020 against a $0.001 cap — 20×.** The pipeline hardcodes
`probe --n 100` (pipelines.py:185) and `ablations --n 300` (pipelines.py:195) with
no override. Running `--task localization --stage run` at *any* `--max-spend`
therefore bills ≥100 probe calls plus ≥300 ablation calls for the first arm —
400+ questions minimum, 4× the whole brief. That is why I ran them by hand at
`--n 6` / `--n 2`. **This pipeline's `run` stage cannot be exercised inside a
100-question budget as shipped.**

### Residual of the fix that *was* made

The `derived` step is bounded now, but not clean. One `--max-spend 0.06`:

```
svg_word_mc: 736 questions at rungs ['large', 'small']
  !! budget ceiling $0.06 reached -- not submitting anything further
  svg_word_mc__…jsonl: +22 rows ($0.083 total, 0 without a prediction)
  !! budget ceiling $0.06 reached -- not submitting anything further
  svg_word_mc__blind_…jsonl: +1 rows ($0.099 total, 0 without a prediction)
svg_counting: 476 questions at rungs ['large', 'small']
  !! budget ceiling $0.06 reached -- not submitting anything further
  svg_counting__…jsonl: +1 rows ($0.125 total, 0 without a prediction)
  !! budget ceiling $0.06 reached -- not submitting anything further
  svg_counting__blind_…jsonl: +1 rows ($0.138 total, 0 without a prediction)
done | $0.138 | 53 calls
```

Two distinct problems. (a) **One `Budget` is shared across all four arms and never
apportioned**, so arm 1 eats it and arms 2–4 get one row each; the blind controls
and the entire counting set are structurally starved. (b) On each starved arm the
window of 8 is already in flight when the ceiling trips, and `g.cancel()`
(run_api.py:99) cannot stop a running future — so **28 of 53 calls were billed and
discarded**. That is the same failure mode the previous review priced at
$3.24/1,417 calls, reduced by ~50× but not eliminated.

Prior review finding #4 — "no documented way to run the derived sets small" —
**is still open.** `run_api derived --help` has no `--limit` and no `--n`.

---

## 5. The rule, and whether the tooling enforces it

`docs/ARTIFACTS.md:179–187`:

> **Drift note.** The scene generator no longer reproduces the committed
> `data/svg_localization`: same seed, 73% of shared uids get different ground
> truth, and some uids bind to a different question entirely. `results/*.jsonl` is
> keyed by uid, so regenerating in place would join existing answers to the wrong
> questions. `generate scenes` therefore requires an explicit `--out` — no default — and both
> the pipeline and the Makefile refuse to target the committed set.

Attacked from nine angles. `data/svg_localization/manifest.jsonl` verified by
`shasum` before and after every single attempt; `git status --porcelain
data/svg_localization` clean throughout.

### 5a. `blindspot.generate scenes` — solid

| cwd | `--out` | result |
|---|---|---|
| repo root | *(omitted)* | exit 1, "`--out` is required and has no default" |
| repo root | `data/svg_localization` | exit 1, "refusing" |
| repo root | `data/svg_localization/` (trailing slash) | exit 1, "refusing" |
| repo root | `./data/svg_localization` | exit 1, "refusing" |
| repo root | `data/../data/svg_localization` | exit 1, "refusing" |
| repo root | absolute path | exit 1, "refusing" |
| `/tmp/otherdir` | absolute path | exit 1, "refusing" |
| `/tmp/otherdir` | absolute + trailing slash | exit 1, "refusing" |
| `/tmp/otherdir` | absolute with `/./` inserted | exit 1, "refusing" |
| `/tmp/otherdir` | `data/svg_localization` | exit 0 — **correct**, wrote `/tmp/otherdir/data/svg_localization`, a genuinely new dir |

**The named fix is real.** `generate.py:2702` requires `--out`; `generate.py:2721`
additionally refuses the *value*, resolved against `_REPO_ROOT` (generate.py:2693),
not cwd. Refuses outright, not merely "give me an `--out`". No angle got through.

### 5b. `blindspot.pipelines --out` — **holed**

From the repo root all six spellings are refused (exit 1). From
`/tmp/otherdir`:

| `--out` | expected | actual |
|---|---|---|
| absolute path to the committed set | refuse | **refuse**, exit 1 |
| `data/svg_localization` | refuse | **exit 0 — guard does not fire** |
| `data/svg_localization/` | refuse | **exit 0 — guard does not fire** |

`pipelines.py:412` compares `Path(opts["out"]).resolve()` — resolved against the
**cwd** — with `(flow.ROOT / COMMITTED).resolve()`. Only one of the two readings
is checked, directly contradicting its own comment at pipelines.py:407–411
("Resolve BOTH against the repository root, not the cwd"). This matters because
`flow.py:168` runs every step with `cwd=ROOT`, so the relative path *does* land
on the committed dataset:

```
$ cd /tmp/otherdir
$ python -m blindspot.pipelines synth_localization_eval --out data/svg_localization --stage generate --dry-run
--- scenes+localization
    -m blindspot.generate scenes --count 200 --complexity 4 --seed 17 --out data/svg_localization
```

Run for real, the generator's backstop catches it (exit 1, dataset intact,
checksums unchanged) — but the pipeline had already scheduled the write, and
printed this while doing so (pipelines.py:165):

```
why: builds a NEW set in data/svg_localization; the committed set is untouched
```

A false assertion emitted by the guard layer, about the file it is guarding.

**The fix is one line.** `pipelines.py:330–339` already defines `_resolutions()`,
whose docstring is a precise description of this exact bug ("Checking one reading
and not the other is exactly how a guard silently fails to fire, so both are
checked"). It is wired to `refuse_finetune_out` (pipelines.py:419) and **not** to
the synth guard. `finetune_data` is protected against an attack that
`synth_localization_eval` is not.

### 5c. The Makefile — claim is inaccurate

`ARTIFACTS.md:184` says "both the pipeline and the Makefile refuse to target the
committed set". The Makefile does **not** check the value:

```
$ make dataset OUT=data/svg_localization COUNT=2
python -m blindspot.generate scenes --count 2 --complexity 4 --seed 17 --out data/svg_localization
scenes: refusing --out data/svg_localization
```

`Makefile:37–39` only requires `OUT` to be *set*. The refusal comes from
`generate.py:2721`. Right outcome, wrong attribution — and if the generator's
backstop is ever removed, the Makefile route opens.

**Net: the dataset is safe, but only because of one guard, at
`generate.py:2721`. Two of the three layers ARTIFACTS.md:184 credits do not
actually check the value.**

---

## 6. My numbers vs the study's

`blindspots.md` Table 4 (:164–170) and Table 3 (:139–144).

| task | study small | study large | study n | mine small | mine large | mine n |
|---|---:|---:|---:|---:|---:|---:|
| Is this word present? | 99.73% | 100.00% | 736 | **100.00%** (59) | **100.00%** (48) | 107 |
| Count the elements | 94.12% | 97.06% | 476 | **97.67%** (43) | **100.00%** (15) | 58 |
| Point at it | 6.68% | 4.41% | 1,587 | **3.70%** (27) | **3.85%** (26) | 53 |

### The contrast — reproduces

| | study | mine |
|---|---|---|
| recognition (word_mc + counting, small+large) | ~99.7% / 94–97% | **99.39%** (164/165) |
| pointing (small+large) | 5.55% | **3.77%** (2/53) |
| ratio | ~18× | **26.3×** |
| vs random click | — | 12.7× chance (all rungs, 3/94) |

Fisher exact, 164/165 vs 2/53: **p = 3.5e-46**. The study's 5.55% sits inside my
Wilson CI for pointing (1.09%–8.97%, all rungs). **The abstract's claim
(`blindspots.md:14–17`) — "knows what is on the page but not where" — reproduces
directionally and at overwhelming significance even at 1/30 of the sample.**

### The precision ladder — reproduces

§6 (:120–122): "Loosen the requirement to the right quarter of the image and
accuracy jumps more than tenfold."

| rung | exact | 2×2 strict | jump |
|---|---:|---:|---:|
| small | 3.70% | 40.7% | **11.0×** |
| large | 3.85% | 61.5% | **16.0×** |

Study Table 3 has 2×2 at 59.4% (small) / 71.9% (large). Mine are lower in
absolute terms with n=27/26, but the shape and the >10× jump both hold, and
`large > small` on the 2×2 tolerance holds too. Full small-rung ladder, strict:
2×2 40.7% → 3×3 33.3% → 4×4 18.5% → 8×8 14.8% → 16×16 11.1%; ratio-to-chance
rises monotonically 1.6× → 3.0× → 3.0× → 9.5× → 28.4×.

### What does NOT reproduce, and how weak the evidence is

**Be blunt about n.** My pointing arm is 2 hits. Two.

| study claim | mine | honest verdict |
|---|---|---|
| pointing *falls* with size (6.68% small > 4.41% large), "moves in the opposite direction" (:172) | 3.70% small **<** 3.85% large | **not reproduced.** k=1 vs k=1, Fisher p = 1.000. Untestable at this n; this is a coin flip, not a contradiction. |
| blind control, word presence 22.6% | 0% at **n=2** | **untestable.** Structurally starved by §4(a). |
| blind control, counting 12.0% | 0% at **n=1** | **untestable.** Same cause. |
| medium-vs-large null control | n=8, 0 hits both arms | **untestable.** 8 complete triples out of 77 uids with any rung. |
| small-vs-medium resolution effect | n=9, 0 hits both arms | **untestable.** |

The pairing block reports `complete_triples: 8, dropped_incomplete: 69`. **Every
paired comparison in the study is out of reach on a partial budget**, because a
budget-truncated run samples uids independently per rung and almost never
completes a triple. Only the unpaired contrast survives — which happens to be the
headline, so the headline is the one thing a cheap run *can* check.

Harness sanity: `probe` put **claude-sonnet-5** on the same small-rung images and
scored **0/6**. Consistent with a genuine perceptual limit rather than a broken
harness, but n=6 carries no weight and I will not claim otherwise.

---

## 7. Artifacts — produced or failed

| artifact | ARTIFACTS.md | status | evidence |
|---|---|---|---|
| `data/svg_localization/verify/index.html` | :114 | **produced** | 253,362 B, 149 `<img>`, 75 unique srcs, **0 broken**, min 13,854 B |
| `data/svg_localization/examples/index.html` | :115 | **produced** | 24,152 B, 15 `<img>`, 0 broken, min 32,171 B |
| `data/svg_localization/counting/examples.html` | :116 | **produced** | 12,648 B, 9 `<img>`, 0 broken |
| `data/svg_localization/word_mc/examples.html` | :116 | **produced** | 11,716 B, 6 `<img>`, 0 broken |
| `results/svg_localization__*.jsonl` | spine :20–21 | **produced** | 130 → 177 rows |
| `results/svg_{counting,word_mc}__*.jsonl` + `__blind_*` | spine :20–21 | **produced** | 70 / 116 / blind 1 / blind 2 |
| `results/svgloc_abl_{repeat,quadrant_mc,crop}__*.jsonl` | :45 (input) | **partial** — 3 of 8 arms | budget-truncated by design |
| `results/svgloc_ablation_uids.json` | :45 (input) | **produced** | 2 uids |
| `results/svg_localization__probe_small_*.jsonl/_uids.json/_summary.json` | **absent from ARTIFACTS.md** | **produced** | 3 files |
| `outputs/svgloc/summary.json` | :43 | **produced** | 94 point / 83 text / 0 unusable |
| `outputs/svgderived/summary.json` | :44 | **produced** | counting 70, word_mc 116 |
| `outputs/svgloc/ablations.json` | :45 | **failed, then produced on retry** | §8 |
| `outputs/svgloc/report.html` | :89 | **produced** | 48,091 B, **24 `<img>`, 24/24 resolve, 6,977–55,556 B, none <1 KB** |
| `outputs/svgloc/assets/*.jpg` | **absent from ARTIFACTS.md** | **produced** | 24 files |
| `outputs/svgderived/report.html` | :90 | **produced** | 19,673 B, 0 `<img>` **by design** — the reference build at `takehome/outputs/svgderived/report.html` also has 0 |

**No image loss anywhere.** Checked by decoding every `data:` URI and stat-ing
every external `src`. Reference comparison: `takehome/outputs/svgloc/report.html`
is 61,955 B with the same 24 `<img>`; mine is smaller because it summarises 94
scored points instead of 2,380, not because anything dropped.

`report svgderived` completed with `word_mc` at exactly **100.00%** — the input
that used to raise `TypeError: unsupported format string passed to
NoneType.__format__`. **That prior fix is confirmed.**

Determinism bonus: `verify/index.html`, `examples/index.html` and both
`examples.html` are **tracked in git** (815 tracked files under `data/`), and
after regenerating all four `git status --porcelain data/svg_localization` is
**empty**. The audit family reproduces byte-identically against the committed
copies.

---

## 8. Everything that broke, with exact text

### 8a. NEW — `eval ablations` → `ZeroDivisionError`

```
Traceback (most recent call last):
  File "/Users/sdoveh/gitrep/haiku-perception-blindspots/blindspot/eval.py", line 2119, in <module>
    raise SystemExit(main())
  File "/Users/sdoveh/gitrep/haiku-perception-blindspots/blindspot/eval.py", line 2115, in main
    return a.fn(a)
  File "/Users/sdoveh/gitrep/haiku-perception-blindspots/blindspot/eval.py", line 1363, in cmd_ablations
    s = analyse_ablations(uids)
  File "/Users/sdoveh/gitrep/haiku-perception-blindspots/blindspot/eval.py", line 1325, in analyse_ablations
    arm_acc = hits / len(uids)
              ~~~~~^~~~~~~~~~~
ZeroDivisionError: division by zero
```

Cause: `eval.py:1272`, `uids = [u for u in rows if u in base]` — the ablation arm
intersected with the **baseline** localization predictions. `abl_sample()` draws
its own sample with no regard for what has been scored, so on any partial run the
intersection is routinely empty. Mine was exactly empty (2 ablation uids, 175
scored localization rows, overlap `set()`). The guard at `eval.py:1270`
(`if not rows: continue`) checks the arm file is non-empty but never the
intersection. `if not uids: continue` fixes it.

**`ARTIFACTS.md:45` is incomplete as a result.** It states the prerequisites as
`results/svgloc_abl_* + svgloc_ablation_uids.json`. Both were present and the
command still crashed. The real third prerequisite — the main
`results/svg_localization__*` must cover the sampled ablation uids — is
undocumented.

Recovered by scoring the two missing baseline uids (`core --uids …`, 2 calls,
$0.008) and re-running; `ablations.json` then wrote at n=2.

Mitigation that *did* work: the step is `optional=True` (pipelines.py:215), so the
pipeline warned and continued to exit 0. The previous review's `eval ablations`
`FileNotFoundError` abort fix holds.

### 8b. NEW — the n=0 abort was fixed for one command out of three

`grep -n ABORT blindspot/eval.py` returns exactly two hits, both inside
`cmd_localization` (eval.py:881, :888). `cmd_derived` and `cmd_ablations` have
none.

Consequence, observed in this run: `eval ablations` **silently replaced** a good
4,419-byte `outputs/svgloc/ablations.json` (Aug 23 16:50) with a 1,631-byte n=2
file, exit 0, no warning. That is precisely the "silent data loss" the fix
comment at eval.py:875–879 describes, in the same output directory, one command
over. `eval derived` did the same to `outputs/svgderived/summary.json` (n=70/116
over whatever was there).

**Disclosure: I destroyed that file.** It is gitignored, and there were no
`results/svgloc_abl_*` on disk to rebuild it from, so it was already
unreproducible before I touched it. A byte-identical 4,419-byte copy survives
read-only at `/Users/sdoveh/gitrep/takehome/outputs/svgloc/ablations.json` and
can be copied back. I backed up `svgloc/summary.json` and
`svgderived/summary.json` before starting and did not back up `ablations.json`,
because `ARTIFACTS.md:45` gives no hint that the command overwrites without
checking.

### 8c. `generate audit --out <DIR>` — fixed at the call site only

The named regression is gone: `pipelines.py:173` now passes
`--out data/svg_localization/verify/index.html`, and the audit stage runs clean
(`4723 questions audited`, `consistency errors: 0`, exit 0). But the underlying
command is unchanged and still dies on a directory:

```
$ python -m blindspot.generate audit --data data/svg_localization --out /tmp/auditdir
  File ".../pathlib.py", line 1044, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
IsADirectoryError: [Errno 21] Is a directory: '/tmp/auditdir'
```

`generate.py:2816` does `out = a.out or (a.data / "verify" / "index.html")` — it
defaults to a file path but never validates a supplied one. Raw traceback, no
message. `ARTIFACTS.md:114` happens to show the correct file-path form, so a
reader following ARTIFACTS.md is safe; a reader following the `--data <dir>`
shape of the neighbouring flags is not.

---

## 9. Fix verification, itemised

| fix under test | verdict | evidence |
|---|---|---|
| `generate scenes` refuses `--out data/svg_localization` **outright**, not merely requiring `--out` | **CONFIRMED** | generate.py:2702 (required) *and* :2721 (value refused). 10 angles, 2 cwds, all correct — §5a |
| … from the repo root | **CONFIRMED** | 5/5 spellings refused |
| … from another directory | **CONFIRMED** | 3/3 absolute spellings refused; relative correctly allowed (targets a new dir) |
| pipeline `audit` stage no longer `IsADirectoryError` | **CONFIRMED** | exit 0, 4723 audited, 0 consistency errors. Fixed at the call site; the bare command still crashes — §8c |
| `eval localization` ABORTS rather than writing n=0 over a good summary | **CONFIRMED** | both branches, exit 2, `shasum` of `summary.json` byte-identical before and after |
| — missing file | | `ABORT: no result file at results/svg_localization__nosuchtag_r0.jsonl (tag 'nosuchtag_r0'). Refusing to overwrite outputs/svgloc/summary.json with an n=0 summary.` |
| — file present, nothing scoreable | | `ABORT: results/svg_localization__zzzprobe_r0.jsonl has 1 line(s) but nothing scoreable (1 unique uid(s), 1 unusable, 0 point, 0 text). Refusing to overwrite outputs/svgloc/summary.json with an n=0 summary.` |
| (prior) `--max-spend` is a hard stop | **PARTIAL** — `derived` only, 1 of 7 subcommands — §4 |
| (prior) `report svgderived` `TypeError` at 100% | **CONFIRMED FIXED** | ran with word_mc at 100.00% |
| (prior) `eval ablations` `FileNotFoundError` aborts the pipeline | **CONFIRMED FIXED** | `optional=True`, warns and continues |

---

## 10. ARTIFACTS.md inaccuracies, with file:line

| # | file:line | claim | reality |
|---|---|---|---|
| 1 | `docs/ARTIFACTS.md` (whole file) | — | **The string `synth_localization_eval` appears 0 times.** No stage list, no run order, no `--task`/`--out`/`--max-spend`. The doc is command-indexed; the brief is pipeline-indexed. It cannot be the sole instruction source without `docs/STRUCTURE.md:113` or `docs/runme/SYNTHETIC.md`. |
| 2 | `docs/ARTIFACTS.md:102` | "`results/probe_uids.json`, written only by `run_api probe`" | Written at `run_api.py:1132`, inside **`cmd_coord_probe`** — subcommand `coord-probe`. `run_api probe` (this pipeline's step) writes `svg_localization__probe_<rung>_{uids,summary}.json` + a results jsonl. After running `probe`, `results/probe_uids.json` does not exist. |
| 3 | `docs/ARTIFACTS.md:45` | needs `results/svgloc_abl_*` + `svgloc_ablation_uids.json` | Incomplete. Also needs the sampled uids to be present in `results/svg_localization__*`. Both stated prerequisites present → `ZeroDivisionError` (§8a). |
| 4 | `docs/ARTIFACTS.md:184` | "both the pipeline and the Makefile refuse to target the committed set" | The pipeline refuses only when `--out` is absolute or cwd is the repo root (§5b). The Makefile never checks the value at all — `Makefile:37–39` only requires `OUT` to be set (§5c). |
| 5 | `docs/ARTIFACTS.md:85` | standalone HTML pages are "self-contained" | `outputs/svgloc/report.html` links **24 external JPEGs** in `outputs/svgloc/assets/`. Move the HTML alone and every image breaks. The reference build is the same, so the doc is wrong, not the build. `assets/` is not listed as an artifact anywhere. |
| 6 | `docs/ARTIFACTS.md:4` | "Nothing here is committed" | Lines :114–116 produce `verify/index.html`, `examples/index.html` and both `{counting,word_mc}/examples.html`, all four **tracked in git**. Line :5–6 partially walks it back, but the lead sentence is false for the audit table directly below it. |
| 7 | `docs/ARTIFACTS.md:20–21` | spine covers `blindspot.core` / `blindspot.run_api` | No table row for `results/svg_localization__*`, `results/svg_{counting,word_mc}__*`, or any `probe`/`ablations` result file. 4 of the pipeline's 11 steps have no artifact entry. |
| 8 | `docs/ARTIFACTS.md:172` | "`generate scenes/questions` output — yes, given `--seed`" | Directly contradicted by the drift note at :179–182 twelve lines later. The cell's own caveat ("but see the drift note below") is doing a lot of work for something the same file elsewhere calls a source-of-truth hazard. |

Secondary sources, since they were consulted:

| file:line | problem |
|---|---|
| `docs/runme/SYNTHETIC.md:264–267` | Still documents the **fixed** n=0 bug as live: "`blindspot.eval localization` reports `0 point / 0 text` and **writes an n=0 summary over a good one**, exit 0. Back up … before running it." It now aborts with exit 2. |
| `docs/runme/SYNTHETIC.md:50–56` | "The three question types". The dataset has five (§3). |
| `blindspots.md:110` | "three kinds of question". Same. |
| `blindspot/flow.py:113` | `--max-spend` help says "split across API steps". It is `min(step_spend, max_spend)` per step (flow.py:73) — with 4 API steps, `--max-spend X` authorises `4X`. |
| `blindspot/pipelines.py:407–411` | Comment claims both readings of `--out` are resolved. Only one is (§5b). |
| `blindspot/pipelines.py:165` | Step note prints "the committed set is untouched" while `--out` points at the committed set. |

---

## 11. Caveats on these numbers

1. **Pooled, not clean-room.** `results/` already held 130 localization / 69
   counting / 94 word_mc rows from a prior agent. `eval` pools them. Every "mine"
   figure in §6 is that pool. My own contribution is isolated in §3 by uid diff
   against a pre-spend snapshot; the deltas match the reported call counts
   exactly, so no cross-contamination was detected.
2. **Another agent was writing to this repo during the run.** `git status` moved
   from `?? finetune/` at start to `M blindspot/report_pages.py` + `?? reviews/finetune_data.md`
   at finish. Neither is mine. I touched no file outside this report,
   `results/`, `outputs/` and the four regenerated (and byte-identical) audit
   pages.
3. **112 billed calls against a ~100 brief.** 12 over: 6 were the deliberate
   $0.001-cap probe that proves §4, 6 were the 3-arm ablation slice needed for
   `ablations.json`. Overshoot on `core` (45 calls vs a $0.12 ≈ 32-call cap) is
   the in-flight window, an inherent ~8-call slack in the metered design, not a
   new bug.
4. **`--offline` end-to-end passes** (`--all --offline`, `--from eval --offline`,
   both exit 0), so the stage wiring is sound independent of everything above.
