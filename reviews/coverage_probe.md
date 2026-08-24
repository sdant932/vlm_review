# Coverage probe — does the repo do what it says?

Agent brief: read the code and data, not the prose; check every headline claim
in the reference reports; report any number you cannot trace. Read-only.

## Verdict

> The study is real, the instrument is well built, and the central finding
> reproduces exactly from raw data. But one headline number is computed from a
> contaminated union of three different experiments, a "cannot drift" generated
> table is one-third hardcoded prose, several quoted numbers trace to nothing in
> the repo, and the repo's own measurements that would *weaken* the conclusion
> are computed, stored, and then omitted from the write-up.

## Reproduced independently, from raw JSONL

small 53/793 = 6.68% · large 35/794 = 4.41% · pooled 88/1587 = **5.55%** —
matching Tables 3/4 to the digit. `RESULTS_MANIFEST.md` sha256 prefixes verify.
**The core measurement is sound.**

## 1. ScreenSpot-Pro 1.8% pools three protocols

`eval.load_rows` globs `results/{dataset}__*.jsonl` and unions every match.

| source | n | hits |
|---|---:|---:|
| `think2000_native_r0` (the arm Table 1 names) | 1381 | 20 |
| `think2000_edge1568_r0` (different resize) | 150 | 5 |
| `tiled3x3_r0` | 50 | 4 |
| **published** | **1581** | **29 = 1.83%** |
| native only | 1581 | 20 = **1.27%** |

12.7% of scored items come from conditions Table 1 does not name, contributing
**31% of the hits**. The tiling arm is the one `blindspots.md` §4 explicitly
disowns: *"that changes how the model is called rather than measuring the
model, so we left it out of scope."*

Latent: two Sonnet-5 files sit in the same glob and only failed to contaminate
the number because `sorted()` places them before a later Haiku file.

## 2. Numbers that trace to nothing

| number | quoted in | problem |
|---|---|---|
| InfoVQA **74.3% → 59.4%** across megapixel quintiles | `blindspots.md` T2, `part3.md` §2 | no module computes it — string literal at `report.py:1167`. **The study's second headline blind spot.** |
| **6.7pp** text-volume control | T2 | same — the control that makes the resolution claim causal |
| SlideVQA **80.7%** format-corrected | §2 | artifact says **81.93** |
| **47%** of zeros format-equivalent | §2 | artifact says 126/256 = **49.2%** |
| **3.8pp** corrected gap | T2 | artifact says **−7.69** |
| **1.3/3.8/12.3%** by target-size bin | `part3.md` §3.2 | only a code comment |

## 3. Measurements that undercut the conclusion, all computed, none published

All eight ablation arms sit in `figures.json`. Three significant:

| arm | acc | baseline | |
|---|---:|---:|---|
| `cell_then_point` | **19.00%** | 6.67% | **significant** |
| `quadrant_mc` | **80.33%** | 66.33% | **significant** |
| `bbox` | **1.33%** | 6.67% | **significant (worse)** |

- A prompt-format change nearly **triples** exact localization. The report closes
  with *"the other a change to the model."*
- **`bbox` at 1.33% is fatal to `part3.md`'s central design choice** — it picks a
  box over a point as *"easier to train."*

## 4. Other undisclosed items

- **A third of the localization data is dropped.** The `medium` rung (793 items)
  appears nowhere. `blindspots.md` says "two sizes"; there are three. It is also
  the designed null control, and it **passes**.
- **117 out-of-range predictions (4.9%)**, asymmetric by rung — 0 small, 56
  medium, 61 large. Excluding them narrows the small→large gap from 2.28pp to
  1.91pp.
- **`report.py` prints "these cannot drift from the results"** over a table that
  is one-third literals — and the on-disk `tables.md` currently contradicts
  itself, T1/T2/T6 at full-study values beside T3/T4/T5 showing `n=16`, `0.00%`,
  `adjusted nan [nan–nan]`.
- Metric labels wrong in the generated table: SlideVQA labelled **ANLS**, scored
  `token_f1`; CharXiv labelled "judged exact match", actually `normalized_match`.

## 5. Honesty

**Good, better than typical.** `STRUCTURE.md` and `METHODOLOGY.md` are candid to
the point of self-harm — generator drift with reproductions, hardcoded
denominators, 10.1% run-to-run disagreement with *"any single-run difference
smaller than that is not a result."* Blind controls run on everything and read
honestly: AI2D 62.7% blind kills AI2D as a vision measure and the report says so.

**The pattern in what is missing:**

> Caveats about *method* are disclosed generously; measurements that would soften
> a *finding* are computed, stored in `figures.json`, and left out of the prose.
