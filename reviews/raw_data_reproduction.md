# Reproduction against the original study's raw data

**Date:** 2026-08-23
**Input:** `/Users/sdoveh/gitrep/takehome/results/` — 37,518 rows across 45 JSONL files (read-only)
**Method:** symlink the reference `results/` and `data/` into an empty tree, run the
consolidated code's full chain, compare every leaf value in every JSON artifact against
the reference artifact.

```bash
cd /tmp/rawtest && ln -s /Users/sdoveh/gitrep/takehome/results results
              ln -s /Users/sdoveh/gitrep/takehome/data    data
python -m blindspot.eval   aggregate localization derived ablations
python -m blindspot.report summary data examples tables index paste
python -m blindspot.report_pages causes
```

## Result

| artifact | leaf keys compared | identical | differing |
|---|---:|---:|---:|
| `outputs/svgloc/summary.json` | 1,094 | **1,094** | 0 |
| `outputs/svgderived/summary.json` | 1,494 | **1,494** | 0 (+2 new keys) |
| `outputs/svgloc/ablations.json` | 135 | **135** | 0 |
| `outputs/report/figures.json` | 2,350 | 2,347 | 3 |
| `outputs/summary.json` | 604 | 545 | 59 |

`causes`: **16/16 pages** reproduce, each exactly 6 bytes smaller. The 6 bytes are the
link fragment `aug22/`, from renaming `outputs/aug22/` to `outputs/report/`. No other
difference on any page.

**All 62 differing values trace to a single root cause: ScreenSpot-Pro.** 53 are
ScreenSpot-Pro keys directly; the other 6 are `primitives.localization_point.pooled*`,
which carry ScreenSpot-Pro's value unchanged. Every other benchmark, every localization
ladder rung, every ablation and every derived-set number reproduces bit-exactly.

## The ScreenSpot-Pro difference is a fix, and the mechanism is confirmed

Reference reports **1.8343%** (29/1581). This code reports **1.6445%** (26/1581).

`results/` holds eight ScreenSpot-Pro tags. The pre-fix `load_rows` unioned them keyed by
uid, so a later tag overwrote an earlier one. Recomputing click-in-bbox with
`core.point_in_bbox` reproduces both numbers exactly and explains the gap:

| arm | n | hits |
|---|---:|---:|
| `haiku-4-5_think2000_native_r0` — the intended arm | 1,581 | **26** |
| `think2000_edge1568_r0` — the resolution ablation | 200 | 5 |
| `tiled3x3_r0` — the tiling ablation | 50 | 4 |
| `haiku-4-5_official_r0` | 1,581 | 0 (`pred` is null; rescore-only) |
| union across all eight tags | 1,581 | **29** |

Provenance of the 1,581 rows the union actually scored:

```
haiku-4-5_think2000_native_r0   1381
think2000_edge1568_r0            150     <- different input resolution
tiled3x3_r0                       50     <- different protocol
```

The union both adds and destroys credit, and the arithmetic closes exactly:

- gains 8 uids credited by a foreign arm (4 from `tiled3x3_r0`, 4 from `edge1568`)
- loses 5 uids where a canonical **hit** was overwritten by a foreign **miss**
- `26 + 8 - 5 = 29` ✓

**Why this matters beyond 0.19pp.** The contamination is not random: **200 of 1,581 rows
(12.7%) came from the two resolution/tiling ablation arms** — precisely the experiments
run to test whether resolution or tiling changes the ScreenSpot-Pro result. The baseline
those ablations were meant to be compared against silently contained them. The direction
of the bias is also order-dependent, since the union resolved collisions by sorted tag
name rather than by protocol.

`eval.py`'s `load_rows` now pins `CANONICAL_TAG = "haiku-4-5_think2000_native_r0"` and
refuses a mixed set of models or protocols outright.

## Consequence for the write-up

`blindspots.md` and `docs/DATASETS.md` still quote **1.8%** for ScreenSpot-Pro. The
generated pages quote 1.6%. The prose is the stale side and needs updating to 1.6%
(26/1581) — see `reviews/causes_analysis.md`, which found the same discrepancy
independently from the rendered pages.
