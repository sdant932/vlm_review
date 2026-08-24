# Pipeline 3 — finetune_data

Agent brief: run it from the docs alone; work out which step costs money.
Compare against `part3.md`.

## Headline

Ran end to end for **~$0.05** (8 calls, capped). The agent found the paid step
from the docs and capped it unprompted. It also **refused to run
`--all`**, because the defaults overwrite unrecoverable artifacts.

## Verified true against the data

| `part3.md` claim | on disk |
|---|---|
| "1,513 targets over 384 images" | exact ✓ |
| "16 chart types, 10 themes, 9 fonts" | 16 / 10 / 9 ✓ |
| Six aspect ratios × four sizes, 510×681 → 1568×671 / 1072×1072 / 928×1238 | exact ✓ |
| "Every image delivered exactly as rendered" | `downscaled_by_api` False on all 1513 ✓ |
| Shape-held labels recorded separately | 505 shape / 1008 padded_text ✓ |
| All 9 example captions | every one matches a manifest row exactly ✓ |
| Background natural share 74/26 | 74.0 / 26.0 ✓ |
| GRPO advantage formula | implemented exactly ✓ |

## Not supported by the data

1. **Size balancing**: document says 3 bins @ 50/30/20; code is 5 bands @
   30/25/20/15/10 (`WEIGHTS = [6,5,4,3,2]`).
2. **Background balancing**: document says 80/20; data is 70/30 — and there is
   **no polarity weighting in the code at all**. The 70/30 is a side effect of
   theme-spreading.
3. **Three data sources: one exists.** No web scraper, no Opus labelling step,
   no `source` field on any record. 100% synthetic.
4. **The supervised/RL split does not exist.** `grep -rin grpo blindspot/` →
   three comments, zero code. No split field, no IoU score, no partition.
5. **The GRPO eligibility rule is stated and not implemented** — the document
   requires samples to disagree *and clear a minimum score*; the code checks
   spread only. Visible in the shipped artifact: all three groups flagged
   `usable_group: true` with mean IoUs of 0.011, 0.041, 0.008 — precisely the
   "everything near zero" case the document says to exclude.
6. **§3 headline numbers don't match the repo**: "99.9% / 95.6% / 5.3%" vs the
   repo's 99.73/100.00, 94.12/97.06, and 5.55%. **5.3% is not derivable from
   anything on disk.**

## Reproducibility

- Ladder **images** regenerate byte-identically — all 384 PNGs md5-match.
- Ladder **manifest** does not: same 1513 uids, but **91% have a different
  `target_text`/`box_px`**. Two fresh runs agree with each other, so this is
  drift since the artifact was built, not nondeterminism.
- `samples --seed 0` reproduces the shipped *distribution* exactly but only
  **5 of 20 uids**.

## Breakages

1. **`report_finetune figures` silently renders the WRONG label.**
   `_scene()` sets `complexity=1`; the ladder builds at `4`. Measured against the
   shipped manifest: **268/1513 target indices out of range, 159 pointing at a
   different string, only 1086 correct.** The committed figure is right *by
   luck*. No error, no warning. *(Fixed.)*
2. Same function **crashes** on any ladder with fewer than 90 scenes —
   `FIG_UID` is hardcoded to graph 89 — and `FINETUNE.md:110` recommends
   `--scenes-per-aspect 4` (24 scenes). *(Fixed.)*
3. **`make test` fails** under system Python 3.14 on a 1px difference in a
   "pixel-exact" box; passes 510/510 under the venv's 3.11. Same Pillow, same
   freetype. Worth knowing, because `part3.md` sells exact-ink boxes as having
   "no label-noise floor to subtract."

## Documentation defects

- `FINETUNE.md:136` says `--out` is required so a bare run cannot overwrite the
  reference artifact — true of the CLI, **false of the pipeline**, which
  hardcodes the reference path back in. *(Fixed.)*
- `FINETUNE.md:83` attributes "0.899 / 18.5%" to `part3.md`, which contains
  neither figure. Three files assert a fourth says something it doesn't.
- `ExactInk.verify` is documented but reachable from no CLI.
- `part3.md` mixes two datasets in adjacent tables without naming either.
