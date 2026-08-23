# Evaluation instructions

Instructions for the agent evaluating `data/svg_localization`. Read this before
running anything.

Dataset as generated: **200 scenes, 4,723 questions** — 2,380 `point`,
1,143 `relation`, 1,200 `reverse` — across 16 chart/diagram types, 10 themes,
9 fonts, `--complexity 4`, `--seed 17`.

---

## 1. What this dataset is testing

Three hypotheses, in priority order. Everything you report should serve one of
them; anything else is secondary.

**H1 — Localization degrades smoothly with required precision, not absolutely.**
The ScreenSpot-Pro arm scored 1.65% exact but 61.8% at 2×2 — a smooth decay whose
ratio-above-chance *rose* the whole way. Test whether the same **shape** appears
here, where target size and image size are knobs rather than confounds. Test the
shape, not the numbers — see §1.1.

**H2 — Effective resolution governs accuracy.** The resolution ladder is the
point of this set. See §4 — a primary deliverable, not an appendix.

**H3 — How much of the deficit is coordinate *emission* rather than
perception.** `relation` asks about position while requiring **no coordinates at
all**, in either the question or the answer, so comparing it against `point`
bounds the expression component.

> The earlier study answered H3 with a *grid control* — a magenta 4×4 grid drawn
> over the screenshot, with the model naming a cell. That is **not part of
> ScreenSpot-Pro**; it was a bespoke ablation in `scripts/run/grid_control.py`. It is
> deliberately **not** reproduced here: measured on a trial build, the overlay
> covered the gold text in **746 of 2,400 grid questions (31%)**. Do not add a
> grid arm to this dataset.

---

## 1.1 What is and is not comparable to ScreenSpot-Pro

This dataset shares ScreenSpot-Pro's **method**. It does not share its **task**.
Conflating the two will produce a wrong report.

**Comparable — the method is identical, and verified rather than assumed:**

* the same `POINT_INSTRUCTION`, the same 0–1000 answer space, the same
  `parse_response`;
* the same `blindspot.core.scoring.point_in_bbox` against a normalized **widget box**
  (§3.1);
* the same analysis structure: precision curve, lenient variant, distance bands,
  area quintiles, Wilson intervals.

**Not comparable — the task differs in four ways:**

| | ScreenSpot-Pro | here |
|---|---|---|
| the ask | *"stop the bilibili download in android virtual machine in android studio"* | `the text "Index Intake"` |
| what that requires | resolve a **functional intent** to a visual referent | **match a string already quoted in the question** |
| targets | 604 icon / 977 text | 2,380 text, **zero icons** |
| target area | 0.0017%–4.73%, median 0.036% | 0.039%–2.75%, median 0.145% |

Quoting the answer string makes this strictly easier and exercises a different
ability. The floor is ~23× larger than ScreenSpot-Pro's, so the small targets
that drove its near-zero score are absent. The images are synthetic vector
charts, not photographic UI chrome.

**Therefore:**

* **Do not report any number here against the 1.65% figure**, or any published
  ScreenSpot-Pro number — not as a delta, not as "better than", not in a shared
  column.
* **Do not claim the icon-vs-text finding** (1.16% vs 1.94%) is confirmed or
  refuted. There are no icons here; it is untestable on this set.
* What *is* valid is every **within-dataset** contrast, because those hold the
  task constant: `small` vs `medium`, `medium` vs `large`, `point` vs
  `relation`, the area quintile gradient, and the **shape** of the precision
  curve.

The only honest cross-dataset sentence is about shape — "the curve decays
smoothly, as it did on ScreenSpot-Pro" — never about magnitude.

---

## 2. Running it

`point` rows follow the repo's ScreenSpot-Pro convention, so use the existing
path rather than writing a new one:

* `question` is the element description only — **prepend
  `blindspot.core.prompts.POINT_INSTRUCTION`**, do not send it bare.
* The model answers `{"x": 0..1000, "y": 0..1000}`;
  `blindspot.core.prompts.parse_response` divides by 1000.
* Score with `blindspot.core.scoring.point_in_bbox(pred, gold)` against
  **`gold_bbox_norm`**.

`relation` and `reverse` are short free-text; send `question` as-is.

Use the same model settings as the main study (`claude-haiku-4-5-20251001`,
thinking enabled, 2000 tokens) or the comparison is not comparable. Record the
settings in your output.

---

## 3. Metrics — keep these tight

### 3.1 The gold box is the *widget*, not the glyphs

`gold_bbox_norm` is a **hit box**: the region a user could click, which is what
ScreenSpot-Pro annotates. It is one of two things, recorded in `hit_source`:

| `hit_source` | n | what it is |
|---|---|---|
| `shape` | 820 | the enclosing widget — a node rectangle, a state circle |
| `padded_text` | 1,560 | the glyph ink grown by button padding (0.42×font wide, 0.34×font tall) |

A shape only qualifies if it fully contains the label's ink and covers under 6%
of the canvas — that guard keeps panel frames and plot backgrounds from being
mistaken for buttons. The ink box is published separately as
`text_ink_bbox_norm` and is guaranteed to sit inside the hit box.

**Score against `gold_bbox_norm`.** Do not score against `text_ink_bbox_norm` —
that is the glyph outline, it is ~3× smaller (mean area 0.084% vs 0.255%), and
scoring it would be a harder and different task from the one ScreenSpot-Pro
poses. Do not score against `gold_bbox_px`; the pixel fields exist for the visual
audit.

### 3.2 Click-in-bbox — primary

A prediction is correct iff it falls inside the hit box:

```
correct = x0 <= x <= x1 and y0 <= y <= y1     # all normalized 0-1
```

No tolerance, no padding beyond what the hit box already carries, no partial
credit, no IoU.

**Always report accuracy against chance.** Chance is the mean hit-box area
fraction, because a uniform random click lands inside a box with probability
equal to its area share:

```
chance = 0.25472%     (mean target_area_frac over point rows)
```

Report raw accuracy **and** ratio-above-chance. Reporting 2% without the ratio is
what made the original ScreenSpot-Pro number read as total failure when it was in
fact 25× chance.

### 3.3 Distance to the box — the continuous companion

Binary in-or-out throws away *how badly* a miss missed. Report distance too:

```
d_box = hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1))
```

— the Euclidean distance from the prediction to the nearest point of the hit box,
in normalized units, and **0 when the click is inside**. This is the natural
companion to §3.2: `point_in_bbox` is exactly `d_box == 0`.

Also compute distance to the box centre, since that is what `blindspot.core.scoring`
records as `center_distance` for the ScreenSpot-Pro arm:

```
d_centre = hypot(x - (x0 + x1) / 2, y - (y0 + y1) / 2)
```

Report the median and the full distribution of both over misses. `d_box` answers
"how far outside the target did it land"; `d_centre` answers "how far from the
thing was it aiming". They diverge most on large targets — precisely where the
hit box is doing work — so report both rather than picking one.

### 3.4 The precision curve — required

Bucket the **same** predictions into coarser cells and report accuracy at each.
This is a post-hoc analysis of predictions: nothing is drawn on any image and no
extra API calls are made.

| granularity | chance |
|---|---|
| 2×2 | 25% |
| 3×3 | 11.11% |
| 4×4 | 6.25% |
| 8×8 | 1.5625% |
| 16×16 | 0.390625% |
| exact hit box | 0.25472% |

A point's cell is `(min(int(x*g), g-1), min(int(y*g), g-1))`; the target's cell
is the one containing its **box centre**. Report ratio-above-chance at every
rung. A smooth decay with a rising ratio is the H1 signature; a cliff is not.

### 3.5 The lenient curve — also required

Alongside the strict curve, report the forgiving definition the study also
published: credit for **any cell the target box touches**, via
`blindspot.reporting.cause_pages.bbox_cells`. The original gave both (4×4: 31.1% strict vs
35.4% lenient). Report both columns; do not quietly pick the friendlier one.

### 3.6 Failure bands

Classify every miss by `d_centre`, in the study's bands: `near_miss` < 10% of the
screen, `moderate_miss` 10–25%, `wrong_region` > 25%. The original split was
23.0% / 39.4% / 37.6%.

Report the same split on `d_box` as well. The two are **not** interchangeable,
and neither is comparable to the original's counts (§1.1) — the band structure is
what transfers, not the numbers.

### 3.7 Text answers (`relation`, `reverse`)

Score **exact match and token-F1, side by side**, via
`blindspot.core.scoring.token_f1`, which returns `(EM, F1)`.

Normalize before comparing: lowercase, strip, collapse internal whitespace. Do
not strip punctuation beyond that, and do not accept substring containment — the
labels are short and substring matching would credit "Close" for "Close Ledger".

> Naming note: the repo's localization benchmark is ScreenSpot-Pro, and §3.2 is
> its click-in-bbox metric. If you meant Google's ScreenQA, that is a
> short-answer UI-QA task whose EM/F1 pair is what this section already uses —
> but its *box* metric is IoU-based BBOX-F1 at threshold 0.1, a different metric
> from §3.2 that must not be mixed with it.

### 3.8 Never do

* **Never average across metrics.** Click-in-bbox, distance and token-F1 are
  different units. Separate columns, as the main study does.
* Never report a mean over resolutions as "the" score — see §4.
* Never score a null or unparseable prediction as wrong. Count it separately and
  say how many. Scoring a dropped API call as a failure is the mistake this whole
  study is about.

### 3.9 Sanity probe — run this FIRST

`scripts/run/coord_probe.py` exists because a near-zero localization score has two
explanations: the model cannot do it, or the harness is broken. On a **dataset
that has never been run against any model**, that ambiguity is sharper than it
was on ScreenSpot-Pro, not weaker.

Before reporting anything, run a **stronger model** (Sonnet) on a sample of
byte-identical inputs.

* If it lands in the boxes, the pipeline is sound and a low Haiku score is a
  capability result.
* If **both** models score near zero, suspect the dataset or the harness first —
  check a handful against `verify/index.html` before writing any conclusion.

Run two conditions, as `coord_probe.py` does: native, and handicapped to Haiku's
~1568px budget. Sonnet's image ceiling is higher (~2576px), so a native win could
be resolution rather than model — which matters more here than there, because
`large` is 3000×1900 and the two models would not receive the same thing.

Report the probe result before the main table.

---

## 4. The multiscale analysis — a primary deliverable

Every scene exists at three rungs:

| variant | on disk | delivered by the API | label font | downscaled |
|---|---|---|---|---|
| `small` | 900×570 (0.51 MP) | 900×570 (0.51 MP) | ≥10px | no |
| `medium` | 1500×950 (1.43 MP) | 1348×853 (1.15 MP) | ≥17px | yes |
| `large` | 3000×1900 (5.70 MP) | 1348×853 (1.15 MP) | ≥33px | yes |

Two comparisons carry the weight, and they mean different things:

**(a) `medium` vs `large` — the null control.** Both deliver at **1348×853**, the
same information budget; they differ only in resampling path. **Expect no
difference.** Whatever gap you measure is this dataset's empirical noise floor,
and it is the yardstick for every other difference you report — the role the
10.1% repeat-disagreement rate played in the main study. Report it first. If
`large` beats `medium` appreciably, something is wrong with your pipeline rather
than interesting about the model.

**(b) `small` vs `medium` — the resolution effect.** The target occupies the
**same fraction** of the image at every rung, so this is not a target-size
effect. It isolates absolute resolution: can the model resolve a 10px label as
well as a 17px one. This is the H2 test.

Report for each rung: click-in-bbox, the precision curve, distance summaries and
token-F1 — then the paired deltas for (a) and (b) with CIs.

### 4.1 Pairing — read this carefully

**Pair on `(graph_id, qtype, target_text, anchor_text)`, never on the uid's
question index.**

The eligible-target pool differs per resolution — a label legible at `medium` may
fall under the 10px floor at `small`, or be occluded at one scale and not another
— so question `:02` is **not** guaranteed to be the same target at every rung.
Measured on this build:

* pairing by uid index: 1,555 complete triples, of which **194 have a different
  target at different resolutions** — silently comparing different questions;
* pairing by `(graph_id, qtype, target_text, anchor_text)`: **1,416 complete
  triples**, all genuinely the same target.

Use the 1,416. Report how many rows you dropped for lack of a complete triple,
and give unpaired totals alongside so the paired subset is visibly not
cherry-picked.

---

## 5. Required breakdowns

Beyond the per-rung table, cut click-in-bbox by:

* **`target_area_frac` quintiles** — the direct H2 gradient, comparable in
  structure to ScreenSpot-Pro's 0.63% → 4.42% finding.
* **`hit_source`** — `shape` (820) vs `padded_text` (1,560). Shape-derived boxes
  are real widgets; padded ones are synthetic. If the two behave differently, the
  padding constant is doing more work than it should, and that is worth knowing.
* **`chart_type`** — 16 types. Call out `dashboard` separately: densest, with
  panels, and the closest analogue to a real UI screenshot.
* **`target_role`** — `label` (694), `point` (238), `cell` (237), `value` (231),
  `category` (136), `tick` (130), `legend` (99), `actor` (90). Whether dense
  table cells behave like isolated node labels is a genuine question.
* **`theme`** (light vs the three dark) and **`font_family`** — these should show
  *nothing*. If they do, you have found a styling sensitivity worth its own
  section.

Use Wilson intervals. State `n` on every cell; suppress cells under n=30 rather
than showing noise.

---

## 6. Things that will bite you

* **Do not pair on the uid index** (§4.1).
* Score `gold_bbox_norm`, not `text_ink_bbox_norm` and not the `_px` fields
  (§3.1).
* `image_px` is the file's size; `effective_px` is what the model resolved. When
  reasoning about resolution, use `effective_px`.
* Targets are already filtered for uniqueness, non-overlap, WCAG AA contrast, a
  10px floor, occlusion, and hit boxes that would swallow a neighbouring label.
  **Do not re-filter.** If a target looks unfair, check `rejected_targets` and
  `readability_report.json` first — the exclusion may already be recorded.
* There is **no ground-truth noise floor**. Gold is measured off the raster. If
  the model disagrees with gold, the model is wrong — unlike the scraped
  benchmarks, where 2.4–5.1% of contested gold had to be subtracted. Do not
  budget for annotation error.
* 25 labels across the 200 scenes fall below AA and are excluded as targets. They
  still appear in the images as distractors, as do the footnotes and badges,
  which are never targets. That is intentional.

---

## 7. What to hand back

0. The §3.9 sanity probe result — is the pipeline sound?
1. A headline table: click-in-bbox per rung, with chance and ratio-above-chance.
2. The `medium` vs `large` null result **first**, as the noise floor.
3. The precision curve per rung, strict **and** lenient columns.
4. Distance summaries (`d_box` and `d_centre`) and the failure bands.
5. `point` vs `relation`, as the perception/emission split.
6. The `target_area_frac` quintile gradient, and the `hit_source` cut.
7. Breakdowns from §5, with `n` and intervals.
8. Every claim tied to a number you can regenerate, and an explicit list of what
   you did **not** test — at minimum: no icon targets, no targets below ~0.039%
   of the image, and no intent-resolution questions, since every target string is
   quoted in its own prompt. See §1.1.

Where a result is null, say so and keep it. Four of the fifteen causes in the
main study are refuted findings, and they are published for a reason.
