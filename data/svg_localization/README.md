# SVG localization probe

A synthetic text-localization set built to isolate what ScreenSpot-Pro conflates.

Regenerate — **into a new directory**, never over this one:

```
python -m blindspot.generate scenes --count 200 --complexity 4 --out /tmp/svgloc_new
python -m blindspot.generate audit  --data /tmp/svgloc_new --open
```

The generator has drifted from the committed set, and `results/*.jsonl` is keyed
by uid, so regenerating in place would bind existing answers to different
questions. See
[../../docs/runme/SYNTHETIC.md §0](../../docs/runme/SYNTHETIC.md#0-do-not-regenerate-in-place).

**Evaluating this set? Read [EVAL.md](EVAL.md) first** — it pins the metrics to
ScreenSpot-Pro's click-in-bbox, makes the resolution ladder a primary result, and
documents a pairing trap that silently compares different questions.

Everything here is generated. Nothing is hand-annotated, so there is no
ground-truth noise floor to subtract — unlike the ~2.4–5.1% contested-gold floor
measured on the scraped benchmarks.

## What it is for

The ScreenSpot-Pro arm of the study could not separate three things:

| confound | how this set separates it |
|---|---|
| target size | `target_area_frac` is recorded per question; targets span 0.0385%–2.75% of the image, median 0.145% |
| image size | the same scene is emitted at three resolutions, two of which deliver *identically* after the API downscale |
| perception vs. coordinate emission | four question types, from "emit x,y" to "name the neighbour" |

## The resolution ladder

| variant | as generated | as the API delivers it | downscaled |
|---|---|---|---|
| `small` | 900×570 (0.51 MP) | 900×570 (0.51 MP) | no |
| `medium` | 1500×950 (1.43 MP) | 1348×853 (1.15 MP) | yes |
| `large` | 3000×1900 (5.70 MP) | 1348×853 (1.15 MP) | yes |

`medium` and `large` are **the same input to the model** despite differing 4× on
disk. Any accuracy difference between them is noise, and any accuracy difference
between `small` and `medium` is a genuine resolution effect rather than a
target-size effect — the target occupies the same *fraction* of the image at
every rung. That pairing is the control the scraped benchmark could not provide.

## Question types

| `qtype` | asks for | `scoring` | isolates |
|---|---|---|---|
| `point` | the label's centre, in a 0-1000 normalized space | `point_in_bbox` | perception **+** coordinate emission |
| `relation` | the label immediately left/right/above/below another | `exact_match` | perception only, no coordinates |
| `reverse` | what text sits at a given pixel | `exact_match` | inverse localization |

There is **no grid arm** — ScreenSpot-Pro has none, and a trial that added one
had its magenta overlay covering the gold text in 31% of those questions.

`point` follows the **ScreenSpot-Pro convention exactly**, so these rows are
scorable by `blindspot.core.score()` unchanged and directly comparable to that
arm:

* `answer_type` is `point`, `gold` is a normalized `[x0,y0,x1,y1]` box;
* `question` is the element description only -- the harness prepends
  `POINT_INSTRUCTION` from `blindspot.core`;
* the model answers `{"x": 0..1000, "y": 0..1000}`, which `parse_response`
  divides by 1000.

Coordinates are normalized rather than pixel-valued for the reason given in
`blindspot/core.py`: the model never sees the native resolution, because the API
downscales first, so asking for pixels would inject a coordinate-space error it
cannot avoid. `gold_bbox_px` is kept alongside for the visual audit.

`relation` is what bounds the coordinate-emission component: it asks about
position while requiring no coordinates anywhere in the question or the answer.

## Ground truth — the hit box is the target

ScreenSpot-Pro's gold is the **widget** box: the region you could click, with its
padding. Scoring against a tight glyph outline would be a harder and different
task, and would leave nothing widget-shaped to measure distance against. So every
label carries a **hit box**, and that is what `gold_bbox_norm` holds:

* the **enclosing shape** where one exists — a flowchart node rectangle, a state
  circle, a legend row (`hit_source: "shape"`);
* otherwise the ink box grown by button-like padding, 0.42×font horizontally and
  0.34×font vertically (`hit_source: "padded_text"`).

A shape only qualifies if it fully contains the label's ink and covers under 6%
of the canvas — that last guard keeps panel frames and plot backgrounds from
being mistaken for buttons.

The glyph outline is still published as `text_ink_bbox_norm` / `text_ink_bbox_px`
for the visual audit, and the ink is guaranteed to sit inside the hit box.

Both boxes are **measured off the rendered raster** with `ImageDraw.textbbox`,
not predicted from font metrics.

The SVG and the PNG are generated from one shared primitive list, so they cannot
drift apart per chart type. Text is pinned in both directions:

* **vertical** — an explicit alphabetic baseline computed from PIL's ink box.
  Relying on SVG `dominant-baseline="central"` instead put the vector source
  2.5px away from the raster, because it centres on font metrics while PIL
  centres on ink.
* **horizontal** — `textLength` + `lengthAdjust="spacingAndGlyphs"`.

Residual SVG↔PNG disagreement is **≤1.0px on a 1500px canvas**, and is
advance-width vs. ink-width side bearings rather than a placement error.

## Validity rules

A label may only become a question target if it passes all of:

| rule | why |
|---|---|
| string unique in the scene | otherwise "where is X" has several right answers |
| ink box overlaps no other label | otherwise the gold box contains someone else's text |
| WCAG AA contrast for its size (4.5:1, or 3:1 at ≥24px) | an unreadable target measures the renderer, not the model |
| ≥10px font at every rung | 7px was squint-legible at best; every label now clears 10px on `small` |
| nothing drawn over the ink | measured against a render with all labels composited last, not inferred from draw order |
| the hit box contains no other label | an enclosing widget must not swallow its neighbour's text |
| fully inside the canvas | no clipped targets |

Contrast is measured against the **actual** background: the scene is rendered a
second time with all text suppressed, and the median colour under each label is
sampled from that pass. A label on a bar, a pie wedge or a table stripe has a
background the theme never names, so assuming the page colour would have been
wrong. This check caught a real bug during development — `line_chart` series
labels on the `high-contrast` theme measured 2.25:1 — which was fixed at the
source rather than filtered away.

Rejection counts are published per row in `rejected_targets`.

## Contents

* `svg/g####.svg` — vector source, one per scene
* `images/g####_{small,medium,large}.png`
* `manifest.jsonl` — one row per (scene, resolution, question)
* `scenes.jsonl` — one row per scene: the semantic content behind the picture
  (bar values, table cells, flowchart edges, sequence messages, …) plus every
  text prim in draw order. Index `i` there is `<text id="ti">` in the SVG and
  `target_idx` in the manifest, so the three files join cleanly.
* `EVAL.md` — instructions for the evaluating agent
* `readability_report.json` — labels below AA and scenes skipped, with reasons
* `verify/index.html` — visual audit; overlays are positioned from the manifest's
  own numbers, so a box that misses its text means the published coordinates are
  wrong

## Diversity

16 chart/diagram builders (`--list-types`): `flowchart`, `bar_chart`,
`line_chart`, `scatter`, `pie_chart`, `table`, `org_chart`, `network`,
`timeline`, `gantt`, `sequence`, `treemap`, `quadrant`, `mindmap`,
`state_machine`, `dashboard`.

`dashboard` composes 2-4 of the others into panels on one canvas. Each panel is
built on a `Scene` sized to the panel, so its builder lays out for the space it
has and its type stays legible; building full-size and shrinking pushed every
label under the legibility floor at `small`. This is the densest type -- 60+
labels, with the smallest targets in the set (from 0.040% of the image, median
0.176%) -- and the closest analogue to a real ScreenSpot-Pro screenshot.

`--complexity 1..5` scales node counts, layer widths, extra skip/back edges,
series and row counts, and adds decoy text (captions, badges, footnotes) that is
never a target. Decoys matter: they put text in the scene that looks like a
target and is not one, which is what stops a localization question being
trivially answerable by "find the only text".

10 themes including three dark and one blueprint, varying background, palette,
stroke weight, corner radius and background pattern; 13 font candidates across
serif, sans, condensed and monospace, of which those present and able to
rasterize at 5-9px are used (8 on this machine). Content is procedural — domain vocabularies
combined under a seed — so scaling to hundreds of scenes does not repeat labels.

Everything is deterministic in `--seed`.

## Key manifest fields

```
uid  graph_id  chart_type  theme  font_family
resolution  scale  image  svg  image_px  effective_px  downscaled_by_api
qtype  question  answer  scoring  answer_type  prompt_style  complexity
gold_bbox_norm  gold_center_norm  gold_bbox_px  gold_center_px
probe_point_px  anchor_text  direction
target_text  target_role  target_area_frac  font_px  target_contrast
n_texts  n_eligible_targets  rejected_targets
```

`answer` is the gold to score against: `{"x": 500, "y": 437}` in 0-1000 space for
`point`, and the label text itself for `relation` and `reverse`.

## Asking non-localization questions later

The SVG carries geometry and strings, but not *relationships* — which value
belongs to which category, which node an edge leaves, what order the messages
went in. None of that is recoverable from the rendered output, so it is captured
at build time in `scenes.jsonl` under `facts`:

| chart type | facts |
|---|---|
| `flowchart`, `network` | `nodes`, `edges`, `layers`, `flow` |
| `bar_chart` | `bars` (category → value), `series_name` |
| `line_chart` | `x_labels`, `series` (name → {x: value}) |
| `table` | `columns`, `row_headers`, `cells` (`"Row\|Col"` → value) |
| `pie_chart` | `wedges` (label, value, pct) |
| `gantt` | `tasks` (task, start_week, duration_weeks) |
| `timeline` | `milestones` (label, date) |
| `treemap` | `blocks` (label, value) |
| `sequence` | `actors`, `messages` (from, to, label, order) |
| `state_machine` | `states`, `transitions` |
| `org_chart` | `levels` |
| `quadrant` | `quadrants`, `items` (label → quadrant) |
| `mindmap` | `root`, `branches` |
| `scatter` | `points`, `x_axis` |
| `dashboard` | `panels[]`, each with its own type, origin, size and facts |

So a later question set can ask read-off ("what is Churn for Pacific?"),
aggregation ("which region has the highest Backlog?"), or topology ("which step
follows Validate Consult?") with exact gold, and still know where the answer
sits on screen.

## Caveats

* **This shares ScreenSpot-Pro's method, not its task.** The metric, prompt
  scaffold and analysis structure are identical and verified.
  But ScreenSpot-Pro asks the model to resolve a functional intent (*"stop the
  bilibili download in android studio"*) onto a referent that is usually an icon;
  here the target string is quoted in the prompt (`the text "Index Intake"`), so
  the model matches a string rather than resolving a reference. That is strictly
  easier and exercises a different ability. **Absolute scores from this set are
  not comparable to the 1.65% baseline** — only within-dataset contrasts and the
  *shape* of the precision curve are. See [EVAL.md §1.1](EVAL.md).
* Targets are rendered text, not UI widgets: 2,882 text targets and **zero
  icons**, against ScreenSpot-Pro's 604 icon / 977 text. The study's icon-vs-text
  finding (1.16% vs 1.94%) is untestable here.
* The hard tail is missing. Targets span 0.0385%–2.75% of the image (median
  0.145%) against ScreenSpot-Pro's 0.0017%–4.73% (median 0.036%) — a floor ~23×
  larger. Deliberate: a ~0.002% target falls below the 7px legibility floor at
  the `small` rung, which would break the resolution ladder for exactly the
  targets that need it most.
* The `relation` questions assume a reading of "immediately left/right/above/
  below" resolved by nearest centre within a band. Ties are possible in dense
  scenes; overlapping and duplicate labels are excluded, but an unusual layout
  could still admit a second defensible answer.
* Nothing here has been run against a model yet. It is an instrument, not a
  result.
* Fonts are probed at 5/7/9px on startup and dropped if they cannot rasterize
  there; macOS `Helvetica.ttc` fails this and is skipped. Generated text is
  ASCII-only for the same reason.
* Some labels still collide in dense scenes -- 17 of 388 across 20 scenes at
  `--complexity 4`. They are excluded from targets by the overlap rule, so gold
  is unaffected, but they do reduce the usable pool.
