# Evaluation instructions — `counting`

Instructions for the agent evaluating `data/svg_localization/counting`. Read this
before running anything.

**714 questions over 200 scenes**, 16 chart types, 19 distinct question forms, at
three resolutions (238 each). Derived from the existing scenes — no image was
re-rendered, so a model can be scored on this, `word_mc`, and the localization set
over **identical pixels**.

---

## 1. What this set is for

One counting question per chart type, about the structure that type is made of:
bars in a bar chart, data rows in a table, slices in a pie, boxes in a flowchart,
messages in a sequence diagram.

The main study found counting **degrades with the count** — InfographicVQA fell
63% → 33% across count bins — and, more usefully, that the *sign* of the error
separates two different failures: undercounting objects (losing track in a crowd)
versus over-counting repeated marks (estimating a pattern instead of enumerating
it). That analysis was limited by scraped ground truth. Here gold is exact, so
the dose-response curve and the signed error can be measured without a noise
floor.

---

## 2. Running it

This maps onto the harness's existing count path:

* `answer_type` is `count`; `blindspot.core.prompts` prepends `COUNT_INSTRUCTION`.
* `blindspot.core.prompts.parse_response` returns `int(obj["answer"])`.
* Score with `blindspot.core.scoring.count_score`, which returns `score`, `abs_error`
  **and `signed_error`**. The signed error is the point — see §3.2.

The assembled prompt looks like:

```
Count carefully and answer with a single whole number.
Count only what is actually drawn in the image.

How many labelled boxes are in this flowchart?
```

Same model settings as the main study (`claude-haiku-4-5-20251001`, thinking
enabled, 2000 tokens).

---

## 3. Metrics

### 3.1 Accuracy

Exact integer match. No tolerance, no off-by-one credit — an off-by-one is a
wrong count, and the interesting question is how the errors are distributed, not
how many can be forgiven.

There is **no meaningful chance baseline** for a free-response integer, so do not
invent one. Report accuracy, and report it against the true count (§3.3).

### 3.2 Signed error — do not skip this

`count_score` returns `signed_error = predicted − true`. Report its **mean per
count bin**, not just the absolute error. This is the whole reason the metric
exists:

* consistently **negative** as the count rises → the model stops early, the
  signature of losing track in a crowded field;
* consistently **positive** → it over-reports, the signature of estimating a
  repeating pattern rather than enumerating it;
* scattered around zero → it is simply noisy.

The main study found *both* patterns in different families on the same
benchmark — objects drifting negative, tick marks drifting positive — and noted
that pooling them into one "counting accuracy" number hides both. Do not pool.

### 3.3 The dose-response curve — the primary result

Bin by the **true** count and report accuracy and mean signed error in each bin.
The distribution here:

| bin | n |
|---|---|
| 1–4 | 117 |
| 5–6 | 183 |
| 7–9 | 177 |
| 10–15 | 210 |
| 16+ | 27 |

Gold ranges 3–27, median 7. Suppress the 16+ bin if it splits too thin after
cutting by anything else — 27 rows is 9 per resolution.

A monotone decline is the finding. A flat curve is also a finding, and should be
reported as one rather than buried.

### 3.4 Never do

* Never report absolute error alone — it destroys the sign, which carries the
  mechanism (§3.2).
* Never pool counting families into one number (§3.2).
* Never score a null or unparseable prediction as wrong. Count separately and say
  how many.

---

## 4. The resolution ladder

**Gold is identical at all three rungs** — verified, 0 questions have a gold that
differs across resolutions. The structure does not change with resolution, only
how many pixels describe it.

| variant | on disk | delivered by the API |
|---|---|---|
| `small` | 900×570 (0.51 MP) | 900×570 (0.51 MP) |
| `medium` | 1500×950 (1.43 MP) | 1348×853 (1.15 MP) |
| `large` | 3000×1900 (5.70 MP) | 1348×853 (1.15 MP) |

That makes this set an unusually clean resolution probe: any accuracy difference
across the three is a resolution effect and nothing else. `medium` vs `large` is
the null control (same delivered size); `small` vs `medium` is the real test.

**Pairing is trivial here and you should exploit it.** All **238 questions appear
at all three rungs**, so every comparison can be fully paired — no dropped rows,
no cherry-picking. Pair on `(graph_id, question)`.

Report the interaction: does the count at which accuracy collapses move with
resolution? That is the sharpest thing this set can show, and neither the
resolution ladder nor the counting curve can show it alone.

---

## 5. Required breakdowns

* **true-count bins** (§3.3) — primary.
* **`chart_type`** (16). Counting bars is not counting table rows; the main study
  found tick-counting behaved differently from object-counting on the same
  benchmark.
* **what is being counted** — 19 distinct question forms. Group them into
  *objects* (bars, boxes, nodes, slices, points), *rows* (table rows, gantt
  tasks), and *connections* (messages, edges). Connections are the ones with no
  enclosing shape to anchor on, and are where undercounting should appear first
  if it appears at all.
* **`theme`** and **`font_family`** — should show nothing.

Wilson intervals for accuracy; state `n`; suppress cells under n=30.

---

## 6. Validity — what has already been guaranteed

Gold comes from the semantic record captured when the scene was built, not from
counting marks in a rendered image. On top of that, **678 of 714 rows are
cross-checked against the labels actually drawn**, and any mismatch is dropped
rather than published:

| chart type | count | cross-checked against |
|---|---|---|
| flowchart / network / state machine / org chart / mind map | nodes, states, branches | one `label` per item |
| table | rows, columns | rows×columns `cell` labels, and `header` labels |
| bar chart | bars | one `category` label per bar |
| pie chart | slices | one `legend` entry per slice |
| sequence | participants, messages | `actor` and `message` labels |
| gantt / timeline / scatter / quadrant / line chart / dashboard | tasks, milestones, points, lines, panels | matching role labels |

`cross_check_failures.json` records any that failed; it is currently empty.

**The 36 unchecked rows are all `treemap`, deliberately.** A block too small for
text is still drawn as a rectangle, so labels and rectangles are genuinely not
1:1 there — the question asks for rectangles, which is visually answerable but
not verifiable against label counts. Treat treemap results with slightly more
caution and say so.

## 7. Limits

* One or two questions per scene, so `chart_type` cells are small — 36–78 rows
  each across all three resolutions, i.e. 12–26 per rung. Several breakdowns will
  not survive being cut twice.
* Counts are small by real-chart standards (median 7, max 27). The main study's
  steepest decline was in the 16+ bin, which has only 27 rows here. **This set
  cannot resolve the high-count tail**, and a flat curve should not be reported
  as "counting does not degrade" — it may only mean the range is too narrow.
* **Run the blind control.** Some counts may be guessable from the question alone
  ("how many columns does this table have" has a narrow plausible range). Ask the
  same questions with the image withheld; whatever survives was never a
  perception task.
