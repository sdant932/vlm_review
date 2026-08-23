# Evaluation instructions — `word_mc`

Instructions for the agent evaluating `data/svg_localization/word_mc`. Read this
before running anything.

**1,104 questions over 191 scenes**, 16 chart types, at three resolutions
(small 366 / medium 368 / large 370). Derived from the existing scenes — no image
was re-rendered, so a model can be scored on this, `counting`, and the
localization set over **identical pixels**.

---

## 1. What this set is for

One question: *which of these four words appears in the figure?* One option is
present; the other three appear nowhere in it.

This isolates **reading** from **localization**. The localization set asks where
a named string is; this asks only whether a string is there at all. Comparing the
two on the same scenes separates "cannot find it" from "cannot read it" — if a
model reliably knows a word is present but cannot point at it, the deficit is
spatial, not textual.

It is also the cheapest probe of the resolution ladder in the whole collection:
the answer does not depend on where anything is, only on whether glyphs resolved.

---

## 2. Running it

This maps onto the harness's existing multiple-choice path:

* `answer_type` is `choice`; put `options` into `Example.meta["options"]`.
* `blindspot.core.prompts` prepends `CHOICE_INSTRUCTION` **and appends the lettered
  option block** — the manifest stores `question` and `options` separately, and
  the `A.`/`B.`/`C.`/`D.` lines are assembled at prompt time, not stored. Do not
  hand-roll this; use the existing builder or you will send a different prompt.
* The model replies with a single letter, constrained by the response schema's
  `enum: ["A","B","C","D"]`.

The assembled prompt looks like:

```
Answer with the single letter of the correct option.
Base your answer only on what is shown in the diagram.

Which of the following words appears in the figure?

A. Aggregate
B. Analyst
C. Discharge
D. Dispute
```

Use the same model settings as the main study (`claude-haiku-4-5-20251001`,
thinking enabled, 2000 tokens).

---

## 3. Metric

**Exact letter match**, via `blindspot.core.scoring` (`multiple_choice`). Nothing else
— no credit for naming the word instead of the letter, no partial credit.

**Chance is 25%.** Report accuracy against it. An accuracy near 25% means the
model is guessing, not reading, and every downstream cut is then meaningless.

### 3.1 Position bias — check it before anything else

The answer key is close to uniform by construction, which is what makes a slot
preference detectable:

| option | n | share |
|---|---|---|
| A | 293 | 26.5% |
| B | 278 | 25.2% |
| C | 261 | 23.6% |
| D | 272 | 24.6% |

Run the same χ² test the main study ran on AI2D: compare the model's pick
distribution against this key, **and** the distribution of its picks among wrong
answers, which is the sharper test because a guessing model has nothing else to
go on. The AI2D result was a clean null (max deviation 0.8pp, χ²=1.76 against a
critical value of 7.81). If this set shows a bias, every accuracy number below is
contaminated and you must say so before reporting them.

### 3.2 What a wrong answer means here

A wrong answer is one of two things, and they are worth separating:

* the model picked a distractor that is genuinely absent — a **hallucinated
  reading**, and directly comparable in spirit to the absence-detection finding
  in the main study (10.6% invention on CharXiv "Not Applicable" items);
* the model failed to spot the word that is present.

Both look identical in the accuracy number. Report the distribution of *which*
distractor was chosen: if wrong picks cluster on particular vocabulary, that is a
prior about what charts contain overriding what this chart contains.

---

## 4. The resolution ladder

Report accuracy per rung:

| variant | on disk | delivered by the API | label font |
|---|---|---|---|
| `small` | 900×570 (0.51 MP) | 900×570 (0.51 MP) | ≥10px |
| `medium` | 1500×950 (1.43 MP) | 1348×853 (1.15 MP) | ≥17px |
| `large` | 3000×1900 (5.70 MP) | 1348×853 (1.15 MP) | ≥33px |

`medium` and `large` deliver at the **same** size, so they are the null control —
any gap between them is noise, and it is the yardstick for the `small` vs
`medium` gap, which is the real resolution test.

This set is the cleanest resolution probe available, because the answer has no
spatial component at all. If accuracy falls from `medium` to `small` here, that is
glyph legibility and nothing else.

**Pairing.** The vetted word pool differs per rung, so pair on
`(graph_id, answer_text)`, never on the uid index. **148 words appear at all
three resolutions**; use those for paired tests and report the unpaired totals
alongside.

---

## 5. Required breakdowns

* **`chart_type`** (16) — is a word in a dense table harder to spot than one in a
  flowchart node?
* **`theme`** (10, three dark) and **`font_family`** (9) — these should show
  nothing. If they do, it is a styling sensitivity and deserves its own section.
* **word length** of the correct answer — the shortest words are the hardest to
  resolve at `small`, and this is the most likely place for a real effect.

Wilson intervals; state `n`; suppress cells under n=30.

---

## 6. Validity — what has already been guaranteed

Do not re-filter. Every row satisfies, by construction and verified:

| guarantee | how |
|---|---|
| the correct word is genuinely present | substring check across all scene text — 0 failures over 1,104 rows |
| every distractor is genuinely absent | substring check across **all** text including titles, footnotes and badges — 0 failures over 3,312 distractors |
| the correct word is readable | drawn only from labels that passed the localization filters: ≥10px at this rung, WCAG AA contrast, not occluded, unique in the scene |
| options are well formed | 4 distinct options, letter matches the correct option — 0 failures |

The last one matters most: a word the model physically cannot read would make the
item unanswerable rather than hard, so correct answers are restricted to labels
already proven legible at that specific resolution.

## 7. Limits

* Distractors come from the same vocabulary families the scenes are built from,
  so they are plausible — but they are single words, and a model with a strong
  prior over this vocabulary could exploit it. **Run the blind control**: ask the
  same questions with the image withheld. Whatever survives was never a
  perception task. The main study found AI2D was 63% answerable blind against 25%
  chance; do not assume this set is immune.
* Presence is not localization. A high score here says nothing about whether the
  model can point at the word — that is what the localization set measures, and
  the interesting result is the *gap* between them.
* One word per question. This does not test reading a phrase, ordering, or any
  relationship between labels.
