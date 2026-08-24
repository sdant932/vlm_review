# Pipeline — `report_pages causes`, substance review

Agent brief: generate the 15 per-cause pages against the *full* study
(`takehome/results`, `takehome/data`), read six of them in full plus evidence
sections of the rest, and judge whether the claims hold up — not whether the
files exist.

## Verdict

**The pages themselves are unusually rigorous — every verdict badge I checked
matches its own table, controls are run where a control is needed, and thin
evidence is labelled thin. The serious problem is one level up: `causes`
disagrees with the study's shipped report, `blindspots.md`, on three
checkable numbers, and in one case the report states as settled a finding
that `causes`' own page shows is 87% a scoring artifact.**

**The most misleading thing a reader would come away believing**, if they
read `blindspots.md` alone and trusted its Table 2: that "general OCR
reasoning" (read-vs-derive) is corroborated on two independent benchmarks at
comparable strength. `blindspots.md:83` reports CharXiv's 27-point
descriptive-vs-reasoning gap and then adds, uncaveated, "SlideVQA shows the
same gap." The `causes` page for exactly this claim
(`derivation_vs_reading.html`, built by `blindspot/report_pages.py:2049-2111`)
shows the SlideVQA gap is 26.5pp *as scored* but **3.8pp once the metric's
formatting penalty is corrected for** — a 7x shrinkage — and says so in its
own words: *"Anyone reporting the 27-point number as a reasoning deficit is
reporting a scoring artifact"* (`report_pages.py:2090-2092`). The report's
single corroborating benchmark for this cause contributes almost nothing once
measured the way the study's own tooling now measures it, and `blindspots.md`
never mentions the correction.

## What I ran

```
mkdir -p /tmp/causes_read && cd /tmp/causes_read
ln -s /Users/sdoveh/gitrep/takehome/results results
ln -s /Users/sdoveh/gitrep/takehome/data data
PYTHONPATH=.../haiku-perception-blindspots .venv/bin/python -m blindspot.report_pages causes --no-images
```

15 causes + index built cleanly against the full result files (CharXiv 5000,
InfographicVQA 2801, SlideVQA 1003 + 494 all-pages, AI2D 3086, ScreenSpot-Pro
1581 — 13,965 scored rows total). No API calls, no writes outside `/tmp` and
this report.

**Caveat on provenance.** `blindspot/report_pages.py` has an uncommitted
100-insertion/49-deletion diff on disk (`git diff --stat`), made by another
agent mid-session. I read the diff line by line: it is entirely (a) `None`
vs `0.0` formatting hygiene (`pct_or_dash`/`pct_bare`, `report_pages.py:398-421`)
and (b) a new abort guard, `MIN_CAUSE_ROWS = 200` at `report_pages.py:398`
wired into `cmd_causes` (`report_pages.py:3199-3222`), which refuses to build
`causes` at all against a thin `results/` rather than rendering em-dashes that
"still read as evidence." Against the full takehome tree every benchmark is
≥1000 rows, so this guard is a no-op here and none of the arithmetic in the
diff changes a single computed number. I did not modify the file.

## Per-page substance: the six required pages, plus spot checks on the rest

| page | claim | evidence shown | n | verdict vs. table | honesty |
|---|---|---|---|---|---|
| `label_reference_binding` | AI2D label-reference (61%) is the weakest single operation; CharXiv legend-naming (97%) is not | 605 vs 2481 (AI2D split), 154/136 (CharXiv), blind control 22.8% vs 25% chance | good | **MIXED matches**: two benchmarks disagree, page says so and narrows the claim rather than picking a side | Explicit "what would refute this" + blind control that rules out a knowledge-only explanation |
| `language_prior_override` | AI2D score is mostly text prior (blind 62.7% vs 25% chance); CharXiv is mostly vision (blind 26.6%) | paired blind/sighted, n=499/500/500/215 across 4 benchmarks | good | **PROVEN matches** — reproduces the corrected 62.7% AI2D blind figure exactly | States plainly: "A benchmark score is an upper bound on perception, not a measurement of it" |
| `absence_detection` | Model invents an answer 10.6% of the time when the true answer is "Not Applicable"; almost never over-abstains (0.40%) | n=1000 absent items [95% CI 8.8–12.7%], n=4000 over-abstention check, per-template breakdown down to n=27 | adequate, CI shown | **SUPPORTED matches** | Labelled single-benchmark up front: "Without a second source this is a claim about Haiku on scientific charts, not a general property" |
| `counting` | Accuracy falls with true count on 2 of 3 families; the *sign* of the error separates two distinct failure modes | n=314/224/267, binned by count with bins <5 blanked, signed-error table | good, small bins disclosed | **SUPPORTED matches** — explicitly states CharXiv tick-counting is flat and "the claim rests entirely on the signed error" for that family | "the interesting regime (30, 50, 100 objects) is barely sampled... Not tested, and this matters" |
| `effective_resolution` | Accuracy falls with delivered pixels, not with text volume | InfographicVQA 5 quintiles n=560×5, OCR-word-count control (flat, non-monotone), CharXiv panel-count n=1544→152, ScreenSpot-Pro target-size quintiles | good | **PROVEN matches** — control (OCR words) is the right control and it's flat where the pixel bin is monotone | Discloses the untested causal experiment (a resolution ablation) and a live confound ("very large infographics may also be intrinsically harder") |
| `index` | Summary/navigation page | recomputed scoring table, control-arm table, exclusion rules | — | consistent with the 15 detail pages | States the ScreenSpot-Pro 4×4-grid file "was read defensively in case it was still being written" — an odd thing to need to say about a shipped artifact, but honest |

Spot-checked the remaining nine (`answer_expression`, `derivation_vs_reading`,
`cross_page_integration`, `retrieval_search`, `subplot_scope`,
`list_answer_integrity`, `position_bias`, `wrong_element_not_near_miss`,
`ground_truth_noise`) at the evidence-table level. All read as internally
consistent — verdict matches table in every case — and several are notably
self-limiting:

- `subplot_scope` (REFUTED): the decisive test (same figure addressed via two
  different subplots) has **zero eligible items** in the 1,000-figure sample
  and the page says so instead of dropping the test quietly. The axis-confusion
  check runs on **n=3**, and the page flags it as "weak evidence" rather than
  reporting 0/3 as proof of anything.
- `position_bias` (REFUTED): χ²=1.76 against a critical value of 7.81 on
  n=3086, reported with the actual statistic rather than just "no bias found."
- `wrong_element_not_near_miss` (PROVEN): runs a **permutation control** on its
  own best piece of evidence (83% of wrong InfographicVQA numbers appear
  elsewhere in the page's OCR) and reports that the control deflates it — real
  excess is +9.3pp, not the raw 83% — then explicitly says the cause's weight
  rests on the error-magnitude histogram, not the OCR match.
- `ground_truth_noise` (SUPPORTED) catches a real bug in the underlying
  SlideVQA manifest: qa_id 1620/1621, annotated expression `220-50`,
  annotated *answer* `17` (should evaluate to 170) — and discloses that this
  same item is used elsewhere on the site as a model failure (it is: the
  model's own answer, 450, is wrong regardless of which gold is right, so the
  mislabeled gold doesn't corrupt that page's classification, but it is worth
  knowing the annotation itself is broken).

I found **no case** of a page's own printed verdict contradicting its own
table — the single most serious failure mode this review was looking for —
across all 15 pages.

## Cross-check against `takehome/outputs/report/blindspots.md`

| # | topic | `blindspots.md` | `causes` page (this run) | agree? |
|---|---|---|---|---|
| 1 | ScreenSpot-Pro click accuracy | **1.8%** (`blindspots.md:38`, Table 1; repeated `:81` Table 2) | **1.6%** exact / 1.64% precisely (`resolution_precision.html`; index.html row `screenspot_pro ... 1.6% 1.6`) | **no.** This matches the known correction (1.83%→1.6445%, old loader pooled three protocols) that the causes pages already reflect and the shipped report does not |
| 2 | AI2D label-reference split | 58.7% (n=499) vs 86.0% (n=2,587) (`blindspots.md:82`) | 60.8% (n=605) vs 86.7% (n=2,481) (`label_reference_binding.html`) | **no — different classification, undisclosed.** Both sum to 3,086, so this is a reclassification of the same question pool, not new data. `blindspot/core.py:706-710`'s own comment says the split moved from "keyword guessing" to "option shape" (`core.py:720,731`) — i.e., the classifier was rewritten after the report was written, and the change is documented in code comments but never reconciled in either `blindspots.md` or the causes page |
| 3 | Derive-vs-read, SlideVQA corroboration | "SlideVQA shows the same gap" (`blindspots.md:83`), no number given | 26.5pp as scored → **3.8pp format-corrected** (`derivation_vs_reading.html`; `report_pages.py:2073-2077,2090-2092`) | **directionally yes, magnitude no** — see lead finding above |
| 4 | CharXiv aggregate score | 84.7% (`blindspots.md:34`, Table 1) | 85.3% accuracy@0.5 / 84.1% mean metric (`index.html`) | **neither matches exactly** — minor, likely a different metric blend, but no page states which |
| 5 | "none [land] at all when the resize leaves the target under 12px" | asserted (`blindspots.md:81`) | **not tested anywhere in `causes`** — no page bins by absolute pixel size of the *rendered* target post-downscale, only by target size as a fraction of screen | claim in the report with no supporting page |
| 6 | Resolution bias (InfographicVQA 74.3%→59.4%, CharXiv 9.5pp 1-panel→13+) | `blindspots.md:80` | `effective_resolution.html` | **exact match** |
| 7 | Hallucination (10.6% invention, 45.7% worst template) | `blindspots.md:84` | `absence_detection.html` (46% rounded, 45.7% in detail table) | **match** |
| 8 | Counting (63%→33% InfographicVQA bins, 78.1% vs 93.3% CharXiv) | `blindspots.md:85` | `counting.html` | **exact match** |
| 9 | 13-benchmark → 5/6 kept, item counts | `blindspots.md:19-27` | index totals differ (13,965 vs 13,471) | **explained, not a bug** — the 494-item difference is exactly the SlideVQA-all-20-pages control arm, which the report's Table 1 doesn't include in scope |

Items 6–9 reproduce cleanly; items 1–5 do not, and none of the five
discrepancies is flagged anywhere in either document.

## Honesty assessment

**Good, unprompted disclosure — representative quotes:**

- `effective_resolution.html`: *"A second confound remains: very large
  infographics may also be intrinsically harder (longer, more multi-part
  questions), and the OCR-word control only partly rules that out."*
- `resolution_precision.html`: *"Note one confound: the grid screenshots were
  pre-downscaled to 1568px before the grid was drawn, while the main run sent
  native resolution... which makes its advantage a lower bound."*
- `subplot_scope.html`: *"The decisive test could not be run on this
  sample... This is recorded as untestable rather than quietly dropped."*
- `answer_expression.html` exists purely to say two other causes on the same
  site look worse than they are: *"it contaminates two other causes...
  and because it is the single largest correction available to any number in
  this study."*
- Index page (`report_pages.py` → `index.html`): negative results are kept
  and counted ("4 of these hypotheses are refuted, and that is reported
  rather than dropped"), and the exclusion rule for the example galleries is
  stated plainly enough to audit: items the blind arm solved are dropped
  from every gallery except the one whose subject is exactly that.

**Weaker spot:** the causes pages are honest *within themselves* but do not
reconcile with the shipped report at all — there is no note anywhere in
`causes` saying "this number supersedes blindspots.md's Table X," even though
at least one of the differences (ScreenSpot-Pro) is a known, deliberate
correction. A reader who has both documents open gets two different headline
numbers for the model's worst benchmark with no pointer explaining why.

## Bottom line

Trust the `causes` pages over `blindspots.md` where they conflict — the
`causes` pipeline recomputes everything from the result files with disclosed
controls, and the report predates at least one known fix. But do not read
`blindspots.md` Section 3/Table 2 as still-accurate cross-benchmark
corroboration for the read-vs-derive claim: SlideVQA's contribution to that
claim is a metric artifact, and the study's own more careful page already
says so.
