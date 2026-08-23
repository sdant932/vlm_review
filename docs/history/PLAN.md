> **Historical.** Preserved verbatim from the working tree for provenance.
> It describes an earlier state of the project and is **not** current guidance.
> For how things actually work now, read the top-level `README.md` and
> `docs/PIPELINE.md`.

---

# Coordinate-Diagnostics Explainer (HTML + SVG + annotated PNGs)

## Context

We validated the ScreenSpot / ScreenSpot-Pro grounding pilot and found the low scores
(21% / 3%) are **real**, not a harness bug:

- bbox convention proven `[x0,y0,x1,y1]` (xywh would push 642/1272 and 551/1581 boxes
  past their own image edge)
- 1581/1581 Pro images match their recorded `img_size`; 1 known clamped edge case
- re-interpreting model output in any other coordinate space makes accuracy *worse*
  (÷1000 21.0% > src-pixels 16.5% > ÷100 0.0%)

The remaining finding is a **systematic coordinate compression**, explained in prose
that didn't land. Regressing predicted centre on gold centre:

| dataset | slope x | slope y | intercept x | intercept y |
|---|---|---|---|---|
| ScreenSpot-v2 (~960×540) | 1.023 | 0.873 | 0.008 | 0.064 |
| ScreenSpot-Pro (4K) | 0.685 | 0.620 | 0.227 | 0.107 |

Slope ≈1 = tracks the target. Slope 0.62–0.69 with a positive intercept = predictions
collapsing toward screen centre — weak positional signal falling back on a prior. This
is the resolution hypothesis as a continuous quantity, and it escapes the 3% floor.

**Goal:** make this visible rather than argued — a self-contained HTML page plus
annotated PNGs, built from real pilot data, with a stronger model run on the same
inputs as the control.

---

## Part A — Sonnet control run (~$0.25)

Purpose is a **harness sanity check**, not a model comparison: if a stronger model lands
inside the gold boxes on identical inputs, the harness is sound and Haiku's score is a
capability result.

**Sample:** 10 examples, stratified — 5 ScreenSpot-v2 + 5 ScreenSpot-Pro, mixing cases
Haiku hit and missed, and small vs large targets. Fixed uids, recorded in the script so
the set is reproducible.

**Two conditions, because Sonnet 5's image ceiling (~2576px) is higher than Haiku's
(~1568px):**
- `native` — Sonnet's real-world behaviour
- `--max-edge 1568` — Sonnet handicapped to Haiku's effective pixel budget

Sonnet winning at native but *not* at 1568 means the gap is resolution, not model.
Winning at both means it's model capability. Either result is informative; conflating
them would not be.

**Runner change required:** `runner.py` hardcodes `MODEL` and uses
`thinking={"type":"enabled","budget_tokens":N}`. That parameter is **removed on Sonnet 5
and returns a 400** — Sonnet 5 needs `thinking={"type":"adaptive"}`. Add `--model` and
branch the thinking config by model family. Keep structured outputs (supported on both).

## Part B — Annotated PNGs

`outputs/probe/` — one PNG per probe example, plus a contact sheet.

Each image shows, over the real screenshot:
- **gold bbox** — green rectangle (`STATUS["good"]`)
- **Haiku prediction** — red crosshair (`STATUS["critical"]`)
- **Sonnet prediction** — blue crosshair (`SERIES` slot 1)
- caption: instruction, target size in native px and after downscale, hit/miss per model

Two crops per example: full screen for context, and a zoom around the gold target —
on a 4K screenshot a 47px target is invisible at full-page scale. `report.py:render_grounding`
already implements exactly this framing (gold box + predicted point + padded zoom);
extend it to take multiple predictions rather than writing a second renderer.

## Part C — HTML explainer

`outputs/coord_diagnostics.html` — one file, no external assets.

1. **Bbox anatomy** — screen rectangle with `[x0,y0,x1,y1]` labelled on real ScreenSpot
   row 0 (`"close"`, `[0.948,0.144,0.994,0.207]`): top-left origin, y increasing
   downward, and why the same numbers read as `[x,y,w,h]` land off-screen.
2. **Units differ per dataset** — v2 normalized 0–1; Pro absolute pixels + `img_size`.
   Same geometry, two encodings, one conversion in `adapters.py`.
3. **The compression finding** — two scatter plots (v2, Pro): gold-centre vs
   predicted-centre, ideal `y=x` dashed, fitted line solid, slope/intercept annotated.
   v2 hugs the diagonal; Pro flattens toward the mean.
4. **What that looks like on screen** — screen outline, real gold targets, arrows to
   where Haiku actually clicked. The abstract slope becomes "arrows all point inward."
5. **Probe gallery** — the Part B PNGs inlined, Haiku vs Sonnet side by side.

Plus a plain-language reading of each panel and the falsifiable prediction: slope should
degrade monotonically with `--max-edge`; if flat, the resolution hypothesis is wrong.

---

## Implementation

- Read the `dataviz` skill before writing any chart code.
- `scripts/coord_probe.py` — selects the fixed 10 uids, runs both conditions, writes
  `results/probe_*.jsonl`.
- `scripts/coord_diagnostics.py` — renders PNGs and builds the HTML.
- Reuse rather than reimplement: `blindspot.adapters.load`, `blindspot.report.load_results`,
  `render_grounding`, the `SERIES`/`STATUS` palette, and `esc()`. Emit SVG as strings the
  way `report.py:pipeline_diagram` already does.
- Real data only — every number traceable to `results/*.jsonl`.

## Verification

```bash
conda activate takehome && cd ~/gitrep/takehome
python scripts/coord_probe.py --max-spend 1        # Sonnet, both conditions
python scripts/coord_diagnostics.py                # PNGs + HTML
open outputs/coord_diagnostics.html
```

- [ ] Sonnet runs without a 400 (adaptive thinking branch works)
- [ ] Both conditions recorded; per-condition hit rate reported separately
- [ ] 10 PNGs written, each showing gold box + both model predictions + zoom crop
- [ ] Page opens standalone, no network requests
- [ ] Fitted slopes on the page match the table above
- [ ] Panel 1 coordinates match `data/screenspot/manifest.jsonl` row 0 exactly
- [ ] Readable light and dark; every charted value also present as text

## Out of scope

Full-split scale-out, resolution ablation across `--max-edge` settings, repeat-variance
runs, and the remaining datasets (FlowLearn, SlideVQA, DocVQA, LiveXiv). This deliverable
is the diagnostic that decides whether those are worth running.
