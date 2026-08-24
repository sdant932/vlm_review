# Pipeline 3 — finetune_data (re-run after the fixes)

Agent brief: run it end to end from `docs/ARTIFACTS.md`; verify three claimed
fixes; check the artifacts against `outputs/part3/part3.md`.

## Verdict

**The pipeline runs clean end to end — 8/8 steps, every artifact produced, for
$0.0789 and 31 API calls.** Two of the three fixes hold under attack. The third
does not.

**`--out` refusal is bypassable in one keystroke on this machine.**
`--out outputs/PART3` is accepted and schedules writes into `outputs/PART3/assets`,
which on APFS *is* `outputs/part3/assets`. The guard compares `Path.resolve()`
strings; `resolve()` does not case-fold and macOS is case-insensitive by default.
Same hole via `--out DATA` (→ `data/svgloc_mr`) and `--out outputs/Finetune`
(→ `gallery.html`, `worked_examples.json`). Four of the six protected artifacts
are reachable. I stopped at `--list` and did not execute.

Against `part3.md`, the previous review's verdict stands unchanged: **the ladder
claims verify exactly, and the training-plan claims do not.** Nothing in §3's
balancing, source mix or SFT/RL split has been implemented since.

---

## Gate

| gate | result |
|---|---|
| `pytest` | **583 passed, 4 skipped, 0 failed**, exit 0 ✓ |
| `tools verify-install` | ok — imports, 15 CLIs, 2 libraries skipped, 4723 questions ✓ |
| `pipelines finetune_data --list` | 1 step, `[verify] multires-audit` ✓ |

Note: `addopts = "-q"` (`pyproject.toml`) plus a pipe swallows the summary line —
`pytest -q | tail` shows dots and no count. Redirect to a file to see it.

---

## Commands run, in order

All writes routed to `/tmp/ft` and `/tmp/ftguard`. **405 reference files
checksummed before and after: byte-identical.**

| # | command | ARTIFACTS.md | result |
|---|---|---|---|
| 1 | `pipelines finetune_data --list` | :186 | 1 step ✓ |
| 2 | `--out outputs/part3 \| data \| outputs/finetune --list` ×7 forms | :142 | refused ✓ |
| 3 | `--out outputs/PART3 \| DATA \| outputs/Finetune --list` | :142 | **ACCEPTED — bypass** ✗ |
| 4 | `pipelines finetune_data --all` (bare) | :186 | 1 step, 2s, exit 0 ✓ |
| 5 | `pipelines finetune_data --all --out /tmp/ft --offline` | :143 | 7 steps, 23s, exit 0 ✓ |
| 6 | `report_worked --max-spend 0` | :153 | 0 calls, no file, exit 2 ✓ |
| 7 | `report_worked --max-spend 0.02` | :153 | 7 calls, $0.0203, TRUNCATED ✓ |
| 8 | `pipelines finetune_data --from report --out /tmp/ft` (no cap) | — | refused, "1 step(s) call the API" ✓ |
| 9 | `pipelines finetune_data --from report --out /tmp/ft --max-spend 0.50` | :143 | 24 calls, $0.0586, 1162s, exit 0 ✓ |
| 10 | `report_finetune examples --dataset data/svgloc_mr --out-dir /tmp/…` | :152 | ✓ (caption audit) |
| 11 | `report_finetune gallery --records data/sft_bbox/… --out /tmp/…` | :149 | byte-identical ✓ |
| 12 | `generate_finetune ladder --scenes-per-aspect 4 --out /tmp/ftdev/…` | :147 | ✓ |

**Spend: $0.0789 total, 31 model-scored questions** of the ~100 budgeted.
Skipped: nothing. `report_worked` is the only paid step in the pipeline
(`pipelines.py:299-303`, `needs_api=True, spend=0.5`); everything else is offline.
I did not re-score any benchmark — none of this pipeline does.

---

## The three fixes

### 1. `--all` no longer overwrites six artifacts — **PARTIAL**

| sub-claim | result |
|---|---|
| Without `--out`, no reference artifact is scheduled | ✓ plan is 1 step |
| Bare `--all` leaves all 405 reference files untouched | ✓ verified by md5 |
| "schedules **no writing steps at all**" (`FINETUNE.md:23`, `pipelines.py:310`) | ✗ **false** |
| `--out` onto a reference is refused by name | ✓ for exact-case paths |
| …resistant to symlink / `..` / trailing slash / absolute / `./` | ✓ all 7 forms refused |
| …resistant to **case variants** | ✗ **bypassed** |
| …covers non-listed artifacts | ✗ `--out outputs` accepted |

**The bypass.** `pipelines.py:330-339` (`_resolutions`) compares resolved *strings*
against `FINETUNE_REFERENCE` (`pipelines.py:237-247`). `Path.resolve()` preserves
case; APFS does not distinguish it. Confirmed live — `head outputs/PART3/part3.md`
prints the shipped document.

```
$ python -m blindspot.pipelines finetune_data --out outputs/PART3 --list
  gen-multires   ... ladder --out outputs/PART3/svgloc_mr
  figures        ... figures --out-dir outputs/PART3/assets      <-- reference artifact
  examples       ... examples --out-dir outputs/PART3/assets
  render         ... render_markdown --out outputs/PART3/part3.html
```

`--out DATA` reaches `data/svgloc_mr`; `--out outputs/Finetune` reaches
`gallery.html` and `worked_examples.json`. Fix: compare `os.path.realpath` +
`os.path.normcase`, or `os.path.samefile` when the target exists.

**Secondary gap.** `--out outputs` is accepted and points `figures`/`examples` at
`outputs/assets` — the 28MB `eval annotate` asset tree (`ARTIFACTS.md:97`). Not on
the six-item list, so not refused. The guard is an allowlist of six names, not a
rule about the repo.

**"No writing steps at all" is not true.** The bare plan's one step,
`generate_finetune audit`, defaults `--out` to
`outputs/finetune/gallery_<dataset>.html` (`generate_finetune.py:722`) with
`--gallery-n 24` (`:751`). Bare `--all` wrote a 973 KB
`outputs/finetune/gallery_svgloc_mr.html`. It is deterministic — the rewrite was
byte-identical — and it is not one of the six, so no harm done. But
`FINETUNE.md:222` labels that exact command `# read-only` and `pipelines.py:310`
says "Read-only against the ladder". Both are wrong.

### 2. `report_finetune figures` label agreement — **FIXED, verified**

| check | before (prev. review) | now |
|---|---|---|
| shipped ladder, 1,513 rows | 268 out of range, 159 wrong string, 1,086 ok | **1,513 / 1,513 agree, 0 / 0** ✓ |
| regenerated ladder, 1,513 rows | — | **1,513 / 1,513 agree** ✓ |
| poisoned `target_text` | drew silently | `SystemExit`, **no file written** ✓ |
| poisoned `target_idx` (out of range) | drew silently | `SystemExit`, **no file written** ✓ |
| `--scenes-per-aspect 4` (24 scenes, no graph 89) | crashed on `FIG_UID` | both figures produced ✓ |

`report_finetune.py:355-357` now calls `GF.build_scene(...)` with the row's own
`canvas_px` and `GF.COMPLEXITY` (=4). The assertion at `:425-429` raises:

```
mr:0089:portrait:r70:03: rebuilt scene does not match the manifest --
target 15 of 32 is 'Score Receipt', manifest says 'XXX-POISONED'
```

### 3. `report_worked` spend ceiling — **FIXED, verified**

`report_worked.py:99` adds `--max-spend` (default 0.50); `:132` checks
`budget.exhausted()` before each call; `:166-171` refuses to write on an empty run.

| cap | calls | spent | file | overshoot |
|---|---|---|---|---|
| `$0` | 0 | $0.0000 | **not written**, exit 2 | — |
| `$0.02` | 7 | $0.0203 | written, `truncated: true`, run marked PARTIAL | **$0.0003 = 1 call** ✓ |
| `$0.50` | 24 | $0.0586 | written, 3 complete groups | — |

Serial-by-construction bound holds exactly as documented.

---

## Artifacts

Every artifact `ARTIFACTS.md` attributes to this pipeline. **Embedded-image counts
checked: no gallery lost its images.**

| artifact | ARTIFACTS.md | status | `<img>` | base64 |
|---|---|---|---|---|
| `svgloc_mr/` (384 PNG + manifest) | :147 | produced ✓ | — | — |
| `sft_bbox/sft_bbox_20.jsonl` (20 rec, 28 KB) | :148 | produced ✓ | — | — |
| `gallery.html` (1.1 MB) | :149 | produced ✓ | 40 | **40/40** |
| `gallery_svgloc_mr.html` (980 KB) | :150 | produced ✓ | 48 | **48/48** |
| `assets/fig_box_extraction.png`, `fig_frames.png` | :151 | produced ✓ | — | — |
| `assets/ex01–ex09.png` | :152 | produced ✓ | — | — |
| `worked_examples.json` (3 groups × 8) | :153 | produced ✓ | — | — |
| `part3.html` (380 KB) | :154 | produced ✓ | 9 | **9/9** |
| `part3_paste.html` (404 KB) | :154 | produced ✓ | 9 | **9/9** |

Shipped copies checked too: `outputs/finetune/gallery{,_svgloc_mr,_multires}.html`
= 40/40, 48/48, 48/48 base64; `outputs/part3/part3{,_paste}.html` = 9/9. No losses
anywhere. Nothing failed.

### But `part3.html` does not use the figures the pipeline just built

`/tmp/ft/part3.html` is **byte-identical** (`bca2a6f4…`) to
`outputs/part3/part3.html`, while `/tmp/ft/assets/ex01.png` ≠
`outputs/part3/assets/ex01.png`. Cause: the pipeline hardcodes
`--src outputs/part3/part3.md` (`pipelines.py:227, 304`) and `render_markdown.py:360`
inlines images with `base=src.parent`. So `render` always embeds the **shipped**
assets and ignores `--out-dir`. In a scratch build the `figures` and `examples`
steps are orphaned — nothing downstream consumes them. `ARTIFACTS.md:143` sells
`--out DIR` as making the whole thing self-contained; two of its eight steps are not.

---

## Reproducibility (`ARTIFACTS.md:168-187`)

| claim | line | measured |
|---|---|---|
| `gallery.html` byte-identical | :173 | ✓ md5 `62aa0e26…` both |
| ladder: same 1,513 uids | :182 | ✓ 1,513 shared |
| ladder: **91%** different `target_text`/`box_px` | :130 (FINETUNE) | **90.9% (1,376/1,513)** — both fields, same rows ✓ |
| ladder images | — | **384/384 byte-identical** ✓ |
| `samples --seed 0` reproduces **5** of 20 | :131 (FINETUNE) | **5 of 20** ✓ |
| `worked_examples.json` not reproducible | :177 | ✓ — my 3 uids ≠ the shipped 3 |
| `--scenes-per-aspect 4` ladder | :192 (FINETUNE) | 385 files, 12 MB, manifest 1.1 MB ✓ |

---

## part3.md vs the data on disk

### Verified true

| claim (`part3.md`) | on disk |
|---|---|
| "1,513 targets over 384 images" | 1,513 rows / 384 unique images ✓ |
| "16 chart and diagram types, 10 colour themes, 9 font families" | 16 / 10 / 9 ✓ |
| "Six aspect ratios at four sizes each" | 6 aspects × 4 rungs (r55/r70/r85/r100) ✓ |
| "from 510×681 up to … 1568×671 at 21:9, 1072×1072 square, 928×1238 portrait" | all four exact ✓ |
| "Every image delivered exactly as rendered" | `downscaled_by_api` False on 1,513/1,513 ✓ |
| "that shape is recorded separately and flagged" | `hit_source` 505 shape / 1008 padded_text; `hit_box_answers_the_question` False on exactly those 505 ✓ |
| box is the supervision target, hit box flagged | `target.box_norm` vs `accept_region` with `source: "gold_bbox -- Part 2 click-in-bbox region"` ✓ |
| GRPO advantage `A_i = (r_i − mean r)/std r` | `report_worked.py:147` implements it exactly ✓ |
| size-bin endpoints 0.039% – 2.751% | band labels span `0.039-0.094%` … `0.432-2.751%` ✓ |

### Not supported by the data

| # | `part3.md` says | disk says |
|---|---|---|
| 1 | size balancing: **3 bins @ 50/30/20** | **5 bands @ 30/25/20/15/10** — `WEIGHTS = [6,5,4,3,2]`, `generate_finetune.py:423`; shipped record counts 6/5/4/3/2 |
| 2 | background: natural **74/26**, use **80/20** | shipped records are **14 light / 6 dark = 70/30**, and `pick_spread` (`generate_finetune.py:483`) weights only `chart_type + theme + target_role`. **No polarity term anywhere.** The 70/30 is a side effect. Ladder polarity is 70.4/29.6, also not 74/26 |
| 3 | **three** sources @ 50/30/20 (synthetic / web / Opus-labelled) | **one.** All 20 records come from `data/svg_localization`, all 900×570. No web scraper, no Opus step, no `source` field on any record. 100% synthetic |
| 4 | supervised/RL split "measured, not chosen" | **does not exist.** `grep -rin grpo blindspot/*.py` → 3 hits, all comments (`pipelines.py:224,369`, `report_worked.py:6`). No split field, no partition step, no IoU scoring of the pool. `def iou` exists once, in `report_worked.py:50`, a reporting tool |
| 5 | GRPO eligibility: samples must "both disagree **and clear a minimum score**" | `report_worked.py:152` is `usable_group: sd > 1e-9`. Spread only. My own run reproduced the failure: `mr:0055` scored **mean IoU 0.0044**, flagged `usable_group: true` — precisely the "everything near zero" case the document excludes |
| 6 | §3 headline "99.9% … 95.6% … 5.3%" | word_mc **100.00%** (n=85), counting **98.25%** (n=57), localization **4.35%** (n=69, k=3). Per-rung localization 5.56 / 2.94 / 5.88. **5.3% matches nothing** |
| 7 | §2 "74.3 to 59.4 across megapixel quintiles", "6.7 point spread" | `outputs/report/figures.json` has `resolution_gradient.image_size = null` and `text_volume = null`. Neither figure appears in `figures.json` or `outputs/summary.json`. By `ARTIFACTS.md:49-51`'s own rule — "if a number appears in a page but not here, it was typed by hand" — these were typed by hand |
| 8 | accuracy by size 1.3 / 3.8 / 12.3% | `outputs/svgloc/summary.json` area quintiles are n=3–4 per cell, all `suppressed: true`, all acc 0.0. Not derivable |
| 9 | background accuracy 2.6% light / 13.8% dark | disk: **1.82%** light (n=55) / **14.29%** dark (n=14) |

Caveat on 6–9: this tree's `results/` is a reduced subset (localization n=69 vs the
study's full run), so these are "not reproducible from what is on disk here",
not proof of fabrication — except **#7**, which is null in the auditable bundle
`figures.json` is defined to be.

### The chained-ladder defect is unchanged

`FINETUNE.md:320` flags it and it is still true: `samples` reads
`data/svg_localization/manifest.jsonl`. All 20 SFT records are 900×570.
**The ladder feeds nothing that becomes training data.** `part3.md`'s
"Resolution and aspect ratio" section describes the ladder as if it were the
training set; it is not.

---

## ARTIFACTS.md inaccuracies

| line | says | actual |
|---|---|---|
| `ARTIFACTS.md:162` | "**8 of 9** captions name a different chart, size and target than the image they label" | **9 of 9.** Regenerating `examples` from the shipped ladder reproduces `ex01–ex09.png` **byte-identically** (9/9 md5 match) with captions that differ from `part3.md` on every line. `ex06` is captioned `"Yield" — ultrawide 1098×470, line chart` and is actually `"W4" — square 750×750, gantt`. The one string that appears in both is at a different index |
| `ARTIFACTS.md:186` + `pipelines.py:310` + `FINETUNE.md:222` | the no-`--out` step is read-only / "schedules nothing that writes" | it writes `outputs/finetune/gallery_svgloc_mr.html` (973 KB). Harmless — deterministic, not a reference artifact — but the word is wrong in three places |
| `ARTIFACTS.md:142-143` | "`finetune_data --out DIR` … refuses to target any of these by name. The pipeline is protected" | protected against exact-case paths only. See the bypass above |
| `ARTIFACTS.md:123` | "Every command in **this table** … writes to a reference artifact by default" | the table (`:132-139`) omits `generate_finetune audit`, which also defaults its `--out`. Reader following the table will not know |
| `ARTIFACTS.md:153` | gives the artifact's command as `python -m blindspot.report_worked --max-spend 0.50` — **no `--out`** | that bare form defaults straight onto the reference artifact, as `:138` itself warns. The table prints the footgun the warning box above it forbids |
| `ARTIFACTS.md:143` | "drive the whole thing through `pipelines finetune_data --out DIR`" | that command aborts: `--max-spend` is mandatory once `worked-examples` is selected. `ARTIFACTS.md` never mentions `--max-spend` on the pipeline or `--offline`. **Following ARTIFACTS.md alone, the pipeline does not run.** This is the one place it does not stand on its own |
| `ARTIFACTS.md:151-154` | implies `--out DIR` produces a coherent build | `render` is hardcoded to `outputs/part3/part3.md` and embeds shipped assets; the scratch `figures`/`examples` output is never consumed |

Secondary docs, for completeness:

| line | says | actual |
|---|---|---|
| `FINETUNE.md:321` | "`data/svgloc_mr` is undocumented — absent from `data/README.md`" | **`data/README.md:12` documents both `svgloc_mr/` and `sft_bbox/`.** Stale |
| `FINETUNE.md:222` | `audit --dataset data/svgloc_mr  # read-only` | writes a 973 KB gallery |
| `FINETUNE.md:311` | "The size bins it quotes (0.039% – 2.751%) match the shipped manifest" | endpoints match `data/svg_localization`; the section is about `data/svgloc_mr`, whose range is 0.0067–0.4269%. Ambiguous, and it papers over the 3-vs-5 bin mismatch |
| `FINETUNE.md:83` | attributes "0.899 / 18.5%" to a measurement | neither figure is in `part3.md`. `ExactInk.verify` is documented and called from no CLI (`grep -rn '\.verify(' blindspot/*.py` → 0 hits) |

---

## Nothing broke

No command errored except where designed to. Every non-zero exit was a guard
firing correctly: the `--out` refusals, the missing `--max-spend`, the `$0` no-op
(exit 2), and the two poisoned-scene `SystemExit`s.

## Housekeeping

`git status` gained ` M blindspot/report_pages.py` (95 insertions) at 21:41 —
**mid-session, and not mine.** I wrote to `/tmp` only; my one repo write is this
file. Flagging it because the working tree changed under the run.
