# `blindspot/report_pages.py` — port fidelity audit

**Date:** 2026-08-23 · Source audit, no API calls, no source modified. A second,
independent pass over the same file as `report_pages_audit.md`; it goes deeper on port
fidelity and finds the silent-wrong output that the first pass missed.

## Verdict

**The port is faithful. The two things the port *added* are the broken ones.**

After normalising the seven documented rename families, **131 of 146 function and class
bodies are byte-identical to their `legacy/` originals, and 77 of 82 module-level
constants are byte-identical.** All 15 body and 5 constant differences are deliberate and
accounted for. No string literal, keyword argument, f-string expression or method name was
collaterally renamed — verified by re-running the diff with attribute renaming disabled
and getting the same counts. So "matches the original to within a few bytes" is not luck.

The defects live in the ~150 new lines.

## Findings

| # | finding | where | class | port-introduced |
|---|---|---|---|---|
| F1 | guard checks 6 of 13 input families; `causes` still `TypeError`s past it | `:3216`, `:210-215` | crash | **yes** |
| F2 | `MIN_CAUSE_ROWS = 200` exceeds what this checkout's `data/` can produce | `:398` | unrunnable | **yes** |
| F3 | 7 `A − B` sites with both operands nullable, unpatched | below | crash | inherited |
| F4 | the whole `slidevqa` section was never patched; same bug at `:5084`, `:5116` | `:5048` | crash | inherited |
| **F5** | **`slidevqa` publishes 1-decimal effect sizes from n=1 with no n, exit 0** | `:5993-5996` | **silent-wrong** | inherited |
| F6 | `drilldown` writes all three artifacts, *then* crashes (exit 1) | `:4859` | crash-after-write | inherited |
| F7 | `slidevqa --skip-images` emits 60 broken `<img>`; `causes --no-images` does not | `:5676` vs `:900` | broken output | inherited |
| F8 | `pct_bare` at `:420` is dead, shadowed by the identical one at `:796` | `:420` | dead dup | **yes** |
| F9 | the run tag literal is duplicated 11 ways in one file | below | latent | **yes** (merge) |
| F13 | `eval.load_rows` ≠ legacy `aggregate.load_rows`; `tasks` reads a different file set | `eval.py:219` | input change | **yes** |
| F14 | orphaned `outputs/aug22/summary.json` still on disk with different numbers | `:7721` | trap | **yes** |

### F5 — the actual silent-wrong output, and it is not `causes`

`slidevqa` on a thin tree exits 0 and writes a 71 KB page. Its headline tiles:

| tile | published | n behind it | n shown |
|---|---|---|---|
| Evidence-condition F1 | `67.0` | 20 | yes |
| Retrieval cost | `+0.0` | 8 paired | yes |
| Integration cost | `−7.5` | **8 vs 12** | **no** |
| Derivation cost | **`−70.5`** | **1 vs 19** | **no** |

The n *exists* — `a["costs"][i]["basis"]` carries it (`:5113`, `:5118`, `:5123`) and it
reaches the bar-chart tooltip at `:6009` (`evidence condition, n=1 vs 19`). The **tile**,
which is the number a reader quotes, discards it for a hardcoded string. Grepping all
1,485 lines of that section for `small`, `MIN_`, `min_n` finds two hits, both in prose:
no `SMALL_N`, no CI, no precondition. Contrast `drilldown`, which labels every node under
`SMALL_N = 30` (`:3271`), and `tasks`, which prints `accuracy 83% / n=12 · CI 55-95%` on
the same tree.

A one-decimal F1 delta from a single question, on a page titled *"what does it cost to
find the slide?"*, is the worst outcome available here.

### F1 — the guard's blind spot, measured

`d.counts` is populated only by the loop over `CAUSE_MAIN_FILES` (`:210-211`). The four
other input families load on the lines immediately below and are never counted:

| input | loaded | counted | builders needing it |
|---|---|---|---|
| `CAUSE_MAIN_FILES` (6) | `:210-211` | **yes** | all 15 |
| `control_blind.jsonl` | `:212` | no | `c_language_prior_override`, `c_label_reference_binding` |
| `control_onepage0.jsonl` | `:213` | no | `c_cross_page` |
| `control_grid4.jsonl` | `:214` | no | `c_resolution_precision` |
| `{ds}__gtaudit.jsonl` | `:288`, `:2951` | no | `c_gt_noise` |
| `data/{ds}/manifest.jsonl` | `:226-231` | no | `Data.__init__` |

With the three controls truncated to zero bytes and the threshold satisfied, `causes`
gets past the guard and dies at `:1549` with the *same error class the guard exists to
prevent*. Per-builder, controls emptied: `c_language_prior_override` `:1549`,
`c_resolution_precision` `:1366`, `c_cross_page` `:2313`, `c_counting` `:2612` all
`TypeError`; `c_gt_noise` `:2951` `FileNotFoundError`. Ten pass.

### F3 — the shape the patch could not see

The `pct_or_dash`/`pct_bare` campaign patched 48 sites. Every site that **subtracts before
formatting** was skipped, because there is no formatter to substitute in. At `:2313`:

```python
("Take one of the two slides away", f"{(one - both) * 100:+.1f}pp",
 f"F1 {pct_bare(both, 1)} &rarr; {pct_bare(one, 1)} on the same {len(paired)} questions",
```

The *note* was patched. The *value on the same tile* was not. Same at `:2347`, `:1548`,
`:2609`. Live sites: `:1366`, `:1549`, `:1627`, `:2313`, `:2347`, `:2397`, `:2612`, plus
`:2854` latent. Division/format variants at `:1351`, `:1377`, `:2428-2445`, `:3023`,
`:3919`, `:4859`.

### F9 — the run tag, 11 times

`TAG` (`:151`), five inline literals in `DRILL_MAIN_FILES` (`:3278-3282`), `ALLPAGES_FILE`
(`:3284`), `JUDGE_FILE` (`:3285`), `EVIDENCE_JSONL` (`:4867`), `ALLPAGES_JSONL` (`:4868`),
`RUN` (`:7367`). Across seven modules this was expected duplication; in one file, changing
`TAG` alone silently leaves `drilldown`, `slidevqa` and `candidates` reading a different
run than `causes`.

### F13 — the one non-equivalent substitution

| | legacy `aggregate.load_rows` | port `eval.load_rows` |
|---|---|---|
| selection | glob union of every run | canonical tag alone, or a loud `ValueError` |
| two protocols | silently pooled | picks `CANONICAL_TAG`, refuses to guess |
| duplicate uid | last-writer-wins by filename | `ValueError` unless `allow_mixed_protocol` |

`cmd_tasks` (`:6636`) calls `load_rows(ds)` untagged, so `tasks` now reads a different
file set than `task_pages` did. Measured on this tree the pages are identical — the two
`screenspot_pro` arms happen to cover the same 26 uids — so this is coincidence of
coverage, not equivalence. The new behaviour is better; the docstring doesn't mention it.

## Two of its claims did not hold when I checked them

- **"the abort leaves `outputs/causes/` behind, which `:3200` promises it will not."**
  Not reproduced. `outputs/causes/` and `outputs/assets_causes/` exist here but are empty
  and pre-date the run; re-running the abort left the directory mtime **unchanged**. The
  mkdir-after-check ordering holds.
- **F2's "can never succeed in this repository."** Overstated. The 200 threshold is
  unreachable because this checkout's `data/` holds only 20–50 downloaded rows per
  benchmark, not because of any fixed ceiling — `blindspot.download` lifts it. The
  narrower true finding stands: `causes` cannot run on the sample-sized data the repo
  ships with, and the abort message says *score the benchmarks first* without mentioning
  you must **download** more of them first.

## Fidelity checks that passed

- 19 of 23 cross-module helper substitutions byte-identical to their origins
  (`is_na`/`wilson`/`quantiles`/`cell_of`/`centre_cell`/`bbox_cells`, `score`/`anls`/
  `token_f1`, `classify`/`classify_point`, `load`/`slidevqa`, `as_model_saw`/`fit`/
  `draw_target`/`busiest_crop`). The other four are renames only, plus F13.
- **`LABELS` and `FM_LABELS` are not crossed** — two dicts of the same name and arity
  imported into one module was the likeliest merge error, and it did not happen.
- `esc`: 7 legacy definitions, 5 byte-identical, the other 2 behaviourally identical
  (`html.escape`'s `quote` defaults to `True`).
- No unused imports; no unreachable branch; only three dead symbols (`:420`, `:792`,
  `:3484`), two of them already dead in `legacy/`.
- `outputs/aug22/summary.json` → `outputs/report/summary.json` (F14) is the **correct**
  move; the trap is only that the stale 29 KB file is still on disk with different values.
