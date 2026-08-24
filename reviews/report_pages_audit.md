# `blindspot/report_pages.py` — source audit of the 8-module merge

**Date:** 2026-08-23 · **Method:** source audit, no API calls. Findings below were
re-verified against the file directly after the agent reported them.

## Verdict

The port is materially clean — an AST-level diff of all 172 defs against
`legacy/blindspot/reporting/*.py` found **no unintentional silent behaviour change**.
But the `MIN_CAUSE_ROWS` guard added during the fix round covers **less than half the
surface it needs to**.

`cause_mean()` returns `None` for an empty group. `pct_or_dash`/`pct_bare` make that safe
wherever a single mean is *displayed*. Nothing makes it safe where two means are
*subtracted, divided or multiplied* before display — and that composition is unguarded in
**8 of the 15** `c_*` builders. `MIN_CAUSE_ROWS` checks only the row counts of the six
main per-benchmark files, so it cannot see any of the three real triggers:

1. a **control file** (`control_grid4`/`control_onepage0`/`control_blind`) empty or with
   zero overlapping uids — `Data.counts` is built only from `CAUSE_MAIN_FILES`
   (`report_pages.py:208-211`); `self.blind`/`onepage`/`grid4` load afterwards
   (`:213-215`) and are never counted;
2. a **within-benchmark subgroup** (a qtype, `split`, or `operation` tag) empty while the
   parent benchmark passes;
3. a **cross-file intersection** (`slidevqa` ∩ `slidevqa_allpages` by uid) empty while
   both files individually pass.

All three were reproduced by calling the real builders with constructed `Data` fixtures.

### Reproduced

| # | trigger | crash |
|---|---|---|
| a | `control_grid4.jsonl` empty, ScreenSpot-Pro fine | `TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'` at `:1366`, `delta = name_acc - click_acc` |
| b | `slidevqa` ∩ `slidevqa_allpages` empty, both files individually fine | `TypeError` at `:2424`; sibling at `:2428` is `ti_a / ti_e`, a **division** — partial overlap gives `ZeroDivisionError` or `TypeError` depending on which side survives |
| c | 250 AI2D rows all one qtype — **over** the 200 threshold | `TypeError: ... for *: 'NoneType' and 'int'` at `:1649`, `f"{a_lr * 100:.1f}%"` |

(c) is the sharpest counter-example: the threshold checks the benchmark total, not the
partition a builder actually needs.

**The tell that this was missed:** at `:1649` `a_lr * 100` is interpolated bare, while
`:1650` on the very next line *does* call `pct_or_dash`. The mechanical patch edited the
adjacent site and skipped this one. `c_derivation_vs_reading` is 100% AST-identical to
legacy past the `mean`→`cause_mean` rename — the patch never reached it at all.

### Full catalogue

| function | line | nullable source | trigger |
|---|---|---|---|
| `c_resolution_precision` | 1332 | `name_acc`/`click_acc` (1365→1366) | `grid4` empty — **repro a** |
| `c_language_prior_override` | 1502 | `cxp` inline 1549, 1626 | `blind` has no CharXiv pairs |
| `c_label_reference_binding` | 1632 | `a_lr`/`a_dr` (1636), `a13`/`a12` (1646); 16 sites 1649-1734 | any qtype/qid subgroup — **repro c** |
| `c_derivation_vs_reading` | 2049 | `a_d`/`a_r`, `f_l`/`f_a`, `c_l`/`c_a`; 18 sites 2063-2166 | charxiv `split` / slidevqa `arithmetic` empty |
| `c_cross_page` | 2282 | `f_s`/`f_m`, `c_s`/`c_m`, `one`/`both`/`still`/`both_ok`; 12 sites | `onepage0` empty, or `single`/`multi` empty |
| `c_retrieval` | 2406 | 10 means keyed on `common`; 15 sites | uid intersection — **repro b** |
| `c_counting` | 2529 | `iv_c`/`iv_rest` inline 2612 | infographicvqa `counting` subset empty |
| `c_subplot_scope` | 2745 | `named`/`noprefix` inline 2854 | no multi-panel questions sampled |

A ninth, `c_gt_noise` (`total_floor`, 2994→3023), is unguarded but unlikely in practice.

**All nine are inherited from the legacy study code, not introduced by the port.** The
patch touched only the ~49 sites where a *single* mean was interpolated bare; not one
arithmetic composition of two means was touched.

## The other six subcommands on sparse results

No silent-wrong-output in any of them — every number on every page carries its own `n=`,
visibly small where small. The gap is uglier-but-louder:

| subcommand | empty `results/` | thin-but-real `results/` |
|---|---|---|
| `causes` | clean abort, exit 2 | clean abort, exit 2 |
| `drilldown` | **bare `FileNotFoundError`**, exit 1 (`:4779`) | exit 0, 0 arithmetic violations |
| `slidevqa` | **bare `FileNotFoundError`**, exit 1 (`:5000`) | exit 0, 20/20 + 8/8 verified |
| `tasks` | clean abort, exit 2 (new vs legacy) | exit 0 |
| `primitives` | **bare `FileNotFoundError`** on `outputs/summary.json` | exit 0 after `eval aggregate` |
| `headline` | **bare `FileNotFoundError`** on `report/summary.json` | exit 0 after `report summary` |
| `candidates` | clean abort, exit 2 (new vs legacy) | exit 0; skips a zero-candidate cause without aborting |

`drilldown`/`slidevqa` crash with no `ABORT:` message and no exit-2 convention when one
required file is absent — a green-field clone hits this on first run.
`primitives`/`headline` crash the same way but against a documented file dependency.

## `--skip-images` does not keep its promise

`causes --no-images` is honest: `_thumb_html` (`:900-906`) omits the `<figure>` when the
asset is missing, so no broken refs.

`slidevqa --skip-images` is not. Thumbnails are built **client-side** —
`thumbSrc = (dk,p) => \`assets_slidevqa/${dk}_${pad2(p)}_t.jpg\`` (`:5676`) interpolated
unconditionally into `<img src>` (`:5803`) with no existence check. Reproduced: exits 0,
prints "assets 0.0 MB in 0 files", writes 12 examples whose thumbnails all 404 because
the directory does not exist. Both flags' help text makes the same promise; one
implementation keeps it.

## What the test suite would and would not catch

24 passed. It would notice a missing file. It would **not** notice a wrong number:

- `test_causes_writes_one_page_per_builder_plus_an_index` monkeypatches `rp.BUILDERS` to
  trivial closures with hardcoded strings. Not one real `c_*`, not one real `cause_mean`,
  is ever executed.
- `test_causes_refuses_a_run_too_thin_to_compare` patches `Data` and returns before
  `BUILDERS` is iterated. Its own fixture sets `_FakeData.grid4 = {"__meta__": ...}`
  (`tests/test_report_pages.py:132`) — **exactly the shape that crashes
  `c_resolution_precision`** — and never feeds it to a real builder.
- The six non-`causes` subcommands are never run beyond `--help`.

The suite's docstring names this exact `TypeError` as its motivating history and still
does not test for it.

## Smaller findings

- **The module docstring is wrong about itself** (`:81-85`): it claims `OUT` is "defined
  once, below", but `AUG22_OUT` (`:7009`) and `CAND_OUT` (`:7365`) are the same concept
  under two names with different paths. Code correct, manifest overclaims.
- `as_model_saw`/`fit`/`draw_target` (`:132`) are attributed to `blindspot.report` but
  originate in legacy `report_examples.py`. Import is safe; attribution is wrong.
- Dead: `pctf` (`:792`, duplicate of `pct_or_dash`), `_b` (`:3484`) — both already dead in
  legacy. `[x for x in hard if False][:0]` (`:2830`) is a permanently-empty gallery slot,
  inherited verbatim.
- `read_jsonl` (`:173`) deliberately shadows `core.read_jsonl` with a different arity —
  misuse fails loudly, not silently.
- Three deliberate new behaviours (`causes`/`tasks`/`candidates` aborting instead of
  writing empty output) are undocumented in the docstring's "collisions resolved" manifest.

## Invariants confirmed

`is_na`/`wilson`/`quantiles`/`cell_of`/`centre_cell`/`bbox_cells` imported from `core`,
not redefined. `RESULTS`/`GOOD`/`BAD` defined once each. The three `*_format_equivalent`
and two `*_pct` functions are genuinely distinct, no cross-calling. No unused imports, no
duplicate top-level defs besides the known `pct_bare`, no unreachable branch from the merge.
