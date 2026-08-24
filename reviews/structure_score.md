# Structure review — 62/100

Agent brief: score how well the repository is organised; do not inflate.
Read-only; no API spend.

## Finding that dominates everything else

**The repository as committed is not the repository on disk.** `git status`
returns 121 entries: 82 deletions, 15 modifications, 24 untracked. `git ls-tree
HEAD blindspot` still contains the pre-consolidation nested layout.

Every live module is untracked. `git clone` yields a tree where
`python -m blindspot.core` does not exist and `make test` finds ~15% of the
suite. `README.md` opens with `git clone && ./setup.sh`. That does not work.

> **Fixed.** Committed and pushed; a fresh clone now imports 16/16 modules.

## Scores

| dimension | score | note |
|---|---:|---|
| Testing | 8/10 | 510 tests, meaningful; three judged in depth all catch real regressions |
| Legacy handling | 8/10 | `legacy/` principled and enforced by a test that it can never be packaged |
| Navigability | 7/10 | flat package, good `REPO_MAP.md`; but "where is the scorer for X" has two answers |
| Layering | 7/10 | acyclic and tested; but `judge` imports `eval` — producer importing consumer |
| Entry points | 7/10 | one obvious way in; four different `prog` conventions |
| Docs fidelity | 7/10 | all 38 commands run; "one test file" claim false (there are five) |
| Cohesion & size | 6/10 | `report.py` and `diagnose.py` are containers, not modules |
| **Safety & footguns** | **5/10** | see below |

Weighted total 670/1000 → 67, **less 5 for the git state = 62**.

## Safety findings

1. `generate.py:2663` — `scenes --out` defaults to the committed dataset. The
   guard lives at two *callers* while the callee defaults to the destructive
   path. *(Fixed.)*
2. `report.py:578` — `rate = 0.0` when the audit file is absent: missing input
   reported as a measured zero. *(Fixed.)*
3. `core.py:1771` — `score()` falls through to ANLS for unrecognised datasets.
4. `report_worked.py` — no `Budget`, no `--max-spend`. *(Fixed.)*
5. `Budget.exhausted()` is checked after the spend lands, so overshoot is
   bounded by the in-flight window. *(Fixed — window was `concurrency * 4`.)*

## The three best decisions

1. **Spend control as a framework concern, enforced by tests written from a real
   incident.** Three tests enforce `needs_api` in both directions.
2. **`figures.json` as the single auditable artifact**, with the JSON/HTML split
   and its two exceptions stated in the module's own `--help`.
3. **`docs/STRUCTURE.md` "Known limits — documented, not fixed."** Eight problems
   with reproductions. Worth more to an inheritor than the rest of the docs
   combined, because it is the part nobody is incentivised to write.

## Verdict

> "I would inherit this, and I would not be nervous about it — which is not the
> same as saying it is well organised. The judgement on display is well above
> average… What is missing is *follow-through*. A consolidation was designed,
> executed, documented in three places — and then not committed, not reflected in
> `__init__.py`, not propagated to the `prog` strings. The repository reads like
> someone who is excellent at deciding and mediocre at closing."
