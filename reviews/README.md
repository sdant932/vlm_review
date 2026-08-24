# Reviews

Five independent agent reviews of this repository, each written by an agent that
arrived with no prior knowledge of it. **These are not build artifacts** — see
[../docs/REVIEWS.md](../docs/REVIEWS.md) for how to regenerate them.

| report | verdict in one line |
|---|---|
| [structure_score.md](structure_score.md) | **62/100.** Judgement well above average, follow-through poor. |
| [coverage_probe.md](coverage_probe.md) | Central finding reproduces exactly; one headline is contaminated; two numbers have no code. |
| [run_literature_eval.md](run_literature_eval.md) | Chain intact download → `summary.json`, broken from there to `outputs/report/`. |
| [run_synth_localization_eval.md](run_synth_localization_eval.md) | Headline contrast reproduces at 1/20 sample. Asked for $0.50, was billed $3.40. |
| [run_finetune_data.md](run_finetune_data.md) | Ladder claims verified exactly; the plan is ~30% implemented. |

Each was compared against the study's reference reports — `blindspots.md` and
`part3.md` — which is where the substantive findings came from. Reading the code
alone produced engineering complaints; reading it *against the claims* produced
the ones that mattered.

Findings that are about the study rather than the code came back as decisions,
not fixes. The fixes that were made are noted inline below each report.
