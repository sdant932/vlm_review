# Review protocol

The reports in `reviews/` are **not** build artifacts. Nothing in `blindspot/`
produces them: they are written by LLM agents given a brief, a budget and no
prior knowledge of this repository. This file is how you regenerate them.

Why they exist: the people who build a study cannot audit its documentation,
because they already know what the commands are. Every finding in `reviews/`
came from an agent that did not.

---

## The reference reports

Every review is judged against the study's own written output. These are the
ground truth a reviewer compares its numbers to, and **neither is in this
repository** — they are separate deliverables:

| document | covers | where it lives |
|---|---|---|
| `blindspots.md` | the benchmark evaluation (§1–4) and the generated dataset (§5–7) | `/Users/sdoveh/gitrep/takehome/outputs/report/blindspots.md` |
| `part3.md` | the finetuning plan | `outputs/part3/part3.md` (generated, gitignored) |

A reviewer given no reference report will tell you the pipeline ran. It will not
tell you the number is wrong. Both of the findings that mattered most —
a headline pooled from three protocols, and two numbers with no code behind them
— came from comparing against `blindspots.md`, not from reading the code.

If you move or lose these, the protocol below still runs but stops being an
audit and becomes a smoke test. Point the briefs at wherever they actually are.

---

## The five reviews

| report | question it answers | spends money? |
|---|---|---|
| `run_literature_eval.md` | can a stranger run pipeline 1 from the docs alone? | yes, capped |
| `run_synth_localization_eval.md` | …pipeline 2? | yes, capped |
| `run_finetune_data.md` | …pipeline 3? | one step, capped |
| `coverage_probe.md` | does the repo do what it says it does? | no |
| `structure_score.md` | how well is it organised, out of 100? | no |

---

## How to regenerate

Spawn one agent per report, in parallel. They are independent and must not share
context — the value comes from each arriving cold.

### The three pipeline runs

Give each agent **only** this much:

> You have been handed the repository at `<path>`. You know nothing about it
> beyond this message.
>
> Your job: get the **`<pipeline>`** pipeline running end to end on a small
> budget, and report whether it works.
>
> Figure out how from the repository's own documentation. Start at `README.md`
> and follow it. Do not ask me how it works — the point is to find out whether
> the docs stand on their own. If you get stuck, say exactly where and what was
> missing.
>
> Constraints: total spend under **$0.50**; use whatever flag the docs say
> controls that. `ANTHROPIC_API_KEY` is in the environment — never print, echo,
> cat or grep it or any `.env` file. Use `./.venv/bin/python`. Never
> `rm -rf`/`rm -r`. Do not modify any source file — if something is broken,
> report it, do not patch it. Write generated output to /tmp.
>
> Then compare your results against the study's published findings — read the
> reference report for your pipeline (paths in "The reference reports" above):
> `blindspots.md` for pipelines 1–2, `part3.md` for pipeline 3.
> Your sample will be far smaller, so exact agreement is not expected. Assess
> whether your numbers are *consistent* with the claims or flatly contradict
> them.
>
> Report: (a) what you ran and where you found each instruction; (b) spend and
> question count; (c) your numbers vs the report's; (d) every place the
> documentation was wrong, missing, ambiguous or a dead end — file and line;
> (e) anything that broke, with exact error text.
>
> Be blunt. A list of things that did not work is more useful than a success
> story.

**Deliberately withheld.** Do not name the commands, the module paths, or the
traps. Two things worth leaving as bait, phrased as they were:

- pipeline 2: *"there is at least one thing the docs tell you NOT to do — find
  it, and say whether the tooling actually enforces it or merely asks nicely."*
- pipeline 3: *"work out from the docs which part is the exception"* on cost,
  without naming the step that spends.

If an agent finds neither, the documentation has failed.

### The coverage probe

> Probe the repository and report what it actually covers. You are an outside
> reviewer. Read the code and the data, not just the prose. Then answer: **does
> what this repository does match what it says it does?**
>
> Work out for yourself: what the study is; which datasets, tasks, metrics and
> controls the code covers, and which are declared but not reachable; where the
> description overstates or understates the code; what is missing to reproduce
> the study.
>
> Then check each headline claim in the two reference reports — `blindspots.md`
> and `part3.md`, paths above — against the code and data. Where a number is quoted, find where it comes from. **Report
> any number you cannot trace.**
>
> Also assess honesty: does the repo disclose its own limitations, or present
> results more confidently than the evidence supports? Look for stated caveats,
> blind controls and known-issue sections — and for problems that are NOT
> disclosed.
>
> Lead with anything a reader would be misled by. Cite file:line. Do not be
> diplomatic.

### The structure score

> Score the structure of this repository. Give a numeric score and defend it.
>
> Assess, each out of 10 with reasons: navigability; module cohesion and size
> (judge each large file individually, with line counts); layering and
> dependency direction (verify by reading imports, not the docs); entry points;
> testing (**pick three tests and judge whether they would actually catch a
> regression**); documentation-to-code fidelity (spot-check at least ten
> documented commands by running them); handling of legacy and dead code;
> safety and footguns.
>
> Then: an overall score out of 100 with your chosen weighting; the three things
> most worth fixing, in priority order, with reasoning for the order; the three
> best decisions in the repository; and a one-paragraph verdict — would you be
> comfortable inheriting this, and what would you do first?
>
> Be harsh but fair. **Do not inflate the score to be encouraging — an honest 60
> with specific reasons is far more useful than a diplomatic 85.**

---

## Reading the results

Judge the reviews before you judge the repository:

- A pipeline agent that finishes without listing a documentation defect either
  got lucky or was told too much. Check what you put in the brief.
- A structure score above ~85 on first pass usually means the reviewer was
  primed, not that the repo is excellent.
- The coverage probe is the one that finds *scientific* problems rather than
  engineering ones. Its findings are the expensive kind: an untraceable number
  is a claim nobody can check.

Findings that are about the study rather than the code — a contaminated
headline, an unpublished ablation — are **not** for the reviewer to fix, and the
briefs say so. They come back as decisions.

## Cost

Roughly **$1.50** for the three pipeline runs at a $0.50 cap each; the probe and
the score spend nothing. Budget more if you raise the caps — and read
`reviews/run_synth_localization_eval.md` first, which is the report of a run that
asked for $0.10 and was billed $3.24 because `--max-spend` was not being
enforced. That defect is fixed; the report is kept as the reason the fix exists.
