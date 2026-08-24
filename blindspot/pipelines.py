"""Every pipeline, in one file.

Three efforts, one launcher. A pipeline owns no logic -- it is an ordered list of
steps, each naming a module in this package and the arguments this
effort wants from it. Keeping all three here makes the differences between them
visible on one screen, which is the point: they share a scorer, a runner and a
report chain, and only the argument lists differ.

    python -m blindspot.pipelines --list                       # all three, summarised
    python -m blindspot.pipelines literature_eval --list
    python -m blindspot.pipelines literature_eval --all --max-spend 40
    python -m blindspot.pipelines synth_localization_eval --task counting --stage run eval --max-spend 3
    python -m blindspot.pipelines finetune_data --all --offline

Runbooks: docs/runme/{BENCHMARKS,SYNTHETIC,FINETUNE}.md
"""

from __future__ import annotations

import sys
from pathlib import Path

from blindspot import flow
from blindspot.flow import Step, main

# ============================================================ 1. literature_eval
# Evaluate on the published benchmarks we download. Feeds blindspots.md 1-4.

BENCHMARKS = ["charxiv", "ai2d", "slidevqa", "infographicvqa", "screenspot_pro"]


def literature_eval(_opts) -> dict[str, list[Step]]:
    return {
        "download": [
            Step("hf-hosted", ["-m", "blindspot.download", "hf", "--only", *BENCHMARKS[:4]],
                 note="streaming; writes data/<name>/images + manifest.jsonl"),
            Step("screenspot-pro", ["-m", "blindspot.download", "screenspot-pro"],
                 note="not load_dataset-able: per-app JSON + images"),
            Step("github-sources", ["-m", "blindspot.download", "github-sources"], optional=True,
                 note="needs clones in third_party/"),
        ],
        "run": [
            Step(f"eval:{ds}", ["-m", "blindspot.core", "--datasets", ds],
                 needs_api=True, spend=8.0)
            for ds in BENCHMARKS
        ] + [
            Step("official:screenspot_pro",
                 ["-m", "blindspot.run_api", "official", "--datasets", "screenspot_pro", "--full"],
                 needs_api=True, spend=5.0,
                 note="the benchmark's own protocol, so the number is leaderboard-comparable"),
        ],
        "controls": [
            Step("blind", ["-m", "blindspot.run_api", "controls", "--arm", "blind",
                           "--datasets", *BENCHMARKS, "--n", "400"],
                 needs_api=True, spend=3.0,
                 note="the same question with the image withheld -- run this for anything "
                      "you intend to call a perception failure"),
            Step("onepage", ["-m", "blindspot.run_api", "controls", "--arm", "onepage",
                             "--datasets", "slidevqa", "--n", "400"],
                 needs_api=True, spend=2.0),
            Step("grid", ["-m", "blindspot.run_api", "grid", "--n", "350"],
                 needs_api=True, spend=1.0, note="fails to locate, or fails to say where?"),
        ],
        "judge": [
            Step("charxiv", ["-m", "blindspot.judge", "charxiv", "--dataset", "charxiv",
                             "--judge-model", "claude-opus-5"],
                 needs_api=True, spend=4.0,
                 note="string matching is a LOWER BOUND on free text; the published "
                      "protocol requires this"),
            Step("infovqa-equiv", ["-m", "blindspot.judge", "equiv",
                                   "--dataset", "infographicvqa"], needs_api=True, spend=2.0),
            Step("gt-audit", ["-m", "blindspot.judge", "gt-audit", "--dataset", "charxiv",
                              "--per-category", "5"],
                 needs_api=True, spend=2.0, note="is the BENCHMARK wrong? -> contested-gold floor"),
        ],
        "eval": [Step("aggregate", ["-m", "blindspot.eval", "aggregate"],
                      note="-> outputs/summary.json")],
        "diagnose": [
            Step("failure-modes", ["-m", "blindspot.diagnose", "failure-modes"]),
            Step("coordinates", ["-m", "blindspot.diagnose", "coordinates"]),
            Step("capability", ["-m", "blindspot.diagnose", "capability"]),
            Step("gt-quality", ["-m", "blindspot.diagnose", "gt-quality"], optional=True),
            Step("dataset-page", ["-m", "blindspot.diagnose", "dataset-page"]),
        ],
        "report": [
            Step("figures-json", ["-m", "blindspot.report", "data"]),
            Step("figures-png", ["-m", "blindspot.report", "examples"]),
            Step("tables", ["-m", "blindspot.report", "tables"]),
            Step("index", ["-m", "blindspot.report", "index"]),
            Step("paste-html", ["-m", "blindspot.report", "paste"], optional=True),
        ],
    }


# =================================================== 2. synth_localization_eval
# Generate our dataset and evaluate on it. Feeds blindspots.md 5-7.
#
# Three question types over ONE set of 200 scenes, derived without re-rendering
# so all three ask about identical pixels. The contrast between them is the
# finding: the model knows what is on the page, not where.

TASKS = {
    # `eval`/`report` are full argv tails: the per-task modules were merged into
    # blindspot.eval / blindspot.report, so the task is now the SUBCOMMAND. Two
    # tasks sharing a subcommand still collapse to one step, as they did when
    # they shared a module.
    "localization":   {"dataset": "svg_localization", "spend": 5.0,
                       "eval": ["blindspot.eval", "localization"],
                       "report": ["blindspot.report", "svgloc"]},
    "text_existence": {"dataset": "svg_word_mc", "spend": 2.0,
                       "eval": ["blindspot.eval", "derived"],
                       "report": ["blindspot.report", "svgderived"]},
    "counting":       {"dataset": "svg_counting", "spend": 2.0,
                       "eval": ["blindspot.eval", "derived"],
                       "report": ["blindspot.report", "svgderived"]},
}
COMMITTED = "data/svg_localization"      # source of truth -- never a generate target


def synth_localization_eval(opts) -> dict[str, list[Step]]:
    tasks, out = opts["tasks"], opts["out"]
    derived = [k for k in tasks if k in ("text_existence", "counting")]
    data = out or COMMITTED

    gen = []
    if out:
        gen.append(Step("scenes+localization",
                        ["-m", "blindspot.generate", "scenes", "--count", "200",
                         "--complexity", "4", "--seed", "17", "--out", out],
                        note=f"builds a NEW set in {out}; the committed set is untouched"))
        if derived:
            gen.append(Step("derive:" + "+".join(derived),
                            ["-m", "blindspot.generate", "questions", "--data", out],
                            note="re-renders nothing -- new questions over existing images"))

    audit = [Step("ground-truth",
                  ["-m", "blindspot.generate", "audit", "--data", data,
                   "--out", f"{data}/verify"],
                  note="overlays positioned from manifest.jsonl ALONE, so a bug in the "
                       "layout code cannot cancel itself out"),
             Step("examples", ["-m", "blindspot.generate", "examples",
                               "--data", data, "--out", f"{data}/examples"], optional=True)]

    run: list[Step] = []
    if "localization" in tasks:
        run += [Step("localization", ["-m", "blindspot.core",
                                      "--datasets", "svg_localization"],
                     needs_api=True, spend=5.0),
                Step("localization:probe", ["-m", "blindspot.run_api", "probe",
                                            "--rung", "small", "--n", "100"],
                     needs_api=True, spend=1.0,
                     note="harness sanity check -- run before believing a low score")]
    if derived:
        run.append(Step("derived:" + "+".join(derived),
                        ["-m", "blindspot.run_api", "derived", "--datasets",
                         *[TASKS[k]["dataset"] for k in derived]],
                        needs_api=True, spend=sum(TASKS[k]["spend"] for k in derived),
                        note="includes the BLIND control for each -- the image withheld"))
    if "localization" in tasks:
        run.append(Step("localization:ablations", ["-m", "blindspot.run_api", "ablations", "--n", "300"],
                        needs_api=True, spend=8.0,
                        note="8 arms: repeat careful describe cell_then_point landmark "
                             "crop bbox quadrant_mc"))

    ev, rep, seen_e, seen_r = [], [], set(), set()
    for k in tasks:
        if tuple(TASKS[k]["eval"]) not in seen_e:
            seen_e.add(tuple(TASKS[k]["eval"]))
            ev.append(Step(k, ["-m", *TASKS[k]["eval"]]))
        if tuple(TASKS[k]["report"]) not in seen_r:
            seen_r.add(tuple(TASKS[k]["report"]))
            rep.append(Step(k, ["-m", *TASKS[k]["report"]], optional=True))
    if "localization" in tasks:
        ev.append(Step("localization:ablations", ["-m", "blindspot.eval", "ablations"]))

    return {"generate": gen, "audit": audit, "run": run, "eval": ev, "report": rep}


# ================================================================ 3. finetune_data
# Build the SFT / GRPO training data. Feeds part3.md. Constructs data, never trains.

MR_DATA = "data/svgloc_mr"


def finetune_data(_opts) -> dict[str, list[Step]]:
    return {
        "ladder": [
            Step("gen-multires", ["-m", "blindspot.generate_finetune", "ladder",
                                  "--out", MR_DATA],
                 note="6 aspect ratios x 4 sizes, up to the largest that reaches the "
                      "model without downscaling (1568px edge, ~1.15MP)"),
        ],
        "build": [
            Step("sft-bbox", ["-m", "blindspot.generate_finetune", "samples",
                              "--n", "20", "--seed", "0",
                              "--out", "data/sft_bbox/sft_bbox_20.jsonl"],
                 note="target is a BOX, not a point: a point has thousands of equally "
                      "correct answers, a box has one and overlap is a graded signal"),
        ],
        "verify": [
            Step("multires-audit", ["-m", "blindspot.generate_finetune", "audit",
                                    "--dataset", MR_DATA]),
            Step("gallery", ["-m", "blindspot.report_finetune", "gallery"],
                 note="every record shown twice -- full frame and a zoom -- because at "
                      "900x570 a 0.02% target is a few pixels"),
        ],
        "report": [
            Step("figures", ["-m", "blindspot.report_finetune", "figures"]),
            Step("examples", ["-m", "blindspot.report_finetune", "examples"]),
            # CALLS THE API. The script has no --max-spend of its own, so the
            # framework can gate it (key check, --offline, ceiling prompt) but
            # cannot cap it mid-run. Output is model-sampled: --seed picks WHICH
            # records are asked about, not what comes back.
            Step("worked-examples", ["-m", "blindspot.report_worked"],
                 needs_api=True, optional=True,
                 note="~24 model calls; no internal spend cap"),
            Step("render", ["-m", "blindspot.render_markdown",
                            "--src", "outputs/part3/part3.md", "--paste"],
                 note="HTML generated from the markdown, never hand-edited"),
        ],
    }


PIPELINES = {
    "literature_eval": (literature_eval, "evaluate on the published benchmarks"),
    "synth_localization_eval": (synth_localization_eval, "generate our dataset and evaluate on it"),
    "finetune_data": (finetune_data, "build the SFT / GRPO training data"),
}


def _usage() -> int:
    print(__doc__.strip().split("\n\n")[0])
    print("\npipelines:")
    for name, (fn, desc) in PIPELINES.items():
        stages = list(fn({"tasks": list(TASKS), "out": None}))
        print(f"  {name:<26} {desc}")
        print(f"  {'':<26} stages: {' -> '.join(s for s in stages if s)}")
    print("\n  python -m blindspot.pipelines <pipeline> --list")
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help") or (argv[0] == "--list" and len(argv) == 1):
        raise SystemExit(_usage())
    name, argv = argv[0], argv[1:]
    if name not in PIPELINES:
        raise SystemExit(f"unknown pipeline {name!r}. choose from {list(PIPELINES)}")

    # pipeline-specific options, peeled off before the generic parser
    opts = {"tasks": list(TASKS), "out": None}
    if "--task" in argv:
        i = argv.index("--task"); picked, j = [], i + 1
        while j < len(argv) and not argv[j].startswith("-"):
            picked.append(argv[j]); j += 1
        bad = [p for p in picked if p not in TASKS]
        if bad:
            raise SystemExit(f"unknown task(s): {bad}. choose from {list(TASKS)}")
        opts["tasks"] = picked or opts["tasks"]; del argv[i:j]
    if "--out" in argv:
        i = argv.index("--out"); opts["out"] = argv[i + 1]; del argv[i:i + 2]
        # Resolve BOTH against the repository root, not the cwd. Comparing
        # `Path(COMMITTED).resolve()` resolves the guard's own reference against
        # wherever the process happens to be, so from any other directory an
        # absolute --out pointing straight at the committed dataset compares
        # unequal and the guard silently does not fire.
        if Path(opts["out"]).resolve() == (flow.ROOT / COMMITTED).resolve():
            raise SystemExit(
                f"refusing --out {COMMITTED}: that is the committed dataset and the source\n"
                f"of truth for every published number. The generator has drifted from it, so\n"
                f"regenerating in place would rebind uids to different questions.\n"
                f"See docs/runme/SYNTHETIC.md section 0.")

    sys.argv = [f"blindspot.pipelines {name}", *argv]
    stages = {k: v for k, v in PIPELINES[name][0](opts).items() if v}
    raise SystemExit(main(name, stages, __doc__))
