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
    python -m blindspot.pipelines finetune_data --all --out /tmp/finetune_dev --offline

Two pipelines take `--out`, and both mean the same thing by it: build somewhere
new. `synth_localization_eval --out` and `finetune_data --out` are opt-in --
without one, the steps that would write over a committed or reference artifact
are not scheduled at all, and an --out that resolves onto one is refused by name.

Runbooks: docs/runme/{BENCHMARKS,SYNTHETIC,FINETUNE}.md
"""

from __future__ import annotations

import os
import sys
import textwrap
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
            Step("summary", ["-m", "blindspot.report", "summary"],
                 note="-> outputs/report/summary.json, which `data` READS as a FILE, "
                      "not an import -- so it has to come first or `data` assembles "
                      "figures.json from a stale (or absent) summary"),
            Step("figures-json", ["-m", "blindspot.report", "data"]),
            Step("figures-png", ["-m", "blindspot.report", "examples"]),
            Step("tables", ["-m", "blindspot.report", "tables"]),
            Step("index", ["-m", "blindspot.report", "index"]),
            Step("paste-html", ["-m", "blindspot.report", "paste"], optional=True),
        ],
        # The standalone pages: everything the report links out to rather than
        # contains. They read results/*.jsonl directly and recompute their own
        # numbers, so only the two summary-fed ones care about ordering --
        # `primitives` after `eval:aggregate`, `headline` after `report:summary`,
        # both of which this stage already follows.
        "pages": [
            Step("causes", ["-m", "blindspot.report_pages", "causes"],
                 note="-> outputs/causes/*.html + assets_causes/ -- 15 blind spots, "
                      "one page each, plus the index that ranks them"),
            Step("drilldown", ["-m", "blindspot.report_pages", "drilldown"],
                 note="-> outputs/drilldown.{html,json,csv}"),
            Step("slidevqa", ["-m", "blindspot.report_pages", "slidevqa"],
                 note="-> outputs/slidevqa.html + assets_slidevqa/"),
            Step("tasks", ["-m", "blindspot.report_pages", "tasks"],
                 note="-> outputs/tasks/*.html, one per perceptual primitive"),
            Step("primitives", ["-m", "blindspot.report_pages", "primitives"],
                 note="-> outputs/report.html, the overview the four pages above "
                      "crumb back to; reads outputs/summary.json as a FILE"),
            Step("headline", ["-m", "blindspot.report_pages", "headline"],
                 note="-> outputs/aug22/report.html; reads outputs/report/summary.json "
                      "as a FILE, so `report:summary` has to have run"),
            Step("candidates", ["-m", "blindspot.report_pages", "candidates"], optional=True,
                 note="-> outputs/report/candidates.html; optional because one of its "
                      "six pools needs the synth_localization_eval results"),
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
                   "--out", f"{data}/verify/index.html"],
                  note="overlays positioned from manifest.jsonl ALONE, so a bug in the "
                       "layout code cannot cancel itself out"),
             Step("examples", ["-m", "blindspot.generate", "examples",
                               "--data", data, "--out", f"{data}/examples/index.html"], optional=True)]

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
        # Optional because its only input, results/svgloc_ablation_uids.json, is
        # written by the `run` stage's `localization:ablations` API pass. Offline,
        # or on a clone that never spent the money, that file is simply absent --
        # which is not a broken pipeline and must not abort one. Documented at
        # docs/runme/SYNTHETIC.md, whose `--from eval --offline` line used to die here.
        ev.append(Step("localization:ablations", ["-m", "blindspot.eval", "ablations"],
                       optional=True,
                       note="the ablations were never run: this reads "
                            "results/svgloc_ablation_uids.json, which only a prior "
                            "`run_api ablations` pass (the `run` stage, needs the API) writes"))

    return {"generate": gen, "audit": audit, "run": run, "eval": ev, "report": rep}


# ================================================================ 3. finetune_data
# Build the SFT / GRPO training data. Feeds part3.md. Constructs data, never trains.

MR_DATA = "data/svgloc_mr"
PART3_MD = "outputs/part3/part3.md"      # prose, hand-written: read, never written

# Every artifact this pipeline used to overwrite on a bare `--all`. All six are
# gitignored and untracked, so an overwrite is unrecoverable -- and regeneration
# does NOT reproduce them: a fresh `ladder` emits the same 1513 uids with a
# different target_text/box_px for 91% of them, and `samples --seed 0` reproduces
# 5 of the 20 shipped records. So the steps that write them are scheduled ONLY
# against an explicit --out, exactly as synth_localization_eval's generate stage
# is. The irony this replaces: `generate_finetune samples` already refuses to
# default its --out, and this file used to hand the reference path straight back.
FINETUNE_REFERENCE = {
    MR_DATA: "the committed multi-resolution ladder; a rerun keeps the uids and "
             "changes 91% of the boxes",
    "data/sft_bbox/sft_bbox_20.jsonl": "the shipped SFT records; --seed 0 "
                                       "reproduces 5 of the 20",
    "outputs/finetune/gallery.html": "the gallery of those records",
    "outputs/finetune/worked_examples.json": "real model samples; not "
                                             "reproducible at all",
    "outputs/part3/assets": "the figures and example strip part3.md embeds",
    "outputs/part3/part3.html": "the rendered deliverable",
}


def finetune_out(base: str) -> dict[str, str]:
    """Where each writing step goes, given an --out base directory.

    One place, so the guard below checks the paths the steps will actually be
    given rather than a hand-kept copy of them.
    """
    return {
        "ladder": f"{base}/svgloc_mr",
        "sft-records": f"{base}/sft_bbox/sft_bbox_20.jsonl",
        "audit-gallery": f"{base}/gallery_svgloc_mr.html",
        "gallery": f"{base}/gallery.html",
        "assets": f"{base}/assets",
        "worked-examples": f"{base}/worked_examples.json",
        "part3-html": f"{base}/part3.html",
    }


def finetune_data(opts) -> dict[str, list[Step]]:
    out = opts.get("out")
    p = finetune_out(out) if out else None
    data = p["ladder"] if p else MR_DATA

    ladder: list[Step] = []
    build: list[Step] = []
    report: list[Step] = []
    if p:
        ladder = [
            Step("gen-multires", ["-m", "blindspot.generate_finetune", "ladder",
                                  "--out", p["ladder"]],
                 note="6 aspect ratios x 4 sizes, up to the largest that reaches the "
                      "model without downscaling (1568px edge, ~1.15MP)"),
        ]
        build = [
            Step("sft-bbox", ["-m", "blindspot.generate_finetune", "samples",
                              "--n", "20", "--seed", "0",
                              "--out", p["sft-records"]],
                 note="target is a BOX, not a point: a point has thousands of equally "
                      "correct answers, a box has one and overlap is a graded signal"),
        ]
        report = [
            Step("figures", ["-m", "blindspot.report_finetune", "figures",
                             "--dataset", data, "--out-dir", p["assets"]]),
            Step("examples", ["-m", "blindspot.report_finetune", "examples",
                              "--dataset", data, "--out-dir", p["assets"]]),
            # CALLS THE API. It now owns a --max-spend of its own and checks the
            # ceiling before every call, so the `spend` declared here reaches it:
            # Step.rendered appends the share only to a step that declares one.
            # Output is model-sampled: --seed picks WHICH records are asked
            # about, not what comes back.
            Step("worked-examples", ["-m", "blindspot.report_worked",
                                     "--dataset", data,
                                     "--out", p["worked-examples"]],
                 needs_api=True, optional=True, spend=0.5,
                 note="~24 model calls, capped by its own --max-spend"),
            Step("render", ["-m", "blindspot.render_markdown", "--src", PART3_MD,
                            "--out", p["part3-html"], "--paste"],
                 note="HTML generated from the markdown, never hand-edited"),
        ]

    verify = [
        # Read-only against the ladder, so it runs with or without --out. Its
        # gallery follows the dataset audited and never lands on gallery.html.
        Step("multires-audit", ["-m", "blindspot.generate_finetune", "audit",
                                "--dataset", data]
             + (["--out", p["audit-gallery"]] if p else []),
             note=("checks every box against the pixels" if p else
                   "checks every box against the pixels of the COMMITTED ladder. "
                   "The steps that would write are omitted: pass --out DIR to "
                   "build a new set (docs/runme/FINETUNE.md section 0)")),
    ]
    if p:
        verify.append(
            Step("gallery", ["-m", "blindspot.report_finetune", "gallery",
                             "--records", p["sft-records"], "--out", p["gallery"]],
                 note="every record shown twice -- full frame and a zoom -- because at "
                      "900x570 a 0.02% target is a few pixels"))

    return {"ladder": ladder, "build": build, "verify": verify, "report": report}


def _resolutions(p: str) -> set[Path]:
    """Every filesystem location the string `p` could name, resolved.

    A relative --out means one thing to the shell that typed it (its own cwd)
    and another to the steps, which `flow.main` always runs with cwd=ROOT.
    Checking one reading and not the other is exactly how a guard silently fails
    to fire, so both are checked.
    """
    q = Path(p)
    return {q.resolve(), (flow.ROOT / q).resolve()}




def as_step_would_see(out: str) -> Path:
    """Resolve --out the way the STEP will, not the way the caller's shell does.

    `flow.main` runs every step with `cwd=ROOT`, so a relative `--out` is
    relative to the repository, not to wherever the operator happened to be
    standing. Checking it against the caller's cwd means `cd /tmp && ... --out
    data/svg_localization` passes the guard and then writes the committed
    dataset, because the step re-interprets the same string against ROOT.
    """
    p = Path(out)
    return p if p.is_absolute() else flow.ROOT / p
def same_path(a: Path, b: Path) -> bool:
    """Is `a` the same location as `b`, whatever the filesystem thinks?

    `Path.resolve()` does not case-fold, and macOS/APFS is case-INSENSITIVE by
    default. So `outputs/PART3`.resolve() != `outputs/part3`.resolve() as strings
    while `stat()` reports the same inode -- the two paths ARE one directory. A
    guard built on string equality therefore waves through
    `--out outputs/PART3` and writes over the very artifact it exists to protect.

    Ask the filesystem when it can answer (samefile compares device+inode), and
    fall back to a case-normalised comparison when the path does not exist yet,
    which is the common case for an --out that has never been created.
    """
    try:
        return a.samefile(b)
    except (OSError, ValueError):
        return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))
def within_path(child: Path, parent: Path) -> bool:
    """Is `child` `parent`, or anywhere INSIDE it?

    `same_path` answers identity, and identity was the whole guard. It says
    nothing about containment, so `--out data/svg_localization/subdir` passed
    every check and the printed plan claimed "the committed set is untouched"
    while scheduling writes inside the committed set. Nothing gets clobbered
    that way -- a subdirectory cannot overwrite its siblings -- but it drops
    untracked files into a git-tracked dataset that every published number is
    read from, which is the state the guard exists to prevent.

    A guard whose job is keeping writes out of a directory has to test the
    directory, not just its name.
    """
    if same_path(child, parent):
        return True
    # `os.path.normcase` is a NO-OP on POSIX -- it only folds case on Windows.
    # So a normcase string compare cannot see that DATA/SVG_LOCALIZATION/sub is
    # inside data/svg_localization on a case-insensitive APFS volume, which is
    # exactly the bypass this function was added to close. Walk up to the
    # nearest ancestor that exists and ask the filesystem, which does know;
    # fall back to an explicit casefold for the part that does not exist yet.
    here = child.resolve()
    for ancestor in here.parents:
        if same_path(ancestor, parent):
            return True
    return str(here).casefold().startswith(str(parent.resolve()).casefold() + os.sep)
def refuse_finetune_out(out: str) -> None:
    """Reject an --out that would land a finetune step on a reference artifact.

    The guard is on the DERIVED paths, not just the base, because the base is
    never the thing overwritten: `--out outputs/part3` writes
    `outputs/part3/assets`, and `--out data` writes `data/svgloc_mr`.
    """
    targets = {"--out itself": str(as_step_would_see(out))}
    targets.update({f"the {k} it would write": v for k, v in finetune_out(out).items()})
    for label, target in targets.items():
        hit = _resolutions(target)
        for ref, why in FINETUNE_REFERENCE.items():
            # within_path, not same_path: `--out data/svgloc_mr/subdir` lands
            # nothing ON a reference artifact and so passed an identity check,
            # while still writing inside one of the directories this refuses to
            # touch. Containment is the property being protected.
            if any(within_path(h, flow.ROOT / ref) for h in hit):
                raise SystemExit(
                    f"refusing --out {out}: {label} lands on, or inside,\n\n"
                    f"    {ref}\n\n"
                    + textwrap.fill(
                        f"which is {why}. It is gitignored and untracked, so there "
                        f"is no copy to restore, and rerunning this pipeline does "
                        f"not reproduce it. Point --out at a scratch directory "
                        f"instead, e.g. --out /tmp/finetune_dev.", 78)
                    + "\nSee docs/runme/FINETUNE.md section 0.")


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
        i = argv.index("--out")
        if i + 1 >= len(argv):
            raise SystemExit("--out needs a directory")
        opts["out"] = argv[i + 1]; del argv[i:i + 2]
        # Resolve BOTH against the repository root, not the cwd. Comparing
        # `Path(COMMITTED).resolve()` resolves the guard's own reference against
        # wherever the process happens to be, so from any other directory an
        # absolute --out pointing straight at the committed dataset compares
        # unequal and the guard silently does not fire.
        if within_path(as_step_would_see(opts["out"]), flow.ROOT / COMMITTED):
            raise SystemExit(
                f"refusing --out {opts['out']}: that is {COMMITTED}, or a path inside it.\n"
                f"That is the committed dataset and the source of truth for every published\n"
                f"number. The generator has drifted from it, so regenerating in place would\n"
                f"rebind uids to different questions -- and writing to a SUBDIRECTORY of it\n"
                f"leaves untracked files in a git-tracked dataset even though it clobbers\n"
                f"nothing. Build into a directory outside it.\n"
                f"See docs/runme/SYNTHETIC.md section 0.")
        if name == "finetune_data":
            refuse_finetune_out(opts["out"])

    sys.argv = [f"blindspot.pipelines {name}", *argv]
    stages = {k: v for k, v in PIPELINES[name][0](opts).items() if v}
    raise SystemExit(main(name, stages, __doc__))
