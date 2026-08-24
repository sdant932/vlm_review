"""The three pipelines and the launcher that runs them.

`blindspot/pipelines.py` is pure aggregation: it owns no logic, only argument
lists. That is exactly what makes it fragile. Nothing in Python checks that
`["-m", "blindspot.eval", "aggregate"]` still names a module that exists, still
names a subcommand that module still accepts, or still declares that it spends
money. A rename three files away leaves the pipeline syntactically perfect and
functionally dead, and you find out ten minutes into a run -- or, worse, after
the bill.

So this file tests the seams, not the arithmetic:

    1. REGISTRY        -- all three pipelines build, and build lists of Steps
    2. RESOLVABLE ARGV -- every step's argv is accepted by its module's parser
    3. SPEND GATING    -- the only thing standing between a typo and real money
    4. TASK PRUNING    -- --task really removes work, and fails loudly if wrong
    5. COMMITTED DATA  -- no stage can overwrite a shipped or reference artifact
    6. STAGE SELECTION -- --list / --stage / --from pick the right subsets
    7. OPTIONAL STEPS  -- a step marked optional must not abort the run

Nothing here calls the API, downloads anything, or writes into the repository.
Group 3 drives `flow.main` with `subprocess.call` replaced by a recorder, so the
plan is inspected as data rather than parsed out of a log. Groups 4-6 shell out,
because the options they cover are parsed in `__main__` and cannot be imported;
they only ever pass `--list`, which returns before a single step is executed.

Three failures found while writing these, all since fixed, are the reason the
groups exist in this shape: every `blindspot.eval` / `blindspot.report` step was
missing its subcommand after those modules were merged (group 2); the
`worked-examples` step reached the API with no declared spend, so `--max-spend`
could not reach it (group 3 now pins that the ceiling both is appended and
parses); and `finetune_data --all` -- the command its own runbook gives --
rebuilt every reference artifact of Part 3 in place, none of which is in git and
none of which regenerates (group 5).
"""

import argparse
import importlib
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from blindspot import flow, pipelines
from blindspot.flow import Step

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_NAMES = ["literature_eval", "synth_localization_eval", "finetune_data"]

# A scratch --out that is emphatically not the committed dataset. Never created:
# every test that uses it stops at --list or at the plan.
SCRATCH_OUT = "/tmp/blindspot_test_out"


# =============================================================================
# helpers
# =============================================================================

def _stages(name, tasks=None, out=None):
    """Build one pipeline's {stage: [Step]} map."""
    build_fn = pipelines.PIPELINES[name][0]
    return build_fn({"tasks": list(tasks or pipelines.TASKS), "out": out})


def _steps(name, **kw):
    """[(stage, Step)] for one pipeline, flattened in order."""
    return [(stage, s) for stage, steps in _stages(name, **kw).items() for s in steps]


def _every_step():
    """Every step of every pipeline, with the generate stage populated.

    `out` is set so the opt-in generate steps are covered too -- they are the
    steps most likely to rot, since they are the ones nobody runs.
    """
    out = [(f"{name}/{stage}", s)
           for name in PIPELINE_NAMES
           for stage, s in _steps(name, out=SCRATCH_OUT)]
    assert out, "no steps at all -- has the registry moved?"
    return out


class _ParserCaptured(Exception):
    pass


_PARSER_CACHE: dict[str, tuple] = {}


def _module_and_parser(modname):
    """Import a module and get its fully-built ArgumentParser, running nothing.

    Every CLI here builds its parser inside `main()`, so the parser is not
    reachable as an attribute. Rather than shelling out to `--help` once per
    step, hijack `parse_args` to hand the parser back before it parses anything:
    the parser is then a real object we can ask about subcommands and validate
    argv against, in-process and without side effects.
    """
    if modname not in _PARSER_CACHE:
        mod = importlib.import_module(modname)
        box = {}
        real = argparse.ArgumentParser.parse_args

        def spy(self, args=None, namespace=None):
            box["parser"] = self
            raise _ParserCaptured

        argparse.ArgumentParser.parse_args = spy
        argv = sys.argv
        sys.argv = [modname]
        try:
            try:
                mod.main()
            except _ParserCaptured:
                pass
        finally:
            argparse.ArgumentParser.parse_args = real
            sys.argv = argv
        assert "parser" in box, (
            f"{modname}.main() did not reach parse_args -- it does work before "
            f"parsing its arguments, which makes it unsafe to introspect and "
            f"unsafe to --help")
        _PARSER_CACHE[modname] = (mod, box["parser"])
    return _PARSER_CACHE[modname]


def _subcommands(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


def _named_subcommand(step):
    """The subcommand word in a step's argv, if it names one."""
    tail = step.argv[2:]
    return tail[0] if tail and not tail[0].startswith("-") else None


def _handler_for(step):
    """The function a step's argv would ultimately call."""
    mod, parser = _module_and_parser(step.argv[1])
    subs = _subcommands(parser)
    sub = _named_subcommand(step)
    if subs and sub in subs:
        return next((v for v in subs[sub]._defaults.values() if callable(v)), None)
    return mod.main


def _reaches_anthropic(fn, seen=None):
    """Does calling `fn` reach the `anthropic` module, through blindspot code?

    A module-level `import anthropic` is too coarse a signal now that the CLIs
    are merged: `blindspot.eval` imports it for one subcommand out of six, and
    grading `aggregate` as an API step would make --max-spend look mandatory
    when it is not. So walk the actual call graph from the subcommand's handler,
    following globals that are blindspot functions.
    """
    seen = set() if seen is None else seen
    if fn is None or fn in seen:
        return False
    seen.add(fn)
    code = getattr(fn, "__code__", None)
    if code is None:
        return False
    names, stack = set(), [code]
    while stack:                                   # nested defs / comprehensions
        c = stack.pop()
        names |= set(c.co_names)
        stack += [k for k in c.co_consts if isinstance(k, types.CodeType)]
    if "anthropic" in names:
        return True
    for n in names:
        v = fn.__globals__.get(n)
        if (isinstance(v, types.FunctionType)
                and str(v.__module__).startswith("blindspot")
                and _reaches_anthropic(v, seen)):
            return True
    return False


def _run_flow(stages, argv, monkeypatch, *, key=True, real=False):
    """Drive flow.main and return (exit code, [argv of each command it ran]).

    `subprocess.call` is replaced by a recorder, so the launcher runs to
    completion and the commands it *would* have issued are available as data --
    including the --max-spend share appended to each one, which is the thing
    worth asserting on. `real=True` lets the commands actually run, for the
    exit-code handling in group 7.
    """
    ran = []
    monkeypatch.setattr(sys, "argv", ["pipelines-under-test", *argv])
    # The repo ships a .env, so the key check must never depend on the machine.
    monkeypatch.setattr(flow, "_have_api_key", lambda: key)
    if not real:
        def recorder(cmd, **kw):
            ran.append(list(cmd))
            return 0
        monkeypatch.setattr(flow.subprocess, "call", recorder)
    return flow.main("testflow", stages), ran


def _cli(*args, cwd="/tmp"):
    """Run the pipelines CLI. Only ever called with --list, which runs nothing.

    The option parsing under test (--task, --out) lives in `if __name__ ==
    "__main__"`, which is unreachable by import, so these few tests shell out.
    """
    return subprocess.run([sys.executable, "-m", "blindspot.pipelines", *args],
                          capture_output=True, text=True, cwd=cwd)


# a step line is `    $ name<padding>argv`; the argv half always starts with the
# interpreter flag, which is what tells a step line apart from a note line.
_PLAN_STEP = re.compile(r"^ {4}([$ ]) (\S+) +(-[mc] .*)$")
_PLAN_STAGE = re.compile(r"^ {2}\[([^\]]+)\]")


def _parse_plan(stdout):
    """(stage names, {step name: argv string}) out of a --list plan."""
    stages, steps = [], {}
    for line in stdout.splitlines():
        m = _PLAN_STAGE.match(line)
        if m:
            stages.append(m.group(1))
            continue
        m = _PLAN_STEP.match(line)
        if m:
            steps[m.group(2)] = m.group(3)
    return stages, steps


# =============================================================================
# 1. REGISTRY
# =============================================================================
"""The registry is the entry point for all three efforts; if it does not build,
nothing else in this file has anything to test."""


def test_all_three_pipelines_are_registered():
    assert list(pipelines.PIPELINES) == PIPELINE_NAMES


def test_every_pipeline_is_a_build_function_and_a_description():
    for name, entry in pipelines.PIPELINES.items():
        build_fn, desc = entry
        assert callable(build_fn), f"{name}: not callable"
        assert desc and isinstance(desc, str), f"{name}: no description"


@pytest.mark.parametrize("name", PIPELINE_NAMES)
def test_every_pipeline_builds_named_stages_of_steps(name):
    stages = _stages(name, out=SCRATCH_OUT)
    assert isinstance(stages, dict) and stages, f"{name}: built no stages"
    for stage, steps in stages.items():
        assert stage and stage.strip(), f"{name}: a stage has no name"
        assert isinstance(steps, list), f"{name}/{stage}: not a list"
        assert all(isinstance(s, Step) for s in steps), f"{name}/{stage}: not all Steps"
    assert any(stages.values()), f"{name}: every stage is empty"


@pytest.mark.parametrize("name", PIPELINE_NAMES)
def test_step_names_are_unique_within_a_stage(name):
    """--from/--stage address stages; the log addresses steps by name."""
    for stage, steps in _stages(name, out=SCRATCH_OUT).items():
        names = [s.name for s in steps]
        assert all(names), f"{name}/{stage}: a step has no name"
        assert len(names) == len(set(names)), f"{name}/{stage}: duplicate step names {names}"


# =============================================================================
# 2. RESOLVABLE ARGV
# =============================================================================
"""Every step's argv must still name something runnable.

This is the test that catches a rename. A step is a string list; renaming a
module or a subcommand leaves it untouched and still perfectly valid Python. The
check is not "does the file exist" -- it is "would argparse accept exactly these
arguments", which also covers a dropped flag and an out-of-range --datasets
choice. Parsing stops before any handler runs, so nothing here executes.
"""


def test_every_step_invokes_a_module_not_a_path():
    """`-m blindspot.x` survives a cwd change; `scripts/x.py` did not."""
    for label, s in _every_step():
        assert s.argv[:1] == ["-m"], f"{label}/{s.name}: not a -m invocation: {s.argv}"
        assert s.argv[1].startswith("blindspot."), f"{label}/{s.name}: {s.argv[1]}"


def test_every_step_targets_a_module_file_that_exists():
    missing = [f"{label}/{s.name} -> {s.argv[1]}"
               for label, s in _every_step()
               if not (ROOT / (s.argv[1].replace(".", "/") + ".py")).is_file()]
    assert not missing, "pipeline steps name modules that do not exist:\n  " + "\n  ".join(missing)


def test_every_step_names_a_subcommand_its_module_still_accepts():
    bad = []
    for label, s in _every_step():
        _, parser = _module_and_parser(s.argv[1])
        subs = _subcommands(parser)
        sub = _named_subcommand(s)
        if not subs:
            assert sub is None, f"{label}/{s.name}: {s.argv[1]} takes no subcommands"
            continue
        if sub is None:
            bad.append(f"{label}/{s.name}: {s.argv[1]} requires a subcommand, none given")
        elif sub not in subs:
            bad.append(f"{label}/{s.name}: {s.argv[1]} has no subcommand {sub!r} "
                       f"(has {sorted(subs)})")
    assert not bad, "pipeline steps name subcommands that no longer exist:\n  " + "\n  ".join(bad)


def test_every_step_argv_parses(capsys):
    """The full check: argparse accepts the whole argument list, flags included."""
    bad = []
    for label, s in _every_step():
        _, parser = _module_and_parser(s.argv[1])
        try:
            parser.parse_args(s.argv[2:])
        except SystemExit:
            err = capsys.readouterr().err.strip().splitlines()
            bad.append(f"{label}/{s.name}: {' '.join(s.argv)}\n      -> "
                       f"{err[-1] if err else 'rejected'}")
        except Exception as e:                       # a type= callable blew up
            bad.append(f"{label}/{s.name}: {type(e).__name__}: {e}")
    capsys.readouterr()
    assert not bad, "pipeline steps their own module would reject:\n  " + "\n  ".join(bad)


# =============================================================================
# 3. SPEND GATING
# =============================================================================
"""The only thing between a typo and a real bill.

`flow` gates three things on `Step.needs_api`: the key precheck, the --offline
skip, and the refusal to start without --max-spend. A step that reaches the API
without the flag defeats all three at once. That is not hypothetical -- the
`worked-examples` step shipped without it and a test run spent real money
discovering that the runbook was wrong about the flow making no API calls.

`needs_api` is therefore checked against what the step's *subcommand* actually
calls, not against what its module happens to import, and the ceiling arithmetic
is checked by reading the commands the launcher issues.
"""


def test_steps_that_reach_the_api_declare_it():
    undeclared = [f"{label}/{s.name} -> {' '.join(s.argv)}"
                  for label, s in _every_step()
                  if not s.needs_api and _reaches_anthropic(_handler_for(s))]
    assert not undeclared, (
        "step(s) reach the API without needs_api=True, so they run uncapped, "
        "survive --offline and never warn:\n  " + "\n  ".join(undeclared))


def test_steps_that_declare_the_api_really_reach_it():
    """The inverse: a false needs_api makes --max-spend look mandatory when it is not."""
    overdeclared = [f"{label}/{s.name} -> {' '.join(s.argv)}"
                    for label, s in _every_step()
                    if s.needs_api and not _reaches_anthropic(_handler_for(s))]
    assert not overdeclared, (
        "needs_api=True but nothing on the call path touches anthropic:\n  "
        + "\n  ".join(overdeclared))


def test_api_steps_without_a_declared_spend_stay_optional_and_documented():
    """--max-spend can only reach a step that declares a share of it.

    `Step.rendered` appends the ceiling only when `spend` is set, so an API step
    with no `spend` runs uncapped no matter what the operator passes. There is no
    such step left -- `worked-examples` was the one, and it now owns a
    `--max-spend` of its own -- so this holds vacuously today. It stays because
    the next one must not be added silently.
    """
    for label, s in _every_step():
        if s.needs_api and s.spend is None:
            assert s.rendered(5.0) == [sys.executable, *s.argv], (
                f"{label}/{s.name}: declares no spend yet got a ceiling")
            assert s.optional and s.note, (
                f"{label}/{s.name} reaches the API with no declared spend, so "
                f"--max-spend cannot cap it. Give it a spend, or mark it "
                f"optional=True and say so in its note.")


def test_the_worked_examples_step_declares_a_spend_and_can_be_capped():
    """The B7 seam, from the launcher's side.

    `blindspot.report_worked` is `--prompts x --samples` calls with no natural
    end -- 100 x 100 is ten thousand of them. It used to declare no `spend`, and
    `Step.rendered` appends `--max-spend` only to a step that declares one, so
    the operator's ceiling could not reach it even when they passed one.
    """
    step = next(s for _, s in _steps("finetune_data", out=SCRATCH_OUT)
                if s.name == "worked-examples")
    assert step.needs_api, "the step that samples the model must say so"
    assert step.spend, "no declared spend means --max-spend is never appended"
    assert "--max-spend" in step.rendered(0.25), step.rendered(0.25)


def test_the_ceiling_the_launcher_appends_is_one_report_worked_accepts():
    """Appending a flag the callee rejects is the same as having no cap.

    Asserted against the module's real parser, so renaming the flag on either
    side fails here instead of at the end of a paid run.
    """
    step = next(s for _, s in _steps("finetune_data", out=SCRATCH_OUT)
                if s.name == "worked-examples")
    _, parser = _module_and_parser("blindspot.report_worked")
    args = parser.parse_args(step.rendered(0.25)[3:])     # [python, -m, module, ...]
    assert args.max_spend == 0.25


def test_flow_refuses_to_start_api_steps_without_max_spend(monkeypatch, capsys):
    rc, ran = _run_flow(_gated_stages(), ["--all"], monkeypatch)
    assert rc != 0, "API steps started with no ceiling at all"
    assert not ran, f"commands ran despite the refusal: {ran}"
    assert "--max-spend" in capsys.readouterr().err


def test_flow_refuses_to_start_api_steps_without_a_key(monkeypatch, capsys):
    rc, ran = _run_flow(_gated_stages(), ["--all", "--max-spend", "4"],
                        monkeypatch, key=False)
    assert rc != 0 and not ran
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_flow_runs_everything_once_a_ceiling_is_given(monkeypatch):
    rc, ran = _run_flow(_gated_stages(), ["--all", "--max-spend", "4"], monkeypatch)
    assert rc == 0
    assert len(ran) == 4, ran


def test_offline_skips_api_steps_and_only_api_steps(monkeypatch):
    stages = _gated_stages()
    rc, ran = _run_flow(stages, ["--all", "--offline"], monkeypatch, key=False)
    assert rc == 0, "an offline run should not need a key or a ceiling"
    ran_names = {_recorded_name(cmd) for cmd in ran}
    expected = {s.name for st in stages.values() for s in st if not s.needs_api}
    assert ran_names == expected, f"offline ran {ran_names}, expected {expected}"


def test_max_spend_is_split_across_api_steps_that_declare_a_share(monkeypatch):
    rc, ran = _run_flow(_gated_stages(), ["--all", "--max-spend", "4"], monkeypatch)
    assert rc == 0
    caps = {_recorded_name(cmd):
            (cmd[cmd.index("--max-spend") + 1] if "--max-spend" in cmd else None)
            for cmd in ran}
    # two API steps share the $4: $2 each, and the declared $8 is clipped to it
    assert caps["api-big"] == "2", caps
    # an API step with no declared spend gets no ceiling -- it has nowhere to put it
    assert caps["api-uncapped"] is None, caps
    # a non-API step never gets one, even though it declares a spend
    assert caps["local-with-spend"] is None, caps
    assert caps["local"] is None, caps


def test_a_step_is_never_given_more_than_it_declared(monkeypatch):
    """--max-spend 100 across two steps must not hand $50 to a step that asked for $8."""
    _, ran = _run_flow(_gated_stages(), ["--all", "--max-spend", "100"], monkeypatch)
    cmd = next(c for c in ran if _recorded_name(c) == "api-big")
    assert cmd[cmd.index("--max-spend") + 1] == "8", cmd


def _recorded_name(cmd):
    """The step name out of a recorded command: [python, "-c", "pass", <name>, ...]."""
    return cmd[3]


def _gated_stages():
    """A fabricated flow covering every combination the gate has to handle.

    Fabricated rather than real so the spend arithmetic is asserted on numbers
    chosen for the test; the real pipelines are checked by the two declaration
    tests above. `-c pass` never runs -- subprocess.call is a recorder -- and the
    step name is carried in argv so the recorded commands are identifiable.
    """
    return {
        "prep": [Step("local", ["-c", "pass", "local"])],
        "run": [Step("api-big", ["-c", "pass", "api-big"], needs_api=True, spend=8.0),
                Step("api-uncapped", ["-c", "pass", "api-uncapped"], needs_api=True)],
        "wrap": [Step("local-with-spend", ["-c", "pass", "local-with-spend"], spend=3.0)],
    }


# =============================================================================
# 4. TASK PRUNING
# =============================================================================
"""--task must remove work, not just relabel it.

The three synthetic question types share one set of scenes, so it is easy for a
"counting only" run to quietly drag the localization arms along with it -- they
are the expensive ones. The pruning happens in `__main__`, before the generic
parser sees the arguments, so these tests go through the CLI.
"""


def test_task_counting_prunes_the_localization_arms():
    r = _cli("synth_localization_eval", "--task", "counting", "--list")
    assert r.returncode == 0, r.stderr
    _, steps = _parse_plan(r.stdout)
    assert steps, r.stdout
    localization = [n for n in steps if n.startswith("localization")]
    assert not localization, f"--task counting still runs {localization}"
    assert "derived:counting" in steps, steps
    assert "svg_counting" in steps["derived:counting"]


def test_no_task_selection_runs_all_three_tasks():
    r = _cli("synth_localization_eval", "--list")
    assert r.returncode == 0, r.stderr
    _, steps = _parse_plan(r.stdout)
    joined = " ".join(steps.values())
    for dataset in [t["dataset"] for t in pipelines.TASKS.values()]:
        assert dataset in joined, f"{dataset} missing from the default plan:\n{r.stdout}"


def test_an_unknown_task_fails_loudly():
    r = _cli("synth_localization_eval", "--task", "counting", "typo", "--list")
    assert r.returncode != 0, "an unknown task was silently ignored"
    assert "typo" in (r.stderr + r.stdout)


def test_an_unknown_pipeline_fails_loudly():
    r = _cli("no_such_pipeline", "--list")
    assert r.returncode != 0
    assert "unknown pipeline" in (r.stderr + r.stdout)


# =============================================================================
# 5. THE COMMITTED-DATASET GUARD
# =============================================================================
"""data/svg_localization is the source of truth for every published number.

The generator has drifted from it, so regenerating in place would rebind uids to
different questions and silently invalidate the results already in results/.
The guard is two-sided: generation is opt-in (no generate stage at all unless
--out names somewhere else), and --out is refused outright if it resolves to the
committed set.
"""


def test_generation_is_opt_in():
    stages = _stages("synth_localization_eval")
    assert stages["generate"] == [], (
        "the generate stage is present without --out; a plain --all would "
        "regenerate over the committed dataset")


def test_generation_appears_only_where_out_points():
    stages = _stages("synth_localization_eval", out=SCRATCH_OUT)
    gen = stages["generate"]
    assert gen, "--out was given and still nothing generates"
    for s in gen:
        assert SCRATCH_OUT in s.argv, f"generate step {s.name} ignores --out: {s.argv}"
        assert pipelines.COMMITTED not in " ".join(s.argv), (
            f"generate step {s.name} writes to the committed dataset: {s.argv}")


def test_audit_reads_the_committed_set_when_no_out_is_given():
    """The other half of opt-in: without --out the flow still audits, in place."""
    audit = _stages("synth_localization_eval")["audit"]
    assert audit
    assert all(pipelines.COMMITTED in s.argv for s in audit), [s.argv for s in audit]


@pytest.mark.parametrize("out", [pipelines.COMMITTED, str(ROOT / pipelines.COMMITTED)])
def test_the_cli_refuses_to_generate_into_the_committed_dataset(out):
    # run from ROOT: the guard compares resolved paths, so the relative form only
    # means the committed set when the cwd is the repository.
    r = _cli("synth_localization_eval", "--out", out, "--list", cwd=str(ROOT))
    assert r.returncode != 0, f"--out {out} was accepted:\n{r.stdout}"
    assert "refusing" in (r.stderr + r.stdout)


@pytest.mark.parametrize("suffix", [
    "subdir",                 # the plain case: one level inside
    "a/b/c",                  # several levels, none of which exist
    "sub/",                   # trailing separator
])
def test_the_guard_refuses_a_path_INSIDE_the_committed_dataset(suffix):
    """Identity is not containment, and the guard used to test only identity.

    `--out data/svg_localization/subdir` is not the committed dataset, so
    `samefile` said no and the printed plan claimed "the committed set is
    untouched" while scheduling writes inside it. Nothing gets clobbered -- a
    subdirectory cannot overwrite its siblings -- but it leaves untracked files
    in the git-tracked dataset every published number is read from.
    """
    out = f"{pipelines.COMMITTED}/{suffix}"
    r = _cli("synth_localization_eval", "--out", out, "--list", cwd=str(ROOT))
    assert r.returncode != 0, f"--out {out} was accepted:\n{r.stdout}"
    assert "refusing" in (r.stderr + r.stdout)


def test_the_containment_guard_folds_case_because_normcase_does_not():
    """`os.path.normcase` is a no-op on POSIX -- it folds case on Windows only.

    So the string half of the guard could not see that DATA/SVG_LOCALIZATION/sub
    is inside data/svg_localization on a case-insensitive APFS volume, which is
    the bypass that survived the first containment fix.
    """
    out = f"{pipelines.COMMITTED.upper()}/sub"
    r = _cli("synth_localization_eval", "--out", out, "--list", cwd=str(ROOT))
    assert r.returncode != 0, f"--out {out} was accepted:\n{r.stdout}"


@pytest.mark.parametrize("out", [
    "data/svg_localization_other",   # shares the prefix, is NOT inside
    "data/svgloc2",
])
def test_the_containment_guard_does_not_over_reach(out, tmp_path):
    """A sibling that merely shares the prefix must still be allowed.

    A `startswith` without the separator would swallow
    data/svg_localization_other, which is a different directory.
    """
    r = _cli("synth_localization_eval", "--out", out, "--list", cwd=str(ROOT))
    assert r.returncode == 0, f"--out {out} was wrongly refused:\n{r.stderr}"


# ----------------------------------------------- the same rule, for finetune_data
"""`finetune_data` writes six artifacts that are not in git and do not come back.

`python -m blindspot.pipelines finetune_data --all` -- the command printed in
docs/runme/FINETUNE.md -- used to rebuild all of them in place: the ladder, the
SFT records, the gallery, the worked examples, the part3 assets and part3.html.
Every one is gitignored, and regeneration is not restoration: a fresh `ladder`
emits the same uids with different boxes, `samples --seed 0` reproduces a
quarter of the shipped records, and `report_worked` samples the model.

So the same two-sided rule as above: writing is opt-in behind --out, and an
--out that would land a step on one of those artifacts is refused by name.
"""

# A writing flag; anything else in an argv is an input the step only reads.
WRITE_FLAGS = ("--out", "--out-dir")


def _write_targets(step):
    return [v for f, v in zip(step.argv, step.argv[1:]) if f in WRITE_FLAGS]


def _on_a_reference_artifact(path):
    return next((ref for ref in pipelines.FINETUNE_REFERENCE
                 if (ROOT / ref).resolve() in pipelines._resolutions(path)), None)


def test_finetune_writing_steps_are_opt_in():
    """Without --out, nothing that writes a reference artifact is scheduled."""
    stages = _stages("finetune_data")
    hits = [(stage, s.name, t, ref)
            for stage, steps in stages.items() for s in steps
            for t in _write_targets(s) if (ref := _on_a_reference_artifact(t))]
    assert not hits, (
        "a plain --all would overwrite unrecoverable artifacts:\n  "
        + "\n  ".join(f"{stage}/{name} writes {t} -> {ref}" for stage, name, t, ref in hits))


def test_the_bare_pipeline_still_audits_the_committed_ladder():
    """The other half of opt-in: read-only work is not thrown away with it."""
    verify = _stages("finetune_data")["verify"]
    assert [s.name for s in verify] == ["multires-audit"], [s.name for s in verify]
    assert pipelines.MR_DATA in verify[0].argv, verify[0].argv
    assert "--out" in verify[0].note or "FINETUNE.md" in verify[0].note, (
        "the bare plan must say how to get the omitted steps back")


def test_every_finetune_step_writes_under_out_when_one_is_given():
    stages = _stages("finetune_data", out=SCRATCH_OUT)
    assert all(stages[k] for k in ("ladder", "build", "verify", "report")), (
        "--out was given and some stage still has nothing to do")
    for stage, steps in stages.items():
        for s in steps:
            targets = _write_targets(s)
            assert targets, f"{stage}/{s.name} writes nowhere: {s.argv}"
            for t in targets:
                assert t.startswith(SCRATCH_OUT), (
                    f"{stage}/{s.name} ignores --out and writes {t}")


@pytest.mark.parametrize("ref", sorted(pipelines.FINETUNE_REFERENCE))
def test_the_guard_refuses_an_out_that_lands_on_any_reference_artifact(ref):
    """Both readings: --out AT the artifact, and an --out that derives it.

    Derived by inverting `finetune_out`, so an artifact that the steps can no
    longer produce -- or a step whose destination moved out from under an entry
    here -- fails this rather than leaving an unenforceable line in the table.
    """
    with pytest.raises(SystemExit) as e:
        pipelines.refuse_finetune_out(ref)
    assert ref in str(e.value), f"the refusal must name {ref}: {e.value}"
    # the reason is wrapped to the terminal, so compare it unwrapped
    said = " ".join(str(e.value).split())
    assert " ".join(pipelines.FINETUNE_REFERENCE[ref].split()) in said, "and say why"

    layout = pipelines.finetune_out("BASE")
    bases = [ref[: -len(suffix)] for suffix in
             (t[len("BASE"):] for t in layout.values()) if ref.endswith(suffix)]
    assert bases, f"no --out produces {ref} any more -- the guard cannot fire"
    for base in bases:
        with pytest.raises(SystemExit):
            pipelines.refuse_finetune_out(base)


def test_a_scratch_out_is_accepted():
    pipelines.refuse_finetune_out(SCRATCH_OUT)          # must not raise


@pytest.mark.parametrize("out", ["data", "outputs/finetune", "outputs/part3",
                                 pipelines.MR_DATA,
                                 str(ROOT / "outputs/part3")])
def test_the_cli_refuses_a_destructive_finetune_out(out):
    # from ROOT, since a relative --out only names a repository path from there
    r = _cli("finetune_data", "--out", out, "--list", cwd=str(ROOT))
    assert r.returncode != 0, f"--out {out} was accepted:\n{r.stdout}"
    assert "refusing" in (r.stderr + r.stdout)
    assert "FINETUNE.md" in (r.stderr + r.stdout), "the refusal must say where to read"


def test_the_bare_cli_plan_schedules_no_writing_step():
    """End to end, through the argument parsing that only `__main__` does."""
    r = _cli("finetune_data", "--list", cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    _, steps = _parse_plan(r.stdout)
    assert list(steps) == ["multires-audit"], steps
    for ref in pipelines.FINETUNE_REFERENCE:
        assert f"--out {ref}" not in r.stdout and f"--out-dir {ref}" not in r.stdout


# ------------------------------------------ and for the generator underneath it
"""`generate scenes --out` used to default to the committed dataset.

The pipeline refused that path and the Makefile refused it, but the callee they
both wrap still pointed at it, so typing the documented command by hand -- which
the Makefile header invites -- went straight past both guards. The default is
gone; the flag is required.
"""


def _tree(d: Path):
    return {str(p.relative_to(d)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in d.rglob("*") if p.is_file()}


def test_scenes_out_has_no_default():
    _, parser = _module_and_parser("blindspot.generate")
    out = next(a for a in _subcommands(parser)["scenes"]._actions
               if "--out" in a.option_strings)
    assert out.default is None, (
        f"scenes --out defaults to {out.default!r}; a bare run would write there")


def test_a_bare_scenes_run_exits_non_zero_and_writes_nothing():
    committed = ROOT / pipelines.COMMITTED
    before = _tree(committed)
    r = subprocess.run([sys.executable, "-m", "blindspot.generate", "scenes",
                        "--count", "1"], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode != 0, "a bare `scenes` ran and produced a dataset somewhere"
    assert "--out" in (r.stdout + r.stderr)
    assert _tree(committed) == before, "the committed dataset changed"


def test_list_types_still_works_without_an_out():
    """The one thing `scenes` does that writes nothing keeps working bare."""
    r = subprocess.run([sys.executable, "-m", "blindspot.generate", "scenes",
                        "--list-types"], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    assert len(r.stdout.split()) > 5, r.stdout


# =============================================================================
# 6. STAGE SELECTION
# =============================================================================
"""--list, --stage and --from are how a half-finished run is resumed.

Getting the subset wrong re-runs an API stage that already succeeded, which
costs money, or skips one that did not, which corrupts the numbers.
"""


@pytest.mark.parametrize("name", PIPELINE_NAMES)
def test_list_exits_zero_and_names_every_stage(name):
    r = _cli(name, "--list")
    assert r.returncode == 0, r.stderr
    listed, _ = _parse_plan(r.stdout)
    expected = [k for k, v in _stages(name).items() if v]
    assert listed == expected, f"{name}: plan lists {listed}, expected {expected}"


def test_a_pipeline_with_no_selector_errors():
    r = _cli("literature_eval")
    assert r.returncode != 0, "a bare pipeline ran something"
    assert "--all" in r.stderr


def test_stage_runs_only_that_stage(monkeypatch):
    _, ran = _run_flow(_gated_stages(), ["--stage", "prep"], monkeypatch)
    assert [_recorded_name(c) for c in ran] == ["local"]


def test_from_runs_that_stage_and_everything_after(monkeypatch):
    _, ran = _run_flow(_gated_stages(), ["--from", "run", "--max-spend", "4"], monkeypatch)
    assert [_recorded_name(c) for c in ran] == ["api-big", "api-uncapped",
                                                "local-with-spend"]


def test_list_runs_nothing(monkeypatch, capsys):
    rc, ran = _run_flow(_gated_stages(), ["--list"], monkeypatch, key=False)
    assert rc == 0 and not ran
    out = capsys.readouterr().out
    assert set(_parse_plan(out)[0]) == set(_gated_stages())


# =============================================================================
# 7. OPTIONAL STEPS
# =============================================================================
"""`optional=True` means "this one is allowed to fail".

Several steps depend on artifacts that only exist on a full run -- a gallery
needs images, a paste-page needs prose that lives outside the repo. Marking them
optional is what makes `--all` usable on a fresh clone; the risk is the marking
silently stops working and one missing font aborts a forty-minute run.

These run a real subprocess -- `python -c` that exits non-zero -- because the
thing under test is the exit-code handling, and a fabricated flow is used so no
real pipeline has to contain a deliberately failing step.
"""


def _failing_stages(optional):
    boom = ["-c", "raise SystemExit(3)"]
    return {
        "one": [Step("boom", boom, optional=optional), Step("after-boom", ["-c", "pass"])],
        "two": [Step("later", ["-c", "pass"])],
    }


def test_an_optional_failure_does_not_abort_the_run(monkeypatch, capsys):
    rc, _ = _run_flow(_failing_stages(True), ["--all"], monkeypatch, real=True)
    assert rc == 0, "an optional failure aborted the run"
    out = capsys.readouterr().out
    assert "optional step failed" in out
    assert "later" in out, "the run stopped before the next stage"


def test_a_required_failure_aborts_with_the_step_s_exit_code(monkeypatch, capsys):
    rc, _ = _run_flow(_failing_stages(False), ["--all"], monkeypatch, real=True)
    assert rc == 3, "a failing required step did not abort with its own exit code"
    assert "later" not in capsys.readouterr().out, "the run continued past a hard failure"


def test_continue_on_error_finishes_and_reports_the_failure(monkeypatch, capsys):
    rc, _ = _run_flow(_failing_stages(False), ["--all", "--continue-on-error"],
                      monkeypatch, real=True)
    assert rc == 1, "--continue-on-error must still exit non-zero"
    out = capsys.readouterr().out
    assert "later" in out, "--continue-on-error stopped early"
    assert "one/boom (exit 3)" in out, "the failure was not reported at the end"
