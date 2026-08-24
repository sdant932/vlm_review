"""Generic launcher framework. Shared by every pipeline in `blindspot.pipelines`.

A *pipeline* (historically "flow") is an aggregation, not a code container: it
declares an ordered list of stages, each stage a list of steps, each step an
invocation of a **generic** script under `blindspot/<module>.py`. Nothing
pipeline-specific lives here, and nothing generic lives in a pipeline.

    blindspot/<module>.py   generic, dataset-agnostic, reusable
    blindspot/pipelines.py  declares which of them to call, in what order, with what args

A pipeline builds a `{stage_name: [Step, ...]}` mapping and hands it to
`main()`. Everything below -- listing, dry-running, resuming from a stage,
propagating --max-spend, failing fast -- is the same for all three pipelines and
is implemented once, here.

Usage:

    from blindspot.flow import Step, main
    STAGES = {"download": [Step("charxiv", [...])], ...}
    raise SystemExit(main("literature_eval", STAGES, __doc__))

The name passed to `main()` is the pipeline's name as `blindspot.pipelines`
knows it, because that is what `--help` prints back as the command to retype.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The one entry point. `main()` is handed a pipeline name, and the command that
# runs that pipeline is this module path plus that name -- so `--help` can print
# something the reader can retype verbatim.
ENTRY_POINT = "python -m blindspot.pipelines"


def prog_for(flow: str) -> str:
    """The command line that reaches this pipeline, for argparse's `prog`.

    This used to read `python -m {flow}.run`, left over from when each pipeline
    was its own package with a `run.py`. Those packages are gone, so every
    `--help` was advertising a module that does not exist.
    """
    return f"{ENTRY_POINT} {flow}"


@dataclass
class Step:
    """One invocation of a generic script.

    `argv` is passed to the interpreter verbatim, so a step is exactly what you
    would have typed. `needs_api` gates the step behind a key check and lets
    --offline skip it; `spend` is the share of --max-spend it may use.
    """

    name: str
    argv: list[str]
    needs_api: bool = False
    spend: float | None = None
    note: str = ""
    optional: bool = False          # a failure here warns instead of aborting
    env: dict[str, str] = field(default_factory=dict)

    def rendered(self, max_spend: float | None) -> list[str]:
        argv = list(self.argv)
        if self.spend is not None and max_spend is not None:
            share = min(self.spend, max_spend)
            argv += ["--max-spend", f"{share:.4g}"]
        return [sys.executable, *argv]


def _have_api_key() -> bool:
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY=") and line.split("=", 1)[1].strip():
                return True
    return False


def _print_plan(flow: str, stages: dict[str, list[Step]], chosen: list[str]) -> None:
    print(f"flow: {flow}")
    for stage in chosen:
        steps = stages[stage]
        api = sum(1 for s in steps if s.needs_api)
        tag = f"  ({api} call the API)" if api else ""
        print(f"\n  [{stage}]  {len(steps)} step(s){tag}")
        for s in steps:
            mark = "$" if s.needs_api else " "
            print(f"    {mark} {s.name:<24} {shlex.join(s.argv)}")
            if s.note:
                print(f"      {s.note}")


def main(flow: str, stages: dict[str, list[Step]], doc: str | None = None) -> int:
    order = list(stages)
    ap = argparse.ArgumentParser(
        prog=prog_for(flow),
        description=(doc or "").strip().splitlines()[0] if doc else None,
    )
    ap.add_argument("--stage", nargs="+", choices=order, help="run only these stages")
    ap.add_argument("--from", dest="from_", choices=order, help="run this stage and everything after")
    ap.add_argument("--all", action="store_true", help="run every stage in order")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--dry-run", action="store_true", help="print each command instead of running it")
    ap.add_argument("--offline", action="store_true", help="skip every step that calls the API")
    ap.add_argument("--max-spend", type=float, default=None, help="USD ceiling, split across API steps")
    ap.add_argument("--continue-on-error", action="store_true")
    a = ap.parse_args()

    if a.stage:
        chosen = [s for s in order if s in a.stage]
    elif a.from_:
        chosen = order[order.index(a.from_):]
    elif a.all or a.list:
        chosen = order
    else:
        ap.error("pick one of --all, --stage, --from, or --list")

    if a.list:
        _print_plan(flow, stages, chosen)
        return 0

    steps = [s for st in chosen for s in stages[st]]
    api_steps = [s for s in steps if s.needs_api]
    if api_steps and not a.offline and not a.dry_run:
        if not _have_api_key():
            print(
                f"!! {len(api_steps)} step(s) need ANTHROPIC_API_KEY, which is not set.\n"
                f"   Set it, or re-run with --offline to skip them.",
                file=sys.stderr,
            )
            return 2
        if a.max_spend is None:
            print(
                f"!! {len(api_steps)} step(s) call the API and --max-spend was not set.\n"
                f"   A plain API key cannot read a credit balance, so this ceiling is the\n"
                f"   only spend control there is. Set it, or use --offline.",
                file=sys.stderr,
            )
            return 2

    per_step = (a.max_spend / len(api_steps)) if (a.max_spend and api_steps) else None
    failures: list[str] = []
    t0 = time.time()

    for stage in chosen:
        print(f"\n=== [{stage}] " + "=" * (60 - len(stage)))
        for s in stages[stage]:
            if s.needs_api and a.offline:
                print(f"--- {s.name}: skipped (--offline)")
                continue
            cmd = s.rendered(per_step if s.needs_api else None)
            print(f"--- {s.name}\n    {shlex.join(cmd[1:])}")
            if a.dry_run:
                continue
            rc = subprocess.call(cmd, cwd=ROOT, env={**_environ(), **s.env})
            if rc != 0:
                label = f"{stage}/{s.name} (exit {rc})"
                # A step's note is the one place that says what it needs. Echo it
                # on failure so a missing prerequisite reads as a precondition
                # rather than as whatever traceback the script happened to emit.
                why = f"\n      why: {s.note}" if s.note else ""
                if s.optional:
                    print(f"    ! optional step failed, continuing: {label}{why}")
                elif a.continue_on_error:
                    failures.append(label)
                    print(f"    ! FAILED, continuing: {label}{why}")
                else:
                    print(f"\n!! {label} -- aborting. Re-run with --from {stage} after fixing.{why}",
                          file=sys.stderr)
                    return rc

    dt = time.time() - t0
    if failures:
        print(f"\ndone in {dt:.0f}s with {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\ndone in {dt:.0f}s | {len(chosen)} stage(s)")
    return 0


def _environ() -> dict[str, str]:
    import os

    return dict(os.environ)
