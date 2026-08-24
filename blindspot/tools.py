"""Small maintenance tools that inspect the repository rather than produce results.

Two subcommands:

    python -m blindspot.tools verify-install
    python -m blindspot.tools compare GENERATED REFERENCE [--show-values]

Both exit non-zero on failure so they work in a Makefile.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import pkgutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # blindspot/tools.py -> repo root


# =============================================================================
# verify-install
# =============================================================================
# Check the install: every module imports, every CLI parses, the dataset loads.
#
# Run after `pip install -e .` and after any refactor that moves modules around.
# Exits non-zero on the first category that fails.

def check_imports() -> list[str]:
    """Import every module in the package.

    `scripts/` and `pipelines/` were folded into `blindspot/`, so there is one
    package to sweep rather than two. Discovery is still by walk, not by a
    hand-kept list, so a new module is covered the moment it lands.

    stdout is swallowed: a couple of modules print progress at import time, and
    that noise would bury a real failure.
    """
    import blindspot
    bad = []
    for m in pkgutil.walk_packages(blindspot.__path__, blindspot.__name__ + "."):
        if m.name == "blindspot.tools":       # already imported, as __main__
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                importlib.import_module(m.name)
        except Exception as e:                # noqa: BLE001 -- report, don't raise
            bad.append(f"{m.name}: {type(e).__name__}: {e}")
    return bad


# Every entry point is now a module in the one flat package, run as
# `python -m blindspot.<name>`. Matched by SHAPE, not by name: a module is an
# entry point iff it has an `if __name__ == "__main__"` guard. That is the same
# rule the flattening used, so a module cannot quietly drop out of the sweep by
# being renamed -- which is exactly how fifteen CLIs stopped being checked when
# they last moved, with nothing to say so.
MAIN_GUARD = '__name__ == "__main__"'


def _cli_modules() -> list[str]:
    return sorted(p.stem for p in (ROOT / "blindspot").glob("*.py")
                  if p.name != "__init__.py" and MAIN_GUARD in p.read_text())


def check_clis() -> list[str]:
    """`--help` on every entry point.

    Modules without a `__main__` guard are skipped deliberately: they are
    libraries, and there is nothing to parse. The import sweep above already
    covers whether they load.
    """
    bad = []
    mods = _cli_modules()
    no_cli = sorted({p.stem for p in (ROOT / "blindspot").glob("*.py")
                     if p.name != "__init__.py"} - set(mods))
    for name in mods:
        p = subprocess.run([sys.executable, "-m", f"blindspot.{name}", "--help"],
                           cwd=ROOT, capture_output=True, text=True)
        if p.returncode != 0:
            tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["(no output)"]
            bad.append(f"blindspot/{name}.py: exit {p.returncode}: {tail[0]}")
    print(f"   {len(mods)} CLI(s) checked")
    if no_cli:
        print(f"   {len(no_cli)} module(s) are libraries, skipped: " + ", ".join(no_cli))
    return bad


def check_dataset() -> list[str]:
    from blindspot.core import load
    try:
        rows = load("svg_localization")
    except Exception as e:                    # noqa: BLE001
        return [f"svg_localization failed to load: {type(e).__name__}: {e}"]
    if not rows:
        return ["svg_localization loaded 0 questions"]
    print(f"   {len(rows)} questions, first uid {rows[0].uid} ({rows[0].answer_type})")
    return []


def verify_install() -> int:
    failed = False
    for label, fn in (("imports", check_imports),
                      ("CLIs", check_clis),
                      ("dataset", check_dataset)):
        print(f"==> {label}")
        bad = fn()
        for b in bad:
            print(f"   FAIL {b}")
        print(f"   {'ok' if not bad else f'{len(bad)} failed'}")
        failed |= bool(bad)
    return 1 if failed else 0


# =============================================================================
# compare
# =============================================================================
# Structurally compare a generated artifact against a reference one.
#
# Two runs of this pipeline over different sample sizes produce the same SHAPE
# and different NUMBERS. So a byte diff is useless and a value diff is noise;
# what is worth checking is whether the schema survived.
#
# This walks both JSON trees and reports:
#   * keys present in one and not the other (the real regressions)
#   * leaves whose type changed
#   * for numeric leaves that exist in both, the value delta -- informational only

def paths(node, prefix="") -> dict[str, object]:
    """Flatten a JSON tree to {dotted.path: leaf}. Lists collapse to [] so a
    2,380-row run and a 60-row run compare as the same shape."""
    out: dict[str, object] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(paths(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(node, list):
        out[prefix + "[]"] = f"list[{len(node)}]"
        if node:
            # compare the shape of the first element only
            out.update(paths(node[0], prefix + "[0]"))
    else:
        out[prefix] = node
    return out


def compare(generated: str, reference: str, show_values: bool = False) -> int:
    g, r = Path(generated), Path(reference)
    for p in (g, r):
        if not p.is_file():
            print(f"missing: {p}")
            return 2

    G = paths(json.loads(g.read_text()))
    R = paths(json.loads(r.read_text()))
    gk, rk = set(G), set(R)

    only_ref = sorted(rk - gk)
    only_gen = sorted(gk - rk)
    shared = gk & rk
    type_changed = [k for k in sorted(shared)
                    if type(G[k]) is not type(R[k])
                    and not (isinstance(G[k], (int, float)) and isinstance(R[k], (int, float)))]

    print(f"generated : {g}  ({len(gk)} leaves)")
    print(f"reference : {r}  ({len(rk)} leaves)")
    print()
    print(f"  shared keys        : {len(shared)}")
    print(f"  MISSING (in ref, not generated): {len(only_ref)}")
    print(f"  EXTRA   (generated, not in ref): {len(only_gen)}")
    print(f"  type changed       : {len(type_changed)}")

    for label, keys in (("MISSING", only_ref), ("EXTRA", only_gen), ("TYPE CHANGED", type_changed)):
        if keys:
            print(f"\n{label}:")
            for k in keys[:40]:
                if label == "TYPE CHANGED":
                    print(f"  {k}: {type(R[k]).__name__} -> {type(G[k]).__name__}")
                else:
                    print(f"  {k}")
            if len(keys) > 40:
                print(f"  ... and {len(keys) - 40} more")

    if show_values:
        num = [k for k in sorted(shared)
               if isinstance(G[k], (int, float)) and isinstance(R[k], (int, float))
               and not isinstance(G[k], bool)]
        print(f"\nnumeric leaves shared: {len(num)}  (values differ by sample size, not a defect)")
        for k in num[:25]:
            print(f"  {k:<58} gen={G[k]!s:>12}  ref={R[k]!s:>12}")

    # Schema fidelity is the pass/fail. Values are expected to differ.
    ok = not only_ref and not type_changed
    print("\n" + ("SCHEMA OK -- generated artifact matches the reference shape"
                  if ok else "SCHEMA MISMATCH -- see MISSING / TYPE CHANGED above"))
    return 0 if ok else 1


# =============================================================================
# CLI
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m blindspot.tools",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "verify-install",
        help="every module imports, every CLI parses, the dataset loads",
        description="Check the install: every module imports, every CLI parses, "
                    "the dataset loads. Run after `pip install -e .` and after any "
                    "refactor that moves modules around.")

    cp = sub.add_parser(
        "compare",
        help="structurally compare a generated artifact against a reference one",
        description="Structurally compare two JSON artifacts. Two runs over "
                    "different sample sizes produce the same SHAPE and different "
                    "NUMBERS, so this diffs the schema, not the values.")
    cp.add_argument("generated")
    cp.add_argument("reference")
    cp.add_argument("--show-values", action="store_true",
                    help="also print numeric deltas for shared leaves")

    a = ap.parse_args(argv)
    if a.cmd == "verify-install":
        return verify_install()
    return compare(a.generated, a.reference, a.show_values)


if __name__ == "__main__":
    raise SystemExit(main())
