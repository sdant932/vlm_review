"""Check the install: every module imports, every CLI parses, the dataset loads.

Run after `pip install -e .` and after any refactor that moves modules around.
Exits non-zero on the first category that fails, so it works in a Makefile.

Usage:
    python scripts/verify_install.py
"""
from __future__ import annotations

import contextlib
import importlib
import io
import pkgutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check_imports() -> list[str]:
    """Import every module in both packages.

    stdout is swallowed: a couple of modules print progress at import time, and
    that noise would bury a real failure.
    """
    import blindspot
    import scripts
    bad = []
    for pkg in (blindspot, scripts):
        for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            if m.name == "scripts.verify_install":
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    importlib.import_module(m.name)
            except Exception as e:            # noqa: BLE001 -- report, don't raise
                bad.append(f"{m.name}: {type(e).__name__}: {e}")
    return bad


def check_clis() -> list[str]:
    """`--help` on every entry point that has an argparse CLI.

    Scripts without one are skipped deliberately: they start work as soon as
    they are invoked, so running them here would execute a full analysis (and
    write into outputs/) just to prove they parse. The import sweep above
    already covers whether they load.
    """
    bad, no_cli = [], []
    for f in sorted(ROOT.glob("scripts/*/*.py")):
        if f.name == "__init__.py":
            continue
        if "argparse" not in f.read_text():
            no_cli.append(f.relative_to(ROOT))
            continue
        p = subprocess.run([sys.executable, str(f), "--help"],
                           cwd=ROOT, capture_output=True, text=True)
        if p.returncode != 0:
            tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["(no output)"]
            bad.append(f"{f.relative_to(ROOT)}: exit {p.returncode}: {tail[0]}")
    if no_cli:
        print(f"   {len(no_cli)} script(s) take no arguments, skipped: "
              + ", ".join(str(n.name) for n in no_cli))
    return bad


def check_dataset() -> list[str]:
    from blindspot.core.adapters import load
    try:
        rows = load("svg_localization")
    except Exception as e:                    # noqa: BLE001
        return [f"svg_localization failed to load: {type(e).__name__}: {e}"]
    if not rows:
        return ["svg_localization loaded 0 questions"]
    print(f"   {len(rows)} questions, first uid {rows[0].uid} ({rows[0].answer_type})")
    return []


def main() -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
