"""Structural invariants of the repository itself.

These exist because of a real bug. During the reorganization that split
`blindspot/` and `scripts/` into subpackages, two modules kept computing the
repository root as `Path(__file__).resolve().parent.parent` -- correct when they
sat one level below the root, wrong once they moved one level deeper. One of
them silently wrote empty manifests into `scripts/data/`.

Nothing caught it: importing a module does not evaluate its paths against the
filesystem, and both modules lack an argparse CLI so the `--help` sweep skipped
them. Hence this file.
"""

import ast
import importlib
import io
import contextlib
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PY_FILES = sorted(
    p for p in list((ROOT / "blindspot").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))
    if "__pycache__" not in p.parts
)


def test_we_found_the_repository_root():
    assert (ROOT / "pyproject.toml").is_file()


# ------------------------------------------------- the bug that prompted this

#  Path(__file__)... .parent.parent  |  .parents[2]
DEPTH_RE = re.compile(
    r"Path\(__file__\)\.resolve\(\)(\.parents\[(\d+)\]|(?:\.parent)+)"
)


def _declared_depth(match: re.Match) -> int:
    if match.group(2) is not None:
        return int(match.group(2))
    return match.group(1).count(".parent")


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_file_relative_roots_actually_resolve_to_the_repository_root(path):
    """Any `Path(__file__)`-relative walk upward must land on the repo root.

    A module that moves between directories keeps its old depth unless someone
    notices. The repo root is identified by `pyproject.toml`, so this fails the
    moment the arithmetic and the location disagree.
    """
    for m in DEPTH_RE.finditer(path.read_text()):
        depth = _declared_depth(m)
        target = path.resolve().parents[depth - 1] if m.group(2) is None else path.resolve().parents[depth]
        assert (target / "pyproject.toml").is_file(), (
            f"{path.relative_to(ROOT)}: `{m.group(0)}` resolves to {target}, "
            f"which is not the repository root"
        )


def test_no_module_writes_into_the_source_tree():
    """`scripts/data/` and `blindspot/data/` are what the bug produced.

    Data belongs in data/, results in results/, rendered artefacts in outputs/.
    A directory of those names inside a package means a path went wrong.
    """
    for pkg in ("blindspot", "scripts"):
        for junk in ("data", "results", "outputs", "cache"):
            assert not (ROOT / pkg / junk).exists(), \
                f"{pkg}/{junk}/ exists -- something computed its root wrongly"


# ------------------------------------------------------------ import health

def test_every_module_imports():
    """stdout is swallowed: a couple of modules print at import time, and that
    noise would bury a real failure."""
    import blindspot
    import scripts
    bad = []
    for pkg in (blindspot, scripts):
        for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    importlib.import_module(m.name)
            except Exception as e:                     # noqa: BLE001 -- collect, don't raise
                bad.append(f"{m.name}: {type(e).__name__}: {e}")
    assert not bad, "modules failed to import:\n  " + "\n  ".join(bad)


def test_no_module_uses_a_sys_path_shim():
    """The package is installable; `pip install -e .` puts it on the path.

    A reintroduced `sys.path.insert` means someone hit an import error and
    patched the symptom.
    """
    offenders = [p.relative_to(ROOT) for p in PY_FILES if "sys.path.insert" in p.read_text()]
    assert not offenders, f"sys.path shims are back in: {offenders}"


def test_the_analysis_layer_does_not_import_the_reporting_layer():
    """Dependency direction: core <- judging/analysis <- reporting.

    `analysis` importing `reporting` is what made a 3,000-line HTML renderer a
    prerequisite for computing a confidence interval. The shared helpers now
    live in `core.stats`; this keeps them there.
    """
    offenders = []
    for p in (ROOT / "blindspot" / "analysis").rglob("*.py"):
        for line in p.read_text().splitlines():
            if re.match(r"\s*from blindspot\.reporting", line):
                offenders.append(f"{p.relative_to(ROOT)}: {line.strip()}")
    assert not offenders, "analysis -> reporting imports:\n  " + "\n  ".join(offenders)


def test_core_depends_on_nothing_above_it():
    offenders = []
    for p in (ROOT / "blindspot" / "core").rglob("*.py"):
        for line in p.read_text().splitlines():
            if re.match(r"\s*from blindspot\.(judging|analysis|reporting)", line):
                offenders.append(f"{p.relative_to(ROOT)}: {line.strip()}")
    assert not offenders, "core imports upward:\n  " + "\n  ".join(offenders)


# ---------------------------------------------------------------- the CLIs

ARGPARSE_SCRIPTS = [p for p in sorted((ROOT / "scripts").glob("*/*.py"))
                    if p.name != "__init__.py" and "argparse" in p.read_text()]


@pytest.mark.parametrize("path", ARGPARSE_SCRIPTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_argparse_cli_parses_its_arguments(path):
    p = subprocess.run([sys.executable, str(path), "--help"],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, (p.stderr or p.stdout).strip()[-400:]


def test_scripts_without_a_cli_are_still_guarded():
    """A script with no argparse must at least not do its work on import.

    `analyse_gtaudit.py` used to run a full analysis merely by being imported,
    which made any import sweep both slow and side-effecting.
    """
    unguarded = []
    for p in sorted((ROOT / "scripts").glob("*/*.py")):
        if p.name == "__init__.py":
            continue
        tree = ast.parse(p.read_text())
        has_guard = any(
            isinstance(n, ast.If) and "__main__" in ast.dump(n.test)
            for n in tree.body
        )
        does_work = any(
            isinstance(n, (ast.Expr, ast.For, ast.While))
            and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
            for n in tree.body
        )
        if does_work and not has_guard:
            unguarded.append(str(p.relative_to(ROOT)))
    assert not unguarded, f"these run work at import time: {unguarded}"
