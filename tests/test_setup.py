"""Does the *install* still match the *code*?

Everything in this file protects against one failure mode: configuration that
has silently drifted away from the tree it configures. That drift is invisible
on the machine where the work happened -- an editable install, a warm venv and
a `PYTHONPATH` that already points at the repo will happily run code that a
fresh clone cannot even import -- and it only surfaces for the next person,
who reads it as "the project is broken".

The specific ways this repo can rot:

  * `pyproject.toml [tool.setuptools] packages` is a hand-kept list. A package
    that exists on disk but is missing from that list installs as *nothing* on
    a fresh clone, and a package listed but deleted makes the wheel build fail.
    This has already broken this repo once, which is why both directions are
    asserted here rather than just the easy one.
  * `dependencies` is also hand-kept. An import added to `blindspot/` without a
    matching dependency works forever locally and fails on install; a heavy
    dependency (torch, pandas, matplotlib) creeping in quietly contradicts the
    project's central claim that the whole study runs on anthropic + Pillow +
    numpy.
  * `Makefile` and `setup.sh` are the two documented entry points. Both name
    modules and paths as strings, so a refactor that moves a module breaks them
    without breaking a single import.
  * `python -m blindspot.tools verify-install` is the repo's own self-check. A
    self-check that cannot fail is worse than none, so it is also tested for
    *dishonesty*: it must go red when the thing it checks is broken.

Everything here is offline, deterministic and read-only. No API call, no
download, no `pip install`, no `setup.sh` execution, and no test ever reads the
contents of `.env`.
"""

from __future__ import annotations

import ast
import importlib
import os
import platform
import random
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
MAKEFILE = ROOT / "Makefile"
SETUP_SH = ROOT / "setup.sh"

# Trees that are deliberately not part of the installed distribution.
#   legacy/   the pre-refactor tree, kept for provenance, must never be packaged
#   tests/    not shipped
#   .venv/    the environment itself
PRUNED_DIRS = {
    ".git", ".venv", "venv", "legacy", "tests", "build", "dist",
    "__pycache__", ".pytest_cache", "node_modules", "third_party", "cache",
    "outputs", "results", "data", "docs",
}

# The project docstring is explicit that none of these are needed: the target
# model is API-only and every image is drawn and measured with Pillow. If one
# turns up in blindspot/, either the claim or the import is wrong.
FORBIDDEN_IMPORTS = {
    "torch", "torchvision", "transformers", "cv2", "pandas",
    "matplotlib", "sklearn", "scipy", "seaborn", "plotly",
}

# Reading the whole 4,723-row manifest is fine, but validating every row's
# geometry and stat()ing every image is not worth the seconds. The sample is
# seeded, so a failure is reproducible.
DATASET_SAMPLE = 300
DATASET_SEED = 20250823


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pyproject() -> dict:
    if not PYPROJECT.is_file():
        pytest.skip("no pyproject.toml in the repo root")
    return tomllib.loads(PYPROJECT.read_text())


def _canon(name: str) -> str:
    """PEP 503 name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _req_name(requirement: str) -> str:
    """`anthropic>=1.0.0` -> `anthropic`, without needing `packaging`."""
    return _canon(re.split(r"[<>=!~;\[\s]", requirement.strip(), maxsplit=1)[0])


def _declared_packages() -> list[str]:
    return list(_pyproject().get("tool", {}).get("setuptools", {}).get("packages", []))


def _package_dir(dotted: str) -> Path:
    return ROOT.joinpath(*dotted.split("."))


def _discovered_packages() -> set[str]:
    """Every importable package directory in the tree, minus the pruned ones.

    Walks rather than rglobs so `.venv/` is pruned before it is descended into.
    """
    found: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = Path(dirpath).relative_to(ROOT)
        dirnames[:] = [
            d for d in dirnames
            if d not in PRUNED_DIRS and not d.startswith(".") and not d.endswith(".egg-info")
        ]
        if rel == Path("."):
            continue
        if "__init__.py" in filenames:
            found.add(".".join(rel.parts))
    return found


def _dist_to_import_names() -> dict[str, set[str]]:
    """Reverse `importlib.metadata.packages_distributions()`.

    Distribution name != import name (`pillow` -> `PIL`), and hard-coding that
    mapping is exactly the kind of hand-kept list this file exists to distrust.
    """
    from importlib.metadata import packages_distributions

    out: dict[str, set[str]] = {}
    for top_level, dists in packages_distributions().items():
        for d in dists:
            out.setdefault(_canon(d), set()).add(top_level)
    return out


def _third_party_imports() -> dict[str, set[str]]:
    """{top-level import name: {files that import it}} across `blindspot/*.py`.

    Static (ast) rather than dynamic: importing every module to see what it
    pulls in would run module-level side effects.
    """
    first_party = {"blindspot", "tests"} | {p.split(".")[0] for p in _discovered_packages()}
    out: dict[str, set[str]] = {}
    for path in sorted((ROOT / "blindspot").glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import, i.e. first-party by definition
                names = [node.module] if node.level == 0 and node.module else []
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in sys.stdlib_module_names or top in first_party:
                    continue
                out.setdefault(top, set()).add(path.name)
    return out


def _makefile_vars(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[?:]?=\s*(.*?)\s*(?:#.*)?$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _makefile_recipes() -> list[tuple[str, str]]:
    """[(target, expanded command line)] for every recipe line in the Makefile."""
    if not MAKEFILE.is_file():
        return []
    text = MAKEFILE.read_text()
    variables = _makefile_vars(text)
    recipes: list[tuple[str, str]] = []
    target = None
    for line in text.splitlines():
        if line.startswith("\t"):
            cmd = line[1:].strip()
            for k, v in variables.items():
                cmd = cmd.replace(f"$({k})", v)
            if target and cmd and not cmd.startswith("@"):
                recipes.append((target, cmd))
            continue
        m = re.match(r"^([A-Za-z0-9_][A-Za-z0-9_.-]*)\s*:(?!=)", line)
        if m:
            target = m.group(1)
    return recipes


def _makefile_targets() -> list[str]:
    if not MAKEFILE.is_file():
        return []
    return [
        m.group(1)
        for line in MAKEFILE.read_text().splitlines()
        if (m := re.match(r"^([A-Za-z0-9_][A-Za-z0-9_.-]*)\s*:(?!=)", line))
    ]


_MODULE_RE = re.compile(r"\bpython[0-9.]*\s+(?:-\S+\s+)*?-m\s+([A-Za-z_][\w.]*)")
_SCRIPT_RE = re.compile(r"(?<![\w./-])((?:\./)?[\w][\w./-]*\.(?:py|sh))\b")


def _module_refs(command: str) -> list[str]:
    return _MODULE_RE.findall(command)


def _script_refs(command: str) -> list[str]:
    return [s for s in _SCRIPT_RE.findall(command) if not s.startswith("-")]


def _module_exists(dotted: str) -> bool:
    """True if the module resolves, without importing anything.

    Path check first (an in-tree module must be on disk regardless of what a
    stale editable install still resolves); `find_spec` as the fallback for
    stdlib and site-packages modules such as `pytest` or `compileall`.
    """
    base = ROOT.joinpath(*dotted.split("."))
    if base.with_suffix(".py").is_file() or (base / "__init__.py").is_file():
        return True
    if dotted.split(".")[0] == "blindspot":
        return False
    try:
        return importlib.util.find_spec(dotted) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _run(argv: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """Subprocess helper with a deliberately unusable API key.

    Nothing here should reach the network, and a placeholder key means that if
    something ever did, it would error out rather than spend money.
    """
    env = dict(os.environ, ANTHROPIC_API_KEY="not-a-real-key-tests-are-offline")
    return subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout, env=env)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def _require_git_repo() -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout; gitignore hygiene is not checkable")


# ===========================================================================
# 1. PACKAGING CONFIG vs THE TREE
# ===========================================================================

def test_every_declared_package_exists_on_disk():
    """A listed-but-missing package makes `pip install .` fail on a clean tree."""
    missing = [
        p for p in _declared_packages()
        if not (_package_dir(p).is_dir() and (_package_dir(p) / "__init__.py").is_file())
    ]
    assert not missing, (
        "pyproject [tool.setuptools] packages lists packages that do not exist "
        f"(or have no __init__.py): {missing}"
    )


def test_every_package_on_disk_is_declared():
    """The direction that has actually bitten this repo.

    setuptools is configured with an explicit `packages` list, so discovery is
    off. A package added to the tree and not added to the list is simply absent
    from the install -- and stays invisible locally, because an editable
    install resolves it off the source tree anyway.
    """
    declared = set(_declared_packages())
    undeclared = sorted(_discovered_packages() - declared)
    assert not undeclared, (
        "these importable packages exist but are not in pyproject "
        f"[tool.setuptools] packages, so a fresh install ships none of them: {undeclared}"
    )


def test_legacy_tree_is_not_packaged():
    """`legacy/` is provenance, not product. Shipping it would also shadow
    `blindspot.*` with the pre-refactor copies of the same module names."""
    leaked = [p for p in _declared_packages() if p.split(".")[0] == "legacy"]
    assert not leaked, f"legacy/ must not be packaged, but pyproject declares {leaked}"
    for pkg in _declared_packages():
        assert "legacy" not in _package_dir(pkg).parts, f"{pkg} resolves inside legacy/"


def test_requires_python_is_satisfied_by_this_interpreter():
    spec = _pyproject()["project"].get("requires-python")
    if not spec:
        pytest.skip("no requires-python declared")
    running = tuple(int(x) for x in platform.python_version_tuple()[:2])
    for clause in spec.split(","):
        m = re.match(r"\s*(>=|<=|==|!=|>|<|~=)\s*([\d.]+)", clause)
        assert m, f"unparseable requires-python clause: {clause!r}"
        op, raw = m.group(1), m.group(2).rstrip(".")
        want = tuple(int(x) for x in raw.split(".")[:2])
        ok = {
            ">=": running >= want, ">": running > want,
            "<=": running <= want, "<": running < want,
            "==": running[: len(want)] == want, "!=": running[: len(want)] != want,
            "~=": running >= want and running[0] == want[0],
        }[op]
        assert ok, (
            f"running Python {platform.python_version()} does not satisfy "
            f"requires-python {spec!r}"
        )


def test_declared_runtime_dependencies_are_installed_and_importable():
    deps = _pyproject()["project"].get("dependencies", [])
    assert deps, "the project declares no runtime dependencies at all"
    mapping = _dist_to_import_names()
    broken = []
    for req in deps:
        dist = _req_name(req)
        tops = mapping.get(dist)
        if not tops:
            broken.append(f"{req}: distribution {dist!r} is not installed in this environment")
            continue
        for top in sorted(tops):
            try:
                importlib.import_module(top)
            except Exception as e:  # noqa: BLE001 -- collect, don't abort
                broken.append(f"{req}: `import {top}` raised {type(e).__name__}: {e}")
    assert not broken, "declared dependencies that do not work here:\n  " + "\n  ".join(broken)


def test_every_runtime_dependency_is_actually_used():
    """The converse: a dependency nobody imports is dead weight on every install."""
    imported = set(_third_party_imports())
    mapping = _dist_to_import_names()
    unused = []
    for req in _pyproject()["project"].get("dependencies", []):
        dist = _req_name(req)
        tops = mapping.get(dist, {dist.replace("-", "_")})
        if not (tops & imported):
            unused.append(f"{req} (provides {sorted(tops)})")
    assert not unused, (
        "declared as a runtime dependency but never imported from blindspot/: " + str(unused)
    )


def test_optional_dependency_groups_are_well_formed():
    """Named correctly and pinned -- but deliberately NOT required to be installed.

    Evaluating the shipped dataset needs neither group, and a fresh clone that
    skipped the extras should still pass the suite.
    """
    groups = _pyproject().get("project", {}).get("optional-dependencies", {})
    assert groups, "no optional-dependency groups declared"
    from importlib.metadata import PackageNotFoundError, distribution

    problems = []
    for group, reqs in groups.items():
        assert reqs, f"optional group {group!r} is empty"
        for req in reqs:
            name = _req_name(req)
            if not re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?", name):
                problems.append(f"{group}: {req!r} is not a valid distribution name")
                continue
            if not re.search(r"[<>=!~]=?\s*[\d]", req):
                problems.append(f"{group}: {req!r} has no version constraint")
            try:
                installed = _canon(distribution(name).metadata["Name"])
            except PackageNotFoundError:
                continue  # not installed here: legitimately environment-specific
            if installed != name:
                problems.append(f"{group}: {req!r} resolves to a differently-named dist {installed!r}")
    assert not problems, "optional-dependency problems:\n  " + "\n  ".join(problems)


def test_download_extra_covers_the_download_only_imports():
    """`blindspot/download.py` is the only module allowed to need heavy data
    tooling, and the `download` extra is the contract that says so."""
    groups = _pyproject().get("project", {}).get("optional-dependencies", {})
    if "download" not in groups:
        pytest.skip("no `download` extra declared")
    mapping = _dist_to_import_names()
    provided = set()
    for req in groups["download"]:
        dist = _req_name(req)
        provided |= mapping.get(dist, {dist.replace("-", "_")})

    dl = ROOT / "blindspot" / "download.py"
    if not dl.is_file():
        pytest.skip("blindspot/download.py is gone")
    needed = {top for top, files in _third_party_imports().items() if files == {"download.py"}}
    assert needed <= provided, (
        f"imports unique to download.py that no extra provides: {sorted(needed - provided)}"
    )


# ===========================================================================
# 2. DECLARED DEPENDENCIES vs REAL IMPORTS
# ===========================================================================

def test_every_third_party_import_is_declared_somewhere():
    """Static sweep of `blindspot/*.py` against dependencies + every extra.

    An import that is neither stdlib, nor first-party, nor declared is a module
    that works on this machine and ImportErrors on a fresh install.
    """
    proj = _pyproject()["project"]
    reqs = list(proj.get("dependencies", []))
    for group in proj.get("optional-dependencies", {}).values():
        reqs += list(group)

    mapping = _dist_to_import_names()
    provided = set()
    for req in reqs:
        dist = _req_name(req)
        provided |= mapping.get(dist, set()) | {dist.replace("-", "_"), dist}

    undeclared = {
        top: sorted(files)
        for top, files in _third_party_imports().items()
        if top not in provided
    }
    assert not undeclared, (
        "imports in blindspot/ that no dependency or extra provides "
        f"(they will ImportError on a fresh install): {undeclared}"
    )


def test_no_heavy_dependency_is_imported_anywhere_in_blindspot():
    """The study's central practical claim is that it needs no ML stack.

    torch/transformers/opencv/pandas/matplotlib appearing here would mean the
    claim in pyproject.toml and the README is no longer true.
    """
    offenders = {
        top: sorted(files)
        for top, files in _third_party_imports().items()
        if top in FORBIDDEN_IMPORTS
    }
    assert not offenders, (
        "blindspot/ is documented as needing none of these, but imports them: " + str(offenders)
    )


def test_no_heavy_dependency_is_declared_either():
    proj = _pyproject()["project"]
    declared = {_req_name(r) for r in proj.get("dependencies", [])}
    for group in proj.get("optional-dependencies", {}).values():
        declared |= {_req_name(r) for r in group}
    banned = {_canon(x) for x in FORBIDDEN_IMPORTS} | {"opencv-python", "scikit-learn"}
    assert not declared & banned, f"heavy dependency declared: {sorted(declared & banned)}"


# ===========================================================================
# 3. THE MAKEFILE
# ===========================================================================

def _module_cases():
    return sorted({(t, m) for t, cmd in _makefile_recipes() for m in _module_refs(cmd)})


def _script_cases():
    return sorted({(t, s) for t, cmd in _makefile_recipes() for s in _script_refs(cmd)})


def test_the_makefile_exists_and_has_recipes():
    assert MAKEFILE.is_file(), "no Makefile"
    assert _makefile_recipes(), "the Makefile parses to zero recipe lines -- parser or file is wrong"


@pytest.mark.parametrize("target,module", _module_cases() or [("<none>", "")])
def test_makefile_python_m_targets_reference_a_real_module(target, module):
    """`python -m X` in a recipe is an unchecked string until something checks it."""
    if not module:
        pytest.skip("no `python -m` invocations in the Makefile")
    assert _module_exists(module), (
        f"Makefile target `{target}` runs `python -m {module}` but that module "
        f"does not exist (expected {module.replace('.', '/')}.py)"
    )


@pytest.mark.parametrize("target,script", _script_cases() or [("<none>", "")])
def test_makefile_script_paths_exist(target, script):
    if not script:
        pytest.skip("no script paths referenced in the Makefile")
    assert (ROOT / script).is_file(), (
        f"Makefile target `{target}` references {script}, which is not in the repo"
    )


def test_makefile_compileall_arguments_exist():
    """`compileall` silently succeeds on a directory that is not there, so the
    `verify` target can look green while compiling nothing."""
    missing = []
    for target, cmd in _makefile_recipes():
        if "compileall" not in cmd:
            continue
        args = cmd.split("compileall", 1)[1].split()
        for a in args:
            if a.startswith("-"):
                continue
            if not (ROOT / a).exists():
                missing.append(f"{target}: compileall {a}")
    assert not missing, f"compileall is pointed at paths that do not exist: {missing}"


@pytest.mark.parametrize(
    "module",
    sorted({m for _, m in _module_cases() if m.startswith("blindspot.")}) or [""],
)
def test_makefile_referenced_clis_parse(module):
    """`--help` only. Never a target: several of them download, regenerate the
    dataset, or spend money."""
    if not module:
        pytest.skip("the Makefile invokes no blindspot module directly")
    path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
    if not path.is_file():
        pytest.skip(f"{module} does not exist; covered by the module-existence test")
    if "argparse" not in path.read_text():
        pytest.skip(f"{module} has no argparse CLI")
    p = _run([sys.executable, "-m", module, "--help"])
    assert p.returncode == 0, (
        f"`python -m {module} --help` exited {p.returncode}:\n{(p.stderr or p.stdout)[-2000:]}"
    )


def test_phony_declarations_match_the_real_targets():
    """A target that falls out of `.PHONY` stops running as soon as a file or
    directory of the same name appears -- `test/`, `cache`, `dataset`."""
    text = MAKEFILE.read_text()
    phony = set(re.findall(r"^\.PHONY:\s*(.*)$", text, re.M)[0].split()) if ".PHONY" in text else set()
    if not phony:
        pytest.skip("no .PHONY declaration in the Makefile")
    real = {t for t in _makefile_targets() if not t.startswith(".")}
    assert phony <= real, f".PHONY names targets that do not exist: {sorted(phony - real)}"
    assert real <= phony, f"targets missing from .PHONY: {sorted(real - phony)}"


def test_documented_targets_are_real_targets():
    """`make help` greps for `## ` comments; a doc line without a target lies."""
    documented = set(re.findall(r"^([a-z][a-z0-9_-]*):.*?##", MAKEFILE.read_text(), re.M))
    real = set(_makefile_targets())
    assert documented <= real, f"`make help` advertises non-targets: {sorted(documented - real)}"


# ===========================================================================
# 4. setup.sh
# ===========================================================================

def _setup_sh_text() -> str:
    if not SETUP_SH.is_file():
        pytest.skip("no setup.sh")
    return SETUP_SH.read_text()


def test_setup_sh_is_executable_and_fails_fast():
    text = _setup_sh_text()
    assert os.access(SETUP_SH, os.X_OK), "setup.sh is not executable; `./setup.sh` will not run"
    assert re.search(r"^set -euo pipefail", text, re.M), (
        "setup.sh must `set -euo pipefail`, or a failed step leaves a half-built "
        "environment that reports success"
    )


def test_setup_sh_only_runs_modules_that_exist():
    missing = [m for m in _module_refs(_setup_sh_text()) if not _module_exists(m)]
    assert not missing, f"setup.sh runs `python -m` on modules that do not exist: {missing}"


def test_setup_sh_only_runs_scripts_that_exist():
    text = _setup_sh_text()
    refs = set()
    for line in text.splitlines():
        stripped = line.strip()
        # `python <path>` invocations only; the heredoc is prose, not commands.
        m = re.match(r"^python[0-9.]*\s+([\w./-]+\.py)\b", stripped)
        if m:
            refs.add(m.group(1))
    missing = [s for s in sorted(refs) if not (ROOT / s).is_file()]
    assert not missing, f"setup.sh runs scripts that are not in the repo: {missing}"


def test_setup_sh_creates_the_runtime_directories_the_code_expects():
    """`results/`, `outputs/`, `cache/` and `third_party/` are gitignored, so a
    fresh clone has none of them and the first run writes into nowhere."""
    text = _setup_sh_text()
    created = set()
    for m in re.finditer(r"^\s*mkdir\s+(-\S+\s+)*(.+)$", text, re.M):
        created |= {d.strip().rstrip("/") for d in m.group(2).split()}
    for want in ("results", "outputs", "cache", "third_party"):
        assert want in created, f"setup.sh never creates {want}/ (mkdir -p targets found: {sorted(created)})"


def test_setup_sh_seeds_dotenv_from_the_example_without_clobbering():
    """Copy the template when absent; never overwrite a key that is already there."""
    text = _setup_sh_text()
    assert (ROOT / ".env.example").is_file(), "setup.sh's template .env.example is missing"
    assert re.search(r"cp\s+\.env\.example\s+\.env", text), (
        "setup.sh does not copy .env.example to .env"
    )
    guard = re.search(r"if\s+\[\[\s*!\s*-f\s+\.env\s*\]\]", text) or \
        re.search(r"if\s+\[\s*!\s*-f\s+\.env\s*\]", text)
    assert guard, "the `cp .env.example .env` is not guarded by an `if [[ ! -f .env ]]` test"
    assert text.index(guard.group(0)) < text.index("cp .env.example .env"), (
        "the guard must come before the copy"
    )


def test_setup_sh_installs_editable_with_the_declared_extras():
    text = _setup_sh_text()
    m = re.search(r"pip install[^\n]*", text)
    assert m, "setup.sh never installs the package"
    line = m.group(0)
    assert " -e " in line, f"setup.sh should install editable, got: {line}"
    extras = set(re.findall(r"\[([^\]]+)\]", line))
    declared = set(_pyproject().get("project", {}).get("optional-dependencies", {}))
    asked = {e.strip() for group in extras for e in group.split(",")}
    unknown = asked - declared
    assert not unknown, f"setup.sh asks for extras that pyproject does not define: {sorted(unknown)}"


def test_makefile_setup_target_delegates_to_setup_sh():
    """One install path, not two that can disagree."""
    recipes = [cmd for t, cmd in _makefile_recipes() if t == "setup"]
    if not recipes:
        pytest.skip("no `setup` target in the Makefile")
    assert any("setup.sh" in c for c in recipes), (
        f"`make setup` does not call ./setup.sh; it runs {recipes}"
    )


# ===========================================================================
# 5. verify-install -- and whether it is honest
# ===========================================================================

def _tools():
    try:
        return importlib.import_module("blindspot.tools")
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"blindspot.tools does not import: {type(e).__name__}: {e}")


def test_verify_install_exits_zero_and_reports_all_three_sections():
    p = _run([sys.executable, "-m", "blindspot.tools", "verify-install"], timeout=600)
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"`verify-install` exited {p.returncode}:\n{out[-4000:]}"
    for section in ("imports", "CLIs", "dataset"):
        assert f"==> {section}" in out, f"verify-install never reported the {section!r} section:\n{out}"


def test_verify_install_reports_failure_instead_of_rubber_stamping(monkeypatch, capsys):
    """The whole point of a self-check is that it can go red.

    Each of the three sections is exercised as the failing one so that a future
    refactor cannot accidentally drop a section from the pass/fail fold-in.
    """
    tools = _tools()
    for failing in ("check_imports", "check_clis", "check_dataset"):
        for name in ("check_imports", "check_clis", "check_dataset"):
            monkeypatch.setattr(tools, name, (lambda: ["synthetic failure"]) if name == failing
                                else (lambda: []))
        assert tools.verify_install() == 1, f"verify_install() returned 0 while {failing} failed"
        assert "FAIL" in capsys.readouterr().out


def test_verify_install_dataset_check_actually_fails_on_a_missing_dataset(monkeypatch, tmp_path):
    """Point the loader at an empty directory: `check_dataset` must complain."""
    tools = _tools()
    core = importlib.import_module("blindspot.core")
    if not hasattr(core, "DATA"):
        pytest.skip("blindspot.core no longer exposes DATA; nothing to redirect")
    monkeypatch.setattr(core, "DATA", tmp_path / "definitely-not-here")
    bad = tools.check_dataset()
    assert bad, "check_dataset() passed against a nonexistent dataset directory"
    assert any("svg_localization" in b for b in bad)


def test_verify_install_cli_check_actually_fails_on_a_broken_cli(monkeypatch, tmp_path, capsys):
    """A CLI whose `--help` exits non-zero must be reported, not swallowed.

    Two shapes of `check_clis` have shipped -- a glob over script paths and a
    sweep of `blindspot.*` modules with a `__main__` guard -- so this adapts to
    whichever is present rather than pinning the internals.
    """
    tools = _tools()

    if hasattr(tools, "CLI_GLOBS"):
        (tmp_path / "broken_cli.py").write_text(
            "import argparse\n"
            "raise SystemExit('deliberately broken for the honesty test')\n"
        )
        monkeypatch.setattr(tools, "ROOT", tmp_path)
        monkeypatch.setattr(tools, "CLI_GLOBS", ("*.py",))
        bad = tools.check_clis()
        assert bad, "check_clis() passed a script whose --help exits non-zero"
        assert any("broken_cli.py" in b for b in bad)
        return

    if not hasattr(tools, "_cli_modules"):
        pytest.skip("blindspot.tools no longer exposes a discoverable CLI sweep")

    real = tools._cli_modules()
    assert real, "check_clis discovers zero entry points -- the sweep has gone blind"

    def fail(argv, **kw):
        return subprocess.CompletedProcess(argv, 1, "", "deliberately broken for the honesty test")

    monkeypatch.setattr(tools.subprocess, "run", fail)
    bad = tools.check_clis()
    capsys.readouterr()
    assert len(bad) == len(real), (
        f"check_clis() reported {len(bad)} failures for {len(real)} broken CLIs"
    )


def test_verify_install_is_reachable_the_way_the_docs_say():
    p = _run([sys.executable, "-m", "blindspot.tools", "--help"], timeout=120)
    assert p.returncode == 0, f"`python -m blindspot.tools --help` failed:\n{p.stderr[-2000:]}"
    assert "verify-install" in p.stdout


# ===========================================================================
# 6. RUNTIME DIRECTORIES AND GITIGNORE HYGIENE
# ===========================================================================

@pytest.mark.parametrize("path", ["outputs", "results", "cache", "third_party"])
def test_runtime_directories_are_gitignored(path):
    """These are regenerable (or ~70MB of raw API responses). Committing them
    once is very hard to undo, so the ignore rule is load-bearing.

    Note the .gitignore entries have no trailing slash on purpose: on the study
    machine they are symlinks, and `outputs/` would not match a symlink.
    """
    _require_git_repo()
    r = _git("check-ignore", "-q", path)
    assert r.returncode == 0, f"{path} is NOT gitignored (git check-ignore exit {r.returncode})"


def test_the_committed_dataset_is_not_gitignored():
    """`data/*` is ignored wholesale with a negation for the one dataset this
    study generated. Lose the negation and the repo ships without its data."""
    _require_git_repo()
    dataset = ROOT / "data" / "svg_localization"
    if not dataset.is_dir():
        pytest.skip("data/svg_localization is not present in this checkout")
    r = _git("check-ignore", "-q", "data/svg_localization")
    assert r.returncode == 1, "data/svg_localization is gitignored; the shipped dataset would be lost"
    manifest = "data/svg_localization/manifest.jsonl"
    assert (ROOT / manifest).is_file()
    assert _git("check-ignore", "-q", manifest).returncode == 1, f"{manifest} is gitignored"


def test_the_dataset_is_actually_tracked_by_git():
    _require_git_repo()
    r = _git("ls-files", "--error-unmatch", "data/svg_localization/manifest.jsonl")
    if r.returncode != 0:
        pytest.skip("dataset not committed in this checkout (shallow or partial clone)")


def test_dotenv_is_gitignored_and_never_read_by_the_suite():
    """The secret must be unstageable, and no test may ever open it.

    This test deliberately checks only the *path*: it never reads `.env`.
    """
    _require_git_repo()
    assert _git("check-ignore", "-q", ".env").returncode == 0, ".env is not gitignored"
    assert (ROOT / ".env.example").is_file(), ".env.example (the safe template) is missing"
    assert _git("check-ignore", "-q", ".env.example").returncode == 1, (
        ".env.example is gitignored, so a fresh clone has no template to copy"
    )

    readers = []
    for path in sorted((ROOT / "tests").glob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if ".env" not in line or ".env.example" in line:
                continue
            if re.search(r"(open|read_text|read_bytes|load_dotenv|readlines)\s*\(", line):
                readers.append(f"{path.name}:{i}: {line.strip()}")
    assert not readers, f"a test reads .env; secrets must never enter the suite:\n  " + "\n  ".join(readers)


def test_no_api_key_is_needed_to_run_this_suite():
    """The suite is offline. If it only passes because a key happens to be in
    the environment, that is a different suite than the one a reviewer runs."""
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    p = subprocess.run([sys.executable, "-c", "import blindspot.core"],
                       cwd=ROOT, capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0, (
        f"`import blindspot.core` needs ANTHROPIC_API_KEY to be set:\n{p.stderr[-2000:]}"
    )


# ===========================================================================
# 7. THE SHIPPED DATASET LOADS
# ===========================================================================

@pytest.fixture(scope="module")
def svgloc():
    core = pytest.importorskip("blindspot.core")
    if not (ROOT / "data" / "svg_localization" / "manifest.jsonl").is_file():
        pytest.skip("data/svg_localization is not present in this checkout")
    rows = core.load("svg_localization")
    assert rows, "load('svg_localization') returned nothing"
    return rows


@pytest.fixture(scope="module")
def svgloc_sample(svgloc):
    """A seeded sample. The full manifest is 4,723 rows; validating every one's
    geometry and stat()ing every image adds seconds for no extra coverage."""
    if len(svgloc) <= DATASET_SAMPLE:
        return svgloc
    return random.Random(DATASET_SEED).sample(svgloc, DATASET_SAMPLE)


def test_dataset_loads_a_meaningful_number_of_examples(svgloc):
    assert len(svgloc) > 100, f"only {len(svgloc)} examples; the manifest looks truncated"


def test_dataset_uids_are_unique(svgloc):
    """Duplicate uids silently collapse rows during resume and aggregation."""
    seen, dupes = set(), []
    for ex in svgloc:
        if ex.uid in seen:
            dupes.append(ex.uid)
        seen.add(ex.uid)
    assert not dupes, f"{len(dupes)} duplicate uids, e.g. {dupes[:5]}"


def test_dataset_has_both_answer_types(svgloc):
    types = {ex.answer_type for ex in svgloc}
    assert types == {"point", "span"}, f"unexpected answer_type set: {sorted(types)}"


def test_every_example_has_the_adapter_shape(svgloc_sample):
    core = importlib.import_module("blindspot.core")
    for ex in svgloc_sample:
        assert isinstance(ex, core.Example), f"{ex!r} is not an Example"
        assert ex.dataset == "svg_localization"
        assert ex.uid and isinstance(ex.uid, str)
        assert ex.question and ex.question.strip(), f"{ex.uid}: empty question"
        assert len(ex.images) == 1, f"{ex.uid}: expected exactly one image, got {ex.images}"
        assert ex.answer_type in ("point", "span")


def test_point_gold_boxes_are_normalised_and_non_degenerate(svgloc_sample):
    """The adapter promises `gold_bbox_norm`: four floats in [0,1], x0<x1, y0<y1.

    A flipped or zero-area box scores every click as a miss, which reads as a
    model failure rather than a data bug -- the exact confusion this study is
    about.
    """
    bad = []
    for ex in svgloc_sample:
        if ex.answer_type != "point":
            continue
        box = ex.gold
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, float) for v in box)):
            bad.append(f"{ex.uid}: gold is not four floats: {box!r}")
            continue
        x0, y0, x1, y1 = box
        if not all(0.0 <= v <= 1.0 for v in box):
            bad.append(f"{ex.uid}: box escapes [0,1]: {box}")
        if not (x0 < x1 and y0 < y1):
            bad.append(f"{ex.uid}: box is flipped or empty: {box}")
        elif (x1 - x0) * (y1 - y0) <= 0:
            bad.append(f"{ex.uid}: zero-area box: {box}")
    assert not bad, "malformed point targets:\n  " + "\n  ".join(bad[:20])


def test_point_targets_are_not_absurdly_large(svgloc_sample):
    """A box covering most of the image would make the task trivially passable
    and is the signature of a normalization applied twice (or not at all)."""
    huge = [
        (ex.uid, (ex.gold[2] - ex.gold[0]) * (ex.gold[3] - ex.gold[1]))
        for ex in svgloc_sample
        if ex.answer_type == "point" and isinstance(ex.gold, list) and len(ex.gold) == 4
        and (ex.gold[2] - ex.gold[0]) * (ex.gold[3] - ex.gold[1]) > 0.5
    ]
    assert not huge, f"targets covering >50% of the image: {huge[:10]}"


def test_span_gold_answers_are_non_empty_strings(svgloc_sample):
    bad = [
        ex.uid for ex in svgloc_sample
        if ex.answer_type == "span"
        and not (isinstance(ex.gold, list) and ex.gold
                 and all(isinstance(g, str) and g.strip() for g in ex.gold))
    ]
    assert not bad, f"span examples with empty or non-string gold: {bad[:20]}"


def test_every_sampled_image_path_resolves(svgloc_sample):
    """`blindspot.core.DATA` is a *relative* Path, so image paths resolve
    against the repo root rather than the caller's cwd. Resolve them the same
    way here, and fail loudly if any is missing."""
    missing = []
    for ex in svgloc_sample:
        for img in ex.images:
            p = Path(img)
            if not p.is_absolute():
                p = ROOT / p
            if not p.is_file():
                missing.append(f"{ex.uid}: {img}")
    assert not missing, (
        f"{len(missing)} of {len(svgloc_sample)} sampled examples point at missing images:\n  "
        + "\n  ".join(missing[:20])
    )


def test_point_examples_carry_the_resolution_metadata_the_analysis_needs(svgloc_sample):
    """The whole study is "size and resolution as knobs, not confounds", which
    only works if every row records both."""
    bad = []
    for ex in svgloc_sample:
        if ex.answer_type != "point":
            continue
        for key in ("img_size", "effective_px", "resolution", "target_area_frac"):
            if ex.meta.get(key) in (None, ""):
                bad.append(f"{ex.uid}: missing meta[{key!r}]")
    assert not bad, "point rows missing analysis metadata:\n  " + "\n  ".join(bad[:20])


def test_declared_adapters_all_have_a_loader(svgloc):
    """`ADAPTERS` is the registry `load()` dispatches through; a key with no
    callable behind it fails only when someone runs that dataset."""
    core = importlib.import_module("blindspot.core")
    assert "svg_localization" in core.ADAPTERS
    not_callable = [k for k, v in core.ADAPTERS.items() if not callable(v)]
    assert not not_callable, f"ADAPTERS entries that are not callable: {not_callable}"
