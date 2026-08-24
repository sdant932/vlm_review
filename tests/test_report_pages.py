"""`blindspot.report_pages` -- the seven standalone page builds.

These are the pages the report points at rather than the report itself, and they
were the last part of the harness to be brought into the flat package: for a
while `outputs/causes/`, `outputs/drilldown.*`, `outputs/slidevqa.html` and
`outputs/tasks/` existed in the published output tree with no live code that
could rebuild them. That is what this file guards against happening again.

The `--help` sweep here is deliberately not left to the one in `test_all.py`.
That sweep discovers subcommands with a regex over `add_parser("literal")`, and
both this module and `blindspot/report.py` register theirs through a local
`add()` helper that passes the name as a *variable* -- so the regex finds none
of them and every subcommand's parser goes unchecked. Naming them here is the
cheap fix; the expensive one is teaching the sweep to resolve indirection.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "blindspot" / "report_pages.py"

SUBCOMMANDS = ["causes", "drilldown", "slidevqa", "tasks",
               "primitives", "headline", "candidates"]

# The pages the published study shipped in outputs/causes/. One per hypothesised
# blind spot, plus the index that ranks them.
STUDY_CAUSE_PAGES = {
    "absence_detection", "answer_expression", "counting", "cross_page_integration",
    "derivation_vs_reading", "effective_resolution", "ground_truth_noise",
    "label_reference_binding", "language_prior_override", "list_answer_integrity",
    "position_bias", "resolution_precision", "retrieval_search", "subplot_scope",
    "wrong_element_not_near_miss",
}


def test_the_module_imports():
    """It is 7,600 lines of merged reporting code; an import error is the first
    thing a bad merge produces and the cheapest thing to catch."""
    import blindspot.report_pages as rp

    assert rp.__doc__
    assert callable(rp.main)


def test_it_is_registered_in_the_package_exports():
    import blindspot

    assert "report_pages" in blindspot.__all__


@pytest.mark.parametrize("sub", [None] + SUBCOMMANDS)
def test_every_subcommand_help_parses(sub):
    argv = [sys.executable, "-m", "blindspot.report_pages"]
    argv += [sub] if sub else []
    p = subprocess.run(argv + ["--help"], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, (p.stderr or p.stdout).strip()[-400:]
    assert "usage:" in p.stdout


def test_the_subcommand_list_is_exactly_what_main_registers():
    """A subcommand added without a `--help` case above would slip through."""
    import blindspot.report_pages as rp

    p = subprocess.run([sys.executable, "-m", "blindspot.report_pages", "--help"],
                       cwd=ROOT, capture_output=True, text=True)
    for name in SUBCOMMANDS:
        assert re.search(rf"\b{name}\b", p.stdout), f"{name} missing from --help"
    assert {f.__name__ for f in
            (rp.cmd_causes, rp.cmd_drilldown, rp.cmd_slidevqa, rp.cmd_tasks,
             rp.cmd_primitives, rp.cmd_headline, rp.cmd_candidates)} == \
        {f"cmd_{n}" for n in SUBCOMMANDS}


# --------------------------------------------------------------- causes


def _declared_cause_ids() -> set[str]:
    """The id each `c_*` builder gives its `Cause`, read out of the source.

    Every builder needs the full scored study to run, so the ids cannot be
    collected by calling them. They are the first positional argument of the
    single `Cause(...)` construction in each, and reading them statically is
    enough to catch a page that was dropped or silently renamed.
    """
    tree = ast.parse(MODULE.read_text())
    builders = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "BUILDERS" for t in node.targets):
            builders = [e.id for e in node.value.elts]
    assert builders, "BUILDERS is not a flat list of builder names any more"

    ids = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in builders:
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                        and sub.func.id == "Cause" and sub.args
                        and isinstance(sub.args[0], ast.Constant)):
                    ids.add(sub.args[0].value)
    return ids


def test_the_builders_declare_the_studys_fifteen_causes():
    assert _declared_cause_ids() == STUDY_CAUSE_PAGES


class _FakeData:
    """The four attributes `index_page` reads, and nothing else."""

    def __init__(self, benchmarks):
        meta = {"lines": 2, "unique": 2, "malformed": 0}
        self.counts = {b: {"lines": 2, "unique": 2, "unusable": 0,
                           "malformed": 0, "scored": 2} for b in benchmarks}
        self.rows = {b: [{"score": 1.0}, {"score": 0.0}] for b in benchmarks}
        self.blind = {"__meta__": dict(meta)}
        self.onepage = {"__meta__": dict(meta)}
        self.grid4 = {"__meta__": dict(meta)}


class _FakeBuilder:
    def __init__(self, *_a, **_k):
        self.jobs = []
        self.kept = {"confirmed": 0, "untested": 0, "exempt": 0}
        self.dropped = {"answerable_blind": 0, "contested_gold": 0}


def test_causes_writes_one_page_per_builder_plus_an_index(tmp_path, monkeypatch, capsys):
    """The whole point of the restore: `causes` must emit a page set.

    The fifteen real builders each need the full scored study, which is not in
    this repository, so the *data* is a fixture and the *writing* is real:
    `cmd_causes`, `cause_page`, `index_page` and the CSS/lightbox chrome all run
    unmodified, and what lands on disk is what a full build would lay out.
    """
    import blindspot.report_pages as rp

    wanted = ["effective_resolution", "language_prior_override", "counting"]

    def make(cid):
        def builder(_d, _b):
            return rp.Cause(cid, cid.replace("_", " ").title(),
                            "One sentence of claim.", "SUPPORTED", "+4.0pp",
                            ["charxiv", "ai2d"], 0.5,
                            body="<p>evidence</p>", refute="<p>refutation</p>")
        return builder

    monkeypatch.setattr(rp, "Data", lambda: _FakeData(["charxiv", "ai2d"]))
    monkeypatch.setattr(rp, "Builder", _FakeBuilder)
    monkeypatch.setattr(rp, "BUILDERS", [make(c) for c in wanted])
    monkeypatch.chdir(tmp_path)

    assert rp.main(["causes", "--no-images"]) == 0
    capsys.readouterr()

    pages = tmp_path / "outputs" / "causes"
    assert {p.name for p in pages.glob("*.html")} == \
        {f"{c}.html" for c in wanted} | {"index.html"}
    assert (tmp_path / "outputs" / "assets_causes").is_dir()

    for p in pages.glob("*.html"):
        text = p.read_text()
        assert text.startswith("<!doctype html>"), p.name
        assert len(text) > 2000, f"{p.name} is a stub, not a page"

    index = (pages / "index.html").read_text()
    for c in wanted:
        assert f'href="{c}.html"' in index


def test_causes_pages_link_to_the_overview_that_primitives_builds():
    """`causes`, `tasks`, `slidevqa` and `drilldown` all crumb back to
    `report.html`. Only `report_pages primitives` writes it -- which is why that
    build was restored alongside them rather than left in `legacy/`."""
    src = MODULE.read_text()
    assert '("../report.html", "Summary report")' in src
    assert 'default=str(OUT / "report.html")' in src


# ------------------------------------------------- the other six, structurally


# The artifact each build advertises, exactly as its `--help` prints it. A build
# that quietly changes where it writes stops matching the reference output tree,
# and this is the cheapest place to notice.
ARTIFACTS = [
    ("causes", "outputs/causes/*.html"),
    ("drilldown", "outputs/drilldown.{html,json,csv}"),
    ("slidevqa", "outputs/slidevqa.html"),
    ("tasks", "outputs/tasks/*.html"),
    ("primitives", "outputs/report.html"),
    ("headline", "outputs/aug22/report.html"),
    ("candidates", "outputs/report/candidates.html"),
]


@pytest.mark.parametrize("sub,artifact", ARTIFACTS, ids=[a for a, _ in ARTIFACTS])
def test_every_build_advertises_the_artifact_it_writes(sub, artifact):
    p = subprocess.run([sys.executable, "-m", "blindspot.report_pages", sub, "--help"],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, (p.stderr or p.stdout).strip()[-400:]
    flat = " ".join(p.stdout.split())
    assert artifact in flat, f"{sub} --help no longer names {artifact}"


def test_the_two_summary_fed_builds_read_the_files_their_writers_write():
    """`primitives` and `headline` are the only builds here with a file
    dependency on another command, and the two summaries are different files
    with different schemas. Crossing them renders a page of blanks, silently."""
    import blindspot.report_pages as rp

    prim = subprocess.run([sys.executable, "-m", "blindspot.report_pages",
                           "primitives", "--help"], cwd=ROOT,
                          capture_output=True, text=True).stdout
    head = subprocess.run([sys.executable, "-m", "blindspot.report_pages",
                           "headline", "--help"], cwd=ROOT,
                          capture_output=True, text=True).stdout
    assert "outputs/summary.json" in " ".join(prim.split())
    assert "outputs/report/summary.json" in " ".join(head.split())

    from blindspot import eval as bs_eval, report as bs_report
    assert bs_eval.OUT / "summary.json" == rp.Path("outputs/summary.json")
    assert bs_report.SUMMARY_OUT == rp.Path("outputs/report/summary.json")


def test_nothing_here_reimports_the_pre_consolidation_layout():
    """The four originals were written against `blindspot.core.scoring`,
    `blindspot.analysis.aggregate` and friends. Those packages are gone; a
    surviving reference is a module that only fails when its subcommand runs."""
    src = MODULE.read_text()
    stale = re.findall(r"blindspot\.(core|analysis|reporting|judging)\.\w+", src)
    assert stale == [], f"pre-consolidation imports are back: {sorted(set(stale))}"
