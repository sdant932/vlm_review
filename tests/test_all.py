"""The whole test suite, in one file, in five sections.

Offline and deterministic: no API calls, no downloads, no dependency on
results/. It runs on a fresh clone in a few seconds.

    1. SCORING              -- the scorers that produce every number in the study
    2. STATS                -- confidence intervals, quantiles, the grid
    3. DATASET INVARIANTS   -- the adapter contract, against the shipped manifest
    4. REPO STRUCTURE       -- path arithmetic, import health, layering, the CLIs
    5. PIPELINE STEPS       -- every step that reaches the API must declare it

Each section keeps the module docstring it was written with. Those explain WHY
the group exists -- section 4 in particular records a bug that shipped -- and
they are the reason the tests are worth keeping, so they travel with the code.
"""

import ast
import contextlib
import importlib
import io
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest

from blindspot.core import ADAPTERS, Example, load
from blindspot.core import (
    ANLS_THRESHOLD, CHARXIV_FUZZY_QIDS, CHARXIV_STRICT_QIDS,
    anls, boolean_match, charxiv_grading_confidence, count_score,
    numeric_or_text_match, point_in_bbox, score, token_f1,
)
from blindspot.core import (
    bbox_cells, cell_of, centre_cell, is_na, quantiles, wilson,
)

ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# 1. SCORING
# =============================================================================
"""The scorers, which produce every number in the study.

These pin behaviour that is easy to break by accident and hard to notice: the
ANLS threshold boundary, the deliberate difference between ANLS normalization
and the general one, and the refusal to credit substring containment.
"""


# --------------------------------------------------------------- ANLS

def test_anls_exact_match_scores_one():
    assert anls("42 patients", ["42 patients"]) == 1.0


def test_anls_is_case_and_whitespace_insensitive():
    assert anls("  Total Revenue  ", ["total revenue"]) == 1.0


def test_anls_takes_the_best_gold():
    assert anls("blue", ["red", "blue", "green"]) == 1.0


def test_anls_scores_zero_at_and_above_the_threshold():
    """The published definition zeroes at NL >= tau, not NL > tau.

    "ab" vs "cd" is a normalized distance of exactly 1.0; "ab" vs "ad" is
    exactly 0.5, which is the boundary and must score 0 rather than 0.5.
    """
    assert anls("ab", ["cd"]) == 0.0
    assert anls("ab", ["ad"]) == 0.0


def test_anls_scores_a_near_miss_below_the_threshold():
    # one edit in eight characters -> NL = 0.125, score = 0.875
    assert anls("revenues", ["revenuex"]) == pytest.approx(0.875)


def test_anls_does_not_strip_commas():
    """Deliberately stricter than the harness's general normalizer.

    `_normalize` removes commas, which would score "1,000" against "1000" as a
    perfect match. The official DocVQA/InfographicVQA script does not, and being
    more lenient than the benchmark is as wrong as being stricter.
    """
    assert anls("1,000", ["1000"]) < 1.0


def test_anls_both_empty_scores_one():
    assert anls("", [""]) == 1.0


def test_anls_threshold_constant_matches_the_published_value():
    assert ANLS_THRESHOLD == 0.5


# ------------------------------------------------- numeric / text match

@pytest.mark.parametrize("pred,gold", [
    ("0.28", "0.280"),
    (" 0.28 ", "0.28"),
    ("28%", "28"),
    ("$1500", "1500"),
    ("1,500", "1500"),
])
def test_numeric_match_tolerates_formatting(pred, gold):
    assert numeric_or_text_match(pred, gold and [gold]) == 1.0


def test_numeric_match_rejects_genuinely_different_values():
    assert numeric_or_text_match("0.28", ["0.29"]) == 0.0
    assert numeric_or_text_match("100", ["1000"]) == 0.0


def test_numeric_match_falls_back_to_text_when_either_side_is_not_a_number():
    assert numeric_or_text_match("Fig. 3", ["fig. 3"]) == 1.0
    assert numeric_or_text_match("Fig. 3", ["Fig. 4"]) == 0.0


# ------------------------------------------------------------ token F1

def test_token_f1_refuses_substring_containment():
    """EVAL.md 3.7: "Close" must not be credited for "Close Ledger".

    The labels in the generated dataset are short and often prefixes of each
    other, so a containment-based metric would manufacture accuracy.
    """
    em, f1 = token_f1("Close", ["Close Ledger"])
    assert em == 0.0
    assert f1 == pytest.approx(2 / 3)   # partial credit, not a match


def test_token_f1_exact_match_scores_one_on_both():
    assert token_f1("Close Ledger", ["Close Ledger"]) == (1.0, 1.0)


def test_token_f1_ignores_token_order_for_f1_but_not_for_em():
    em, f1 = token_f1("Ledger Close", ["Close Ledger"])
    assert em == 0.0
    assert f1 == 1.0


def test_token_f1_no_overlap_scores_zero():
    assert token_f1("Open Journal", ["Close Ledger"]) == (0.0, 0.0)


# --------------------------------------------------------- click-in-bbox

def test_point_in_bbox_inside_and_outside():
    box = [0.2, 0.4, 0.6, 0.8]
    assert point_in_bbox((0.4, 0.6), box) == 1.0
    assert point_in_bbox((0.1, 0.6), box) == 0.0
    assert point_in_bbox((0.4, 0.9), box) == 0.0


@pytest.mark.parametrize("pt", [(0.2, 0.4), (0.6, 0.8), (0.2, 0.8), (0.6, 0.4)])
def test_point_in_bbox_counts_the_boundary_as_a_hit(pt):
    """Inclusive on all four edges. A corner-exact prediction is a hit."""
    assert point_in_bbox(pt, [0.2, 0.4, 0.6, 0.8]) == 1.0


def test_point_in_bbox_on_a_realistically_tiny_target():
    """ScreenSpot-Pro targets get down to ~0.01% of the frame; no epsilon slop."""
    box = [0.5000, 0.5000, 0.5010, 0.5010]
    assert point_in_bbox((0.5005, 0.5005), box) == 1.0
    assert point_in_bbox((0.5020, 0.5005), box) == 0.0


# ------------------------------------------------------- boolean / count

def test_boolean_match_is_case_insensitive():
    assert boolean_match("Yes", ["yes"]) == 1.0
    assert boolean_match(" NO ", ["no"]) == 1.0
    assert boolean_match("yes", ["no"]) == 0.0


def test_count_score_returns_signed_error():
    """The sign is the interesting half: consistent undercounting as object
    count rises is a different failure from noisy counting."""
    assert count_score(3, [5]) == {"score": 0.0, "abs_error": 2, "signed_error": -2}
    assert count_score(7, [5]) == {"score": 0.0, "abs_error": 2, "signed_error": 2}
    assert count_score(5, [5]) == {"score": 1.0, "abs_error": 0, "signed_error": 0}


def test_count_score_survives_an_unparseable_prediction():
    r = count_score("several", [5])
    assert r == {"score": 0.0, "abs_error": None, "signed_error": None}


# ------------------------------------------------- CharXiv confidence split

def test_charxiv_confidence_split_is_exhaustive_and_disjoint():
    """Every descriptive qid is classified exactly once.

    The report separates strict from fuzzy rather than pooling them, so a qid
    falling through the gap would be silently mis-tiered.
    """
    assert CHARXIV_STRICT_QIDS & CHARXIV_FUZZY_QIDS == set()
    assert CHARXIV_STRICT_QIDS | CHARXIV_FUZZY_QIDS == set(range(1, 20))


def test_charxiv_reasoning_split_is_graded_fuzzy():
    """qid None is the reasoning split -- short free text, graded approximately."""
    assert charxiv_grading_confidence(None) == "fuzzy"


@pytest.mark.parametrize("qid", sorted(CHARXIV_STRICT_QIDS))
def test_charxiv_strict_qids_report_strict(qid):
    assert charxiv_grading_confidence(qid) == "strict"


@pytest.mark.parametrize("qid", sorted(CHARXIV_FUZZY_QIDS))
def test_charxiv_fuzzy_qids_report_fuzzy(qid):
    assert charxiv_grading_confidence(qid) == "fuzzy"


# =============================================================================
# 2. STATS
# =============================================================================
"""Statistics and grid helpers.

These were moved out of a 3,000-line HTML renderer into `core.stats` during the
repository reorganization. Pinned here so the move stays honest and so a future
edit to a confidence interval shows up as a failing test rather than a shifted
number in a report.
"""


# ------------------------------------------------------ Wilson interval

def test_wilson_brackets_the_point_estimate():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi


def test_wilson_narrows_as_n_grows():
    """Same proportion, more data: the interval must tighten."""
    small = wilson(5, 10)
    large = wilson(500, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_stays_inside_zero_one_at_the_extremes():
    """The reason to use Wilson rather than the normal approximation: at k=0 or
    k=n the naive interval runs off the end of the scale.

    Tolerance rather than a hard bound because the closed form leaves a float
    artifact at the endpoints -- wilson(0, 30) returns a lower bound of about
    -1.2e-17 rather than exactly 0. Harmless at any display precision, and left
    unclamped so the function stays byte-identical to the version whose outputs
    the report was built from.
    """
    eps = 1e-12
    for k, n in [(0, 30), (30, 30), (0, 1), (1, 1)]:
        lo, hi = wilson(k, n)
        assert -eps <= lo <= hi <= 1.0 + eps
        assert abs(lo) < eps or lo > 0, "a nonzero lower bound must be genuinely positive"


def test_wilson_on_zero_observations_is_maximally_uninformative():
    assert wilson(0, 0) == (0.0, 1.0)


def test_wilson_matches_a_known_value():
    """95% Wilson for 57/412 -- the dark-background cell in the polarity table."""
    lo, hi = wilson(57, 412)
    assert lo == pytest.approx(0.1084, abs=5e-4)
    assert hi == pytest.approx(0.1750, abs=5e-4)


# ------------------------------------------------------------ quantiles

def test_quantiles_splits_into_k_bins_covering_every_row():
    rows = [{"v": i} for i in range(100)]
    bins = quantiles(rows, lambda r: r["v"], k=5)
    assert len(bins) == 5
    assert sum(len(chunk) for _, _, chunk in bins) == 100


def test_quantiles_bins_are_ordered_and_report_their_realised_range():
    rows = [{"v": i} for i in range(100)]
    bins = quantiles(rows, lambda r: r["v"], k=4)
    assert [(lo, hi) for lo, hi, _ in bins] == [(0, 24), (25, 49), (50, 74), (75, 99)]


def test_quantiles_drops_rows_with_a_none_key():
    """Rows missing the binning key are excluded, not sorted as zero."""
    rows = [{"v": 1}, {"v": None}, {"v": 3}, {"v": None}]
    bins = quantiles(rows, lambda r: r["v"], k=2)
    assert sum(len(chunk) for _, _, chunk in bins) == 2


def test_quantiles_omits_empty_bins_rather_than_returning_them():
    """Fewer rows than bins: report the bins that exist, not padding."""
    bins = quantiles([{"v": 1}, {"v": 2}], lambda r: r["v"], k=5)
    assert all(chunk for _, _, chunk in bins)
    assert len(bins) <= 2


# ------------------------------------------------------------- the grid

def test_cell_of_places_points_in_the_expected_cell():
    assert cell_of(0.1, 0.1, 4) == (0, 0)
    assert cell_of(0.3, 0.6, 4) == (1, 2)
    assert cell_of(0.9, 0.9, 4) == (3, 3)


def test_cell_of_clamps_the_far_edge():
    """x=1.0 must land in the last cell, not one past it."""
    assert cell_of(1.0, 1.0, 4) == (3, 3)


def test_centre_cell_uses_the_box_centre_not_its_corner():
    # a box spanning the left half vertically centred -> centre at (0.25, 0.5)
    assert centre_cell([0.0, 0.4, 0.5, 0.6], 4) == (1, 2)


def test_bbox_cells_covers_every_cell_a_box_touches():
    """Not just the cell holding the centre -- a wide box spans several."""
    cells = bbox_cells([0.05, 0.05, 0.95, 0.20], 4)
    assert cells == {(0, 0), (1, 0), (2, 0), (3, 0)}


def test_bbox_cells_of_a_tiny_box_is_a_single_cell():
    assert bbox_cells([0.51, 0.51, 0.52, 0.52], 4) == {(2, 2)}


def test_bbox_cells_always_contains_the_centre_cell():
    box = [0.13, 0.62, 0.44, 0.71]
    assert centre_cell(box, 8) in bbox_cells(box, 8)


# --------------------------------------------------------------- is_na

@pytest.mark.parametrize("v", ["Not Applicable", "not applicable", "  NOT APPLICABLE  ",
                              "The answer is not applicable here"])
def test_is_na_recognises_the_charxiv_sentinel(v):
    assert is_na(v)


@pytest.mark.parametrize("v", ["applicable", "42", "", None, "N/A"])
def test_is_na_rejects_everything_else(v):
    assert not is_na(v)


# =============================================================================
# 3. DATASET INVARIANTS
# =============================================================================
"""Invariants the adapter layer promises, checked against the shipped dataset.

`data/svg_localization` is committed, so these run on a fresh clone with no
downloads and no API key. They are the closest thing here to an integration
test: a real manifest, through the real adapter, asserting the contract that
`scoring.score()` relies on.

The contract that matters most is the bbox one. Gold boxes arrive from each
source in a different encoding -- normalized 0-1 for ScreenSpot-v2, absolute
pixels plus an `img_size` for ScreenSpot-Pro -- and `adapters.py` is where they
become one thing. If that slips, every localization number moves silently.
"""


@pytest.fixture(scope="module")
def rows():
    return load("svg_localization")


def test_the_shipped_dataset_loads(rows):
    assert len(rows) == 4723, "the committed manifest is a fixed size; a change here is a real change"


def test_every_row_is_an_example_with_the_required_fields(rows):
    for r in rows[:200]:
        assert isinstance(r, Example)
        assert r.uid and r.dataset == "svg_localization"
        assert r.images and r.question
        assert r.answer_type in {"point", "span"}


def test_uids_are_unique(rows):
    uids = [r.uid for r in rows]
    assert len(uids) == len(set(uids))


def test_every_point_gold_is_a_normalized_bbox(rows):
    """The contract: [0,1] as (x0, y0, x1, y1), whatever the source encoded.

    An xywh box would fail x1 <= 1 on most rows, and a pixel-space box would
    fail it on all of them -- which is how the convention was pinned down in the
    first place.
    """
    points = [r for r in rows if r.answer_type == "point"]
    assert points, "expected point questions in this dataset"
    for r in points:
        x0, y0, x1, y1 = r.gold
        assert 0.0 <= x0 < x1 <= 1.0, f"{r.uid}: x out of order or out of range: {r.gold}"
        assert 0.0 <= y0 < y1 <= 1.0, f"{r.uid}: y out of order or out of range: {r.gold}"


def test_point_targets_are_small_but_never_degenerate(rows):
    """Targets are text labels: a few hundredths of the frame at most, and
    never zero-area. A zero-area gold would be unhittable and would show up as
    a capability result rather than the data bug it is."""
    for r in (r for r in rows if r.answer_type == "point"):
        x0, y0, x1, y1 = r.gold
        area = (x1 - x0) * (y1 - y0)
        assert 0.0 < area < 0.25, f"{r.uid}: implausible target area {area}"


def test_the_gold_centre_is_inside_the_gold_box(rows):
    """Trivially true if the box is well-formed, which is the point: this is the
    same arithmetic `score()` does to compute centre distance."""
    for r in (r for r in rows if r.answer_type == "point")  :
        x0, y0, x1, y1 = r.gold
        assert point_in_bbox(((x0 + x1) / 2, (y0 + y1) / 2), r.gold) == 1.0


def test_image_paths_are_resolvable(rows):
    """A manifest that points at missing files scores 0% and looks like a
    capability result."""
    from pathlib import Path
    for r in rows[::50]:
        for img in r.images:
            assert Path(img).is_file(), f"{r.uid}: missing image {img}"


def test_scoring_a_point_row_produces_the_expected_shape(rows):
    r = next(r for r in rows if r.answer_type == "point")
    x0, y0, x1, y1 = r.gold
    centre = ((x0 + x1) / 2, (y0 + y1) / 2)

    hit = score(r, centre)
    assert hit["score"] == 1.0
    assert hit["metric"] == "click_in_bbox"
    assert hit["center_distance"] == pytest.approx(0.0)

    miss = score(r, (1.0, 1.0) if centre[0] < 0.5 else (0.0, 0.0))
    assert miss["score"] == 0.0
    assert miss["center_distance"] > 0


def test_span_rows_score_by_token_f1(rows):
    """`svg_localization` spans are scored EM + token F1, never by containment."""
    r = next((r for r in rows if r.answer_type == "span"), None)
    if r is None:
        pytest.skip("no span questions in this manifest")
    out = score(r, r.gold[0])
    assert out["metric"] == "svgloc_token_f1"
    assert out["score"] == 1.0 and out["exact_match"] == 1.0


def test_the_adapter_registry_is_intact():
    """Every dataset the pipeline documents is registered and callable."""
    expected = {
        "charxiv", "infographicvqa", "ai2d", "slidevqa", "slidevqa_allpages",
        "screenspot", "screenspot_pro", "flowlearn_sim",
        "svg_localization", "svg_counting", "svg_word_mc",
    }
    assert expected <= set(ADAPTERS), f"missing adapters: {expected - set(ADAPTERS)}"
    assert all(callable(v) for v in ADAPTERS.values())


# =============================================================================
# 4. REPO STRUCTURE
# =============================================================================
"""Structural invariants of the repository itself.

These exist because of a real bug. During the reorganization that split
`blindspot/` and `scripts/` into subpackages, two modules kept computing the
repository root as `Path(__file__).resolve().parent.parent` -- correct when they
sat one level below the root, wrong once they moved one level deeper. One of
them silently wrote empty manifests into `scripts/data/`.

Nothing caught it: importing a module does not evaluate its paths against the
filesystem, and both modules lack an argparse CLI so the `--help` sweep skipped
them. Hence this file.

The sweep is deliberately shape-based rather than a hand-kept list, because it
was a hand-kept list once: two `sys.path.insert` shims lived undetected in the
finetune package the whole time `test_no_module_uses_a_sys_path_shim` was
passing, because the file list never looked there. Everything shipped now lives
in the one `blindspot` package, so the sweep is that package plus this file.
"""

# `legacy/` is excluded on purpose. It holds frozen pre-consolidation copies of
# the modules kept for reference only: not importable, not packaged, not run.
# They carry sys.path shims and `parents[2]` roots that were correct for the
# nested layout they were written in, so sweeping them would fail the structural
# tests below for code that no longer executes.
PY_FILES = sorted(
    p for p in (list((ROOT / "blindspot").rglob("*.py"))
                + list((ROOT / "tests").rglob("*.py")))
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

    Since the flattening every module sits exactly one level below the root, so
    `parents[1]` is the only correct form and the `parents[2]` that survived the
    last move is now wrong. This does not pattern-match on the number: it
    resolves the expression against the file's real location and checks what it
    actually lands on.
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
    for pkg in ("blindspot",):
        for junk in ("data", "results", "outputs", "cache"):
            assert not (ROOT / pkg / junk).exists(), \
                f"{pkg}/{junk}/ exists -- something computed its root wrongly"


# ------------------------------------------------------------ import health

def test_every_module_imports():
    """stdout is swallowed: a couple of modules print at import time, and that
    noise would bury a real failure."""
    import blindspot
    bad = []
    for m in pkgutil.walk_packages(blindspot.__path__, blindspot.__name__ + "."):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                importlib.import_module(m.name)
        except Exception as e:                         # noqa: BLE001 -- collect, don't raise
            bad.append(f"{m.name}: {type(e).__name__}: {e}")
    assert bad == [], "modules failed to import:\n  " + "\n  ".join(bad)
    assert len(list(pkgutil.iter_modules(blindspot.__path__))) >= 15, \
        "the sweep found almost nothing -- has the package moved?"


def test_no_module_uses_a_sys_path_shim():
    """The package is installable; `pip install -e .` puts it on the path.

    A reintroduced sys-path shim means someone hit an import error and patched
    the symptom. Matched as a call rather than as a substring so that prose
    naming the anti-pattern -- this docstring, for one -- does not trip it.
    """
    shim = re.compile(r"sys\.path\.(insert|append)\s*\(")
    offenders = [str(p.relative_to(ROOT)) for p in PY_FILES if shim.search(p.read_text())]
    assert not offenders, f"sys.path shims are back in: {offenders}"


# `from blindspot.x import y`, `import blindspot.x`, `from blindspot import x`
IMPORT_RE = re.compile(
    r"^\s*(?:from\s+blindspot\.(\w+)|import\s+blindspot\.(\w+)"
    r"|from\s+blindspot\s+import\s+([\w,\s]+))",
    re.M,
)


def _blindspot_imports(module: str) -> set[str]:
    """Which sibling `blindspot` modules does `blindspot/<module>.py` import?"""
    src = (ROOT / "blindspot" / f"{module}.py").read_text()
    found: set[str] = set()
    for m in IMPORT_RE.finditer(src):
        if m.group(3):
            found.update(n.strip() for n in m.group(3).split(",") if n.strip())
        else:
            found.add(m.group(1) or m.group(2))
    return found - {module}


def test_the_eval_layer_does_not_import_the_reporting_layer():
    """Dependency direction: core <- eval <- report.

    `eval` importing `report` is what made a 3,000-line HTML renderer a
    prerequisite for computing a confidence interval. The shared helpers live in
    `blindspot.core`; this keeps them there. The layering used to be enforced
    between packages and is now enforced inside the one package, but the
    direction it protects is unchanged.
    """
    offenders = sorted(_blindspot_imports("eval") & {"report", "report_finetune",
                                                     "report_worked", "render_markdown"})
    assert offenders == [], f"blindspot.eval imports the reporting layer: {offenders}"


def test_core_depends_on_nothing_above_it():
    """`blindspot.core` is the bottom of the stack.

    Adapters, prompts, the runner and the scorers must be usable for a new
    dataset without dragging in evaluation, judging or reporting. The one
    permitted sibling is `blindspot.charxiv`, which is vendored constants
    (the CharXiv question map) and imports nothing itself.
    """
    offenders = sorted(_blindspot_imports("core") - {"charxiv"})
    assert offenders == [], f"blindspot.core imports upward: {offenders}"


def test_the_layering_test_can_actually_see_imports():
    """Guard the guard: a regex that matched nothing would pass both tests above."""
    assert "charxiv" in _blindspot_imports("core")
    assert "core" in _blindspot_imports("eval")


# ---------------------------------------------------------------- the CLIs

# Matched by shape, not by name, so the next entry point to land in the package
# is covered the moment it arrives. That matters: the previous glob named the
# subpackages entry points used to live in, and when they were hoisted up a
# level it stopped matching them -- fifteen CLIs lost their --help check without
# a single test turning red. Every entry point now sits at `blindspot/*.py`.
CLI_GLOBS = ("blindspot/*.py",)


def _entry_points() -> list[Path]:
    return sorted({p for g in CLI_GLOBS for p in ROOT.glob(g) if p.name != "__init__.py"})


ARGPARSE_SCRIPTS = [p for p in _entry_points() if "argparse" in p.read_text()]

# `python -m blindspot.download --help` parsing proves almost nothing: the six
# downloaders each build their own subparser, and a broken one only shows up
# when that subcommand is asked for help. So every `add_parser("name")` gets its
# own invocation.
SUBCOMMAND_RE = re.compile(r"""add_parser\(\s*["']([A-Za-z0-9][\w.-]*)["']""")

CLI_INVOCATIONS = [
    (p, sub)
    for p in ARGPARSE_SCRIPTS
    for sub in [None] + sorted(set(SUBCOMMAND_RE.findall(p.read_text())))
]
CLI_IDS = [f"{p.relative_to(ROOT)}" + (f" {s}" if s else "") for p, s in CLI_INVOCATIONS]


@pytest.mark.parametrize("path,sub", CLI_INVOCATIONS, ids=CLI_IDS)
def test_every_argparse_cli_parses_its_arguments(path, sub):
    # `-m blindspot.foo`, not the file path: that is how the pipelines and the
    # docs invoke these, and running the file directly would put `blindspot/` on
    # sys.path -- a module named `datasets.py` would shadow the Hugging Face
    # `datasets` package, which is why the CharXiv constants live in
    # `charxiv.py` and not under a generic name.
    mod = ".".join(path.relative_to(ROOT).with_suffix("").parts)
    p = subprocess.run([sys.executable, "-m", mod] + ([sub] if sub else []) + ["--help"],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, (p.stderr or p.stdout).strip()[-400:]


def test_scripts_without_a_cli_are_still_guarded():
    """A script with no argparse must at least not do its work on import.

    `analyse_gtaudit.py` used to run a full analysis merely by being imported,
    which made any import sweep both slow and side-effecting.
    """
    unguarded = []
    for p in _entry_points():
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


# =============================================================================
# 5. PIPELINE STEPS
# =============================================================================
"""Every pipeline step that reaches the API must declare it.

`blindspot/flow.py` gates three things on `Step.needs_api`: the key precheck,
the `--offline` skip, and the refusal to start without `--max-spend`. A step
that calls the model without the flag defeats all three at once -- it runs
uncapped, survives `--offline`, and never warns.

That is not hypothetical. The `worked-examples` step shipped without the flag,
and a test run spent real money discovering it: the runbook said the flow made
no API calls, the launcher showed no cost marker, and the module underneath
opens an `anthropic.Anthropic` client.

So this test resolves each Step's argv back to its source file and asserts the
declaration matches what the file actually imports.
"""

PIPELINES_MODULE = "blindspot.pipelines"


def _all_stages(mod) -> dict[str, list]:
    """Flatten every pipeline the module defines into one {stage: [Step]} map.

    Three shapes are supported, so a pipeline may declare its stages any of them:
    a PIPELINES registry (all pipelines in one file, which is the shape in use),
    a module-level STAGES, or a parameterised build().
    """
    if hasattr(mod, "PIPELINES"):
        opts = {"tasks": list(getattr(mod, "TASKS", {})), "out": None}
        merged: dict[str, list] = {}
        for name, (fn, _desc) in mod.PIPELINES.items():
            for stage, steps in fn(opts).items():
                merged.setdefault(f"{name}/{stage}", []).extend(steps)
        return merged
    if hasattr(mod, "STAGES"):
        return mod.STAGES
    return mod.build(list(mod.TASKS), None)


def _step_source(argv: list[str]) -> Path | None:
    """Resolve a step's argv to the source file it will execute.

    `-m blindspot.run_api official --datasets ...` runs `blindspot/run_api.py`:
    take the module name and drop everything after it, since the trailing words
    are that CLI's own subcommand and flags, not part of the import path.
    """
    if argv[0] == "-m":
        rel = Path(argv[1].replace(".", "/"))
        for cand in (ROOT / f"{rel}.py", ROOT / rel / "__init__.py"):
            if cand.is_file():
                return cand
        return None
    if argv[0] == "-c":
        return None
    f = ROOT / argv[0]
    return f if f.is_file() else None


_ANTHROPIC = re.compile(r"^\s*(import anthropic|from anthropic)", re.M)


def _target_imports_anthropic(argv: list[str]) -> bool | None:
    """Does this STEP reach the API? True/False, or None when it cannot be told.

    Resolving to the module is not enough. A merged module can hold one
    subcommand that calls the model and five that do not -- `eval` is exactly
    that: `tiling` spends, while `aggregate`, `localization`, `derived`,
    `ablations` and `annotate` are offline analyses. Asking only "does this file
    import anthropic" would mark all six as spending, which makes --max-spend
    look mandatory for work that costs nothing, and trains the reader to ignore
    the warning.

    So when the argv carries a subcommand, narrow to that subcommand's handler
    (`cmd_<name>`, hyphens as underscores) and look only inside it. The module
    body is still checked, because a top-level client construction affects every
    subcommand.
    """
    if argv and argv[0] == "-c":
        return False                    # an inline snippet, not a module
    f = _step_source(argv)
    if f is None:
        return None
    src = f.read_text()

    # a subcommand word is the first bare token after `-m <module>`
    sub = None
    if len(argv) > 2 and argv[0] == "-m" and not argv[2].startswith("-"):
        sub = argv[2].replace("-", "_")

    tree = ast.parse(src)
    module_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if any(_ANTHROPIC.match(ast.unparse(n)) for n in module_level):
        return True
    if sub is None:
        return bool(_ANTHROPIC.search(src))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (f"cmd_{sub}", sub):
            return bool(_ANTHROPIC.search(ast.unparse(node)))
    # no handler found -- fall back to the whole file rather than claiming safe
    return bool(_ANTHROPIC.search(src))


PIPELINE_MODULES = [PIPELINES_MODULE] if (ROOT / "blindspot" / "pipelines.py").is_file() else []


def test_there_is_at_least_one_pipeline():
    assert PIPELINE_MODULES, "blindspot/pipelines.py is missing -- has the layout moved?"
    mod = importlib.import_module(PIPELINES_MODULE)
    assert getattr(mod, "PIPELINES", None), "the PIPELINES registry is empty"
    assert all(callable(fn) and isinstance(desc, str)
               for fn, desc in mod.PIPELINES.values()), \
        "PIPELINES must map name -> (build_fn, description)"


def test_every_pipeline_step_resolves_to_a_real_file():
    """A step naming a module that does not exist fails at run time, not here.

    The flattening renamed every entry point at once, so an argv left pointing
    at the old path is the likeliest way for this file to rot.
    """
    mod = importlib.import_module(PIPELINES_MODULE)
    missing = [f"{stage}/{s.name} -> {' '.join(s.argv)}"
               for stage, steps in _all_stages(mod).items()
               for s in steps
               if s.argv and s.argv[0] != "-c" and _step_source(s.argv) is None]
    assert missing == [], (
        "step(s) point at a file that does not exist:\n  " + "\n  ".join(missing))


@pytest.mark.parametrize("modname", PIPELINE_MODULES)
def test_api_steps_are_declared(modname):
    mod = importlib.import_module(modname)
    undeclared = []
    for stage, steps in _all_stages(mod).items():
        for s in steps:
            if _target_imports_anthropic(s.argv) and not s.needs_api:
                undeclared.append(f"{stage}/{s.name} -> {' '.join(s.argv)}")
    assert not undeclared, (
        f"{modname}: step(s) reach the API without needs_api=True, so they run "
        f"uncapped and survive --offline:\n  " + "\n  ".join(undeclared))


@pytest.mark.parametrize("modname", PIPELINE_MODULES)
def test_declared_api_steps_really_touch_the_api(modname):
    """The inverse: a false needs_api makes --max-spend look mandatory when it is not."""
    mod = importlib.import_module(modname)
    overdeclared = []
    for stage, steps in _all_stages(mod).items():
        for s in steps:
            if s.needs_api and _target_imports_anthropic(s.argv) is False:
                overdeclared.append(f"{stage}/{s.name}")
    assert not overdeclared, f"{modname}: needs_api=True but no anthropic import: {overdeclared}"
