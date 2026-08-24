"""Function-level tests for `blindspot/core.py` and `blindspot/eval.py`.

WHY THIS FILE EXISTS
--------------------
Every number in the study is produced by a few dozen small pure functions in
these two modules. A scorer that shifts by one threshold, a Wilson interval
that starts clamping, a prompt that grows an extra instruction line -- none of
those raise, none of them look wrong in a diff, and all of them move a
published result. `tests/test_all.py` covers the happy paths of scoring and
stats; this file covers the parts that only bite at the edges:

  * threshold boundaries in both directions, and the deliberate places where
    two near-identical helpers disagree (`_anls_normalize` vs `_normalize`,
    `core.wilson` vs `eval.wilson_of_values`, `band` vs `classify_point`,
    `is_na` vs `is_not_applicable`);
  * degenerate inputs -- empty lists, n=0, unknown keys -- which reach these
    functions in real runs and must not raise or silently invent a number;
  * the invariants the module docstrings *promise*: one Example is one
    scoreable question, gold boxes are always normalised [0,1] (x0,y0,x1,y1),
    CharXiv descriptive prompts go through untouched, sampling is stratified
    and deterministic, semantic judgements are deferred rather than guessed;
  * two bugs recently fixed in `eval.py` that were both silent-data-loss
    class: a zero-row scoring run overwriting a good summary, and `overall`
    pooling rungs that `headline` does not report.

Offline, deterministic, no API calls. Real committed data is used wherever it
exists (`data/charxiv`, `data/ai2d`, `data/svg_localization` and its two
derived sets); invented fixtures are kept tiny and obvious.
"""

from __future__ import annotations

import collections
import contextlib
import json
import math
import os
import random
from pathlib import Path

import pytest

from blindspot import core
from blindspot import eval as bs_eval
from blindspot.core import (
    ADAPTERS,
    AI2D_QTYPE,
    CHARXIV_QID,
    FAILURE_MODE_LABELS,
    LABELS,
    MODES,
    PRIMITIVE_LABELS,
    PRIMITIVES,
    SCHEMAS,
    SPAN_INSTRUCTION,
    Example,
    anls,
    bbox_cells,
    boolean_match,
    by_label,
    cell_key_for,
    cell_of,
    centre_cell,
    classify,
    classify_point,
    count_score,
    edge_style,
    encode_image,
    graph_f1,
    has_edge,
    hops,
    is_na,
    is_not_applicable,
    load,
    numeric_or_text_match,
    parse_mermaid,
    parse_response,
    point_in_bbox,
    primitive_for,
    prompt_text,
    quantiles,
    score,
    stratify,
    token_f1,
    wilson,
)
from blindspot.core import summarize as summarize_failure_modes

ROOT = Path(__file__).resolve().parents[1]

# Adapters whose manifest is committed. Everything else is downloaded on
# demand, so those datasets are skipped rather than failed.
SHIPPED = [ds for ds in ("charxiv", "ai2d", "svg_localization",
                         "svg_counting", "svg_word_mc")
           if ds in ADAPTERS]


@contextlib.contextmanager
def at_repo_root():
    """`core.DATA` / `core.RESULTS` are relative paths, so anything that reads
    a manifest or a result file needs the repository root as cwd."""
    prev = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


def _load(dataset: str):
    with at_repo_root():
        return load(dataset)


def ex(answer_type="span", *, dataset="synthetic", question="Q?", gold=None, **meta):
    """A minimal Example. Only the fields the function under test reads."""
    return Example(uid="u", dataset=dataset, images=[], question=question,
                   answer_type=answer_type, gold=gold if gold is not None else ["g"],
                   meta=meta)


# #############################################################################
# #                                                                           #
# #   SUBJECT A -- blindspot/core.py                                          #
# #                                                                           #
# #############################################################################


# =============================================================================
# A1. SCORING -- each scorer follows its benchmark's own published metric
# =============================================================================
"""The scorers. `test_all.py` pins the common cases; these pin the edges where
a plausible "cleanup" would silently change a published number: the ANLS
threshold approached from both sides, the normalizer that is deliberately
stricter than the general one, and which scorers take the best gold.
"""


def test_anls_threshold_boundary_from_both_sides():
    """NL < tau scores 1-NL; NL == tau scores 0. The boundary is strict.

    Approached from below and from above with the same shape of edit, so a
    change from `<` to `<=` fails here rather than moving InfographicVQA by a
    fraction of a point.
    """
    # 5 characters, 2 substitutions -> NL = 0.4, just under tau
    assert anls("abcde", ["abcxy"]) == pytest.approx(0.6)
    # 2 characters, 1 substitution -> NL = 0.5, exactly tau
    assert anls("ab", ["ad"]) == 0.0
    # 3 characters, 2 substitutions -> NL = 2/3, over tau
    assert anls("abc", ["axy"]) == 0.0


def test_anls_threshold_is_a_parameter_and_the_comparison_stays_strict():
    """Raising tau widens the credited band, but `nl < threshold` stays strict:
    a normalized distance of exactly 1.0 scores 0 even at tau = 1.0."""
    assert anls("abc", ["axy"], threshold=0.7) == pytest.approx(1 / 3)
    assert anls("ab", ["cd"], threshold=1.0) == 0.0


def test_anls_normalization_is_stricter_than_the_harness_general_one():
    """"1,000" vs "1000" is the case that separates them.

    `_normalize` strips commas, so `numeric_or_text_match` calls it a perfect
    match. The official DocVQA/InfographicVQA script does not strip commas, so
    ANLS must charge one edit for the separator. Being *more lenient* than the
    benchmark is as wrong as being stricter, and this is the single input where
    that difference is visible.
    """
    assert anls("1,000", ["1000"]) == pytest.approx(0.8)   # one deletion in 5 chars
    assert numeric_or_text_match("1,000", ["1000"]) == 1.0
    assert core._anls_normalize("1,000") == "1,000"
    assert core._normalize("1,000") == "1000"


def test_anls_empty_prediction_against_a_nonempty_gold_scores_zero():
    """Only the both-empty case is a match; a blank answer never earns credit."""
    assert anls("", ["x"]) == 0.0
    assert anls("x", [""]) == 0.0
    assert anls("", [""]) == 1.0


def test_anls_with_no_golds_at_all_scores_zero_rather_than_raising():
    assert anls("anything", []) == 0.0


def test_token_f1_refuses_containment_in_both_directions():
    """`test_all.py` pins prediction-inside-gold; the reverse must not be
    credited either, or a verbose answer would buy exact-match for free."""
    assert token_f1("Close", ["Close Ledger"]) == (0.0, pytest.approx(2 / 3))
    assert token_f1("Close Ledger Now", ["Close Ledger"]) == (0.0, pytest.approx(0.8))


def test_token_f1_takes_the_best_gold_not_the_first():
    """SlideVQA and svg_localization both ship alternative phrasings."""
    em, f1 = token_f1("Close Ledger", ["Open Journal", "Close Ledger"])
    assert (em, f1) == (1.0, 1.0)


def test_token_f1_on_an_empty_prediction_scores_zero_without_dividing_by_zero():
    assert token_f1("", ["Close Ledger"]) == (0.0, 0.0)
    assert token_f1("Close", [""]) == (0.0, 0.0)
    assert token_f1("", [""]) == (1.0, 1.0)


def test_point_in_bbox_is_inclusive_on_all_four_edges_of_a_tiny_target():
    """ScreenSpot-Pro targets reach ~0.01% of the frame. There is no epsilon
    slop, so the four corners and the four edges of a 0.001-wide box are the
    boundary test that matters."""
    box = [0.5000, 0.5000, 0.5010, 0.5010]
    for pt in [(0.5000, 0.5000), (0.5010, 0.5010), (0.5000, 0.5010),
               (0.5010, 0.5000), (0.5005, 0.5000), (0.5000, 0.5005)]:
        assert point_in_bbox(pt, box) == 1.0, pt
    for pt in [(0.4999, 0.5005), (0.5011, 0.5005), (0.5005, 0.4999), (0.5005, 0.5011)]:
        assert point_in_bbox(pt, box) == 0.0, pt


def test_point_in_bbox_on_a_degenerate_box_only_credits_the_exact_point():
    """A zero-area gold is a data bug, not a capability result -- but the
    scorer must still behave predictably if one reaches it."""
    assert point_in_bbox((0.3, 0.7), [0.3, 0.7, 0.3, 0.7]) == 1.0
    assert point_in_bbox((0.3001, 0.7), [0.3, 0.7, 0.3, 0.7]) == 0.0


def test_count_score_reports_signed_error_not_just_accuracy():
    """The sign is the mechanism: undercounting a crowd and overcounting a
    repeating pattern are different failures and must not pool."""
    assert count_score(3, [7]) == {"score": 0.0, "abs_error": 4, "signed_error": -4}
    assert count_score(9, [7]) == {"score": 0.0, "abs_error": 2, "signed_error": 2}
    # A numeric string still parses; a negative answer is simply wrong.
    assert count_score("7", [7])["score"] == 1.0
    assert count_score(-1, [7]) == {"score": 0.0, "abs_error": 8, "signed_error": -8}


def test_count_score_uses_the_first_gold_only_unlike_every_other_scorer():
    """CURRENT BEHAVIOUR, pinned deliberately.

    `anls`, `token_f1`, `numeric_or_text_match` and `boolean_match` all take
    the best-matching gold. `count_score` reads `golds[0]` and ignores the
    rest, so a multi-gold count question would be scored against one of them
    arbitrarily. Every count dataset ships exactly one gold today, which is why
    this is invisible -- and why it needs a test before someone adds a second.
    """
    assert count_score(5, [3, 5])["score"] == 0.0
    assert count_score(3, [3, 5])["score"] == 1.0


def test_count_score_on_an_empty_gold_list_degrades_instead_of_raising():
    assert count_score(3, []) == {"score": 0.0, "abs_error": None, "signed_error": None}


def test_numeric_or_text_match_prefers_numeric_comparison_when_both_sides_parse():
    """Numeric vs text form is the whole point of this scorer: CharXiv's strict
    types answer with values, and "0.30" must equal "0.3" without "3" equalling
    "three"."""
    assert numeric_or_text_match("0.30", ["0.3"]) == 1.0
    assert numeric_or_text_match(" -2.50 ", ["-2.5"]) == 1.0
    assert numeric_or_text_match("3", ["three"]) == 0.0        # text side does not parse
    assert numeric_or_text_match("three", ["3"]) == 0.0
    assert numeric_or_text_match("three", ["Three"]) == 1.0    # falls back to text


def test_numeric_or_text_match_takes_the_best_gold():
    assert numeric_or_text_match("0.28", ["0.99", "0.280"]) == 1.0


def test_numeric_or_text_match_tolerance_is_relative_above_one_and_absolute_below():
    """The tolerance is `1e-6 * max(1, |gold|)`.

    Above 1 it scales with the value, so a rounded large number still matches;
    at or below 1 it is a flat 1e-6, so small values are not given proportional
    slack they do not need. Both halves matter: CharXiv answers span axis
    fractions and six-figure counts.
    """
    assert numeric_or_text_match("1000000.5", ["1000000.0"]) == 1.0   # slack = 1.0
    assert numeric_or_text_match("1000002.0", ["1000000.0"]) == 0.0
    assert numeric_or_text_match("0.2800005", ["0.28"]) == 1.0        # slack = 1e-6
    assert numeric_or_text_match("0.280002", ["0.28"]) == 0.0


def test_boolean_match_commits_to_an_exact_yes_or_no():
    assert boolean_match("YES", ["yes"]) == 1.0
    assert boolean_match("  no  ", ["yes", "no"]) == 1.0       # best gold wins
    assert boolean_match("yes", ["no"]) == 0.0
    assert boolean_match("probably yes", ["yes"]) == 0.0       # no hedging credit
    assert boolean_match(None, ["yes"]) == 0.0


@pytest.mark.parametrize("answer_type,pred,expected_metric", [
    ("point", (0.5, 0.5), "click_in_bbox"),
    ("bbox", (0.4, 0.4, 0.6, 0.6), "bbox_centre_in_gold"),
    ("boolean", "yes", "exact_yes_no"),
    ("choice", "B", "multiple_choice"),
    ("count", 4, "exact_count"),
])
def test_score_dispatches_on_answer_type_before_dataset(answer_type, pred, expected_metric):
    """answer_type wins over dataset: a `point` row in any dataset is graded
    click-in-bbox. This is the dispatch order the adapter docstring promises."""
    golds = {"point": [0.4, 0.4, 0.6, 0.6], "bbox": [0.4, 0.4, 0.6, 0.6],
             "boolean": ["yes"], "choice": ["B"], "count": [4]}
    out = score(ex(answer_type, dataset="charxiv", gold=golds[answer_type]), pred)
    assert out["metric"] == expected_metric
    assert out["score"] == 1.0


def test_score_charxiv_splits_strict_from_fuzzy_grading():
    """Strict qids get normalized (numeric-aware) match; fuzzy qids get ANLS,
    and the row says which -- the report separates them rather than pooling."""
    strict = score(ex("span", dataset="charxiv", gold=["0.28"], qid=8), "0.280")
    assert (strict["metric"], strict["grading_confidence"], strict["score"]) == \
        ("normalized_match", "strict", 1.0)

    fuzzy = score(ex("span", dataset="charxiv", gold=["revenues"], qid=1), "revenuex")
    assert fuzzy["metric"] == "anls"
    assert fuzzy["grading_confidence"] == "fuzzy"
    assert fuzzy["score"] == pytest.approx(0.875)


def test_score_point_reports_centre_distance_alongside_the_hit():
    """The binary hit cannot distinguish a near miss from a click in the far
    corner; centre_distance is what the failure-mode pass reads."""
    out = score(ex("point", gold=[0.4, 0.4, 0.6, 0.6]), (1.0, 0.5))
    assert out["score"] == 0.0
    assert out["center_distance"] == pytest.approx(0.5)


# =============================================================================
# A2. STATS -- confidence intervals, quantiles, the coarse grid
# =============================================================================
"""Wilson, quantiles and the grid. The one thing worth stating loudly: this
`wilson` does NOT clamp its endpoints, and that is intentional -- the report
was built from its unclamped output.
"""


def test_wilson_at_k_zero_k_equals_n_and_n_zero():
    lo, hi = wilson(0, 20)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0 < hi < 0.2

    lo, hi = wilson(20, 20)
    assert 0.8 < lo < 1.0 and hi == pytest.approx(1.0, abs=1e-12)

    # No observations: maximally uninformative, never a crash or a nan.
    assert wilson(0, 0) == (0.0, 1.0)


def test_wilson_does_not_clamp_to_zero_one():
    """CURRENT BEHAVIOUR, asserted rather than fixed.

    A real run produced a lower bound of about -3e-17 at k=0, and an upper
    bound a shade over 1.0 at k=n. The published artifacts contain those
    values. Clamping here would be a defensible change to make deliberately,
    but it must not happen by accident -- so the *absence* of clamping is the
    assertion. `eval.wilson_of_values` is the clamped variant; see subject B.
    """
    over = [(k, n) for n in range(2, 60) for k in (0, n)
            if wilson(k, n)[0] < 0.0 or wilson(k, n)[1] > 1.0]
    assert over, "wilson() appears to have started clamping; see eval.wilson_of_values"
    # ...and the excursions are float dust, not a real error.
    for k, n in over:
        lo, hi = wilson(k, n)
        assert lo > -1e-9 and hi < 1 + 1e-9


def test_wilson_interval_always_brackets_the_point_estimate():
    for n in (1, 3, 7, 40, 411):
        for k in (0, 1, n // 2, n):
            lo, hi = wilson(k, n)
            assert lo <= k / n <= hi, (k, n)


def test_quantiles_of_an_empty_input_is_an_empty_list():
    assert quantiles([], lambda r: r["v"], k=5) == []


def test_quantiles_with_fewer_rows_than_bins_never_returns_an_empty_bin():
    """A thin cell must report the bins it has, not padding that renders as
    a zero-height bar."""
    bins = quantiles([{"v": 1}], lambda r: r["v"], k=5)
    assert len(bins) == 1
    assert bins[0][0] == bins[0][1] == 1


def test_quantiles_bins_partition_the_rows_with_no_overlap():
    rows = [{"v": i} for i in range(37)]
    bins = quantiles(rows, lambda r: r["v"], k=5)
    seen = [r["v"] for _, _, chunk in bins for r in chunk]
    assert sorted(seen) == list(range(37))
    assert len(seen) == len(set(seen))


def test_quantiles_keeps_a_zero_key_but_drops_a_none_key():
    """0 is a legitimate area fraction; None means "not measured". Treating the
    two alike would sort unmeasured rows into the smallest-target bin."""
    bins = quantiles([{"v": 0}, {"v": None}, {"v": 5}], lambda r: r["v"], k=2)
    assert sorted(r["v"] for _, _, c in bins for r in c) == [0, 5]


@pytest.mark.parametrize("g", [2, 3, 4, 8, 16])
def test_cell_of_clamps_the_far_edge_at_every_grid_size(g):
    """x=1.0 is in-range for a normalized coordinate and must land in the last
    cell, not index g (which would be an off-grid cell nothing else produces)."""
    assert cell_of(1.0, 1.0, g) == (g - 1, g - 1)
    assert cell_of(0.0, 0.0, g) == (0, 0)
    assert cell_of(0.999999, 0.999999, g) == (g - 1, g - 1)


def test_cell_of_boundaries_belong_to_the_higher_cell():
    """int() truncation, so an exact cell boundary rounds up, consistently."""
    assert cell_of(0.25, 0.25, 4) == (1, 1)
    assert cell_of(0.2499999, 0.2499999, 4) == (0, 0)


def test_centre_cell_of_a_box_straddling_a_boundary_follows_the_centre():
    # centre is (0.26, 0.74): right of the 0.25 line, above the 0.75 line
    assert centre_cell([0.02, 0.48, 0.50, 1.00], 4) == (1, 2)


def test_bbox_cells_over_a_full_grid_and_a_partial_span():
    """A box covering the whole plane touches every cell; a box inside one
    column touches only that column."""
    assert bbox_cells([0.0, 0.0, 1.0, 1.0], 3) == {(i, j) for i in range(3) for j in range(3)}
    assert bbox_cells([0.35, 0.05, 0.65, 0.95], 3) == {(1, 0), (1, 1), (1, 2)}


def test_bbox_cells_is_a_superset_of_centre_cell_on_a_stress_grid():
    """The lenient precision metric must never be stricter than the strict one.
    Checked over a deterministic sweep rather than one hand-picked box."""
    for i in range(30):
        x0, y0 = i / 60, (29 - i) / 60
        box = [x0, y0, x0 + 0.31, y0 + 0.07]
        for g in (2, 4, 8, 16):
            assert centre_cell(box, g) in bbox_cells(box, g), (box, g)


def test_is_na_matches_a_substring_but_is_not_applicable_matches_the_whole_gold():
    """Two near-identical helpers with deliberately different semantics.

    `is_na(v)` is a substring test used on free-text values; `is_not_applicable(ex)`
    compares the *whole* first gold. Unifying them would reclassify any CharXiv
    answer that merely mentions the phrase as an inapplicable question.
    """
    phrase = "The answer is not applicable here"
    assert is_na(phrase) is True
    assert is_not_applicable(ex("span", gold=[phrase])) is False
    assert is_not_applicable(ex("span", gold=["  Not Applicable "])) is True
    # Degenerate golds must not raise.
    assert is_not_applicable(ex("span", gold=[])) is False
    assert is_not_applicable(ex("span", gold=None)) is False


# =============================================================================
# A3. ADAPTERS -- one Example is one scoreable question; gold boxes normalized
# =============================================================================
"""The adapter contract, against every dataset whose manifest is committed.

`test_all.py` checks `svg_localization` alone. These run over each shipped
dataset, and pin the two promises the docstring makes explicitly: an Example is
a *question* rather than a manifest row, and a gold box is always [0,1] as
(x0, y0, x1, y1) no matter how the source encoded it.
"""


@pytest.fixture(scope="module")
def shipped_rows():
    return {ds: _load(ds) for ds in SHIPPED}


def _manifest_line_count(dataset: str) -> int:
    # svg_counting / svg_word_mc read a manifest nested under svg_localization.
    candidates = [ROOT / "data" / dataset / "manifest.jsonl",
                  ROOT / "data" / "svg_localization" / dataset.replace("svg_", "") / "manifest.jsonl"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        pytest.skip(f"no committed manifest for {dataset}")
    return sum(1 for line in path.read_text().splitlines() if line.strip())


@pytest.mark.parametrize("dataset", SHIPPED)
def test_uids_are_unique_within_every_shipped_dataset(dataset, shipped_rows):
    """The runner resumes by uid and the analysis layer joins on it. A collision
    silently drops one of the two questions from every downstream number."""
    uids = [e.uid for e in shipped_rows[dataset]]
    dupes = [u for u, c in collections.Counter(uids).items() if c > 1]
    assert not dupes, f"{dataset}: duplicate uids {dupes[:5]}"


@pytest.mark.parametrize("dataset", SHIPPED)
def test_answer_type_is_one_the_prompt_builder_and_scorer_both_understand(dataset, shipped_rows):
    """`answer_type` drives prompt construction *and* scoring. A type with a
    schema but no scorer branch (or the reverse) silently produces zeros."""
    for e in shipped_rows[dataset]:
        assert e.answer_type in SCHEMAS, f"{e.uid}: no response schema for {e.answer_type!r}"


@pytest.mark.parametrize("dataset", SHIPPED)
def test_feeding_the_gold_back_as_the_prediction_scores_one(dataset, shipped_rows):
    """The end-to-end adapter/scorer handshake, over every shipped row.

    If a gold is stored in a shape the scorer does not expect -- an int where a
    string is compared, a bbox in the wrong order, a choice stored as its text
    instead of its letter -- this is where it shows, rather than as a suspiciously
    low accuracy in a report.
    """
    for e in shipped_rows[dataset]:
        if e.answer_type == "point":
            x0, y0, x1, y1 = e.gold
            pred = ((x0 + x1) / 2, (y0 + y1) / 2)
        else:
            assert e.gold, f"{e.uid}: empty gold"
            pred = e.gold[0]
        assert score(e, pred)["score"] == 1.0, f"{e.uid}: gold does not score itself"


@pytest.mark.parametrize("dataset", SHIPPED)
def test_gold_point_boxes_are_normalized_ordered_and_non_degenerate(dataset, shipped_rows):
    """The bbox contract, stated as one assertion per promise so a failure names
    which promise broke."""
    points = [e for e in shipped_rows[dataset] if e.answer_type == "point"]
    if not points:
        pytest.skip(f"{dataset} has no point questions")
    for e in points:
        assert len(e.gold) == 4, f"{e.uid}: gold is not a 4-tuple"
        x0, y0, x1, y1 = e.gold
        assert all(isinstance(v, float) for v in e.gold), f"{e.uid}: gold not float"
        assert 0.0 <= x0 <= 1.0 and 0.0 <= x1 <= 1.0, f"{e.uid}: x outside [0,1]: {e.gold}"
        assert 0.0 <= y0 <= 1.0 and 0.0 <= y1 <= 1.0, f"{e.uid}: y outside [0,1]: {e.gold}"
        assert x0 < x1, f"{e.uid}: x0 >= x1 ({e.gold}) -- xywh encoding leaking through?"
        assert y0 < y1, f"{e.uid}: y0 >= y1 ({e.gold})"
        assert (x1 - x0) * (y1 - y0) > 0.0, f"{e.uid}: zero-area target is unhittable"


@pytest.mark.parametrize("dataset", SHIPPED)
def test_image_paths_resolve(dataset, shipped_rows):
    """A manifest pointing at missing files scores 0% and reads as a capability
    result. Sampled, because some of these sets ship thousands of rows."""
    rows = shipped_rows[dataset]
    for e in rows[::max(1, len(rows) // 40)]:
        assert e.images, f"{e.uid}: no images"
        for img in e.images:
            assert (ROOT / img).is_file(), f"{e.uid}: missing image {img}"


def test_one_example_is_one_question_not_one_manifest_row():
    """CharXiv ships four descriptive questions plus one reasoning question per
    figure, so its question count is 5x its row count.

    This is the promise the adapters docstring leads with, and the one that a
    "tidy up the loop" refactor breaks by yielding once per row.
    """
    n_rows = _manifest_line_count("charxiv")
    rows = _load("charxiv")
    assert len(rows) == 5 * n_rows
    splits = collections.Counter(e.meta["split"] for e in rows)
    assert splits["descriptive"] == 4 * n_rows
    assert splits["reasoning"] == n_rows
    # ...and each figure's five questions share one image.
    per_image = collections.Counter(e.images[0] for e in rows)
    assert set(per_image.values()) == {5}


@pytest.mark.parametrize("dataset", SHIPPED)
def test_every_shipped_row_maps_to_a_primitive_or_explicitly_to_none(dataset, shipped_rows):
    """`primitive_for` must return a *known* primitive or an honest (None, None).
    A typo'd primitive name would silently create a new row in the headline
    matrix with one dataset in it."""
    for e in shipped_rows[dataset]:
        prim, prov = primitive_for(e)
        assert (prim is None) == (prov is None)
        if prim is not None:
            assert prim in PRIMITIVES, f"{e.uid}: unknown primitive {prim!r}"
            assert prov in ("construction", "dataset_label"), f"{e.uid}: {prov!r}"


def test_norm_bbox_converts_pixels_and_flags_a_clamp():
    """The ScreenSpot-Pro path (absolute pixels + img_size), tested directly
    because that manifest is downloaded rather than committed.

    Its clamp flag is the audit trail for the one upstream annotation that runs
    a pixel past the top edge: clamped rows stay visible instead of being
    silently corrected.
    """
    box, clamped = core._norm_bbox(100, 50, 300, 150, 1000.0, 500.0)
    assert box == [0.1, 0.1, 0.3, 0.3]
    assert clamped is False

    box, clamped = core._norm_bbox(0, -1, 40, 20, 800.0, 600.0)
    assert clamped is True
    assert box[1] == 0.0 and all(0.0 <= v <= 1.0 for v in box)

    # The ScreenSpot path passes w = h = 1.0: already-normalized input is a
    # no-op, which is what makes one code path safe for two encodings.
    box, clamped = core._norm_bbox(0.2, 0.4, 0.6, 0.8, 1.0, 1.0)
    assert box == [0.2, 0.4, 0.6, 0.8] and clamped is False


# =============================================================================
# A4. PROMPTS -- request construction and response parsing, by answer_type
# =============================================================================
"""What actually gets sent. The load-bearing rule is negative: CharXiv
descriptive questions must go through UNTOUCHED, because those templates carry
their own answer-format rules and adding ours changes the task.
"""


def test_prompt_text_prefixes_the_right_instruction_for_each_answer_type():
    assert prompt_text(ex("point", question="the Save icon")).endswith("Element: the Save icon")
    assert prompt_text(ex("point")).startswith(core.POINT_INSTRUCTION[:20])
    assert prompt_text(ex("boolean")).startswith(core.BOOLEAN_INSTRUCTION)
    assert prompt_text(ex("count")).startswith(core.COUNT_INSTRUCTION)
    assert prompt_text(ex("span")) == "Q?\n\n" + SPAN_INSTRUCTION


def test_prompt_text_renders_choice_options_as_lettered_lines():
    """The schema constrains the answer to A-D, so the options must be
    presented in exactly that order or the letter means nothing."""
    text = prompt_text(ex("choice", question="Which word appears?",
                          options=["alpha", "beta", "gamma", "delta"]))
    assert text.endswith("A. alpha\nB. beta\nC. gamma\nD. delta")
    assert text.startswith(core.CHOICE_INSTRUCTION)


def test_prompt_text_pairs_only_as_many_options_as_letters():
    """zip() truncation: a malformed 5-option row renders four lines rather
    than raising, and a 2-option row renders two."""
    assert prompt_text(ex("choice", options=["a", "b"])).endswith("A. a\nB. b")
    assert prompt_text(ex("choice", options=list("abcde"))).endswith("A. a\nB. b\nC. c\nD. d")


def test_charxiv_descriptive_questions_are_sent_untouched():
    """The comparability rule, checked against the real vendored templates.

    CharXiv's descriptive templates already end in "* Your final answer should
    be ...". Appending SPAN_INSTRUCTION would change the task and make the
    score incomparable to CharXiv's published setup, so the prompt must be
    byte-identical to the question.
    """
    rows = _load("charxiv")
    desc = [e for e in rows if e.meta["split"] == "descriptive"]
    assert desc
    for e in desc:
        assert prompt_text(e) == e.question, f"{e.uid}: descriptive prompt was modified"
        assert SPAN_INSTRUCTION not in prompt_text(e)
        # every vendored template carries its own bulleted answer-format rules
        assert "\n    * " in e.question, \
            f"{e.uid}: vendored answer-format rule missing from the template"


def test_charxiv_reasoning_questions_do_get_our_span_instruction_appended():
    """CURRENT BEHAVIOUR, and worth a second look.

    The reasoning question is wrapped by CharXiv's own `reasoning_question()`
    and therefore *also* arrives carrying vendored answer-format rules -- but
    `prompt_text` exempts only the descriptive split, so SPAN_INSTRUCTION is
    appended on top of them. That is the same comparability concern the
    descriptive exemption exists to avoid. Pinned as-is so the behaviour cannot
    drift silently in either direction.
    """
    rows = _load("charxiv")
    reas = [e for e in rows if e.meta["split"] == "reasoning"]
    assert reas
    e = reas[0]
    assert "* Your final answer" in e.question          # vendored wrapper is present
    assert prompt_text(e) == f"{e.question}\n\n{SPAN_INSTRUCTION}"
    assert all(prompt_text(r).endswith(SPAN_INSTRUCTION) for r in reas)


def test_svg_localization_span_questions_are_self_contained():
    """They state their own answer format in the generated question text, so
    the harness must not double-instruct them either."""
    rows = _load("svg_localization")
    e = next(r for r in rows if r.answer_type == "span")
    assert prompt_text(e) == e.question


def test_prompt_override_replaces_the_whole_block_for_every_answer_type():
    """The ablation lever: vary the wording while image encoding, schema, model
    and thinking budget stay byte-identical."""
    for at in ("span", "point", "boolean", "count", "choice"):
        assert prompt_text(ex(at, prompt_override="ONLY THIS", options=list("abcd"))) == "ONLY THIS"


def test_parse_response_converts_each_schema_payload_to_scorer_input():
    """Coordinates come back in a 0-1000 space and must land in [0,1]; count is
    an int; boolean is lowercased; choice is uppercased."""
    assert parse_response(ex("point"), '{"x": 250, "y": 500}') == (0.25, 0.5)
    assert parse_response(ex("bbox"), '{"x0":0,"y0":100,"x1":500,"y1":1000}') == (0.0, 0.1, 0.5, 1.0)
    assert parse_response(ex("count"), '{"answer": 7}') == 7
    assert parse_response(ex("boolean"), '{"answer": "YES"}') == "yes"
    assert parse_response(ex("choice"), '{"answer": "b"}') == "B"
    assert parse_response(ex("span"), '{"answer": "  Total Revenue "}') == "  Total Revenue "


def test_parse_response_normalizes_the_boolean_and_choice_surface_form():
    """The schema enums make these unnecessary in the happy path; they matter
    for the ablation arms, which answer through free text."""
    assert parse_response(ex("boolean"), '{"answer": " No "}') == "no"
    assert parse_response(ex("choice"), '{"answer": " c "}') == "C"


@pytest.mark.parametrize("raw", ['not json at all', '', '{"x": 1}', '{}', 'null'])
def test_parse_response_raises_on_a_malformed_payload_rather_than_guessing(raw):
    """The runner catches this and records `parse_error`, and the analysis layer
    counts those rows separately instead of scoring them as wrong. Returning a
    fallback value here would turn an API failure into a capability result.
    """
    with pytest.raises(Exception):
        parse_response(ex("point"), raw)


def test_encode_image_respects_max_edge_and_reports_the_size_as_sent():
    """The resolution ablation's only lever. `(width, height) as sent` is what
    the analysis correlates failures against, so it must be the post-resize
    size, and the flag must say whether a resize happened."""
    large = str(ROOT / "data/svg_localization/images/g0000_large.png")
    _b64, media, size, shrunk = encode_image(large, max_edge=core.HAIKU_MAX_EDGE)
    assert max(size) == core.HAIKU_MAX_EDGE
    assert shrunk is True
    assert media == "image/jpeg"
    # Aspect ratio preserved: 3000x1900 -> 1568x993
    assert size == (1568, round(1900 * core.HAIKU_MAX_EDGE / 3000))


def test_encode_image_sends_the_original_untouched_when_it_fits():
    """max_edge=None is the default for the main study: send native so the
    downscale ablation has something to compare against."""
    small = str(ROOT / "data/svg_localization/images/g0000_small.png")
    b64, media, size, shrunk = encode_image(small)
    assert size == (900, 570)
    assert shrunk is False
    assert media == "image/png"          # no re-encode
    import base64 as _b
    assert _b.b64decode(b64) == (ROOT / "data/svg_localization/images/g0000_small.png").read_bytes()


def test_encode_image_with_max_edge_re_encodes_to_jpeg_even_when_no_resize_is_needed():
    """CURRENT BEHAVIOUR, and a confound worth knowing about.

    Passing `max_edge` routes through the resize branch unconditionally, so an
    image already under the cap is still re-encoded as JPEG q=90 while the
    native run sends the original PNG. `was_downscaled` correctly reports False
    (the pixel dimensions did not change), but a max_edge ablation is therefore
    not a pure resolution contrast on already-small rungs -- it also changes the
    codec.
    """
    small = str(ROOT / "data/svg_localization/images/g0000_small.png")
    _b64, media, size, shrunk = encode_image(small, max_edge=core.HAIKU_MAX_EDGE)
    assert size == (900, 570) and shrunk is False
    assert media == "image/jpeg"


def test_encode_image_never_exceeds_the_api_ingestion_limits():
    """API_MAX_DIM / API_MAX_B64_BYTES reject the request outright, before the
    model sees anything -- and skipping the offenders would drop the largest,
    most interesting images from the eval."""
    large = str(ROOT / "data/svg_localization/images/g0000_large.png")
    for max_edge in (None, core.HAIKU_MAX_EDGE, 320):
        b64, _media, size, _shrunk = encode_image(large, max_edge=max_edge)
        assert max(size) <= core.API_MAX_DIM
        assert len(b64) <= core.API_MAX_B64_BYTES


# =============================================================================
# A5. TAXONOMY AND FAILURE MODES
# =============================================================================
"""The primitive spine, and the classifier that says *why* an answer was wrong.

The name collision guard is the important one here: `PRIMITIVE_LABELS` and
`FAILURE_MODE_LABELS` were both called `LABELS` before the merge, and
`aggregate.py` iterates one of them to enumerate primitives. Swapping them
produces a wrong report with no error.
"""


def test_primitive_and_failure_mode_label_sets_cannot_be_swapped():
    """Each dict covers exactly its own domain, and the two domains are
    disjoint -- so a future edit that binds the wrong one fails loudly here
    instead of rendering a report full of failure-mode names."""
    assert set(PRIMITIVE_LABELS) == set(PRIMITIVES)
    assert set(FAILURE_MODE_LABELS) == set(MODES)
    assert set(PRIMITIVE_LABELS).isdisjoint(FAILURE_MODE_LABELS)
    assert all(isinstance(v, str) and v for v in PRIMITIVE_LABELS.values())
    assert all(isinstance(v, str) and v for v in FAILURE_MODE_LABELS.values())


def test_the_historical_LABELS_alias_still_points_at_the_taxonomy_dict():
    """`aggregate.py` does `for prim in LABELS`. Rebinding this alias to the
    failure-mode dict silently produces an empty primitive matrix."""
    assert LABELS is PRIMITIVE_LABELS


@pytest.mark.parametrize("qid,expected", sorted(CHARXIV_QID.items()))
def test_primitive_for_maps_every_charxiv_descriptive_template(qid, expected):
    """All 19 templates are mapped by construction; an unmapped one would drop
    silently out of the headline matrix."""
    prim, prov = primitive_for(ex("span", dataset="charxiv", split="descriptive", qid=qid))
    assert (prim, prov) == (expected, "construction")


def test_charxiv_qid_map_is_complete_and_uses_only_known_primitives():
    assert set(CHARXIV_QID) == set(range(1, 20))
    assert set(CHARXIV_QID.values()) <= set(PRIMITIVES)


def test_primitive_for_the_remaining_mapped_cases():
    # reasoning split: composition, by construction
    assert primitive_for(ex("span", dataset="charxiv", split="reasoning")) == \
        ("composition", "construction")
    # ScreenSpot-Pro is localization by definition
    assert primitive_for(ex("point", dataset="screenspot_pro")) == \
        ("localization_point", "construction")
    # AI2D: only the label-reference half is a perception measurement
    assert primitive_for(ex("choice", dataset="ai2d", qtype="label_reference")) == \
        ("localization_read", "construction")
    assert primitive_for(ex("choice", dataset="ai2d", qtype="diagram_reasoning")) == (None, None)
    assert set(AI2D_QTYPE.values()) <= set(PRIMITIVES)
    # InfographicVQA: the dataset's own annotation, never a guess
    assert primitive_for(ex("span", dataset="infographicvqa", operation=["counting"])) == \
        ("counting", "dataset_label")
    assert primitive_for(ex("span", dataset="infographicvqa", operation=["comparison"])) == \
        ("comparison", "dataset_label")


def test_primitive_for_declines_rather_than_guessing():
    """There is deliberately no keyword-guessed tier. Anything unmapped comes
    back (None, None) and is reported separately, not folded into a primitive."""
    assert primitive_for(ex("span", dataset="charxiv", split="descriptive")) == (None, None)
    assert primitive_for(ex("span", dataset="infographicvqa")) == (None, None)
    assert primitive_for(ex("span", dataset="infographicvqa", operation=["visual/layout"])) == (None, None)
    assert primitive_for(ex("point", dataset="svg_localization")) == (None, None)
    assert primitive_for(ex("point", dataset="screenspot")) == (None, None)
    assert primitive_for(ex("span", dataset="a_dataset_added_tomorrow")) == (None, None)


@pytest.mark.parametrize("gold,pred,mode", [
    (["A, B, C"], "C, B, A", "order_only"),        # same set, wrong sequence
    (["A, B, C"], "A, B, C, D", "extra_items"),    # superset
    (["A, B, C"], "A, B", "missing_items"),        # strict subset
    (["A, B, C"], "A, B, X", "partial_overlap"),   # some right, some wrong
    (["A, B, C"], "a, b, c", "format_only"),       # same list, different case
    (["310.5"], "310.5 million", "format_only"),   # same number, extra unit
    (["Revenue"], "revenue ", "format_only"),      # same string, different case
    (["28%"], "28", "format_only"),                # unit stripped
])
def test_classify_decides_the_deterministic_cases(gold, pred, mode):
    assert classify(gold, pred) == mode


@pytest.mark.parametrize("gold,pred", [
    (["A, B, C"], "X, Y"),          # disjoint lists: is it wrong, or synonyms?
    (["42"], "forty-two"),          # same value, but only a reader knows that
    (["increasing"], "goes up"),    # paraphrase
])
def test_classify_leaves_semantic_judgement_to_the_llm_pass(gold, pred):
    """`unclassified` is the honest answer, not a fallback. Guessing
    `wrong_value` here would report a perception failure for what may be a
    phrasing difference."""
    assert classify(gold, pred) == "unclassified"


def test_classify_never_emits_the_modes_that_belong_to_other_passes():
    """`wrong_value` is a judge verdict, and the point/choice modes come from
    `classify_point` and the MC scorer. `classify` must produce none of them,
    or a deterministic string comparison would masquerade as a judgement.
    """
    produced = set()
    for gold, pred in [(["A, B, C"], "C, B, A"), (["A, B, C"], "A, B, C, D"),
                       (["A, B, C"], "A, B"), (["A, B, C"], "A, B, X"),
                       (["310.5"], "310.5 million"), (["A"], "A"),
                       (["A, B, C"], "X, Y"), (["42"], "forty-two"), ([""], "")]:
        produced.add(classify(gold, pred))
    assert produced <= set(MODES)
    assert produced.isdisjoint(core.POINT_MODES)
    assert produced.isdisjoint(core.CHOICE_MODES)
    assert "wrong_value" not in produced


def test_classify_accepts_a_bare_gold_as_well_as_a_list():
    assert classify("310.5", "310.5 million") == "format_only"


def test_classify_point_bands_a_miss_by_per_axis_distance():
    """L-infinity, not Euclidean: <=0.10 on both axes is a near miss, >0.25 on
    either is the wrong region, and everything between is moderate. (`eval.band`
    uses a Euclidean distance instead -- see subject B.)

    The two boxes are chosen so the offsets land on the boundary *exactly* in
    binary floating point: 0.2 - 0.1 == 0.10 and 0.75 - 0.5 == 0.25. Both
    boundaries are inclusive-below, so an offset sitting on the line stays in
    the gentler band.
    """
    box = [0.00, 0.00, 0.20, 0.20]                     # centre (0.10, 0.10)
    assert classify_point(box, (0.20, 0.10)) == "near_miss"        # dx = 0.10 exactly
    assert classify_point(box, (0.20, 0.20)) == "near_miss"        # both axes at 0.10
    assert classify_point(box, (0.2000001, 0.10)) == "moderate_miss"

    box = [0.20, 0.20, 0.80, 0.80]                     # centre (0.50, 0.50)
    assert classify_point(box, (0.60, 0.50)) == "near_miss"
    assert classify_point(box, (0.70, 0.50)) == "moderate_miss"    # dx = 0.20
    assert classify_point(box, (0.75, 0.50)) == "moderate_miss"    # dx = 0.25 exactly
    assert classify_point(box, (0.7501, 0.50)) == "wrong_region"   # dx > 0.25
    assert classify_point(box, (0.50, 0.99)) == "wrong_region"     # one axis is enough


def test_classify_point_on_an_unusable_prediction_is_unclassified():
    assert classify_point([0.4, 0.4, 0.5, 0.5], "somewhere on the left") == "unclassified"
    assert classify_point([0.4, 0.4, 0.5, 0.5], None) == "unclassified"
    assert classify_point([], (0.5, 0.5)) == "unclassified"


def test_summarize_counts_missing_modes_as_unclassified_and_drops_empty_ones():
    out = summarize_failure_modes([{"failure_mode": "near_miss"},
                                   {"failure_mode": "near_miss"},
                                   {"failure_mode": "format_only"},
                                   {}])
    assert out == {"near_miss": 2, "format_only": 1, "unclassified": 1}
    assert summarize_failure_modes([]) == {}


def test_every_mode_summarize_can_report_has_a_human_label():
    """`judge.py` renders `LABELS.get(mode, mode)`; a mode with no label prints
    a raw identifier into the report."""
    for mode in MODES:
        assert mode in FAILURE_MODE_LABELS


# =============================================================================
# A6. MERMAID -- FlowLearn ground truth -> graph
# =============================================================================
"""Golds are derived from the parsed Mermaid rather than trusted from the
shipped QA labels, so the parser *is* the ground truth for that dataset.
"""

MERMAID = """```mermaid
flowchart LR
entity0(alpha)
entity1[beta]
entity2{gamma}
entity3((delta))
entity0 --> entity1
entity1 ==> entity2
entity2 -..-> entity3
entity0 --- entity3
```"""


@pytest.fixture(scope="module")
def graph():
    return parse_mermaid(MERMAID)


def test_parse_mermaid_reads_direction_all_four_node_shapes_and_all_edge_styles(graph):
    assert graph.direction == "LR"
    assert graph.labels == {"entity0": "alpha", "entity1": "beta",
                            "entity2": "gamma", "entity3": "delta"}
    assert graph.n_nodes == 4 and graph.n_edges == 4
    assert graph.edges == (("entity0", "entity1", "solid"),
                           ("entity1", "entity2", "thick"),
                           ("entity2", "entity3", "dotted"),
                           ("entity0", "entity3", "open"))


def test_parse_mermaid_on_empty_or_unparseable_input_yields_an_empty_graph():
    """The adapter skips figures with no labels (`if not g.labels: continue`),
    so this must be an empty graph rather than an exception."""
    for src in ("", None, "not mermaid at all\njust prose"):
        g = parse_mermaid(src)
        assert g.n_nodes == 0 and g.n_edges == 0
        assert g.direction == "TB"      # documented default


def test_by_label_inverts_the_id_map(graph):
    assert by_label(graph) == {"alpha": "entity0", "beta": "entity1",
                               "gamma": "entity2", "delta": "entity3"}


def test_has_edge_is_directed_by_default_and_none_for_an_absent_label(graph):
    assert has_edge(graph, "alpha", "beta") is True
    assert has_edge(graph, "beta", "alpha") is False                  # direction matters
    assert has_edge(graph, "beta", "alpha", directed=False) is True
    assert has_edge(graph, "alpha", "gamma") is False                 # genuinely unconnected
    # None, not False: "the label is not in this figure" is a different answer
    # from "there is no arrow", and the adapter tells them apart to build the
    # phantom_node false_mode.
    assert has_edge(graph, "alpha", "nonexistent") is None
    assert has_edge(graph, "nonexistent", "alpha") is None


def test_edge_style_reports_the_arrow_style_in_either_direction(graph):
    assert edge_style(graph, "alpha", "beta") == "solid"
    assert edge_style(graph, "beta", "alpha") == "solid"     # symmetric lookup
    assert edge_style(graph, "beta", "gamma") == "thick"
    assert edge_style(graph, "gamma", "delta") == "dotted"
    assert edge_style(graph, "alpha", "delta") == "open"     # --- , no arrowhead
    assert edge_style(graph, "alpha", "gamma") is None       # no edge
    assert edge_style(graph, "alpha", "nonexistent") is None


def test_hops_is_undirected_by_default(graph):
    assert hops(graph, "alpha", "beta") == 1
    assert hops(graph, "alpha", "gamma") == 2
    assert hops(graph, "gamma", "alpha") == 2               # undirected
    assert hops(graph, "alpha", "alpha") == 0               # a node to itself
    assert hops(graph, "alpha", "nonexistent") is None


def test_hops_directed_returns_none_when_the_arrows_point_the_other_way(graph):
    assert hops(graph, "alpha", "gamma", directed=True) == 2
    assert hops(graph, "gamma", "alpha", directed=True) is None


def test_graph_f1_against_a_hand_computed_expected_value(graph):
    """A prediction that finds 3 of 4 nodes and 1 of 4 edges, and invents one
    edge to a node it never declared.

    nodes: tp=3, precision 3/3, recall 3/4 -> F1 = 6/7
    edges: tp=1, precision 1/2, recall 1/4 -> F1 = 1/3
    `score` is the edge F1, because edges are the thing being tested.
    """
    pred = parse_mermaid("flowchart LR\n"
                         "n0(alpha)\nn1(beta)\nn2(gamma)\n"
                         "n0 --> n1\n"
                         "n1 --> n9\n")
    out = graph_f1(graph, pred)
    assert out["node_f1"] == pytest.approx(6 / 7)
    assert out["edge_f1"] == pytest.approx(1 / 3)
    assert out["score"] == out["edge_f1"]
    assert out["exact"] == 0.0


def test_graph_f1_compares_labels_not_ids(graph):
    """The model has no reason to reproduce `entity0`; only the human labels are
    visible in the figure."""
    same = parse_mermaid("flowchart LR\n"
                         "a(alpha)\nb(beta)\nc(gamma)\nd(delta)\n"
                         "a --> b\nb ==> c\nc -..-> d\na --- d\n")
    out = graph_f1(graph, same)
    assert out == {"node_f1": 1.0, "edge_f1": 1.0, "exact": 1.0, "score": 1.0}


def test_graph_f1_of_two_empty_graphs_is_one_and_a_blank_prediction_is_zero():
    empty = parse_mermaid("")
    assert graph_f1(empty, empty)["score"] == 1.0
    assert graph_f1(parse_mermaid(MERMAID), empty) == {
        "node_f1": 0.0, "edge_f1": 0.0, "exact": 0.0, "score": 0.0}


# =============================================================================
# A7. SAMPLING -- stratify by the cell you intend to report
# =============================================================================
"""The pilot sampled CharXiv by figure and produced per-question-type counts of
3 to 16, which rendered as confident bars. `stratify` is the fix, and it has to
be deterministic so a resumed run selects the same questions.
"""


def _cells():
    """A pool deliberately dominated by one cell, like CharXiv's qid mix."""
    def mk(i, cell):
        return ex("span", question=f"q{i}", cell=cell)
    return ([mk(i, "big") for i in range(100)]
            + [mk(100 + i, "mid") for i in range(20)]
            + [mk(200 + i, "thin") for i in range(3)])


def test_stratify_takes_per_cell_from_every_cell_and_reports_the_realised_n():
    """Cells smaller than `per_cell` contribute their whole pool, and the pool
    size comes back so under-filled cells are reported rather than shipped."""
    out, realised = stratify(_cells(), lambda e: e.meta["cell"], per_cell=10, seed=0)
    assert realised == {"big": (10, 100), "mid": (10, 20), "thin": (3, 3)}
    assert len(out) == 23
    assert all(taken == min(pool, 10) for taken, pool in realised.values())


def test_stratify_is_deterministic_in_both_membership_and_order():
    """A resumed run must select the same questions *and* emit them in the same
    sequence -- the runner writes results in order and resumes by position in
    more than one script."""
    a, ra = stratify(_cells(), lambda e: e.meta["cell"], per_cell=10, seed=0)
    b, rb = stratify(_cells(), lambda e: e.meta["cell"], per_cell=10, seed=0)
    assert [e.question for e in a] == [e.question for e in b]
    assert ra == rb
    # ...and the cells themselves come out in a stable (string-sorted) order.
    assert list(ra) == sorted(ra, key=str)
    assert [e.meta["cell"] for e in a] == ["big"] * 10 + ["mid"] * 10 + ["thin"] * 3


def test_stratify_changes_its_selection_with_the_seed():
    a, _ = stratify(_cells(), lambda e: e.meta["cell"], per_cell=10, seed=0)
    b, _ = stratify(_cells(), lambda e: e.meta["cell"], per_cell=10, seed=1)
    assert [e.question for e in a] != [e.question for e in b]


def test_stratify_balances_the_reported_cell_where_naive_sampling_does_not():
    """The bug this function exists to prevent, on the real CharXiv manifest.

    CharXiv is stratified by question-template id. Drawing the same number of
    questions with `random.sample` over the rows gives per-template counts
    spanning an order of magnitude; stratifying gives exactly `per_cell`
    everywhere it can.
    """
    rows = _load("charxiv")
    key = cell_key_for("charxiv")
    out, realised = stratify(rows, key, per_cell=3, seed=0)

    stratified = collections.Counter(key(e) for e in out)
    assert set(stratified.values()) == {3}, "stratified sample is not flat"
    assert len(realised) == len(stratified) > 1

    naive = collections.Counter(key(e) for e in random.Random(0).sample(rows, len(out)))
    assert max(naive.values()) > min(naive.values()) + 1, \
        "the naive comparison is not exercising the imbalance it is meant to show"


def test_cell_key_for_an_unregistered_dataset_falls_back_to_the_dataset_name():
    """One cell, so an unregistered dataset is sampled as a single pool rather
    than crashing or silently stratifying on nothing."""
    assert cell_key_for("not_a_dataset")(ex("span", dataset="whatever")) == "whatever"


def test_area_bucket_uses_the_size_the_model_actually_receives():
    """Target size is computed after the API's ~1568x882 cap, because that is
    the frame the model sees. A missing measurement buckets as the smallest,
    which keeps an unmeasured row out of the "large targets are fine" cell."""
    assert core._area_bucket(ex("point", target_area_frac=1e-5)) == "<12px"
    assert core._area_bucket(ex("point", target_area_frac=0.01)) == ">=56px"
    assert core._area_bucket(ex("point")) == "<12px"
    assert core._area_bucket(ex("point", target_area_frac=None)) == "<12px"


# #############################################################################
# #                                                                           #
# #   SUBJECT B -- blindspot/eval.py                                          #
# #                                                                           #
# #############################################################################


# =============================================================================
# B1. A ZERO-ROW SCORING RUN MUST ABORT, NOT PUBLISH AN EMPTY SUMMARY
# =============================================================================
"""The regression guard for a bug that shipped.

`cmd_localization` used to score whatever it found and write the summary
unconditionally, exiting 0. A missing or mistyped `--tag` therefore replaced a
good outputs/svgloc/summary.json with an n=0 one and reported success -- silent
data loss, with the report downstream rendering a page of zeros without a
complaint. Both abort paths are covered here, plus a positive control so the
guard cannot regress into "always abort".
"""

GOOD_SUMMARY = '{"headline": "a real result that must survive"}'


@pytest.fixture
def loc_run(tmp_path, monkeypatch):
    """Point the localization analysis at a scratch results/ directory."""
    monkeypatch.setattr(bs_eval, "RESULTS", tmp_path)
    out = tmp_path / "summary.json"
    out.write_text(GOOD_SUMMARY)

    def run(tag, lines=None):
        if lines is not None:
            (tmp_path / f"{bs_eval.DS}__{tag}.jsonl").write_text("\n".join(lines))
        with at_repo_root():
            return bs_eval.main(["localization", "--tag", tag, "--out", str(out)])

    run.out = out
    return run


def test_a_missing_result_file_aborts_without_touching_the_existing_summary(loc_run, capsys):
    rc = loc_run("a-tag-that-was-never-run")
    assert rc != 0
    assert loc_run.out.read_text() == GOOD_SUMMARY
    assert "ABORT" in capsys.readouterr().err


def test_a_results_file_whose_uids_match_no_example_aborts(loc_run, capsys):
    """The subtler half of the bug: the file exists, so the path check passes,
    but every row joins to nothing. That is exactly what a stale tag against a
    regenerated dataset looks like."""
    lines = [json.dumps({"uid": f"svgloc:9999:small:{i:02d}", "pred": [0.5, 0.5]})
             for i in range(5)]
    rc = loc_run("ghost-uids", lines)

    assert rc != 0, "a zero-row scoring run must not exit 0"
    assert loc_run.out.read_text() == GOOD_SUMMARY, "the good summary was overwritten"
    err = capsys.readouterr().err
    assert "ABORT" in err and "nothing scoreable" in err


def test_an_empty_result_file_aborts(loc_run):
    assert loc_run("empty", []) != 0
    assert loc_run.out.read_text() == GOOD_SUMMARY


def test_a_result_file_of_only_unusable_rows_aborts(loc_run, capsys):
    """Null predictions are counted, never scored as wrong -- so a run where
    every request failed also scores zero rows and must abort too."""
    exs = [e for e in _load(bs_eval.DS) if e.answer_type == "point"][:4]
    lines = [json.dumps({"uid": e.uid, "pred": None, "error": "overloaded"}) for e in exs]
    rc = loc_run("all-null", lines)
    assert rc != 0
    assert loc_run.out.read_text() == GOOD_SUMMARY
    assert "4 unusable" in capsys.readouterr().err


def test_a_run_with_real_rows_writes_the_summary_and_exits_zero(loc_run):
    """Positive control: the abort must be triggered by zero scoreable rows,
    not by anything about the scratch directory."""
    rows = _load(bs_eval.DS)
    pts = [e for e in rows if e.answer_type == "point"][:4]
    spans = [e for e in rows if e.answer_type == "span"][:2]
    lines = [json.dumps({"uid": e.uid,
                         "pred": [(e.gold[0] + e.gold[2]) / 2, (e.gold[1] + e.gold[3]) / 2]})
             for e in pts]
    lines += [json.dumps({"uid": e.uid, "pred": e.gold[0]}) for e in spans]

    assert loc_run("real", lines) == 0
    s = json.loads(loc_run.out.read_text())
    assert s["counts"]["point_scored"] == 4
    assert s["counts"]["span_scored"] == 2
    assert s["overall"]["n"] == 4 and s["overall"]["acc"] == 1.0


def test_rows_with_an_unknown_uid_are_dropped_without_appearing_in_any_count(loc_run):
    """CURRENT BEHAVIOUR, and the residue of the same class of bug.

    A row whose uid is not in the manifest is skipped by `load_loc_run` before
    the usable/unusable split, so `counts.unique` exceeds
    `point_scored + span_scored + unusable` with nothing saying so. The abort
    only fires when *everything* fails to join; a partial join -- a stale tag
    against a partly-regenerated dataset -- still publishes.
    """
    pts = [e for e in _load(bs_eval.DS) if e.answer_type == "point"][:3]
    lines = [json.dumps({"uid": e.uid,
                         "pred": [(e.gold[0] + e.gold[2]) / 2, (e.gold[1] + e.gold[3]) / 2]})
             for e in pts]
    lines.append(json.dumps({"uid": "svgloc:9999:small:00", "pred": [0.5, 0.5]}))

    assert loc_run("partial", lines) == 0
    c = json.loads(loc_run.out.read_text())["counts"]
    assert c["unique"] == 4
    assert c["point_scored"] + c["span_scored"] + c["unusable"] == 3


# =============================================================================
# B2. `overall` MUST NOT POOL RUNGS OUTSIDE THE REPORTED SET
# =============================================================================
"""The second fixed bug.

`headline` is built from DERIVED_RUNGS (small, large) but `overall` was built
from *all* rows and labelled "both rungs", so a stray `medium` row made a
three-rung pool wear a two-rung label and `overall.n` stopped equalling the sum
of the headline cells. The shipped manifests do contain `medium` rows for both
derived sets, which is exactly how this happened.
"""


@pytest.fixture
def derived_run(tmp_path, monkeypatch):
    """Write a synthetic result file covering all three rungs, then analyse it."""
    monkeypatch.setattr(bs_eval, "RESULTS", tmp_path)

    def run(dataset, analyse, n_per_rung=(5, 4, 3)):
        with at_repo_root():
            rows = load(dataset)
            by_rung = collections.defaultdict(list)
            for e in rows:
                by_rung[e.meta["resolution"]].append(e)
            picked = {rung: by_rung[rung][:n]
                      for rung, n in zip(("small", "large", "medium"), n_per_rung)}
            lines = [json.dumps({"uid": e.uid, "pred": e.gold[0]})
                     for sel in picked.values() for e in sel]
            (tmp_path / f"{dataset}__t.jsonl").write_text("\n".join(lines))
            return analyse("t"), picked

    return run


@pytest.mark.parametrize("dataset,analyse", [
    ("svg_counting", bs_eval.analyse_counting),
    ("svg_word_mc", bs_eval.analyse_word_mc),
])
def test_overall_covers_exactly_the_rows_headline_reports(dataset, analyse, derived_run):
    out, picked = derived_run(dataset, analyse)

    headline_n = sum(c["n"] for c in out["headline"])
    assert [c["label"] for c in out["headline"]] == list(bs_eval.DERIVED_RUNGS)
    assert out["overall"]["n"] == headline_n, \
        "'both rungs' must not pool a rung the headline does not show"
    assert out["overall"]["k"] == sum(c["k"] for c in out["headline"])


@pytest.mark.parametrize("dataset,analyse", [
    ("svg_counting", bs_eval.analyse_counting),
    ("svg_word_mc", bs_eval.analyse_word_mc),
])
def test_rows_outside_the_reported_rungs_are_surfaced_not_silently_dropped(dataset, analyse, derived_run):
    """The excluded rows get a count of their own, so "5 fewer than I ran" is
    answerable from the artifact instead of requiring a re-run."""
    out, picked = derived_run(dataset, analyse)
    c = out["counts"]

    assert c["outside_reported_rungs"] == len(picked["medium"]) > 0
    assert c["scored"] == out["overall"]["n"] + c["outside_reported_rungs"]


@pytest.mark.parametrize("dataset,analyse", [
    ("svg_counting", bs_eval.analyse_counting),
    ("svg_word_mc", bs_eval.analyse_word_mc),
])
def test_the_per_cut_breakdowns_still_include_the_excluded_rung(dataset, analyse, derived_run):
    """The exclusion is from the *labelled pool* only. Dropping those rows from
    every cut as well would be a different, larger change to the numbers, and
    the CLI's note says they remain."""
    out, _picked = derived_run(dataset, analyse)
    assert sum(c["n"] for c in out["chart_type"]) == out["counts"]["scored"]
    assert sum(c["n"] for c in out["theme"]) == out["counts"]["scored"]


def test_in_derived_rungs_filters_on_the_reported_set():
    rows = [{"meta": {"resolution": r}} for r in
            ("small", "medium", "large", "small", "tiny")]
    kept = bs_eval.in_derived_rungs(rows)
    assert [r["meta"]["resolution"] for r in kept] == ["small", "large", "small"]
    assert bs_eval.in_derived_rungs([]) == []
    assert set(bs_eval.DERIVED_RUNGS) < set(bs_eval.LOC_RUNGS), \
        "the derived sets report a subset of the localization rungs"


# =============================================================================
# B3. TWO WILSONS, DELIBERATELY DIFFERENT
# =============================================================================
"""`core.wilson(k, n)` and `eval.wilson_of_values(vals)` disagree on purpose, in
three ways. Substituting either for the other changes published numbers, so
both behaviours are asserted here rather than left to a docstring.
"""


def test_wilson_of_values_returns_none_on_empty_where_core_wilson_returns_the_full_range():
    """None lets the caller emit a null; (0, 1) is a real, if uninformative,
    interval. An empty cell must not render as "0-100%"."""
    assert bs_eval.wilson_of_values([]) is None
    assert wilson(0, 0) == (0.0, 1.0)


def test_wilson_of_values_clamps_to_zero_one_where_core_wilson_does_not():
    lo_v, _hi_v = bs_eval.wilson_of_values([0.0] * 5)
    lo_k, _hi_k = wilson(0, 5)
    assert lo_v == 0.0
    assert lo_k < 0.0                      # the unclamped float artifact, kept

    _lo_v, hi_v = bs_eval.wilson_of_values([1.0] * 5)
    _lo_k, hi_k = wilson(5, 5)
    assert hi_v == 1.0
    assert hi_k > 1.0


def test_wilson_of_values_accepts_fractional_scores_which_core_wilson_cannot_express():
    """CharXiv scores are ANLS, so a row can be worth 0.875. `wilson(k, n)` has
    no way to represent that."""
    lo, hi = bs_eval.wilson_of_values([0.875, 0.5, 1.0, 0.0])
    assert 0.0 <= lo < 0.59375 < hi <= 1.0


def test_the_two_wilsons_agree_on_binary_data_away_from_the_extremes():
    """Same closed form; the differences are only the three above. If this
    diverges, one of them has been re-derived."""
    for k, n in [(7, 20), (57, 412), (1, 3), (99, 100)]:
        vals = [1.0] * k + [0.0] * (n - k)
        assert bs_eval.wilson_of_values(vals) == pytest.approx(wilson(k, n))


def test_agg_cell_returns_none_for_an_empty_or_all_null_slice():
    """summary.json carries a null cell rather than a zero, so the report can
    say "not measured" instead of "0%"."""
    assert bs_eval.agg_cell([]) is None
    assert bs_eval.agg_cell([{"score": None}, {"score": None}]) is None
    cell = bs_eval.agg_cell([{"score": 1.0}, {"score": 0.0}, {"score": None}])
    assert cell["n"] == 2 and cell["acc"] == 0.5
    assert 0.0 <= cell["ci_lo"] <= 0.5 <= cell["ci_hi"] <= 1.0


# =============================================================================
# B4. PURE HELPERS
# =============================================================================
"""Small functions whose output lands directly in a published JSON artifact."""


def _pt_row(pred, gold, area=0.01):
    return {"pred": pred, "gold": gold,
            "hit": bool(point_in_bbox(pred, gold)),
            "d_box": bs_eval.d_box(pred, gold),
            "d_centre": bs_eval.d_centre(pred, gold),
            "meta": {"target_area_frac": area}}


def test_d_box_is_zero_exactly_when_the_click_is_a_hit():
    """The docstring's claim, checked over a sweep rather than a sample:
    `point_in_bbox` is exactly `d_box == 0`, which is what makes the continuous
    distance the natural companion to the binary metric."""
    box = [0.30, 0.40, 0.55, 0.60]
    for i in range(21):
        for j in range(21):
            pred = (i / 20, j / 20)
            assert (bs_eval.d_box(pred, box) == 0.0) == bool(point_in_bbox(pred, box)), pred


def test_band_is_euclidean_and_its_boundaries_are_asymmetric():
    """`< 0.10` but `<= 0.25`, so 0.10 is a moderate miss and 0.25 still is.
    Note this disagrees with `core.classify_point` at exactly 0.10 -- that one
    is per-axis and `<= .10`. Both are correct for their own study; the point
    is that they are not interchangeable."""
    assert bs_eval.band(0.0) == "near_miss"
    assert bs_eval.band(0.099999) == "near_miss"
    assert bs_eval.band(0.10) == "moderate_miss"
    assert bs_eval.band(0.25) == "moderate_miss"
    assert bs_eval.band(0.2500001) == "wrong_region"
    # the disagreement, made explicit
    assert classify_point([0.4, 0.5, 0.4, 0.5], (0.5, 0.5)) == "near_miss"
    assert bs_eval.band(0.10) != "near_miss"


def test_precision_curve_reports_every_grid_plus_the_exact_hit_box():
    rows = [_pt_row((0.45, 0.45), [0.40, 0.40, 0.60, 0.60])]
    curve = bs_eval.precision_curve(rows)

    assert [c["grid"] for c in curve] == [f"{g}x{g}" for g in bs_eval.GRIDS] + ["exact hit box"]
    for c, g in zip(curve, bs_eval.GRIDS):
        assert c["chance"] == 1 / (g * g)
        assert c["n"] == 1
    # The click is inside the box, so the exact-hit-box row scores 1...
    assert curve[-1]["strict"] == 1.0
    assert curve[-1]["lenient"] is None          # meaningless without a grid
    assert curve[-1]["chance"] == pytest.approx(0.01)   # mean target area share
    # ...while strict cell agreement depends on where the grid lines fall:
    # at 3x3 the click and the box centre share a cell; at 2x2 they do not.
    assert {c["grid"]: c["strict"] for c in curve[:5]} == {
        "2x2": 0.0, "3x3": 1.0, "4x4": 0.0, "8x8": 0.0, "16x16": 0.0}
    # lenient is never stricter than strict, at any grid
    assert all(c["lenient"] >= c["strict"] for c in curve[:5])


def test_precision_curve_on_zero_rows_emits_nulls_not_zeros():
    """A suppressed cell must be visibly unmeasured. Reporting 0% would read as
    "the model never got one right"."""
    curve = bs_eval.precision_curve([])
    assert len(curve) == len(bs_eval.GRIDS) + 1
    for c in curve:
        assert c["n"] == 0
        assert c["strict"] is None and c["lenient"] is None and c["ratio"] is None
        assert (c["lo"], c["hi"]) == (0.0, 1.0)


def test_dist_summary_describes_the_misses_and_the_whole_set_separately():
    """`median_d_centre` is over misses only; `median_d_centre_all` includes the
    hits. Pooling them would flatter the distance distribution."""
    box = [0.40, 0.40, 0.60, 0.60]             # centre (0.50, 0.50)
    rows = [_pt_row((0.50, 0.50), box),        # hit:  d_box 0.00, d_centre 0.00
            _pt_row((0.50, 0.65), box),        # miss: d_box 0.05, d_centre 0.15
            _pt_row((0.50, 0.80), box),        # miss: d_box 0.20, d_centre 0.30
            _pt_row((0.50, 0.95), box)]        # miss: d_box 0.35, d_centre 0.45
    out = bs_eval.dist_summary(rows)

    assert out["n_miss"] == 3
    assert out["median_d_box"] == pytest.approx(0.20)
    assert out["median_d_centre"] == pytest.approx(0.30)
    assert out["median_d_centre_all"] == pytest.approx(0.225)   # the hit pulls it down
    assert out["bands_d_centre"] == {"moderate_miss": 1, "wrong_region": 2}
    assert out["bands_d_box"] == {"near_miss": 1, "moderate_miss": 1, "wrong_region": 1}


def test_dist_summary_with_no_misses_reports_nulls_and_empty_bands():
    box = [0.40, 0.40, 0.60, 0.60]
    out = bs_eval.dist_summary([_pt_row((0.50, 0.50), box)])
    assert out["n_miss"] == 0
    assert out["median_d_box"] is None and out["p90_d_box"] is None
    assert out["bands_d_box"] == {} and out["bands_d_centre"] == {}
    assert out["median_d_centre_all"] == 0.0
    assert bs_eval.dist_summary([])["median_d_centre_all"] is None


def test_dist_summary_p90_is_unreliable_at_tiny_n():
    """CURRENT BEHAVIOUR on two counts, both of which look wrong.

    1. `pct` needs more than two values for `statistics.quantiles`, and below
       that returns `vals[0]` -- of an already-sorted list, i.e. the *minimum*.
       A two-miss cell publishes its smallest distance in a field called
       `p90_d_box`.
    2. Just above that, `statistics.quantiles` defaults to the exclusive
       method, which extrapolates: three misses of 0.05/0.20/0.35 yield a
       "p90" of 0.44, larger than any observed distance.

    Harmless at the shipped n's (the reported cells have hundreds of misses)
    and both are floors/ceilings rather than sign errors -- but they are wrong
    numbers in published fields rather than nulls, so they are pinned here.
    """
    box = [0.40, 0.40, 0.60, 0.60]
    two = bs_eval.dist_summary([_pt_row((0.50, 0.50), box),      # hit
                                _pt_row((0.50, 0.65), box),      # miss, d_box 0.05
                                _pt_row((0.50, 0.95), box)])     # miss, d_box 0.35
    assert two["n_miss"] == 2
    assert two["p90_d_box"] == pytest.approx(0.05)   # the minimum, not the p90

    three = bs_eval.dist_summary([_pt_row((0.50, 0.65), box),    # d_box 0.05
                                  _pt_row((0.50, 0.80), box),    # d_box 0.20
                                  _pt_row((0.50, 0.95), box)])   # d_box 0.35
    assert three["p90_d_box"] == pytest.approx(0.44)             # above the maximum


@pytest.mark.parametrize("v,expected", [
    (1, "1-4"), (4, "1-4"), (5, "5-6"), (6, "5-6"), (7, "7-9"), (9, "7-9"),
    (10, "10-15"), (15, "10-15"), (16, "16+"), (10 ** 6, "16+"),
])
def test_count_bin_boundaries(v, expected):
    assert bs_eval.count_bin(v) == expected


@pytest.mark.parametrize("v", [0, -1, 10 ** 6 + 1])
def test_count_bin_returns_a_question_mark_outside_the_ladder(v):
    """0 and negatives are impossible counts and a gold of 0 would be a data
    bug; they get an explicit "?" bin rather than joining "1-4"."""
    assert bs_eval.count_bin(v) == "?"


def test_count_bins_are_contiguous_and_non_overlapping():
    """The ladder is the dose-response x-axis; a gap would silently drop a
    count value out of every bin."""
    bins = bs_eval.COUNT_BINS
    for (_lo_a, hi_a, _na), (lo_b, _hi_b, _nb) in zip(bins, bins[1:]):
        assert lo_b == hi_a + 1
    assert bins[0][0] == 1


def test_signed_stats_separates_the_whole_set_from_the_wrong_answers():
    """`mean_signed` includes the correct rows (which contribute 0);
    `mean_signed_when_wrong` is the one that shows the direction of the error."""
    rows = [{"signed_error": -2, "abs_error": 2, "hit": False},
            {"signed_error": 0, "abs_error": 0, "hit": True},
            {"signed_error": 0, "abs_error": 0, "hit": True},
            {"signed_error": 3, "abs_error": 3, "hit": False}]
    out = bs_eval.signed_stats(rows)
    assert out["mean_signed"] == pytest.approx(0.25)
    assert out["mean_signed_when_wrong"] == pytest.approx(0.5)
    assert out["median_abs"] == pytest.approx(1.0)
    assert (out["under"], out["over"]) == (1, 1)


def test_signed_stats_ignores_unparseable_rows_and_survives_an_empty_set():
    """`count_score` returns None errors for an unparseable prediction. Those
    must not be counted as an exact-zero error, which would drag the mean
    toward "no bias"."""
    out = bs_eval.signed_stats([{"signed_error": None, "abs_error": None, "hit": False},
                                {"signed_error": -4, "abs_error": 4, "hit": False}])
    assert out["mean_signed"] == pytest.approx(-4)
    assert (out["under"], out["over"]) == (1, 0)

    empty = bs_eval.signed_stats([])
    assert empty == {"mean_signed": None, "mean_signed_when_wrong": None,
                     "median_abs": None, "under": 0, "over": 0}


def test_chi2_against_a_uniform_expectation():
    """Position bias is tested before any accuracy number is trusted, so this
    has to be right in both directions."""
    uniform = {k: 0.25 for k in "ABCD"}
    flat = bs_eval.chi2_against({"A": 25, "B": 25, "C": 25, "D": 25}, uniform, 100)
    assert flat["chi2"] == pytest.approx(0.0)
    assert flat["biased"] is False

    skewed = bs_eval.chi2_against({"A": 100}, uniform, 100)
    assert skewed["chi2"] == pytest.approx(300.0)
    assert skewed["biased"] is True
    assert skewed["crit"] == bs_eval.CHI2_CRIT_DF3


def test_chi2_against_a_non_uniform_expectation_uses_the_actual_key_share():
    """The all-picks test compares against the answer key's own share, not 0.25
    -- the key is only near-uniform, and assuming uniform would manufacture a
    small bias every time."""
    share = {"A": 0.4, "B": 0.2, "C": 0.2, "D": 0.2}
    out = bs_eval.chi2_against({"A": 40, "B": 20, "C": 20, "D": 20}, share, 100)
    assert out["chi2"] == pytest.approx(0.0)
    rows = {r["option"]: r for r in out["rows"]}
    assert rows["A"]["obs_share"] == pytest.approx(0.4)
    assert rows["A"]["deviation_pp"] == pytest.approx(0.0)


def test_chi2_against_reports_all_four_options_and_survives_n_zero():
    """A zero-expectation option contributes nothing to chi2 rather than
    dividing by zero, and n=0 emits nulls instead of raising."""
    out = bs_eval.chi2_against({}, {k: 0.25 for k in "ABCD"}, 0)
    assert out["n"] == 0 and out["chi2"] == 0.0 and out["biased"] is False
    assert [r["option"] for r in out["rows"]] == list("ABCD")
    assert all(r["obs_share"] is None and r["deviation_pp"] is None for r in out["rows"])

    zero_expect = bs_eval.chi2_against({"A": 10}, {"A": 0.0, "B": 1.0}, 10)
    assert math.isfinite(zero_expect["chi2"])


def test_read_jsonl_tolerates_a_torn_final_line_and_a_missing_file(tmp_path):
    """A results JSONL is written by a run that can be killed mid-line. A
    half-written last line must not take down an analysis of the other 4,000."""
    p = tmp_path / "r.jsonl"
    p.write_text('{"uid": "a"}\n\n{"uid": "b"}\n{"uid": "c", "pre')
    assert bs_eval.read_jsonl(p) == [{"uid": "a"}, {"uid": "b"}]
    assert bs_eval.read_jsonl(tmp_path / "nope.jsonl") == []
    assert core.read_jsonl(p) == bs_eval.read_jsonl(p)


def test_loc_cell_and_derived_cell_flag_a_cell_too_thin_to_interpret():
    """EVAL.md 5: cells under n=30 are suppressed rather than shown as noise.
    The flag travels with the cell so the renderer cannot forget to check."""
    rows = [_pt_row((0.50, 0.50), [0.40, 0.40, 0.60, 0.60]) for _ in range(5)]
    thin = bs_eval.loc_cell(rows, "thin")
    assert thin["n"] == 5 and thin["suppressed"] is True
    assert thin["acc"] == 1.0 and thin["ratio"] == pytest.approx(100.0)

    fat = bs_eval.loc_cell(rows * 6, "fat")
    assert fat["n"] == 30 and fat["suppressed"] is False

    empty = bs_eval.loc_cell([], "none")
    assert empty["n"] == 0 and empty["acc"] is None and empty["ratio"] is None
    assert (empty["lo"], empty["hi"]) == (0.0, 1.0)

    assert bs_eval.derived_cell([{"hit": True}] * 3, "d")["suppressed"] is True
    assert bs_eval.derived_cell([], "d")["acc"] is None
