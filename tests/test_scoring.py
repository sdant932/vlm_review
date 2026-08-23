"""The scorers, which produce every number in the study.

These pin behaviour that is easy to break by accident and hard to notice: the
ANLS threshold boundary, the deliberate difference between ANLS normalization
and the general one, and the refusal to credit substring containment.
"""

import pytest

from blindspot.core.scoring import (
    ANLS_THRESHOLD, CHARXIV_FUZZY_QIDS, CHARXIV_STRICT_QIDS,
    anls, boolean_match, charxiv_grading_confidence, count_score,
    numeric_or_text_match, point_in_bbox, token_f1,
)


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
