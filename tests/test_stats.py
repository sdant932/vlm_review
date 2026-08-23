"""Statistics and grid helpers.

These were moved out of a 3,000-line HTML renderer into `core.stats` during the
repository reorganization. Pinned here so the move stays honest and so a future
edit to a confidence interval shows up as a failing test rather than a shifted
number in a report.
"""

import pytest

from blindspot.core.stats import (
    bbox_cells, cell_of, centre_cell, is_na, quantiles, wilson,
)


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
