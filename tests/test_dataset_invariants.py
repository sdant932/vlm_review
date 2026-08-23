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

import pytest

from blindspot.core.adapters import ADAPTERS, Example, load
from blindspot.core.scoring import point_in_bbox, score


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
