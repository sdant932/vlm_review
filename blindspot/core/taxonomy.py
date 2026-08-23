"""One primitive taxonomy across datasets -- the spine of the whole report.

The brief asks which *perceptual primitives* underlie business visual tasks and
where they break. A per-dataset accuracy table cannot answer that; the same
primitive has to be measurable across independent sources so agreement between
them is evidence about the model rather than about one benchmark.

Every mapping here is either:

  construction    the task itself guarantees the primitive (CharXiv's templates
                  are generated per type; ScreenSpot-Pro is localization by
                  definition)
  dataset_label   the dataset's own human annotation (InfographicVQA's
                  operation/reasoning labels)

There is deliberately no keyword-guessed tier: a regex over question text would
look identical in the matrix while being materially less trustworthy, and
pooling the two would make the headline chart partly fiction.
"""

from __future__ import annotations

from blindspot.core.adapters import Example

PRIMITIVES = (
    "counting", "line_following", "localization_read", "localization_point",
    "value_interpolation",
    "binding", "structure", "text_in_situ", "comparison", "arithmetic",
    "composition",
)

# Human-readable, in the order the report should show them.
LABELS = {
    "counting": "Counting objects",
    "line_following": "Following lines",
    # Two genuinely different tasks, split apart after they diverged 91% vs 4%.
    # CharXiv asks you to find a spatial extreme and *read what is written there*
    # -- the answer is a string. ScreenSpot-Pro asks you to find an element and
    # *emit its coordinates* -- the answer is a number pair, graded by whether
    # the point lands inside a box. Calling both "localization" hid the fact
    # that one of them additionally requires expressing position numerically.
    "localization_read": "Localization \u2192 read the value there",
    "localization_point": "Localization \u2192 emit a coordinate",
    "value_interpolation": "Reading values off an axis",
    "binding": "Binding legend to series",
    "structure": "Parsing layout structure",
    "text_in_situ": "Reading text in place",
    "comparison": "Comparison",
    "arithmetic": "Arithmetic over visual values",
    "composition": "Composing multiple readings",
}

# CharXiv descriptive template id -> primitive. Each template isolates one
# operation by construction, which is what makes CharXiv the backbone here.
CHARXIV_QID = {
    10: "counting", 12: "counting", 17: "counting", 19: "counting",
    11: "line_following",
    4: "localization_read", 5: "localization_read",
    6: "localization_read", 7: "localization_read",
    8: "value_interpolation", 9: "value_interpolation",
    14: "value_interpolation", 15: "value_interpolation",
    13: "binding",
    18: "structure",
    1: "text_in_situ", 2: "text_in_situ", 3: "text_in_situ",
    16: "comparison",
}

# AI2D, added after FlowLearn was dropped. Only the label-reference half is a
# perception measurement: a blind control (diagram withheld) scored 80.0% on the
# reasoning half against 88.2% with the diagram -- an 8pp gain, so that split is
# mostly world knowledge. Label-reference went 31.5% blind (chance is 25%) to
# 59.8% seeing, so nearly all of its signal comes from looking.
AI2D_QTYPE = {"label_reference": "localization_read"}

INFOVQA_OP = {"counting": "counting", "comparison": "comparison",
              "arithmetic": "arithmetic"}


def primitive_for(ex: Example) -> tuple[str | None, str | None]:
    """(primitive, provenance). (None, None) when the item maps to no primitive."""
    ds, meta = ex.dataset, ex.meta

    if ds == "charxiv":
        if meta.get("split") == "reasoning":
            return "composition", "construction"
        qid = meta.get("qid")
        if qid is not None:
            return CHARXIV_QID.get(int(qid)), "construction"
        return None, None

    if ds == "screenspot_pro":
        return "localization_point", "construction"

    if ds == "ai2d":
        prim = AI2D_QTYPE.get(meta.get("qtype"))
        return (prim, "construction") if prim else (None, None)

    if ds == "infographicvqa":
        for op in (meta.get("operation") or []):
            if op in INFOVQA_OP:
                return INFOVQA_OP[op], "dataset_label"
        return None, None  # unlabelled direct lookup: reported separately

    return None, None


def is_not_applicable(ex: Example) -> bool:
    """CharXiv golds are often 'Not Applicable' -- and the rate varies wildly by
    template (colorbar 86%, title 59%, lines-intersect 45%, most others 0%).

    Pooling those in measures "can you tell this doesn't apply", not the
    primitive. The report scores the non-N/A subset separately for any cell
    where the rate is material.
    """
    try:
        return str(ex.gold[0]).strip().lower() == "not applicable"
    except (IndexError, TypeError):
        return False
