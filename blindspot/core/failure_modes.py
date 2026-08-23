"""Classify *why* an answer was scored wrong.

A single accuracy number treats these as identical failures:

    gold "A, B, C"  pred "C, B, A"          -> read everything, ignored the order instruction
    gold "A, B, C"  pred "A, B, C, D"       -> read everything, plus something that isn't there
    gold "A, B, C"  pred "A, B"             -> missed one
    gold "310.5"    pred "310.5 million"    -> right value, wrong format
    gold "A, B, C"  pred "X, Y"             -> actually wrong

They are not the same finding. The first is an instruction-following failure on a
question whose perception succeeded; the last is a perception failure. Reporting
them together overstates how much the model failed to *see*.

Order matters here specifically because several CharXiv templates ask for it --
qid 13 is literally "(from top to bottom, then left to right)" -- so listing the
right items in the wrong sequence is disobeying an instruction, not misreading
the figure.

List cases are decided deterministically (set and sequence comparison); anything
that needs semantic judgement is left as `unclassified` for the LLM pass.
"""

from __future__ import annotations

import re
from typing import Any

# Coordinate answers cannot be categorised by comparing strings; they get their
# own modes from how far the click landed, which is the distinction that matters
# for localization: never found the area vs found it and missed a tiny target.
POINT_MODES = ("near_miss", "moderate_miss", "wrong_region")

# Multiple choice has exactly one way to fail: pick the wrong letter. There is no
# formatting, ordering or completeness failure available, which is precisely why
# an MC benchmark cannot suffer the transcription errors the others do.
CHOICE_MODES = ("wrong_option",)

MODES = (
    "near_miss", "moderate_miss", "wrong_region", "wrong_option",
    "order_only",       # same items, wrong sequence
    "extra_items",      # everything correct, plus items that are not in the gold
    "missing_items",    # a strict subset of the gold
    "partial_overlap",  # some right, some wrong
    "format_only",      # same value, different surface form (unit, separator, wording)
    "wrong_value",      # genuinely different answer
    "unclassified",     # needs semantic judgement
)

LABELS = {
    "wrong_option": "chose the wrong option",
    "near_miss": "right area, missed the box (<10% off)",
    "moderate_miss": "roughly the wrong place (10-25% off)",
    "wrong_region": "nowhere near (>25% off)",
    "order_only": "right items, wrong order",
    "extra_items": "right items, plus extras",
    "missing_items": "missed some items",
    "partial_overlap": "partly right",
    "format_only": "right value, wrong format",
    "wrong_value": "wrong answer",
    "unclassified": "needs judgement",
}

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_UNITS = re.compile(r"\b(million|billion|thousand|percent|pct|usd|dollars?|people|users?|"
                    r"countries|cases|years?|days?|m|bn|k)\b", re.I)


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" .")


def _items(s: Any) -> list[str]:
    """Split a list-shaped answer. Commas first, then semicolons/newlines."""
    t = str(s).strip()
    for sep in (",", ";", "\n"):
        if sep in t:
            return [_norm(x) for x in t.split(sep) if _norm(x)]
    return [_norm(t)] if _norm(t) else []


def _numbers(s: Any) -> list[str]:
    return _NUM.findall(str(s).replace(",", ""))


def _strip_units(s: Any) -> str:
    return _norm(_UNITS.sub("", str(s).replace("%", "").replace("$", "")))


def classify(gold: Any, pred: Any) -> str:
    """Return one of MODES for a prediction already known to have scored wrong."""
    golds = gold if isinstance(gold, (list, tuple)) else [gold]
    p_raw = str(pred)

    # --- list-shaped answers first ---------------------------------------
    # Checked before the format shortcut: for a legend list the extra terms may
    # be non-numeric ("p", "n"), so a numbers-match test would wrongly call an
    # over-complete answer a formatting difference.
    for g in golds:
        gi, pi = _items(g), _items(p_raw)
        if len(gi) < 2 and len(pi) < 2:
            continue  # not a list; fall through
        gs, ps = set(gi), set(pi)
        if not gs:
            continue
        if gs == ps:
            return "order_only" if gi != pi else "format_only"
        if gs < ps:
            return "extra_items"
        if ps < gs:
            return "missing_items"
        if gs & ps:
            return "partial_overlap"

    # --- same value, different surface form ------------------------------
    for g in golds:
        if _norm(g) == _norm(p_raw):
            return "format_only"
        gn, pn = _numbers(g), _numbers(p_raw)
        if gn and gn == pn:
            # identical numbers, different wrapper: "310.5" vs "310.5 million"
            return "format_only"
        if _strip_units(g) and _strip_units(g) == _strip_units(p_raw):
            return "format_only"

    return "unclassified"


def classify_point(gold: list, pred: Any) -> str:
    """Spatial failure mode for a click that fell outside the target box."""
    try:
        cx, cy = (gold[0] + gold[2]) / 2, (gold[1] + gold[3]) / 2
        ex, ey = abs(pred[0] - cx), abs(pred[1] - cy)
    except Exception:
        return "unclassified"
    if ex <= .10 and ey <= .10:
        return "near_miss"
    if ex > .25 or ey > .25:
        return "wrong_region"
    return "moderate_miss"


def summarize(rows: list[dict]) -> dict[str, int]:
    """Count modes across a set of already-failed rows."""
    out = dict.fromkeys(MODES, 0)
    for r in rows:
        out[r.get("failure_mode", "unclassified")] += 1
    return {k: v for k, v in out.items() if v}
