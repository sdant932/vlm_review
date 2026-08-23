"""Rate how trustworthy each question's ground truth is, so scores can be
reported on a clean subset as well as in full.

The audits found that reference-answer errors are not spread evenly: they track
what the ANNOTATOR had to do. Where the gold is a string printed in the image and
the annotator only had to copy it, error rates are near zero. Where the gold
required a judgement that is written nowhere -- what counts as a "line", whether
a panel label is a "title", which categories "total" includes -- rates run 13-16%.

So confidence is assigned from three sources, in order of strength:

  1. FORMAT      multiple choice cannot suffer transcription error at all: the
                 gold is an index into options supplied with the question.
  2. AUDIT       measured per-cell error rates from the vision adjudicator
                 (results/<dataset>__gtaudit.jsonl). Empirical, not assumed.
  3. STRUCTURE   the dataset's own annotation axes, where they distinguish
                 extractive answers from derived ones.

Nothing here rescores anything. It adds a `gt_confidence` field so a reader can
ask "what is the score on questions whose ground truth we can actually trust?"
alongside the official number.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

RESULTS = Path("results")

# Above this measured error rate, a cell's ground truth is not dependable enough
# to read a model result off it without saying so.
LOW_THRESHOLD = 0.10
HIGH_THRESHOLD = 0.03


def cell_of(ex) -> str:
    """The slice an item belongs to, for pooling audit verdicts."""
    ds, m = ex.dataset, ex.meta
    if ds == "charxiv":
        return f"qid{m['qid']}" if m.get("qid") else "reasoning"
    if ds == "infographicvqa":
        return (m.get("operation") or ["direct_lookup"])[0]
    if ds == "screenspot_pro":
        return m.get("ui_type") or "?"
    if ds == "ai2d":
        return m.get("qtype") or "?"
    return ds


def audited_rates(dataset: str, examples: dict) -> dict[str, tuple[float, int]]:
    """cell -> (fraction of audited failures with non-unambiguous GT, n audited)."""
    p = RESULTS / f"{dataset}__gtaudit.jsonl"
    if not p.exists():
        return {}
    tally = collections.defaultdict(lambda: [0, 0])
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "error" in r or r["uid"] not in examples:
            continue
        c = cell_of(examples[r["uid"]])
        tally[c][1] += 1
        tally[c][0] += r.get("gt_quality") != "unambiguous"
    return {c: (b / n, n) for c, (b, n) in tally.items() if n}


def confidence(ex, rates: dict[str, tuple[float, int]]) -> tuple[str, str]:
    """(high | medium | low, reason)."""
    # 1. Format: MC has no transcription step to get wrong.
    if ex.answer_type == "choice":
        return "high", "multiple choice: gold is an index into supplied options"

    # 2. Audit: measured rate for this cell, when enough of it was adjudicated.
    c = cell_of(ex)
    if c in rates:
        rate, n = rates[c]
        if n >= 15:
            if rate <= HIGH_THRESHOLD:
                return "high", f"audited: {rate*100:.0f}% bad GT over {n} failures"
            if rate >= LOW_THRESHOLD:
                return "low", f"audited: {rate*100:.0f}% bad GT over {n} failures"
            return "medium", f"audited: {rate*100:.0f}% bad GT over {n} failures"

    # 3. Structure: the dataset's own extractive/derived distinction.
    if ex.dataset == "infographicvqa":
        at = (ex.meta.get("gold_answer_type") or [""])[0]
        if not (ex.meta.get("operation") or []) and at in ("single span", "multi-span"):
            return "high", "extractive span, no derivation required"
        if at == "non-extractive":
            return "low", "answer is not printed on the page; annotator derived it"
    if ex.dataset == "charxiv":
        # Verbatim readout of a printed tick or colorbar label.
        if ex.meta.get("qid") in (4, 5, 6, 7, 14, 15):
            return "high", "verbatim readout of a printed axis/colorbar label"
        if ex.meta.get("split") == "reasoning" or ex.meta.get("qid") in (1, 10, 16):
            return "low", "requires an annotator judgement call"
    return "medium", "not audited and not structurally classifiable"


def annotate_rows(rows: list[dict], dataset: str) -> list[dict]:
    examples = {r["uid"]: r["_ex"] for r in rows if "_ex" in r}
    rates = audited_rates(dataset, examples)
    for r in rows:
        if "_ex" in r:
            lvl, why = confidence(r["_ex"], rates)
            r["gt_confidence"], r["gt_confidence_reason"] = lvl, why
    return rows
