"""Normalize each dataset's manifest into a common `Example` record.

One `Example` == one scoreable question. Note this is not one manifest row:
CharXiv ships five questions per figure (four descriptive + one reasoning), so
its row count and its question count differ by ~5x.

`answer_type` drives both prompt construction (prompts.py) and scoring
(scoring.py), so it is the field to look at when adding a dataset:

    span   -> short text answer, graded by ANLS against one or more golds
    point  -> a click location, graded by whether it lands inside a gold bbox

Gold bboxes are always stored normalized to [0, 1] as (x0, y0, x1, y1),
regardless of how the source dataset encoded them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

from blindspot.core.vendor.charxiv_constants import (DESCRIPTIVE_RESP_INST, DESCRIPTIVE_GRADING_QMAP,
                                                     reasoning_question)

DATA = Path("data")


@dataclass
class Example:
    uid: str
    dataset: str
    images: list[str]
    question: str
    answer_type: str  # "span" | "point"
    gold: Any
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _rows(dataset: str) -> Iterator[dict]:
    with open(DATA / dataset / "manifest.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _img(dataset: str, rel: str) -> str:
    return str(DATA / dataset / rel)


def _norm_bbox(x0: float, y0: float, x1: float, y1: float, w: float, h: float) -> tuple[list[float], bool]:
    """Normalize a pixel bbox to [0, 1] and clamp it to the image.

    ScreenSpot-Pro has at least one upstream annotation that runs a pixel past
    the top edge (`inventor_windows_60`, y0=-1). Clamping is the right call for
    an off-by-one on an edge-adjacent toolbar icon; the caller gets a flag so
    clamped rows stay auditable rather than silently corrected.
    """
    box = [x0 / w, y0 / h, x1 / w, y1 / h]
    clamped = [min(max(v, 0.0), 1.0) for v in box]
    return clamped, clamped != box


# --------------------------------------------------------------------------
# InfographicVQA -- real infographics; charts + dense text + icons + layout.
# `answers` is a list of acceptable strings; ANLS scores against the best one.
# --------------------------------------------------------------------------
def infographicvqa() -> Iterator[Example]:
    for r in _rows("infographicvqa"):
        yield Example(
            uid=f"infographicvqa:{r['questionId']}",
            dataset="infographicvqa",
            images=[_img("infographicvqa", r["image"])],
            question=r["question"],
            answer_type="span",
            gold=r["answers"],
            meta={
                # answer_type/operation are the dataset's own difficulty labels;
                # they are the primary slice axes in the blind-spot report.
                "gold_answer_type": r.get("answer_type") or [],
                "operation": r.get("operation/reasoning") or [],
            },
        )


# --------------------------------------------------------------------------
# CharXiv -- real scientific charts from arXiv.
#
# Descriptive questions are stored as integer template IDs. Reconstructed here
# exactly as CharXiv's own `descriptive_query_helper` does, because a different
# prompt makes the numbers incomparable to anything published:
#   - qids 18/19 (subplot layout/count) take no subplot prefix
#   - subplot_loc as a string means the subplots do not form a grid
#   - row 0 means the figure has a single plot
# --------------------------------------------------------------------------
def _charxiv_question(qid: int, subplot_loc: Any, row: Any, col: Any) -> str:
    if qid in (18, 19):
        return DESCRIPTIVE_RESP_INST[qid]
    if isinstance(subplot_loc, str) and subplot_loc:
        prefix = f"For {subplot_loc}, "
    else:
        if row is None or col is None:
            raise ValueError(f"qid {qid}: no subplot location (loc={subplot_loc!r})")
        prefix = (
            "For the current plot, "
            if int(row) == 0
            else f"For the subplot at row {int(row)} and column {int(col)}, "
        )
    return DESCRIPTIVE_RESP_INST[qid].format(prefix)


def charxiv() -> Iterator[Example]:
    # `original_id` is the source arXiv paper, which contributes more than one
    # figure, so the row index is what makes the uid unique.
    for i, r in enumerate(_rows("charxiv")):
        img = [_img("charxiv", r["image"])]
        base = {
            "paper_id": r.get("original_id"),
            "category": r.get("category"),
            "year": r.get("year"),
            "num_subplots": r.get("num_subplots"),
        }
        for j in range(1, 5):
            qid, gold = r.get(f"descriptive_q{j}"), r.get(f"descriptive_a{j}")
            if qid is None or gold is None:
                continue
            yield Example(
                uid=f"charxiv:{i:05d}:d{j}",
                dataset="charxiv",
                images=img,
                question=_charxiv_question(
                    int(qid), r.get("subplot_loc"), r.get("subplot_row"), r.get("subplot_col")
                ),
                answer_type="span",
                gold=[str(gold)],
                meta={
                    **base,
                    "split": "descriptive",
                    "qid": int(qid),
                    "qlabel": DESCRIPTIVE_GRADING_QMAP[int(qid)],
                },
            )
        if r.get("reasoning_q") and r.get("reasoning_a") is not None:
            yield Example(
                uid=f"charxiv:{i:05d}:r",
                dataset="charxiv",
                images=img,
                # CharXiv does not send the bare reasoning question: it wraps it
                # in an answer-format instruction selected by `reasoning_a_type`.
                # Sending the unwrapped question changes the task and makes the
                # score incomparable to CharXiv's published numbers.
                question=reasoning_question(r["reasoning_q"], r["reasoning_a_type"],
                                            r["reasoning_a"]),
                answer_type="span",
                gold=[str(r["reasoning_a"])],
                meta={
                    **base,
                    "split": "reasoning",
                    "reasoning_a_type": r.get("reasoning_a_type"),
                    "reasoning_q_source": r.get("reasoning_q_source"),
                },
            )


# --------------------------------------------------------------------------
# ScreenSpot / ScreenSpot-Pro -- UI element grounding.
#
# The two encode bboxes differently and mixing them up silently produces
# plausible-but-wrong accuracy, so each is converted explicitly:
#   screenspot     : normalized (x0, y0, x1, y1)
#   screenspot_pro : absolute pixels (x0, y0, x1, y1) + img_size (w, h)
# --------------------------------------------------------------------------
def screenspot() -> Iterator[Example]:
    for i, r in enumerate(_rows("screenspot")):
        x0, y0, x1, y1 = (float(v) for v in r["bbox"])
        gold, clamped = _norm_bbox(x0, y0, x1, y1, 1.0, 1.0)  # already normalized
        yield Example(
            uid=f"screenspot:{i:05d}",
            dataset="screenspot",
            images=[_img("screenspot", r["image"])],
            question=r["instruction"],
            answer_type="point",
            gold=gold,
            meta={
                "ui_type": r.get("data_type"),      # icon vs text
                "platform": r.get("data_source"),   # windows / macos / mobile / web
                "source_file": r.get("file_name"),
                "target_area_frac": abs((gold[2] - gold[0]) * (gold[3] - gold[1])),
                "bbox_clamped": clamped,
            },
        )


def screenspot_pro() -> Iterator[Example]:
    for r in _rows("screenspot_pro"):
        x0, y0, x1, y1 = (float(v) for v in r["bbox"])
        w, h = (float(v) for v in r["img_size"])
        gold, clamped = _norm_bbox(x0, y0, x1, y1, w, h)
        yield Example(
            uid=f"screenspot_pro:{r['id']}",
            dataset="screenspot_pro",
            images=[_img("screenspot_pro", r["image"])],
            question=r["instruction"],
            answer_type="point",
            gold=gold,
            meta={
                "ui_type": r.get("ui_type"),
                "platform": r.get("platform"),
                "application": r.get("application"),
                "group": r.get("group"),
                "img_size": [int(w), int(h)],
                # Fraction of the screen the target occupies. High-res pro
                # screenshots get downscaled to Haiku's ~1568px budget, so this
                # is the number to correlate failures against.
                "target_area_frac": abs((gold[2] - gold[0]) * (gold[3] - gold[1])),
                "bbox_clamped": clamped,
            },
        )


ADAPTERS = {
    "infographicvqa": infographicvqa,
    "charxiv": charxiv,
    "screenspot": screenspot,
    "screenspot_pro": screenspot_pro,
}


def load(dataset: str) -> list[Example]:
    return list(ADAPTERS[dataset]())


# --------------------------------------------------------------------------
# FlowLearn (simulated) -- flowcharts / process diagrams.
#
# The only source here with usable arrow-following ground truth, and the only
# one immune to world-knowledge shortcuts: node labels are nonsense words
# ("dihedron cushite") in the `word` variant and random characters in `char`,
# so the model has to trace the arrow rather than guess a plausible edge.
#
# Golds are RE-DERIVED from the parsed Mermaid, not copied from the shipped QA
# fields. Verified on the 150-row sample: `Arrow_AtoB` false pairs are clean
# (0 mislabelled), but 38% of `Arrow_betweenAB` false pairs DO have an
# undirected edge -- that field is directed despite its name. Asking "is there
# an arrow between A and B?" while scoring against it would invent a blind spot
# that isn't there. The shipped fields are used only to choose which node pair
# to ask about; `meta["gold_disagrees_upstream"]` keeps the disagreement rate
# visible instead of silently corrected.
# --------------------------------------------------------------------------
def flowlearn_sim() -> Iterator[Example]:
    from blindspot.analysis.mermaid import parse_mermaid, has_edge, edge_style, hops

    def unconnected_pair(g, stem: str):
        """Pick a genuinely unconnected node pair, deterministically.

        The shipped `Arrow_betweenAB` false pair was chosen for a *directed*
        question, so 38% of them are in fact connected. Reusing it for an
        undirected question and then correcting the gold afterwards leaves the
        yes/no split lopsided (357/243 measured), which quietly rewards a
        constant-"yes" model. Choosing our own negative keeps the probe
        balanced by construction. Preference for hops >= 2 avoids trivially
        distant pairs while staying deterministic across runs.
        """
        labels = sorted(g.labels.values())
        cands = []
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                if has_edge(g, a, b, directed=False) is False:
                    cands.append((hops(g, a, b, directed=False) or 99, a, b))
        if not cands:
            return None, None
        far = [c for c in cands if c[0] >= 2] or cands
        return far[hash(stem) % len(far)][1:]


    for r in _rows("flowlearn_sim"):
        g = parse_mermaid(r.get("Flowchart-to-Mermaid") or "")
        if not g.labels:
            continue  # unparseable; counted by validate.py
        variant = r.get("variant", "word")
        stem = Path(r["image"]).stem
        img = [_img("flowlearn_sim", r["image"])]
        base = {
            "variant": variant,
            "direction": g.direction,          # TB/TD/BT/LR/RL -- BT and RL run against reading order
            "n_nodes": g.n_nodes,
            "n_arrows": g.n_edges,
        }

        # --- arrow following: directed and undirected, each balanced yes/no ---
        for family, field, directed in (
            ("atob", "Arrow_AtoB", True),
            ("between", "Arrow_betweenAB", False),
        ):
            pairs = r.get(field) or {}
            for polarity in ("true", "false"):
                pair = pairs.get(polarity) or {}
                a, b = pair.get("a"), pair.get("b")
                # For the undirected negative, substitute a truly unconnected
                # pair when the shipped one turns out to be connected.
                if (family == "between" and polarity == "false" and a and b
                        and has_edge(g, a, b, directed=False) is not False):
                    a, b = unconnected_pair(g, stem)
                if not a or not b:
                    continue
                derived = has_edge(g, a, b, directed=directed)
                present = derived is not None
                gold = "yes" if derived else "no"
                if directed:
                    q = (f'Does the node labelled "{a}" have an arrow pointing to the node '
                         f'labelled "{b}"? Direction matters: an arrow from "{b}" to "{a}" '
                         f'does not count.')
                else:
                    q = (f'Is the node labelled "{a}" connected to the node labelled "{b}" '
                         f'by an arrow, in either direction?')
                yield Example(
                    uid=f"flowlearn_sim:{variant}:{stem}:{family}:{polarity}",
                    dataset="flowlearn_sim",
                    images=img,
                    question=q,
                    answer_type="boolean",
                    gold=[gold],
                    meta={
                        **base,
                        "family": family,
                        "polarity": polarity,
                        # A "no" answer is right for different reasons; never pool these.
                        "false_mode": (
                            None if polarity == "true" else
                            "phantom_node" if not present else
                            "reversed_edge" if has_edge(g, b, a, directed=True) else
                            "unconnected"
                        ),
                        "edge_style": edge_style(g, a, b),
                        "hops": hops(g, a, b, directed=False),
                        "gold_disagrees_upstream": (gold == "yes") != (polarity == "true"),
                    },
                )

        # --- counting, with difficulty parameterized by the true count ---
        for family, question, gold in (
            ("nodes", "How many nodes (boxes) are in this flowchart?", g.n_nodes),
            ("arrows", "How many arrows are in this flowchart?", g.n_edges),
        ):
            yield Example(
                uid=f"flowlearn_sim:{variant}:{stem}:{family}",
                dataset="flowlearn_sim",
                images=img,
                question=question,
                answer_type="count",
                gold=[gold],
                meta={**base, "family": family, "true_count": gold},
            )


ADAPTERS["flowlearn_sim"] = flowlearn_sim


# --------------------------------------------------------------------------
# AI2D -- grade-school science diagrams, 4-way multiple choice.
#
# Added after FlowLearn was excluded: it is the only remaining source of
# arrow/flow-following on real diagrams, which the brief names explicitly and
# which CharXiv covers with a single question type (97 usable items).
#
# Multiple choice is also the cleanest instrument in the study -- the model
# picks a letter, so a correct reading can never be scored wrong for phrasing.
#
# Two structurally distinguishable question types, separated by option shape
# rather than by keyword guessing: when every option is a bare label ("a", "D")
# the question is asking the model to resolve a letter printed on the diagram to
# the thing it points at; otherwise it is reasoning over the diagram's structure.
# --------------------------------------------------------------------------
def ai2d() -> Iterator[Example]:
    for i, r in enumerate(_rows("ai2d")):
        opts = list(r.get("options") or [])
        if len(opts) != 4:
            continue
        try:
            gold_letter = "ABCD"[int(r["answer"])]
        except (ValueError, TypeError, IndexError, KeyError):
            continue
        label_ref = all(len(str(o).strip()) <= 2 for o in opts)
        yield Example(
            uid=f"ai2d:{i:05d}",
            dataset="ai2d",
            images=[_img("ai2d", r["image"])],
            question=r["question"],
            answer_type="choice",
            gold=[gold_letter],
            meta={
                "options": opts,
                "gold_text": opts[int(r["answer"])],
                "qtype": "label_reference" if label_ref else "diagram_reasoning",
            },
        )


ADAPTERS["ai2d"] = ai2d


# --------------------------------------------------------------------------
# SlideVQA -- real slide decks. The only multi-page source here: every question
# ships all 20 pages of its deck, and 567 of 2,215 need evidence from more than
# one slide.
#
# Run as TWO conditions, because a single score would conflate two different
# abilities:
#   evidence   only the annotated evidence pages -> can it reason across the
#              right slides once you hand them over?
#   all_pages  the whole 20-page deck -> can it also find them?
# The gap between the two is retrieval; the evidence score is multi-hop reading.
#
# 20 images is also the API's boundary for full-resolution handling (above 20,
# each image is capped at 2000px), so the all-pages condition sits exactly on it.
# --------------------------------------------------------------------------
def slidevqa(condition: str = "evidence") -> Iterator[Example]:
    for i, r in enumerate(_rows("slidevqa")):
        pages = [r.get(f"page_{k}") for k in range(1, 21)]
        pages = [p for p in pages if p]
        ev = [int(x) for x in (r.get("evidence_pages") or []) if 0 < int(x) <= len(pages)]
        if not pages or not ev:
            continue
        imgs = ([_img("slidevqa", pages[e - 1]) for e in sorted(ev)]
                if condition == "evidence"
                else [_img("slidevqa", p) for p in pages])
        arith = r.get("arithmetic_expression")
        yield Example(
            uid=f"slidevqa:{condition}:{r.get('qa_id', i)}",
            dataset="slidevqa",
            images=imgs,
            question=r["question"],
            answer_type="span",
            gold=[str(r["answer"])],
            meta={
                "condition": condition,
                "n_evidence": len(ev),
                "multi_page": len(ev) > 1,
                "n_pages_sent": len(imgs),
                "deck": r.get("deck_name"),
                "arithmetic": arith not in (None, "None", ""),
            },
        )


ADAPTERS["slidevqa"] = lambda: slidevqa("evidence")
ADAPTERS["slidevqa_allpages"] = lambda: slidevqa("all_pages")


# --------------------------------------------------------------------------
# svg_localization -- synthetic vector charts built so that target size and
# image resolution are knobs rather than confounds.
#
# Three question types share one manifest and split by `answer_type`:
#   point     -> click-in-bbox against the *widget* hit box (gold_bbox_norm)
#   relation  -> "which label is immediately below X" -- position, no coords
#   reverse   -> "what text is at (x, y)" -- coords in, text out
#
# The gold box is `gold_bbox_norm`, never `text_ink_bbox_norm`: the latter is
# the glyph outline and is ~3x smaller, which would be a harder and different
# task from the one ScreenSpot-Pro poses. The ink box rides along in meta so an
# analysis can report it without being able to accidentally score against it.
# --------------------------------------------------------------------------
def svg_localization() -> Iterator[Example]:
    for r in _rows("svg_localization"):
        qtype = r["qtype"]
        common = {
            "qtype": qtype,
            "graph_id": r["graph_id"],
            "resolution": r["resolution"],
            "scale": r.get("scale"),
            "chart_type": r["chart_type"],
            "theme": r["theme"],
            "font_family": r["font_family"],
            "font_px": r.get("font_px"),
            "complexity": r.get("complexity"),
            "target_idx": r.get("target_idx"),
            "target_text": r.get("target_text"),
            "target_role": r.get("target_role"),
            "anchor_text": r.get("anchor_text"),      # relation only; absent elsewhere
            "direction": r.get("direction"),
            "hit_source": r.get("hit_source"),
            "target_area_frac": r.get("target_area_frac"),
            "target_ink_area_frac": r.get("target_ink_area_frac"),
            "target_contrast": r.get("target_contrast"),
            "target_occluded_frac": r.get("target_occluded_frac"),
            "img_size": [int(v) for v in r["image_px"]],
            "effective_px": [int(v) for v in r["effective_px"]],
            "downscaled_by_api": r.get("downscaled_by_api"),
            "n_eligible_targets": r.get("n_eligible_targets"),
        }
        if qtype == "point":
            yield Example(
                uid=r["uid"], dataset="svg_localization",
                images=[_img("svg_localization", r["image"])],
                question=r["question"],       # bare description; POINT_INSTRUCTION prepended later
                answer_type="point",
                gold=[float(v) for v in r["gold_bbox_norm"]],
                meta={**common,
                      "gold_center_norm": r.get("gold_center_norm"),
                      "text_ink_bbox_norm": r.get("text_ink_bbox_norm"),
                      "probe_point_px": r.get("probe_point_px")},
            )
        else:
            yield Example(
                uid=r["uid"], dataset="svg_localization",
                images=[_img("svg_localization", r["image"])],
                question=r["question"],       # already self-contained and format-instructed
                answer_type="span",
                gold=[str(r["answer"])],
                meta={**common, "probe_point_px": r.get("probe_point_px")},
            )


ADAPTERS["svg_localization"] = svg_localization


# --------------------------------------------------------------------------
# svg_counting / svg_word_mc -- derived from the same 200 scenes as
# svg_localization, over byte-identical pixels. No image was re-rendered, so a
# model can be scored on all three sets and the differences are the task, not
# the input.
#
# counting  -> "how many bars are in this bar chart" -- exact integer, and the
#              SIGNED error is the point: undercounting is losing track in a
#              crowd, overcounting is estimating a repeating pattern.
# word_mc   -> "which of these four words appears in the figure" -- isolates
#              reading from localization, since the answer has no spatial
#              component at all.
# --------------------------------------------------------------------------

# EVAL.md 5: group the 19 question forms by what is being counted. Connections
# have no enclosing shape to anchor on and are where undercounting should show
# up first if it shows up at all.
_COUNT_FAMILY = {
    "boxes": "objects", "bars": "objects", "lines": "objects", "points": "objects",
    "slices": "objects", "nodes": "objects", "states": "objects",
    "rectangles": "objects", "branches": "objects", "panels": "objects",
    "participants": "objects", "milestones": "objects", "labels": "objects",
    "rows": "rows", "columns": "rows", "tasks": "rows",
    "messages": "connections", "arrows": "connections",
}


def _count_family(question: str) -> str:
    q = question.lower()
    if "message arrows" in q or "arrows" in q:
        return "connections"
    if "task rows" in q or "data rows" in q or "columns" in q:
        return "rows"
    return "objects"


def _svg_derived_common(r: dict) -> dict:
    return {
        "qtype": r["qtype"],
        "graph_id": r["graph_id"],
        "resolution": r["resolution"],
        "chart_type": r["chart_type"],
        "theme": r["theme"],
        "font_family": r["font_family"],
        "complexity": r.get("complexity"),
        "img_size": [int(v) for v in r["image_px"]],
        "effective_px": [int(v) for v in r["effective_px"]],
        "downscaled_by_api": r.get("downscaled_by_api"),
    }


def svg_counting() -> Iterator[Example]:
    for r in _rows("svg_localization/counting"):
        yield Example(
            uid=r["uid"], dataset="svg_counting",
            images=[_img("svg_localization", r["image"])],
            question=r["question"],
            answer_type="count",
            gold=[int(r["answer"])],
            meta={**_svg_derived_common(r),
                  "true_count": int(r["true_count"]),
                  "question_form": r["question"],
                  "count_family": _count_family(r["question"]),
                  "cross_checked_against": r.get("cross_checked_against")},
        )


def svg_word_mc() -> Iterator[Example]:
    for r in _rows("svg_localization/word_mc"):
        yield Example(
            uid=r["uid"], dataset="svg_word_mc",
            images=[_img("svg_localization", r["image"])],
            question=r["question"],
            answer_type="choice",
            gold=[str(r["answer"])],
            meta={**_svg_derived_common(r),
                  "options": list(r["options"]),          # prompts.py builds A./B./C./D.
                  "answer_text": r.get("answer_text"),
                  "distractors": list(r.get("distractors") or []),
                  "answer_len": len(str(r.get("answer_text") or ""))},
        )


ADAPTERS["svg_counting"] = svg_counting
ADAPTERS["svg_word_mc"] = svg_word_mc
