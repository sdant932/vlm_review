"""Core of the Haiku 4.5 perception blind-spot study.

The layer everything else builds on, and the only one with no upward
dependencies. Sections below run in dependency order behind
`# ---- section ----` banners.

`blindspot.charxiv` stays outside: it is vendored verbatim from the CharXiv
authors and must remain uneditable, so it is imported rather than inlined.

Each section records the decision behind it, because every one of these is
something a later change can undo by accident:

adapters
    One `Example` == one scoreable question, which is *not* one manifest row:
    CharXiv ships five questions per figure (four descriptive + one reasoning),
    so its row count and its question count differ by ~5x. `answer_type` drives
    both prompt construction and scoring, so it is the field to look at when
    adding a dataset:

        span   -> short text answer, graded by ANLS against one or more golds
        point  -> a click location, graded by whether it lands inside a gold bbox

    Gold bboxes are ALWAYS stored normalized to [0, 1] as (x0, y0, x1, y1),
    regardless of how the source dataset encoded them.

prompts
    CharXiv descriptive prompts already carry their own answer-format rules
    (vendored verbatim); adding our own instructions on top would change the
    task and make the numbers incomparable to CharXiv's published setup, so
    those questions are sent through untouched. Answers come back via structured
    outputs rather than a regex over free text -- with thinking enabled the model
    reasons in its thinking block and emits only schema-conforming JSON, which
    removes the whole class of bugs that come from parsing prose answers.

runner
    Metered, resumable, and safe to kill at any point. A plain API key cannot
    read credit balance or spend caps (every /v1/organizations/* endpoint returns
    401), so the harness meters itself and stops at `--max-spend` rather than
    discovering the ceiling as a mid-run 400. Billing failures are fatal, not
    retryable. Every run is resumable: results append to JSONL keyed by uid and a
    rerun skips what is already there. And the whole thing is non-deterministic
    by construction -- `temperature` is unavailable in anthropic 1.0.0 and
    thinking pins it to 1 regardless -- so `--repeat`/`--run` exists to measure
    run-to-run variance instead of pretending scores are exact.

scoring
    Each scorer follows its benchmark's OWN published metric so the numbers stay
    comparable to published work:

        InfographicVQA  ANLS against the best-matching gold (threshold 0.5)
        CharXiv         normalized match; numeric-aware where the answer is a value
        ScreenSpot/-Pro click-in-bbox accuracy
        SlideVQA        exact match + token F1
        svg_localization token F1 (EM alongside), never substring containment

    Each benchmark's caveat, at comparable weight -- none of these is the
    "main" one:

        CharXiv         the official grader is an LLM judge with per-question-type
                        rubrics. String matching here is a LOWER BOUND on the
                        free-text descriptive types; a right answer phrased
                        differently scores wrong. `charxiv_grading_confidence()`
                        says which side of that line a question falls on, and the
                        report separates them rather than pooling.
        AI2D            the largest threat to any perception claim in this study:
                        with the diagram REMOVED it still scores 62.7% against
                        25% for guessing, so most of the number does not need the
                        image. Treat it as a lower bound on language priors, not a
                        measure of diagram reading.
        SlideVQA        token F1 gives zero to "22%" when the key says "22". A
                        quarter of its questions score zero and 47% of those are
                        the right value written differently.
        InfographicVQA  ANLS at threshold 0.5 against the best-matching gold; a
                        near-miss below threshold scores exactly zero.
        ScreenSpot-Pro  click-in-bbox is all-or-nothing, so it cannot distinguish
                        a near miss from a wild one. `run_api coord-probe` and the
                        distance bands exist because of that.

sampling
    Stratify by the cell you intend to report. The pilot sampled CharXiv by
    *figure*; each figure contributes four of nineteen randomly-chosen question
    types, so a 200-figure sample produced per-question-type counts of 3 to 16 --
    numbers with no statistical content that nevertheless rendered as confident
    bars ("count lines: 100%, n=3").

stats
    Small statistics and coarse-grid helpers shared across the analysis layer.
    They belong here rather than beside a renderer: computing a confidence
    interval must not require importing a page builder, and a number the report
    quotes has to be checkable without running the report.

taxonomy
    One primitive taxonomy across datasets -- the spine of the whole report. The
    brief asks which *perceptual primitives* underlie business visual tasks and
    where they break; a per-dataset accuracy table cannot answer that, so the
    same primitive has to be measurable across independent sources. Every mapping
    is either `construction` (the task itself guarantees the primitive) or
    `dataset_label` (the dataset's own human annotation). There is deliberately
    no keyword-guessed tier: a regex over question text would look identical in
    the matrix while being materially less trustworthy.

failure_modes
    Classify *why* an answer was scored wrong. A single accuracy number treats an
    instruction-following failure on a question whose perception succeeded as
    identical to a perception failure, which overstates how much the model failed
    to *see*. List cases are decided deterministically; anything needing semantic
    judgement is left `unclassified` for the LLM pass.

mermaid
    Parse FlowLearn's Mermaid ground truth into a graph and score against it. The
    Mermaid source is the only *complete* description of each figure, so golds
    are derived from the parsed graph rather than trusted from the shipped QA
    labels -- which also lets us measure how often those labels disagree with the
    figure.

Usage:
    python -m blindspot.core --datasets screenspot --limit 20 --max-spend 1
    python -m blindspot.core --datasets screenspot_pro --max-edge 1568   # ablation
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import anthropic
from PIL import Image

# The one part of core that is NOT merged: vendored verbatim from the CharXiv
# authors, and must stay byte-identical to theirs. Re-exported here because
# `adapters` re-exported it and callers import these names off core.
from blindspot.charxiv import (DESCRIPTIVE_RESP_INST, DESCRIPTIVE_GRADING_QMAP,
                                                     reasoning_question)


# ---- shared paths and IO helpers ---------------------------------------------

DATA = Path("data")
RESULTS = Path("results")


def read_jsonl(path) -> list[dict]:
    """Read a JSONL file tolerantly: missing file -> [], torn/bad line -> skipped.

    One copy of what had drifted into six near-identical private helpers across
    the analysis and reporting layers (`read_jsonl` in svgderived_eval and
    svgloc_eval, `_rows` in svgloc_ablation_eval and report_data, and friends).
    They all agreed on the semantics -- a results JSONL is written by a run that
    can be killed mid-line, and a half-written final line must not take down an
    analysis -- so there is no reason for six of them.

    NOT to be confused with `_rows(dataset)` below, which is a *generator* over
    DATA/<dataset>/manifest.jsonl and is strict about its input. Different thing,
    unfortunately similar name; kept private for exactly that reason.
    """
    out: list[dict] = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


# ==============================================================================
# ---- mermaid: FlowLearn ground truth -> graph --------------------------------
# ==============================================================================
#
# FlowLearn ships four QA fields per flowchart (Arrow_AtoB, Arrow_betweenAB,
# Num_Nodes, Num_Arrows) but the Mermaid source is the only *complete*
# description of the figure. Deriving golds from the parsed graph rather than
# trusting the QA labels lets us (a) phrase questions to match the semantics we
# actually score, and (b) measure how often the shipped labels disagree with the
# figure -- which turns out to matter a great deal for Arrow_betweenAB, whose
# "false" pairs are not undirected-false at all.
#
# Mermaid we need to handle looks like:
#
#     ```mermaid
#     flowchart TB
#     entity0(outsteered karite)
#     entity1(dihedron cushite)
#     entity0 --> entity1
#     entity0 ==> entity4
#     entity1 -..-> entity2
#     ```
#
# This section leads the file because `flowlearn_sim` (adapters) uses it. In the
# split layout that import was function-local to keep adapters free of a
# core-internal edge; merged, it is purely a question of definition order.

# Node declaration: entity0(label), entity0[label], entity0{label}, entity0((label))
_NODE = re.compile(r"^\s*(\w+)\s*(?:\(\(|\(|\[|\{)(.*?)(?:\)\)|\)|\]|\})\s*$")
# Edge: entity0 --> entity1 / ==> / -..-> / -.-> / --- , optional |label|
_EDGE = re.compile(r"^\s*(\w+)\s*(-{2,}>|={2,}>|-\.+-?>|-{3,})\s*(?:\|[^|]*\|\s*)?(\w+)\s*$")
_DIR = re.compile(r"^\s*(?:flowchart|graph)\s+(TB|TD|BT|LR|RL)\s*$", re.I)


@dataclass(frozen=True)
class Graph:
    direction: str
    labels: dict[str, str]                       # entity id -> human label
    edges: tuple[tuple[str, str, str], ...]      # (src, dst, style)

    @property
    def n_nodes(self) -> int:
        return len(self.labels)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def parse_mermaid(src: str) -> Graph:
    direction, labels, edges = "TB", {}, []
    for raw in (src or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        m = _DIR.match(line)
        if m:
            direction = m.group(1).upper()
            continue
        m = _EDGE.match(line)
        if m:
            src_id, style, dst_id = m.group(1), m.group(2), m.group(3)
            edges.append((src_id, dst_id, _norm_style(style)))
            continue
        m = _NODE.match(line)
        if m:
            labels[m.group(1)] = m.group(2).strip()
    return Graph(direction, labels, tuple(edges))


def _norm_style(s: str) -> str:
    if s.startswith("="):
        return "thick"       # ==>
    if "." in s:
        return "dotted"      # -..-> / -.->
    if s.endswith(">"):
        return "solid"       # -->
    return "open"            # ---


def by_label(g: Graph) -> dict[str, str]:
    """label -> entity id. Labels are unique in this corpus (verified)."""
    return {v: k for k, v in g.labels.items()}


def has_edge(g: Graph, a_label: str, b_label: str, directed: bool = True) -> bool | None:
    """True/False, or None when either label does not appear in the figure."""
    idx = by_label(g)
    a, b = idx.get(a_label), idx.get(b_label)
    if a is None or b is None:
        return None
    for s, d, _ in g.edges:
        if s == a and d == b:
            return True
        if not directed and s == b and d == a:
            return True
    return False


def edge_style(g: Graph, a_label: str, b_label: str) -> str | None:
    idx = by_label(g)
    a, b = idx.get(a_label), idx.get(b_label)
    for s, d, st in g.edges:
        if (s, d) == (a, b) or (s, d) == (b, a):
            return st
    return None


def hops(g: Graph, a_label: str, b_label: str, directed: bool = False) -> int | None:
    """Shortest path length in edges; None if unreachable or a label is absent."""
    idx = by_label(g)
    a, b = idx.get(a_label), idx.get(b_label)
    if a is None or b is None:
        return None
    if a == b:
        return 0
    adj: dict[str, set[str]] = {}
    for s, d, _ in g.edges:
        adj.setdefault(s, set()).add(d)
        if not directed:
            adj.setdefault(d, set()).add(s)
    seen, frontier, dist = {a}, [a], 0
    while frontier:
        dist += 1
        nxt = []
        for n in frontier:
            for m in adj.get(n, ()):
                if m == b:
                    return dist
                if m not in seen:
                    seen.add(m)
                    nxt.append(m)
        frontier = nxt
    return None


def graph_f1(gold: Graph, pred: Graph) -> dict:
    """Node/edge F1 against a predicted graph, compared on labels not ids."""
    gl, pl = set(gold.labels.values()), set(pred.labels.values())
    gi, pi = gold.labels, pred.labels
    ge = {(gi.get(s, s), gi.get(d, d)) for s, d, _ in gold.edges}
    pe = {(pi.get(s, s), pi.get(d, d)) for s, d, _ in pred.edges}

    def f1(g_set, p_set):
        if not g_set and not p_set:
            return 1.0
        tp = len(g_set & p_set)
        if not tp:
            return 0.0
        prec, rec = tp / len(p_set), tp / len(g_set)
        return 2 * prec * rec / (prec + rec)

    nf, ef = f1(gl, pl), f1(ge, pe)
    return {"node_f1": nf, "edge_f1": ef,
            "exact": float(gl == pl and ge == pe),
            "score": ef}


# ==============================================================================
# ---- adapters: each dataset's manifest -> a common `Example` -----------------
# ==============================================================================
#
# One `Example` == one scoreable question, not one manifest row (CharXiv ships
# five questions per figure). `answer_type` drives both prompt construction and
# scoring, so it is the field to look at when adding a dataset:
#
#     span   -> short text answer, graded by ANLS against one or more golds
#     point  -> a click location, graded by whether it lands inside a gold bbox
#
# Gold bboxes are always stored normalized to [0, 1] as (x0, y0, x1, y1),
# regardless of how the source dataset encoded them.


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
#
# `parse_mermaid` / `has_edge` / `edge_style` / `hops` are defined above
# function-locally, to keep adapters free of a core-internal import edge. Now
# that mermaid lives above in this same file, that is just ordering.
# --------------------------------------------------------------------------
def flowlearn_sim() -> Iterator[Example]:

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
                  "options": list(r["options"]),          # prompts section builds A./B./C./D.
                  "answer_text": r.get("answer_text"),
                  "distractors": list(r.get("distractors") or []),
                  "answer_len": len(str(r.get("answer_text") or ""))},
        )


ADAPTERS["svg_counting"] = svg_counting
ADAPTERS["svg_word_mc"] = svg_word_mc


# ==============================================================================
# ---- prompts: request construction and response schemas, by `answer_type` ----
# ==============================================================================
#
# Two rules drive everything here:
#
# 1. CharXiv descriptive prompts already carry their own answer-format rules
#    (vendored verbatim). Adding our own instructions on top would change the
#    task and make the numbers incomparable to CharXiv's published setup, so
#    those questions are sent through untouched.
#
# 2. Answers come back via structured outputs rather than a regex over free
#    text. With thinking enabled the model reasons in its thinking block and
#    emits only the schema-conforming JSON, which removes the whole class of
#    bugs that come from parsing prose answers.

# Haiku 4.5 downsizes anything larger to roughly this long edge (~1568 image
# tokens), measured directly against count_tokens. Sending native-resolution
# 4K screenshots therefore buys nothing but upload time -- but we still send
# native by default so the downscale ablation has something to compare against.
HAIKU_MAX_EDGE = 1568

# Hard API ingestion limits, hit for real during the pilot:
#   - a 2534x8369 InfographicVQA page -> 400 "image dimensions exceed max allowed
#     size: 8000 pixels"
#   - 5120x2880 Retina ScreenSpot-Pro screenshots -> 400 "image exceeds 10 MB"
# These reject the request outright, before the model sees anything. Since Haiku
# downscales to ~1568px regardless, shrinking to fit costs no model-visible
# fidelity -- but skipping it silently drops the largest, most interesting images
# from the eval, which would bias exactly the cases we care about.
API_MAX_DIM = 8000
API_MAX_B64_BYTES = 9_500_000  # margin under the 10 MB ceiling

MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
               ".gif": "image/gif", ".webp": "image/webp"}

SCHEMAS: dict[str, dict] = {
    "span": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    },
    "bbox": {
        "type": "object",
        "properties": {
            "x0": {"type": "integer"}, "y0": {"type": "integer"},
            "x1": {"type": "integer"}, "y1": {"type": "integer"},
        },
        "required": ["x0", "y0", "x1", "y1"],
        "additionalProperties": False,
    },
    "point": {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
        },
        "required": ["x", "y"],
        "additionalProperties": False,
    },
    # Constrained to the enum so a hedged "it appears there may be" can never
    # reach the scorer -- the model must commit.
    "boolean": {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
        "required": ["answer"],
        "additionalProperties": False,
    },
    # No `minimum`: structured outputs reject numerical constraints
    # ("For 'integer' type, property 'minimum' is not supported"). The count
    # scorer treats a negative as simply wrong, which is the right behaviour.
    # Multiple choice removes the answer-expression confound entirely: the model
    # picks rather than phrases, so a correct reading cannot be scored wrong for
    # wording -- which is the failure mode that costs InfographicVQA ~8 points.
    "choice": {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},
        "required": ["answer"],
        "additionalProperties": False,
    },
    "count": {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    },
}

SPAN_INSTRUCTION = (
    "Answer the question using only what is visible in the image. "
    "Respond with the shortest exact answer -- a value, name, or phrase copied "
    "from the image. Do not explain, and do not restate the question."
)

BOOLEAN_INSTRUCTION = (
    "Answer strictly yes or no, based only on what is drawn in the image.\n"
    "Commit to one answer even if you are uncertain.\n\n"
)

CHOICE_INSTRUCTION = (
    "Answer with the single letter of the correct option.\n"
    "Base your answer only on what is shown in the diagram.\n\n"
)

COUNT_INSTRUCTION = (
    "Count carefully and answer with a single whole number.\n"
    "Count only what is actually drawn in the image.\n\n"
)

# Coordinates are requested in a 0-1000 normalized space rather than pixels:
# the model never sees the native resolution (the API downscales first), so
# asking for pixel coordinates in an unknown coordinate space would inject an
# avoidable source of error into a localization measurement.
POINT_INSTRUCTION = (
    "Locate the described UI element in the screenshot and return the point at "
    "its center.\n"
    "Use a normalized coordinate system where x=0 is the left edge, x=1000 the "
    "right edge, y=0 the top edge, and y=1000 the bottom edge.\n"
    "Always return your single best guess, even if you are uncertain.\n\n"
    "Element: "
)


def encode_image(path: str, max_edge: int | None = None) -> tuple[str, str, tuple[int, int], bool]:
    """Return (base64, media_type, (width, height) as sent, was_downscaled).

    `max_edge` pre-downscales client-side. That is the lever for the resolution
    ablation: if scores are unchanged at max_edge=1568, a localization failure is
    perceptual rather than an artifact of the API resizing a 4K screenshot out
    from under the model.

    With `max_edge=None` the original is sent untouched *unless* it would breach
    an API ingestion limit, in which case it is shrunk just enough to pass.
    """
    p = Path(path)

    if max_edge is None:
        with Image.open(p) as im:
            size = im.size
        raw = p.read_bytes()
        b64 = base64.b64encode(raw).decode()
        if max(size) <= API_MAX_DIM and len(b64) <= API_MAX_B64_BYTES:
            return b64, MEDIA_TYPES[p.suffix.lower()], size, False
        # Too big to ingest: fall through and shrink to fit.

    with Image.open(p) as im:
        im = im.convert("RGB")
        original = im.size
        target = min(max_edge or API_MAX_DIM, API_MAX_DIM)
        if max(im.size) > target:
            scale = target / max(im.size)
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                           Image.LANCZOS)

        # Shrink until the encoded payload fits, rather than guessing a quality.
        for _ in range(8):
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode()
            if len(b64) <= API_MAX_B64_BYTES:
                return b64, "image/jpeg", im.size, im.size != original
            im = im.resize((max(1, int(im.width * 0.8)), max(1, int(im.height * 0.8))), Image.LANCZOS)

        raise ValueError(f"{path}: cannot fit under API image limits")


def prompt_text(ex: Example) -> str:
    """The exact text block sent alongside the image.

    Split out of build_request so tooling can show the real prompt without
    paying to re-encode the image -- and so the two can never drift apart.

    `meta["prompt_override"]` replaces the whole block. It exists so a prompt
    ablation can vary the wording while every other part of the request --
    image encoding, schema, model, thinking budget -- stays byte-identical.
    """
    ov = ex.meta.get("prompt_override")
    if ov:
        return ov
    if ex.answer_type == "point":
        return POINT_INSTRUCTION + ex.question
    if ex.answer_type == "boolean":
        return BOOLEAN_INSTRUCTION + ex.question
    if ex.answer_type == "count":
        return COUNT_INSTRUCTION + ex.question
    if ex.answer_type == "choice":
        opts = "\n".join(f"{k}. {v}" for k, v in zip("ABCD", ex.meta.get("options", [])))
        return f"{CHOICE_INSTRUCTION}{ex.question}\n\n{opts}"
    if ex.dataset == "svg_localization":
        return ex.question  # already self-contained and states its own answer format
    if ex.dataset == "charxiv" and ex.meta.get("split") == "descriptive":
        return ex.question  # vendored template already specifies the answer format
    return f"{ex.question}\n\n{SPAN_INSTRUCTION}"


def build_request(ex: Example, max_edge: int | None = None) -> tuple[list[dict], dict, list[tuple[int, int]], bool]:
    """Return (message content blocks, response schema, as-sent sizes, any_downscaled)."""
    content: list[dict] = []
    sizes: list[tuple[int, int]] = []
    downscaled = False
    for path in ex.images:
        b64, media_type, size, shrunk = encode_image(path, max_edge)
        sizes.append(size)
        downscaled |= shrunk
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        })

    content.append({"type": "text", "text": prompt_text(ex)})
    return content, SCHEMAS[ex.answer_type], sizes, downscaled


def parse_response(ex: Example, raw: str) -> Any:
    """Convert the schema-conforming JSON into the value the scorer expects."""
    import json

    obj = json.loads(raw)
    if ex.answer_type == "point":
        return (obj["x"] / 1000.0, obj["y"] / 1000.0)
    if ex.answer_type == "bbox":
        return (obj["x0"] / 1000.0, obj["y0"] / 1000.0,
                obj["x1"] / 1000.0, obj["y1"] / 1000.0)
    if ex.answer_type == "count":
        return int(obj["answer"])
    if ex.answer_type == "boolean":
        return str(obj["answer"]).strip().lower()
    if ex.answer_type == "choice":
        return str(obj["answer"]).strip().upper()
    return obj["answer"]


# ==============================================================================
# ---- sampling: stratify by the primitive cell you intend to report -----------
# ==============================================================================
#
# The pilot sampled CharXiv by *figure* with `random.sample(examples, limit)`.
# Each figure contributes four of nineteen randomly-chosen question types, so a
# 200-figure sample produced per-question-type counts of 3 to 16 -- numbers with
# no statistical content that nevertheless rendered as confident bars ("count
# lines: 100%, n=3"). Sampling by the cell you intend to report is the fix.
#
# Cells smaller than `per_cell` contribute their whole pool; the realised n is
# returned so under-filled cells are reported rather than silently shipped.

# The slice each dataset is reported by. Keep in step with the report's axes:
# whatever you stratify on is what you can make claims about.
CELL_KEYS: dict[str, Callable[[Example], Any]] = {
    "charxiv": lambda e: e.meta.get("qid") or e.meta.get("split", "reasoning"),
    "infographicvqa": lambda e: tuple(e.meta.get("operation") or ["direct_lookup"])[0],
    "flowlearn_sim": lambda e: (e.meta.get("family"), e.meta.get("variant"),
                                e.meta.get("polarity")),
    "screenspot_pro": lambda e: (e.meta.get("ui_type"), _area_bucket(e)),
    "ai2d": lambda e: e.meta.get("qtype"),
    "slidevqa": lambda e: ("multi-page" if e.meta.get("multi_page") else "single-page",
                           "arithmetic" if e.meta.get("arithmetic") else "lookup"),
    "slidevqa_allpages": lambda e: ("multi-page" if e.meta.get("multi_page") else "single-page",
                                    "arithmetic" if e.meta.get("arithmetic") else "lookup"),
    "screenspot": lambda e: (e.meta.get("ui_type"), _area_bucket(e)),
}


def _area_bucket(e: Example) -> str:
    """Target size as it reaches the model, after the API's 1568px cap."""
    import math
    frac = e.meta.get("target_area_frac", 0) or 0
    side = math.sqrt(max(frac, 0) * 1568 * 882)
    for lim, name in ((12, "<12px"), (20, "12-20px"), (32, "20-32px"), (56, "32-56px")):
        if side < lim:
            return name
    return ">=56px"


def cell_key_for(dataset: str) -> Callable[[Example], Any]:
    return CELL_KEYS.get(dataset, lambda e: e.dataset)


def stratify(examples: Iterable[Example], key_fn: Callable[[Example], Any],
             per_cell: int = 300, seed: int = 0) -> tuple[list[Example], dict[Any, tuple[int, int]]]:
    """Return (sampled examples, {cell: (taken, pool_size)}).

    Deterministic for a given seed so a resumed run selects the same questions.
    """
    cells: dict[Any, list[Example]] = defaultdict(list)
    for e in examples:
        cells[key_fn(e)].append(e)

    rng = random.Random(seed)
    out: list[Example] = []
    realised: dict[Any, tuple[int, int]] = {}
    for cell in sorted(cells, key=lambda c: str(c)):
        pool = cells[cell]
        take = pool if len(pool) <= per_cell else rng.sample(pool, per_cell)
        out.extend(take)
        realised[cell] = (len(take), len(pool))
    return out, realised


def report_cells(dataset: str, realised: dict[Any, tuple[int, int]], min_n: int = 30) -> None:
    """Print realised n per cell and warn on any too small to interpret."""
    thin = [(c, t) for c, (t, _) in realised.items() if t < min_n]
    total = sum(t for t, _ in realised.values())
    print(f"  {dataset}: {total} questions across {len(realised)} cells")
    for cell, (took, pool) in sorted(realised.items(), key=lambda kv: kv[1][0]):
        mark = "  <-- thin" if took < min_n else ""
        print(f"    {str(cell):<44} n={took:>4} / pool {pool}{mark}")
    if thin:
        print(f"  !! {len(thin)} cell(s) below n={min_n}; report them as indicative only")


# ==============================================================================
# ---- taxonomy: one primitive vocabulary across datasets ----------------------
# ==============================================================================
#
# The spine of the whole report. The brief asks which *perceptual primitives*
# underlie business visual tasks and where they break. A per-dataset accuracy
# table cannot answer that; the same primitive has to be measurable across
# independent sources so agreement between them is evidence about the model
# rather than about one benchmark.
#
# Every mapping here is either:
#
#   construction    the task itself guarantees the primitive (CharXiv's templates
#                   are generated per type; ScreenSpot-Pro is localization by
#                   definition)
#   dataset_label   the dataset's own human annotation (InfographicVQA's
#                   operation/reasoning labels)
#
# There is deliberately no keyword-guessed tier: a regex over question text would
# look identical in the matrix while being materially less trustworthy, and
# pooling the two would make the headline chart partly fiction.

PRIMITIVES = (
    "counting", "line_following", "localization_read", "localization_point",
    "value_interpolation",
    "binding", "structure", "text_in_situ", "comparison", "arithmetic",
    "composition",
)

# Human-readable, in the order the report should show them.
#
# NAME COLLISION, resolved deliberately: this was `taxonomy.LABELS`, and the
# failure-mode section below has its own `LABELS`. Two dicts cannot share one
# name in a merged module, so each gets an explicit name and the bare `LABELS`
# alias stays bound to THIS one -- `aggregate.py` iterates it to enumerate the
# primitives (`for prim in LABELS`), which silently produces a wrong report if
# it gets the other dict, whereas `judge.py`'s `LABELS.get(mode, mode)` merely
# prints raw mode names. Prefer the explicit names in new code.
PRIMITIVE_LABELS = {
    "counting": "Counting objects",
    "line_following": "Following lines",
    # Two genuinely different tasks, split apart after they diverged 91% vs 4%.
    # CharXiv asks you to find a spatial extreme and *read what is written there*
    # -- the answer is a string. ScreenSpot-Pro asks you to find an element and
    # *emit its coordinates* -- the answer is a number pair, graded by whether
    # the point lands inside a box. Calling both "localization" hid the fact
    # that one of them additionally requires expressing position numerically.
    "localization_read": "Localization → read the value there",
    "localization_point": "Localization → emit a coordinate",
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


# ==============================================================================
# ---- failure_modes: classify *why* an answer was scored wrong ----------------
# ==============================================================================
#
# A single accuracy number treats these as identical failures:
#
#     gold "A, B, C"  pred "C, B, A"          -> read everything, ignored the order instruction
#     gold "A, B, C"  pred "A, B, C, D"       -> read everything, plus something that isn't there
#     gold "A, B, C"  pred "A, B"             -> missed one
#     gold "310.5"    pred "310.5 million"    -> right value, wrong format
#     gold "A, B, C"  pred "X, Y"             -> actually wrong
#
# They are not the same finding. The first is an instruction-following failure on
# a question whose perception succeeded; the last is a perception failure.
# Reporting them together overstates how much the model failed to *see*.
#
# Order matters here specifically because several CharXiv templates ask for it --
# qid 13 is literally "(from top to bottom, then left to right)" -- so listing the
# right items in the wrong sequence is disobeying an instruction, not misreading
# the figure.
#
# List cases are decided deterministically (set and sequence comparison); anything
# that needs semantic judgement is left as `unclassified` for the LLM pass.

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

# See the note on PRIMITIVE_LABELS above: this was `failure_modes.LABELS`, and
# the bare `LABELS` alias is bound to the taxonomy dict, not this one.
FAILURE_MODE_LABELS = {
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

# The historical spelling. Bound to the taxonomy dict; see PRIMITIVE_LABELS.
LABELS = PRIMITIVE_LABELS

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


# ==============================================================================
# ---- scoring: each benchmark's own published metric --------------------------
# ==============================================================================
#
# Metrics follow each benchmark's own convention so numbers stay comparable to
# published work:
#
#     InfographicVQA  ANLS against the best-matching gold (threshold 0.5)
#     CharXiv         normalized match; numeric-aware where the answer is a value
#     ScreenSpot/-Pro click-in-bbox accuracy
#
# One honest caveat, surfaced rather than buried: CharXiv's official grader is an
# LLM judge with per-question-type rubrics ("same term, different form scores 1").
# This harness does not run a judge, so free-text descriptive types (title, axis
# labels, legend contents, trend) are graded approximately and their scores are a
# **lower bound** -- a correct answer phrased differently can be marked wrong.
# `charxiv_grading_confidence()` reports which side of that line each question
# falls on, and the report separates them instead of pooling them into one number.

# CharXiv descriptive question types whose answers are values, counts, or a
# closed set -- normalized matching is reliable here.
CHARXIV_STRICT_QIDS = {4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18, 19}
# ...and those whose answers are free text, where strict matching undercounts.
CHARXIV_FUZZY_QIDS = {1, 2, 3, 13, 16}

ANLS_THRESHOLD = 0.5


def charxiv_grading_confidence(qid: int | None) -> str:
    if qid is None:
        return "fuzzy"  # reasoning split: short free-text answers
    return "strict" if qid in CHARXIV_STRICT_QIDS else "fuzzy"


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _normalize(s: Any) -> str:
    return " ".join(str(s).strip().lower().replace(",", "").split())


def _as_float(s: Any) -> float | None:
    t = _normalize(s).rstrip("%").replace("$", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


def _anls_normalize(s: Any) -> str:
    """Lowercase + strip, exactly as the official DocVQA/InfographicVQA script.

    Deliberately NOT `_normalize`: that one also strips commas, which would score
    "1,000" against "1000" as a perfect match. That is more lenient than the
    official metric, and being more lenient than the benchmark is just as wrong
    as being stricter.
    """
    return str(s).lower().strip()


def anls(pred: str, golds: list[str], threshold: float = ANLS_THRESHOLD) -> float:
    """Official ANLS: 1 - NL against the best gold, zeroed below the threshold.

    Follows the published definition: NL = levenshtein / max(len); the score is
    1 - NL when NL < tau (tau = 0.5), otherwise 0; take the max over ground
    truths. Note the boundary is strict -- a normalized distance of exactly 0.5
    scores 0, not 0.5.
    """
    p = _anls_normalize(pred)
    best = 0.0
    for g in golds:
        g = _anls_normalize(g)
        if not p and not g:
            best = max(best, 1.0)
            continue
        denom = max(len(p), len(g))
        if not denom:
            continue
        nl = _levenshtein(p, g) / denom
        best = max(best, 1.0 - nl if nl < threshold else 0.0)
    return best


def numeric_or_text_match(pred: str, golds: list[str]) -> float:
    """Exact match after normalization, comparing numerically when both sides parse.

    Numeric comparison uses a relative tolerance so that 0.28 == 0.280 == "0.28 "
    without letting genuinely different values through.
    """
    pf = _as_float(pred)
    for g in golds:
        gf = _as_float(g)
        if pf is not None and gf is not None:
            if abs(pf - gf) <= 1e-6 * max(1.0, abs(gf)):
                return 1.0
        elif _normalize(pred) == _normalize(g):
            return 1.0
    return 0.0


def boolean_match(pred: Any, golds: list) -> float:
    """Exact yes/no match.

    A raw accuracy number is not sufficient for these families and the report
    must never show one alone: FlowLearn's arrow probes are balanced by
    construction (one matched positive and negative per figure), so a model that
    answers "yes" to everything scores ~50% while having perceived nothing. The
    runner records polarity so the report can show balanced accuracy and the
    yes-rate alongside.
    """
    p = str(pred).strip().lower()
    return 1.0 if any(p == str(g).strip().lower() for g in golds) else 0.0


def count_score(pred: Any, golds: list) -> dict:
    """Exact-count accuracy plus the signed error.

    Signed error is the interesting half: consistent undercounting as object
    count rises is a different failure from noisy counting, and only the sign
    distinguishes them.
    """
    try:
        p = int(pred)
        g = int(golds[0])
    except (TypeError, ValueError, IndexError):
        return {"score": 0.0, "abs_error": None, "signed_error": None}
    return {"score": float(p == g), "abs_error": abs(p - g), "signed_error": p - g}


def token_f1(pred: Any, golds: list) -> tuple[float, float]:
    """(EM, F1) over normalized tokens -- SlideVQA's official pair of metrics.

    Not ANLS: SlideVQA reports exact match and token-level F1, so using ANLS here
    would make the number incomparable to the published results.
    """
    import collections
    def toks(x):
        return [t for t in _anls_normalize(x).replace(",", " ").split() if t]
    p = toks(pred)
    best_em = best_f1 = 0.0
    for g in golds:
        gt = toks(g)
        best_em = max(best_em, float(p == gt))
        if not p or not gt:
            best_f1 = max(best_f1, float(p == gt))
            continue
        common = collections.Counter(p) & collections.Counter(gt)
        n = sum(common.values())
        if n:
            prec, rec = n / len(p), n / len(gt)
            best_f1 = max(best_f1, 2 * prec * rec / (prec + rec))
    return best_em, best_f1


def point_in_bbox(pred: tuple[float, float], bbox: list[float]) -> float:
    x, y = pred
    x0, y0, x1, y1 = bbox
    return 1.0 if (x0 <= x <= x1 and y0 <= y <= y1) else 0.0


def score(example, pred: Any) -> dict:
    """Score one prediction. Returns the value plus how it was obtained."""
    ds, meta = example.dataset, example.meta

    if example.answer_type == "point":
        correct = point_in_bbox(pred, example.gold)
        x, y = pred
        x0, y0, x1, y1 = example.gold
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return {
            "score": correct,
            "metric": "click_in_bbox",
            "grading_confidence": "strict",
            # Distance to target center, for asking "near miss or nowhere near?"
            "center_distance": ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5,
        }

    if example.answer_type == "bbox":
        # Scored as centre-of-predicted-box inside the gold box, so the number
        # is directly comparable to click-in-bbox rather than a new unit. IoU
        # would be a different metric and EVAL.md forbids mixing the two.
        x0, y0, x1, y1 = pred
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return {"score": point_in_bbox((cx, cy), example.gold),
                "metric": "bbox_centre_in_gold", "grading_confidence": "strict",
                "pred_centre": [cx, cy], "pred_area": abs((x1 - x0) * (y1 - y0))}

    if example.answer_type == "boolean":
        return {"score": boolean_match(pred, example.gold),
                "metric": "exact_yes_no", "grading_confidence": "strict",
                "polarity": meta.get("polarity"), "pred_yes": str(pred).strip().lower() == "yes"}

    if example.answer_type == "choice":
        # Record the chosen letter so position bias is checkable: AI2D's answer
        # key is near-uniform across A-D, so a model that favours one slot would
        # show up here rather than hiding inside the accuracy number.
        return {"score": 1.0 if str(pred).strip().upper() == str(example.gold[0]).strip().upper() else 0.0,
                "metric": "multiple_choice", "grading_confidence": "strict",
                "picked": str(pred).strip().upper()}

    if example.answer_type == "count":
        r = count_score(pred, example.gold)
        return {**r, "metric": "exact_count", "grading_confidence": "strict",
                "true_count": meta.get("true_count")}

    if ds == "charxiv":
        conf = charxiv_grading_confidence(meta.get("qid"))
        if conf == "strict":
            return {"score": numeric_or_text_match(pred, example.gold),
                    "metric": "normalized_match", "grading_confidence": "strict"}
        return {"score": anls(pred, example.gold),
                "metric": "anls", "grading_confidence": "fuzzy"}

    if ds == "svg_localization":
        # EVAL.md 3.7: EM and token-F1 reported side by side, no substring
        # containment -- the labels are short and "Close" must not be credited
        # for "Close Ledger". token_f1 compares whole normalized tokens, so it
        # already refuses containment; F1 is the headline, EM rides alongside.
        em, f1 = token_f1(pred, example.gold)
        return {"score": f1, "exact_match": em, "metric": "svgloc_token_f1",
                "grading_confidence": "strict"}

    if ds in ("slidevqa", "slidevqa_allpages"):
        em, f1 = token_f1(pred, example.gold)
        # F1 is the headline in SlideVQA's own reporting; EM is kept alongside.
        return {"score": f1, "exact_match": em, "metric": "slidevqa_f1",
                "grading_confidence": "strict"}

    # InfographicVQA and anything else span-shaped
    return {"score": anls(pred, example.gold), "metric": "anls", "grading_confidence": "strict"}


# ==============================================================================
# ---- stats: confidence intervals, quantiles, coarse grid --------------------
# ==============================================================================
#
# Small statistics and coarse-grid helpers. They live in core, not beside a
# renderer: computing a confidence interval must not require importing a page
# builder, and a number the report quotes has to be checkable without running
# the report.
#
# NOTE for anyone comparing implementations: `wilson` here is the canonical one.
# Two divergent copies live OUTSIDE core and are deliberately not touched by this
# merge -- `analysis/aggregate.py:wilson(vals)` takes a list of per-row scores and
# clamps the interval to [0, 1], and `blindspot/diagnose.py:wilson(k, n, z)` returns
# percentages rather than fractions. Both are reachable; neither is this one.


def is_na(v) -> bool:
    """True for the 'Not Applicable' gold CharXiv uses for inapplicable questions."""
    return "not applicable" in str(v).strip().lower()


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval; shown wherever an n is small enough to matter."""
    if n == 0:
        return (0.0, 1.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def quantiles(rows, keyfn, k=5):
    """Split `rows` into k equal-count bins by `keyfn`, dropping rows keyed None.

    Returns (low, high, rows) per bin, so a caller can label the bin by its
    realised range rather than a nominal cut point.
    """
    rs = sorted((r for r in rows if keyfn(r) is not None), key=keyfn)
    out = []
    for i in range(k):
        chunk = rs[i * len(rs) // k:(i + 1) * len(rs) // k]
        if chunk:
            out.append((keyfn(chunk[0]), keyfn(chunk[-1]), chunk))
    return out


# --- coarse g x g grid over the normalized [0,1] image plane.
# Used to ask "did the prediction land in the right region?" when it missed the
# box outright -- a near miss and a miss to the far corner are different failures.

def cell_of(x, y, g):
    """Grid cell containing normalized point (x, y). Clamps at the far edge."""
    return (min(int(x * g), g - 1), min(int(y * g), g - 1))


def centre_cell(box, g):
    """Grid cell containing the centre of normalized box [x0,y0,x1,y1]."""
    return cell_of((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, g)


def bbox_cells(box, g):
    """Every grid cell a normalized box touches, not just the one holding its centre."""
    x0, y0, x1, y1 = box
    return {(i, j)
            for i in range(min(int(x0 * g), g - 1), min(int(x1 * g), g - 1) + 1)
            for j in range(min(int(y0 * g), g - 1), min(int(y1 * g), g - 1) + 1)}


# ==============================================================================
# ---- runner: metered, resumable, safe to kill at any point -------------------
# ==============================================================================
#
# Design constraints that shaped this, all of them learned the hard way earlier
# in this project:
#
# * **No budget visibility.** A plain API key cannot read credit balance or spend
#   caps (every /v1/organizations/* endpoint returns 401), so the harness meters
#   itself and stops at `--max-spend` rather than discovering the ceiling as a
#   mid-run 400.
# * **Billing failures are fatal, not retryable.** Backing off and retrying a
#   credit-balance error just burns minutes going nowhere, so it aborts the run.
# * **Every run is resumable.** Results append to JSONL keyed by uid; a rerun
#   skips what is already there. A crash costs the in-flight requests, not the run.
# * **Non-deterministic by construction.** `temperature` is unavailable in
#   anthropic 1.0.0, and thinking pins it to 1 regardless, so `--repeat` exists to
#   measure run-to-run variance instead of pretending scores are exact.

MODEL = "claude-haiku-4-5-20251001"

# Per-model pricing (USD per million tokens) and thinking dialect.
#
# The thinking dialect is not cosmetic: `budget_tokens` was REMOVED on the 4.6+
# generation and returns a 400 there, while `adaptive` does not exist on 4.5-era
# models. Sending the wrong one is an immediate hard failure, so the model
# registry owns this rather than the call site.
MODELS = {
    "claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00, "thinking": "budget"},
    "claude-haiku-4-5":          {"in": 1.00, "out": 5.00, "thinking": "budget"},
    "claude-sonnet-4-5":         {"in": 3.00, "out": 15.00, "thinking": "budget"},
    "claude-sonnet-5":           {"in": 3.00, "out": 15.00, "thinking": "adaptive"},
    "claude-opus-5":             {"in": 5.00, "out": 25.00, "thinking": "adaptive"},
}


def client(timeout: float = 120.0) -> anthropic.Anthropic:
    """The one way to build an API client for this project.

    max_retries=0: the SDK's default of 2 multiplies with the runner's own 3
    into six hidden attempts, which turns a 529 wave into a 6x wall-time
    blowout that is invisible in the logs. Retries belong in one place.

    timeout=120s (a plain float): the SDK default is 600s, but p90 latency is
    ~12s -- a stalled connection would otherwise park a worker for ten minutes
    with no signal. A plain float, deliberately: anthropic 1.0.0 rejects an
    httpx.Timeout object with "APIConnectionError: Connection error" before any
    request leaves the process -- which reads exactly like a network outage.

    Six modules outside core hand-roll this identical call (judge.py three
    times, controls/svgloc_ablations/run_svg_derived/svgloc_probe/grid_control
    once each). This is that call, in one place. Callers wanting a different
    read timeout -- judge.py's long-context pass uses 180s -- pass it here
    rather than re-deriving max_retries.
    """
    return anthropic.Anthropic(max_retries=0, timeout=timeout)


def model_spec(model: str) -> dict:
    if model not in MODELS:
        raise SystemExit(
            f"unknown model {model!r}; add it to MODELS with its pricing and "
            f"thinking dialect. Known: {', '.join(sorted(MODELS))}"
        )
    return MODELS[model]


def short_name(model: str) -> str:
    """Filename-safe tag so results from different models never collide."""
    return model.replace("claude-", "").replace("-20251001", "")


def thinking_config(model: str, budget: int) -> dict:
    return ({"type": "enabled", "budget_tokens": budget}
            if model_spec(model)["thinking"] == "budget"
            else {"type": "adaptive"})


class Budget:
    """Thread-safe spend tally with a hard ceiling."""

    def __init__(self, limit_usd: float | None):
        self.limit = limit_usd
        self.spent = 0.0
        self.calls = 0
        self._lock = threading.Lock()

    def add(self, in_tok: int, out_tok: int, model: str = MODEL) -> None:
        spec = model_spec(model)
        with self._lock:
            self.spent += in_tok / 1e6 * spec["in"] + out_tok / 1e6 * spec["out"]
            self.calls += 1

    def exhausted(self) -> bool:
        return self.limit is not None and self.spent >= self.limit


class FatalBillingError(RuntimeError):
    """Raised on credit-balance / spend-cap errors, which must not be retried."""


def is_billing_error(exc: Exception) -> bool:
    """True for errors that must abort the run rather than be retried.

    Public because it was never really private: `blindspot.run_api`
    reimplements the runner's loop deliberately and imported it across the module
    boundary as `_is_billing_error`. The underscore alias below keeps that import
    working; new call sites should use this name.
    """
    msg = str(exc).lower()
    return any(s in msg for s in ("credit balance", "billing", "spend limit", "quota"))


# Back-compat alias for the pre-merge private name. Same object, not a wrapper,
# so `is_billing_error is _is_billing_error` and monkeypatching either is visible
# to the other.
_is_billing_error = is_billing_error


def run_one(client, ex, budget: Budget, thinking_budget: int, max_edge: int | None,
            model: str = MODEL, max_retries: int = 3) -> dict:
    content, schema, sizes, preflight_downscaled = build_request(ex, max_edge)
    rec = {
        "uid": ex.uid,
        "dataset": ex.dataset,
        "answer_type": ex.answer_type,
        "gold": ex.gold,
        "meta": ex.meta,
        "sent_image_sizes": sizes,
        "preflight_downscaled": preflight_downscaled,
        "max_edge": max_edge,
        "thinking_budget": thinking_budget,
        "model": model,
    }

    for attempt in range(max_retries):
        try:
            t0 = time.monotonic()
            resp = client.messages.create(
                model=model,
                # Thinking output plus the JSON answer must both fit. 1024 tokens
                # of headroom was not enough -- two pilot rows came back with
                # stop_reason=max_tokens and a truncated answer after thinking ran
                # long, so a failure here silently reads as a model failure when
                # it is really a budgeting bug. 2048 still truncated 4 of the
                # hardest ScreenSpot-Pro localizations, hence 4096.
                max_tokens=thinking_budget + 4096,
                thinking=thinking_config(model, thinking_budget),
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": content}],
            )
            latency = time.monotonic() - t0
            budget.add(resp.usage.input_tokens, resp.usage.output_tokens, model)

            text = next((b.text for b in resp.content if b.type == "text"), None)
            thinking = next((b.thinking for b in resp.content if b.type == "thinking"), "")
            rec.update({
                "raw": text,
                "thinking": thinking,
                "stop_reason": resp.stop_reason,
                "request_id": resp._request_id,
                "latency_s": round(latency, 2),
                "usage": {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
            })
            try:
                rec["pred"] = parse_response(ex, text) if text else None
            except Exception as e:  # schema held but content was unusable
                rec["pred"] = None
                rec["parse_error"] = f"{type(e).__name__}: {e}"
            return rec

        except Exception as e:
            if is_billing_error(e):
                raise FatalBillingError(str(e)) from e
            if attempt == max_retries - 1:
                rec.update({"pred": None, "error": f"{type(e).__name__}: {e}"})
                return rec
            time.sleep(2**attempt + random.random())

    return rec


def existing_uids(path: Path) -> set[str]:
    """uids that already have a *usable* prediction.

    Anything without one -- an API error, a truncated answer, a parse failure --
    is deliberately excluded so a rerun retries it. Otherwise a transient 5xx or
    a since-fixed budgeting bug gets permanently baked into the results as if it
    were a model failure.
    """
    if not path.exists():
        return set()
    uids = set()
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue  # tolerate a torn final line from a killed run
            if not rec.get("error") and not rec.get("parse_error") and rec.get("pred") is not None:
                uids.add(rec["uid"])
    return uids


def run_dataset(client, dataset: str, args, budget: Budget) -> Path:
    model = getattr(args, "model", MODEL)
    res = "native" if args.max_edge is None else f"edge{args.max_edge}"
    # Model goes in the tag so a Sonnet control run can never overwrite, or be
    # silently pooled with, the Haiku results it exists to be compared against.
    tag = f"{short_name(model)}_think{args.thinking_budget}_{res}"
    out = RESULTS / f"{dataset}__{tag}_r{args.run}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    examples = load(dataset)
    uids = getattr(args, "uids", None)
    if uids:
        want = set(uids)
        examples = [e for e in examples if e.uid in want]
        missing = want - {e.uid for e in examples}
        if missing:
            raise SystemExit(f"{dataset}: uids not found: {sorted(missing)[:5]}")
    elif getattr(args, "per_cell", None):
        # Stratify by the slice we intend to report. Sampling by row instead
        # gave per-cell counts of 3-16 in the pilot -- noise rendered as
        # findings ("count lines: 100%, n=3"). See the sampling section.
        examples, realised = stratify(examples, cell_key_for(dataset),
                                      per_cell=args.per_cell, seed=args.seed)
        report_cells(dataset, realised)
    elif args.limit:
        rng = random.Random(args.seed)
        examples = rng.sample(examples, min(args.limit, len(examples)))

    done = existing_uids(out)
    todo = [e for e in examples if e.uid not in done]
    print(f"\n{dataset}: {len(examples)} selected, {len(done)} already done, {len(todo)} to run -> {out}")
    if not todo:
        return out

    # Record exactly what this run intends to do. Without it, a budget stop
    # leaves uids simply absent from the JSONL -- byte-identical to "never
    # selected" -- so dropped work cannot be told from work never asked for.
    out.with_suffix(".todo.json").write_text(json.dumps({
        "dataset": dataset, "tag": tag, "model": model, "seed": args.seed,
        "per_cell": getattr(args, "per_cell", None), "limit": args.limit,
        "max_edge": args.max_edge, "thinking_budget": args.thinking_budget,
        "uids": [e.uid for e in examples],
    }, indent=1))

    lock = threading.Lock()
    written = failed = 0
    fatal: list[Exception] = []
    stop = threading.Event()
    recent: list[bool] = []

    # Sliding submission window rather than queueing every future up front: at
    # 20k+ todo, f.cancel() cannot stop running work and ThreadPoolExecutor's
    # __exit__ blocks until the queue drains, so Ctrl-C and the budget stop both
    # appear to hang.
    pending = iter(todo)
    # The window is what actually bounds an overspend, not `--max-spend`: every
    # call already in flight when the ceiling trips still completes and still
    # bills. A 4x multiplier over --concurrency put 128 requests in the air at
    # the default, and a measured run crossed a $0.029 cap at record 15 and did
    # not stop until 140 -- 12.8x over.
    #
    # One wave, not four. On top of that, do not open a wave larger than the
    # budget can plausibly cover: until a call has returned there is no cost
    # estimate, so start narrow and widen once one is known.
    window = max(args.concurrency, 1)
    first_wave = min(window, 8)

    with open(out, "a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        def submit(e):
            return pool.submit(run_one, client, e, budget, args.thinking_budget,
                               args.max_edge, model)

        inflight = {submit(e) for e in itertools.islice(pending, first_wave)}
        while inflight:
            done_set, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done_set:
                try:
                    rec = fut.result()
                except FatalBillingError as e:
                    fatal.append(e); stop.set(); continue
                except Exception as e:
                    rec = {"uid": "?", "error": f"{type(e).__name__}: {e}"}
                with lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    written += 1
                    err = bool(rec.get("error"))
                    failed += err
                    recent.append(err)
                    if len(recent) > 200:
                        recent.pop(0)
                    if written % 100 == 0 or written == len(todo):
                        print(f"  {written}/{len(todo)} | ${budget.spent:.3f} | errors {failed}", flush=True)
                    if written % 500 == 0:
                        os.fsync(fh.fileno())
                # A wrong model id or revoked key would otherwise burn the run.
                if len(recent) >= 200 and sum(recent) / len(recent) > 0.5:
                    print("  !! >50% of the last 200 calls failed -- aborting", flush=True)
                    stop.set()
                # Widen to the full window only once a call has priced itself
                # and the remaining budget can cover a whole wave.
                if written and budget.limit:
                    per_call = budget.spent / written
                    afford = int((budget.limit - budget.spent) / per_call) if per_call else window
                    window = max(1, min(max(args.concurrency, 1), afford))
                if budget.exhausted():
                    print(f"  !! spend cap ${budget.limit} reached -- stopping cleanly", flush=True)
                    stop.set()
            if not stop.is_set():
                inflight |= {submit(e) for e in itertools.islice(pending, len(done_set))}

    if fatal:
        raise fatal[0]

    # Reconcile: anything selected but not usably answered is reported, not hidden.
    missing = [e.uid for e in examples if e.uid not in existing_uids(out)]
    if missing:
        out.with_suffix(".missing.json").write_text(json.dumps(missing, indent=1))
        print(f"  !! {len(missing)} selected uids have no usable answer "
              f"-> {out.with_suffix('.missing.json').name}", flush=True)
    return out


# ==============================================================================
# ---- CLI ---------------------------------------------------------------------
# ==============================================================================
#
# `python -m blindspot.core --datasets ... --max-spend ...`, identical in flags
# and behaviour of the study's published runs.


def main() -> int:
    p = argparse.ArgumentParser(description="Haiku 4.5 perception blind-spot eval")
    p.add_argument("--datasets", nargs="+", default=["charxiv", "infographicvqa", "screenspot_pro"],
                   choices=sorted(ADAPTERS))
    p.add_argument("--limit", type=int, default=None, help="random sample of N (prefer --per-cell)")
    p.add_argument("--per-cell", type=int, default=None,
                   help="stratified: N per primitive cell (the statistically meaningful option)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (keeps subsets stable across runs)")
    p.add_argument("--thinking-budget", type=int, default=2000)
    p.add_argument("--max-edge", type=int, default=None,
                   help="pre-downscale images to this long edge (ablation; Haiku caps at ~1568)")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-spend", type=float, default=5.0, help="USD hard stop")
    p.add_argument("--run", type=int, default=0, help="run index, for repeat-variance measurement")
    p.add_argument("--model", default=MODEL, choices=sorted(MODELS),
                   help="target model; thinking dialect and pricing follow from it")
    p.add_argument("--uids", nargs="+", default=None,
                   help="run only these example uids (overrides --limit)")
    args = p.parse_args()

    # Bound to a differently-named local: `client` is the module-level factory,
    # and `client = client()` here would make it a local and raise
    # UnboundLocalError on the call itself. See client() for the max_retries=0 /
    # timeout=120.0 rationale.
    api = client()
    budget = Budget(args.max_spend)
    t0 = time.monotonic()

    try:
        for ds in args.datasets:
            if budget.exhausted():
                print(f"spend cap reached; skipping {ds}")
                continue
            run_dataset(api, ds, args, budget)
    except FatalBillingError as e:
        print(f"\nFATAL billing error -- run aborted, results so far are saved.\n  {e}", file=sys.stderr)
        return 2

    print(f"\ndone in {time.monotonic()-t0:.0f}s | {budget.calls} calls | ${budget.spent:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
