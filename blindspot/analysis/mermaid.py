"""Parse FlowLearn's Mermaid ground truth into a graph, and score against it.

Why this exists: FlowLearn ships four QA fields per flowchart (Arrow_AtoB,
Arrow_betweenAB, Num_Nodes, Num_Arrows) but the Mermaid source is the only
*complete* description of the figure. Deriving golds from the parsed graph
rather than trusting the QA labels lets us (a) phrase questions to match the
semantics we actually score, and (b) measure how often the shipped labels
disagree with the figure -- which turns out to matter a great deal for
Arrow_betweenAB, whose "false" pairs are not undirected-false at all.

Mermaid we need to handle looks like:

    ```mermaid
    flowchart TB
    entity0(outsteered karite)
    entity1(dihedron cushite)
    entity0 --> entity1
    entity0 ==> entity4
    entity1 -..-> entity2
    ```
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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
