"""Derive two extra question sets from the *existing* svg_localization scenes.

Nothing here re-renders. It reads `scenes.jsonl` (semantic content, captured when
the scenes were built) and `manifest.jsonl` (the localization set), and writes new
question files against the images already on disk. Same SVGs, same PNGs, same
pixels -- only new questions.

    word_mc   Which of four words appears in the figure?
              answer_type "choice"  -> blindspot.core.scoring multiple_choice

    counting  How many bars / rows / slices / nodes ... ?
              answer_type "count"   -> blindspot.core.scoring count_score, which
                                       returns signed error as well as accuracy

Two validity rules do the real work.

For `word_mc`, the correct word is drawn only from labels that already passed
the localization set's filters -- legible at this rung, not occluded, unique in
the scene -- because a word the model cannot read is not a fair question. Every
distractor is checked to appear **nowhere** in the scene as a substring, across
every label including titles, footnotes and badges.

For `counting`, gold comes from the semantic record rather than from counting
marks in the raster, and each count is cross-checked against the labels actually
drawn where the two should agree (bars against category labels, and so on).
Mismatches are reported, not silently published.

Usage:
    python scripts/generate/gen_svg_derived.py
    python scripts/generate/gen_svg_derived.py --set counting --per-scene 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from scripts.generate.gen_svg_localization import (VERBS, TERMINALS, REGIONS, METRICS, ROLES,
                                                   ACTORS, QUARTERS, DOMAINS)

WORD_RE = re.compile(r"[A-Za-z][A-Za-z-]{3,}")


def distractor_pool() -> list[str]:
    """Plausible words, drawn from the same families the scenes are built from."""
    out = set(VERBS) | set(TERMINALS) | set(REGIONS) | set(METRICS) | set(ROLES) | set(ACTORS)
    for _, nouns in DOMAINS:
        out |= set(nouns)
    return sorted(w for w in out if WORD_RE.fullmatch(w))


# --------------------------------------------------------------- word_mc

def scene_blob(scene) -> str:
    return " ".join(t["text"] for t in scene["texts"]).lower()


def make_word_mc(scene, vetted: set[str], rng: random.Random, pool: list[str],
                 n: int) -> list[dict]:
    """MC over word presence. Correct word is readable; distractors are absent."""
    blob = scene_blob(scene)
    # candidate correct answers: words inside labels that passed every filter
    cands = []
    for lab in sorted(vetted):
        for w in WORD_RE.findall(lab):
            if blob.count(w.lower()) >= 1:
                cands.append(w)
    cands = sorted(set(cands))
    if not cands:
        return []
    absent = [w for w in pool if w.lower() not in blob]
    if len(absent) < 3:
        return []

    out = []
    rng.shuffle(cands)
    for correct in cands[:n]:
        picks = rng.sample(absent, 3)
        options = picks + [correct]
        rng.shuffle(options)
        letter = "ABCD"[options.index(correct)]
        out.append({
            "qtype": "word_mc", "answer_type": "choice",
            "prompt_style": "choice",
            "question": "Which of the following words appears in the figure?",
            "options": options,
            "answer": letter, "answer_text": correct,
            "distractors": picks,
            "scoring": "multiple_choice",
        })
    return out


# --------------------------------------------------------------- counting
# (question, key into facts, how to count, what the reader should count)

def counting_specs(scene) -> list[tuple[str, int, tuple | str | None]]:
    """(question, gold, check) where check is None or (role, expected_label_count)."""
    f, ct = scene["facts"], scene["chart_type"]
    q = []
    if ct == "bar_chart" and "bars" in f:
        q.append(("How many bars are in this bar chart?", len(f["bars"]), "category"))
    elif ct == "table":
        nR, nC = len(f["row_headers"]), len(f["columns"])
        # every cell in the grid carries a label, so rows*cols must equal the
        # number of 'cell' texts actually drawn
        q.append(("How many data rows does this table have, not counting the "
                  "header row?", nR, ("cell", nR * nC)))
        q.append(("How many columns does this table have?", nC, ("header", nC)))
    elif ct == "pie_chart":
        q.append(("How many slices are in this pie chart?", len(f["wedges"]), "legend"))
    elif ct == "flowchart":
        q.append(("How many labelled boxes are in this flowchart?",
                  len(f["nodes"]), ("label", len(f["nodes"]))))
    elif ct == "network":
        q.append(("How many labelled nodes are in this network diagram?",
                  len(f["nodes"]), ("label", len(f["nodes"]))))
    elif ct == "timeline":
        q.append(("How many milestones are marked on this timeline?",
                  len(f["milestones"]), "milestone"))
    elif ct == "gantt":
        q.append(("How many task rows are in this Gantt chart?", len(f["tasks"]), "task"))
    elif ct == "treemap":
        # deliberately NOT cross-checked: a block too small for text is still
        # drawn as a rectangle, so labels and rectangles are not 1:1 here
        q.append(("How many rectangles are in this treemap?", len(f["blocks"]), None))
    elif ct == "sequence":
        q.append(("How many participants are in this sequence diagram?",
                  len(f["actors"]), "actor"))
        q.append(("How many message arrows are in this sequence diagram?",
                  len(f["messages"]), "message"))
    elif ct == "state_machine":
        q.append(("How many states (circles) are in this state machine?",
                  len(f["states"]), ("label", len(f["states"]))))
    elif ct == "org_chart":
        _org = len({n for lvl in f["levels"] for n in lvl})
        q.append(("How many labelled boxes are in this org chart?",
                  _org, ("label", _org)))
    elif ct == "scatter":
        q.append(("How many labelled points are in this scatter plot?",
                  len(f["points"]), "point"))
    elif ct == "quadrant":
        q.append(("How many labelled points are plotted in this quadrant chart?",
                  len(f["items"]), "point"))
    elif ct == "mindmap":
        q.append(("How many branches radiate from the centre of this mind map?",
                  len(f["branches"]), ("label", len(f["branches"]))))
    elif ct == "line_chart":
        q.append(("How many separate lines are plotted in this line chart?",
                  len(f["series"]), "series"))
        q.append(("How many labels are on the x-axis of this line chart?",
                  len(f["x_labels"]), "category"))
    elif ct == "dashboard":
        q.append(("How many separate chart panels are in this dashboard?",
                  len(f.get("panels", [])), "panel_title"))
    return [(a, b, c) for a, b, c in q if b > 0]


def cross_check(scene, check, gold: int) -> str | None:
    """Verify a count against the labels actually drawn, where the two must agree.

    A bare role string means "there should be exactly `gold` labels of this role".
    A (role, expected) pair is for counts where the relationship is not 1:1 --
    a table's row count is checked as rows*cols against the cell labels drawn.
    """
    if check is None:
        return None
    role, expected = (check, gold) if isinstance(check, str) else check
    n = sum(1 for t in scene["texts"] if t["role"] == role)
    if n == expected:
        return None
    return f"expected {expected} '{role}' labels, {n} drawn"


# --------------------------------------------------------------- driver

def build(data: Path, which: set[str], per_scene: int, seed: int):
    scenes = [json.loads(l) for l in (data / "scenes.jsonl").read_text().splitlines() if l.strip()]
    loc = [json.loads(l) for l in (data / "manifest.jsonl").read_text().splitlines() if l.strip()]

    vetted = defaultdict(set)          # (gid, res) -> labels that passed every filter
    meta = {}
    for m in loc:
        vetted[(m["graph_id"], m["resolution"])].add(m["target_text"])
        meta[(m["graph_id"], m["resolution"])] = {
            "image_px": m["image_px"], "effective_px": m["effective_px"],
            "downscaled_by_api": m["downscaled_by_api"], "scale": m["scale"],
        }
    pool = distractor_pool()
    by_id = {s["graph_id"]: s for s in scenes}

    mc_rows, ct_rows, warnings = [], [], []
    for gid, scene in sorted(by_id.items()):
        for res, img in scene["images"].items():
            k = (gid, res)
            if k not in meta:
                continue
            base = {
                "graph_id": gid, "chart_type": scene["chart_type"],
                "theme": scene["theme"], "font_family": scene["font_family"],
                "resolution": res, "image": img, "svg": scene["svg"],
                "title": scene["title"], "complexity": scene["complexity"],
                **meta[k],
            }
            if "word_mc" in which:
                rng = random.Random(seed * 977 + gid * 31 + len(res))
                for i, q in enumerate(make_word_mc(scene, vetted[k], rng, pool,
                                                   per_scene)):
                    mc_rows.append({"uid": f"svgmc:{gid:04d}:{res}:{i:02d}",
                                    **base, **q})
            if "counting" in which:
                for i, (qs, gold, check) in enumerate(counting_specs(scene)):
                    bad = cross_check(scene, check, gold)
                    if bad:
                        warnings.append({"graph_id": gid, "chart_type": scene["chart_type"],
                                         "question": qs, "detail": bad})
                        continue
                    ct_rows.append({
                        "uid": f"svgcount:{gid:04d}:{res}:{i:02d}", **base,
                        "qtype": "counting", "answer_type": "count",
                        "prompt_style": "count", "question": qs,
                        "answer": gold, "true_count": gold,
                        "cross_checked_against": (check if isinstance(check, str)
                                                  else (check[0] if check else None)),
                        "scoring": "exact_count",
                    })

    if "word_mc" in which:
        d = data / "word_mc"; d.mkdir(exist_ok=True)
        (d / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in mc_rows))
    if "counting" in which:
        d = data / "counting"; d.mkdir(exist_ok=True)
        (d / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ct_rows))
        (d / "cross_check_failures.json").write_text(json.dumps(warnings, indent=1))
    return mc_rows, ct_rows, warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/svg_localization"))
    ap.add_argument("--set", default="both", choices=["both", "word_mc", "counting"])
    ap.add_argument("--per-scene", type=int, default=2,
                    help="word_mc questions per scene per resolution")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args(argv)
    which = {"word_mc", "counting"} if a.set == "both" else {a.set}

    mc, ct, warn = build(a.data, which, a.per_scene, a.seed)
    if "word_mc" in which:
        print(f"word_mc : {len(mc)} questions -> {a.data / 'word_mc/manifest.jsonl'}")
    if "counting" in which:
        print(f"counting: {len(ct)} questions -> {a.data / 'counting/manifest.jsonl'}")
        if warn:
            print(f"  ! {len(warn)} count(s) failed cross-check and were dropped:")
            seen = set()
            for w in warn:
                if w["chart_type"] in seen:
                    continue
                seen.add(w["chart_type"])
                print(f"      {w['chart_type']}: {w['detail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
