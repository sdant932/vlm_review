"""Example pages for the derived question sets (`word_mc`, `counting`).

Shows the full prompt exactly as the harness would assemble it -- including the
instruction prefix and, for multiple choice, the lettered option block appended
by `blindspot.core.prompts` rather than stored in the manifest. Reproducing that
assembly here is the point: a reader comparing the manifest against what the
model receives should see where every part comes from.

Usage:
    python scripts/generate/examples_svg_derived.py --set word_mc
    python scripts/generate/examples_svg_derived.py --set counting --per-type 3
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    from blindspot.core.prompts import CHOICE_INSTRUCTION, COUNT_INSTRUCTION
except Exception:
    CHOICE_INSTRUCTION = ("Answer with the single letter of the correct option.\n"
                          "Base your answer only on what is shown in the diagram.\n\n")
    COUNT_INSTRUCTION = ("Count carefully and answer with a single whole number.\n"
                         "Count only what is actually drawn in the image.\n\n")

CSS = Path(__file__).with_name("examples_svg_localization.py").read_text()
CSS = CSS[CSS.index('CSS = """') + 9:CSS.index('"""\n\n\ndef esc')]

SETS = {
    "word_mc": {
        "title": "Word presence — multiple choice",
        "dek": "Which of four words appears in the figure?",
        "blurb": (
            "One option appears somewhere in the figure; the other three appear "
            "nowhere in it. The correct word is drawn only from labels that "
            "already passed the localization set's filters &mdash; legible at "
            "this resolution, not occluded, unique in the scene &mdash; so a "
            "word the model physically cannot read is never the answer. Every "
            "distractor is verified absent as a substring across <i>all</i> "
            "text in the scene, including the title, footnotes and badges."),
        "scoring": "Exact letter match (<code>multiple_choice</code>). Chance is 25%.",
    },
    "counting": {
        "title": "Counting",
        "dek": "How many bars, rows, slices, nodes&hellip;?",
        "blurb": (
            "One counting question per chart type, asking about the structure "
            "that type is made of. Gold comes from the semantic record captured "
            "when the scene was built, not from counting marks in the raster, "
            "and 678 of 714 are additionally cross-checked against the labels "
            "actually drawn &mdash; a table's row count against rows&times;columns "
            "cell labels, a flowchart's box count against its node labels. "
            "Counts are identical at all three resolutions, so the same question "
            "doubles as a resolution probe."),
        "scoring": ("Exact integer match (<code>exact_count</code>), which also "
                    "returns <b>signed error</b> &mdash; the sign separates "
                    "stopping early from over-counting."),
    },
}


def esc(s):
    return html.escape(str(s), quote=True)


def prompt_of(r):
    if r["qtype"] == "word_mc":
        opts = "\n".join(f"{k}. {v}" for k, v in zip("ABCD", r["options"]))
        return f'{CHOICE_INSTRUCTION}{r["question"]}\n\n{opts}'
    return f'{COUNT_INSTRUCTION}{r["question"]}'


def answer_of(r):
    if r["qtype"] == "word_mc":
        return (f'"{r["answer"]}"   ({r["answer_text"]})\n\n'
                f'# the other three appear nowhere in the figure:\n'
                f'#   {", ".join(r["distractors"])}')
    cc = r.get("cross_checked_against")
    note = (f"# cross-checked against the '{cc}' labels actually drawn"
            if cc else "# not cross-checked: small blocks are drawn without labels")
    return f'{r["answer"]}\n\n{note}\n# scored exact, with signed error alongside'


def stage(src, W, H, width=520):
    return (f'<div class="stage" style="width:{min(W, width)}px">'
            f'<img src="{esc(src)}" alt="" width="{W}" height="{H}" loading="lazy">'
            f'</div>')


def card(r):
    W, H = r["image_px"]
    chips = [f'<span class="chip k">{esc(r["qtype"])}</span>',
             f'<span class="chip">{esc(r["chart_type"])}</span>',
             f'<span class="chip">{esc(r["resolution"])} &middot; {W}&times;{H}</span>',
             f'<span class="chip">theme: {esc(r["theme"])}</span>']
    kv = [("delivered to model", f'{r["effective_px"][0]}&times;{r["effective_px"][1]}'
                                 f'{" (downscaled)" if r["downscaled_by_api"] else ""}'),
          ("scene", f'<code>{esc(r["svg"])}</code>'),
          ("uid", f'<code>{esc(r["uid"])}</code>')]
    return (f'<div class="card"><div class="hd">{"".join(chips)}</div><div class="cols">'
            f'<div class="img">{stage("../" + r["image"], W, H)}</div>'
            f'<div class="txt">'
            f'<div class="lbl">prompt, exactly as sent</div><pre>{esc(prompt_of(r))}</pre>'
            f'<div class="lbl">gold answer</div>'
            f'<pre class="ans">{esc(answer_of(r))}</pre>'
            f'<dl class="kv">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in kv)
            + "</dl></div></div></div>")


def build(data: Path, which: str, out: Path, per_type: int) -> int:
    rows = [json.loads(l) for l in (data / which / "manifest.jsonl").read_text().splitlines() if l.strip()]
    spec = SETS[which]
    body = []

    n_scenes = len({r["graph_id"] for r in rows})
    tiles = [("scenes", n_scenes), ("questions", len(rows)),
             ("chart types", len({r["chart_type"] for r in rows}))]
    if which == "counting":
        gold = sorted(r["answer"] for r in rows)
        tiles.append(("gold count range", f'{gold[0]}&ndash;{gold[-1]}'))
    else:
        tiles.append(("chance", "25%"))
    body.append('<div class="tiles">' + "".join(
        f'<div class="tile"><div class="l">{k}</div><div class="v">{v}</div></div>'
        for k, v in tiles) + "</div>")

    body.append(f'<div class="note">{spec["blurb"]}</div>')
    body.append(f'<div class="note"><b>Scoring.</b> {spec["scoring"]}</div>')
    body.append('<div class="note warn"><b>These questions run on the existing '
                'scenes.</b> No image was re-rendered to create them &mdash; they are '
                'derived from <code>scenes.jsonl</code> and point at the same PNGs '
                'and SVGs as the localization set. A model can therefore be scored '
                'on all three sets over identical pixels.</div>')

    if which == "counting":
        body.append('<h2 id="ladder">Same question, three resolutions</h2>')
        by = defaultdict(dict)
        for r in rows:
            by[(r["graph_id"], r["question"])][r["resolution"]] = r
        trio = next((v for v in by.values() if len(v) == 3), None)
        if trio:
            any_r = next(iter(trio.values()))
            body.append(f'<pre>{esc(prompt_of(any_r))}</pre>')
            body.append(f'<div class="note"><b>Gold is '
                        f'{any_r["answer"]} at every rung.</b> The structure does not '
                        f'change with resolution, only how many pixels describe it, so '
                        f'any accuracy difference across these three is a resolution '
                        f'effect and nothing else.</div>')
            body.append('<div class="cols">')
            for rn in ("small", "medium", "large"):
                if rn in trio:
                    x = trio[rn]
                    body.append(f'<div><div class="sub">{rn} &middot; '
                                f'{x["image_px"][0]}&times;{x["image_px"][1]}</div>'
                                + stage("../" + x["image"], *x["image_px"], width=340)
                                + "</div>")
            body.append("</div>")

    body.append("<h2>Examples</h2>")
    seen, picked = set(), []
    for r in rows:
        if r["resolution"] != "medium" or r["chart_type"] in seen:
            continue
        seen.add(r["chart_type"])
        picked.append(r)
        if len(picked) >= per_type:
            break
    for r in picked:
        body.append(card(r))

    doc = ('<!doctype html><html><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>{spec["title"]} &mdash; examples</title>'
           f"<style>{CSS}</style></head><body><div class=\"wrap\">"
           f'<h1>{spec["title"]}</h1><p class="sub">{spec["dek"]}</p>'
           '<nav><a href="../examples/index.html">Localization examples</a>'
           '<a href="../verify/index.html">Ground-truth audit</a>'
           '<a href="EVAL.md">EVAL.md</a></nav>'
           + "".join(body) + "</div></body></html>")
    out.write_text(doc)
    return len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/svg_localization"))
    ap.add_argument("--set", default="both", choices=["both", "word_mc", "counting"])
    ap.add_argument("--per-type", type=int, default=6)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)
    which = ["word_mc", "counting"] if a.set == "both" else [a.set]
    for w in which:
        out = a.data / w / "examples.html"
        n = build(a.data, w, out, a.per_type)
        print(f"{w}: {n} questions -> {out}")
        if a.open:
            subprocess.run(["open", str(out)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
