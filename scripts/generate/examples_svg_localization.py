"""Build a browsable page of example questions and answers.

This is not the ground-truth audit -- `verify_svg_localization.py` does that. The
purpose here is to show, for a reader who has never seen the dataset, *exactly
what a model is asked and exactly what counts as right*.

The prompt shown is the full string as it would be sent, with
`blindspot.core.prompts.POINT_INSTRUCTION` prepended for `point` rows. That prefix is
the single easiest thing to get wrong: the manifest stores only the element
description, and sending it bare turns a localization task into an unanswerable
fragment.

Usage:
    python scripts/generate/examples_svg_localization.py
    python scripts/generate/examples_svg_localization.py --per-type 4 --open
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
    from blindspot.core.prompts import POINT_INSTRUCTION
except Exception:                                    # keep the page buildable
    POINT_INSTRUCTION = (
        "Locate the described UI element in the screenshot and return the point at "
        "its center.\nUse a normalized coordinate system where x=0 is the left edge, "
        "x=1000 the right edge, y=0 the top edge, and y=1000 the bottom edge.\n"
        "Always return your single best guess, even if you are uncertain.\n\nElement: ")

CSS = """
:root{--bg:#0f1116;--panel:#171a21;--panel2:#1d2028;--ink:#e8eaed;--muted:#9aa0aa;
      --line:#2a2f3a;--good:#2ecc71;--bad:#ff5b5b;--accent:#5b8def;--chip:#232833}
@media (prefers-color-scheme:light){:root{--bg:#f7f8fa;--panel:#fff;--panel2:#f0f2f6;
      --ink:#15181d;--muted:#5c636e;--line:#dfe3ea;--chip:#eceff4}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:30px 22px 80px}
h1{font-size:26px;margin:0 0 6px}
h2{font-size:20px;margin:40px 0 10px;padding-bottom:7px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:20px 0 8px;color:var(--muted);font-weight:600}
.sub{color:var(--muted);margin:0 0 20px}
nav{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 24px}
nav a{background:var(--chip);color:var(--ink);text-decoration:none;padding:6px 12px;
      border-radius:7px;font-size:13px;border:1px solid var(--line)}
nav a:hover{border-color:var(--accent)}
.note{background:var(--panel);border-left:3px solid var(--accent);padding:12px 15px;
      border-radius:0 8px 8px 0;margin:14px 0}
.note.warn{border-left-color:#d68a1e}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:12px}
.tile .v{font-size:23px;font-weight:650;font-variant-numeric:tabular-nums}
.tile .l{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;
      padding:16px;margin:16px 0}
.card .hd{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin-bottom:11px}
.chip{display:inline-block;background:var(--chip);border:1px solid var(--line);
      border-radius:20px;padding:2px 9px;font-size:11px;color:var(--muted)}
.chip.k{color:var(--accent);border-color:var(--accent)}
.cols{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
.cols>.img{flex:0 0 auto}
.cols>.txt{flex:1 1 340px;min-width:300px}
.stage{position:relative;display:inline-block;border:1px solid var(--line);
       border-radius:8px;overflow:hidden;background:#fff;line-height:0}
.stage img{display:block;max-width:100%;height:auto}
.box{position:absolute;border:2px solid var(--good);
     box-shadow:0 0 0 1px rgba(0,0,0,.5) inset;pointer-events:none}
.cross{position:absolute;pointer-events:none}
.cross::before,.cross::after{content:"";position:absolute;background:var(--bad)}
.cross::before{left:-10px;top:-1px;width:20px;height:2px}
.cross::after{top:-10px;left:-1px;height:20px;width:2px}
pre{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
    padding:11px 13px;margin:6px 0 12px;overflow-x:auto;white-space:pre-wrap;
    font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink)}
pre.ans{border-left:3px solid var(--good)}
.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
     margin:10px 0 3px;font-weight:600}
code{background:var(--chip);padding:1px 5px;border-radius:4px;font-size:12.5px;
     font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
table{width:100%;border-collapse:collapse;margin:10px 0;background:var(--panel);
      border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{padding:7px 11px;text-align:left;border-bottom:1px solid var(--line);font-size:13px}
th{background:var(--chip);color:var(--muted);font-weight:600;font-size:11px;
   text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:none}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:12.5px;margin-top:8px}
.kv dt{color:var(--muted)} .kv dd{margin:0;font-variant-numeric:tabular-nums}
"""


def esc(s):
    return html.escape(str(s), quote=True)


def stage(src, W, H, box=None, cross=None, width=520):
    o = [f'<div class="stage" style="width:{min(W, width)}px">'
         f'<img src="{esc(src)}" alt="" width="{W}" height="{H}" loading="lazy">']
    if box:
        x0, y0, x1, y1 = box
        o.append(f'<div class="box" style="left:{x0 / W * 100:.4f}%;top:{y0 / H * 100:.4f}%;'
                 f'width:{(x1 - x0) / W * 100:.4f}%;height:{(y1 - y0) / H * 100:.4f}%"></div>')
    if cross:
        o.append(f'<div class="cross" style="left:{cross[0] / W * 100:.4f}%;'
                 f'top:{cross[1] / H * 100:.4f}%"></div>')
    o.append("</div>")
    return "".join(o)


def prompt_of(r):
    """The full string as sent -- the prefix is the thing people forget."""
    if r["qtype"] == "point":
        return POINT_INSTRUCTION + r["question"]
    return r["question"]


def answer_of(r):
    a = r["answer"]
    if r["qtype"] == "point":
        nb = r["gold_bbox_norm"]
        return (json.dumps(a) + "\n\n"
                "# scored: point_in_bbox(pred, gold_bbox_norm)\n"
                f"# gold_bbox_norm = [{nb[0]:.4f}, {nb[1]:.4f}, {nb[2]:.4f}, {nb[3]:.4f}]\n"
                "# a prediction counts iff it lands inside that box -- no tolerance")
    return f'"{a}"\n\n# scored: exact match, and token-F1 alongside'


META = {
    "point": ("Localization", "perception <b>+</b> coordinate emission",
              "The model is given an element description and must return its centre in a "
              "0&ndash;1000 normalized space. Scored <code>point_in_bbox</code> &mdash; "
              "binary, no tolerance, no IoU. Green box is the gold target; red crosshair is "
              "the gold answer."),
    "relation": ("Spatial relation", "perception only, <b>no</b> coordinates",
                 "No coordinates in the question or the answer. The model must find one "
                 "label, work out which label sits beside it, and read that one back."),
    "reverse": ("Inverse localization", "point &rarr; text",
                "The opposite direction: given a pixel coordinate, name the text there. "
                "Red crosshair is the probe point given in the question."),
}


def card(r, data: Path, show_box=True, show_cross=None):
    W, H = r["image_px"]
    box = r["gold_bbox_px"] if show_box else None
    cross = None
    if show_cross == "answer":
        cross = r.get("gold_center_px")
    elif show_cross == "probe":
        cross = r.get("probe_point_px")
    chips = [f'<span class="chip k">{esc(r["qtype"])}</span>',
             f'<span class="chip">{esc(r["chart_type"])}</span>',
             f'<span class="chip">{esc(r["resolution"])} &middot; {W}&times;{H}</span>',
             f'<span class="chip">theme: {esc(r["theme"])}</span>',
             f'<span class="chip">role: {esc(r["target_role"])}</span>']
    kv = [("target area", f'{r["target_area_frac"] * 100:.4f}% of image'),
          ("label size", f'{r["font_px"]}px'),
          ("contrast", f'{r["target_contrast"]:.1f}:1'),
          ("delivered to model", f'{r["effective_px"][0]}&times;{r["effective_px"][1]}'
                                 f'{" (downscaled)" if r["downscaled_by_api"] else ""}'),
          ("uid", f'<code>{esc(r["uid"])}</code>')]
    return (f'<div class="card"><div class="hd">{"".join(chips)}</div><div class="cols">'
            f'<div class="img">{stage("../" + r["image"], W, H, box, cross)}</div>'
            f'<div class="txt">'
            f'<div class="lbl">prompt, exactly as sent</div>'
            f'<pre>{esc(prompt_of(r))}</pre>'
            f'<div class="lbl">gold answer</div>'
            f'<pre class="ans">{esc(answer_of(r))}</pre>'
            f'<dl class="kv">'
            + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in kv)
            + "</dl></div></div></div>")


def build(data: Path, out: Path, per_type: int) -> int:
    rows = [json.loads(l) for l in (data / "manifest.jsonl").read_text().splitlines() if l.strip()]
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["qtype"]].append(r)

    body = []
    n_scenes = len({r["graph_id"] for r in rows})
    body.append('<div class="tiles">'
                f'<div class="tile"><div class="l">scenes</div><div class="v">{n_scenes}</div></div>'
                f'<div class="tile"><div class="l">questions</div><div class="v">{len(rows)}</div></div>'
                f'<div class="tile"><div class="l">question types</div><div class="v">{len(by_type)}</div></div>'
                f'<div class="tile"><div class="l">chart types</div>'
                f'<div class="v">{len({r["chart_type"] for r in rows})}</div></div>'
                '</div>')

    body.append('<div class="note warn"><b>The prompts below are the complete strings.</b> '
                'For <code>point</code> rows the manifest stores only the element description '
                '(<code>the text "Close"</code>); <code>POINT_INSTRUCTION</code> from '
                '<code>blindspot.core.prompts</code> must be prepended. Everything before '
                '<code>Element:</code> comes from that constant, not from the manifest.</div>')

    # one target, three rungs -- the control the set is built around
    ladder = None
    for r in by_type["point"]:
        if r["resolution"] != "medium":
            continue
        sibs = {x["resolution"]: x for x in by_type["point"]
                if x["graph_id"] == r["graph_id"] and x["target_text"] == r["target_text"]}
        if len(sibs) == 3:
            ladder = sibs
            break
    if ladder:
        body.append('<h2 id="ladder">The same target at three resolutions</h2>')
        body.append('<p class="sub">Identical question, identical answer in normalized '
                    'coordinates, identical fraction of the image. Only the pixel budget '
                    'changes &mdash; which is what makes the ladder a clean resolution test '
                    'rather than a target-size test.</p>')
        lr = ladder["medium"]
        body.append(f'<pre>{esc(prompt_of(lr))}</pre>')
        body.append('<table><tr><th>variant</th><th class=num>image</th>'
                    '<th class=num>delivered</th><th class=num>label</th>'
                    '<th class=num>target area</th><th>gold answer (0&ndash;1000)</th></tr>')
        for rn in ("small", "medium", "large"):
            x = ladder[rn]
            body.append(f'<tr><td>{rn}</td>'
                        f'<td class=num>{x["image_px"][0]}&times;{x["image_px"][1]}</td>'
                        f'<td class=num>{x["effective_px"][0]}&times;{x["effective_px"][1]}</td>'
                        f'<td class=num>{x["font_px"]}px</td>'
                        f'<td class=num>{x["target_area_frac"] * 100:.4f}%</td>'
                        f'<td><code>{esc(json.dumps(x["answer"]))}</code></td></tr>')
        body.append("</table>")
        body.append('<div class="cols">')
        for rn in ("small", "medium", "large"):
            x = ladder[rn]
            body.append(f'<div><div class="sub">{rn}</div>'
                        + stage("../" + x["image"], *x["image_px"], x["gold_bbox_px"],
                                x["gold_center_px"], width=340) + "</div>")
        body.append("</div>")
        body.append('<div class="note"><b><code>medium</code> and <code>large</code> both '
                    'deliver at the same size.</b> They are the same input to the model '
                    'despite differing 4&times; on disk, so any measured gap between them is '
                    'noise &mdash; that is the null control. <code>small</code> is the one '
                    'that genuinely carries less detail.</div>')

    order = ["point", "relation", "reverse"]
    for qt in order:
        pool = by_type.get(qt) or []
        if not pool:
            continue
        title, isolates, blurb = META[qt]
        body.append(f'<h2 id="{qt}">{title} &mdash; <code>{qt}</code></h2>')
        body.append(f'<p class="sub">Isolates: {isolates}. {blurb}</p>')
        seen, picked = set(), []
        for r in pool:
            if r["resolution"] != "medium" or r["chart_type"] in seen:
                continue
            seen.add(r["chart_type"])
            picked.append(r)
            if len(picked) >= per_type:
                break
        cross = {"point": "answer", "reverse": "probe"}.get(qt)
        for r in picked:
            body.append(card(r, data, show_box=True, show_cross=cross))

    body.append('<h2 id="scope">What these questions do not ask</h2>')
    body.append('<div class="note warn">Every target string is <b>quoted in its own '
                'prompt</b>, so the model matches a string rather than resolving a '
                'description to a referent. ScreenSpot-Pro instead asks things like '
                '<i>"stop the bilibili download in android virtual machine in android '
                'studio"</i>, usually pointing at an icon with no text at all. This set has '
                '<b>zero icon targets</b>, and its smallest target is ~4.5&times; larger than '
                'ScreenSpot-Pro\'s. Absolute scores here are <b>not</b> comparable to that '
                'arm &mdash; only within-dataset contrasts and the <i>shape</i> of the '
                'precision curve are. See <code>EVAL.md</code> &sect;1.1.</div>')

    doc = ('<!doctype html><html><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           '<title>SVG localization &mdash; example questions and answers</title>'
           f"<style>{CSS}</style></head><body><div class=\"wrap\">"
           '<h1>SVG localization &mdash; example questions and answers</h1>'
           '<p class="sub">What a model is asked, and what counts as correct.</p>'
           '<nav><a href="#ladder">Resolution ladder</a><a href="#point">point</a>'
           '<a href="#relation">relation</a>'
           '<a href="#reverse">reverse</a><a href="#scope">Scope</a>'
           '<a href="../verify/index.html">Ground-truth audit</a></nav>'
           + "".join(body) + "</div></body></html>")
    out.write_text(doc)
    return len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/svg_localization"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--per-type", type=int, default=4)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)
    out = a.out or (a.data / "examples" / "index.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = build(a.data, out, a.per_type)
    print(f"examples page from {n} questions -> {out}")
    if a.open:
        subprocess.run(["open", str(out)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
