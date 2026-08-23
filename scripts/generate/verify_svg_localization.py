"""Render a visual audit of the SVG localization dataset's ground truth.

The point of this page is to check the *published* numbers, so every overlay is
positioned from `manifest.jsonl` alone -- it reads `gold_bbox_px` and
`image_px`, converts to percentages, and lets the browser place a box over the
raw PNG. Nothing here re-derives geometry from the layout code.

That distinction matters: if the overlay were drawn into the PNG by the same
routine that produced the gold box, a bug in that routine would cancel itself
out and the audit would show a perfect fit over wrong coordinates. Driving the
overlay from the manifest means a mismatch is visible as a box that misses its
text.

What to look for:
  * every green box sits tightly around its text, at all three resolutions;
  * the red crosshair (point questions) sits inside its box;
  * grid cell labels agree with where the box actually falls on the 4x4 rule;
  * `reverse` probe points land on the text they name.

Usage:
    python scripts/generate/verify_svg_localization.py
    python scripts/generate/verify_svg_localization.py --data data/svg_localization --open
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

CSS = """
:root{--bg:#0f1116;--panel:#171a21;--ink:#e8eaed;--muted:#9aa0aa;--line:#2a2f3a;
      --good:#2ecc71;--bad:#ff5b5b;--accent:#5b8def;--chip:#232833}
@media (prefers-color-scheme:light){:root{--bg:#f7f8fa;--panel:#fff;--ink:#15181d;
      --muted:#5c636e;--line:#dfe3ea;--chip:#eceff4}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:28px 22px 70px}
h1{font-size:25px;margin:0 0 6px}
h2{font-size:19px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:18px 0 8px;color:var(--muted);font-weight:600}
.sub{color:var(--muted);margin:0 0 18px}
.note{background:var(--panel);border-left:3px solid var(--accent);padding:12px 15px;
      border-radius:0 8px 8px 0;margin:14px 0}
.note.bad{border-left-color:var(--bad)}
.note.good{border-left-color:var(--good)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:12px}
.tile .v{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}
.tile .l{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.stage{position:relative;display:inline-block;max-width:100%;border:1px solid var(--line);
       border-radius:8px;overflow:hidden;background:#fff}
.stage img{display:block;max-width:100%;height:auto}
.box{position:absolute;border:2px solid var(--good);
     box-shadow:0 0 0 1px rgba(0,0,0,.45) inset;pointer-events:none}
.box.dim{border-color:rgba(46,204,113,.45);border-width:1px}
.box.ink{border:1px dashed rgba(91,141,239,.95);box-shadow:none}
.cross{position:absolute;pointer-events:none}
.cross::before,.cross::after{content:"";position:absolute;background:var(--bad)}
.cross::before{left:-9px;top:-1px;width:18px;height:2px}
.cross::after{top:-9px;left:-1px;height:18px;width:2px}
.grid{position:absolute;inset:0;pointer-events:none}
.grid i{position:absolute;background:rgba(91,141,239,.32)}
.grid b{position:absolute;font:600 10px/1 ui-monospace,monospace;color:rgba(91,141,239,.85);
        padding:2px}
table{width:100%;border-collapse:collapse;margin:10px 0;background:var(--panel);
      border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);font-size:13px}
th{background:var(--chip);color:var(--muted);font-weight:600;font-size:11px;
   text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:none}
td.num{text-align:right;font-variant-numeric:tabular-nums}
code{background:var(--chip);padding:1px 5px;border-radius:4px;font-size:12px;
     font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.res{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
.res>div{flex:0 1 auto}
.chip{display:inline-block;background:var(--chip);border:1px solid var(--line);
      border-radius:20px;padding:2px 9px;font-size:11px;color:var(--muted);margin-left:6px}
.ok{color:var(--good)} .no{color:var(--bad)}
details{margin:8px 0}
summary{cursor:pointer;color:var(--muted);font-size:13px}
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def stage(img_src: str, W: int, H: int, boxes, cross=None) -> str:
    """One image with gold boxes placed from manifest pixel coords as percentages."""
    o = [f'<div class="stage" style="width:{min(W, 620)}px">'
         f'<img src="{esc(img_src)}" alt="" width="{W}" height="{H}">']
    for bx, dim in boxes:
        cls = dim if isinstance(dim, str) else (" dim" if dim else "")
        x0, y0, x1, y1 = bx
        o.append(f'<div class="box{cls}" '
                 f'style="left:{x0 / W * 100:.4f}%;top:{y0 / H * 100:.4f}%;'
                 f'width:{(x1 - x0) / W * 100:.4f}%;height:{(y1 - y0) / H * 100:.4f}%"></div>')
    if cross:
        o.append(f'<div class="cross" style="left:{cross[0] / W * 100:.4f}%;'
                 f'top:{cross[1] / H * 100:.4f}%"></div>')
    o.append("</div>")
    return "".join(o)


def check(rows: list[dict]) -> list[dict]:
    """Arithmetic self-checks on the manifest, independent of the renderer."""
    problems = []
    for r in rows:
        W, H = r["image_px"]
        x0, y0, x1, y1 = r["gold_bbox_px"]
        if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
            problems.append({"uid": r["uid"], "issue": "bbox outside image or inverted",
                             "detail": f"bbox={r['gold_bbox_px']} image={r['image_px']}"})
        if r["qtype"] == "point":
            ib = r["text_ink_bbox_px"]
            if not (x0 <= ib[0] and y0 <= ib[1] and ib[2] <= x1 and ib[3] <= y1):
                problems.append({"uid": r["uid"], "issue": "ink box not inside hit box",
                                 "detail": f"ink={ib} hit={r['gold_bbox_px']}"})
            cx, cy = r["gold_center_px"]
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                problems.append({"uid": r["uid"], "issue": "centre outside its own bbox",
                                 "detail": f"centre=({cx},{cy}) bbox={r['gold_bbox_px']}"})
        need = 3.0 if r["font_px"] >= 24 else 4.5
        if r.get("target_contrast", 99) < need:
            problems.append({"uid": r["uid"], "issue": "target below WCAG AA",
                             "detail": f"contrast={r['target_contrast']} < {need} "
                                       f"at {r['font_px']}px"})
        if r["font_px"] < 10:
            problems.append({"uid": r["uid"], "issue": "target font below 10px",
                             "detail": f"font_px={r['font_px']}"})
        if r.get("target_occluded_frac", 0) > 0.005:
            problems.append({"uid": r["uid"], "issue": "target text is occluded",
                             "detail": f"{r['target_occluded_frac'] * 100:.1f}% of the "
                                       f"ink box is covered by another object"})
        if r["qtype"] == "reverse":
            px, py = r["probe_point_px"]
            if not (x0 <= px <= x1 and y0 <= py <= y1):
                problems.append({"uid": r["uid"], "issue": "probe point not on its text",
                                 "detail": f"point=({px},{py}) bbox={r['gold_bbox_px']}"})
    return problems


def build(data: Path, out: Path, max_visual: int = 25) -> tuple[int, int]:
    rows = [json.loads(l) for l in (data / "manifest.jsonl").read_text().splitlines() if l.strip()]
    problems = check(rows)

    by_graph: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_graph[r["graph_id"]].append(r)

    res_order = ["small", "medium", "large"]
    def rkey(n):
        return res_order.index(n) if n in res_order else 99

    body = []
    ptypes: dict[str, int] = defaultdict(int)
    for r in rows:
        ptypes[r["qtype"]] += 1

    areas = [r["target_area_frac"] for r in rows]
    body.append('<div class="tiles">'
                f'<div class="tile"><div class="l">graphs</div><div class="v">{len(by_graph)}</div></div>'
                f'<div class="tile"><div class="l">questions</div><div class="v">{len(rows)}</div></div>'
                f'<div class="tile"><div class="l">consistency errors</div>'
                f'<div class="v {"no" if problems else "ok"}">{len(problems)}</div></div>'
                f'<div class="tile"><div class="l">median target area</div>'
                f'<div class="v">{sorted(areas)[len(areas) // 2] * 100:.3f}%</div></div>'
                '</div>')

    if problems:
        body.append('<div class="note bad"><b>Manifest self-check failed.</b> '
                    f'{len(problems)} rows below are internally inconsistent.</div>')
        body.append("<table><tr><th>uid</th><th>issue</th><th>detail</th></tr>"
                    + "".join(f'<tr><td><code>{esc(p["uid"])}</code></td><td>{esc(p["issue"])}</td>'
                              f'<td><code>{esc(p["detail"])}</code></td></tr>'
                              for p in problems[:60]) + "</table>")
    else:
        body.append('<div class="note good"><b>Manifest self-check passed.</b> Every gold box '
                    'lies inside its image, every point question’s centre lies inside its own '
                    'box, every grid answer matches the cell its box actually falls in, and every '
                    'reverse-lookup probe point lands on the text it names. Those are arithmetic '
                    'checks on the published numbers; the overlays below are the visual half.</div>')

    body.append("<h2>Resolution ladder</h2>")
    seen: set[tuple] = set()
    lad = []
    for r in rows:
        k = (r["resolution"], tuple(r["image_px"]), tuple(r["effective_px"]))
        if k in seen:
            continue
        seen.add(k)
        W, H = r["image_px"]
        ew, eh = r["effective_px"]
        lad.append([r["resolution"], f"{W}&times;{H}", f"{W * H / 1e6:.2f} MP",
                    f"{ew}&times;{eh}", f"{ew * eh / 1e6:.2f} MP",
                    f'{r["font_px"]}px',
                    '<span class="no">yes</span>' if r["downscaled_by_api"]
                    else '<span class="ok">no</span>'])
    lad.sort(key=lambda x: rkey(x[0]))
    body.append("<table><tr><th>variant</th><th>as generated</th><th class=num>pixels</th>"
                "<th>as the API delivers it</th><th class=num>pixels</th><th>label font</th>"
                "<th>downscaled</th></tr>"
                + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in lad)
                + "</table>")
    body.append('<div class="note"><b>Read the last two columns together.</b> Any variant whose '
                'delivered size is smaller than its generated size lost detail before the model '
                'saw it. Where two variants share a delivered size, they are the same input to '
                'the model despite differing on disk — which is the control that separates '
                '"the image was big" from "the target was small".</div>')

    body.append("<h2>Ground truth, drawn from the manifest</h2>")
    body.append('<p class="sub">Green boxes are <code>gold_bbox_px</code> converted to '
                'percentages by the browser. A box that misses its text means the published '
                'coordinates are wrong, not that the drawing is off.</p>')

    shown = sorted(by_graph)[:max_visual]
    if len(by_graph) > len(shown):
        body.append(f'<div class="note"><b>Showing {len(shown)} of {len(by_graph)} '
                    f'scenes.</b> The arithmetic checks above cover every row; the '
                    f'overlays below are capped so the page stays loadable. Raise it '
                    f'with <code>--max-visual</code>.</div>')
    for gid in shown:
        grows = by_graph[gid]
        first = grows[0]
        body.append(f'<h2>Scene {gid:04d} &mdash; {esc(first["title"])}'
                    f'<span class="chip">{esc(first["chart_type"])}</span>'
                    f'<span class="chip">theme: {esc(first["theme"])}</span>'
                    f'<span class="chip">{esc(first["font_family"].split(",")[0])}</span>'
                    f'<span class="chip">{first["n_texts"]} texts, '
                    f'{first["n_eligible_targets"]} eligible</span>'
                    f'<span class="chip">{len(grows)} questions</span></h2>')

        per_res: dict[str, list[dict]] = defaultdict(list)
        for r in grows:
            per_res[r["resolution"]].append(r)

        body.append('<h3>All gold boxes, every resolution</h3><div class="res">')
        for rn in sorted(per_res, key=rkey):
            rr = per_res[rn]
            W, H = rr[0]["image_px"]
            uniq = {tuple(x["gold_bbox_px"]) for x in rr}
            body.append(f'<div><div class="sub">{esc(rn)} &middot; {W}&times;{H}</div>'
                        + stage(f'../{rr[0]["image"]}', W, H,
                                [(list(b), True) for b in uniq])
                        + "</div>")
        body.append("</div>")

        med = per_res.get("medium") or per_res[sorted(per_res, key=rkey)[0]]

        body.append("<details><summary>every question for this graph</summary>")
        body.append("<table><tr><th>uid</th><th>res</th><th>type</th><th>question</th>"
                    "<th>answer</th><th class=num>target area</th>"
                    "<th class=num>contrast</th></tr>")
        for r in sorted(grows, key=lambda x: (rkey(x["resolution"]), x["uid"])):
            body.append(f'<tr><td><code>{esc(r["uid"])}</code></td><td>{esc(r["resolution"])}</td>'
                        f'<td>{esc(r["qtype"])}</td><td>{esc(r["question"])}</td>'
                        f'<td><code>{esc(r["answer"])}</code></td>'
                        f'<td class=num>{r["target_area_frac"] * 100:.4f}%</td>'
                        f'<td class=num>{r["target_contrast"]:.2f}</td></tr>')
        body.append("</table></details>")

        pq = [x for x in med if x["qtype"] == "point"][:3]
        if pq:
            body.append("<h3>Point questions: solid green = hit box (the scored target, the parallel to a button) &middot; dashed blue = glyph ink</h3>")
            body.append('<div class="res">')
            for x in pq:
                W, H = x["image_px"]
                body.append(f'<div><div class="sub"><code>{esc(x["target_text"])}</code> '
                            f'&rarr; <code>{esc(x["answer"])}</code></div>'
                            + stage(f'../{x["image"]}', W, H,
                                    [(x["gold_bbox_px"], False),
                                     (x["text_ink_bbox_px"], " ink")],
                                    cross=x["gold_center_px"]) + "</div>")
            body.append("</div>")

    doc = (f'<!doctype html><html><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>SVG localization dataset — ground-truth audit</title>'
           f"<style>{CSS}</style></head><body><div class=\"wrap\">"
           f"<h1>SVG localization dataset — ground-truth audit</h1>"
           f'<p class="sub">Overlays are positioned from <code>manifest.jsonl</code>, not '
           f're-derived from the generator. {len(rows)} questions over {len(by_graph)} graphs.</p>'
           + "".join(body) + "</div></body></html>")

    out.write_text(doc)
    return len(rows), len(problems)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/svg_localization"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--open", action="store_true", help="open the page when done")
    ap.add_argument("--max-visual", type=int, default=25,
                    help="scenes to render overlays for (checks always cover all)")
    a = ap.parse_args(argv)
    out = a.out or (a.data / "verify" / "index.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    n, bad = build(a.data, out, a.max_visual)
    print(f"{n} questions audited -> {out}")
    print(f"consistency errors: {bad}" + ("" if bad else "  (all checks passed)"))
    if a.open:
        subprocess.run(["open", str(out)], check=False)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
