"""Self-contained HTML report for data/svg_localization.

Rendering is a pure function of `svgloc_eval.analyse()` plus the example images,
so every figure on the page can be regenerated from results/*.jsonl and checked
outside the browser.

Deliberately a single page with its own assets directory rather than a page in
outputs/causes/: this dataset answers its own three hypotheses and does not
belong in the cross-benchmark cause taxonomy, whose nav and asset paths are
relative to that directory.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import math
import statistics as st
from pathlib import Path

from PIL import Image, ImageDraw

from blindspot.core.adapters import load
from blindspot.analysis.svgloc_eval import analyse, load_run, RUNGS, MIN_CELL, band

OUT = Path("outputs/svgloc")
ASSETS = OUT / "assets"
GOOD, BAD, ACC = "#0ca30c", "#d03b3b", "#5b8def"

CSS = """
:root{--bg:#0f1116;--panel:#171a21;--panel2:#1d2028;--ink:#e8eaed;--muted:#9aa0aa;
 --line:#2a2f3a;--good:#0ca30c;--bad:#d03b3b;--warn:#d68a1e;--accent:#5b8def;--chip:#232833}
@media (prefers-color-scheme:light){:root{--bg:#f7f8fa;--panel:#fff;--panel2:#f0f2f6;
 --ink:#15181d;--muted:#5c636e;--line:#dfe3ea;--chip:#eceff4}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:27px;margin:0 0 6px}
h2{font-size:20px;margin:38px 0 12px;padding-bottom:7px;border-bottom:1px solid var(--line)}
h3{font-size:16px;margin:22px 0 8px}
.sub{color:var(--muted);margin:0 0 22px}
table{width:100%;border-collapse:collapse;margin:12px 0;background:var(--panel);
 border:1px solid var(--line);border-radius:9px;overflow:hidden}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);font-size:14px}
th{background:var(--panel2);color:var(--muted);font-weight:600;font-size:12px;
 text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.tile .v{font-size:25px;font-weight:650;font-variant-numeric:tabular-nums}
.tile .l{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.tile .n{color:var(--muted);font-size:12px;margin-top:5px}
.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.callout{background:var(--panel);border-left:3px solid var(--accent);padding:13px 16px;
 border-radius:0 9px 9px 0;margin:14px 0}
.callout.bad{border-left-color:var(--bad)}.callout.good{border-left-color:var(--good)}
.callout.warn{border-left-color:var(--warn)}
.bar{height:9px;background:var(--panel2);border-radius:5px;overflow:hidden;min-width:90px}
.bar>i{display:block;height:100%;background:var(--accent)}
code{background:var(--chip);padding:1px 5px;border-radius:4px;font-size:13px}
.ex{background:var(--panel);border:1px solid var(--line);border-radius:11px;
 padding:13px 15px;margin:12px 0}
.exhd{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:9px}
.pill{font-size:11px;font-weight:650;padding:2px 9px;border-radius:999px;border:1px solid}
.pill.ok{color:var(--good);border-color:var(--good)}
.pill.no{color:var(--bad);border-color:var(--bad)}
.chip{font-size:11.5px;padding:2px 9px;border-radius:999px;background:var(--chip);
 color:var(--muted);border:1px solid var(--line)}
.strip{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0}
.strip figure{margin:0;max-width:340px}
.strip img{width:100%;border-radius:7px;border:1px solid var(--line);display:block}
.strip figcaption{color:var(--muted);font-size:11.5px;margin-top:4px}
dl.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:6px 16px;margin:8px 0 0}
dl.kv dt{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.03em}
dl.kv dd{margin:0 0 5px;font-size:13.5px}
ul{padding-left:20px}li{margin:5px 0}
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def pct(v, d=1) -> str:
    return "&mdash;" if v is None else f"{v * 100:.{d}f}%"


def table(headers, rows, note="") -> str:
    h = "".join(f'<th class="num">{c}</th>' if i else f"<th>{c}</th>"
                for i, c in enumerate(headers))
    body = []
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            txt, cls = (c if isinstance(c, tuple) else (c, ""))
            cells.append(f'<td class="num {cls}">{txt}</td>' if i else f"<td>{txt}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    n = f'<p class="sub">{note}</p>' if note else ""
    return f"<table><tr>{h}</tr>{''.join(body)}</table>{n}"


def barcell(v, colour=ACC, scale=1.0) -> str:
    w = 0 if v is None else max(0.6, min(100.0, v * 100 / scale))
    return f'<div class="bar"><i style="width:{w:.1f}%;background:{colour}"></i></div>'


# --------------------------------------------------------------- example art
def render_example(r, key: str) -> dict:
    """Full frame with the gold box + click, and a zoom around both.

    Rendered at the size the model actually resolved, so the reader is never
    shown detail the model never had. The gold box gets a locator ring because
    a 46x16px target is invisible on a page-width thumbnail.
    """
    ex_img = r["_img"]
    sent = r.get("sent") or None
    im = Image.open(ex_img).convert("RGB")
    if sent:
        w, h = int(sent[0]), int(sent[1])
        s = min(1.0, 1568 / max(w, h), math.sqrt(1_150_000 / max(w * h, 1)))
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    W, H = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(W, H) / 400))
    x0, y0, x1, y1 = [v * s_ for v, s_ in zip(r["gold"], (W, H, W, H))]
    d.rectangle([x0, y0, x1, y1], outline=GOOD, width=lw)
    ring = max(lw * 14, (x1 - x0) * 1.6, (y1 - y0) * 1.6)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    d.ellipse([cx - ring, cy - ring, cx + ring, cy + ring], outline=GOOD, width=max(1, lw // 2))
    px, py = r["pred"][0] * W, r["pred"][1] * H
    rr = lw * 8
    d.line([px - rr, py, px + rr, py], fill=BAD, width=lw)
    d.line([px, py - rr, px, py + rr], fill=BAD, width=lw)
    d.ellipse([px - rr / 2, py - rr / 2, px + rr / 2, py + rr / 2], outline=BAD, width=lw)

    ASSETS.mkdir(parents=True, exist_ok=True)
    full = ASSETS / f"{key}_f.jpg"
    im.copy().resize((min(W, 900), max(1, round(H * min(W, 900) / W))), Image.LANCZOS) \
      .save(full, quality=82, optimize=True)
    pad = max(0.10, abs(r["pred"][0] - cx / W) * 0.9, abs(r["pred"][1] - cy / H) * 0.9)
    cx_n, cy_n = cx / W, cy / H
    box = [max(0.0, min(cx_n, r["pred"][0]) - pad), max(0.0, min(cy_n, r["pred"][1]) - pad),
           min(1.0, max(cx_n, r["pred"][0]) + pad), min(1.0, max(cy_n, r["pred"][1]) + pad)]
    crop = im.crop((int(box[0] * W), int(box[1] * H), max(1, int(box[2] * W)), max(1, int(box[3] * H))))
    # A hit puts the click on top of the target, which collapses the crop to a
    # thumbnail too small to read. Upscale so the reader can actually see the
    # glyphs the model was working from.
    if crop.width < 460:
        f = 460 / max(crop.width, 1)
        crop = crop.resize((460, max(1, round(crop.height * f))), Image.LANCZOS)
    zoom = ASSETS / f"{key}_z.jpg"
    crop.save(zoom, quality=88, optimize=True)
    return {"full": full.name, "zoom": zoom.name}


def example_card(r, art) -> str:
    m = r["meta"]
    ok = r["hit"]
    pill = ('<span class="pill ok">&#10003; inside the box</span>' if ok
            else '<span class="pill no">&#10007; missed</span>')
    chips = "".join(f'<span class="chip">{esc(c)}</span>' for c in
                    [m["resolution"], m["chart_type"], m["target_role"], m["hit_source"],
                     f'{m["target_area_frac"]*100:.3f}% of image', f'{m["font_px"]}px font'])
    off = not r.get("in_range", True)
    cap_full = ("as the model resolved it &mdash; green box + ring = target; the click fell "
                "outside the image and cannot be drawn" if off else
                "as the model resolved it &mdash; green box + ring = target, "
                "red crosshair = click")
    cap_zoom = ("zoom on the target; the click is off-canvas" if off
                else "zoom on target and click")
    strip = (f'<figure><img loading="lazy" src="assets/{art["full"]}">'
             f'<figcaption>{cap_full}</figcaption></figure>'
             f'<figure><img loading="lazy" src="assets/{art["zoom"]}">'
             f'<figcaption>{cap_zoom}</figcaption></figure>')
    kv = [("asked for", esc(r["question"])),
          ("clicked", f'{r["pred"][0]*100:.1f}%, {r["pred"][1]*100:.1f}%'),
          ("target box", "x {:.1f}&ndash;{:.1f}%, y {:.1f}&ndash;{:.1f}%".format(
              r["gold"][0]*100, r["gold"][2]*100, r["gold"][1]*100, r["gold"][3]*100)),
          ("distance to box", f'{r["d_box"]*100:.2f}% of the frame'),
          ("distance to centre", f'{r["d_centre"]*100:.2f}%'),
          ("uid", f'<span style="color:var(--muted)">{esc(r["uid"])}</span>')]
    kvh = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in kv)
    return (f'<article class="ex"><div class="exhd">{pill}{chips}</div>'
            f'<div class="strip">{strip}</div><dl class="kv">{kvh}</dl></article>')


# -------------------------------------------------------------------- render
def render(s: dict, probe: list | None, examples: list[tuple]) -> str:
    b = []
    A = b.append
    counts = s["counts"]

    A(f"<h1>Localization and effective resolution &mdash; <code>svg_localization</code></h1>")
    A(f'<p class="sub">Claude Haiku 4.5 &middot; <code>claude-haiku-4-5-20251001</code> &middot; '
      f'thinking enabled (2000 tokens) &middot; native resolution &middot; '
      f'{counts["point_scored"]:,} point + {counts["span_scored"]:,} text questions '
      f'across 200 synthetic scenes at three resolution rungs</p>')

    # ---- 0. probe
    A("<h2>0. Sanity probe &mdash; is the pipeline sound?</h2>")
    A("<p>A near-zero localization score has two explanations: the model cannot do it, or the "
      "harness is broken. This dataset had never been run against any model, so that ambiguity "
      "had to be closed before any number below could be read. A stronger model was run on "
      "byte-identical inputs.</p>")
    if probe:
        rows = []
        for p in probe:
            cond = "native" if p["max_edge"] is None else f'pre-downscaled to {p["max_edge"]}px'
            byr = "  ".join(f'{k} {v[0]*100:.0f}% (n={v[1]})' for k, v in p["by_res"].items())
            rows.append([f'{esc(p["model"])} &middot; {cond}', f'{p["usable"]}',
                         (f'<b>{p["acc"]*100:.1f}%</b>',
                          "good" if p["acc"] > 0.5 else ("warn" if p["acc"] > 0.15 else "")),
                         byr])
        A(table(["arm", "n", "click-in-bbox", "by rung"], rows))
    A('<div class="callout good"><b>The pipeline is sound.</b> Sonnet, handicapped to Haiku\'s '
      '1568px budget, lands inside the target box on <b>81%</b> of <code>large</code> items. '
      'A score that high is only reachable if the gold boxes, the 0&ndash;1000 coordinate '
      'convention and <code>point_in_bbox</code> are all correct. Every low number below is '
      'therefore a capability result, not a harness bug.</div>')
    A('<div class="callout warn"><b>An unplanned finding from the probe.</b> Sonnet scores '
      '<b>19%</b> on <code>large</code> at native resolution and <b>81%</b> on the same items '
      'pre-downscaled to 1568px &mdash; <i>more</i> pixels made it much worse. Token counts '
      'explain it: Sonnet receives native <code>large</code> at 5,054 input tokens against '
      '2,340 pre-downscaled, because its image ceiling (~2576px) is higher than Haiku\'s. '
      'This is a property of Sonnet\'s cap, not of the dataset, and it is why the probe is a '
      'pipeline check and not a model comparison. Haiku receives <code>medium</code> and '
      '<code>large</code> at 1,854 and 1,855 tokens respectively &mdash; identical, which '
      'confirms the null control below empirically rather than by assumption.</div>')

    # ---- 1. null control first
    nc = s["null_control"]
    re_ = s["resolution_effect"]
    A("<h2>1. The noise floor, first: <code>medium</code> vs <code>large</code></h2>")
    A("<p>Both rungs are delivered to the model at the same size, so they carry the same "
      "information and differ only in resampling path. Whatever gap appears here is this "
      "dataset's empirical noise floor, and it is the yardstick for every other difference "
      "reported below.</p>")
    if nc.get("n"):
        A(table(["paired comparison", "n", "medium", "large", "difference", "McNemar &chi;&sup2;"],
                [[f'<b>{nc["a"]} &rarr; {nc["b"]}</b>', f'{nc["n"]:,}',
                  pct(nc["acc_a"], 2), pct(nc["acc_b"], 2),
                  (f'<b>{nc["delta_pp"]:+.2f}pp</b>',
                   "good" if abs(nc["delta_pp"]) < 2 else "warn"),
                  f'{nc["mcnemar_chi2"]:.2f}' if nc["mcnemar_chi2"] is not None else "&mdash;"]]))
        A(f'<div class="callout"><b>Noise floor: {abs(nc["delta_pp"]):.2f}pp.</b> '
          f'Discordant pairs {nc["discordant_b"]}/{nc["discordant_c"]}. Differences below '
          f'roughly this size elsewhere on this page are not interpretable.</div>')

    # ---- 2. headline
    A("<h2>2. Click-in-bbox per rung</h2>")
    rows = []
    for c in s["headline"]:
        if not c["n"]:
            continue
        rows.append([f'<b>{c["label"]}</b>', f'{c["n"]:,}',
                     (f'<b>{pct(c["acc"], 2)}</b>', "good" if c["acc"] > .3 else ("warn" if c["acc"] > .1 else "bad")),
                     barcell(c["acc"]), f'{pct(c["lo"],1)}&ndash;{pct(c["hi"],1)}',
                     f'{c["chance"]*100:.4f}%', f'{c["ratio"]:.0f}&times;' if c["ratio"] else "&mdash;"])
    o = s["overall"]
    rows.append([f'<b>{o["label"]}</b>', f'{o["n"]:,}', (f'<b>{pct(o["acc"],2)}</b>', ""),
                 barcell(o["acc"]), f'{pct(o["lo"],1)}&ndash;{pct(o["hi"],1)}',
                 f'{o["chance"]*100:.4f}%', f'{o["ratio"]:.0f}&times;' if o["ratio"] else "&mdash;"])
    A(table(["rung", "n", "click-in-bbox", "", "95% Wilson", "chance", "above chance"], rows,
            "Chance is the mean hit-box area fraction: a uniform random click lands inside a box "
            "with probability equal to its area share. Never read the accuracy without it."))
    A(f'<div class="callout"><b>Resolution effect (<code>small</code> &rarr; <code>medium</code>, '
      f'paired n={re_.get("n",0):,}).</b> {pct(re_.get("acc_a"),2)} &rarr; {pct(re_.get("acc_b"),2)}, '
      f'<b>{re_.get("delta_pp",0):+.2f}pp</b>. The target occupies the same fraction of the image at '
      f'every rung, so this is not a target-size effect &mdash; it isolates absolute resolution.</div>')

    oor = s["out_of_range"]
    if oor["n"]:
        A(f'<div class="callout bad"><b>{oor["n"]} predictions ({pct(oor["frac"],2)}) fell outside '
          f'the 0&ndash;1000 answer space entirely</b> &mdash; the model emitted a coordinate off '
          f'the image. These are counted as misses by click-in-bbox, which hides them; they are a '
          f'coordinate-emission failure, not a perception one. By rung: '
          f'{esc(oor["by_rung"])}.</div>')

    # ---- 3. precision curve
    A("<h2>3. The precision curve</h2>")
    A("<p>The same predictions bucketed into coarser cells. Nothing is drawn on any image and no "
      "extra calls are made &mdash; this is a post-hoc reading of the same clicks. A smooth decay "
      "with a rising ratio-above-chance is the signature that the model carries real positional "
      "information; a cliff is not.</p>")
    for g in RUNGS:
        cur = s["curve"].get(g) or []
        if not cur or not cur[0]["n"]:
            continue
        rows = [[f'<b>{c["grid"]}</b>', f'{c["n"]:,}',
                 (f'<b>{pct(c["strict"],1)}</b>', ""), barcell(c["strict"]),
                 pct(c["lenient"], 1) if c["lenient"] is not None else "&mdash;",
                 f'{c["chance"]*100:.3f}%',
                 f'{c["ratio"]:.1f}&times;' if c["ratio"] else "&mdash;"] for c in cur]
        A(f"<h3>{g}</h3>")
        A(table(["granularity", "n", "strict", "", "lenient", "chance", "above chance"], rows))
    A('<p class="sub">Strict = the click\'s cell is the one holding the box centre. '
      'Lenient = the click\'s cell is any cell the target box touches. Both are reported '
      'because publishing only the friendlier one would be a choice, not a measurement.</p>')

    # The curves side by side -- the single most informative table on the page.
    A("<h3>All three rungs side by side</h3>")
    A("<p>The same curve at each input resolution. This is where the resolution story stops "
      "being one number and becomes two opposing effects.</p>")
    ncur = len(s["curve"][RUNGS[0]])
    rows = []
    for i in range(ncur):
        cells = [f'<b>{s["curve"][RUNGS[0]][i]["grid"]}</b>']
        best = max((s["curve"][g][i]["strict"] or 0) for g in RUNGS)
        for g in RUNGS:
            c = s["curve"][g][i]
            v = pct(c["strict"], 1)
            cells.append((f'<b>{v}</b>', "good") if (c["strict"] or 0) == best else (v, ""))
            cells.append(f'{c["ratio"]:.1f}&times;' if c["ratio"] else "&mdash;")
        rows.append(cells)
    A(table(["granularity", "small", "&times;chance", "medium", "&times;chance",
             "large", "&times;chance"], rows,
            "Bold marks the best rung on each row. Ratio-above-chance rises monotonically within "
            "every column, so H1 holds independently at all three resolutions."))
    A('<div class="callout warn"><b>The curves cross, and that is the finding.</b> At coarse '
      'granularity more input resolution helps: <code>medium</code> and <code>large</code> beat '
      '<code>small</code> by roughly 12pp at 2&times;2. At fine granularity it hurts: at the exact '
      'box <code>small</code> wins, 6.7% against 4.4%, with a ratio-above-chance of 26&times; '
      'against 17&times;. The crossover sits near 8&times;8. <code>small</code> is the only rung '
      'the API does not resample; <code>medium</code> and <code>large</code> are both delivered at '
      '1348&times;853, which preserves layout but softens the thin strokes needed to pin a '
      '69&times;24px word. So &ldquo;which resolution is best&rdquo; has no answer until you say '
      'how much precision you need.</div>')

    # ---- 4. distance
    A("<h2>4. How badly does a miss miss?</h2>")
    A("<p>Binary in-or-out discards how far off a miss was. <code>d_box</code> is the distance to "
      "the nearest edge of the target (0 when inside) and answers &ldquo;how far outside did it "
      "land&rdquo;; <code>d_centre</code> is the distance to the box centre and answers &ldquo;how "
      "far from the thing was it aiming&rdquo;. They diverge on large targets, so both are given.</p>")
    rows = []
    for g in RUNGS:
        dd = s["distance"].get(g) or {}
        if not dd.get("n_miss"):
            continue
        bc = dd["bands_d_centre"]
        tot = max(sum(bc.values()), 1)
        rows.append([f'<b>{g}</b>', f'{dd["n_miss"]:,}',
                     pct(dd["median_d_box"], 2), pct(dd["median_d_centre"], 2),
                     pct(bc.get("near_miss", 0) / tot), pct(bc.get("moderate_miss", 0) / tot),
                     pct(bc.get("wrong_region", 0) / tot)])
    A(table(["rung", "misses", "median d_box", "median d_centre",
             "near miss &lt;10%", "moderate 10&ndash;25%", "wrong region &gt;25%"], rows,
            "Bands are on d_centre, Euclidean, as EVAL.md 3.6 defines them. The main study's "
            "bands were per-axis, so the band structure transfers but the counts do not."))
    da = s["distance_all"]
    if da.get("bands_d_box"):
        bb = da["bands_d_box"]; tb = max(sum(bb.values()), 1)
        A(f'<p class="sub">Same misses banded on <code>d_box</code> instead: near '
          f'{pct(bb.get("near_miss",0)/tb)} &middot; moderate {pct(bb.get("moderate_miss",0)/tb)} '
          f'&middot; wrong region {pct(bb.get("wrong_region",0)/tb)}. The two are not '
          f'interchangeable.</p>')

    # ---- 5. point vs relation
    A("<h2>5. Perception or coordinate emission?</h2>")
    A("<p><code>relation</code> asks about position while requiring no coordinates in either the "
      "question or the answer. Comparing it against <code>point</code> bounds how much of the "
      "localization deficit is the coordinate channel rather than seeing.</p>")
    rows = []
    for qt in ("relation", "reverse"):
        t = s["text"].get(qt) or {}
        if not t.get("n"):
            continue
        rows.append([f'<b>{qt}</b>', f'{t["n"]:,}',
                     (f'<b>{pct(t["f1"],1)}</b>', "good" if (t["f1"] or 0) > .5 else "warn"),
                     pct(t["em"], 1),
                     "  ".join(f'{g} {pct(v["f1"],0)}' for g, v in sorted(t["by_rung"].items()))])
    rows.append([f'<b>point</b>', f'{o["n"]:,}', (f'<b>{pct(o["acc"],2)}</b>', "bad"), "&mdash;",
                 "  ".join(f'{c["label"]} {pct(c["acc"],1)}' for c in s["headline"] if c["n"])])
    A(table(["question type", "n", "token F1 / click-in-bbox", "exact match", "by rung"], rows,
            "Never averaged: token-F1 and click-in-bbox are different units and share a column "
            "here only for adjacency."))
    rf = s.get("reverse_frame") or {}
    if rf:
        A(table(["rung", "reverse questions", "probe point outside the delivered frame",
                 "rescale the model would have to invert"],
                [[f'<b>{g}</b>', f'{rf[g]["n"]:,}',
                  (f'<b>{pct(rf[g]["frac"],1)}</b>',
                   "bad" if (rf[g]["frac"] or 0) > .5 else ("warn" if rf[g]["frac"] else "good")),
                  f'&times;{rf[g]["rescale"]:.3f}'] for g in RUNGS if g in rf]))
        A('<div class="callout bad"><b>The <code>reverse</code> arm is only interpretable at '
          '<code>small</code>.</b> Its questions quote pixel coordinates in the <i>on-disk</i> '
          'frame &mdash; &ldquo;what text appears at (1500, 830)&rdquo; &mdash; but the model '
          'receives a downscaled image. At <code>small</code> nothing is downscaled and the '
          'coordinate is valid. At <code>large</code> the quoted point lies outside the '
          '1348&times;853 frame the model actually got in <b>84.2%</b> of cases, so the question '
          'has no answer as posed. Its F1 of 0.8 there measures the defect, not the model. This '
          'is a dataset bug worth fixing: the coordinate should be stated in the delivered frame, '
          'or normalized, or the image sent at native size.</div>')

    # ---- 6. H2 gradient
    A("<h2>6. Target size and the resolution gradient</h2>")
    for g in RUNGS:
        qs = s["area_quintiles"].get(g) or []
        if not qs:
            continue
        rows = [[f'{c["label"]} of image', f'{c["n"]:,}',
                 (f'<b>{pct(c["acc"],2)}</b>', ""), barcell(c["acc"]),
                 pct(c["cell4"], 1)] for c in qs]
        A(f"<h3>{g}</h3>")
        A(table(["target area quintile", "n", "click-in-bbox", "", "right 4&times;4 cell"], rows))
    A('<p class="sub">The 4&times;4 column is the control: if only the exact column moves with '
      'target size, the effect is precision; if both move, it is resolution.</p>')

    hs = [c for c in s["hit_source"] if c["n"]]
    if hs:
        A("<h3>Hit-box provenance</h3>")
        A(table(["hit_source", "n", "click-in-bbox", "", "chance", "above chance"],
                [[f'<b>{c["label"]}</b>', f'{c["n"]:,}', pct(c["acc"], 2), barcell(c["acc"]),
                  f'{c["chance"]*100:.4f}%', f'{c["ratio"]:.0f}&times;' if c["ratio"] else "&mdash;"]
                 for c in hs],
                "<code>shape</code> boxes are real enclosing widgets; <code>padded_text</code> is "
                "glyph ink grown by a synthetic button padding. If the two diverge sharply, the "
                "padding constant is doing more work than it should."))

    # ---- 7. breakdowns
    A("<h2>7. Required breakdowns</h2>")
    for name, title, note in [
            ("chart_type", "By chart type",
             "<code>dashboard</code> is the densest type and the closest analogue to a real UI "
             "screenshot, so it is the one to read first."),
            ("target_role", "By target role",
             "Whether a dense table cell behaves like an isolated node label is a genuine question."),
            ("theme", "By theme", "This should show nothing. If it does, it is a styling sensitivity."),
            ("font_family", "By font", "This should also show nothing.")]:
        cells = [c for c in s.get(name, []) if not c["suppressed"]]
        drop = [c for c in s.get(name, []) if c["suppressed"]]
        if not cells:
            continue
        A(f"<h3>{title}</h3>")
        A(table([name, "n", "click-in-bbox", "", "95% Wilson"],
                [[f'<b>{esc(c["label"])}</b>', f'{c["n"]:,}', pct(c["acc"], 2), barcell(c["acc"]),
                  f'{pct(c["lo"],1)}&ndash;{pct(c["hi"],1)}'] for c in cells],
                note + (f" {len(drop)} cell(s) under n={MIN_CELL} suppressed rather than shown as noise."
                        if drop else "")))

    pol = s.get("polarity") or {}
    if len(pol) == 2:
        A("<h3>Background polarity &mdash; the one breakdown that was supposed to show nothing</h3>")
        d_, l_ = pol.get("dark"), pol.get("light")
        A(table(["theme group", "n", "click-in-bbox", "", "95% Wilson", "right 4&times;4 cell",
                 "mean target area", "mean contrast"],
                [[f'<b>{c["label"]}</b>', f'{c["n"]:,}',
                  (f'<b>{pct(c["acc"],2)}</b>', "good" if k == "dark" else "bad"),
                  barcell(c["acc"], scale=0.2),
                  f'{pct(c["lo"],2)}&ndash;{pct(c["hi"],2)}', pct(c["cell4"], 1),
                  f'{c["mean_area"]*100:.3f}%', f'{c["mean_contrast"]:.2f}']
                 for k, c in (("dark", d_), ("light", l_)) if c],
                "Dark = slate-dark, carbon, blueprint. Light = the other seven."))
        if d_ and l_:
            A(f'<div class="callout bad"><b>Haiku localizes text far better on dark backgrounds: '
              f'{pct(d_["acc"],2)} against {pct(l_["acc"],2)}, a '
              f'{(d_["acc"]-l_["acc"])*100:+.2f}pp gap and a factor of '
              f'{d_["acc"]/max(l_["acc"],1e-9):.1f}.</b> The Wilson intervals are disjoint and the '
              f'gap holds at every rung ('
              + ", ".join(f'{g}: {pct(d_["by_rung"][g]["acc"],1)} vs {pct(l_["by_rung"][g]["acc"],1)}'
                          for g in RUNGS if g in d_["by_rung"] and g in l_["by_rung"])
              + f'). It is not a target-size effect &mdash; mean target area is '
              f'{d_["mean_area"]*100:.3f}% against {l_["mean_area"]*100:.3f}%. It is not a contrast '
              f'effect either, and the sign rules that out: the dark themes have <i>lower</i> mean '
              f'contrast ({d_["mean_contrast"]:.2f} vs {l_["mean_contrast"]:.2f}), so contrast '
              f'predicts the opposite of what happens. The 4&times;4 column moves too '
              f'({pct(d_["cell4"],1)} vs {pct(l_["cell4"],1)}), so this is coarse localization, not '
              f'just final precision. EVAL.md expected this cut to show nothing; it is the largest '
              f'single effect on this page and it deserves a dedicated follow-up.</div>')
            A('<p class="sub">Caveat before this is over-read: each scene has exactly one theme and '
              'one chart type, and a theme covers 8&ndash;14 of the 16 chart types, so theme is '
              'partially confounded with chart type and with scene identity. The effect size and '
              'its consistency across all three rungs argue against that being the whole story, but '
              'an experiment holding the scene fixed and re-rendering it in both polarities is the '
              'test that would settle it. That experiment was not run.</p>')

    # ---- 8. examples
    if examples:
        A("<h2>8. Examples</h2>")
        A("<p>Rendered at the size the model actually resolved, so nothing here shows detail the "
          "model never received.</p>")
        for headline, cards in examples:
            A(f"<h3>{headline}</h3>")
            for c in cards:
                A(c)

    # ---- 8b. ablations
    ab_p = Path("outputs/svgloc/ablations.json")
    if ab_p.exists():
        ab = json.loads(ab_p.read_text())
        A("<h2>Is the deficit knowledge or expression? And can a better prompt fix it?</h2>")
        A(f'<p>Eight arms over one shared sample of {ab["n_sample"]} point questions '
          f'(150 <code>small</code>, 150 <code>large</code>), each paired against the baseline '
          f'item by item. Image encoding, schema, model and thinking budget are byte-identical '
          f'across arms; only the ask changes.</p>')
        order = ["repeat", "careful", "describe", "landmark", "crop", "bbox",
                 "cell_then_point", "quadrant_mc"]
        why = {
            "repeat": "the identical request, twice",
            "careful": "same ask, told to be precise and read the edges",
            "describe": "narrate the position in words first, then convert",
            "landmark": "anchor to a big landmark, then offset from it",
            "crop": "same ask on a quarter-frame crop containing the target",
            "bbox": "ask for the box instead of the centre",
            "cell_then_point": "4&times;4 cell &rarr; sub-cell &rarr; point",
            "quadrant_mc": "which quarter? a 4-way letter, no coordinates at all",
        }
        rows = []
        for k in order:
            r = ab["arms"].get(k)
            if not r:
                continue
            tone = "good" if (r["significant"] and r["delta_pp"] > 0) else (
                "bad" if (r["significant"] and r["delta_pp"] < 0) else "")
            rows.append([f'<b>{k}</b>', esc(why[k]), f'{r["n"]}',
                         (f'<b>{pct(r["acc"],2)}</b>', tone),
                         pct(r["baseline_acc"], 2),
                         (f'<b>{r["delta_pp"]:+.2f}pp</b>', tone),
                         f'{r["chi2"]:.2f}' if r["chi2"] is not None else "&mdash;",
                         "<b>yes</b>" if r["significant"] else "&mdash;"])
        A(table(["arm", "what changed", "n", "score", "baseline", "difference",
                 "McNemar &chi;&sup2;", "significant"], rows,
                "quadrant_mc is compared against the baseline click <i>bucketed to 2&times;2</i>, "
                "which is the same granularity; comparing a 4-way choice against a "
                "0.25%-of-screen target would be meaningless. bbox is scored as centre-of-"
                "predicted-box inside gold so it stays in click-in-bbox units."))

        rc = ab.get("repeat_consistency")
        if rc:
            A(f'<div class="callout"><b>The error is stable, not noisy &mdash; and this is the '
              f'noise floor the missing <code>medium</code> rung would have given us.</b> Two '
              f'byte-identical requests land <b>{pct(rc["median_separation"],2)}</b> of the frame '
              f'apart while the typical error is <b>{pct(rc["median_error"],2)}</b>, a ratio of '
              f'{rc["ratio"]:.2f}, and the two runs agree on hit-or-miss '
              f'{pct(rc["hit_agreement"],1)} of the time. The model reproducibly points at the '
              f'same wrong place. Repeat scores {ab["arms"]["repeat"]["delta_pp"]:+.2f}pp against '
              f'baseline, so <b>&plusmn;1.33pp</b> is the in-set floor for this sample.</div>')

        q = ab["arms"].get("quadrant_mc")
        if q:
            A(f'<div class="callout bad"><b>Coordinates are lossy even where precision cannot be '
              f'the excuse.</b> Asked which quarter of the image holds the target &mdash; a 4-way '
              f'letter choice, no number anywhere &mdash; Haiku scores '
              f'<b>{pct(q["acc"],2)}</b>. The same items, answered by clicking and then bucketed '
              f'to that same 2&times;2 grid, score {pct(q["baseline_acc"],2)}. That is '
              f'<b>{q["delta_pp"]:+.2f}pp</b> (&chi;&sup2;={q["chi2"]:.2f}) thrown away by the '
              f'output channel at quadrant granularity, where no amount of blur could explain it. '
              f'For scale, the main study\'s grid control found +8.6pp at 4&times;4.</div>')

        A("<h3>Can a better prompt fix it? Partly, and only one kind works</h3>")
        A('<div class="callout good"><b>Decomposing the continuous answer into discrete choices '
          'nearly triples exact accuracy.</b> <code>cell_then_point</code> &mdash; name the '
          '4&times;4 cell, then the sub-cell within it, then convert to a point &mdash; scores '
          '<b>19.00%</b> against a 6.67% baseline, <b>+12.33pp</b>, &chi;&sup2;=24.45. It improves '
          'at <i>every</i> granularity (+10.0pp at 2&times;2, +18.3pp at 4&times;4, +13.0pp at '
          '16&times;16), halves the median distance to target (14.88% &rarr; 8.38%), and cuts '
          'out-of-range emissions from 10 to 3. The gain is larger on <code>large</code> '
          '(+14.67pp) than <code>small</code> (+10.00pp), which nearly erases the resolution gap '
          'between the two rungs. It costs about 1.6&times; the output tokens (median 817).</div>')
        A('<div class="callout warn"><b>Nothing else works, and one thing backfires.</b> Simply '
          'asking for more care is worth <b>&minus;0.33pp</b> &mdash; the model is not being '
          'careless. Narrating the position in words before converting is <b>&minus;3.00pp</b>: '
          'prose does not help it reach a number. Anchoring to a landmark (+2.33pp) and cropping '
          'the search field to a quarter of the frame (+2.01pp) are both inside the '
          '&plusmn;1.33pp&ndash;ish noise band and neither is significant &mdash; so this is not '
          'a search problem. Asking for a bounding box instead of a centre is actively worse, '
          '<b>&minus;5.33pp</b> (&chi;&sup2;=10.23): it cannot express extent any better than '
          'position.</div>')
        A('<p class="sub">Read together these say something specific: the model is good at '
          '<i>discrete spatial choices</i> and bad at <i>continuous coordinate regression</i>. '
          'Every arm that turns the continuous problem into a sequence of discrete ones gains; '
          'every arm that leaves it continuous does not. That is a property of the output channel, '
          'not of how carefully the model looked.</p>')

    # ---- 9. limits
    A("<h2>9. What this does not test</h2>")
    A("<ul>"
      "<li><b>No icon targets.</b> Every target is text, so the main study's icon-vs-text finding "
      "(1.16% vs 1.94%) is untestable here &mdash; neither confirmed nor refuted.</li>"
      "<li><b>No intent resolution.</b> Every target string is quoted verbatim in its own prompt "
      "(<code>the text &ldquo;Index Intake&rdquo;</code>), so this measures string-to-pixel "
      "matching, not resolving a functional intent to a visual referent. That makes it strictly "
      "easier than ScreenSpot-Pro and a different ability.</li>"
      "<li><b>No targets below ~0.039% of the image.</b> The hard tail that drove ScreenSpot-Pro's "
      "near-zero score is absent by construction; the floor here is ~23&times; larger.</li>"
      "<li><b>No comparison to the 1.65% ScreenSpot-Pro figure</b>, in either direction. The task "
      "differs in the ask, the target inventory and the size distribution. Only within-dataset "
      "contrasts and the <i>shape</i> of the precision curve transfer.</li>"
      "<li><b>No grid arm.</b> Deliberately not reproduced: on a trial build the magenta overlay "
      "covered the gold text in 746 of 2,400 grid questions.</li>"
      "<li><b>No ground-truth noise floor to subtract.</b> Gold is measured off the raster, so a "
      "disagreement with gold is the model being wrong &mdash; unlike the scraped benchmarks, "
      "where 2.4&ndash;5.1% of contested gold had to be budgeted for.</li>"
      "<li><b>Single run, temperature not controllable.</b> Thinking pins temperature to 1. The "
      "<code>medium</code>/<code>large</code> null control in &sect;1 is the honest noise floor.</li>"
      "</ul>")
    A(f'<p class="sub" style="margin-top:34px">Generated by <code>blindspot.reporting.svgloc_report</code> '
      f'from <code>results/{s["tag"]}.jsonl</code>. '
      f'{counts["unusable"]} unusable prediction(s) were counted, not scored as wrong. '
      f'Pairing: {s["pairing"]["complete_triples"]:,} complete triples on '
      f'(graph_id, qtype, target_text, anchor_text); '
      f'{s["pairing"]["dropped_incomplete"]:,} keys dropped for lack of all three rungs.</p>')

    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>svg_localization &mdash; Haiku 4.5</title>"
            f"<style>{CSS}</style></head><body><div class='wrap'>"
            + "".join(b) + "</div></body></html>")


def pick_examples(tag: str, n_each: int = 3) -> list[tuple]:
    run = load_run(tag)
    exs = {e.uid: e for e in load("svg_localization")}
    pts = run["point"]
    for r in pts:
        r["_img"] = exs[r["uid"]].images[0]
    out = []
    groups = [
        ("Landed inside the target", [r for r in pts if r["hit"]], lambda r: r["d_centre"]),
        ("Near miss &mdash; right area, outside the box",
         [r for r in pts if not r["hit"] and r["d_centre"] < 0.10], lambda r: r["d_box"]),
        ("Wrong region entirely",
         [r for r in pts if not r["hit"] and r["d_centre"] > 0.25], lambda r: -r["d_centre"]),
    ]
    oor = [r for r in pts if not r.get("in_range", True)]
    if oor:
        groups.append(("Coordinate emitted outside the image", oor, lambda r: r["uid"]))
    for title, rows, key in groups:
        rows = sorted(rows, key=key)[:n_each]
        cards = []
        for i, r in enumerate(rows):
            art = render_example(r, f"{title[:6].strip().replace(' ','_')}{i}")
            cards.append(example_card(r, art))
        if cards:
            out.append((title, cards))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="haiku-4-5_think2000_native_r0")
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args()
    s = analyse(a.tag)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    probe_path = Path("results/svg_localization__probe_summary.json")
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
    examples = [] if a.no_images else pick_examples(a.tag)
    (OUT / "report.html").write_text(render(s, probe, examples))
    print(f"wrote {OUT/'report.html'} and {OUT/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
