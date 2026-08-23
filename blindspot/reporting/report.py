"""Build a self-contained HTML blind-spot report.

Two audiences in one page:

* **Stats** -- where Haiku 4.5 fails, sliced by the axes that could explain why
  (element type, target size, question type).
* **Evidence** -- the actual images for the hard cases, with the gold target and
  the model's answer drawn on them, because an aggregate score never shows you
  *what* the model was looking at when it missed.

Everything is inlined (images as data URIs, CSS and JS in the page) so the
report is one file you can mail to someone.

Usage:
    python -m blindspot.reporting.report                        # -> outputs/report.html
    python -m blindspot.reporting.report --gallery 8 --out x.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from blindspot.core.adapters import load
from blindspot.core.scoring import score, CHARXIV_FUZZY_QIDS

RESULTS = Path("results")
OUTPUTS = Path("outputs")

# Haiku 4.5's measured image ceiling (verified with count_tokens: 1568x882,
# 2576x1449 and 3840x2160 all bill 1572 input tokens).
HAIKU_MAX_EDGE = 1568
HAIKU_MAX_IMAGE_TOKENS = 1572

# --- palette -------------------------------------------------------------
# Slots 1 and 2 of the reference categorical palette. palette.md documents this
# pair as clearing every hard gate in both modes (worst adjacent CVD dE 9.1
# light / 8.4 dark; normal-vision 19.6 / 19.3), and the first three slots also
# validate all-pairs -- so a two-series chart is inside the documented-safe set.
SERIES = {"light": ["#2a78d6", "#eb6834"], "dark": ["#3987e5", "#d95926"]}
STATUS = {"good": "#0ca30c", "critical": "#d03b3b"}


# =========================================================================
# Load + score
# =========================================================================
def load_results(dataset: str, tag: str = "haiku-4-5_think2000_native_r0") -> list[dict]:
    """One row per uid, preferring a usable prediction over a later failure.

    A uid can appear more than once: reruns append, and the retry logic keys off
    whether a *usable* row exists rather than the newest row. Plain last-wins
    would let a truncated retry overwrite an earlier good answer and show up as a
    model failure. Rule: any row with a prediction beats one without; among rows
    with predictions, the most recent wins.
    """
    path = RESULTS / f"{dataset}__{tag}.jsonl"
    if not path.exists():
        return []
    by_uid: dict[str, dict] = {}
    for line in open(path):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        prev = by_uid.get(rec["uid"])
        if prev is None or prev.get("pred") is None or rec.get("pred") is not None:
            by_uid[rec["uid"]] = rec
    return list(by_uid.values())


def scored_rows(dataset: str, tag: str = "haiku-4-5_think2000_native_r0") -> list[dict]:
    """Attach a score to every successful row. Errors are kept but flagged."""
    examples = {e.uid: e for e in load(dataset)}
    out = []
    for rec in load_results(dataset, tag):
        ex = examples.get(rec["uid"])
        if ex is None:
            continue
        row = dict(rec)
        row["_example"] = ex
        if rec.get("error") or rec.get("pred") is None:
            row.update({"score": None, "metric": None, "grading_confidence": None,
                        "failed_call": True})
        else:
            row.update(score(ex, rec["pred"]))
            row["failed_call"] = False
        out.append(row)
    return out


def _acc(rows: list[dict]) -> float | None:
    vals = [r["score"] for r in rows if r["score"] is not None]
    return sum(vals) / len(vals) if vals else None


def wilson_ci(rows: list[dict]) -> tuple[float, float] | None:
    """95% Wilson interval. ANLS is continuous, so this is indicative, not exact."""
    vals = [r["score"] for r in rows if r["score"] is not None]
    n = len(vals)
    if not n:
        return None
    p, z = sum(vals) / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0)) / d
    return max(0.0, c - m), min(1.0, c + m)


def bucket_area(frac: float) -> str:
    """Target size buckets, labelled by what they become after downscaling."""
    px = frac * HAIKU_MAX_EDGE * (HAIKU_MAX_EDGE * 9 / 16)  # ~px at Haiku's ceiling
    side = math.sqrt(max(px, 0))
    if side < 12:
        return "&lt;12px"
    if side < 20:
        return "12-20px"
    if side < 32:
        return "20-32px"
    if side < 56:
        return "32-56px"
    return "&ge;56px"


AREA_ORDER = ["&lt;12px", "12-20px", "20-32px", "32-56px", "&ge;56px"]

# CharXiv's question text is full prose ("What is the difference between the
# maximum and minimum values of the tick labels on the continuous legend...").
# Charts need an axis label, not a paragraph.
CHARXIV_SHORT = {
    1: "plot title", 2: "x-axis label", 3: "y-axis label",
    4: "x-axis leftmost tick", 5: "x-axis rightmost tick",
    6: "y-axis lowest tick", 7: "y-axis highest tick",
    8: "x-axis tick spacing", 9: "y-axis tick spacing",
    10: "count lines", 11: "lines intersect?", 12: "count legend labels",
    13: "read legend labels", 14: "colorbar range", 15: "colorbar max",
    16: "overall trend", 17: "count all ticks",
    18: "subplot layout", 19: "count subplots",
}


# =========================================================================
# Image rendering for the evidence gallery
# =========================================================================
def _data_uri(im: Image.Image, max_w: int = 560, quality: int = 72) -> str:
    im = im.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def render_grounding(row: dict) -> tuple[str, str]:
    """Full screenshot with gold box + predicted point, plus a zoomed crop.

    The zoom is the point of this figure: on a 4K screenshot the target is often
    a couple of dozen pixels, so the full view alone cannot show whether the
    model was close or in the wrong region entirely.
    """
    ex = row["_example"]
    im = Image.open(ex.images[0]).convert("RGB")
    W, H = im.size
    x0, y0, x1, y1 = [c * s for c, s in zip(ex.gold, (W, H, W, H))]

    full = im.copy()
    d = ImageDraw.Draw(full)
    lw = max(2, round(max(W, H) / 500))
    d.rectangle([x0, y0, x1, y1], outline=STATUS["good"], width=lw)
    if row.get("pred"):
        px, py = row["pred"][0] * W, row["pred"][1] * H
        r = lw * 6
        d.line([px - r, py, px + r, py], fill=STATUS["critical"], width=lw)
        d.line([px, py - r, px, py + r], fill=STATUS["critical"], width=lw)
        d.ellipse([px - r / 2, py - r / 2, px + r / 2, py + r / 2],
                  outline=STATUS["critical"], width=lw)

    # Zoom window: big enough for context, always containing the gold target.
    pad = max((x1 - x0), (y1 - y0)) * 6 + 80
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (max(0, cx - pad), max(0, cy - pad), min(W, cx + pad), min(H, cy + pad))
    zoom = full.crop(tuple(round(v) for v in box))
    return _data_uri(full), _data_uri(zoom, max_w=420)


def render_thumb(row: dict) -> str:
    return _data_uri(Image.open(row["_example"].images[0]), max_w=460)


# =========================================================================
# HTML fragments
# =========================================================================
def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def bar_chart(title: str, subtitle: str, items: list[tuple[str, float | None, int]],
              series_label: str | None = None) -> str:
    """Horizontal bars, one series, slot-1 blue. Values direct-labelled at the end.

    A single series needs no legend box -- the title names it (per the skill's
    accessibility rule).
    """
    rows = []
    for label, val, n in items:
        if val is None:
            rows.append(
                f'<div class="row"><div class="rlab">{label}</div>'
                f'<div class="track"><span class="nodata">no data</span></div>'
                f'<div class="rval">&mdash;</div></div>')
            continue
        pct = val * 100
        rows.append(
            f'<div class="row" tabindex="0" data-tip="{label}: {pct:.1f}% accuracy &middot; n={n}">'
            f'<div class="rlab">{label}</div>'
            f'<div class="track"><div class="bar" style="width:{max(pct,0.6):.2f}%"></div></div>'
            f'<div class="rval">{pct:.0f}<span class="pcts">%</span> '
            f'<span class="nlab">n={n}</span></div></div>')
    sub = f'<p class="sub">{subtitle}</p>' if subtitle else ""
    return f'<figure class="chart"><figcaption><h3>{title}</h3>{sub}</figcaption>{"".join(rows)}</figure>'


def grouped_bar_chart(title: str, subtitle: str, groups: list[str],
                      series: list[tuple[str, dict[str, tuple[float | None, int]]]]) -> str:
    """Two series -> slots 1 and 2, with a legend (identity never color-alone)."""
    legend = "".join(
        f'<span class="lg"><i class="sw" style="background:var(--series-{i+1})"></i>{esc(name)}</span>'
        for i, (name, _) in enumerate(series))
    rows = []
    for g in groups:
        bars = []
        for i, (name, data) in enumerate(series):
            val, n = data.get(g, (None, 0))
            if val is None:
                bars.append('<div class="gbar-wrap"><span class="nodata">&mdash;</span></div>')
                continue
            pct = val * 100
            bars.append(
                f'<div class="gbar-wrap" tabindex="0" '
                f'data-tip="{esc(name)} &middot; {g}: {pct:.1f}% &middot; n={n}">'
                f'<div class="gbar s{i+1}" style="width:{max(pct,0.6):.2f}%"></div>'
                f'<span class="gval">{pct:.0f}%</span></div>')
        rows.append(f'<div class="grow"><div class="rlab">{g}</div>'
                    f'<div class="gbars">{"".join(bars)}</div></div>')
    sub = f'<p class="sub">{subtitle}</p>' if subtitle else ""
    return (f'<figure class="chart"><figcaption><h3>{title}</h3>{sub}</figcaption>'
            f'<div class="legend">{legend}</div>{"".join(rows)}</figure>')


def stat_tile(label: str, value: str, note: str, tone: str = "") -> str:
    return (f'<div class="tile {tone}"><div class="tlab">{esc(label)}</div>'
            f'<div class="tval">{value}</div><div class="tnote">{note}</div></div>')


# =========================================================================
# Image input pipeline diagram
# =========================================================================
def pipeline_diagram(stats: dict) -> str:
    """SVG of what happens to an image between disk and the model.

    Drawn to scale where it matters: the 'what the model sees' panel shows the
    real pixel budget, because the headline finding of this whole eval is that
    resolution is spent before the model gets a vote.
    """
    src_w, src_h = stats["example_src"]
    scale = HAIKU_MAX_EDGE / max(src_w, src_h)
    out_w, out_h = round(src_w * scale), round(src_h * scale)
    kept = (out_w * out_h) / (src_w * src_h) * 100
    tgt_native = stats["median_target_px"]
    tgt_after = tgt_native * scale

    def stage(x, label, detail, tone="n"):
        return f"""
  <g transform="translate({x},0)">
    <rect class="st {tone}" x="0" y="28" width="150" height="70" rx="8"/>
    <text class="sl" x="75" y="56">{label}</text>
    <text class="sd" x="75" y="76">{detail}</text>
  </g>"""

    def arrow(x):
        return (f'<path class="ar" d="M{x} 63 L{x+26} 63"/>'
                f'<path class="arh" d="M{x+26} 63 l-6 -4 v8 z"/>')

    gates = f"""
  <g transform="translate(0,132)">
    <rect class="gate" x="0" y="0" width="440" height="76" rx="8"/>
    <text class="gl" x="14" y="24">Hard ingest gates &mdash; violated = HTTP 400, image never seen</text>
    <text class="gd" x="14" y="44">max dimension 8000px &middot; max 10 MB base64</text>
    <text class="gd" x="14" y="62">hit {stats['preflight_n']}&times; in this run &rarr; pre-flight downscale added to the harness</text>
  </g>"""

    return f"""
<figure class="chart wide">
 <figcaption><h3>How an image reaches Haiku 4.5</h3>
 <p class="sub">Every stage is lossy or gated. By the time the model looks, resolution has
 already been spent &mdash; measured, not assumed (<code>count_tokens</code> bills 1568&times;882,
 2576&times;1449 and 3840&times;2160 identically at {HAIKU_MAX_IMAGE_TOKENS} tokens).</p></figcaption>
 <svg viewBox="0 0 880 235" class="pipe" role="img"
      aria-label="Image pipeline: source, ingest gates, API downscale to 1568px, tokenization, model input">
  <g transform="translate(8,4)">
   {stage(0, "Source on disk", f"{src_w}&times;{src_h}")}
   {arrow(150)}
   {stage(176, "Pre-flight", "fit 8000px / 10MB")}
   {arrow(326)}
   {stage(352, "API downscale", f"long edge &le;{HAIKU_MAX_EDGE}")}
   {arrow(502)}
   {stage(528, "Tokenize", f"&le;{HAIKU_MAX_IMAGE_TOKENS} tokens", "w")}
   {arrow(678)}
   {stage(704, "Haiku 4.5", f"{out_w}&times;{out_h}", "w")}
   {gates}
  </g>
  <g transform="translate(478,152)">
   <text class="gl" x="0" y="0">What survives</text>
   <text class="gd" x="0" y="22">{src_w}&times;{src_h} &rarr; {out_w}&times;{out_h} &mdash; {kept:.0f}% of original pixels kept</text>
   <text class="gd" x="0" y="40">median ScreenSpot-Pro target {tgt_native:.0f}px &rarr; <tspan class="bad">{tgt_after:.0f}px</tspan> on a side</text>
   <text class="gd" x="0" y="58">native resolution buys nothing: identical tokens, identical input</text>
  </g>
 </svg>
</figure>"""


# =========================================================================
# Evidence gallery
# =========================================================================
def gallery_grounding(rows: list[dict], n: int) -> str:
    """Hardest misses: wrong clicks, smallest target first."""
    misses = [r for r in rows if r.get("score") == 0.0 and r.get("pred")]
    misses.sort(key=lambda r: r["meta"].get("target_area_frac", 1))
    cards = []
    for r in misses[:n]:
        full, zoom = render_grounding(r)
        ex, m = r["_example"], r["meta"]
        side = math.sqrt(m.get("target_area_frac", 0) * HAIKU_MAX_EDGE * HAIKU_MAX_EDGE * 9 / 16)
        px, py = r["pred"]
        cards.append(f"""
<article class="case">
  <div class="chead">
    <span class="pill bad">&#10007; missed</span>
    <span class="ctitle">{esc(ex.question)}</span>
  </div>
  <div class="cimgs">
    <figure><img src="{full}" alt="Full screenshot with gold target and model click"><figcaption>full screen &middot; {m.get('img_size',['?','?'])[0]}&times;{m.get('img_size',['?','?'])[1]}</figcaption></figure>
    <figure><img src="{zoom}" alt="Zoomed crop around the gold target"><figcaption>zoom on target</figcaption></figure>
  </div>
  <dl class="cmeta">
    <div><dt>target size after downscale</dt><dd class="bad">&asymp;{side:.0f}px on a side</dd></div>
    <div><dt>model clicked</dt><dd>({px*100:.1f}%, {py*100:.1f}%)</dd></div>
    <div><dt>distance to target centre</dt><dd>{r.get('center_distance',0)*100:.1f}% of screen</dd></div>
    <div><dt>element type</dt><dd>{esc(m.get('ui_type'))} &middot; {esc(m.get('application') or m.get('platform'))}</dd></div>
  </dl>
</article>""")
    if not cards:
        return '<p class="empty">No grounding misses in this slice.</p>'
    return f'<div class="cases">{"".join(cards)}</div>'


def gallery_span(rows: list[dict], n: int, label: str) -> str:
    """Confident-but-wrong reading failures: score 0 with a non-empty answer."""
    misses = [r for r in rows if r.get("score") == 0.0 and r.get("pred")]
    misses.sort(key=lambda r: len(str(r.get("pred", ""))))
    cards = []
    for r in misses[:n]:
        ex = r["_example"]
        q = ex.question.split("\n")[0][:220]
        extra = ""
        if ex.dataset == "charxiv":
            conf = r.get("grading_confidence")
            extra = (f'<div><dt>question type</dt><dd>{esc(ex.meta.get("qlabel") or ex.meta.get("split"))}</dd></div>'
                     f'<div><dt>grading</dt><dd>{esc(conf)}{" (lower bound)" if conf=="fuzzy" else ""}</dd></div>')
        cards.append(f"""
<article class="case">
  <div class="chead">
    <span class="pill bad">&#10007; wrong</span>
    <span class="ctitle">{esc(q)}</span>
  </div>
  <div class="cimgs one"><figure><img src="{render_thumb(r)}" alt="Source image for this question"></figure></div>
  <dl class="cmeta">
    <div><dt>model answered</dt><dd class="bad">{esc(r["pred"])}</dd></div>
    <div><dt>gold</dt><dd class="good">{esc(" | ".join(map(str, ex.gold[:3])))}</dd></div>
    {extra}
  </dl>
</article>""")
    if not cards:
        return f'<p class="empty">No {label} failures in this slice.</p>'
    return f'<div class="cases">{"".join(cards)}</div>'


# =========================================================================
# Table view (accessibility twin -- every charted value readable as text)
# =========================================================================
def table_view(sections: list[tuple[str, list[tuple[str, float | None, int]]]]) -> str:
    blocks = []
    for title, items in sections:
        body = "".join(
            f'<tr><th scope="row">{lab}</th>'
            f'<td>{"&mdash;" if v is None else f"{v*100:.1f}%"}</td><td>{n}</td></tr>'
            for lab, v, n in items)
        blocks.append(f'<h4>{title}</h4><table><thead><tr><th scope="col">Slice</th>'
                      f'<th scope="col">Accuracy</th><th scope="col">n</th></tr></thead>'
                      f'<tbody>{body}</tbody></table>')
    return "".join(blocks)


CSS = """
*{box-sizing:border-box}
:root{
  color-scheme:light;
  --surface-1:#fcfcfb; --page:#f9f9f7;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --series-1:#2a78d6; --series-2:#eb6834;
  --good:#0ca30c; --critical:#d03b3b;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  color-scheme:dark;
  --surface-1:#1a1a19; --page:#0d0d0d;
  --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --series-1:#3987e5; --series-2:#d95926;
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --surface-1:#1a1a19; --page:#0d0d0d;
  --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --series-1:#3987e5; --series-2:#d95926;
}
html{background:var(--page)}
body{margin:0;background:var(--page);color:var(--text-primary);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.viz-root{background:var(--page);color:var(--text-primary)}
.viz-root{max-width:1180px;margin:0 auto;padding:32px 24px 80px}
header.top{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;margin-bottom:8px}
h1{font-size:27px;line-height:1.25;margin:0 0 6px}
.dek{color:var(--text-secondary);margin:0;max-width:68ch}
h2{font-size:19px;margin:44px 0 4px;padding-top:22px;border-top:1px solid var(--grid)}
h2 .h2sub{display:block;font-size:14px;font-weight:400;color:var(--text-secondary);margin-top:4px}
h4{font-size:14px;margin:22px 0 8px;color:var(--text-secondary)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9em}
button.theme{border:1px solid var(--border);background:var(--surface-1);color:var(--text-secondary);
  border-radius:8px;padding:7px 12px;cursor:pointer;font:inherit;font-size:13px;white-space:nowrap}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:12px;margin:20px 0 8px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:15px 16px}
.tlab{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.tval{font-size:33px;line-height:1.1;margin:7px 0 3px}
.tile.bad .tval{color:var(--critical)} .tile.good .tval{color:var(--good)}
.tnote{font-size:12.5px;color:var(--text-secondary)}

.chart{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;
  padding:18px 20px 20px;margin:16px 0}
.chart.wide{padding-bottom:8px}
figcaption h3{font-size:15.5px;margin:0 0 3px}
.sub{font-size:13px;color:var(--text-secondary);margin:0 0 14px;max-width:80ch}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}
@media(max-width:860px){.grid2{grid-template-columns:1fr}}

.row,.grow{display:grid;grid-template-columns:150px 1fr 74px;align-items:center;gap:12px;padding:5px 0}
.grow{grid-template-columns:150px 1fr}
.rlab{font-size:12.5px;color:var(--text-secondary);text-align:right;overflow-wrap:anywhere}
.track{height:15px;background:var(--grid);border-radius:4px;position:relative}
.bar{height:100%;background:var(--series-1);border-radius:0 4px 4px 0}
.rval{font-size:13px;line-height:1.35;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.pcts{color:var(--muted);font-size:11px}
.nlab{display:block;font-size:11px;color:var(--muted)}
.gbars{display:flex;flex-direction:column;gap:2px}
.gbar-wrap{display:flex;align-items:center;gap:7px;height:14px}
.gbar{height:100%;border-radius:0 4px 4px 0;min-width:2px}
.gbar.s1{background:var(--series-1)} .gbar.s2{background:var(--series-2)}
.gval{font-size:11.5px;color:var(--text-secondary);font-variant-numeric:tabular-nums}
.legend{display:flex;gap:16px;margin:0 0 12px 182px;font-size:12.5px;color:var(--text-secondary)}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.nodata{font-size:11.5px;color:var(--muted)}
.row[tabindex],.gbar-wrap[tabindex]{cursor:default}
.row[tabindex]:focus-visible,.gbar-wrap[tabindex]:focus-visible{outline:2px solid var(--series-1);outline-offset:3px}

svg.pipe{width:100%;height:auto;display:block}
.pipe .st{fill:var(--surface-1);stroke:var(--axis);stroke-width:1}
.pipe .st.w{stroke:var(--series-1);stroke-width:1.5}
.pipe .sl{font-size:13px;fill:var(--text-primary);text-anchor:middle;font-weight:600}
.pipe .sd{font-size:11.5px;fill:var(--text-secondary);text-anchor:middle}
.pipe .ar{stroke:var(--axis);stroke-width:1.5;fill:none}
.pipe .arh{fill:var(--axis)}
.pipe .gate{fill:none;stroke:var(--critical);stroke-width:1.5;stroke-dasharray:0}
.pipe .gl{font-size:12.5px;fill:var(--text-primary);font-weight:600}
.pipe .gd{font-size:11.5px;fill:var(--text-secondary)}
.pipe .bad{fill:var(--critical);font-weight:600}

.cases{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.cases{grid-template-columns:1fr}}
.case{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:14px;overflow:hidden}
.chead{display:flex;gap:9px;align-items:flex-start;margin-bottom:11px}
.pill{font-size:11px;padding:2px 8px;border-radius:999px;white-space:nowrap;font-weight:600}
.pill.bad{background:color-mix(in srgb,var(--critical) 15%,transparent);color:var(--critical)}
.ctitle{font-size:13.5px;line-height:1.45}
.cimgs{display:grid;grid-template-columns:1fr 320px;gap:10px;align-items:start}
.cimgs.one{grid-template-columns:1fr}
@media(max-width:640px){.cimgs{grid-template-columns:1fr}}
.cimgs figure{margin:0}
.cimgs img{width:100%;height:auto;max-height:250px;object-fit:contain;object-position:top left;
  background:var(--grid);border-radius:7px;border:1px solid var(--border);display:block}
.cimgs figcaption{font-size:11px;color:var(--muted);margin-top:4px}
.cmeta{margin:12px 0 0;font-size:12.5px;display:grid;gap:5px}
.cmeta>div{display:grid;grid-template-columns:190px 1fr;gap:10px}
.cmeta dt{color:var(--muted)} .cmeta dd{margin:0;overflow-wrap:anywhere}
.cmeta .bad{color:var(--critical)} .cmeta .good{color:var(--good)}
.empty{color:var(--text-secondary);font-size:13.5px}

table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:14px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--grid)}
td{font-variant-numeric:tabular-nums}
th[scope=row]{font-weight:400;color:var(--text-secondary)}
details.tbl{margin-top:14px}
details.tbl summary{cursor:pointer;font-size:13.5px;color:var(--text-secondary);padding:7px 0}

#tip{position:fixed;z-index:99;pointer-events:none;opacity:0;transition:opacity .1s;
  background:var(--text-primary);color:var(--surface-1);font-size:12px;padding:6px 9px;
  border-radius:6px;max-width:280px}
.caveat{background:var(--surface-1);border:1px solid var(--border);border-left:3px solid var(--series-2);
  border-radius:9px;padding:13px 16px;font-size:13.5px;color:var(--text-secondary);margin:16px 0}
.caveat strong{color:var(--text-primary)}
"""

JS = """
const tip=document.getElementById('tip');
function show(e){const t=e.currentTarget.dataset.tip;if(!t)return;
  tip.innerHTML=t;tip.style.opacity=1;const r=e.currentTarget.getBoundingClientRect();
  tip.style.left=Math.min(window.innerWidth-300,r.left+12)+'px';
  tip.style.top=Math.max(8,r.top-38)+'px';}
function hide(){tip.style.opacity=0;}
document.querySelectorAll('[data-tip]').forEach(el=>{
  el.addEventListener('mouseenter',show);el.addEventListener('mouseleave',hide);
  el.addEventListener('focus',show);el.addEventListener('blur',hide);});
const btn=document.querySelector('button.theme');
btn.addEventListener('click',()=>{
  const dark=document.documentElement.dataset.theme==='dark';
  document.documentElement.dataset.theme=dark?'light':'dark';
  btn.textContent=dark?'Dark mode':'Light mode';});
"""


# =========================================================================
# Assembly
# =========================================================================
def build(gallery_n: int, tag: str) -> str:
    data = {ds: scored_rows(ds, tag) for ds in
            ("infographicvqa", "charxiv", "screenspot", "screenspot_pro")}
    data = {k: v for k, v in data.items() if v}
    if not data:
        raise SystemExit("no results found -- run blindspot.core.runner first")

    ground = data.get("screenspot", []) + data.get("screenspot_pro", [])
    total_n = sum(len(v) for v in data.values())
    n_calls_failed = sum(1 for v in data.values() for r in v if r["failed_call"])
    preflight_n = sum(1 for v in data.values() for r in v if r.get("preflight_downscaled"))

    # ---- headline tiles ----
    tiles = []
    for ds, rows in data.items():
        a = _acc(rows)
        ci = wilson_ci(rows)
        tone = "bad" if (a is not None and a < 0.5) else ""
        note = f"n={len(rows)}"
        if ci:
            note += f" &middot; 95% CI {ci[0]*100:.0f}&ndash;{ci[1]*100:.0f}%"
        tiles.append(stat_tile(ds, f"{a*100:.0f}<span class='pcts'>%</span>" if a is not None else "&mdash;",
                               note, tone))

    # ---- per-dataset accuracy ----
    ds_items = [(ds, _acc(rows), len(rows)) for ds, rows in data.items()]
    chart_ds = bar_chart("Accuracy by dataset", "Higher is better. Haiku 4.5, thinking enabled.",
                         sorted(ds_items, key=lambda t: (t[1] is None, t[1])))

    # ---- grounding: element type x dataset ----
    charts = [chart_ds]
    table_sections = [("Accuracy by dataset", sorted(ds_items, key=lambda t: (t[1] is None, t[1])))]

    if ground:
        by_type: dict[str, dict[str, tuple[float | None, int]]] = {}
        for name, key in (("ScreenSpot", "screenspot"), ("ScreenSpot-Pro", "screenspot_pro")):
            rows = data.get(key, [])
            d = {}
            for t in ("text", "icon"):
                sel = [r for r in rows if r["meta"].get("ui_type") == t]
                d[t] = (_acc(sel), len(sel))
            by_type[name] = d
        charts.append(grouped_bar_chart(
            "UI grounding: text labels vs icons",
            "Icons carry no readable string, so they test perception rather than OCR.",
            ["text", "icon"],
            [(k, v) for k, v in by_type.items()]))
        for name, d in by_type.items():
            table_sections.append((f"{name} by element type",
                                   [(t, *d[t]) for t in ("text", "icon")]))

        # ---- the resolution story ----
        buckets = defaultdict(list)
        for r in ground:
            buckets[bucket_area(r["meta"].get("target_area_frac", 0))].append(r)
        area_items = [(b, _acc(buckets[b]), len(buckets[b])) for b in AREA_ORDER if buckets[b]]
        charts.append(bar_chart(
            "Grounding accuracy by target size (after downscaling)",
            "Target size expressed as it reaches the model, once the API has capped the "
            "long edge at 1568px. This is the axis the pipeline predicts should matter.",
            area_items))
        table_sections.append(("Grounding by target size (post-downscale)", area_items))

    # ---- resolution control: does OUR downscaling explain the failures? ----
    # Sending the original file vs pre-downscaling to Haiku's own ceiling. If the
    # harness were destroying the pixels, these two columns would diverge.
    abl_pairs = []
    for name, key in (("ScreenSpot", "screenspot"), ("ScreenSpot-Pro", "screenspot_pro")):
        nat, abl = data.get(key, []), scored_rows(key, "think2000_edge1568_r0")
        if nat and abl:
            abl_pairs.append((name, _acc(nat), len(nat), _acc(abl), len(abl)))
    if abl_pairs:
        charts.append(grouped_bar_chart(
            "Control: original file vs pre-downscaled to 1568px",
            "The same questions, sent two ways. Identical scores mean the harness is not "
            "the thing losing the resolution -- the API downscales either way, so doing it "
            "first changes nothing. This rules out the pipeline as the cause and leaves the "
            "model's perception at that resolution as the explanation.",
            [n for n, *_ in abl_pairs],
            [("sent as original file", {n: (a, na) for n, a, na, _, _ in abl_pairs}),
             ("pre-downscaled to 1568px", {n: (b, nb) for n, _, _, b, nb in abl_pairs})]))
        table_sections.append(("Resolution control (accuracy)",
            [(f"{n} &mdash; original", a, na) for n, a, na, _, _ in abl_pairs] +
            [(f"{n} &mdash; pre-downscaled", b, nb) for n, _, _, b, nb in abl_pairs]))

    # ---- CharXiv by question type ----
    cx = data.get("charxiv", [])
    if cx:
        by_q = defaultdict(list)
        for r in cx:
            qid = r["meta"].get("qid")
            lab = CHARXIV_SHORT.get(qid, "reasoning (free-form)") if qid else "reasoning (free-form)"
            by_q[lab].append(r)
        q_items = sorted(((k, _acc(v), len(v)) for k, v in by_q.items() if len(v) >= 3),
                         key=lambda t: (t[1] is None, t[1]))
        charts.append(bar_chart(
            "CharXiv: accuracy by question type",
            "Each type isolates a different chart-reading primitive. Free-text types are "
            "graded approximately (see caveat) and are a lower bound.",
            q_items))
        table_sections.append(("CharXiv by question type", q_items))

    # ---- pipeline stats ----
    pro = data.get("screenspot_pro", [])
    src = (3840, 2160)
    med_target = 0.0
    if pro:
        sizes = [tuple(r["meta"]["img_size"]) for r in pro if r["meta"].get("img_size")]
        if sizes:
            src = max(sizes, key=lambda s: s[0] * s[1])
        fr = [r["meta"].get("target_area_frac", 0) for r in pro]
        med_target = math.sqrt(statistics.median(fr) * src[0] * src[1]) if fr else 0
    pipe = pipeline_diagram({"example_src": src, "median_target_px": med_target,
                             "preflight_n": preflight_n})

    galleries = ""
    if ground:
        galleries += ('<h2>Evidence &mdash; UI grounding misses'
                      '<span class="h2sub">Smallest targets first. '
                      '<span style="color:var(--good)">&#9633; green</span> = gold element, '
                      '<span style="color:var(--critical)">&#10010; red</span> = where Haiku clicked.'
                      '</span></h2>' + gallery_grounding(ground, gallery_n))
    for ds, label in (("infographicvqa", "infographic reading"), ("charxiv", "chart reading")):
        if data.get(ds):
            galleries += (f'<h2>Evidence &mdash; {label} failures'
                          f'<span class="h2sub">Model answered confidently and was wrong.</span></h2>'
                          + gallery_span(data[ds], max(2, gallery_n // 2), label))

    fuzzy_note = ""
    if cx:
        n_fuzzy = sum(1 for r in cx if r.get("grading_confidence") == "fuzzy")
        fuzzy_note = (f'<div class="caveat"><strong>Grading caveat.</strong> CharXiv\'s official '
                      f'grader is an LLM judge with per-type rubrics; this harness does not run one. '
                      f'{n_fuzzy} of {len(cx)} CharXiv questions ({n_fuzzy/len(cx)*100:.0f}%) are free-text '
                      f'(titles, axis labels, legend contents, trend) and are graded by normalized string '
                      f'similarity, so a correct answer phrased differently is scored wrong. '
                      f'Treat those types as a <em>lower bound</em>, not a measurement.</div>')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Claude Haiku 4.5 &mdash; perception blind spots</title>
<style>{CSS}</style></head>
<body><div class="viz-root">
<header class="top">
  <div>
    <h1>Claude Haiku 4.5 &mdash; perception blind spots</h1>
    <p class="dek">Where a fast, cheap vision model fails on business-shaped visual tasks:
    infographics, scientific charts, and UI screens. {total_n} questions, thinking enabled,
    single run. No comparison model &mdash; this measures failure, not ranking.</p>
  </div>
  <button class="theme" type="button">Dark mode</button>
</header>

<div class="tiles">{"".join(tiles)}
{stat_tile("failed API calls", str(n_calls_failed), "after pre-flight downscaling fix",
           "good" if n_calls_failed == 0 else "bad")}</div>

<h2>The image pipeline<span class="h2sub">Why resolution is the prime suspect &mdash; and
why the harness is not the culprit (see the control chart below).</span></h2>
{pipe}

<h2>Where it fails<span class="h2sub">Sliced by the axes that could explain why.</span></h2>
{"".join(charts[:1])}
<div class="grid2">{"".join(charts[1:3])}</div>
{"".join(charts[3:])}
{fuzzy_note}

<details class="tbl"><summary>Table view &mdash; every charted value as text</summary>
{table_view(table_sections)}</details>

{galleries}

</div><div id="tip" role="status"></div>
<script>{JS}</script></body></html>"""


def main() -> int:
    p = argparse.ArgumentParser(description="Build the blind-spot HTML report")
    p.add_argument("--gallery", type=int, default=6, help="cases per evidence section")
    p.add_argument("--tag", default="haiku-4-5_think2000_native_r0")
    p.add_argument("--out", default=str(OUTPUTS / "report.html"))
    a = p.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(a.gallery, a.tag), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
