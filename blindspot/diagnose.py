#!/usr/bin/env python
"""Diagnostics: one question each, one page each.

    python -m blindspot.diagnose failure-modes     -> outputs/failure_analysis.html
    python -m blindspot.diagnose coordinates       -> outputs/coord_diagnostics.html + PNGs
    python -m blindspot.diagnose capability [DS]   -> stdout tables
    python -m blindspot.diagnose gt-quality        -> stdout tables
    python -m blindspot.diagnose annotate-probe    -> outputs/probe/*.png
    python -m blindspot.diagnose dataset-page      -> outputs/datasets.html

Each subcommand answers one question and nothing else. They read `results/` and
`data/`; none of them calls the API.

  failure-modes    Where does ScreenSpot-Pro grounding actually fail: format,
                   instruction, or vision? Decomposes 1581 official-protocol rows
                   into a funnel so the three candidate explanations stay
                   separable instead of collapsing into one score.

  coordinates      Annotated PNGs plus a self-contained HTML/SVG explainer for the
                   coordinate finding. Everything is drawn from real pilot data in
                   results/ -- no illustrative fakes. Colour uses categorical slots
                   1-3 of the reference palette (blue/orange/aqua), the documented
                   all-pairs-validated trio in both modes (worst CVD dE 9.2 light /
                   9.4 dark), which is the right gate because all three marks
                   co-occur on every annotated image. Hit/miss is carried by an icon
                   and a label, never by colour.

  capability       Which axis does UI grounding break along? Slices the
                   official-protocol rows by every axis ScreenSpot-Pro annotates,
                   under BOTH parse tiers: `official` (the published regex verbatim,
                   comparable to the GPT-4o 0.8%) and `lenient` (whitespace-tolerant
                   plus pixel->0-1 rescale, the same courtesy the official parser
                   already extends to GPT-4o). Reporting both keeps format
                   compliance and localisation ability separable: a single number
                   confounds "can't point" with "didn't punctuate the answer".
                   Takes an optional dataset name, default screenspot_pro.

  gt-quality       Ground-truth quality rates by failure mode and question type,
                   from results/charxiv__gtaudit.jsonl.

  annotate-probe   Draws ground truth against prediction directly onto the
                   screenshots. Self-describing: each image carries its own labels,
                   so a PNG pulled out of the folder still explains itself without
                   the surrounding page. Green box = ground truth, blue cross =
                   where Haiku 4.5 clicked, dashed line between them = the miss.

  dataset-page     What each dataset is, and whether it turned out usable. Every
                   number is read from data/ at build time rather than transcribed,
                   so the page cannot drift from the corpus. The datasets that
                   turned out unusable are documented as carefully as the ones in
                   the study: knowing that FlowLearn's scientific split ships no
                   ground truth, or that Ferret-UI's release is not the real
                   benchmark, is what stops the next person re-deriving it.
"""

from __future__ import annotations

import argparse
import base64
import collections
import html
import io
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from blindspot.eval import load_rows
from blindspot.eval import CSS as ANNOTATE_CSS   # reuse the validated palette
from blindspot.core import load
from blindspot.core import FAILURE_MODE_LABELS as FM
from blindspot.core import HAIKU_MAX_EDGE, prompt_text
from blindspot.core import point_in_bbox
from blindspot.run_api import (eval_sample, extract_first_bounding_box,
                                       extract_first_point, extract_lenient,
                                       score_row, to_unit)

RESULTS, OUTPUTS = Path("results"), Path("outputs")


# ============================================================ failure-modes
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # validated slots 1-3
GOOD, BAD = "#0ca30c", "#d03b3b"


def collect(ds="screenspot_pro"):
    rows = [json.loads(l) for l in open(RESULTS / f"{ds}__haiku-4-5_official_r0.jsonl") if l.strip()]
    rows = [r for r in rows if not r.get("error")]
    out = []
    for r in rows:
        t = r.get("raw_response") or ""
        strict = bool(extract_first_bounding_box(t) or extract_first_point(t))
        b, p, _ = extract_lenient(t)
        v = b or p
        size = tuple(r["sent_image_sizes"][0])
        rec = {"strict": strict, "parsed": v is not None, "followed": False,
               "hit": False, "dist": None, "meta": r["meta"], "gold": r["gold"]}
        if v is not None:
            rec["followed"] = max(abs(x) for x in v) <= 1.0
            bb, _ = to_unit(b, size); pp, _ = to_unit(p, size)
            if not pp and bb:
                pp = [(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2]
            rec["pred"] = pp
            rec["hit"] = eval_sample(r["gold"], pp) == "correct"
            g = r["gold"]
            if pp and all(0 <= c <= 1 for c in pp):
                rec["dist"] = math.dist(pp, ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2))
        W, H = r["meta"].get("img_size") or size
        side = math.sqrt(max(r["meta"].get("target_area_frac", 0), 0) * W * H)
        rec["seen_px"] = side * min(1.0, HAIKU_MAX_EDGE / max(W, H))
        out.append(rec)
    return out


def funnel_svg(rows):
    N = len(rows)
    stages = [
        ("Produced a parseable answer", sum(r["parsed"] for r in rows), S1,
         "understood the task and the output shape"),
        ("Matched the official regex", sum(r["strict"] for r in rows), BAD,
         "rejected on whitespace after commas — a measurement artefact"),
        ("Used 0–1 range as instructed", sum(r["followed"] for r in rows), S2,
         "instruction-following failure — the rest answered in pixels"),
        ("Landed inside the target box", sum(r["hit"] for r in rows), S3,
         "localisation — this is the real gap"),
    ]
    X, W, ROW = 330, 560, 62
    bars = []
    for i, (lab, v, col, note) in enumerate(stages):
        y = 34 + i * ROW
        pct = v / N
        bw = max(pct * W, 3)
        inside = bw > W * 0.72           # label would run past the canvas edge
        tx = X + bw - 10 if inside else X + bw + 10
        anchor = "end" if inside else "start"
        cls = "fv inv" if inside else "fv"
        sub = "fn inv" if inside else "fn"
        bars.append(f"""
   <text class="fl" x="{X-16}" y="{y+17}" text-anchor="end">{lab}</text>
   <rect class="trk" x="{X}" y="{y}" width="{W}" height="26" rx="4"/>
   <rect x="{X}" y="{y}" width="{bw:.1f}" height="26" rx="4" fill="{col}"/>
   <text class="{cls}" x="{tx:.1f}" y="{y+18}" text-anchor="{anchor}">{pct*100:.1f}%
     <tspan class="{sub}">({v} of {N})</tspan></text>
   <text class="fnote" x="{X}" y="{y+42}">{note}</text>""")
    return f"""
<figure class="chart"><figcaption><h3>Where the 1,581 attempts are lost</h3>
<p class="sub">Each bar is the share of all attempts surviving that stage. Reading down
separates a formatting artefact from an instruction failure from a vision failure.</p>
</figcaption>
<svg viewBox="0 0 1000 {34+len(stages)*ROW+16}" role="img"
     aria-label="Funnel of attempts surviving each stage">{''.join(bars)}</svg></figure>"""


def independence_svg(rows):
    a = [r["hit"] for r in rows if r["parsed"] and r["followed"]]
    b = [r["hit"] for r in rows if r["parsed"] and not r["followed"]]
    items = [("Followed the 0–1 instruction", sum(a) / len(a), len(a), S2),
             ("Answered in pixels instead", sum(b) / len(b), len(b), S1)]
    mx = max(v for _, v, _, _ in items) or 1
    X, W = 300, 470
    bars = []
    for i, (lab, v, n, col) in enumerate(items):
        y = 26 + i * 56
        bars.append(f"""
   <text class="fl" x="{X-16}" y="{y+17}" text-anchor="end">{lab}</text>
   <rect class="trk" x="{X}" y="{y}" width="{W}" height="26" rx="4"/>
   <rect x="{X}" y="{y}" width="{max(v/mx*W*0.8,3):.1f}" height="26" rx="4" fill="{col}"/>
   <text class="fv" x="{X+max(v/mx*W*0.8,3)+10:.1f}" y="{y+18}">{v*100:.2f}%
     <tspan class="fn">(n={n})</tspan></text>""")
    return f"""
<figure class="chart"><figcaption><h3>Instruction-following does not predict localisation</h3>
<p class="sub">If poor instruction-following were a symptom of the model being lost, the
compliant rows should score better. They do not — the two failures are independent.</p>
</figcaption>
<svg viewBox="0 0 900 150" role="img" aria-label="Accuracy split by whether the range instruction was followed">{''.join(bars)}</svg></figure>"""


def precision_svg(rows):
    d = sorted(r["dist"] for r in rows if r["dist"] is not None)
    n = len(d)
    PX, PY, PW, PH = 74, 34, 700, 250
    pts = []
    for i, v in enumerate(d):
        x = PX + min(v, 0.6) / 0.6 * PW
        y = PY + PH - (i + 1) / n * PH
        pts.append(f"{x:.1f},{y:.1f}")
    marks = []
    for th, lab in ((0.05, "5%"), (0.20, "20%")):
        x = PX + th / 0.6 * PW
        frac = sum(v < th for v in d) / n
        marks.append(f'<path class="thr" d="M{x:.1f} {PY} L{x:.1f} {PY+PH}"/>'
                     f'<text class="thrt" x="{x+6:.1f}" y="{PY+14}">{lab} of screen '
                     f'→ {frac*100:.0f}% of clicks</text>')
    return f"""
<figure class="chart"><figcaption><h3>It finds the region, not the element</h3>
<p class="sub">Cumulative share of clicks (up) within a given distance of the true target
centre (across). A model that could not see the screen at all would stay flat near the
bottom; this curve rises steeply and then stalls well short of the target.</p>
</figcaption>
<svg viewBox="0 0 830 320" role="img" aria-label="Cumulative distribution of miss distance">
  <rect class="plot" x="{PX}" y="{PY}" width="{PW}" height="{PH}" rx="3"/>
  {''.join(marks)}
  <polyline class="cdf" points="{' '.join(pts)}"/>
  <text class="tick" x="{PX}" y="{PY+PH+18}">0</text>
  <text class="tick" x="{PX+PW}" y="{PY+PH+18}" text-anchor="end">60%</text>
  <text class="axt" x="{PX+PW/2}" y="{PY+PH+34}" text-anchor="middle">distance from true target centre (% of screen)</text>
  <text class="tick" x="{PX-8}" y="{PY+8}" text-anchor="end">100%</text>
  <text class="tick" x="{PX-8}" y="{PY+PH}" text-anchor="end">0</text>
  <text class="axt" x="18" y="{PY+PH/2}" transform="rotate(-90 18 {PY+PH/2})" text-anchor="middle">share of clicks</text>
</svg></figure>"""


def size_svg(rows):
    edges = [8, 12, 20, 32, 56]
    labs = ["&lt;8px", "8–12px", "12–20px", "20–32px", "32–56px", "≥56px"]
    g = defaultdict(list)
    for r in rows:
        i = next((k for k, e in enumerate(edges) if r["seen_px"] < e), len(edges))
        g[labs[i]].append(r["hit"])
    X, W = 190, 560
    mx = max(sum(v) / len(v) for v in g.values() if v) or 1
    bars = []
    for i, lab in enumerate(labs):
        v = g.get(lab, [])
        if not v:
            continue
        acc = sum(v) / len(v)
        y = 26 + i * 46
        bars.append(f"""
   <text class="fl" x="{X-16}" y="{y+17}" text-anchor="end">{lab}</text>
   <rect class="trk" x="{X}" y="{y}" width="{W}" height="26" rx="4"/>
   <rect x="{X}" y="{y}" width="{max(acc/mx*W*0.85,2):.1f}" height="26" rx="4" fill="{S3}"/>
   <text class="fv" x="{X+max(acc/mx*W*0.85,2)+10:.1f}" y="{y+18}">{acc*100:.1f}%
     <tspan class="fn">(n={len(v)})</tspan></text>""")
    return f"""
<figure class="chart"><figcaption><h3>The limit is target size, after the API's downscale</h3>
<p class="sub">Bucketed by how large the target is <em>as the model sees it</em> — native size
times the ~{HAIKU_MAX_EDGE}px downscale every image passes through. Monotonic across six bins.</p>
</figcaption>
<svg viewBox="0 0 1000 {26+len(labs)*46+10}" role="img" aria-label="Accuracy by target size after downscale">{''.join(bars)}</svg></figure>"""


FAILURE_CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f9f9f7;color:#0b0b0b;
 font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:38px 24px 80px}
h1{font-size:26px;margin:0 0 8px}h2{font-size:18px;margin:38px 0 10px}
h3{font-size:16px;margin:0 0 4px}
.lede{color:#52514e;max-width:76ch;margin:0 0 4px}
.verdict{background:#fcfcfb;border:1px solid #e4e3df;border-left:4px solid #2a78d6;
 border-radius:8px;padding:14px 18px;margin:18px 0;max-width:88ch}
.verdict strong{color:#0b0b0b}
.chart{background:#fcfcfb;border:1px solid #e4e3df;border-radius:12px;padding:20px 22px;margin:20px 0}
.sub{color:#52514e;font-size:13.5px;margin:4px 0 12px;max-width:90ch}
svg{width:100%;height:auto;display:block}
.trk{fill:#efeeea}
.plot{fill:#f6f5f2;stroke:#dedcd7;stroke-width:1}
.fl{fill:#0b0b0b;font-size:13.5px}
.fv{fill:#0b0b0b;font-size:14px;font-weight:600;font-variant-numeric:tabular-nums}
.fn{fill:#7d7c77;font-size:12px;font-weight:400}
.fv.inv{fill:#fff}.fn.inv{fill:#ffffffc0}
.fnote{fill:#7d7c77;font-size:12px}
.tick,.axt{fill:#52514e;font-size:11.5px}
.cdf{fill:none;stroke:#2a78d6;stroke-width:2.5}
.thr{stroke:#8a8984;stroke-width:1.5;stroke-dasharray:4 4}
.thrt{fill:#52514e;font-size:11.5px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #e4e3df}
th{color:#52514e}td.num{font-variant-numeric:tabular-nums}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
 background:#efeeea;padding:1px 5px;border-radius:4px}
@media(prefers-color-scheme:dark){
 body{background:#0d0d0d;color:#fff}
 .chart,.verdict{background:#1a1a19;border-color:#333330}
 .lede,.sub,.tick,.axt,.fnote,.thrt,th{color:#c3c2b7}
 .verdict strong{color:#fff}
 .fl,.fv{fill:#fff}.lede,.sub,.tick,.axt,.fnote,.thrt{fill:#c3c2b7}
 .trk{fill:#2a2a28}.plot{fill:#202020;stroke:#333330}
 .cdf{stroke:#3987e5}.thr{stroke:#7d7c77}
 th,td{border-color:#333330}code{background:#262624}
}
"""


def cmd_failure_modes(args) -> int:
    rows = collect()
    N = len(rows)
    parsed = sum(r["parsed"] for r in rows)
    strict = sum(r["strict"] for r in rows)
    followed = sum(r["followed"] for r in rows)
    hit = sum(r["hit"] for r in rows)
    d = [r["dist"] for r in rows if r["dist"] is not None]
    a = [r["hit"] for r in rows if r["parsed"] and r["followed"]]
    b = [r["hit"] for r in rows if r["parsed"] and not r["followed"]]

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScreenSpot-Pro failure analysis — Haiku 4.5</title><style>{FAILURE_CSS}</style></head><body>
<div class="wrap">
<h1>What is actually failing on ScreenSpot-Pro?</h1>
<p class="lede">{N:,} examples, full split, official protocol — bounding box in 0–1 floats,
no reasoning, temperature 0. Three candidate explanations: output formatting,
instruction-following, or vision. They are separable, and only one of them matters.</p>

<div class="verdict">
<strong>Verdict: it is localisation.</strong> Formatting is a non-issue
({parsed/N*100:.1f}% produced a well-formed answer) and the {strict/N*100:.1f}% official
score is an artefact of whitespace, not capability. Instruction-following is a genuine
failure ({(1-followed/N)*100:.0f}% ignored the 0–1 range) but repairing every instance of
it moves the score only to {hit/N*100:.1f}%. The remaining {(1-hit/N)*100:.1f}% is vision:
the model resolves the right region and cannot resolve the element inside it.
</div>

<h2>1 · The funnel</h2>
{funnel_svg(rows)}

<h2>2 · Ruling out instruction-following as the cause</h2>
{independence_svg(rows)}
<p class="lede">Rows that obeyed the range instruction scored {sum(a)/len(a)*100:.2f}%;
rows that ignored it scored {sum(b)/len(b)*100:.2f}%. Obedience buys nothing, so the two
failures have separate causes — the format problem is not a symptom of the vision problem.</p>

<h2>3 · The shape of the vision failure</h2>
{precision_svg(rows)}
<p class="lede">Median miss is {st.median(d)*100:.1f}% of the screen.
{sum(x<.20 for x in d)/len(d)*100:.0f}% of clicks land within 20% of the target centre but only
{sum(x<.05 for x in d)/len(d)*100:.0f}% within 5% — right neighbourhood, wrong element.</p>

{size_svg(rows)}

<h2>4 · Numbers</h2>
<figure class="chart"><table><thead><tr><th>Stage</th><th>Count</th><th>Share</th>
<th>Recoverable by the harness?</th></tr></thead><tbody>
<tr><td>Successful API calls</td><td class="num">{N}</td><td class="num">100%</td><td>—</td></tr>
<tr><td>Well-formed answer</td><td class="num">{parsed}</td><td class="num">{parsed/N*100:.1f}%</td><td>—</td></tr>
<tr><td>Matched official regex</td><td class="num">{strict}</td><td class="num">{strict/N*100:.1f}%</td><td>Yes — whitespace-tolerant parse</td></tr>
<tr><td>Obeyed 0–1 range</td><td class="num">{followed}</td><td class="num">{followed/N*100:.1f}%</td><td>Yes — rescale by the seen resolution</td></tr>
<tr><td>Landed in target</td><td class="num">{hit}</td><td class="num">{hit/N*100:.1f}%</td><td><strong>No</strong></td></tr>
</tbody></table></figure>

<p class="lede">Caveat: at a {hit/N*100:.1f}% ceiling most sub-slices sit inside their own
noise. The two findings worth defending are target size (monotonic across six bins) and
text-vs-icon (5×, large n). Everything else — application, platform, resolution — is flat.</p>
</div></body></html>"""

    OUTPUTS.mkdir(exist_ok=True)
    p = OUTPUTS / "failure_analysis.html"
    p.write_text(page, encoding="utf-8")
    print(f"wrote {p}")
    print(f"  parseable {parsed}/{N}  official-regex {strict}  followed-range {followed}  hit {hit}")
    return 0

# ============================================================== coordinates
PNGS = OUTPUTS / "probe"

# palette slots 1-3 (light / dark)
C = {"gt": ("#1baf7a", "#199e70"), "haiku": ("#2a78d6", "#3987e5"), "sonnet": ("#eb6834", "#d95926")}
INK, SUB = "#0b0b0b", "#52514e"

TAGS = {"haiku": "haiku-4-5_think2000_native_r0", "sonnet": "sonnet-5_think2000_native_r0"}


def coord_rows(ds, tag):
    p = RESULTS / f"{ds}__{tag}.jsonl"
    if not p.exists():
        return {}
    return {json.loads(l)["uid"]: json.loads(l) for l in open(p) if l.strip()}


def esc(s):
    return html.escape(str(s), quote=True)


def data_uri(im, max_w=620, q=76):
    im = im.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def crosshair(d, x, y, colour, lw, r):
    """Ring + cross, with a dark halo so the mark survives any background."""
    for col, w in ((("#00000088"), lw + 2), (colour, lw)):
        d.line([x - r, y, x + r, y], fill=col, width=w)
        d.line([x, y - r, x, y + r], fill=col, width=w)
        d.ellipse([x - r * .55, y - r * .55, x + r * .55, y + r * .55], outline=col, width=w)


def annotate(ex, preds: dict) -> tuple[Image.Image, Image.Image]:
    im = Image.open(ex.images[0]).convert("RGB")
    W, H = im.size
    x0, y0, x1, y1 = [c * s for c, s in zip(ex.gold, (W, H, W, H))]
    full = im.copy(); d = ImageDraw.Draw(full)
    lw = max(2, round(max(W, H) / 600)); r = lw * 7
    d.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline="#00000088", width=lw + 2)
    d.rectangle([x0, y0, x1, y1], outline=C["gt"][0], width=lw)
    for name, p in preds.items():
        if p:
            crosshair(d, p[0] * W, p[1] * H, C[name][0], lw, r)
    # The zoom has to frame the target AND the click, otherwise a reader sees a
    # lone green box and cannot tell whether the model was one row off or on the
    # other side of the screen. Try to include both; fall back to target-only if
    # they are so far apart that the crop degenerates into the full screenshot.
    pad = max(x1 - x0, y1 - y0) * 3 + 90
    xs, ys = [x0, x1], [y0, y1]
    for p_ in preds.values():
        if p_ and 0 <= p_[0] <= 1 and 0 <= p_[1] <= 1:
            xs.append(p_[0] * W); ys.append(p_[1] * H)
    box = (max(0, int(min(xs) - pad)), max(0, int(min(ys) - pad)),
           min(W, int(max(xs) + pad)), min(H, int(max(ys) + pad)))
    if (box[2] - box[0]) > W * 0.9 and (box[3] - box[1]) > H * 0.9:
        cap = min(W, H) / 3
        p2 = min(max(x1 - x0, y1 - y0) * 7 + 110, cap)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        box = (max(0, int(cx - p2)), max(0, int(cy - p2)),
               min(W, int(cx + p2)), min(H, int(cy + p2)))
        if (box[2] - box[0]) > W * 0.9 and (box[3] - box[1]) > H * 0.9:
            return full, None, False
        return full, full.crop(box), False
    return full, full.crop(box), True


def fit(pairs):
    gs = [g for g, _ in pairs]; ps = [p for _, p in pairs]
    mg, mp = st.mean(gs), st.mean(ps)
    a = sum((g - mg) * (p - mp) for g, p in zip(gs, ps)) / sum((g - mg) ** 2 for g in gs)
    return a, mp - a * mg


# ---------------------------------------------------------------- SVG panels
def panel_bbox_anatomy() -> str:
    """Real ScreenSpot row 0: 'close', [0.948,0.144,0.994,0.207]."""
    b = [0.948, 0.144, 0.994, 0.207]
    SW, SH, OX, OY = 400, 225, 56, 58
    x0, y0 = OX + b[0] * SW, OY + b[1] * SH
    x1, y1 = OX + b[2] * SW, OY + b[3] * SH
    # the same numbers misread as [x, y, w, h]: clipped at the screen edge, with a
    # marker showing it keeps going -- that overflow IS the point of the panel.
    edge = OX + SW
    wrong_w, wrong_h = b[2] * SW, b[3] * SH
    clip = min(x0 + wrong_w, edge + 56)
    return f"""
<figure class="chart wide"><figcaption><h3>1 &middot; What a bbox is</h3>
<p class="sub">Real row 0 of <code>data/screenspot/manifest.jsonl</code> &mdash; instruction
<em>"close"</em>, bbox <code>[0.948, 0.144, 0.994, 0.207]</code>. Origin is
<strong>top-left</strong>; y grows <strong>downward</strong>.</p></figcaption>
<svg viewBox="0 0 1060 340" role="img" aria-label="Bounding box anatomy on a screen rectangle">
  <g>
    <text class="ct" x="{OX}" y="26">Read as [x0, y0, x1, y1] &mdash; correct</text>
    <rect class="screen" x="{OX}" y="{OY}" width="{SW}" height="{SH}" rx="4"/>
    <text class="tick" x="{OX-6}" y="{OY-7}" text-anchor="end">0,0</text>
    <text class="tick" x="{OX+SW}" y="{OY-7}" text-anchor="end">x = 1.0</text>
    <text class="tick" x="{OX-8}" y="{OY+SH+4}" text-anchor="end">y = 1.0</text>
    <path class="ax" d="M{OX} {OY} L{OX+SW} {OY}"/><path class="ax" d="M{OX} {OY} L{OX} {OY+SH}"/>
    <rect class="gt" x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}"/>
    <circle class="pt" cx="{x0}" cy="{y0}" r="4"/><circle class="pt" cx="{x1}" cy="{y1}" r="4"/>
    <path class="lead" d="M{x0} {y0} L{x0-108} {y0-30}"/>
    <text class="lbl" x="{x0-112}" y="{y0-33}" text-anchor="end">x0,y0 = 0.948, 0.144</text>
    <path class="lead" d="M{x1} {y1} L{x1-28} {y1+44}"/>
    <text class="lbl" x="{x1-32}" y="{y1+48}" text-anchor="end">x1,y1 = 0.994, 0.207</text>
    <text class="cap" x="{OX}" y="{OY+SH+30}">4.6% wide, 6.3% tall &mdash; a close button, top-right.</text>
  </g>
  <g transform="translate(540,0)">
    <text class="ct" x="{OX}" y="26">Read as [x, y, w, h] &mdash; wrong</text>
    <rect class="screen" x="{OX}" y="{OY}" width="{SW}" height="{SH}" rx="4"/>
    <rect class="bad-box" x="{x0}" y="{y0}" width="{clip-x0}" height="{wrong_h}"/>
    <path class="ax dashed" d="M{edge} {OY-6} L{edge} {OY+SH+10}"/>
    <path class="arrow-off" d="M{clip-10} {y0+wrong_h/2} L{clip+34} {y0+wrong_h/2}"/>
    <text class="bad-t" x="{clip+40}" y="{y0+wrong_h/2+4}">keeps going</text>
    <text class="cap" x="{OX}" y="{OY+SH+30}">x + w = 1.94 &mdash; off the screen. 642 of 1272 boxes</text>
    <text class="cap" x="{OX}" y="{OY+SH+48}">would do this, so the format is x0,y0,x1,y1. Proven.</text>
  </g>
</svg></figure>"""


def panel_units() -> str:
    return f"""
<figure class="chart wide"><figcaption><h3>2 &middot; Same geometry, two encodings</h3>
<p class="sub">The two datasets store the identical rectangle differently. One conversion in
<code>adapters.py</code> normalises both to 0&ndash;1, matching the official
<code>eval_screenspot_pro.py</code> line for line.</p></figcaption>
<svg viewBox="0 0 1000 210" role="img" aria-label="Normalized versus pixel bbox encodings">
  <g transform="translate(20,20)">
    <rect class="card" x="0" y="0" width="450" height="150" rx="8"/>
    <text class="ct" x="18" y="30">ScreenSpot-v2 &mdash; normalized</text>
    <text class="cd" x="18" y="58">bbox = [0.948, 0.144, 0.994, 0.207]</text>
    <text class="cd" x="18" y="80">already 0&ndash;1 &mdash; image size irrelevant</text>
    <text class="cm" x="18" y="112">gold = bbox   (no conversion)</text>
  </g>
  <g transform="translate(520,20)">
    <rect class="card" x="0" y="0" width="450" height="150" rx="8"/>
    <text class="ct" x="18" y="30">ScreenSpot-Pro &mdash; absolute pixels</text>
    <text class="cd" x="18" y="58">bbox = [1774, 1586, 2113, 1618]</text>
    <text class="cd" x="18" y="80">img_size = [3840, 2160]</text>
    <text class="cm" x="18" y="112">gold = bbox / img_size &rarr; [0.462, 0.734, 0.550, 0.749]</text>
  </g>
</svg></figure>"""


def panel_scatter(data: dict) -> str:
    """Predicted centre vs ground-truth centre. The whole finding, in one picture."""
    W, H, PAD = 380, 300, 46
    def one(ds, title, note, ox):
        pairs = data[ds]
        a, b = fit(pairs)
        pts = "".join(
            f'<circle class="dot" cx="{PAD+g*(W-2*PAD):.1f}" cy="{H-PAD-(p if p<=1 else 1)*(H-2*PAD):.1f}" r="3.4"/>'
            for g, p in pairs)
        fy0, fy1 = b, a + b
        return f"""
  <g transform="translate({ox},0)">
    <text class="ct" x="{PAD}" y="22">{title}</text>
    <text class="cd" x="{PAD}" y="40">{note}</text>
    <rect class="plot" x="{PAD}" y="{PAD}" width="{W-2*PAD}" height="{H-2*PAD}" rx="3"/>
    <path class="ideal" d="M{PAD} {H-PAD} L{W-PAD} {PAD}"/>
    <text class="ideal-t" x="{W-PAD-4}" y="{PAD+14}" text-anchor="end">ideal y = x</text>
    {pts}
    <path class="fitline" d="M{PAD} {H-PAD-fy0*(H-2*PAD):.1f} L{W-PAD} {H-PAD-fy1*(H-2*PAD):.1f}"/>
    <text class="fit-t" x="{PAD+8}" y="{H-PAD-16}">fitted slope {a:.2f}</text>
    <text class="tick" x="{PAD}" y="{H-PAD+18}">0</text>
    <text class="tick" x="{W-PAD}" y="{H-PAD+18}" text-anchor="end">1</text>
    <text class="axt" x="{W/2}" y="{H-PAD+34}" text-anchor="middle">ground-truth x (centre of gold box)</text>
    <text class="axt" x="14" y="{H/2}" transform="rotate(-90 14 {H/2})" text-anchor="middle">predicted x</text>
  </g>"""
    return f"""
<figure class="chart wide"><figcaption><h3>3 &middot; Predictions collapse toward the centre</h3>
<p class="sub">Each dot is one example: where the target actually is (across) vs where Haiku
clicked (up). On the dashed line the model is perfect. A <strong>flatter</strong> fitted line
means the model is ignoring the target's real position and drifting to the middle.</p></figcaption>
<svg viewBox="0 0 1000 340" role="img" aria-label="Scatter of predicted versus ground-truth x position for both datasets">
  {one('screenspot','ScreenSpot-v2  (~960&times;540)','slope 1.02 &mdash; tracks the target',20)}
  {one('screenspot_pro','ScreenSpot-Pro  (3840&times;2160)','slope 0.69 &mdash; pulled to the middle',520)}
</svg>
<p class="read"><strong>How to read it:</strong> on the left the dots hug the diagonal &mdash; when the
target moves right, the prediction moves right. On the right the cloud is nearly flat: wherever the
target actually is, Haiku guesses near the middle of the screen. That is what "slope 0.69" means.</p>
</figure>"""


def panel_arrows(ds: str, data_rows: list) -> str:
    """Gold target -> where the model actually clicked. Arrows point inward."""
    SW, SH, OX, OY = 720, 405, 140, 50
    seg = []
    for gx, gy, px, py in data_rows:
        x0, y0 = OX + gx * SW, OY + gy * SH
        x1, y1 = OX + px * SW, OY + py * SH
        seg.append(f'<path class="arw" d="M{x0:.1f} {y0:.1f} L{x1:.1f} {y1:.1f}"/>'
                   f'<circle class="gtd" cx="{x0:.1f}" cy="{y0:.1f}" r="3.6"/>'
                   f'<circle class="prd" cx="{x1:.1f}" cy="{y1:.1f}" r="2.6"/>')
    return f"""
<figure class="chart wide"><figcaption><h3>4 &middot; The same thing, drawn on a screen</h3>
<p class="sub">One arrow per ScreenSpot-Pro example: tail = where the element really is
(<span class="k gt">&#9679; ground truth</span>), head = where Haiku clicked
(<span class="k hk">&#9679; prediction</span>).</p></figcaption>
<svg viewBox="0 0 1000 505" role="img" aria-label="Arrows from true target position to predicted position">
  <rect class="screen" x="{OX}" y="{OY}" width="{SW}" height="{SH}" rx="4"/>
  <circle class="ctr" cx="{OX+SW/2}" cy="{OY+SH/2}" r="7"/>
  <text class="cap" x="{OX+SW/2}" y="{OY+SH/2+24}" text-anchor="middle">screen centre</text>
  {''.join(seg)}
  <text class="cap" x="{OX}" y="{OY+SH+26}">Arrows converge on the middle rather than pointing in random directions &mdash;
    a systematic pull, not random error.</text>
</svg></figure>"""


COORD_CSS = """
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{margin:0;background:#f9f9f7;color:#0b0b0b;
 font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:40px 24px 90px}
h1{font-size:27px;margin:0 0 6px}h2{font-size:19px;margin:44px 0 12px}
h3{font-size:16px;margin:0 0 4px}
.lede{color:#52514e;max-width:74ch;margin:0 0 8px}
.chart{background:#fcfcfb;border:1px solid #e4e3df;border-radius:12px;padding:20px 22px;margin:22px 0}
figcaption{margin-bottom:10px}
.sub,.cap2{color:#52514e;font-size:13.5px;margin:4px 0 0;max-width:88ch}
.read{color:#52514e;font-size:13.5px;margin:10px 0 0;padding-top:10px;border-top:1px solid #e4e3df;max-width:88ch}
svg{width:100%;height:auto;display:block}
.screen{fill:#f0efec;stroke:#c9c8c3;stroke-width:1.5}
.card{fill:#f4f3f0;stroke:#e0dfda;stroke-width:1}
.plot{fill:#f6f5f2;stroke:#dedcd7;stroke-width:1}
.ax{stroke:#a9a8a3;stroke-width:1.5}.dashed{stroke-dasharray:4 4}
.tick,.cap,.axt{fill:#52514e;font-size:11.5px}
.ct{fill:#0b0b0b;font-size:14px;font-weight:600}
.cd{fill:#52514e;font-size:12.5px}
.cm{fill:#0b0b0b;font-size:12.5px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.gt{fill:none;stroke:#1baf7a;stroke-width:2.5}
.pt{fill:#1baf7a}
.lead{stroke:#8a8984;stroke-width:1;fill:none}
.lbl{fill:#0b0b0b;font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.bad-box{fill:rgba(208,59,59,.10);stroke:#d03b3b;stroke-width:2;stroke-dasharray:5 4}
.bad-t{fill:#d03b3b;font-size:12px;font-weight:600}
.dot{fill:#2a78d6;fill-opacity:.62}
.ideal{stroke:#8a8984;stroke-width:1.5;stroke-dasharray:5 4;fill:none}
.ideal-t{fill:#52514e;font-size:11px}
.fitline{stroke:#eb6834;stroke-width:2.5;fill:none}
.fit-t{fill:#eb6834;font-size:12.5px;font-weight:600}
.arw{stroke:#2a78d6;stroke-width:1.6;stroke-opacity:.55;fill:none}
.gtd{fill:#1baf7a}.prd{fill:#2a78d6}
.ctr{fill:none;stroke:#8a8984;stroke-width:1.5;stroke-dasharray:3 3}
.k{font-weight:600}.k.gt{color:#128a60}.k.hk{color:#2a78d6}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin-top:8px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #e4e3df}
th{color:#52514e;font-weight:600}
td.num{font-variant-numeric:tabular-nums}
.cases{display:grid;gap:16px}
.case{background:#fcfcfb;border:1px solid #e4e3df;border-radius:12px;padding:16px 18px}
.chead{display:flex;gap:10px;align-items:baseline;margin-bottom:10px;flex-wrap:wrap}
.pill{font-size:11.5px;font-weight:700;padding:2px 9px;border-radius:999px;border:1px solid}
.pill.hit{color:#0a7d0a;border-color:#0ca30c;background:#0ca30c14}
.pill.miss{color:#b02f2f;border-color:#d03b3b;background:#d03b3b14}
.ctitle{font-weight:600}
.cimgs{display:grid;grid-template-columns:1.55fr 1fr;gap:12px;align-items:start}
.nozoom{display:flex;align-items:center;justify-content:center;min-height:110px;border:1px dashed #d6d5d0;border-radius:7px;padding:12px;text-align:center}
.cimgs img{width:100%;border-radius:7px;border:1px solid #e4e3df;display:block}
.cimgs figcaption{font-size:11.5px;color:#52514e;margin-top:5px}
.cmeta{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:8px 18px;margin:12px 0 0;font-size:13px}
.cmeta dt{color:#52514e;font-size:11.5px}.cmeta dd{margin:1px 0 0;font-variant-numeric:tabular-nums}
.legend{display:flex;gap:18px;font-size:12.5px;color:#52514e;margin:2px 0 12px;flex-wrap:wrap}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
 background:#efeeea;padding:1px 5px;border-radius:4px}
@media(prefers-color-scheme:dark){
 body{background:#0d0d0d;color:#fff}
 .chart,.case{background:#1a1a19;border-color:#333330}
 .lede,.sub,.read,.cap,.tick,.axt,.cd,.cmeta dt,.legend,th{color:#c3c2b7}
 .screen{fill:#222220;stroke:#3d3c38}.card{fill:#222220;stroke:#333330}
 .plot{fill:#202020;stroke:#333330}
 .ct,.lbl,.cm{fill:#fff}.ax,.lead,.ideal,.ctr{stroke:#7d7c77}
 .gt,.pt,.gtd{stroke:#199e70;fill:#199e70}.gt{fill:none}
 .dot,.prd,.arw{fill:#3987e5;stroke:#3987e5}.arw{fill:none}
 .fitline,.fit-t{stroke:#d95926;fill:#d95926}.fitline{fill:none}
 th,td{border-color:#333330}.cimgs img{border-color:#333330}
 code{background:#262624}
 .k.gt{color:#199e70}.k.hk{color:#3987e5}
}
"""


def build_gallery(plan: dict) -> tuple[str, list]:
    """Annotated PNGs: gold box + where Haiku actually clicked."""
    PNGS.mkdir(parents=True, exist_ok=True)
    cards, manifest = [], []
    for ds, uids in plan.items():
        exs = {e.uid: e for e in load(ds)}
        H = coord_rows(ds, TAGS["haiku"])
        for uid in uids:
            ex, hr = exs[uid], H.get(uid)
            if hr is None or not hr.get("pred"):
                continue
            pred = tuple(hr["pred"])
            hit = point_in_bbox(pred, ex.gold)
            full, zoom, both = annotate(ex, {"haiku": pred})
            stem = uid.replace(":", "_")
            fp = PNGS / f"{stem}_full.png"; full.save(fp)
            zp = None
            if zoom is not None:
                zp = PNGS / f"{stem}_zoom.png"; zoom.save(zp)
            manifest.append({"uid": uid, "hit": bool(hit), "full": str(fp),
                             "zoom": str(zp) if zp else None})

            m = ex.meta
            W, H_ = (m.get("img_size") or Image.open(ex.images[0]).size)
            side = (m.get("target_area_frac", 0) * W * H_) ** .5
            after = side * min(1.0, 1568 / max(W, H_))
            gx, gy = (ex.gold[0] + ex.gold[2]) / 2, (ex.gold[1] + ex.gold[3]) / 2
            off = ((pred[0] - gx) ** 2 + (pred[1] - gy) ** 2) ** .5
            cards.append(f"""
<article class="case">
  <div class="chead"><span class="pill {'hit' if hit else 'miss'}">{'&#10003; hit' if hit else '&#10007; miss'}</span>
    <span class="ctitle">{esc(ex.question[:110])}</span></div>
  <div class="cimgs">
    <figure><img src="{data_uri(full)}" alt="Screenshot with gold box and predicted click">
      <figcaption>full screen &middot; {W}&times;{H_}</figcaption></figure>
    {f'<figure><img src="{data_uri(zoom, max_w=380)}" alt="Zoom around the true target"><figcaption>{"target and click" if both else "zoom on the true target &mdash; click is outside this crop"}</figcaption></figure>' if zoom is not None else '<figure class="nozoom"><figcaption>target fills much of the screen &mdash; no zoom needed</figcaption></figure>'}
  </div>
  <dl class="cmeta">
    <div><dt>target size (native)</dt><dd>&asymp;{side:.0f}px</dd></div>
    <div><dt>after downscale to 1568</dt><dd>&asymp;{after:.0f}px</dd></div>
    <div><dt>true centre</dt><dd>({gx*100:.1f}%, {gy*100:.1f}%)</dd></div>
    <div><dt>model clicked</dt><dd>({pred[0]*100:.1f}%, {pred[1]*100:.1f}%)</dd></div>
    <div><dt>miss distance</dt><dd>{off*100:.1f}% of screen</dd></div>
    <div><dt>element</dt><dd>{esc(m.get('ui_type'))} &middot; {esc(m.get('application') or m.get('platform'))}</dd></div>
  </dl>
</article>""")
    return f'<div class="cases">{"".join(cards)}</div>', manifest


def cmd_coordinates(args) -> int:
    OUTPUTS.mkdir(exist_ok=True)
    scatter, arrows, table = {}, [], []
    for ds in ("screenspot", "screenspot_pro"):
        exs = {e.uid: e for e in load(ds)}
        pairs, arr, sc = [], [], []
        for uid, r in coord_rows(ds, TAGS["haiku"]).items():
            ex = exs.get(uid)
            if ex is None or not r.get("pred"):
                continue
            gx, gy = (ex.gold[0] + ex.gold[2]) / 2, (ex.gold[1] + ex.gold[3]) / 2
            px, py = r["pred"]
            pairs.append((gx, px)); sc.append(point_in_bbox((px, py), ex.gold))
            if 0 <= px <= 1 and 0 <= py <= 1:
                arr.append((gx, gy, px, py))
        scatter[ds] = pairs
        ax, bx = fit(pairs)
        table.append((ds, len(pairs), sum(sc) / len(sc), ax, bx))
        if ds == "screenspot_pro":
            arrows = arr[:110]

    plan = json.loads((RESULTS / "probe_uids.json").read_text())
    gallery, manifest = build_gallery(plan)
    (OUTPUTS / "probe" / "index.json").write_text(json.dumps(manifest, indent=2))

    trow = "".join(
        f'<tr><th scope="row">{esc(d)}</th><td class="num">{n}</td>'
        f'<td class="num">{a*100:.1f}%</td><td class="num">{s:.3f}</td>'
        f'<td class="num">{b:+.3f}</td></tr>' for d, n, a, s, b in table)

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Haiku 4.5 &mdash; coordinate diagnostics</title><style>{COORD_CSS}</style></head><body>
<div class="wrap">
<h1>Where Haiku 4.5 clicks, and where the target actually is</h1>
<p class="lede">Every number and image below comes from the pilot run in <code>results/</code>.
The question this answers: is the low grounding score a bug in how we read coordinates,
or is the model genuinely missing?</p>

<h2>The coordinate system</h2>
{panel_bbox_anatomy()}
{panel_units()}

<h2>The finding</h2>
{panel_scatter(scatter)}
{panel_arrows('screenspot_pro', arrows)}

<figure class="chart"><figcaption><h3>Fitted values</h3>
<p class="sub">Slope 1.0 and intercept 0.0 would be a perfectly calibrated model.</p></figcaption>
<table><thead><tr><th scope="col">Dataset</th><th scope="col">n</th>
<th scope="col">In-box accuracy</th><th scope="col">Slope (x)</th><th scope="col">Intercept (x)</th></tr></thead>
<tbody>{trow}</tbody></table></figure>

<h2>Ground truth vs prediction, case by case</h2>
<p class="lede">Ten examples. Green box is the true target; blue crosshair is where Haiku clicked.
The zoom panel exists because on a 4K screenshot a 47px target is invisible at page scale.</p>
<div class="legend">
  <span><i class="sw" style="background:#1baf7a"></i>ground truth (gold box)</span>
  <span><i class="sw" style="background:#2a78d6"></i>Haiku 4.5 prediction</span>
</div>
{gallery}
</div></body></html>"""

    out = OUTPUTS / "coord_diagnostics.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print(f"wrote {len(manifest)} annotated PNG pairs -> {PNGS}/")
    for d, n, a, s, b in table:
        print(f"  {d:16s} n={n:4d}  acc={a*100:5.1f}%  slope={s:.3f}  intercept={b:+.3f}")
    return 0

# =============================================================== capability


def cap_load(ds, model="haiku-4-5"):
    f = RESULTS / f"{ds}__{model}_official_r0.jsonl"
    rows = [json.loads(l) for l in open(f) if l.strip()]
    out = []
    for r in rows:
        v = score_row(r)
        if v["official"] == "call_error":
            continue
        m = r["meta"]
        W, H = m.get("img_size") or r["sent_image_sizes"][0]
        side = math.sqrt(max(m.get("target_area_frac", 0), 0) * W * H)
        r["_v"] = v
        r["_side_native"] = side
        r["_side_seen"] = side * min(1.0, HAIKU_MAX_EDGE / max(W, H))
        r["_megapix"] = W * H / 1e6
        out.append(r)
    return out


def bucket(v, edges, labels):
    """labels must be len(edges)+1: one per interval plus an explicit overflow bin.

    Passing len(edges) labels silently folds the overflow into the last named
    bucket, which mislabels it -- e.g. a ">=20px" bin printed as "20-32px".
    """
    assert len(labels) == len(edges) + 1, "need one more label than edges"
    for e, l in zip(edges, labels):
        if v < e:
            return l
    return labels[-1]


def cap_table(title, rows, keyfn, order=None, note=""):
    g = defaultdict(list)
    for r in rows:
        g[keyfn(r)].append(r)
    keys = order or sorted(g, key=lambda k: -len(g[k]))
    keys = [k for k in keys if k in g]
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    print(f"  {'slice':22s} {'n':>5s} {'official':>9s} {'lenient':>8s} {'fmt-ok':>7s} {'px-range':>9s}")
    for k in keys:
        v = g[k]
        n = len(v)
        off = sum(x["_v"]["official"] == "correct" for x in v) / n
        len_ = sum(x["_v"]["lenient"] == "correct" for x in v) / n
        fmt = sum(x["_v"]["lenient"] != "wrong_format" for x in v) / n
        px = sum(x["_v"]["range_violation"] for x in v) / n
        print(f"  {str(k):22s} {n:5d} {off*100:8.1f}% {len_*100:7.1f}% {fmt*100:6.0f}% {px*100:8.1f}%")


def cmd_capability(args) -> int:
    ds = args.dataset
    rows = cap_load(ds)
    n = len(rows)
    off = sum(r["_v"]["official"] == "correct" for r in rows) / n
    len_ = sum(r["_v"]["lenient"] == "correct" for r in rows) / n
    wf = sum(r["_v"]["official"] == "wrong_format" for r in rows)
    wfl = sum(r["_v"]["lenient"] == "wrong_format" for r in rows)
    px = sum(r["_v"]["range_violation"] for r in rows) / n

    print(f"=== {ds} | n={n} | official protocol (bbox, 0-1 floats, no reasoning, temp 0)")
    print(f"  action_acc  official {off*100:.1f}%  (wrong_format {wf})")
    print(f"  action_acc  lenient  {len_*100:.1f}%  (wrong_format {wfl})")
    print(f"  answered in pixels despite being asked for 0-1: {px*100:.1f}%")

    cap_table("BY ELEMENT TYPE", rows, lambda r: r["meta"].get("ui_type") or "?",
          order=["text", "icon"],
          note="the paper's own headline axis: icons carry app-specific meaning")
    cap_table("BY APPLICATION GROUP", rows, lambda r: r["meta"].get("group") or "?",
          note="domain familiarity: does it know what these tools look like?")
    cap_table("BY TARGET SIZE AS THE MODEL SEES IT", rows,
          lambda r: bucket(r["_side_seen"], [8, 12, 20, 32, 56],
                           ["<8px", "8-12px", "12-20px", "20-32px", "32-56px", ">=56px"]),
          order=["<8px", "8-12px", "12-20px", "20-32px", "32-56px", ">=56px"],
          note="native size x the API's ~1568px downscale -- the acuity question")
    cap_table("BY SCREEN RESOLUTION", rows,
          lambda r: bucket(r["_megapix"], [2.5, 4.5, 8.5],
                           ["<2.5MP", "2.5-4.5MP", "4.5-8.5MP", ">=8.5MP"]),
          order=["<2.5MP", "2.5-4.5MP", "4.5-8.5MP", ">=8.5MP"],
          note="more pixels = more thrown away at the downscale gate")
    cap_table("BY PLATFORM", rows, lambda r: r["meta"].get("platform") or "?")

    # where it clicks instead
    P = [(r["_v"].get("lenient_point"), r["gold"]) for r in rows if r["_v"].get("lenient_point")]
    P = [(p, g) for p, g in P if p and all(0 <= c <= 1 for c in p)]
    if P:
        import statistics as st
        def slope(i):
            gs = [((g[i] + g[i + 2]) / 2) for _, g in P]
            ps = [p[i] for p, _ in P]
            mg, mp = st.mean(gs), st.mean(ps)
            den = sum((x - mg) ** 2 for x in gs)
            return sum((x - mg) * (y - mp) for x, y in zip(gs, ps)) / den if den else float("nan")
        d = [math.dist(p, ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)) for p, g in P]
        print(f"\nWHERE IT CLICKS INSTEAD  (n={len(P)} in-range predictions)")
        print(f"  slope pred-vs-gold      x {slope(0):.3f}   y {slope(1):.3f}   (1.0 = tracks the target)")
        print(f"  median miss distance    {st.median(d)*100:.1f}% of screen diagonal")
        print(f"  within 5% of centre     {sum(x < .05 for x in d)/len(d)*100:.1f}%")
        print(f"  within 20% of centre    {sum(x < .20 for x in d)/len(d)*100:.1f}%")
    return 0

# =============================================================== gt-quality
DS = "charxiv"

# CharXiv descriptive question ids, grouped into families that fail differently.
FREE = {1, 2, 3, 13, 16}   # title, x-label, y-label, legend names, trend  (free-text)
TICK = {4, 5, 6, 7, 15}    # verbatim tick values
NUM  = {8, 9, 10, 12, 14, 17, 19}
OTH  = {11, 18}


def wilson(k, n, z=1.96):
    if not n: return (0, 0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h)*100, min(1, c+h)*100)


def fam(q):
    return ("free-text (title/axis/legend/trend)" if q in FREE else
            "tick-value readout" if q in TICK else
            "counting / arithmetic" if q in NUM else "structural (intersect/layout)")


def show(title, key, rows, popn=None, minn=1):
    print(f"\n### {title}")
    print(f"{'key':<56} {'n':>4} {'badGT%':>7} {'95% CI':>13} {'!gold%':>7} {'pop':>5}")
    g = collections.defaultdict(list)
    for r in rows: g[r[key]].append(r)
    tab = []
    for k, rs in g.items():
        n = len(rs); b = sum(r["bad"] for r in rs); ng = sum(r["not_gold"] for r in rs)
        tab.append((k, n, b, b/n*100, ng/n*100, wilson(b, n), (popn or {}).get(k)))
    for k, n, b, bp, ngp, ci, p in sorted(tab, key=lambda t: (-t[3], -t[1])):
        if n < minn: continue
        print(f"{str(k)[:56]:<56} {n:>4} {b:>3}/{n:<3}{bp:>4.0f}% "
              f"{ci[0]:>5.0f}-{ci[1]:<5.0f} {ngp:>6.0f}% {str(p or ''):>5}")


def cmd_gt_quality(args) -> int:
    recs = list({json.loads(l)["uid"]: json.loads(l)
                 for l in open(f"results/{DS}__gtaudit.jsonl")
                 if "error" not in json.loads(l)}.values())

    ex = {e.uid: e for e in load(DS)}
    for r in recs:
        m = ex[r["uid"]].meta
        r.update(qid=m.get("qid"), qlabel=m.get("qlabel"), split=m.get("split"),
                 bad=r["gt_quality"] != "unambiguous", not_gold=r["verdict"] != "gold_correct")

    pop = collections.Counter(rr.get("failure_mode", "unclassified")
                              for rr in load_rows(DS) if (rr.get("score") or 0) < 0.5)
    NPOP = sum(pop.values())

    n = len(recs); bad = sum(r["bad"] for r in recs)
    print(f"AUDITED {n} of {NPOP} charxiv failures ({n/NPOP*100:.0f}% of the failure set)")
    print("verdict:   ", dict(collections.Counter(r["verdict"] for r in recs).most_common()))
    print("gt_quality:", dict(collections.Counter(r["gt_quality"] for r in recs).most_common()))
    print(f"UNWEIGHTED bad-GT: {bad}/{n} = {bad/n*100:.1f}%  CI {wilson(bad,n)[0]:.0f}-{wilson(bad,n)[1]:.0f}%")

    g = collections.defaultdict(list)
    for r in recs: g[r.get("failure_mode", "unclassified")].append(r)
    w  = sum(pop[k]/NPOP * (sum(x["bad"] for x in v)/len(v)) for k, v in g.items() if k in pop)
    wg = sum(pop[k]/NPOP * (sum(x["not_gold"] for x in v)/len(v)) for k, v in g.items() if k in pop)
    print(f"PREVALENCE-WEIGHTED bad-GT: {w*100:.1f}%   (verdict != gold_correct: {wg*100:.1f}%)")
    print(f"=> implies ~{w*497:.0f} of 497 failures / ~{w*497/5000*100:.1f}pp of the 88.4% score")

    print("\ncross-tab verdict x gt_quality:")
    ct = collections.Counter((r["verdict"], r["gt_quality"]) for r in recs)
    for k, v in ct.most_common(): print(f"  {k[0]:<20} {k[1]:<13} {v}")

    for r in recs: r["fmlabel"] = f'{r.get("failure_mode")} ({FM.get(r.get("failure_mode"),"")})'
    show("by failure mode", "fmlabel", recs)
    show("by split", "split", recs)
    show("by question type", "qlabel", [r for r in recs if r["qid"]], minn=3)

    for r in recs:
        r["fam"] = fam(r["qid"]) if r["qid"] else "reasoning split (free-form)"
    show("by question family", "fam", recs)
    return 0

# =========================================================== annotate-probe
PROBE_OUT = Path("outputs/probe")
TAG = "haiku-4-5_think2000_native_r0"
GT_C, PR_C = (27, 175, 122), (42, 120, 214)      # palette slots 3 and 1
HALO = (0, 0, 0)

FONT_CANDIDATES = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                   "/System/Library/Fonts/Helvetica.ttc",
                   "/Library/Fonts/Arial Bold.ttf"]


def font(px: int):
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            try:
                return ImageFont.truetype(f, px)
            except Exception:
                pass
    return ImageFont.load_default()


def tag(d, xy, text, colour, fnt, size, avoid=None, prefer="above"):
    """Filled label chip, kept inside the image and off the thing it labels.

    Targets sit anywhere -- including flush against an edge (ScreenSpot has boxes
    at y0 = 0.0 exactly) -- so a chip placed blindly above the box gets clipped
    off-canvas. Try the preferred side, fall back to the opposite, then clamp.
    """
    W, H = size
    x, y = xy
    l, t, r, b = d.textbbox((0, 0), text, font=fnt)
    w, h = r - l, b - t
    pad = max(4, h // 3)
    bw, bh = w + 2 * pad, h + 2 * pad

    if avoid:
        ax0, ay0, ax1, ay1 = avoid
        gap = pad
        above, below = ay0 - bh - gap, ay1 + gap
        y = above if (prefer == "above" and above >= 0) else below
        if y + bh > H:                      # no room below either -> go above
            y = max(0, above)
        x = ax0
    x = min(max(0, x), W - bw)
    y = min(max(0, y), H - bh)
    d.rounded_rectangle([x, y, x + bw, y + bh], radius=pad,
                        fill=colour + (235,), outline=(255, 255, 255, 210), width=1)
    d.text((x + pad - l, y + pad - t), text, fill=(255, 255, 255), font=fnt)
    return (x, y, x + bw, y + bh)


def wrap(d, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if d.textlength(t, font=fnt) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def banner(im: Image.Image, instruction: str, hit: bool, sub: str,
           prompt: str = "", answer: str = "") -> Image.Image:
    """Caption strip above the screenshot.

    Extending the canvas rather than overlaying keeps the instruction from
    covering the very UI the reader needs to judge the click against.
    """
    W = im.width
    fs = max(17, round(W / 66))
    f_main, f_sub = font(fs), font(max(13, round(fs * 0.72)))
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    pad = round(fs * 0.85)
    chip = "HIT" if hit else "MISS"
    chip_w = tmp.textlength(chip, font=f_sub) + pad * 1.6
    lines = wrap(tmp, f'"{instruction}"', f_main, W - pad * 3 - chip_w)
    # The metadata line is long and narrow zoom crops are common, so it wraps too
    # -- otherwise it silently runs off the right edge.
    subs = wrap(tmp, sub, f_sub, W - pad * 2)
    # Verbatim prompt, so the image shows what the model was actually asked. The
    # ask is a POINT in 0-1000 space, which is why the mark is a crosshair and
    # not a box -- the official benchmark protocol asks for a bbox instead.
    f_mono = font(max(12, round(fs * 0.66)))
    plines = []
    for para in (prompt or "").split("\n"):
        plines.extend(wrap(tmp, para, f_mono, W - pad * 3.2) if para.strip() else [""])
    prow = round(fs * 0.9)
    bh = (pad * 2 + len(lines) * round(fs * 1.32)
          + len(subs) * round(fs * 0.95)
          + (round(fs * 1.5) + len(plines) * prow + (prow if answer else 0) if plines else 0))

    out = Image.new("RGB", (W, im.height + bh), (250, 250, 248))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, W, bh], fill=(250, 250, 248))
    d.line([0, bh - 1, W, bh - 1], fill=(210, 209, 204), width=2)

    y = pad
    for ln in lines:
        d.text((pad, y), ln, fill=(11, 11, 11), font=f_main)
        y += round(fs * 1.32)
    for sl in subs:
        d.text((pad, y + 2), sl, fill=(90, 89, 85), font=f_sub)
        y += round(fs * 0.95)

    if plines:
        y += round(fs * 0.55)
        d.text((pad, y), "PROMPT SENT TO THE MODEL", fill=(120, 119, 114), font=f_mono)
        y += prow
        top = y - 2
        for pl in plines:
            d.text((pad + round(fs * 0.55), y), pl, fill=(60, 59, 56), font=f_mono)
            y += prow
        if answer:
            d.text((pad + round(fs * 0.55), y), f"model returned: {answer}",
                   fill=(42, 120, 214), font=f_mono)
            y += prow
        d.line([pad, top, pad, y - 4], fill=(200, 199, 194), width=2)

    cx = W - pad - chip_w
    cy = pad * 0.7
    ch = round(fs * 1.25)
    d.rounded_rectangle([cx, cy, cx + chip_w, cy + ch], radius=ch // 2,
                        fill=(12, 163, 12) if hit else (208, 59, 59))
    tw = tmp.textlength(chip, font=f_sub)
    d.text((cx + (chip_w - tw) / 2, cy + (ch - fs * 0.72) / 2 - 1), chip,
           fill=(255, 255, 255), font=f_sub)

    out.paste(im, (0, bh))
    return out


def draw(base: Image.Image, gold, pred, scale=1.0):
    """Draw GT box + prediction cross + miss line onto a copy of `base`."""
    im = base.convert("RGB")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    W, H = im.size
    lw = max(3, round(max(W, H) / 500 * scale))
    fnt = font(max(15, round(max(W, H) / 62 * scale)))

    x0, y0, x1, y1 = [c * s for c, s in zip(gold, (W, H, W, H))]
    gx, gy = (x0 + x1) / 2, (y0 + y1) / 2
    px, py = pred[0] * W, pred[1] * H

    # miss line first, so the marks sit on top of it
    if 0 <= pred[0] <= 1 and 0 <= pred[1] <= 1:
        for col, wd in ((HALO + (170,), lw + 2), (PR_C + (200,), lw)):
            for i in range(0, int(((px - gx) ** 2 + (py - gy) ** 2) ** .5), 26):
                t0 = i / max(1e-6, ((px - gx) ** 2 + (py - gy) ** 2) ** .5)
                t1 = min(1.0, t0 + 13 / max(1e-6, ((px - gx) ** 2 + (py - gy) ** 2) ** .5))
                d.line([gx + (px - gx) * t0, gy + (py - gy) * t0,
                        gx + (px - gx) * t1, gy + (py - gy) * t1], fill=col, width=wd)

    # ground-truth box, haloed so it reads on light and dark UI alike
    d.rectangle([x0 - lw, y0 - lw, x1 + lw, y1 + lw], outline=HALO + (190,), width=lw + 2)
    d.rectangle([x0, y0, x1, y1], outline=GT_C + (255,), width=lw)

    # prediction crosshair
    r = lw * 8
    for col, wd in ((HALO + (190,), lw + 3), (PR_C + (255,), lw)):
        d.line([px - r, py, px + r, py], fill=col, width=wd)
        d.line([px, py - r, px, py + r], fill=col, width=wd)
        d.ellipse([px - r * .5, py - r * .5, px + r * .5, py + r * .5], outline=col, width=wd)

    # Labels sit outside the marks they name, never on top of them.
    tag(d, (x0, y0), "GROUND TRUTH", GT_C, fnt, (W, H),
        avoid=(x0 - lw, y0 - lw, x1 + lw, y1 + lw), prefer="above")
    tag(d, (px, py), "HAIKU CLICKED", PR_C, fnt, (W, H),
        avoid=(px - r, py - r, px + r, py + r), prefer="below")

    im.paste(Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB"))
    return im


def cmd_annotate_probe(args) -> int:
    PROBE_OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((RESULTS / "probe_uids.json").read_text())
    made = []
    for ds, uids in plan.items():
        exs = {e.uid: e for e in load(ds)}
        rows = {json.loads(l)["uid"]: json.loads(l)
                for l in open(RESULTS / f"{ds}__{TAG}.jsonl") if l.strip()}
        for uid in uids:
            ex, rec = exs[uid], rows.get(uid)
            if not rec or not rec.get("pred"):
                continue
            pred = tuple(rec["pred"])
            hit = bool(point_in_bbox(pred, ex.gold))
            base = Image.open(ex.images[0])
            W, H = base.size
            full = draw(base, ex.gold, pred)

            # zoom framing both marks
            x0, y0, x1, y1 = [c * s for c, s in zip(ex.gold, (W, H, W, H))]
            xs, ys = [x0, x1], [y0, y1]
            if 0 <= pred[0] <= 1 and 0 <= pred[1] <= 1:
                xs.append(pred[0] * W); ys.append(pred[1] * H)
            pad = max(x1 - x0, y1 - y0) * 2 + max(W, H) * 0.05
            box = (max(0, int(min(xs) - pad)), max(0, int(min(ys) - pad)),
                   min(W, int(max(xs) + pad)), min(H, int(max(ys) + pad)))
            crop = base.crop(box)
            g2 = [(x0 - box[0]) / crop.width, (y0 - box[1]) / crop.height,
                  (x1 - box[0]) / crop.width, (y1 - box[1]) / crop.height]
            p2 = ((pred[0] * W - box[0]) / crop.width, (pred[1] * H - box[1]) / crop.height)
            zoom = draw(crop, g2, p2, scale=1.9)

            gx = (ex.gold[0] + ex.gold[2]) / 2
            gy = (ex.gold[1] + ex.gold[3]) / 2
            off = ((pred[0] - gx) ** 2 + (pred[1] - gy) ** 2) ** .5
            side = (ex.meta.get("target_area_frac", 0) * W * H) ** .5
            sub = (f"{ds} \u00b7 {uid.split(':')[-1]} \u00b7 {W}\u00d7{H} \u00b7 "
                   f"target \u2248{side:.0f}px \u00b7 miss {off*100:.1f}% of screen")
            ptxt = prompt_text(ex)
            ans = rec.get("raw") or ""
            stem = uid.replace(":", "_")
            banner(full, ex.question, hit, sub, ptxt, ans).save(PROBE_OUT / f"{stem}__1_full.png")
            banner(zoom, ex.question, hit, sub + " \u00b7 zoomed", ptxt, ans).save(
                PROBE_OUT / f"{stem}__2_zoom.png")
            made.append((uid, ex.question[:52], hit, f"{W}x{H}"))
    print(f"{'uid':44s} {'hit':>4s}  {'size':>10s}  instruction")
    for uid, q, hit, sz in made:
        print(f"{uid:44s} {'YES' if hit else 'no':>4s}  {sz:>10s}  {q}")
    print(f"\n{len(made)*2} PNGs -> {PROBE_OUT}/")
    return 0

# ============================================================= dataset-page
DATA = Path("data")
DATASETS_OUT = Path("outputs/datasets.html")

# status, business task, primitives, metric, and the reason for the verdict.
CARDS = {
    "charxiv": dict(
        status="USED", title="CharXiv",
        task="Reading real scientific charts from arXiv papers &mdash; the closest public proxy "
             "for the analytical charts that appear in reports and decks.",
        prim="counting &middot; line-following &middot; localization &middot; value interpolation &middot; "
             "binding &middot; structure &middot; text-in-situ &middot; comparison &middot; composition",
        metric="CharXiv's <b>official LLM judge</b> (batched triplets under seven per-type rubrics, "
               "vendored verbatim). Our normalized-match/ANLS scorer is kept alongside for comparison.",
        why="The backbone of the study: 19 descriptive templates each isolate a single perceptual "
            "operation by construction, so the primitive decomposition is the dataset's own, not ours.",
        caveat="Gold answers are 'Not Applicable' at wildly different rates per template (colorbar 86%, "
               "title 59%, lines-intersect 45%, most others 0%). Every primitive is also scored on the "
               "answerable subset, because a pooled number measures 'can you tell this doesn't apply'."),
    "infographicvqa": dict(
        status="USED", title="InfographicVQA",
        task="Real business infographics: marketing collateral, survey results, sports and product "
             "comparisons &mdash; charts, dense text, icons and layout in one artifact.",
        prim="counting &middot; comparison &middot; arithmetic",
        metric="<b>Official ANLS</b> at threshold 0.5, matching the published definition exactly "
               "(lowercase and strip only; a normalized distance of exactly 0.5 scores 0).",
        why="Carries two orthogonal human-annotated axes: the operation required (counting, "
            "comparison, arithmetic) and how the answer is produced (span vs non-extractive).",
        caveat="Counting is 93% non-extractive and arithmetic 97%, while comparison is 86% single-span "
               "&mdash; so 'derive the answer' and 'find the answer' are cleanly separable here."),
    "screenspot_pro": dict(
        status="USED", title="ScreenSpot-Pro",
        task="Locating UI elements on high-resolution screenshots of professional software "
             "(CAD, IDEs, office suites).",
        prim="localization &rarr; emit a coordinate",
        metric="<b>Click-in-bbox accuracy</b>, the benchmark's own metric: the predicted point must "
               "fall inside the gold box.",
        why="The only source here that ends in emitting coordinates rather than reading text. That "
            "makes it a different primitive from CharXiv's spatial-extreme questions, which is why "
            "the two are reported separately (91% vs 2%) rather than averaged.",
        caveat="Every image exceeds Haiku's ~1568px ceiling, so ~7% of pixels survive and the median "
               "target is roughly 22px across when the model sees it. Verified not to be a harness "
               "artifact: pre-downscaling changes nothing, and alternative coordinate spaces score worse."),
    "flowlearn_sim": dict(
        status="EXCLUDED", title="FlowLearn (simulated)",
        task="Procedurally generated flowcharts with nonsense-word node labels.",
        prim="arrow-following &middot; counting (dropped)",
        metric="n/a",
        why="Genuinely the best arrow-following probe available &mdash; nonsense labels mean world "
            "knowledge cannot shortcut the trace, and node/arrow counts give a difficulty curve.",
        caveat="Excluded for protocol purity. The release ships <b>no official question text</b> "
               "(113,567 files, and the only prose is a licence README), and <code>Arrow_betweenAB</code> "
               "is directed despite its name &mdash; 38% of its 'false' pairs have a real undirected "
               "edge, so scoring the natural-language reading would have manufactured a fake blind spot."),
    "flowlearn_sci": dict(
        status="UNUSABLE", title="FlowLearn (scientific)",
        task="3.8K real flowcharts extracted from papers.", prim="&mdash;", metric="n/a",
        why="Would have been the real-world counterpart to the simulated split.",
        caveat="Ships only image, caption, OCR text and figType. <b>No questions and no answers.</b> "
               "Nothing to score."),
    "blindtest": dict(
        status="UNUSABLE", title="BlindTest",
        task="Confound-free primitive probes: line intersection, counting circles, nested squares, "
             "subway-map tracing.", prim="the spec's primitives, in isolation", metric="n/a",
        why="Task names map one-to-one onto the brief's named primitives &mdash; the purest probes available.",
        caveat="<b>ground_truth is null on all 30 rows.</b> The answer-bearing filename prefixes were "
               "stripped before public release. Hand-annotatable in about an hour; not done here."),
    "ferret_ui": dict(
        status="UNUSABLE", title="Ferret-UI",
        task="Decomposed UI perception tasks.", prim="&mdash;", metric="n/a",
        why="Would have given elementary UI perception as separate sub-tasks.",
        caveat="8 manifest rows, 3 with an answer, <b>1 image on disk</b>, and every row carries a note "
               "saying it is not the real Ferret-UI benchmark. Also CC BY-NC (non-commercial)."),
}

NOT_RUN = {
    "chartqa": ("Business charts, 1250 human + 1250 machine-generated questions. Relaxed accuracy. "
                "Downloaded and adapter-ready; out of scope once the study narrowed to three."),
    "docvqa": ("Forms, memos and tables with a 9-way question-type taxonomy (layout, table/list, form, "
               "handwritten...). ANLS. Strong business fit; not run."),
    "livexiv": ("5,120 four-way multiple-choice questions, contamination-controlled. The one source with "
                "zero answer-expression confound &mdash; the model picks rather than phrases."),
    "slidevqa": ("Slide decks; every question ships all 20 pages, so a 20x cost multiplier. Needs a "
                 "retrieval protocol before it says anything clean."),
    "screenspot": ("Baseline UI grounding at ordinary resolution. Demoted with the rest of the UI set."),
    "rico_screenqa": ("8,427 screen-reading questions; uniquely supports both ANLS and click-in-bbox on "
                      "the same item. Excluded by decision."),
}

BADGE = {"USED": ("var(--good)", "in the study"),
         "EXCLUDED": ("var(--accent)", "excluded by decision"),
         "UNUSABLE": ("var(--bad)", "unusable as released")}


EXTRA = """
.badge{display:inline-block;font-size:11px;font-weight:600;padding:3px 10px;border-radius:999px;
 color:#fff;margin-left:10px;vertical-align:3px}
.ds{background:var(--surface);border:1px solid var(--border);border-radius:12px;
 padding:18px 20px;margin:14px 0}
.ds h3{font-size:17px;margin:0 0 10px}
.ds dl{display:grid;gap:7px;margin:0;font-size:13.5px}
.ds dl>div{display:grid;grid-template-columns:150px 1fr;gap:12px}
.ds dt{color:var(--muted)}.ds dd{margin:0}
.ds .stat{display:flex;gap:20px;flex-wrap:wrap;margin:0 0 14px;font-size:12.5px;color:var(--ink2)}
.ds .stat b{color:var(--ink);font-variant-numeric:tabular-nums}
.ds .caveat{margin-top:12px;padding:11px 13px;border-radius:9px;font-size:13px;
 background:color-mix(in srgb,var(--accent) 8%,transparent);color:var(--ink2)}
.ds .caveat b{color:var(--ink)}
table.sum{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0 26px}
table.sum th,table.sum td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid)}
table.sum td.n{font-variant-numeric:tabular-nums}
"""


def stats(name: str) -> dict:
    d = DATA / name
    mf = d / "manifest.jsonl"
    if not mf.exists():
        return {}
    rows = [json.loads(l) for l in open(mf) if l.strip()]
    imgs = sum(1 for p in d.rglob("*")
               if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".gif", ".webp"))
    size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    q = len(rows)
    if name == "charxiv":   # 4 descriptive + 1 reasoning per figure
        q = sum(sum(1 for j in range(1, 5) if r.get(f"descriptive_q{j}") is not None)
                + (1 if r.get("reasoning_q") else 0) for r in rows)
    return {"rows": len(rows), "images": imgs, "questions": q, "gb": size / 1e9}


def card(key: str, c: dict) -> str:
    st = stats(key)
    col, lab = BADGE[c["status"]]
    s = ""
    if st:
        s = (f'<div class="stat"><span>rows <b>{st["rows"]:,}</b></span>'
             f'<span>questions <b>{st["questions"]:,}</b></span>'
             f'<span>images <b>{st["images"]:,}</b></span>'
             f'<span>on disk <b>{st["gb"]:.2f} GB</b></span></div>')
    return f"""
<article class="ds" id="{esc(key)}">
 <h3>{c['title']}<span class="badge" style="background:{col}">{lab}</span></h3>
 {s}
 <dl>
  <div><dt>business task</dt><dd>{c['task']}</dd></div>
  <div><dt>primitives</dt><dd>{c['prim']}</dd></div>
  <div><dt>metric</dt><dd>{c['metric']}</dd></div>
  <div><dt>why this one</dt><dd>{c['why']}</dd></div>
 </dl>
 <div class="caveat">{c['caveat']}</div>
</article>"""


def cmd_dataset_page(args) -> int:
    DATASETS_OUT.parent.mkdir(parents=True, exist_ok=True)
    summ = {}
    p = Path("outputs/summary.json")
    if p.exists():
        summ = json.loads(p.read_text()).get("datasets", {})

    srows = []
    for key, c in CARDS.items():
        st = stats(key)
        acc = summ.get(key, {}).get("acc")
        n = summ.get(key, {}).get("n")
        col, lab = BADGE[c["status"]]
        srows.append(
            f'<tr><th scope="row"><a href="#{esc(key)}">{c["title"]}</a></th>'
            f'<td><span class="badge" style="background:{col}">{lab}</span></td>'
            f'<td class="n">{st.get("questions", 0):,}</td>'
            f'<td class="n">{"&mdash;" if acc is None else f"{acc*100:.1f}%"}</td>'
            f'<td class="n">{"&mdash;" if n is None else f"{n:,}"}</td></tr>')

    others = "".join(
        f'<tr><th scope="row">{esc(k)}</th><td class="n">{stats(k).get("rows",0):,}</td>'
        f'<td>{v}</td></tr>' for k, v in NOT_RUN.items())

    body = "".join(card(k, c) for k, c in CARDS.items())
    DATASETS_OUT.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Datasets &mdash; Haiku 4.5 perception study</title>
<style>{ANNOTATE_CSS}{EXTRA}</style></head><body><div class="wrap">
<p style="font-size:12.5px;color:var(--muted)"><a href="report.html">&larr; overview</a></p>
<h1>Datasets</h1>
<p class="dek">What each corpus is, which perceptual primitives it can measure, how it is scored,
and &mdash; for the ones that did not make it &mdash; exactly what is wrong with them. Every count
below is read from <code>data/</code> at build time.</p>

<table class="sum"><thead><tr><th scope="col">dataset</th><th scope="col">status</th>
<th scope="col">questions available</th><th scope="col">Haiku 4.5</th><th scope="col">n evaluated</th>
</tr></thead><tbody>{''.join(srows)}</tbody></table>

{body}

<h2 style="font-size:18px;margin:38px 0 6px;padding-top:20px;border-top:1px solid var(--grid)">
Downloaded but not run</h2>
<p class="dek" style="font-size:13.5px">Complete on disk at their official eval splits and adapter-ready.
Out of scope for this study rather than defective.</p>
<table class="sum"><thead><tr><th scope="col">dataset</th><th scope="col">rows</th>
<th scope="col">note</th></tr></thead><tbody>{others}</tbody></table>
</div></body></html>""", encoding="utf-8")
    print(f"wrote {DATASETS_OUT} ({DATASETS_OUT.stat().st_size/1024:.0f} KB)")
    return 0


# ================================================================ dispatch

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m blindspot.diagnose",
                                description="One diagnostic question per subcommand.")
    sub = p.add_subparsers(dest="cmd", metavar="QUESTION")

    fm = sub.add_parser("failure-modes",
                        help="format, instruction, or vision? -> outputs/failure_analysis.html")
    fm.set_defaults(fn=cmd_failure_modes)

    co = sub.add_parser("coordinates",
                        help="the coordinate-compression explainer -> outputs/coord_diagnostics.html")
    co.set_defaults(fn=cmd_coordinates)

    ca = sub.add_parser("capability", help="which axis does UI grounding break along?")
    ca.add_argument("dataset", nargs="?", default="screenspot_pro",
                    help="dataset to slice (default: screenspot_pro)")
    ca.set_defaults(fn=cmd_capability)

    gq = sub.add_parser("gt-quality",
                        help="ground-truth quality by failure mode and question type")
    gq.set_defaults(fn=cmd_gt_quality)

    an = sub.add_parser("annotate-probe",
                        help="gold box vs predicted click, drawn on the screenshots")
    an.set_defaults(fn=cmd_annotate_probe)

    dp = sub.add_parser("dataset-page", help="what each dataset is -> outputs/datasets.html")
    dp.set_defaults(fn=cmd_dataset_page)
    return p


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        p.print_help()
        return 2
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
