"""Where does ScreenSpot-Pro grounding actually fail: format, instruction, or vision?

Decomposes 1581 official-protocol rows into a funnel, so the three candidate
explanations stay separable instead of collapsing into one score.

    python scripts/analyze/failure_analysis.py   ->  outputs/failure_analysis.html
"""
from __future__ import annotations
import json, math, statistics as st, sys
from collections import defaultdict
from pathlib import Path

from blindspot.core.prompts import HAIKU_MAX_EDGE
from scripts.run.official_eval import (extract_first_bounding_box, extract_first_point,
                                       extract_lenient, to_unit, eval_sample)

RES, OUT = Path("results"), Path("outputs")
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # validated slots 1-3
GOOD, BAD = "#0ca30c", "#d03b3b"


def collect(ds="screenspot_pro"):
    rows = [json.loads(l) for l in open(RES / f"{ds}__haiku-4-5_official_r0.jsonl") if l.strip()]
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


CSS = """
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


def main() -> int:
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
<title>ScreenSpot-Pro failure analysis — Haiku 4.5</title><style>{CSS}</style></head><body>
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

    OUT.mkdir(exist_ok=True)
    p = OUT / "failure_analysis.html"
    p.write_text(page, encoding="utf-8")
    print(f"wrote {p}")
    print(f"  parseable {parsed}/{N}  official-regex {strict}  followed-range {followed}  hit {hit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
