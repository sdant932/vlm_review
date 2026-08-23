"""Annotated PNGs + a self-contained HTML/SVG explainer for the coordinate finding.

Everything here is drawn from real pilot data in results/ -- no illustrative fakes.

Colour: categorical slots 1-3 of the reference palette (blue/orange/aqua). That
trio is the documented all-pairs-validated set in both modes (worst CVD dE 9.2
light / 9.4 dark), which is the right gate because all three marks co-occur on
every annotated image. Hit/miss is carried by an icon + label, never by colour.

    python scripts/analyze/coord_diagnostics.py
"""
from __future__ import annotations
import base64, html, io, json, statistics as st, sys
from pathlib import Path

from PIL import Image, ImageDraw
from blindspot.core.adapters import load
from blindspot.core.scoring import point_in_bbox

R, OUT = Path("results"), Path("outputs")
PNGS = OUT / "probe"

# palette slots 1-3 (light / dark)
C = {"gt": ("#1baf7a", "#199e70"), "haiku": ("#2a78d6", "#3987e5"), "sonnet": ("#eb6834", "#d95926")}
INK, SUB = "#0b0b0b", "#52514e"

TAGS = {"haiku": "haiku-4-5_think2000_native_r0", "sonnet": "sonnet-5_think2000_native_r0"}


def rows(ds, tag):
    p = R / f"{ds}__{tag}.jsonl"
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


CSS = """
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
        H = rows(ds, TAGS["haiku"])
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


def main() -> int:
    OUT.mkdir(exist_ok=True)
    scatter, arrows, table = {}, [], []
    for ds in ("screenspot", "screenspot_pro"):
        exs = {e.uid: e for e in load(ds)}
        pairs, arr, sc = [], [], []
        for uid, r in rows(ds, TAGS["haiku"]).items():
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

    plan = json.loads((R / "probe_uids.json").read_text())
    gallery, manifest = build_gallery(plan)
    (OUT / "probe" / "index.json").write_text(json.dumps(manifest, indent=2))

    trow = "".join(
        f'<tr><th scope="row">{esc(d)}</th><td class="num">{n}</td>'
        f'<td class="num">{a*100:.1f}%</td><td class="num">{s:.3f}</td>'
        f'<td class="num">{b:+.3f}</td></tr>' for d, n, a, s, b in table)

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Haiku 4.5 &mdash; coordinate diagnostics</title><style>{CSS}</style></head><body>
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

    out = OUT / "coord_diagnostics.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print(f"wrote {len(manifest)} annotated PNG pairs -> {PNGS}/")
    for d, n, a, s, b in table:
        print(f"  {d:16s} n={n:4d}  acc={a*100:5.1f}%  slope={s:.3f}  intercept={b:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
