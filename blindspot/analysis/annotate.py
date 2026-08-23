"""Per-asset annotations: a sidecar JSON for every evaluated question, plus
paginated HTML galleries for browsing them.

Two rules shape this:

* **Render from what the model saw, not from the source file.** ScreenSpot-Pro
  ships 5120x2880 screenshots but Haiku receives ~1568px. Annotating the
  original would display detail the model never had -- exactly the confusion
  this project exists to avoid.
* **Rendering is CPU-bound and embarrassingly parallel**, so it runs in a
  process pool. The API stage is I/O-bound and threaded; mixing the two in one
  pool would let PIL work starve the requests.

Usage:
    python -m blindspot.analysis.annotate --datasets charxiv --per-page 50
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from blindspot.core.adapters import load
from blindspot.core.scoring import score

OUT = Path("outputs")
ANNOT = OUT / "annotations"
GALLERY = OUT / "gallery"
ASSETS = OUT / "assets"          # full-size annotated images, linked not inlined
FULL_MAX = 1600                  # long edge of the click-through image
GOOD, BAD = "#0ca30c", "#d03b3b"
THUMB_MAX = 560


# ---------------------------------------------------------------- rendering
def _thumb(im: Image.Image, max_w: int = THUMB_MAX, q: int = 72) -> str:
    im = im.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, max(1, round(im.height * max_w / im.width))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


_FONT_PATHS = ("/System/Library/Fonts/Supplemental/Arial.ttf",
               "/System/Library/Fonts/Helvetica.ttc")


def _font(size: int):
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap(draw, text: str, font, width: int) -> list[str]:
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if draw.textlength(t, font=font) <= width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_overlay(im: Image.Image, row: dict, labels: bool = True) -> Image.Image:
    """Gold target and prediction drawn on the image, for point-type answers.

    `labels=False` omits the "ground truth"/"prediction" captions. They are useful
    in the gallery but must be off when the image is sent to a judge: the caption
    is drawn adjacent to the box and can sit on top of the very UI element under
    adjudication, so the judge is asked whether a box contains an element that the
    annotation itself has covered up.
    """
    W, H = im.size
    if row.get("answer_type") != "point" or not row.get("gold"):
        return im
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(W, H) / 400))
    x0, y0, x1, y1 = [c * s for c, s in zip(row["gold"], (W, H, W, H))]
    d.rectangle([x0, y0, x1, y1], outline=GOOD, width=lw)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ring = max(lw * 14, (x1 - x0) * 1.6, (y1 - y0) * 1.6)
    d.ellipse([cx - ring, cy - ring, cx + ring, cy + ring], outline=GOOD, width=max(1, lw // 2))
    if labels:
        d.text((x0, max(0, y0 - lw * 9)), "ground truth", fill=GOOD, font=_font(lw * 8))
    if row.get("pred"):
        px, py = row["pred"][0] * W, row["pred"][1] * H
        r = lw * 8
        d.line([px - r, py, px + r, py], fill=BAD, width=lw)
        d.line([px, py - r, px, py + r], fill=BAD, width=lw)
        d.ellipse([px - r / 2, py - r / 2, px + r / 2, py + r / 2], outline=BAD, width=lw)
        if labels:
            d.text((px + r, py + r), "prediction", fill=BAD, font=_font(lw * 8))
    return im


def render_full(row: dict) -> str | None:
    """Write the click-through image: the asset with both annotations, plus a
    caption band carrying question / ground truth / prediction.

    Written to disk rather than inlined -- a page with 50 full-size data URIs
    would be tens of megabytes, and the whole point of the larger view is that
    it is only fetched when someone actually clicks.
    """
    src = row.get("_image")
    if not src:
        return None
    out_dir = ASSETS / row["dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{row['uid'].replace(':', '_').replace('/', '_')}.jpg"
    dest = out_dir / name

    im = Image.open(src).convert("RGB")
    if max(im.size) > FULL_MAX:
        sc = FULL_MAX / max(im.size)
        im = im.resize((max(1, round(im.width * sc)), max(1, round(im.height * sc))), Image.LANCZOS)
    im = draw_overlay(im, row)

    # Caption band: for span answers this is the only place the annotation can
    # live, since gold and prediction are text rather than a location.
    W = im.width
    f_lab, f_txt = _font(15), _font(17)
    scratch = ImageDraw.Draw(im)
    ok = (row.get("score") or 0) >= 0.5
    def fmt(v, is_box):
        if row.get("answer_type") != "point":
            return str(v)
        try:
            if is_box:
                x0, y0, x1, y1 = v
                return (f"box x {x0*100:.1f}\u2013{x1*100:.1f}%, y {y0*100:.1f}\u2013{y1*100:.1f}% "
                        f"(centre {(x0+x1)/2*100:.1f}%, {(y0+y1)/2*100:.1f}%)")
            return f"point {v[0]*100:.1f}%, {v[1]*100:.1f}%"
        except Exception:
            return str(v)

    blocks = [("question", str(row.get("question", ""))[:400]),
              ("ground truth", fmt(row.get("gold"), True)),
              ("prediction", fmt(row.get("pred"), False))]
    wrapped = [(lab, _wrap(scratch, txt, f_txt, W - 200)) for lab, txt in blocks]
    band = 22 + sum(24 + 22 * len(ls) for _, ls in wrapped)

    canvas = Image.new("RGB", (W, im.height + band), "#111111")
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    y = im.height + 14
    for lab, lines in wrapped:
        col = GOOD if lab == "ground truth" else (GOOD if (lab == "prediction" and ok) else
                                                  BAD if lab == "prediction" else "#898781")
        d.text((18, y), lab.upper(), fill="#898781", font=f_lab)
        for ln in lines:
            d.text((160, y - 2), ln, fill=col, font=f_txt)
            y += 22
        y += 8
    d.text((W - 120, im.height + 14), "CORRECT" if ok else "WRONG",
           fill=GOOD if ok else BAD, font=f_txt)

    canvas.save(dest, format="JPEG", quality=82)
    return str(Path("..") / "assets" / row["dataset"] / name)


def render_asset(row: dict) -> tuple[str, str | None]:
    """(main thumbnail, optional zoom) with the gold target and prediction drawn."""
    path = row["_image"]
    im = Image.open(path).convert("RGB")

    if row.get("answer_type") != "point":
        return _thumb(im), None

    W, H = im.size
    x0, y0, x1, y1 = [c * s for c, s in zip(row["gold"], (W, H, W, H))]
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(W, H) / 500))
    d.rectangle([x0, y0, x1, y1], outline=GOOD, width=lw)
    if row.get("pred"):
        px, py = row["pred"][0] * W, row["pred"][1] * H
        r = lw * 6
        d.line([px - r, py, px + r, py], fill=BAD, width=lw)
        d.line([px, py - r, px, py + r], fill=BAD, width=lw)
    pad = max(x1 - x0, y1 - y0) * 6 + 80
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    zoom = im.crop((round(max(0, cx - pad)), round(max(0, cy - pad)),
                    round(min(W, cx + pad)), round(min(H, cy + pad))))
    return _thumb(im), _thumb(zoom, max_w=420)


def build_one(row: dict) -> dict:
    """Module-level so the process pool can pickle it. Writes the sidecar JSON."""
    main, zoom = render_asset(row)
    full = render_full(row)
    rec = {k: row[k] for k in
           ("uid", "dataset", "answer_type", "question", "gold", "pred",
            "score", "metric", "grading_confidence", "primitive",
            "center_distance", "signed_error", "abs_error", "true_count",
            "polarity", "not_applicable", "judge_score", "string_score") if k in row}
    rec.update({
        "image": row["_image"],
        "sent_image_size": row.get("sent_image_sizes", [None])[0],
        "usage": row.get("usage"),
        "request_id": row.get("request_id"),
        "meta": row.get("meta", {}),
    })
    out = ANNOT / row["dataset"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{row['uid'].replace(':', '_').replace('/', '_')}.json").write_text(
        json.dumps(rec, indent=1, default=str))
    return {**rec, "_thumb": main, "_zoom": zoom, "_full": full}


# ------------------------------------------------------------------ gallery
CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;
 --ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--border:rgba(11,11,11,.10);
 --accent:#2a78d6;--good:#0ca30c;--bad:#d03b3b}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
 --surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--border:rgba(255,255,255,.10);--accent:#3987e5}}
:root[data-theme=dark]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;
 --ink2:#c3c2b7;--grid:#2c2c2a;--border:rgba(255,255,255,.10);--accent:#3987e5}
*{box-sizing:border-box}
html,body{background:var(--page);color:var(--ink);margin:0;
 font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 70px}
h1{font-size:23px;margin:0 0 4px}
.dek{color:var(--ink2);margin:0 0 18px}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:0 0 18px;
 padding:12px 14px;background:var(--surface);border:1px solid var(--border);border-radius:10px}
.bar label{font-size:12.5px;color:var(--ink2)}
select,button{font:inherit;font-size:13px;padding:5px 9px;border-radius:7px;
 border:1px solid var(--border);background:var(--page);color:var(--ink)}
.case{background:var(--surface);border:1px solid var(--border);border-radius:11px;
 padding:13px;margin-bottom:12px}
.hd{display:flex;gap:9px;align-items:flex-start;margin-bottom:10px}
.pill{font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;white-space:nowrap}
.ok{background:color-mix(in srgb,var(--good) 15%,transparent);color:var(--good)}
.no{background:color-mix(in srgb,var(--bad) 15%,transparent);color:var(--bad)}
.tag{font-size:11px;color:var(--muted);border:1px solid var(--border);
 padding:2px 7px;border-radius:999px}
.q{font-size:13.5px;line-height:1.45;flex:1}
.imgs{display:grid;grid-template-columns:1fr 300px;gap:10px;align-items:start}
.imgs.one{grid-template-columns:1fr}
@media(max-width:760px){.imgs{grid-template-columns:1fr}}
.imgs img{width:100%;max-height:250px;object-fit:contain;object-position:top left;
 background:var(--grid);border:1px solid var(--border);border-radius:7px;display:block}
dl{margin:11px 0 0;font-size:12.5px;display:grid;gap:4px}
dl>div{display:grid;grid-template-columns:150px 1fr;gap:10px}
dt{color:var(--muted)} dd{margin:0;overflow-wrap:anywhere}
dd.g{color:var(--good)} dd.b{color:var(--bad)}
nav{display:flex;gap:7px;flex-wrap:wrap;margin:20px 0 0}
nav a{font-size:13px;padding:5px 10px;border:1px solid var(--border);
 border-radius:7px;text-decoration:none;color:var(--ink2);background:var(--surface)}
nav a.cur{background:var(--accent);color:#fff;border-color:transparent}
.imgs a{display:block;position:relative}
.imgs a::after{content:"click to enlarge";position:absolute;right:6px;bottom:6px;
 font-size:10.5px;padding:2px 7px;border-radius:999px;background:rgba(0,0,0,.62);color:#fff;
 opacity:0;transition:opacity .12s}
.imgs a:hover::after,.imgs a:focus-visible::after{opacity:1}
.imgs a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
#lb{position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.92);display:none}
#lb.on{display:block}
#lb .stage{position:absolute;inset:0;overflow:hidden;cursor:grab;touch-action:none}
#lb .stage.grabbing{cursor:grabbing}
#lb img{position:absolute;top:0;left:0;transform-origin:0 0;will-change:transform;
 max-width:none;max-height:none;user-select:none;-webkit-user-drag:none}
#lb .ctrl{position:fixed;top:14px;left:50%;transform:translateX(-50%);display:flex;gap:6px;
 align-items:center;z-index:2;background:rgba(20,20,20,.85);padding:6px 8px;border-radius:10px}
#lb .ctrl button{font:inherit;font-size:14px;line-height:1;min-width:34px;padding:7px 9px;
 border-radius:7px;border:1px solid rgba(255,255,255,.18);background:#222;color:#eee;cursor:pointer}
#lb .ctrl button:hover{background:#333}
#lb .ctrl .lvl{color:#c3c2b7;font-size:12.5px;min-width:52px;text-align:center;
 font-variant-numeric:tabular-nums}
#lb .hint{position:fixed;bottom:14px;left:50%;transform:translateX(-50%);color:#898781;
 font-size:12px;z-index:2;background:rgba(20,20,20,.8);padding:5px 11px;border-radius:999px}
"""

LIGHTBOX_JS = """
(function(){
 const lb=document.getElementById('lb'); if(!lb) return;
 const stage=lb.querySelector('.stage'), img=lb.querySelector('img'),
       lvl=lb.querySelector('.lvl');
 let s=1, fit=1, tx=0, ty=0, drag=false, lx=0, ly=0;

 const apply=()=>{img.style.transform=`translate(${tx}px,${ty}px) scale(${s})`;
                  lvl.textContent=Math.round(s/fit*100)+'%';};
 // Fit on open, then let the user magnify well past 1:1 -- a 22px UI target is
 // the whole reason this exists, so the ceiling is deliberately high.
 const fitView=()=>{const r=stage.getBoundingClientRect();
   fit=Math.min(r.width/img.naturalWidth, r.height/img.naturalHeight);
   s=fit; tx=(r.width-img.naturalWidth*s)/2; ty=(r.height-img.naturalHeight*s)/2; apply();};
 const zoomAt=(px,py,f)=>{const ns=Math.min(fit*40, Math.max(fit*0.5, s*f));
   tx=px-(px-tx)*(ns/s); ty=py-(py-ty)*(ns/s); s=ns; apply();};
 const centreZoom=f=>{const r=stage.getBoundingClientRect(); zoomAt(r.width/2,r.height/2,f);};

 img.addEventListener('load', fitView);
 addEventListener('resize', ()=>{ if(lb.classList.contains('on')) fitView(); });

 stage.addEventListener('wheel', e=>{ e.preventDefault();
   const r=stage.getBoundingClientRect();
   zoomAt(e.clientX-r.left, e.clientY-r.top, e.deltaY<0?1.18:1/1.18); }, {passive:false});
 stage.addEventListener('dblclick', e=>{ const r=stage.getBoundingClientRect();
   if(s>fit*1.5) fitView(); else zoomAt(e.clientX-r.left, e.clientY-r.top, 5); });
 stage.addEventListener('mousedown', e=>{ drag=true; lx=e.clientX; ly=e.clientY;
   stage.classList.add('grabbing'); e.preventDefault(); });
 addEventListener('mousemove', e=>{ if(!drag) return;
   tx+=e.clientX-lx; ty+=e.clientY-ly; lx=e.clientX; ly=e.clientY; apply(); });
 addEventListener('mouseup', ()=>{ drag=false; stage.classList.remove('grabbing'); });

 lb.querySelector('.zin').onclick =()=>centreZoom(1.5);
 lb.querySelector('.zout').onclick=()=>centreZoom(1/1.5);
 lb.querySelector('.zfit').onclick=fitView;
 const close=()=>{ lb.classList.remove('on'); img.removeAttribute('src'); };
 lb.querySelector('.zclose').onclick=close;
 addEventListener('keydown', e=>{ if(!lb.classList.contains('on')) return;
   if(e.key==='Escape') close();
   if(e.key==='+'||e.key==='=') centreZoom(1.5);
   if(e.key==='-') centreZoom(1/1.5);
   if(e.key==='0') fitView(); });

 document.querySelectorAll('a.zoom').forEach(a=>a.addEventListener('click', e=>{
   e.preventDefault(); img.src=a.getAttribute('href'); lb.classList.add('on'); }));
})();
"""

LIGHTBOX_HTML = """
<div id="lb">
 <div class="ctrl">
  <button class="zout" title="zoom out (-)">&minus;</button>
  <span class="lvl">100%</span>
  <button class="zin" title="zoom in (+)">+</button>
  <button class="zfit" title="fit to screen (0)">fit</button>
  <button class="zclose" title="close (Esc)">&times;</button>
 </div>
 <div class="stage"><img alt="full size, annotated"></div>
 <span class="hint">scroll or double-click to zoom &middot; drag to pan &middot; Esc to close</span>
</div>
"""


JS = """
const sel=document.getElementById('f-res'),pr=document.getElementById('f-prim');
function apply(){const r=sel.value,p=pr?pr.value:'all';
 document.querySelectorAll('.case').forEach(c=>{
  const okr = r==='all'||c.dataset.res===r;
  const okp = p==='all'||c.dataset.prim===p;
  c.style.display=(okr&&okp)?'':'none';});}
sel.addEventListener('change',apply); if(pr)pr.addEventListener('change',apply);
document.getElementById('theme').addEventListener('click',()=>{
 const d=document.documentElement.dataset.theme==='dark';
 document.documentElement.dataset.theme=d?'light':'dark';});
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def _coord(v, is_box: bool) -> str:
    """Readable form for normalized boxes/points; passthrough for everything else."""
    try:
        if is_box:
            x0, y0, x1, y1 = v
            return (f"centre {(x0+x1)/2*100:.1f}%, {(y0+y1)/2*100:.1f}% "
                    f"(box {x0*100:.1f}\u2013{x1*100:.1f}% x {y0*100:.1f}\u2013{y1*100:.1f}%)")
        return f"{v[0]*100:.1f}%, {v[1]*100:.1f}%"
    except Exception:
        return str(v)


def case_html(a: dict) -> str:
    """One evaluated question. Images link to the full-size annotated copy."""
    ok = (a.get("score") or 0) >= 0.5
    is_point = a.get("answer_type") == "point"
    full = a.get("_full")

    def linked(img: str) -> str:
        if not full:
            return img
        return f'<a class="zoom" href="{full}" title="open full size with both annotations">{img}</a>'

    main = linked(f'<img src="{a["_thumb"]}" alt="evaluated asset">')
    zoom = ""
    if a.get("_zoom"):
        zoom = ('<figure>'
                + linked(f'<img src="{a["_zoom"]}" alt="zoom on target">')
                + '<figcaption style="font-size:11px;color:var(--muted)">zoom on target</figcaption>'
                + '</figure>')

    prim = a.get("primitive") or "-"
    extra = ""
    if a.get("metric") == "exact_count":
        extra = (f'<div><dt>signed error</dt><dd>{a.get("signed_error")}</dd></div>'
                 f'<div><dt>true count</dt><dd>{a.get("true_count")}</dd></div>')
    elif a.get("metric") == "click_in_bbox":
        cd = a.get("center_distance")
        extra = ('<div><dt>distance to target centre</dt><dd>'
                 + (f"{cd*100:.1f}% of screen" if cd is not None else "&mdash;")
                 + "</dd></div>")
    conf_note = (' <span class="tag">graded approximately</span>'
                 if a.get("grading_confidence") == "fuzzy" and a.get("judge_score") is None else "")
    if a.get("judge_score") is not None:
        agree = (a["judge_score"] >= .5) == ((a.get("string_score") or 0) >= .5)
        extra += ('<div><dt>CharXiv official judge</dt><dd class="{}">{}</dd></div>'
                  .format("g" if a["judge_score"] >= .5 else "b",
                          ("scored correct" if a["judge_score"] >= .5 else "scored wrong")
                          + ("" if agree else " &mdash; disagrees with string matching")))

    return f"""
<article class="case" data-res="{'ok' if ok else 'no'}" data-prim="{esc(prim)}">
 <div class="hd">
  <span class="pill {'ok' if ok else 'no'}">{'&#10003;' if ok else '&#10007;'}</span>
  <span class="tag">{esc(prim)}</span>
  <span class="q">{esc(a.get('question',''))[:400]}</span>
 </div>
 <div class="imgs{'' if zoom else ' one'}"><figure>{main}</figure>{zoom}</div>
 <dl>
  <div><dt>model answered</dt><dd class="{'g' if ok else 'b'}">{esc(_coord(a.get('pred'), False) if is_point else a.get('pred'))}</dd></div>
  <div><dt>gold</dt><dd class="g">{esc(_coord(a.get('gold'), True) if is_point else a.get('gold'))}</dd></div>
  <div><dt>metric</dt><dd>{esc(a.get('metric'))}{conf_note}</dd></div>
  {extra}
  <div><dt>uid</dt><dd style="color:var(--muted)">{esc(a.get('uid'))}</dd></div>
 </dl>
</article>"""


def write_pages(dataset: str, annots: list[dict], per_page: int) -> list[Path]:
    GALLERY.mkdir(parents=True, exist_ok=True)
    prims = sorted({a.get("primitive") or "-" for a in annots})
    pages = [annots[i:i + per_page] for i in range(0, len(annots), per_page)] or [[]]
    paths = []
    for i, chunk in enumerate(pages):
        nav = "".join(
            f'<a href="{dataset}_{j:03d}.html"{" class=cur" if j == i else ""}>{j+1}</a>'
            for j in range(len(pages)))
        opts = "".join(f'<option value="{esc(p)}">{esc(p)}</option>' for p in prims)
        n_ok = sum(1 for a in annots if (a.get("score") or 0) >= 0.5)
        p = GALLERY / f"{dataset}_{i:03d}.html"
        p.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(dataset)} &mdash; annotated assets {i+1}/{len(pages)}</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>{esc(dataset)} &mdash; annotated assets</h1>
<p class="dek">Every evaluated question with its image, gold answer and Haiku 4.5's answer.
{len(annots)} assets &middot; {n_ok} correct ({n_ok/max(len(annots),1)*100:.0f}%) &middot;
page {i+1} of {len(pages)}</p>
<div class="bar">
 <label for="f-res">result</label>
 <select id="f-res"><option value="all">all</option><option value="no">incorrect only</option>
 <option value="ok">correct only</option></select>
 <label for="f-prim">primitive</label>
 <select id="f-prim"><option value="all">all</option>{opts}</select>
 <button id="theme" type="button">toggle theme</button>
</div>
{''.join(case_html(a) for a in chunk)}
<nav>{nav}</nav>
</div>
{LIGHTBOX_HTML}
<script>{JS}
{LIGHTBOX_JS}</script></body></html>""", encoding="utf-8")
        paths.append(p)
    return paths


# --------------------------------------------------------------------- main
def rows_for(dataset: str, tag: str | None = None) -> list[dict]:
    """Rows for the gallery, sourced from the aggregate layer.

    This used to re-implement loading and scoring locally, which meant it
    silently missed everything the aggregate layer adds -- including CharXiv's
    official judge verdict, so the galleries were showing string-match results
    while the report showed judged ones.
    """
    from blindspot.analysis.aggregate import load_rows
    from blindspot.core.taxonomy import LABELS

    rows = []
    for r in load_rows(dataset):
        ex = r["_ex"]
        row = {k: v for k, v in r.items() if k != "_ex"}
        row["_image"] = ex.images[0]
        row["question"] = ex.question
        row["primitive"] = (LABELS.get(r.get("primitive"))
                            or ex.meta.get("qlabel")
                            or (ex.meta.get("operation") or [None])[0]
                            or ex.meta.get("ui_type") or ex.meta.get("split") or "-")
        row.setdefault("answer_type", ex.answer_type)
        row["gold"] = ex.gold
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Build per-asset annotations + galleries")
    ap.add_argument("--datasets", nargs="+",
                    default=["charxiv", "infographicvqa", "screenspot_pro"])
    ap.add_argument("--tag", default="haiku-4-5_think2000_native_r0")
    ap.add_argument("--per-page", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None, help="cap assets per dataset")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8) - 6))
    a = ap.parse_args()

    ANNOT.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets:
        rows = rows_for(ds, a.tag)
        if not rows:
            print(f"{ds}: no results for tag {a.tag}, skipping")
            continue
        if a.limit:
            rows = rows[: a.limit]
        # Incorrect first: the failures are what the report is about.
        rows.sort(key=lambda r: (r.get("score") or 0, r["uid"]))
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            annots = list(pool.map(build_one, rows, chunksize=8))
        pages = write_pages(ds, annots, a.per_page)
        print(f"{ds}: {len(annots)} annotated -> {ANNOT/ds}/  +  {len(pages)} gallery page(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
