"""One HTML page per perceptual primitive: what it tests, how it scores, and
worked examples on both sides of the line.

Each page carries correct AND incorrect cases deliberately. A gallery of
failures alone tells you the model is bad without telling you what "good" looked
like on the same task, and for primitives scoring above 90% the interesting
question is what the residual failures have in common -- which you can only see
next to the successes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from blindspot.analysis.aggregate import load_rows, cell, slice_by
from blindspot.analysis.annotate import CSS as GCSS, LIGHTBOX_HTML, LIGHTBOX_JS, build_one, esc
from blindspot.core.failure_modes import LABELS as FM_LABELS
from blindspot.core.taxonomy import LABELS, primitive_for

OUT = Path("outputs")
TASKS = OUT / "tasks"

# What each primitive actually asks the model to do, and how it is graded.
DESCRIBE = {
    "counting": ("Count discrete objects in a figure &mdash; lines in a plot, labels in a "
                 "legend, ticks across all axes, subplots, or items in an infographic.",
                 "Exact match on the number."),
    "line_following": ("Trace lines through a plot and decide whether any of them cross. "
                       "The only primitive here with a single source.",
                       "Exact match on yes/no."),
    "localization_read": ("Find a spatial extreme &mdash; the leftmost or rightmost x tick, "
                          "the highest or lowest y tick &mdash; and report the value written there. "
                          "Ends in reading text.",
                          "Normalized text/numeric match, or the official CharXiv judge."),
    "localization_point": ("Find a described UI element on a dense professional screenshot and "
                           "return its centre as coordinates. Ends in emitting numbers, not text.",
                           "Click-in-bbox: the predicted point must fall inside the gold box."),
    "value_interpolation": ("Read a value off a continuous scale &mdash; tick spacing, colorbar "
                            "range and maximum &mdash; where the answer is often not printed and "
                            "must be inferred from the axis.",
                            "Numeric match with tolerance."),
    "binding": ("Associate legend entries with the series they label, and report them in "
                "reading order.", "Normalized match on the ordered list."),
    "structure": ("Parse the layout of a composite figure &mdash; how many subplots and in "
                  "what grid.", "Normalized match (\"n by m\")."),
    "text_in_situ": ("Read text embedded in a figure: the plot title and the axis labels.",
                     "Normalized match; free-text, so a lower bound under string matching."),
    "comparison": ("Compare two or more quantities read off the figure and pick the larger, "
                   "the fastest, the third-highest.", "ANLS / normalized match."),
    "arithmetic": ("Derive a number that is written nowhere on the page &mdash; an inverse "
                   "percentage, a difference, a share.", "ANLS against the accepted answers."),
    "composition": ("Answer a question that requires combining several separate readings of "
                    "the same chart.", "Official CharXiv reasoning judge / ANLS."),
}

EXTRA_CSS = """
.hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
.hero .tile{background:var(--surface);border:1px solid var(--border);border-radius:11px;padding:14px 15px}
.hero .tlab{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.hero .tval{font-size:28px;margin:6px 0 2px}.hero .tnote{font-size:12px;color:var(--ink2)}
.what{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent);
 border-radius:10px;padding:14px 16px;margin:14px 0;font-size:13.5px;color:var(--ink2)}
.what b{color:var(--ink)}
.brow{display:grid;grid-template-columns:210px 1fr 74px;align-items:center;gap:11px;padding:4px 0}
.blab{font-size:12.5px;color:var(--ink2);text-align:right;overflow-wrap:anywhere}
.btrack{height:14px;background:var(--grid);border-radius:4px}
.bfill{height:100%;background:var(--accent);border-radius:0 4px 4px 0}
.bval{font-size:12.5px;text-align:right;font-variant-numeric:tabular-nums}
.bval span{display:block;font-size:10.5px;color:var(--muted)}
h2.sec{font-size:17px;margin:34px 0 6px;padding-top:18px;border-top:1px solid var(--grid)}
h2.sec .n{font-size:13px;font-weight:400;color:var(--ink2)}
svg.scatter{width:100%;height:auto;background:var(--surface);border:1px solid var(--border);border-radius:11px}
.scatter .ax{stroke:var(--axis);stroke-width:1}
.scatter .gl{stroke:var(--grid);stroke-width:1}
.scatter .pt{fill:var(--bad);opacity:.45}
.scatter .pt.ok{fill:var(--good);opacity:.75}
.scatter .ideal{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:4 3;fill:none}
.scatter .fit{stroke:var(--accent);stroke-width:2;fill:none}
.scatter text{font-size:11px;fill:var(--ink2)}
"""


def _fmt(v, d=1):
    return "&mdash;" if v is None else f"{v*100:.{d}f}"


def bar_rows(items) -> str:
    out = []
    for lab, acc, n in items:
        out.append(f'<div class="brow"><div class="blab">{esc(lab)}</div>'
                   f'<div class="btrack"><div class="bfill" style="width:{max((acc or 0)*100,0.6):.1f}%"></div></div>'
                   f'<div class="bval">{_fmt(acc,0)}%<span>n={n}</span></div></div>')
    return "".join(out)


def scatter_svg(rows: list[dict]) -> str:
    """Predicted vs gold centre, both axes, with the fitted line against y=x.

    This is the localization finding made visible: if the model tracked the
    target the points would sit on the dashed diagonal. A shallower fitted line
    means predictions are collapsing toward a prior instead.
    """
    import statistics as st
    pts = [r for r in rows if isinstance(r.get("pred"), (list, tuple)) and len(r["pred"]) == 2]
    if len(pts) < 20:
        return ""

    def panel(idx, name, ox):
        gold = [(r["gold"][idx] + r["gold"][idx + 2]) / 2 for r in pts]
        pred = [r["pred"][idx] for r in pts]
        mg, mp = st.mean(gold), st.mean(pred)
        sxx = sum((g - mg) ** 2 for g in gold)
        slope = (sum((g - mg) * (p - mp) for g, p in zip(gold, pred)) / sxx) if sxx else 0
        icept = mp - slope * mg
        S = 250
        dots = "".join(
            f'<circle class="pt{" ok" if (r.get("score") or 0) >= .5 else ""}" '
            f'cx="{g*S:.1f}" cy="{S-p*S:.1f}" r="2.4"/>'
            for g, p, r in zip(gold, pred, pts))
        y0, y1 = icept, slope + icept
        # Clip: with slope 0.865 and intercept 0.192 the x fit exits the top of
        # the box before gold=1, and an unclipped line drawn outside its own axes
        # reads as a rendering bug rather than as extrapolation.
        return f"""
  <g transform="translate({ox},14)">
    <clipPath id="clip{idx}"><rect x="0" y="0" width="{S}" height="{S}"/></clipPath>
    <text x="0" y="-2">{name} &mdash; slope {slope:.3f}, intercept {icept:.3f}</text>
    <rect class="gl" x="0" y="0" width="{S}" height="{S}" fill="none"/>
    <line class="ideal" x1="0" y1="{S}" x2="{S}" y2="0"/>
    <g clip-path="url(#clip{idx})">{dots}
    <line class="fit" x1="0" y1="{S-y0*S:.1f}" x2="{S}" y2="{S-y1*S:.1f}"/></g>
    <line class="ax" x1="0" y1="{S}" x2="{S}" y2="{S}"/>
    <line class="ax" x1="0" y1="0" x2="0" y2="{S}"/>
    <text x="{S/2-28}" y="{S+16}">gold centre</text>
    <text transform="translate(-8,{S/2+30}) rotate(-90)">predicted</text>
  </g>"""

    return f"""<figure style="margin:14px 0">
<svg class="scatter" viewBox="0 0 610 300" role="img"
     aria-label="Predicted versus gold centre coordinates on both axes">
 {panel(0,'horizontal (x)',34)}{panel(1,'vertical (y)',330)}
</svg>
<figcaption style="font-size:12.5px;color:var(--ink2);margin-top:8px">
Dashed diagonal = perfect tracking. Solid line = the actual fit. Green points landed
inside the gold box, red missed. A slope well under 1 means predictions are compressing
toward a central prior rather than following the target.</figcaption></figure>"""


def localization_decomposition(rows: list[dict]) -> str:
    """Split the failure into 'never found the region' vs 'found it, missed the box'.

    A single accuracy number hides that these are different problems with
    different fixes. The regression slope alone is actively misleading here: it
    reads as compression when what is really happening is a bimodal population
    -- a band that tracks the target plus a diffuse cloud that does not.
    """
    pts = [r for r in rows if isinstance(r.get("pred"), (list, tuple)) and len(r["pred"]) == 2]
    if not pts:
        return ""
    def err(r, i):
        return abs(r["pred"][i] - (r["gold"][i] + r["gold"][i + 2]) / 2)
    n = len(pts)
    track = [r for r in pts if err(r, 0) <= .10 and err(r, 1) <= .10]
    lost = [r for r in pts if err(r, 0) > .25 or err(r, 1) > .25]
    hit = lambda s: sum(r.get("score") or 0 for r in s) / max(len(s), 1) * 100
    mx = sorted(err(r, 0) for r in pts)[n // 2] * 100
    my = sorted(err(r, 1) for r in pts)[n // 2] * 100
    return f"""
<h2 class="sec">Two different failures, not one</h2>
<div class="card">
{bar_rows([("found the region (within 10% on both axes)", len(track)/n, len(track)),
           ("landed in the box, given the region was right", hit(track)/100, len(track)),
           ("lost entirely (&gt;25% off on either axis)", len(lost)/n, len(lost))])}
</div>
<div class="what">
<b>The model finds roughly the right area {len(track)/n*100:.0f}% of the time, and still hits the
box in only {hit(track):.1f}% of those cases.</b> Those are separate problems: coarse search fails on
{len(lost)/n*100:.0f}% of items outright, while on the ones it does locate, the target is around
22px across after the API downscales the screenshot &mdash; small enough that "close" is not close
enough. Median error is {mx:.1f}% of screen width horizontally and {my:.1f}% vertically, so the
vertical axis is the <em>more</em> accurate one; the shallow y regression slope reflects the outlier
cloud, not compression.
</div>"""


def build_page(prim: str, rows: list[dict], n_examples: int) -> Path:
    """One primitive: stats, sub-slices, scatter (if pointing), worked examples."""
    TASKS.mkdir(parents=True, exist_ok=True)
    c = cell(rows) or {}
    answerable = [r for r in rows if not r["not_applicable"]]
    ca = cell(answerable) or {}
    na_rate = 1 - len(answerable) / max(len(rows), 1)
    by_ds = slice_by(rows, lambda r: r["dataset"])
    what, metric = DESCRIBE.get(prim, ("", ""))

    # Sub-slices that make sense for this primitive.
    sub = []
    if prim == "localization_point":
        def bucket(r):
            f = r["_ex"].meta.get("target_area_frac", 0) or 0
            s = math.sqrt(f * 1568 * 882)
            return next((n for l, n in ((12, "<12px"), (20, "12-20px"), (32, "20-32px"),
                                        (56, "32-56px")) if s < l), ">=56px")
        sub = [("by target size after downscale", slice_by(rows, bucket, 5)),
               ("by element type", slice_by(rows, lambda r: r["_ex"].meta.get("ui_type") or "?", 5))]
    else:
        sub = [("by question type", slice_by(rows, lambda r: r["_ex"].meta.get("qlabel")
                                             or (r["_ex"].meta.get("operation") or ["?"])[0], 5))]

    # Examples from both sides. Hardest failures first; successes for contrast.
    wrong = sorted([r for r in answerable if (r.get("score") or 0) < 0.5],
                   key=lambda r: (r["_ex"].meta.get("target_area_frac", 1), r["uid"]))[:n_examples]
    right = [r for r in answerable if (r.get("score") or 0) >= 0.5][:max(2, n_examples // 2)]
    # Strip `_ex` (an Example) before crossing the process boundary and carry the
    # few fields the renderer needs explicitly -- spawn pickles every argument.
    jobs = []
    for r in wrong + right:
        j = {k: v for k, v in r.items() if k != "_ex"}
        j.update({"_image": r["_ex"].images[0], "question": r["_ex"].question,
                  "answer_type": r.get("answer_type") or r["_ex"].answer_type,
                  "gold": r["_ex"].gold})
        jobs.append(j)
    with ProcessPoolExecutor(max_workers=8) as pool:
        cards = list(pool.map(build_one, jobs, chunksize=2))
    neg, pos = cards[:len(wrong)], cards[len(wrong):]

    def case(a, ok):
        full = a.get("_full")

        def link(img: str) -> str:
            if not full:
                return img
            return f'<a class="zoom" href="{full}" title="open full size with both annotations">{img}</a>'

        main_img = link('<img src="%s" alt="asset">' % a["_thumb"])
        zoom = ""
        if a.get("_zoom"):
            zoom = "<figure>%s</figure>" % link('<img src="%s" alt="zoom">' % a["_zoom"])
        return f"""<article class="case">
 <div class="hd"><span class="pill {'ok' if ok else 'no'}">{'&#10003; correct' if ok else '&#10007; wrong'}</span>
 <span class="q">{esc(a.get('question',''))[:340]}</span></div>
 <div class="imgs{'' if zoom else ' one'}"><figure>{main_img}</figure>{zoom}</div>
 <dl><div><dt>model answered</dt><dd class="{'g' if ok else 'b'}">{esc(a.get('pred'))}</dd></div>
 <div><dt>gold</dt><dd class="g">{esc(a.get('gold'))}</dd></div>
 <div><dt>uid</dt><dd style="color:var(--muted)">{esc(a.get('uid'))}</dd></div></dl></article>"""

    # Why this primitive's failures failed.
    import collections as _c
    fails = [r for r in rows if (r.get("score") or 0) < 0.5]
    fm = _c.Counter(r.get("failure_mode", "unclassified") for r in fails)
    fm_block = ""
    if fm:
        fm_block = ('<h2 class="sec">Why the failures failed <span class="n">&mdash; '
                    f'{len(fails)} scored-wrong answers</span></h2><div class="card">'
                    + bar_rows([(FM_LABELS.get(m, m), c / len(fails), c)
                                for m, c in fm.most_common()]) + "</div>")

    na_note = (f'<div class="what"><b>{na_rate*100:.0f}% of golds are "Not Applicable"</b> for this '
               f'primitive &mdash; the figure has nothing to count or trace. Headline accuracy below is '
               f'the answerable subset; pooling the rest in would measure "can you tell this does not '
               f'apply" instead.</div>') if na_rate > 0.1 else ""

    subs = "".join(f'<h2 class="sec">{esc(t)}</h2><div class="card">'
                   f'{bar_rows([(d["label"], d["acc"], d["n"]) for d in items])}</div>'
                   for t, items in sub if items)

    return _write(prim, f"""
<h1>{LABELS[prim]}</h1>
<p class="dek">{what}</p>
<div class="what"><b>How it is scored.</b> {metric}</div>
{na_note}
<div class="hero">
 <div class="tile"><div class="tlab">accuracy</div><div class="tval">{_fmt(ca.get('acc'),0)}%</div>
  <div class="tnote">n={ca.get('n',0)}{f" &middot; CI {_fmt(ca.get('ci_lo'),0)}&ndash;{_fmt(ca.get('ci_hi'),0)}%" if ca.get('ci_lo') is not None else ""}</div></div>
 <div class="tile"><div class="tlab">questions asked</div><div class="tval">{len(rows)}</div>
  <div class="tnote">{len(answerable)} answerable</div></div>
 <div class="tile"><div class="tlab">sources</div><div class="tval">{len(by_ds)}</div>
  <div class="tnote">{esc(', '.join(d['label'] for d in by_ds))}</div></div>
</div>
{'<h2 class="sec">by dataset</h2><div class="card">' + bar_rows([(d['label'], d['acc'], d['n']) for d in by_ds]) + '</div>' if len(by_ds) > 1 else ''}
{subs}
{fm_block}
{('<h2 class="sec">Where the predictions actually land</h2>' + scatter_svg(rows) + localization_decomposition(rows)) if prim == 'localization_point' else ''}
<h2 class="sec">Failures <span class="n">&mdash; hardest cases first</span></h2>
{''.join(case(a, False) for a in neg) or '<p class="dek">No failures in this slice.</p>'}
<h2 class="sec">Successes <span class="n">&mdash; the same task, done right</span></h2>
{''.join(case(a, True) for a in pos) or '<p class="dek">No successes in this slice.</p>'}
""")


def _write(prim: str, body: str) -> Path:
    p = TASKS / f"{prim}.html"
    nav = " &middot; ".join(f'<a href="{k}.html">{esc(LABELS[k])}</a>' for k in LABELS)
    p.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(LABELS[prim])} &mdash; Haiku 4.5</title>
<style>{GCSS}{EXTRA_CSS}</style></head><body><div class="wrap">
<p style="font-size:12.5px;color:var(--muted)"><a href="../report.html">&larr; overview</a></p>
{body}
<h2 class="sec">Other primitives</h2><p class="dek" style="font-size:13px">{nav}</p>
</div>
{LIGHTBOX_HTML}
<script>{LIGHTBOX_JS}</script></body></html>""", encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-primitive task pages")
    ap.add_argument("--examples", type=int, default=8)
    a = ap.parse_args()
    rows = [r for ds in ("charxiv", "infographicvqa", "screenspot_pro") for r in load_rows(ds)]
    by_prim = defaultdict(list)
    for r in rows:
        if r["primitive"]:
            by_prim[r["primitive"]].append(r)
    for prim in LABELS:
        if prim in by_prim:
            p = build_page(prim, by_prim[prim], a.examples)
            print(f"  {LABELS[prim]:42} n={len(by_prim[prim]):<5} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
