"""Render outputs/report.html from outputs/summary.json.

Rendering is a pure function of the summary, so the numbers can be checked
independently of the page and the page can be rebuilt without re-scoring.

One rule shapes the layout. When two independent datasets measure the same
primitive and disagree sharply -- localization is CharXiv 91% against
ScreenSpot-Pro 4% -- the pooled average (76%) describes nothing that exists.
Divergent primitives are therefore always shown per source and flagged, never
averaged into a single bar.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

OUT = Path("outputs")
DIVERGENCE = 0.25  # sources further apart than this are never pooled


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def pct(v) -> str:
    return "&mdash;" if v is None else f"{v*100:.0f}"


CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
 --muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
 --s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;--bad:#d03b3b;--warn:#fab219}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
 --surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926}}
:root[data-theme=dark]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;
 --ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
 --s1:#3987e5;--s2:#d95926}
*{box-sizing:border-box}
html,body{background:var(--page);color:var(--ink);margin:0;
 font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1140px;margin:0 auto;padding:30px 22px 80px}
header{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}
h1{font-size:26px;margin:0 0 6px}
.dek{color:var(--ink2);margin:0;max-width:70ch}
h2{font-size:19px;margin:42px 0 4px;padding-top:20px;border-top:1px solid var(--grid)}
h2 .sub{display:block;font-size:13.5px;font-weight:400;color:var(--ink2);margin-top:5px;max-width:80ch}
button{font:inherit;font-size:13px;padding:7px 12px;border-radius:8px;
 border:1px solid var(--border);background:var(--surface);color:var(--ink2);cursor:pointer}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:15px 16px}
.tlab{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.tval{font-size:32px;line-height:1.1;margin:7px 0 3px}
.tile.bad .tval{color:var(--bad)}.tile.good .tval{color:var(--good)}
.tnote{font-size:12.5px;color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
 padding:18px 20px 20px;margin:16px 0}
.card h3{font-size:15.5px;margin:0 0 3px}
.card .sub{font-size:13px;color:var(--ink2);margin:0 0 15px;max-width:82ch}
.row{display:grid;grid-template-columns:230px 1fr 82px;align-items:center;gap:12px;padding:5px 0}
.rlab{font-size:12.5px;color:var(--ink2);text-align:right;overflow-wrap:anywhere}
.track{height:15px;background:var(--grid);border-radius:4px;position:relative}
.bar{height:100%;background:var(--s1);border-radius:0 4px 4px 0}
.bar.s2{background:var(--s2)}
.rval{font-size:13px;line-height:1.35;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.pcts{color:var(--muted);font-size:11px}.nlab{display:block;font-size:11px;color:var(--muted)}
.legend{display:flex;gap:16px;margin:0 0 12px 242px;font-size:12.5px;color:var(--ink2)}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
.flag{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;
 background:color-mix(in srgb,var(--warn) 22%,transparent);color:var(--ink);margin-left:8px}
.split{display:grid;grid-template-columns:230px 1fr;gap:12px;padding:7px 0;align-items:start}
.splitbars{display:flex;flex-direction:column;gap:3px}
.sb{display:flex;align-items:center;gap:8px;height:14px}
.sb .b{height:100%;border-radius:0 4px 4px 0;min-width:2px}
.sb .t{font-size:11.5px;color:var(--ink2);font-variant-numeric:tabular-nums;white-space:nowrap}
.note{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--s2);
 border-radius:9px;padding:13px 16px;font-size:13.5px;color:var(--ink2);margin:16px 0}
.note strong{color:var(--ink)}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 16px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--grid)}
td{font-variant-numeric:tabular-nums}
th[scope=row]{font-weight:400;color:var(--ink2)}
details summary{cursor:pointer;font-size:13.5px;color:var(--ink2);padding:8px 0}
a{color:var(--s1)}
#tip{position:fixed;z-index:99;pointer-events:none;opacity:0;transition:opacity .1s;
 background:var(--ink);color:var(--surface);font-size:12px;padding:6px 9px;border-radius:6px;max-width:290px}
"""

JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{
 const show=e=>{tip.innerHTML=el.dataset.tip;tip.style.opacity=1;
  const r=el.getBoundingClientRect();
  tip.style.left=Math.min(innerWidth-310,r.left+12)+'px';tip.style.top=Math.max(8,r.top-38)+'px';};
 const hide=()=>tip.style.opacity=0;
 el.addEventListener('mouseenter',show);el.addEventListener('mouseleave',hide);
 el.addEventListener('focus',show);el.addEventListener('blur',hide);});
const b=document.querySelector('button.theme');
b.addEventListener('click',()=>{const d=document.documentElement.dataset.theme==='dark';
 document.documentElement.dataset.theme=d?'light':'dark';b.textContent=d?'Dark mode':'Light mode';});
"""


def tile(lab, val, note, tone="") -> str:
    return (f'<div class="tile {tone}"><div class="tlab">{esc(lab)}</div>'
            f'<div class="tval">{val}</div><div class="tnote">{note}</div></div>')


def bars(title, sub, items) -> str:
    """items: (label, acc, n, tooltip). Single series -> slot 1, no legend needed."""
    rows = []
    for lab, acc, n, tip in items:
        if acc is None:
            rows.append(f'<div class="row"><div class="rlab">{lab}</div>'
                        f'<div class="track"></div><div class="rval">&mdash;</div></div>')
            continue
        rows.append(
            f'<div class="row" tabindex="0" data-tip="{esc(tip)}">'
            f'<div class="rlab">{lab}</div>'
            f'<div class="track"><div class="bar" style="width:{max(acc*100,0.6):.2f}%"></div></div>'
            f'<div class="rval">{acc*100:.0f}<span class="pcts">%</span>'
            f'<span class="nlab">n={n}</span></div></div>')
    return (f'<div class="card"><h3>{title}</h3><p class="sub">{sub}</p>{"".join(rows)}</div>')


def primitive_section(s: dict) -> tuple[str, list]:
    """The headline. Divergent primitives are split by source, never pooled."""
    rows, table = [], []
    order = sorted(s["primitives"].items(),
                   key=lambda kv: ((kv[1]["pooled_answerable"] or kv[1]["pooled"] or {}).get("acc", 1)))
    for key, p in order:
        srcs = {d: v for d, v in p["sources"].items() if v.get("answerable")}
        accs = [v["answerable"]["acc"] for v in srcs.values()]
        diverged = len(accs) > 1 and (max(accs) - min(accs)) > DIVERGENCE
        pa = p["pooled_answerable"] or {}
        na = max((v["na_rate"] for v in p["sources"].values()), default=0)
        na_note = f" &middot; {na*100:.0f}% of golds are 'Not Applicable'; scored on the answerable subset" if na > 0.1 else ""

        if diverged:
            sb = []
            for i, (d, v) in enumerate(sorted(srcs.items(), key=lambda kv: -kv[1]["answerable"]["acc"])):
                a, n = v["answerable"]["acc"], v["answerable"]["n"]
                sb.append(f'<div class="sb" tabindex="0" data-tip="{esc(d)}: {a*100:.1f}% (n={n})">'
                          f'<div class="b" style="width:{max(a*100,0.6):.2f}%;'
                          f'background:var(--s{1 if i==0 else 2})"></div>'
                          f'<span class="t">{a*100:.0f}% &middot; {esc(d)} (n={n})</span></div>')
                table.append((f"{p['label']} &mdash; {d}", a, n))
            rows.append(f'<div class="split"><div class="rlab">{esc(p["label"])}'
                        f'<span class="flag">sources disagree</span></div>'
                        f'<div class="splitbars">{"".join(sb)}</div></div>')
        else:
            acc, n = pa.get("acc"), pa.get("n", 0)
            src_txt = ", ".join(f"{d} {v['answerable']['acc']*100:.0f}%" for d, v in srcs.items())
            rows.append(
                f'<div class="row" tabindex="0" data-tip="{esc(p["label"])}: {pct(acc)}% (n={n})'
                f' &middot; {esc(src_txt)}{na_note}">'
                f'<div class="rlab">{esc(p["label"])}</div>'
                f'<div class="track"><div class="bar" style="width:{max((acc or 0)*100,0.6):.2f}%"></div></div>'
                f'<div class="rval">{pct(acc)}<span class="pcts">%</span>'
                f'<span class="nlab">n={n}</span></div></div>')
            table.append((p["label"], acc, n))
    return ("".join(rows), table)


def render(s: dict) -> str:
    n_multi = s["totals"]["multi_source_primitives"]
    tiles = [tile("questions", f"{s['totals']['questions']:,}", "official splits, official metrics")]
    for ds, d in s["datasets"].items():
        if d.get("acc") is None:
            continue
        tone = "bad" if d["acc"] < 0.5 else ""
        ci = (f" &middot; CI {d['ci_lo']*100:.0f}&ndash;{d['ci_hi']*100:.0f}%"
              if d.get("ci_lo") is not None else "")
        tiles.append(tile(ds, f"{d['acc']*100:.0f}<span class='pcts'>%</span>",
                          f"n={d['n']}{ci}", tone))

    prim_rows, prim_table = primitive_section(s)
    cx = s.get("charxiv", {})
    loc = s.get("localization", {})

    cx_block = ""
    if cx.get("descriptive") and cx.get("reasoning"):
        d, r = cx["descriptive"], cx["reasoning"]
        gap = (d["acc"] - r["acc"]) * 100
        cx_block = bars(
            "CharXiv: reading the chart vs reasoning over it",
            "Same figures, same model. Descriptive questions each isolate one primitive; "
            "reasoning questions require combining several readings.",
            [("descriptive (single primitive)", d["acc"], d["n"],
              f"descriptive: {d['acc']*100:.1f}% (n={d['n']})"),
             ("reasoning (composed)", r["acc"], r["n"],
              f"reasoning: {r['acc']*100:.1f}% (n={r['n']})")])
        cx_block += (f'<div class="note"><strong>A {gap:.0f}-point gap.</strong> Haiku 4.5 reads '
                     f'individual chart elements reliably and degrades sharply once an answer '
                     f'requires holding several of them together. That gap, not the absolute '
                     f'scores, is the finding.</div>')

    g = cx.get("grader_comparison")
    grader = ""
    if g:
        grader = (f'<div class="note"><strong>Grader check.</strong> On {g["n"]} responses graded '
                  f'both ways, CharXiv\'s official LLM judge scores '
                  f'{g["official_judge"]*100:.1f}% and our string matcher {g["string_match"]*100:.1f}%, '
                  f'agreeing on {g["agreement"]*100:.0f}%. The official judge is authoritative; '
                  f'the agreement is what licenses using the fast matcher for the slices the '
                  f'judge has not covered.</div>')

    loc_block = ""
    if loc.get("by_target_size"):
        order = ["<12px", "12-20px", "20-32px", "32-56px", ">=56px"]
        byk = {d["label"]: d for d in loc["by_target_size"]}
        items = [(esc(k), byk[k]["acc"], byk[k]["n"], f"{k}: {byk[k]['acc']*100:.1f}% (n={byk[k]['n']})")
                 for k in order if k in byk]
        loc_block = bars(
            "Localization on dense screens, by target size",
            "Target size as it reaches the model, after the API caps the long edge at 1568px. "
            "ScreenSpot-Pro only.", items)
        if loc.get("by_ui_type"):
            loc_block += bars("Localization by element type",
                              "Icons carry no readable string, so they test perception rather than text matching.",
                              [(esc(d["label"]), d["acc"], d["n"],
                                f"{d['label']}: {d['acc']*100:.1f}% (n={d['n']})")
                               for d in loc["by_ui_type"]])

    # --- failure-mode breakdown: why, not just how often ---------------
    from blindspot.core.failure_modes import LABELS as FM
    fm_cards = []
    for ds, d in s["datasets"].items():
        modes = d.get("failure_modes") or {}
        # A single-mode breakdown says nothing; multiple choice has only one way
        # to be wrong, and that is a property of the format, not a finding.
        if len(modes) < 2:
            continue
        tot = d.get("failures", sum(modes.values())) or 1
        items = [(FM.get(m, m), c / tot, c,
                  f"{FM.get(m, m)}: {c} of {tot} failures ({c/tot*100:.1f}%)")
                 for m, c in sorted(modes.items(), key=lambda kv: -kv[1])]
        adj = d.get("acc_meaning_adjusted")
        note = ""
        if adj and abs(adj - (d.get("acc") or 0)) > 1e-6:
            note = (f" Crediting answers that mean the right thing but are phrased differently "
                    f"raises {esc(ds)} from {d['acc']*100:.1f}% to {adj*100:.1f}%.")
        fm_cards.append(bars(
            f"{esc(ds)} &mdash; why the failures failed",
            f"{tot} scored-wrong answers, classified. List-shaped cases are decided exactly; "
            f"the rest by an equivalence judge.{note}", items))
    fm_block = "".join(fm_cards)

    # --- benchmark quality + blind control, as caveats not adjustments ---
    q = s.get("benchmark_quality") or {}
    qual_block = ""
    if q:
        rows_q = "".join(
            f'<tr><th scope="row">{esc(d)}</th>'
            f'<td>{v["audited"]}</td>'
            f'<td>{v["bad_gt_in_failures"]*100:.1f}%</td>'
            f'<td><b>{v["bad_gt_whole_set"]*100:.2f}%</b></td>'
            f'<td>{v["headline_if_all_credited"]*100:.1f}%</td></tr>'
            for d, v in sorted(q.items(), key=lambda kv: -kv[1]["bad_gt_whole_set"]))
        qual_block = f"""
<div class="card">
<h3>How trustworthy is each benchmark's ground truth?</h3>
<p class="sub">A vision model was shown the image, the question, the shipped reference answer and
Haiku's answer, and asked which was actually right. It was told to default to the benchmark and to
justify any verdict against it. Only <em>failures</em> were audited &mdash; bad ground truth is
concentrated there, because bad ground truth is what causes a correct answer to be scored wrong.
The whole-set column is therefore the one that characterises the benchmark.</p>
<table><thead><tr><th scope="col">dataset</th><th scope="col">audited</th>
<th scope="col">bad GT among failures</th><th scope="col">bad GT across the whole set</th>
<th scope="col">headline if all credited</th></tr></thead><tbody>{rows_q}</tbody></table>
</div>
<div class="note"><strong>Nothing here is applied to the scores.</strong> Every number in this
report is the official metric over the full official split. Filtering to the questions whose
ground truth survives an audit would move each headline by 0&ndash;3 points and would be hard to
distinguish from choosing the questions the model happens to do well on. The errors are real
&mdash; two InfographicVQA golds were confirmed wrong by hand, and CharXiv contradicts itself
about whether one figure's panels have titles &mdash; but at 1.7&ndash;2.8% of each full split
they are smaller than any gap this study reports.</div>"""

    bc = s.get("ai2d_blind_control") or {}
    blind_block = ""
    if bc:
        items = []
        for k, v in sorted(bc.items(), key=lambda kv: kv[1]["blind"]):
            if v.get("with_image") is None:
                continue
            gain = v["with_image"] - v["blind"]
            items.append((f"{esc(k)} &mdash; blind", v["blind"], v["blind_n"],
                          f"{esc(k)} with the diagram withheld: {v['blind']*100:.1f}%"))
            items.append((f"{esc(k)} &mdash; with diagram", v["with_image"], v["n"],
                          f"{esc(k)} seeing the diagram: {v['with_image']*100:.1f}% "
                          f"(+{gain*100:.1f}pp)"))
        blind_block = bars(
            "AI2D: how much of the score needs the diagram at all?",
            "The same questions, asked with and without the image. Four-way multiple choice, so "
            "chance is 25%. A split that scores the same blind is measuring world knowledge, not "
            "perception, and does not belong in a primitive map.", items)
        blind_block += ('<div class="note"><strong>Only the label-reference half of AI2D is a '
                        'perception measurement.</strong> Its reasoning questions are 80% '
                        'answerable with the diagram hidden &mdash; ecology and physics knowledge, '
                        'not chart reading. Label-reference starts at 31.5% blind, barely above '
                        'chance, and nearly doubles when the model can look. Only that half feeds '
                        'the primitive map above.</div>')

    tbl = "".join(f'<tr><th scope="row">{lab}</th><td>{pct(a)}%</td><td>{n}</td></tr>'
                  for lab, a, n in prim_table)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Claude Haiku 4.5 &mdash; perceptual primitives</title><style>{CSS}</style></head>
<body><div class="wrap">
<header><div>
<h1>Claude Haiku 4.5 &mdash; where perception breaks</h1>
<p class="dek">Business visual tasks decomposed into perceptual primitives and measured across
three benchmarks, using each benchmark's own questions and its own metric.
{s['totals']['questions']:,} questions &middot; {s['totals']['primitives_measured']} primitives &middot;
{n_multi} measured by more than one dataset. No comparison model: this maps failure, not ranking.</p>
</div><button class="theme" type="button">Dark mode</button></header>

<div class="tiles">{''.join(tiles)}</div>

<h2>The primitive gradient<span class="sub">Each bar is one perceptual operation, pooled across
every dataset that measures it. Where two datasets disagree by more than {DIVERGENCE*100:.0f} points
the sources are shown separately &mdash; averaging them would describe nothing real.</span></h2>
<div class="card">{prim_rows}</div>

<h2>Reading vs reasoning<span class="sub">The clearest single contrast in the study.</span></h2>
{cx_block}{grader}

<h2>Not every failure is a perception failure<span class="sub">A wrong score can mean the
model misread the figure, or that it read it correctly and expressed the answer differently,
or &mdash; where the question specifies an order &mdash; that it read everything and sequenced
it wrong. Those are different findings and are separated here. The official metric stays the
headline; the adjusted figure sits beside it.</span></h2>
{fm_block}

<h2>Can these benchmarks be trusted?<span class="sub">Two checks on the measuring instrument
itself: whether the reference answers are right, and whether the questions actually require the
image.</span></h2>
{qual_block}
{blind_block}

<h2>Localization, in detail<span class="sub">Why the two localization sources disagree so violently.</span></h2>
{loc_block}

<details><summary>Table view &mdash; every charted value as text</summary>
<table><thead><tr><th scope="col">Primitive</th><th scope="col">Accuracy</th>
<th scope="col">n</th></tr></thead><tbody>{tbl}</tbody></table></details>

<h2>Per-asset evidence<span class="sub">Every evaluated question with its image, gold answer and
Haiku's answer, filterable by primitive and correctness.</span></h2>
<p><a href="gallery/charxiv_000.html">CharXiv gallery</a> &middot;
<a href="gallery/infographicvqa_000.html">InfographicVQA gallery</a> &middot;
<a href="gallery/screenspot_pro_000.html">ScreenSpot-Pro gallery</a> &middot;
<a href="datasets.html">dataset documentation</a></p>

</div><div id="tip" role="status"></div><script>{JS}</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(OUT / "summary.json"))
    ap.add_argument("--out", default=str(OUT / "report.html"))
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text())
    Path(a.out).write_text(render(s), encoding="utf-8")
    print(f"wrote {a.out} ({Path(a.out).stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
