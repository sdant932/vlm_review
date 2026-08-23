"""outputs/aug22/report.html -- the corrected headline report.

Written fresh rather than patched onto the original renderer for two reasons:
the control arms have no place in the old schema, and the original report is
kept unmodified so the before/after is auditable.

Design rule throughout: an accuracy number alone is not a finding. Every headline
is shown next to the control that says what it means -- the blind score beside
the sighted score, the coarse-grid accuracy beside the exact one, the as-scored
F1 beside the format-corrected one. Where a number was retracted during the
study, the retraction is printed, not quietly dropped.

CSS custom properties are declared on :root only. Declaring them on a wrapper
class shipped a black-on-black report earlier in this project -- custom
properties inherit downward, so `body` never saw them.
"""
from __future__ import annotations

import argparse, html, json
from pathlib import Path

OUT = Path("outputs/aug22")

CSS = """
:root{
  --bg:#0f1116; --panel:#171a21; --panel2:#1d2028; --ink:#e8eaed; --muted:#9aa0aa;
  --line:#2a2f3a; --good:#0ca30c; --bad:#d03b3b; --warn:#d68a1e; --accent:#5b8def;
  --chip:#232833;
}
@media (prefers-color-scheme: light){
  :root{ --bg:#f7f8fa; --panel:#fff; --panel2:#f0f2f6; --ink:#15181d; --muted:#5c636e;
         --line:#dfe3ea; --chip:#eceff4; }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:27px;margin:0 0 6px} h2{font-size:20px;margin:38px 0 12px;padding-bottom:7px;
   border-bottom:1px solid var(--line)} h3{font-size:16px;margin:22px 0 8px;color:var(--ink)}
p{color:var(--ink)} .sub{color:var(--muted);margin:0 0 22px}
nav{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0 26px}
nav a{background:var(--chip);color:var(--ink);text-decoration:none;padding:6px 12px;
      border-radius:7px;font-size:13px;border:1px solid var(--line)}
nav a:hover{border-color:var(--accent)}
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
.good{color:var(--good)} .bad{color:var(--bad)} .warn{color:var(--warn)}
.callout{background:var(--panel);border-left:3px solid var(--accent);padding:13px 16px;
         border-radius:0 9px 9px 0;margin:14px 0}
.callout.bad{border-left-color:var(--bad)} .callout.good{border-left-color:var(--good)}
.callout.warn{border-left-color:var(--warn)}
.bar{height:9px;background:var(--panel2);border-radius:5px;overflow:hidden;min-width:90px}
.bar>i{display:block;height:100%;background:var(--accent)}
.badge{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;
       font-weight:600;background:var(--chip);border:1px solid var(--line)}
.badge.p{color:var(--good);border-color:var(--good)}
.badge.r{color:var(--bad);border-color:var(--bad)}
code{background:var(--chip);padding:1px 5px;border-radius:4px;font-size:13px}
"""


def esc(s): return html.escape(str(s))
def pct(v, d=1): return "--" if v is None else f"{v*100:.{d}f}%"
def f1(v, d=1): return "--" if v is None else f"{v:.{d}f}"


def bar(frac, tone=""):
    w = max(0.0, min(1.0, frac or 0)) * 100
    col = {"good": "var(--good)", "bad": "var(--bad)", "warn": "var(--warn)"}.get(tone, "var(--accent)")
    return f'<div class="bar"><i style="width:{w:.1f}%;background:{col}"></i></div>'


def render(s: dict) -> str:
    c = s.get("controls", {})
    ds = s["datasets"]
    P = []
    A = P.append

    A(f'<!doctype html><html><head><meta charset="utf-8">'
      f'<meta name="viewport" content="width=device-width,initial-scale=1">'
      f'<title>Haiku 4.5 perception blind spots -- corrected report</title>'
      f'<style>{CSS}</style></head><body><div class="wrap">')
    A("<h1>Claude Haiku 4.5 -- perception blind spots</h1>")
    A(f'<p class="sub">Corrected report, {esc(s.get("generated",""))} &middot; '
      f'<code>{esc(s["model"])}</code> &middot; thinking enabled (2000 tokens) &middot; '
      f'{s["totals"]["questions"]:,} questions across {len(ds)} benchmark arms plus 4 control ablations</p>')
    A('<nav>'
      '<a href="drilldown.html">Drill-down (top&rarr;bottom)</a>'
      '<a href="slidevqa.html">SlideVQA explorer</a>'
      '<a href="causes/index.html">Per-cause evidence</a>'
      '<a href="summary.json">summary.json</a>'
      '<a href="../datasets.html">Dataset documentation</a>'
      '<a href="../gallery/">Galleries</a></nav>')

    # ---------- what changed --------------------------------------------
    dup = c.get("duplication", {})
    dupline = ", ".join(f"{k} {v['duplicate_lines']}" for k, v in dup.items() if v["duplicate_lines"])
    A('<div class="callout warn"><b>What changed in this rebuild.</b> '
      'Resumed runs appended rather than replaced, leaving duplicate lines '
      f'({esc(dupline)}). The aggregate loader always collapsed these by uid, but '
      'ad-hoc analyses during the session did not, so some interim figures were quoted '
      'with inflated <i>n</i>. Every number below is deduplicated. SlideVQA is now '
      'included (it was missing from the original dataset list, silently omitting '
      '1,497 questions), and the four control ablations are reported alongside the '
      'accuracies they explain.</div>')

    A('<div class="callout bad"><b>Retracted.</b> An interim claim that arithmetic '
      'costs ~6&times; what retrieval costs on SlideVQA does not survive. Roughly half '
      'that gap is a metric artifact: token-F1 scores <code>22%</code> against a gold of '
      '<code>22</code> as zero. Format-corrected, the three costs are comparable. Both '
      'columns are shown below.</div>')

    # ---------- headline -------------------------------------------------
    A("<h2>Headline accuracy</h2>")
    A('<table><tr><th>Benchmark</th><th class="num">n</th><th>Metric</th>'
      '<th class="num">Score</th><th style="width:180px"></th><th class="num">95% CI</th></tr>')
    METRIC = {"charxiv": "string match / official judge", "infographicvqa": "ANLS",
              "screenspot_pro": "click-in-bbox", "ai2d": "MC accuracy",
              "slidevqa": "token F1 (evidence pages)", "slidevqa_allpages": "token F1 (all 20 slides)"}
    for k, d in ds.items():
        if d.get("acc") is None:
            continue
        tone = "bad" if d["acc"] < 0.35 else "good" if d["acc"] > 0.75 else "warn"
        ci = (f'{d["ci_lo"]*100:.1f}-{d["ci_hi"]*100:.1f}'
              if d.get("ci_lo") is not None else "--")
        A(f'<tr><td><b>{esc(k)}</b></td><td class="num">{d["n"]:,}</td>'
          f'<td>{esc(METRIC.get(k,""))}</td>'
          f'<td class="num {tone}"><b>{pct(d["acc"],2)}</b></td>'
          f'<td>{bar(d["acc"], tone)}</td><td class="num">{ci}</td></tr>')
    A("</table>")

    rp = c.get("reproducibility", {})
    if rp.get("disagreement_rate") is not None:
        A(f'<div class="callout"><b>Noise floor.</b> {rp["repeated_items"]:,} CharXiv items were '
          f'answered twice by the resumed runs -- same question, same settings. '
          f'<b>{pct(rp["disagreement_rate"])}</b> returned a different answer the second time. '
          'That is a measured reproducibility floor on real repeated trials: single-item '
          'differences below roughly this size are not interpretable.</div>')

    # ---------- blind control -------------------------------------------
    A("<h2>How much of each score is actually vision?</h2>")
    A('<p>The same questions, asked with the image withheld. Whatever survives was '
      'never a perception task -- it was recoverable from the question text and world '
      'knowledge. This is the ceiling that must be subtracted before calling a number "vision".</p>')
    A('<table><tr><th>Benchmark</th><th class="num">Blind</th><th class="num">Sighted</th>'
      '<th class="num">Vision adds</th><th style="width:180px"></th><th class="num">Chance</th><th class="num">n</th></tr>')
    for k, v in sorted(c.get("blind", {}).items(), key=lambda x: -x[1]["vision_adds_pp"]):
        tone = "bad" if v["vision_adds_pp"] < 20 else "good"
        A(f'<tr><td><b>{esc(k)}</b></td><td class="num">{pct(v["blind"])}</td>'
          f'<td class="num">{pct(v["sighted"])}</td>'
          f'<td class="num {tone}"><b>{v["vision_adds_pp"]:.1f}pp</b></td>'
          f'<td>{bar(v["vision_adds_pp"]/100, tone)}</td>'
          f'<td class="num">{pct(v["chance"],0)}</td><td class="num">{v["n"]}</td></tr>')
    A("</table>")
    cs = c.get("blind_charxiv_split", {})
    if cs:
        A("<h3>CharXiv, by split</h3><table><tr><th>Split</th><th class='num'>Blind</th>"
          "<th class='num'>Sighted</th><th class='num'>Vision adds</th><th class='num'>n</th></tr>")
        for k, v in cs.items():
            A(f'<tr><td>{esc(k)}</td><td class="num">{pct(v["blind"])}</td>'
              f'<td class="num">{pct(v["sighted"])}</td>'
              f'<td class="num good">{(v["sighted"]-v["blind"])*100:.1f}pp</td>'
              f'<td class="num">{v["n"]}</td></tr>')
        A("</table>")
    b = c.get("blind", {})
    if "ai2d" in b:
        v = b["ai2d"]
        above = (v["sighted"] - v["chance"]) * 100
        A(f'<div class="callout bad"><b>AI2D is largely not a vision benchmark.</b> Of the '
          f'{above:.0f} points it scores above chance, only <b>{v["vision_adds_pp"]:.1f}</b> come from '
          'seeing the diagram. Its headline overstates diagram perception by roughly 3&times;. '
          'CharXiv is the opposite and is the most trustworthy arm here.</div>')

    # ---------- localization ---------------------------------------------
    cl = c.get("coarse_localization", {})
    if cl:
        A("<h2>Localization: not blind, imprecise</h2>")
        A('<p>ScreenSpot-Pro scores near zero on exact clicks. But bucketing the '
          '<i>same</i> predictions by how much precision you demand shows a smooth falloff, '
          'not an absence of perception.</p>')
        A('<table><tr><th>Granularity</th><th class="num">Accuracy</th><th style="width:200px"></th>'
          '<th class="num">Chance</th><th class="num">Above chance</th></tr>')
        for lab, g in cl["grids"].items():
            A(f'<tr><td>{esc(lab)} grid</td><td class="num"><b>{pct(g["acc"])}</b></td>'
              f'<td>{bar(g["acc"])}</td><td class="num">{pct(g["chance"])}</td>'
              f'<td class="num">{g["acc"]/g["chance"]:.1f}&times;</td></tr>')
        ex, ar = cl["exact_click_in_bbox"], cl["mean_target_area_frac"]
        A(f'<tr><td><b>exact click-in-bbox</b></td><td class="num bad"><b>{pct(ex,2)}</b></td>'
          f'<td>{bar(ex,"bad")}</td><td class="num">{pct(ar,3)}</td>'
          f'<td class="num">{ex/ar:.0f}&times;</td></tr></table>')
        A(f'<p class="sub">Mean target occupies {pct(ar,3)} of the screen (n={cl["n"]:,}).</p>')

    g = c.get("grid_control")
    if g:
        A("<h3>Is it perception, or coordinate emission?</h3>")
        A('<p>Same items, same 4&times;4 granularity; only the answer format differs. '
          'The model either names a cell ("B3") or emits coordinates that we bucket.</p>')
        A('<table><tr><th>Condition</th><th class="num">Accuracy</th><th style="width:200px"></th></tr>'
          f'<tr><td><b>names the cell</b></td><td class="num good"><b>{pct(g["named_cell_acc"])}</b></td>'
          f'<td>{bar(g["named_cell_acc"],"good")}</td></tr>'
          f'<tr><td>clicks, bucketed to same cell</td><td class="num">{pct(g["click_derived_cell_acc"])}</td>'
          f'<td>{bar(g["click_derived_cell_acc"])}</td></tr>'
          f'<tr><td>chance</td><td class="num">{pct(g["chance"])}</td><td>{bar(g["chance"])}</td></tr></table>')
        p = g.get("mcnemar_p")
        A(f'<div class="callout"><b>Both are true, in proportion.</b> Naming beats pointing by '
          f'<b>{g["delta_pp"]:+.1f}pp</b> (McNemar {g["mcnemar_b"]}/{g["mcnemar_c"]} discordant, '
          f'p={p:.2g}) -- coordinate emission is genuinely lossy and a UI agent using element '
          f'selection rather than raw pixels would recover it. But even with coordinates removed, '
          f'accuracy is {pct(g["named_cell_acc"])} at 4&times;4. The majority of the deficit is '
          'perceptual, and no output format rescues it.</div>')

    # ---------- slidevqa costs -------------------------------------------
    sc = c.get("slidevqa_costs")
    if sc:
        A("<h2>SlideVQA: what does each operation actually cost?</h2>")
        A(f'<p>Retrieval is measured paired on n={sc["paired_n"]} questions present in both '
          'conditions. Format-corrected columns neutralise the token-F1 unit artifact.</p>')
        A('<table><tr><th>Operation</th><th class="num">As scored</th>'
          '<th class="num">Format-corrected</th><th>What it means</th></tr>')
        rows = [("retrieval", "find the right slide among 20"),
                ("integration", "combine 2+ slides"),
                ("derivation", "compute on what was read")]
        for k, note in rows:
            a_, b_ = sc["as_scored"][k], sc["format_corrected"][k]
            A(f'<tr><td><b>{esc(k)}</b></td><td class="num">{a_:+.1f}</td>'
              f'<td class="num"><b>{b_:+.1f}</b></td><td class="sub">{esc(note)}</td></tr>')
        A("</table>")
        A('<table><tr><th>Slice</th><th class="num">As scored F1</th><th class="num">Format-corrected F1</th></tr>')
        for k, lab in (("overall_f1", "overall"), ("lookup_f1", "plain lookup"),
                       ("arithmetic_f1", "needs arithmetic"), ("single_page_f1", "single-page evidence"),
                       ("multi_page_f1", "multi-page evidence")):
            A(f'<tr><td>{esc(lab)}</td><td class="num">{f1(sc["as_scored"][k])}</td>'
              f'<td class="num">{f1(sc["format_corrected"][k])}</td></tr>')
        A("</table>")

    op = c.get("onepage")
    if op:
        A("<h3>Are the multi-page questions really multi-page?</h3>")
        A('<p>Give the model only one of the two annotated evidence slides. If accuracy holds, '
          'the "multi-hop" label was a dataset artifact.</p>')
        A(f'<table><tr><th>Condition</th><th class="num">F1</th><th style="width:200px"></th></tr>'
          f'<tr><td>both evidence slides</td><td class="num">{f1(op["both_slides_f1"])}</td>'
          f'<td>{bar(op["both_slides_f1"]/100)}</td></tr>'
          f'<tr><td><b>only one of the two</b></td><td class="num bad"><b>{f1(op["one_slide_f1"])}</b></td>'
          f'<td>{bar(op["one_slide_f1"]/100,"bad")}</td></tr></table>')
        A(f'<div class="callout good"><b>Genuinely multi-page, and integration is a strength.</b> '
          f'Removing one slide costs <b>{op["collapse_f1"]:.1f} F1</b>; only '
          f'{pct(op["still_answerable_frac"])} stay answerable. The information really is '
          'distributed -- yet combining two slides costs only ~4-6 F1. These are '
          '<i>bridge questions</i>: the target is identified by a property on a different '
          'slide ("the Trading Operating Profit in the year Nestl&eacute; achieved the third '
          'largest Organic Growth"). Haiku handles that indirection well.</div>')

    # ---------- abstention ------------------------------------------------
    ab = c.get("abstention")
    if ab and ab.get("gold_na_n"):
        A("<h2>When the thing is not there</h2>")
        A(f'<p>{ab["gold_na_n"]:,} CharXiv questions have "Not Applicable" as the gold answer -- '
          'the chart genuinely has no legend, no second axis, no intersecting lines.</p>')
        A('<div class="tiles">'
          f'<div class="tile"><div class="l">Correctly abstained</div>'
          f'<div class="v good">{pct(ab["correctly_abstained"])}</div>'
          f'<div class="n">n={ab["gold_na_n"]:,}</div></div>'
          f'<div class="tile"><div class="l">Invented a value</div>'
          f'<div class="v bad">{pct(ab["invented_a_value"])}</div>'
          f'<div class="n">confident fabrication</div></div>'
          f'<div class="tile"><div class="l">Over-abstained</div>'
          f'<div class="v">{pct(ab["over_abstained"])}</div>'
          f'<div class="n">n={ab["gold_value_n"]:,}</div></div></div>')
        A('<p>The aggregate hides the shape of it. Absence detection is '
          '<b>structure-dependent</b>:</p>')
        A('<table><tr><th>Question</th><th class="num">n</th><th class="num">Correctly abstained</th>'
          '<th style="width:180px"></th></tr>')
        for q in ab["by_question"]:
            tone = "bad" if q["abstained"] < 0.8 else "good"
            A(f'<tr><td>{esc(q["qlabel"])}</td><td class="num">{q["n"]}</td>'
              f'<td class="num {tone}">{pct(q["abstained"])}</td>'
              f'<td>{bar(q["abstained"],tone)}</td></tr>')
        A("</table>")
        A('<div class="callout bad"><b>The worst failure mode for business use.</b> Asked how many '
          'legend entries a legend-less chart has, Haiku invents a count roughly 4 times in 10. '
          'That is a <i>detection</i> failure -- does this structure exist at all -- not a reading '
          'failure, and it produces confident fabricated output rather than an error.</div>')

    # ---------- numeric error --------------------------------------------
    ne = c.get("numeric_error", [])
    if ne:
        A("<h2>When a number is wrong, how wrong is it?</h2>")
        A('<p>If the model were misreading values off an axis, errors would cluster below 10%. '
          'They do not.</p>')
        A('<table><tr><th>Benchmark</th><th class="num">numeric n</th><th class="num">exact</th>'
          '<th class="num">median error when wrong</th><th class="num">within 10%</th>'
          '<th class="num">&gt;100%</th></tr>')
        for r in ne:
            A(f'<tr><td>{esc(r["label"])}</td><td class="num">{r["n_numeric"]:,}</td>'
              f'<td class="num">{pct(r["exact_frac"])}</td>'
              f'<td class="num bad"><b>{pct(r["median_rel_error"])}</b></td>'
              f'<td class="num">{pct(r["within_10pct_frac"])}</td>'
              f'<td class="num">{pct(r["over_100pct_frac"])}</td></tr>')
        A("</table>")
        A('<div class="callout"><b>Wrong element, not near miss.</b> A 33-90% relative error is the '
          'signature of grabbing a neighbouring bar or row -- a discrete jump to the wrong element -- '
          'not of imprecise interpolation, which would cluster under 10%. Only ~14-19% of errors '
          'land within 10%. This is consistent across four independent benchmarks, which makes it '
          'a property of the model rather than a dataset artifact.</div>')

    # ---------- format artifact ------------------------------------------
    fa = c.get("format_artifact", {})
    if fa:
        A("<h2>How much of the &quot;failure&quot; is the metric?</h2>")
        A('<table><tr><th>Benchmark</th><th>Metric</th><th class="num">hard zeros</th>'
          '<th class="num">of which format-equivalent</th></tr>')
        MET = {"slidevqa": "token F1", "charxiv": "ANLS", "infographicvqa": "ANLS"}
        for k in ("slidevqa", "charxiv", "infographicvqa"):
            v = fa.get(k)
            if not v or not v["hard_zeros"]:
                continue
            frac = v["hard_zeros_format_equivalent"] / v["hard_zeros"]
            tone = "bad" if frac > 0.2 else ""
            A(f'<tr><td><b>{esc(k)}</b></td><td>{esc(MET[k])}</td>'
              f'<td class="num">{v["hard_zeros"]:,}</td>'
              f'<td class="num {tone}"><b>{pct(frac)}</b> ({v["hard_zeros_format_equivalent"]:,})</td></tr>')
        A("</table>")
        A('<div class="callout warn"><b>Metric artifact, not a model blind spot.</b> Token-F1 scores '
          '<code>22%</code> against <code>22</code> as a flat zero; ANLS, being edit-distance based, '
          'barely notices. This is why the SlideVQA arithmetic gap needed retracting and why the '
          'ANLS-scored benchmarks did not.</div>')

    # ---------- limits ----------------------------------------------------
    A("<h2>Limits of this evaluation</h2><ul>")
    for t in [
        "<b>Single run, temperature not controllable.</b> Thinking pins temperature to 1 and the "
        "SDK exposes no override. The measured repeat-disagreement rate above is the honest noise floor.",
        "<b>CharXiv free-text types are string-scored where the official judge did not run</b>, which "
        "is a lower bound: a correct answer phrased differently scores zero. Judge-scored and "
        "string-scored cells are never averaged together.",
        "<b>Ground-truth noise sets the floor.</b> Audits put questionable gold at roughly 1.5% of "
        "CharXiv and 2.8% of InfographicVQA whole-set; 3 SlideVQA arithmetic items have expressions "
        "that do not evaluate to their own annotated answer.",
        "<b>ScreenSpot-Pro conflates perception with coordinate emission</b> -- quantified above, "
        "but the grid control covers 350 of 1,581 items, not all.",
        "<b>Blind control samples 500 per benchmark</b>, not the full splits.",
        "<b>Haiku downscales images to ~1568px long edge</b> regardless of what is sent, so no result "
        "here speaks to native-resolution performance. A prior ablation confirmed pre-downscaling "
        "changes nothing.",
        "<b>Benchmarks measure what they measure.</b> The blind control shows AI2D is substantially "
        "a language task; conclusions about diagram perception should lean on CharXiv.",
    ]:
        A(f"<li>{t}</li>")
    A("</ul>")

    A('<p class="sub" style="margin-top:34px">Generated by <code>blindspot.reporting.aug22_report</code>. '
      'Numbers recomputed from <code>results/*.jsonl</code>; see <code>summary.json</code> '
      'and <code>drilldown.csv</code> to verify any figure outside the browser.</p>')
    A("</div></body></html>")
    return "\n".join(P)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(OUT / "summary.json"))
    ap.add_argument("--out", default=str(OUT / "report.html"))
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text())
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(s))
    print(f"wrote {p} ({p.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
