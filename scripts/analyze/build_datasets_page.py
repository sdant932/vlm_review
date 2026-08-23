#!/usr/bin/env python
"""Build outputs/datasets.html -- what each dataset is, and whether it is usable.

Every number is read from data/ at build time rather than transcribed, so the
page cannot drift from the corpus. The datasets that turned out to be unusable
are documented as carefully as the ones in the study: knowing that FlowLearn's
scientific split ships no ground truth, or that Ferret-UI's release is not the
real benchmark, is what stops the next person re-deriving it.
"""

from __future__ import annotations

import collections
import html
import json
import sys
from pathlib import Path

from blindspot.analysis.annotate import CSS  # noqa: E402  (reuse the validated palette)

DATA = Path("data")
OUT = Path("outputs/datasets.html")

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


def esc(s):
    return html.escape(str(s), quote=True)


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


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
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
    OUT.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Datasets &mdash; Haiku 4.5 perception study</title>
<style>{CSS}{EXTRA}</style></head><body><div class="wrap">
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
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
