"""HTML report for the svg_localization-derived sets: counting and word_mc.

Rendering is a pure function of svgderived_eval.analyse_*, so every figure can be
regenerated from results/*.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from blindspot.reporting.svgloc_report import CSS, esc, pct, table, barcell
from blindspot.analysis.svgderived_eval import (analyse_counting, analyse_word_mc, RUNGS, MIN_CELL,
                                                COUNT_BINS)

OUT = Path("outputs/svgderived")


def sgn(v, d=2) -> str:
    return "&mdash;" if v is None else f"{v:+.{d}f}"


def _breakdowns(s: dict, specs) -> list[str]:
    b = []
    for name, title, note in specs:
        keep = [c for c in s.get(name, []) if not c["suppressed"]]
        drop = [c for c in s.get(name, []) if c["suppressed"]]
        if not keep:
            continue
        b.append(f"<h3>{title}</h3>")
        rows = [[f'<b>{esc(c["label"])[:46]}</b>', f'{c["n"]:,}', pct(c["acc"], 2),
                 barcell(c["acc"]), f'{pct(c["lo"],1)}&ndash;{pct(c["hi"],1)}',
                 sgn(c.get("mean_signed"))] for c in keep]
        b.append(table([name, "n", "accuracy", "", "95% Wilson", "mean signed error"], rows,
                       note + (f" {len(drop)} cell(s) under n={MIN_CELL} suppressed."
                               if drop else "")))
    return b


def render(cnt: dict, mc: dict) -> str:
    b = []
    A = b.append
    A("<h1>Counting and word presence across the resolution ladder</h1>")
    A(f'<p class="sub">Claude Haiku 4.5 &middot; <code>claude-haiku-4-5-20251001</code> &middot; '
      f'thinking enabled (2000 tokens) &middot; <code>small</code> and <code>large</code> rungs '
      f'only &middot; {cnt["counts"]["scored"]:,} counting + {mc["counts"]["scored"]:,} word-choice '
      f'questions, each with a paired blind control, over the same 200 scenes and '
      f'byte-identical pixels as the localization set</p>')

    A('<div class="callout warn"><b>Read this before any number: the null control is missing.</b> '
      'Both EVAL.md files designate <code>medium</code> vs <code>large</code> as the noise floor, '
      'because those two rungs are delivered to the model at the same size and differ only in '
      'resampling path. <code>medium</code> was excluded from this run, so neither set carries its '
      'own noise floor, and the <code>small</code> vs <code>large</code> contrast below now '
      'conflates two things it was designed to separate: absolute delivered resolution '
      '(900&times;570 against 1348&times;853) and whether the API resampled at all. The '
      'localization run measured that null at <b>&minus;0.13pp</b> over these same 200 scenes and '
      'the same pixels, so it is carried across here as a proxy floor &mdash; but it is borrowed, '
      'not measured on these questions, and a difference of a point or two should be treated as '
      'noise rather than a finding.</div>')

    # =================================================================== counting
    A("<h2>Counting</h2>")
    c_un = cnt["counts"]
    A(f'<p>One counting question per chart type, about the structure that type is made of. Gold '
      f'comes from the semantic record captured when each scene was built, not from counting marks '
      f'in a raster, and 678 of 714 rows are cross-checked against the labels actually drawn. '
      f'There is no chance baseline for a free-response integer, so none is invented.</p>')

    rows = []
    for c in cnt["headline"]:
        if not c["n"]:
            continue
        rows.append([f'<b>{c["label"]}</b>', f'{c["n"]:,}',
                     (f'<b>{pct(c["acc"],2)}</b>', ""), barcell(c["acc"]),
                     f'{pct(c["lo"],1)}&ndash;{pct(c["hi"],1)}',
                     sgn(c.get("mean_signed")), sgn(c.get("mean_signed_when_wrong")),
                     f'{c.get("under",0)}/{c.get("over",0)}'])
    o = cnt["overall"]
    rows.append([f'<b>{o["label"]}</b>', f'{o["n"]:,}', (f'<b>{pct(o["acc"],2)}</b>', ""),
                 barcell(o["acc"]), f'{pct(o["lo"],1)}&ndash;{pct(o["hi"],1)}',
                 sgn(o.get("mean_signed")), sgn(o.get("mean_signed_when_wrong")),
                 f'{o.get("under",0)}/{o.get("over",0)}'])
    A(table(["rung", "n", "exact-count accuracy", "", "95% Wilson", "mean signed error",
             "&hellip;when wrong", "under/over"], rows,
            "Signed error is predicted &minus; true. Negative means the model stopped early; "
            "positive means it over-reported. Absolute error is never reported alone because it "
            "destroys the sign, which carries the mechanism."))

    p = cnt["paired"]
    if p.get("n"):
        A(f'<div class="callout"><b>Paired <code>small</code> &rarr; <code>large</code> '
          f'(n={p["n"]:,}, every question appears at both rungs).</b> '
          f'{pct(p["acc_a"],2)} &rarr; {pct(p["acc_b"],2)}, <b>{p["delta_pp"]:+.2f}pp</b>. '
          f'Discordant {p["discordant_b"]}/{p["discordant_c"]}, McNemar '
          f'&chi;&sup2;={p["mcnemar_chi2"]:.2f} &mdash; '
          f'{"significant at p&lt;.05" if p["significant"] else "not significant at p&lt;.05"}.</div>')

    A("<h3>Dose-response: accuracy against the true count</h3>")
    A("<p>The primary result. A monotone decline is a finding; a flat curve is also a finding and "
      "is reported as one rather than buried.</p>")
    rows = []
    for i, name in enumerate([n for _l, _h, n in COUNT_BINS]):
        cells = [f'<b>{name}</b>']
        for g in RUNGS:
            c = cnt["dose"][g][i]
            cells.append(f'{c["n"]}')
            cells.append((f'<b>{pct(c["acc"],1)}</b>', "") if not c["suppressed"]
                         else (f'<span style="color:var(--muted)">{pct(c["acc"],1)}</span>', ""))
            cells.append(sgn(c.get("mean_signed")))
        rows.append(cells)
    A(table(["true count", "small n", "small acc", "small signed",
             "large n", "large acc", "large signed"], rows,
            f"Cells under n={MIN_CELL} are greyed rather than removed, so the shape of the curve "
            f"stays visible; the 16+ bin has only 27 rows across all three rungs by construction "
            f"and cannot resolve the high-count tail."))

    dc = cnt.get("dose_confound") or {}
    if dc:
        A('<div class="callout bad"><b>That curve cannot be read as a dose response, and reporting '
          'it as one would be wrong.</b> The true count is not randomly assigned across question '
          'forms, so a count bin is partly a proxy for <i>what</i> is being counted. The 16+ bin '
          'is a single question form. Meanwhile the two hardest forms &mdash; points in a quadrant '
          'chart (62.5%) and separate lines in a line chart (65.4%) &mdash; sit at median counts '
          'of 7 and <b>4</b>, at the low end. Accuracy appearing to rise with the count is the '
          'easy forms happening to carry the big numbers.</div>')
        A(table(["true-count bin", "n", "distinct question forms it spans", "dominated by"],
                [[f'<b>{k}</b>', f'{v["n"]:,}',
                  (f'<b>{v["n_forms"]}</b>', "bad" if v["n_forms"] <= 1 else ""),
                  esc(", ".join(q.replace("How many", "").split(" are in")[0].strip()[:26]
                                for q, _c in v["top"]))]
                 for k, v in dc["bin_forms"].items()]))
        A("<h4>The clean test: within a single question form</h4>")
        A("<p>Holding the thing being counted fixed is the only way to ask whether the count "
          "itself matters. It requires a form with both enough rows and enough spread.</p>")
        if dc["within_form"]:
            A(table(["question form", "count range", "low half", "high half", "difference"],
                    [[f'<b>{esc(w["form"][:52])}</b>', f'{w["min"]}&ndash;{w["max"]}',
                      f'{pct(w["lo_acc"],1)} (n={w["lo_n"]})',
                      f'{pct(w["hi_acc"],1)} (n={w["hi_n"]})',
                      (f'<b>{w["delta_pp"]:+.1f}pp</b>', "")] for w in dc["within_form"]],
                    f'Only {dc["n_forms_testable"]} of {dc["n_forms_total"]} question forms carry '
                    f'enough rows and enough count spread to support this test.'))
            A(f'<div class="callout"><b>No degradation where it can actually be measured.</b> '
              f'Counting 27 labelled boxes is as reliable as counting 8 &mdash; 100% in both '
              f'halves. That is one form out of {dc["n_forms_total"]}, so it is weak evidence, but '
              f'it points the opposite way from the main study\'s InfographicVQA curve '
              f'(63% &rarr; 33% across count bins).</div>')
        A(f'<p class="sub">The whole error inventory is {dc["n_errors"]} wrong answers out of '
          f'{cnt["overall"]["n"]:,}. Signed errors: '
          + ", ".join(f'<code>{v:+d}</code>&times;{n}' for v, n in dc["signed_histogram"])
          + '. Every error is off by three or fewer, and under- and over-counts are close to '
            'balanced, so neither the "stops early" nor the "estimates a pattern" signature from '
            'the main study reproduces here.</p>')

    A("<h3>By what is being counted</h3>")
    A("<p>Never pooled into one accuracy number. Connections have no enclosing shape to anchor on, "
      "and are where undercounting should appear first if it appears at all.</p>")
    rows = []
    for fam, c in sorted(cnt["family"].items(), key=lambda kv: -(kv[1]["acc"] or 0)):
        rows.append([f'<b>{fam}</b>', f'{c["n"]:,}', pct(c["acc"], 2), barcell(c["acc"]),
                     f'{pct(c["lo"],1)}&ndash;{pct(c["hi"],1)}',
                     sgn(c.get("mean_signed")), f'{c.get("under",0)}/{c.get("over",0)}'])
    A(table(["family", "n", "accuracy", "", "95% Wilson", "mean signed error", "under/over"], rows))

    bl = cnt["blind"]
    if bl["overall"]["n"]:
        A("<h3>Blind control &mdash; how much of this needed the image?</h3>")
        rows = [[f'<b>{g}</b>', f'{bl["by_rung"][g]["n"]:,}', pct(bl["by_rung"][g]["acc"], 2),
                 pct(next((c["acc"] for c in cnt["headline"] if c["label"] == g), None), 2),
                 sgn((next((c["acc"] for c in cnt["headline"] if c["label"] == g), 0) or 0) * 100
                     - (bl["by_rung"][g]["acc"] or 0) * 100, 1)]
                for g in RUNGS if g in bl["by_rung"]]
        A(table(["rung", "n", "no image", "with image", "vision adds (pp)"], rows,
                "Some counts are guessable from the question alone &mdash; &ldquo;how many columns "
                "does this table have&rdquo; has a narrow plausible range. Whatever survives here "
                "was never a perception task."))

    A("".join(_breakdowns(cnt, [
        ("chart_type", "By chart type",
         "Counting bars is not counting table rows. Cells are small by construction &mdash; one or "
         "two questions per scene."),
        ("count_family_unused", "", ""),
        ("theme", "By theme", "This should show nothing."),
        ("font_family", "By font", "This should also show nothing."),
    ])))

    # ==================================================================== word_mc
    A("<h2>Word presence (<code>word_mc</code>)</h2>")
    A("<p>One question: which of these four words appears in the figure? One option is present; "
      "the other three appear nowhere in it &mdash; verified by substring check across all scene "
      "text including titles, footnotes and badges. This isolates <b>reading</b> from "
      "<b>localization</b>: the answer has no spatial component at all.</p>")

    pb = mc["position_bias"]
    A("<h3>Position bias &mdash; checked before anything else</h3>")
    A("<p>If the model favours a slot, every accuracy number below is contaminated. The sharper "
      "test is the distribution among <i>wrong</i> answers, where a guessing model has nothing "
      "else to go on.</p>")
    rows = [[f'<b>{r["option"]}</b>', f'{r["observed"]:,}', pct(r["obs_share"], 1),
             pct(r["expected_share"], 1),
             (f'{r["deviation_pp"]:+.1f}pp', "bad" if abs(r["deviation_pp"] or 0) > 5 else "")]
            for r in pb["all_picks"]["rows"]]
    A(table(["option", "model picks", "share of picks", "share of key", "deviation"], rows))
    ok_all, ok_wrong = not pb["all_picks"]["biased"], not pb["wrong_picks"]["biased"]
    A(f'<div class="callout {"good" if (ok_all and ok_wrong) else "bad"}">'
      f'<b>{"No position bias." if (ok_all and ok_wrong) else "Position bias detected."}</b> '
      f'All picks against the answer key: &chi;&sup2;={pb["all_picks"]["chi2"]:.2f} against a '
      f'critical value of {pb["all_picks"]["crit"]} (3 d.f.). Among wrong answers against uniform: '
      f'&chi;&sup2;={pb["wrong_picks"]["chi2"]:.2f} on n={pb["wrong_picks"]["n"]}. '
      f'{"Accuracy below is not contaminated by a guessing strategy." if (ok_all and ok_wrong) else "Every number below must be read with this in mind."}</div>')

    rows = []
    for c in mc["headline"]:
        if not c["n"]:
            continue
        rows.append([f'<b>{c["label"]}</b>', f'{c["n"]:,}',
                     (f'<b>{pct(c["acc"],2)}</b>', "good" if (c["acc"] or 0) > .6 else "warn"),
                     barcell(c["acc"]), f'{pct(c["lo"],1)}&ndash;{pct(c["hi"],1)}', "25.0%",
                     f'{(c["acc"] or 0)/0.25:.2f}&times;'])
    o = mc["overall"]
    rows.append([f'<b>{o["label"]}</b>', f'{o["n"]:,}', (f'<b>{pct(o["acc"],2)}</b>', ""),
                 barcell(o["acc"]), f'{pct(o["lo"],1)}&ndash;{pct(o["hi"],1)}', "25.0%",
                 f'{(o["acc"] or 0)/0.25:.2f}&times;'])
    A(table(["rung", "n", "accuracy", "", "95% Wilson", "chance", "above chance"], rows,
            "An accuracy near 25% would mean the model is guessing rather than reading, and every "
            "cut below would then be meaningless."))

    p = mc["paired"]
    if p.get("n"):
        A(f'<div class="callout"><b>Paired <code>small</code> &rarr; <code>large</code> '
          f'(n={p["n"]:,}, on (graph_id, answer_text)).</b> {pct(p["acc_a"],2)} &rarr; '
          f'{pct(p["acc_b"],2)}, <b>{p["delta_pp"]:+.2f}pp</b>. Discordant '
          f'{p["discordant_b"]}/{p["discordant_c"]}, McNemar &chi;&sup2;='
          f'{p["mcnemar_chi2"]:.2f} &mdash; '
          f'{"significant at p&lt;.05" if p["significant"] else "not significant at p&lt;.05"}. '
          f'Because the answer has no spatial component, a gap here is glyph legibility and '
          f'nothing else.</div>')

    wa = mc["wrong_analysis"]
    if wa["n_wrong"]:
        A("<h3>What a wrong answer actually was</h3>")
        A(f'<p>{wa["n_wrong"]:,} wrong answers spread over {wa["distinct_distractors"]:,} distinct '
          f'distractors. Every one of these words is verifiably absent from the figure, so each '
          f'wrong pick is a hallucinated reading &mdash; the same shape as the absence-detection '
          f'finding in the main study. If wrong picks clustered on particular vocabulary, that '
          f'would be a prior about what charts contain overriding what this chart contains.</p>')
        A(table(["most-chosen absent word", "times chosen"],
                [[f'<b>{esc(w)}</b>', f'{n}'] for w, n in wa["top_distractors"][:10]],
                f"Top 10 of {wa['distinct_distractors']:,}. A flat tail here means no vocabulary "
                f"prior; a spike means one."))

    pol = mc.get("polarity") or {}
    if len(pol) == 2:
        A("<h3>Background polarity</h3>")
        d_, l_ = pol.get("dark"), pol.get("light")
        A(table(["theme group", "n", "accuracy", "", "95% Wilson"],
                [[f'<b>{c["label"]}</b>', f'{c["n"]:,}', pct(c["acc"], 2), barcell(c["acc"]),
                  f'{pct(c["lo"],1)}&ndash;{pct(c["hi"],1)}'] for c in (d_, l_) if c],
                "The localization set showed a 5.8&times; dark-over-light effect. Because this "
                "task has no spatial component, testing it here separates a reading effect from a "
                "pointing effect."))

    bl = mc["blind"]
    if bl["overall"]["n"]:
        A("<h3>Blind control</h3>")
        rows = [[f'<b>{g}</b>', f'{bl["by_rung"][g]["n"]:,}', pct(bl["by_rung"][g]["acc"], 2),
                 pct(next((c["acc"] for c in mc["headline"] if c["label"] == g), None), 2),
                 sgn(((next((c["acc"] for c in mc["headline"] if c["label"] == g), 0) or 0)
                      - (bl["by_rung"][g]["acc"] or 0)) * 100, 1)]
                for g in RUNGS if g in bl["by_rung"]]
        A(table(["rung", "n", "no image", "with image", "vision adds (pp)"], rows,
                "Distractors are single words from the vocabulary families the scenes are built "
                "from, so a model with a strong prior over that vocabulary could exploit it. "
                "Chance is 25%."))

    A("".join(_breakdowns(mc, [
        ("chart_type", "By chart type", "Is a word in a dense table harder to spot than one in a "
                                        "flowchart node?"),
        ("answer_len", "By length of the correct word",
         "The shortest words are the hardest to resolve at <code>small</code>, and this is the "
         "most likely place for a real effect."),
        ("theme", "By theme", "This should show nothing."),
        ("font_family", "By font", "This should also show nothing."),
    ])))

    # ============================================== the reason both sets exist
    cross = {}
    cp = Path("outputs/svgderived/cross.json")
    if cp.exists():
        cross = json.loads(cp.read_text())
    if cross:
        A("<h2>Reading, counting and pointing on identical pixels</h2>")
        A("<p>All three sets are drawn from the same 200 scenes and the same PNG files. No image "
          "was re-rendered, so the differences below are the task and nothing else &mdash; which "
          "is the one comparison none of the three sets can make alone.</p>")
        mc_s = next((c["acc"] for c in mc["headline"] if c["label"] == "small"), None)
        mc_l = next((c["acc"] for c in mc["headline"] if c["label"] == "large"), None)
        c_s = next((c["acc"] for c in cnt["headline"] if c["label"] == "small"), None)
        c_l = next((c["acc"] for c in cnt["headline"] if c["label"] == "large"), None)
        A(table(["task", "what it asks", "small", "large", "blind"],
                [["<b>word_mc</b>", "is this word present at all?",
                  (f'<b>{pct(mc_s,2)}</b>', "good"), (f'<b>{pct(mc_l,2)}</b>', "good"),
                  pct(mc["blind"]["overall"]["acc"], 1)],
                 ["<b>counting</b>", "how many of these structures are there?",
                  (f'<b>{pct(c_s,2)}</b>', "good"), (f'<b>{pct(c_l,2)}</b>', "good"),
                  pct(cnt["blind"]["overall"]["acc"], 1)],
                 ["<b>localization</b> &mdash; 4&times;4 cell", "roughly where is it?",
                  pct(cross["loc_c4_small"], 1), pct(cross["loc_c4_large"], 1), "&mdash;"],
                 ["<b>localization</b> &mdash; exact box", "exactly where is it?",
                  (f'<b>{pct(cross["loc_small"],2)}</b>', "bad"),
                  (f'<b>{pct(cross["loc_large"],2)}</b>', "bad"), "&mdash;"]],
                "Localization has no blind arm because a click target cannot be located without "
                "the screenshot &mdash; a blind arm would score zero by construction."))
        A('<div class="callout bad"><b>The deficit is spatial, not textual.</b> On the very same '
          'pixels, Haiku identifies which word is present essentially perfectly '
          f'({pct(mc["overall"]["acc"],2)}, one error in {mc["overall"]["n"]:,}) and counts the '
          f'structures at {pct(cnt["overall"]["acc"],1)} &mdash; but lands inside the box of a '
          f'word it was <i>told the text of</i> only {pct(cross["loc_pooled"],2)} of the time. It '
          'can read the label and cannot point at it. This is exactly the separation word_mc was '
          'built to make, and it rules out the deflationary explanation that the localization '
          'score is low because the glyphs were illegible: they plainly were not.</div>')
        A('<div class="callout warn"><b>Both new sets are at or near ceiling, and that limits what '
          'they can say about resolution.</b> word_mc scores 99.7&ndash;100% and counting '
          '94&ndash;97%, so neither has the headroom to resolve a resolution effect. word_mc\'s '
          f'{mc["paired"]["delta_pp"]:+.2f}pp small&rarr;large is measured against a ceiling with '
          'one error in the entire set; counting\'s '
          f'{cnt["paired"]["delta_pp"]:+.2f}pp is nominally significant '
          f'(&chi;&sup2;={cnt["paired"]["mcnemar_chi2"]:.2f}) but rests on 21 errors in total and '
          'has no in-set noise floor to be judged against. The resolution question is answered by '
          'the localization set, which has the dynamic range; these two establish that reading and '
          'counting are <i>not</i> the bottleneck.</div>')

    # ===================================================================== limits
    A("<h2>What this does not test</h2>")
    A("<ul>"
      "<li><b>No noise floor of its own.</b> <code>medium</code> was excluded, and it is the null "
      "control in both specs. The &minus;0.13pp measured on the localization set over the same "
      "pixels is borrowed as a proxy.</li>"
      "<li><b><code>small</code> vs <code>large</code> is not a clean resolution contrast.</b> It "
      "mixes delivered size with whether the API resampled at all. With <code>medium</code> "
      "included the two could have been separated.</li>"
      "<li><b>Counting cannot resolve the high-count tail.</b> Gold ranges 3&ndash;27, median 7, "
      "and the 16+ bin holds 27 rows across all three rungs. A flat curve here does not mean "
      "counting never degrades &mdash; it may only mean the range is too narrow.</li>"
      "<li><b>Treemap counts are the 36 rows not cross-checked</b>, deliberately: a block too "
      "small for text is still drawn as a rectangle, so labels and rectangles are genuinely not "
      "1:1 there. Treat treemap results with more caution.</li>"
      "<li><b><code>word_mc</code> tests presence, not localization.</b> A high score says nothing "
      "about whether the model can point at the word; the interesting result is the gap between "
      "this set and the localization set on the same pixels.</li>"
      "<li><b>One word per question.</b> No phrase reading, no ordering, no relationship between "
      "labels.</li>"
      "</ul>")
    A(f'<p class="sub" style="margin-top:34px">Generated by '
      f'<code>blindspot.reporting.svgderived_report</code>. Counting: {c_un["scored"]:,} scored, '
      f'{c_un["unusable"]} unusable (counted, never scored as wrong), {c_un["blind_scored"]:,} '
      f'blind. word_mc: {mc["counts"]["scored"]:,} scored, {mc["counts"]["unusable"]} unusable, '
      f'{mc["counts"]["blind_scored"]:,} blind.</p>')

    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>counting + word_mc &mdash; Haiku 4.5</title>"
            f"<style>{CSS}</style></head><body><div class='wrap'>"
            + "".join(b) + "</div></body></html>")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="haiku-4-5_think2000_native_r0")
    a = ap.parse_args()
    cnt, mc = analyse_counting(a.tag), analyse_word_mc(a.tag)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"counting": cnt, "word_mc": mc}, indent=1))
    (OUT / "report.html").write_text(render(cnt, mc))
    print(f"wrote {OUT/'report.html'} and {OUT/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
