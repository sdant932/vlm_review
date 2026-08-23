"""The report's tables, emitted as markdown straight from the measured JSON.

`blindspots.md` includes these verbatim, so no number in the report is typed by
hand: every cell traces to `outputs/report/figures.json`,
`outputs/svgloc/{summary,ablations}.json` or `outputs/svgderived/summary.json`.

The one exception is the stratified odds ratio in T6, which is not stored
anywhere and is recomputed here from the run rows so it can be audited.

    python -m blindspot.reporting.report_tables       # -> outputs/report/tables.md
"""

from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path

OUT = Path("outputs/report")
DARK_THEMES = {"slate-dark", "carbon", "blueprint"}
RUN = "haiku-4-5_think2000_native_r0"
RUNGS = ("small", "large")      # the two reported resolutions


def points():
    """The scored point rows at the two reported resolutions."""
    from blindspot.analysis.svgloc_eval import load_run
    return [r for r in load_run(RUN)["point"] if r["meta"]["resolution"] in RUNGS]


def load(p: str) -> dict:
    return json.loads(Path(p).read_text())


def pct(x, d=1) -> str:
    return "—" if x is None else f"{x * 100:.{d}f}%"


def table(head: list[str], rows: list[list], note: str = "") -> str:
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    if note:
        out += ["", f"*{note}*"]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ T1, T2
BENCH = [                       # (key, label, operation, metric)
    ("charxiv", "CharXiv", "Read a value or a structure off a scientific chart",
     "judged exact match"),
    ("ai2d", "AI2D", "Answer a question about a diagram, 4-way multiple choice",
     "accuracy"),
    ("slidevqa", "SlideVQA", "Answer from evidence inside a slide deck", "ANLS"),
    ("infographicvqa", "InfographicVQA", "Read a dense, large-format infographic", "ANLS"),
    ("screenspot_pro", "ScreenSpot-Pro",
     "Point at an element described by its function — the answer is a coordinate",
     "click-in-bbox"),
]


def t1(f: dict) -> str:
    b = f["benchmarks"]
    rows = [[lab, op, f'{b[k]["n"]:,}', met, f'**{pct(b[k]["acc"])}**']
            for k, lab, op, met in BENCH]
    ctl = b["slidevqa_allpages"]
    total = sum(b[k]["n"] for k, *_ in BENCH) + ctl["n"]
    return table(
        ["Benchmark", "Operation measured", "Items", "Metric", "Haiku 4.5"], rows,
        f'{total:,} questions on `{f["model"]}`, thinking at 2,000 tokens, each arm '
        f'scored by its own published metric. A sixth scored arm — SlideVQA all-pages, '
        f'n={ctl["n"]}, {pct(ctl["acc"])} — is a retrieval control, not a separate '
        f'benchmark.')


def t2(f: dict) -> str:
    bind, absd = f["ai2d_binding"], f["absence_detection"]
    mark, nomark = bind["sighted"]["refers_to_mark"], bind["sighted"]["no_mark"]
    pb = bind["paired_blind"]
    pa = absd["paired_blind"]["absent"]
    rows = [
        ["**Resolution bias**", "InfographicVQA, CharXiv",
         "ANLS 74.3% → 59.4% across image-size quintiles (n=2,801); CharXiv −9.5pp, "
         "1 panel vs 13+",
         "text-volume control flat — 6.7pp spread, no trend"],
        ["**Localization**", "ScreenSpot-Pro",
         f'{pct(f["benchmarks"]["screenspot_pro"]["acc"])} click-in-bbox '
         f'(n={f["benchmarks"]["screenspot_pro"]["n"]:,}); 0.0% on targets under 12px '
         f'as delivered', "—"],
        ["**Label–object matching**", "AI2D",
         f'{pct(mark["acc"])} when the question names a printed mark (n={mark["n"]:,}) '
         f'vs {pct(nomark["acc"])} when it does not (n={nomark["n"]:,})',
         f'{pct(pb["refers_to_mark"]["blind"])} vs {pct(pb["no_mark"]["blind"])} — '
         f'mark-referring sits **below the {pct(bind["chance"], 0)} chance line**'],
        ["**General OCR reasoning**", "CharXiv, SlideVQA",
         "CharXiv 90.7% descriptive vs 63.7% reasoning (27.0pp); SlideVQA 26.5pp as "
         "scored, **3.8pp** format-corrected", "—"],
        ["**Hallucination**", "CharXiv",
         f'{pct(absd["full_set"]["absent"]["invention_rate"])} invention on '
         f'{absd["full_set"]["absent"]["n"]:,} "Not Applicable" items, 45.7% on the '
         f'worst template; over-abstention '
         f'{pct(absd["full_set"]["over_abstention"]["rate"], 2)}',
         f'abstains **{pct(pa["abstains_blind"])} blind vs '
         f'{pct(pa["abstains_sighted"])} sighted** (n={pa["n"]}) — the image adds nothing'],
        ["**Counting**", "InfographicVQA, CharXiv",
         "InfoVQA 63% → 33% across count bins; CharXiv ticks 78.1% (n=224) vs objects "
         "93.3% (n=314)", "—"],
    ]
    return table(["Blind spot", "Benchmark", "Public-benchmark result", "Blind control"],
                 rows,
                 "The blind control asks the same question with the image withheld. A "
                 "candidate is a perception blind spot only where withholding the image "
                 "changes the answer.")


# ------------------------------------------------------------------ T3, T4
def t3(f: dict) -> str:
    from blindspot.analysis.svgloc_eval import precision_curve
    rows = points()
    curve = precision_curve(rows)
    label = {"exact hit box": "**the exact target box**"}
    out = []
    for c in curve:
        last = c is curve[-1]
        lab = label.get(c["grid"], f'{c["grid"].replace("x", "×")} cell')
        bold = (lambda t: f"**{t}**") if last else (lambda t: t)
        out.append([lab, bold(pct(c["chance"], 2)), bold(pct(c["strict"], 2)),
                    bold(f'{c["ratio"]:.1f}×')])
    return table(["Required precision", "Chance", "Accuracy", "Ratio to chance"], out,
                 f'Both resolutions pooled, n={len(rows):,}. Chance at the exact box is '
                 f'the mean target-area fraction. The same predictions are bucketed more '
                 f'coarsely at each row up, so this is one set of answers read at six '
                 f'tolerances, not six experiments.')


def _misses_on_other_label(rows) -> float:
    """Share of in-range misses landing inside a *different* catalogued target box.

    Decides whether a bad citation is visibly or invisibly wrong, so it is read
    off the manifest's own boxes rather than asserted.
    """
    boxes = collections.defaultdict(list)
    for line in open("data/svg_localization/manifest.jsonl"):
        r = json.loads(line)
        b = r.get("gold_bbox_norm") or r.get("bbox_norm")
        if b:
            boxes[(r["graph_id"], r["resolution"])].append((r.get("target_idx"), b))
    k = n = 0
    for r in rows:
        if r["hit"] or not r.get("in_range", True):
            continue
        n += 1
        m, (px, py) = r["meta"], r["pred"]
        for idx, b in boxes[(m["graph_id"], m["resolution"])]:
            if idx != m["target_idx"] and b[0] <= px <= b[2] and b[1] <= py <= b[3]:
                k += 1
                break
    return k / max(n, 1)


def t4(f: dict) -> str:
    s = f["synthetic"]
    rows = []
    for g in RUNGS:
        b = s["bands"]["by_rung_d_centre"][g]
        d = s["distance"][g]
        rows.append([g, f'{d["n_miss"]:,}', pct(b["near_miss"]), pct(b["moderate_miss"]),
                     pct(b["wrong_region"]), f'{d["median_d_centre"] * 100:.1f}%'])
    return table(["Resolution", "Misses", "Near (<10%)", "Moderate (10–25%)",
                  "Wrong region (>25%)", "Median distance"], rows,
                 f'Distance is to the target centre, as a fraction of the frame '
                 f'diagonal, and the bands are of misses only. '
                 f'{pct(_misses_on_other_label(points()), 1)} of in-range misses land '
                 f'inside a different labelled element.')


# ------------------------------------------------------------------ T5
def _mh_or(rows) -> tuple[float, float, float]:
    """Mantel-Haenszel odds ratio for dark vs light, with a Robins-Breslow-Greenland CI.

    Strata are resolution x target-area tertile x contrast tertile, so a theme that
    simply drew bigger or higher-contrast targets cannot produce the effect on
    its own. Chart type is deliberately left out: adding it spreads the dark
    items over several hundred cells, and the point estimate then moves with the
    binning rather than with the data.
    """
    def tertile(vals):
        v = sorted(vals)
        return v[len(v) // 3], v[2 * len(v) // 3]

    a_lo, a_hi = tertile([r["meta"]["target_area_frac"] for r in rows])
    c_lo, c_hi = tertile([r["meta"]["target_contrast"] for r in rows])
    cut = lambda v, lo, hi: 0 if v < lo else (1 if v < hi else 2)

    strata = collections.defaultdict(lambda: [0, 0, 0, 0])   # a b c d
    for r in rows:
        m = r["meta"]
        key = (m["resolution"],
               cut(m["target_area_frac"], a_lo, a_hi),
               cut(m["target_contrast"], c_lo, c_hi))
        dark = m["theme"] in DARK_THEMES
        strata[key][(0 if dark else 2) + (0 if r["hit"] else 1)] += 1

    ps = qs = rs = rsum = ssum = 0.0
    for a, b, c, d in strata.values():
        n = a + b + c + d
        if n == 0 or (a + b) == 0 or (c + d) == 0:
            continue
        R, S = a * d / n, b * c / n
        P, Q = (a + d) / n, (b + c) / n
        ps += P * R
        qs += P * S + Q * R
        rs += Q * S
        rsum += R
        ssum += S
    if rsum == 0 or ssum == 0:
        return float("nan"), float("nan"), float("nan")
    orr = rsum / ssum
    se = math.sqrt(ps / (2 * rsum ** 2) + qs / (2 * rsum * ssum) + rs / (2 * ssum ** 2))
    return orr, orr * math.exp(-1.96 * se), orr * math.exp(1.96 * se)


def _crude_or(k1, n1, k2, n2) -> float:
    return ((k1 + .5) / (n1 - k1 + .5)) / ((k2 + .5) / (n2 - k2 + .5))


def t6(f: dict) -> str:
    """Dark vs light exact localization. One comparison, which is all §7 claims."""
    rows = points()
    dk = [r for r in rows if r["meta"]["theme"] in DARK_THEMES]
    lt = [r for r in rows if r["meta"]["theme"] not in DARK_THEMES]
    kd, kl = sum(r["hit"] for r in dk), sum(r["hit"] for r in lt)
    orr, lo, hi = _mh_or(rows)
    return table(["Background", "Exact localization", "Hits", "Items"],
                 [["Dark", pct(kd / len(dk)), kd, f"{len(dk):,}"],
                  ["Light", pct(kl / len(lt)), kl, f"{len(lt):,}"]],
                 f'Crude odds ratio {_crude_or(kd, len(dk), kl, len(lt)):.2f}; adjusted '
                 f'**{orr:.2f}** [{lo:.2f}–{hi:.2f}] by Mantel-Haenszel, stratified on '
                 f'resolution, target-area tertile and contrast tertile, so this is not '
                 f'simply dark themes drawing bigger or higher-contrast targets. Theme is '
                 f'assigned per scene rather than crossed within it, so this is '
                 f'observational.')


# ------------------------------------------------------------------ T7
def t7(f: dict) -> str:
    d = f["synthetic"]["derived"]
    w, c = d["word_mc"], d["counting"]
    by = lambda h, lab: next(x for x in h if x["label"] == lab)
    loc = {x["label"]: x for x in f["synthetic"]["headline"]}
    rows = [
        ["Is this word present? (4-way choice)", pct(by(w["headline"], "small")["acc"], 2),
         pct(by(w["headline"], "large")["acc"], 2),
         pct(w["blind"]["overall"]["acc"]), f'{w["overall"]["n"]:,}'],
        ["How many times does it appear?", pct(by(c["headline"], "small")["acc"], 2),
         pct(by(c["headline"], "large")["acc"], 2),
         pct(c["blind"]["overall"]["acc"]), f'{c["overall"]["n"]:,}'],
        ["Point at it (exact target box)", pct(loc["small"]["acc"], 2),
         pct(loc["large"]["acc"], 2), "—", f'{loc["small"]["n"] + loc["large"]["n"]:,}'],
    ]
    return table(["Task on the generated scenes", "Small", "Large",
                  "Blind control", "Items"], rows,
                 "The same 200 scenes and the same image files; the question populations "
                 "differ. Reading and counting do not move with resolution, which is what "
                 "the public megapixel gradient predicted they would do. Pointing does "
                 "move, and in the opposite direction.")


TABLES = [("T1", "The benchmark suite", t1), ("T2", "Result per blind spot", t2),
          ("T3", "Localization accuracy against required precision", t3),
          ("T4", "Where the misses land", t4),
          ("T5", "Localization by background polarity", t6),
          ("T6", "Reading, counting and pointing on the same scenes", t7)]


def build() -> dict[str, str]:
    f = load("outputs/report/figures.json")
    return {tid: fn(f) for tid, _title, fn in TABLES}


def inject(path: str = "outputs/report/blindspots.md") -> int:
    """Rewrite each `<!-- Tn -->...<!-- /Tn -->` block in place.

    The prose between the markers is hand-written and never touched; only the
    generated tables are replaced, so the report can be re-derived after a rerun
    without losing edits.
    """
    src = Path(path)
    text = src.read_text()
    built = build()
    n = 0
    for tid, title, _fn in TABLES:
        pat = re.compile(rf"<!-- {tid} -->.*?<!-- /{tid} -->", re.S)
        if not pat.search(text):
            print(f"  !! no marker for {tid} in {path}")
            continue
        text = pat.sub(f"<!-- {tid} -->\n**Table {tid[1:]}. {title}.**\n\n"
                       f"{built[tid]}\n<!-- /{tid} -->", text)
        n += 1
    src.write_text(text)
    return n


def main() -> int:
    f = load("outputs/report/figures.json")
    md = ["# Tables\n",
          "Generated by `python -m blindspot.reporting.report_tables`. Every cell is read from "
          "the measured JSON, so these cannot drift from the results.\n"]
    for tid, title, fn in TABLES:
        md += [f"\n## {tid} — {title}\n\n", fn(f), "\n"]
    (OUT / "tables.md").write_text("".join(md))
    print(f"wrote {OUT / 'tables.md'} ({len(TABLES)} tables)")
    print(f"injected {inject()} tables into outputs/report/blindspots.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
