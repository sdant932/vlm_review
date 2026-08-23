"""Where does Haiku 4.5's UI grounding break down, and along which axis?

Reads the official-protocol rows and slices them by every axis ScreenSpot-Pro
annotates, under BOTH parse tiers:

    official  published regex verbatim -- comparable to the GPT-4o 0.8%
    lenient   whitespace-tolerant + pixel->0-1 rescale (the same courtesy the
              official parser already extends to GPT-4o)

Reporting both keeps format-compliance and localisation ability separable: a
single number confounds "can't point" with "didn't punctuate the answer".

    python scripts/analyze/capability_report.py
"""
from __future__ import annotations
import json, math, sys
from collections import defaultdict
from pathlib import Path

from blindspot.core.prompts import HAIKU_MAX_EDGE
from scripts.run.official_eval import score_row

RESULTS = Path("results")


def load(ds, model="haiku-4-5"):
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


def table(title, rows, keyfn, order=None, note=""):
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


def main() -> int:
    ds = sys.argv[1] if len(sys.argv) > 1 else "screenspot_pro"
    rows = load(ds)
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

    table("BY ELEMENT TYPE", rows, lambda r: r["meta"].get("ui_type") or "?",
          order=["text", "icon"],
          note="the paper's own headline axis: icons carry app-specific meaning")
    table("BY APPLICATION GROUP", rows, lambda r: r["meta"].get("group") or "?",
          note="domain familiarity: does it know what these tools look like?")
    table("BY TARGET SIZE AS THE MODEL SEES IT", rows,
          lambda r: bucket(r["_side_seen"], [8, 12, 20, 32, 56],
                           ["<8px", "8-12px", "12-20px", "20-32px", "32-56px", ">=56px"]),
          order=["<8px", "8-12px", "12-20px", "20-32px", "32-56px", ">=56px"],
          note="native size x the API's ~1568px downscale -- the acuity question")
    table("BY SCREEN RESOLUTION", rows,
          lambda r: bucket(r["_megapix"], [2.5, 4.5, 8.5],
                           ["<2.5MP", "2.5-4.5MP", "4.5-8.5MP", ">=8.5MP"]),
          order=["<2.5MP", "2.5-4.5MP", "4.5-8.5MP", ">=8.5MP"],
          note="more pixels = more thrown away at the downscale gate")
    table("BY PLATFORM", rows, lambda r: r["meta"].get("platform") or "?")

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


if __name__ == "__main__":
    raise SystemExit(main())
