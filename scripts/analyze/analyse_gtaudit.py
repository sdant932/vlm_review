"""Analyse results/charxiv__gtaudit.jsonl: GT-quality rates by failure mode and question type.

Usage:
    python scripts/analyze/analyse_gtaudit.py
"""
import collections, json, math

from blindspot.core.adapters import load
from blindspot.analysis.aggregate import load_rows
from blindspot.core.failure_modes import LABELS as FM

DS = "charxiv"

# CharXiv descriptive question ids, grouped into families that fail differently.
FREE = {1, 2, 3, 13, 16}   # title, x-label, y-label, legend names, trend  (free-text)
TICK = {4, 5, 6, 7, 15}    # verbatim tick values
NUM  = {8, 9, 10, 12, 14, 17, 19}
OTH  = {11, 18}


def wilson(k, n, z=1.96):
    if not n: return (0, 0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h)*100, min(1, c+h)*100)


def fam(q):
    return ("free-text (title/axis/legend/trend)" if q in FREE else
            "tick-value readout" if q in TICK else
            "counting / arithmetic" if q in NUM else "structural (intersect/layout)")


def show(title, key, rows, popn=None, minn=1):
    print(f"\n### {title}")
    print(f"{'key':<56} {'n':>4} {'badGT%':>7} {'95% CI':>13} {'!gold%':>7} {'pop':>5}")
    g = collections.defaultdict(list)
    for r in rows: g[r[key]].append(r)
    tab = []
    for k, rs in g.items():
        n = len(rs); b = sum(r["bad"] for r in rs); ng = sum(r["not_gold"] for r in rs)
        tab.append((k, n, b, b/n*100, ng/n*100, wilson(b, n), (popn or {}).get(k)))
    for k, n, b, bp, ngp, ci, p in sorted(tab, key=lambda t: (-t[3], -t[1])):
        if n < minn: continue
        print(f"{str(k)[:56]:<56} {n:>4} {b:>3}/{n:<3}{bp:>4.0f}% "
              f"{ci[0]:>5.0f}-{ci[1]:<5.0f} {ngp:>6.0f}% {str(p or ''):>5}")


def main():
    recs = list({json.loads(l)["uid"]: json.loads(l)
                 for l in open(f"results/{DS}__gtaudit.jsonl")
                 if "error" not in json.loads(l)}.values())

    ex = {e.uid: e for e in load(DS)}
    for r in recs:
        m = ex[r["uid"]].meta
        r.update(qid=m.get("qid"), qlabel=m.get("qlabel"), split=m.get("split"),
                 bad=r["gt_quality"] != "unambiguous", not_gold=r["verdict"] != "gold_correct")

    pop = collections.Counter(rr.get("failure_mode", "unclassified")
                              for rr in load_rows(DS) if (rr.get("score") or 0) < 0.5)
    NPOP = sum(pop.values())

    n = len(recs); bad = sum(r["bad"] for r in recs)
    print(f"AUDITED {n} of {NPOP} charxiv failures ({n/NPOP*100:.0f}% of the failure set)")
    print("verdict:   ", dict(collections.Counter(r["verdict"] for r in recs).most_common()))
    print("gt_quality:", dict(collections.Counter(r["gt_quality"] for r in recs).most_common()))
    print(f"UNWEIGHTED bad-GT: {bad}/{n} = {bad/n*100:.1f}%  CI {wilson(bad,n)[0]:.0f}-{wilson(bad,n)[1]:.0f}%")

    g = collections.defaultdict(list)
    for r in recs: g[r.get("failure_mode", "unclassified")].append(r)
    w  = sum(pop[k]/NPOP * (sum(x["bad"] for x in v)/len(v)) for k, v in g.items() if k in pop)
    wg = sum(pop[k]/NPOP * (sum(x["not_gold"] for x in v)/len(v)) for k, v in g.items() if k in pop)
    print(f"PREVALENCE-WEIGHTED bad-GT: {w*100:.1f}%   (verdict != gold_correct: {wg*100:.1f}%)")
    print(f"=> implies ~{w*497:.0f} of 497 failures / ~{w*497/5000*100:.1f}pp of the 88.4% score")

    print("\ncross-tab verdict x gt_quality:")
    ct = collections.Counter((r["verdict"], r["gt_quality"]) for r in recs)
    for k, v in ct.most_common(): print(f"  {k[0]:<20} {k[1]:<13} {v}")

    for r in recs: r["fmlabel"] = f'{r.get("failure_mode")} ({FM.get(r.get("failure_mode"),"")})'
    show("by failure mode", "fmlabel", recs)
    show("by split", "split", recs)
    show("by question type", "qlabel", [r for r in recs if r["qid"]], minn=3)

    for r in recs:
        r["fam"] = fam(r["qid"]) if r["qid"] else "reasoning split (free-form)"
    show("by question family", "fam", recs)


if __name__ == "__main__":
    main()
