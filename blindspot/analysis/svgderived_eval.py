"""Analysis for the svg_localization-derived sets: `counting` and `word_mc`.

Both were run at `small` and `large` only, at the user's direction. That has one
consequence worth stating before any number is read, and it is stated on the
page rather than buried here: **the null control is gone**. Both EVAL.md files
designate `medium` vs `large` as the noise floor, because those two rungs deliver
at the same size and differ only in resampling path. Without `medium` there is no
within-set noise floor, so the localization run's measured null (-0.13pp over the
same 200 scenes and byte-identical pixels) is carried across as a proxy and
labelled as one.

Rules enforced here rather than left to the caller:
* counting is scored by exact integer match, and the SIGNED error is reported per
  bin -- absolute error alone destroys the mechanism (undercount vs overcount);
* counting families are never pooled into one accuracy number;
* word_mc position bias is tested before any accuracy number is trusted;
* a null or unparseable prediction is counted, never scored as wrong.
"""

from __future__ import annotations

import collections
import json
import statistics as st
from pathlib import Path

from blindspot.core.adapters import load
from blindspot.core.scoring import score as official_score
from blindspot.core.stats import wilson

RESULTS = Path("results")
RUNGS = ("small", "large")
MIN_CELL = 30
COUNT_BINS = [(1, 4, "1-4"), (5, 6, "5-6"), (7, 9, "7-9"), (10, 15, "10-15"), (16, 10**6, "16+")]
CHI2_CRIT_DF3 = 7.815          # 95%, 3 degrees of freedom


def read_jsonl(p: Path) -> list[dict]:
    rows = []
    if not p.exists():
        return rows
    for line in open(p):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def load_run(ds: str, tag: str, blind: bool = False) -> tuple[list[dict], list[dict]]:
    """Join a result file to its examples and score it. Returns (scored, unusable)."""
    exs = {e.uid: e for e in load(ds)}
    path = RESULTS / (f"{ds}__blind_{tag}.jsonl" if blind else f"{ds}__{tag}.jsonl")
    best: dict[str, dict] = {}
    for r in read_jsonl(path):
        uid = r["uid"][6:] if blind and r["uid"].startswith("blind:") else r["uid"]
        if r.get("pred") is None:
            best.setdefault(uid, r)
        else:
            best[uid] = r
    scored, unusable = [], []
    for uid, r in best.items():
        e = exs.get(uid)
        if e is None:
            continue
        if r.get("pred") is None:
            unusable.append({"uid": uid, "reason": r.get("parse_error") or r.get("error") or "null"})
            continue
        row = {"uid": uid, "pred": r["pred"], "meta": e.meta, "gold": e.gold,
               "question": e.question, "thinking": r.get("thinking") or ""}
        row.update(official_score(e, r["pred"]))
        row["hit"] = bool(row.get("score", 0) >= 0.5)
        scored.append(row)
    return scored, unusable


def by(rows, keyfn) -> dict:
    g = collections.defaultdict(list)
    for r in rows:
        g[keyfn(r)].append(r)
    return dict(g)


def cell(rows, label, extra=None) -> dict:
    n = len(rows)
    k = sum(r["hit"] for r in rows)
    lo, hi = wilson(k, n)
    out = {"label": label, "n": n, "k": k, "acc": (k / n) if n else None,
           "lo": lo, "hi": hi, "suppressed": n < MIN_CELL}
    if extra:
        out.update(extra)
    return out


def count_bin(v: int) -> str:
    for lo, hi, name in COUNT_BINS:
        if lo <= v <= hi:
            return name
    return "?"


def signed_stats(rows) -> dict:
    se = [r["signed_error"] for r in rows if r.get("signed_error") is not None]
    ae = [r["abs_error"] for r in rows if r.get("abs_error") is not None]
    wrong = [r["signed_error"] for r in rows
             if r.get("signed_error") is not None and not r["hit"]]
    return {"mean_signed": st.mean(se) if se else None,
            "mean_signed_when_wrong": st.mean(wrong) if wrong else None,
            "median_abs": st.median(ae) if ae else None,
            "under": sum(1 for v in wrong if v < 0), "over": sum(1 for v in wrong if v > 0)}


def paired(rows_by_rung: dict, a: str, b: str, keyfn) -> dict:
    ka = {keyfn(r): r for r in rows_by_rung.get(a, [])}
    kb = {keyfn(r): r for r in rows_by_rung.get(b, [])}
    both = sorted(set(ka) & set(kb), key=str)
    if not both:
        return {"a": a, "b": b, "n": 0}
    ha = sum(ka[k]["hit"] for k in both)
    hb = sum(kb[k]["hit"] for k in both)
    b_ = sum(1 for k in both if ka[k]["hit"] and not kb[k]["hit"])
    c_ = sum(1 for k in both if not ka[k]["hit"] and kb[k]["hit"])
    chi = ((abs(b_ - c_) - 1) ** 2 / (b_ + c_)) if (b_ + c_) else None
    return {"a": a, "b": b, "n": len(both), "acc_a": ha / len(both), "acc_b": hb / len(both),
            "delta_pp": (hb - ha) / len(both) * 100,
            "discordant_b": b_, "discordant_c": c_, "mcnemar_chi2": chi,
            "significant": bool(chi is not None and chi > 3.841)}


# ------------------------------------------------------------------- counting
def analyse_counting(tag: str) -> dict:
    rows, unusable = load_run("svg_counting", tag)
    blind, blind_bad = load_run("svg_counting", tag, blind=True)
    out = {"dataset": "svg_counting", "tag": tag,
           "counts": {"scored": len(rows), "unusable": len(unusable),
                      "blind_scored": len(blind), "blind_unusable": len(blind_bad),
                      "unusable_detail": unusable[:10]}}
    byr = by(rows, lambda r: r["meta"]["resolution"])
    out["headline"] = [cell(byr.get(g, []), g, signed_stats(byr.get(g, []))) for g in RUNGS]
    out["overall"] = cell(rows, "both rungs", signed_stats(rows))
    out["paired"] = paired(byr, "small", "large",
                           lambda r: (r["meta"]["graph_id"], r["question"]))

    # 3.3 dose-response, and the interaction: does the collapse point move with
    # resolution? Neither the ladder nor the counting curve shows that alone.
    out["dose"] = {}
    for g in RUNGS:
        cells = []
        for _lo, _hi, name in COUNT_BINS:
            sub = [r for r in byr.get(g, []) if count_bin(r["meta"]["true_count"]) == name]
            cells.append(cell(sub, name, signed_stats(sub)))
        out["dose"][g] = cells
    out["dose_all"] = [cell([r for r in rows if count_bin(r["meta"]["true_count"]) == name],
                            name, signed_stats([r for r in rows
                                                if count_bin(r["meta"]["true_count"]) == name]))
                       for _lo, _hi, name in COUNT_BINS]

    # The dose-response above is not interpretable on its own: the true count is
    # not randomly assigned across question forms, so a count bin is partly a
    # proxy for *what* is being counted. Measure that confound, then run the
    # clean version -- within a single form, where the thing counted is fixed.
    forms = by(rows, lambda r: r["meta"]["question_form"])
    bin_forms = {}
    for _lo, _hi, name in COUNT_BINS:
        sub = [r for r in rows if count_bin(r["meta"]["true_count"]) == name]
        cf = collections.Counter(r["meta"]["question_form"] for r in sub)
        bin_forms[name] = {"n": len(sub), "n_forms": len(cf),
                           "top": [(q, c) for q, c in cf.most_common(3)]}
    within = []
    for q, v in sorted(forms.items()):
        tc = [r["meta"]["true_count"] for r in v]
        if len(v) < 20 or (max(tc) - min(tc)) < 4:
            continue
        med = st.median(tc)
        lo = [r for r in v if r["meta"]["true_count"] <= med]
        hi = [r for r in v if r["meta"]["true_count"] > med]
        if len(lo) < 6 or len(hi) < 6:
            continue
        al = sum(r["hit"] for r in lo) / len(lo)
        ah = sum(r["hit"] for r in hi) / len(hi)
        within.append({"form": q, "min": min(tc), "max": max(tc),
                       "lo_n": len(lo), "lo_acc": al, "hi_n": len(hi), "hi_acc": ah,
                       "delta_pp": (ah - al) * 100})
    errs = [r for r in rows if not r["hit"]]
    out["dose_confound"] = {
        "bin_forms": bin_forms, "within_form": within,
        "n_forms_testable": len(within), "n_forms_total": len(forms),
        "n_errors": len(errs),
        "signed_histogram": sorted(collections.Counter(
            r["signed_error"] for r in errs if r.get("signed_error") is not None).items()),
    }

    # 3.2/5: families are reported separately and never pooled.
    out["family"] = {}
    for fam, v in by(rows, lambda r: r["meta"]["count_family"]).items():
        c = cell(v, fam, signed_stats(v))
        c["by_rung"] = {g: cell(vv, g, signed_stats(vv))
                        for g, vv in by(v, lambda r: r["meta"]["resolution"]).items()}
        out["family"][fam] = c

    for name, keyfn in [("chart_type", lambda r: r["meta"]["chart_type"]),
                        ("question_form", lambda r: r["meta"]["question_form"]),
                        ("theme", lambda r: r["meta"]["theme"]),
                        ("font_family", lambda r: r["meta"]["font_family"])]:
        out[name] = sorted((cell(v, k, signed_stats(v)) for k, v in by(rows, keyfn).items()),
                           key=lambda c: -(c["acc"] or 0))

    bb = by(blind, lambda r: r["meta"]["resolution"])
    out["blind"] = {"overall": cell(blind, "blind"),
                    "by_rung": {g: cell(v, g) for g, v in bb.items()},
                    "by_family": {k: cell(v, k) for k, v in
                                  by(blind, lambda r: r["meta"]["count_family"]).items()}}
    return out


# -------------------------------------------------------------------- word_mc
def chi2_against(observed: dict, expected_share: dict, n: int) -> dict:
    chi = 0.0
    rows = []
    for k in "ABCD":
        o = observed.get(k, 0)
        e = expected_share.get(k, 0.0) * n
        if e > 0:
            chi += (o - e) ** 2 / e
        rows.append({"option": k, "observed": o, "obs_share": o / n if n else None,
                     "expected_share": expected_share.get(k, 0.0),
                     "deviation_pp": ((o / n) - expected_share.get(k, 0.0)) * 100 if n else None})
    return {"chi2": chi, "crit": CHI2_CRIT_DF3, "biased": chi > CHI2_CRIT_DF3, "rows": rows, "n": n}


def analyse_word_mc(tag: str) -> dict:
    rows, unusable = load_run("svg_word_mc", tag)
    blind, blind_bad = load_run("svg_word_mc", tag, blind=True)
    out = {"dataset": "svg_word_mc", "tag": tag, "chance": 0.25,
           "counts": {"scored": len(rows), "unusable": len(unusable),
                      "blind_scored": len(blind), "blind_unusable": len(blind_bad),
                      "unusable_detail": unusable[:10]}}

    # 3.1: position bias first. If this fails, nothing below is trustworthy.
    key = collections.Counter(r["gold"][0] for r in rows)
    n = len(rows)
    share = {k: key.get(k, 0) / n for k in "ABCD"}
    picks = collections.Counter(r.get("picked") for r in rows)
    wrong_picks = collections.Counter(r.get("picked") for r in rows if not r["hit"])
    nw = sum(wrong_picks.values())
    out["position_bias"] = {
        "all_picks": chi2_against(picks, share, n),
        # among wrong answers the model has nothing to go on, so a slot
        # preference shows up here first -- the sharper of the two tests.
        "wrong_picks": chi2_against(wrong_picks, {k: 0.25 for k in "ABCD"}, nw),
        "key_share": share,
    }

    byr = by(rows, lambda r: r["meta"]["resolution"])
    out["headline"] = [cell(byr.get(g, []), g) for g in RUNGS]
    out["overall"] = cell(rows, "both rungs")
    out["paired"] = paired(byr, "small", "large",
                           lambda r: (r["meta"]["graph_id"], r["meta"]["answer_text"]))

    # 3.2: a wrong answer is either a hallucinated reading (picked an absent
    # distractor) or a failure to spot the present word. Both look identical in
    # the accuracy number.
    miss = [r for r in rows if not r["hit"]]
    chosen = collections.Counter()
    for r in miss:
        opts = r["meta"].get("options") or []
        p = r.get("picked")
        if p and p in "ABCD" and len(opts) == 4:
            chosen[opts["ABCD".index(p)]] += 1
    out["wrong_analysis"] = {"n_wrong": len(miss),
                             "top_distractors": chosen.most_common(15),
                             "distinct_distractors": len(chosen)}

    for name, keyfn in [("chart_type", lambda r: r["meta"]["chart_type"]),
                        ("theme", lambda r: r["meta"]["theme"]),
                        ("font_family", lambda r: r["meta"]["font_family"]),
                        ("answer_len", lambda r: _lenbin(r["meta"].get("answer_len") or 0))]:
        out[name] = sorted((cell(v, k) for k, v in by(rows, keyfn).items()),
                           key=lambda c: -(c["acc"] or 0))

    dark = {"slate-dark", "carbon", "blueprint"}
    out["polarity"] = {k: cell(v, k) for k, v in
                       by(rows, lambda r: "dark" if r["meta"]["theme"] in dark else "light").items()}

    bb = by(blind, lambda r: r["meta"]["resolution"])
    out["blind"] = {"overall": cell(blind, "blind"),
                    "by_rung": {g: cell(v, g) for g, v in bb.items()}}
    return out


def _lenbin(n: int) -> str:
    if n <= 4:
        return "<=4 chars"
    if n <= 6:
        return "5-6"
    if n <= 8:
        return "7-8"
    if n <= 10:
        return "9-10"
    return "11+"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="haiku-4-5_think2000_native_r0")
    ap.add_argument("--out", default="outputs/svgderived/summary.json")
    a = ap.parse_args()
    s = {"counting": analyse_counting(a.tag), "word_mc": analyse_word_mc(a.tag)}
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, indent=1))
    for k in ("counting", "word_mc"):
        d = s[k]
        print(f"{k}: {d['counts']['scored']} scored, {d['counts']['unusable']} unusable, "
              f"blind {d['counts']['blind_scored']}")
        for c in d["headline"]:
            if c["n"]:
                print(f"   {c['label']:7s} n={c['n']:4d}  acc {c['acc']*100:6.2f}% "
                      f"[{c['lo']*100:.1f}-{c['hi']*100:.1f}]"
                      + (f"  mean signed {c['mean_signed']:+.2f}" if c.get("mean_signed") is not None else ""))
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
