"""Corrected summary for the 22 Aug run: outputs/aug22/summary.json.

Differs from the original `blindspot.analysis.aggregate` output in three ways:

1. **SlideVQA is included.** The original DATASETS list predates that arm, so the
   published summary silently omitted 1,497 questions across two conditions.
2. **The control arms are first-class.** blind / one-page / grid / coarse-grid are
   what turn an accuracy table into a causal account, and none of them existed
   when the original schema was written.
3. **Every number is deduplicated by uid.** Resumed runs appended rather than
   replaced, leaving 2,121 duplicate lines in CharXiv and 175 in ScreenSpot-Pro.
   `aggregate.load_rows` already collapsed these correctly; the ad-hoc analyses
   during the session did not, so this module re-derives the affected figures
   from a single deduplicating loader and records the duplicate counts so the
   discrepancy is auditable rather than invisible.

Nothing here overwrites the original outputs/summary.json or outputs/report.html.
"""
from __future__ import annotations

import collections, json, re, statistics as st
from pathlib import Path

from blindspot.analysis.aggregate import summarize
from blindspot.core.scoring import anls, token_f1, numeric_or_text_match, point_in_bbox, charxiv_grading_confidence

OUT = Path("outputs/aug22")
RES = Path("results")
DATASETS = ["charxiv", "infographicvqa", "screenspot_pro", "ai2d", "slidevqa", "slidevqa_allpages"]

# Scale words must be normalised before two numbers can be compared: a model that
# answers "3.3bn" against a gold of "3300000000" has read the chart correctly and
# only dressed the answer differently.
SCALE = {"bn": 1e9, "billion": 1e9, "billions": 1e9, "tn": 1e12, "trillion": 1e12,
         "m": 1e6, "mn": 1e6, "million": 1e6, "millions": 1e6,
         "k": 1e3, "thousand": 1e3, "thousands": 1e3}


def _dedup(path: Path) -> list[dict]:
    """Last write per uid wins; rows without a usable prediction are dropped."""
    d = {}
    for line in path.open():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        d[r["uid"]] = r
    return [r for r in d.values() if r.get("pred") is not None]


def _dup_stats(path: Path) -> dict:
    uids = []
    for line in path.open():
        if line.strip():
            try:
                uids.append(json.loads(line)["uid"])
            except Exception:
                pass
    c = collections.Counter(uids)
    return {"lines": len(uids), "unique": len(c),
            "duplicate_lines": sum(v - 1 for v in c.values() if v > 1)}


def canon_num(s):
    t = str(s).strip().lower().replace(",", "")
    for ch in "$€£%":
        t = t.replace(ch, "")
    mult = 1.0
    for w, f in sorted(SCALE.items(), key=lambda x: -len(x[0])):
        if re.search(rf"(?<![a-z]){re.escape(w)}\b", t):
            mult = f
            t = re.sub(rf"(?<![a-z]){re.escape(w)}\b", "", t)
            break
    m = re.fullmatch(r"\s*(-?\d+\.?\d*)\s*[a-z]*\s*", t)
    try:
        return float(m.group(1)) * mult if m else None
    except ValueError:
        return None


def canon_txt(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def fmt_equiv(pred, golds) -> bool:
    """Conservative: numeric equality (sign-sensitive) or exact folded-text match.

    No substring fallback -- an early version scored "7" as equivalent to "17".
    """
    pn = canon_num(pred)
    for g in golds:
        gn = canon_num(g)
        if pn is not None and gn is not None:
            if pn == gn:
                return True
            continue
        if canon_txt(pred) and canon_txt(pred) == canon_txt(g):
            return True
    return False


def _f(path: Path) -> Path:
    return RES / path


def controls() -> dict:
    """The four ablations, each paired against the sighted/full run it modifies."""
    out: dict = {}

    # ---- provenance: how much duplication the resumed runs left behind -------
    out["duplication"] = {
        ds: _dup_stats(RES / f"{ds}__haiku-4-5_think2000_native_r0.jsonl")
        for ds in ("charxiv", "infographicvqa", "ai2d", "screenspot_pro", "slidevqa")
    }

    # Repeated items are an unplanned but real reproducibility measurement: the
    # same question asked twice, same settings. Disagreement here is the noise
    # floor under every other comparison in this report.
    raw = collections.defaultdict(list)
    for line in (RES / "charxiv__haiku-4-5_think2000_native_r0.jsonl").open():
        if line.strip():
            r = json.loads(line)
            raw[r["uid"]].append(r.get("pred"))
    rep = {u: v for u, v in raw.items() if len(v) > 1}
    dis = sum(1 for v in rep.values() if len({str(x) for x in v}) > 1)
    out["reproducibility"] = {
        "repeated_items": len(rep), "disagreed": dis,
        "disagreement_rate": dis / len(rep) if rep else None,
        "note": "same question, same settings, asked twice; thinking pins temperature to 1",
    }

    # ---- blind control ------------------------------------------------------
    sight = {}
    for ds in ("charxiv", "infographicvqa", "slidevqa", "ai2d"):
        for r in _dedup(RES / f"{ds}__haiku-4-5_think2000_native_r0.jsonl"):
            sight[r["uid"]] = r

    def sc(r):
        if r["answer_type"] == "choice":
            return 1.0 if str(r["pred"]).strip().upper() == str(r["gold"][0]).strip().upper() else 0.0
        if r["dataset"].startswith("slidevqa"):
            return token_f1(r["pred"], r["gold"])[1]
        if r["dataset"].startswith("charxiv"):
            conf = charxiv_grading_confidence(r["meta"].get("qid"))
            return numeric_or_text_match(r["pred"], r["gold"]) if conf == "strict" else anls(r["pred"], r["gold"])
        return anls(r["pred"], r["gold"])

    blind = collections.defaultdict(lambda: ([], []))
    cx_split = collections.defaultdict(lambda: ([], []))
    for b in _dedup(RES / "control_blind.jsonl"):
        src = b["meta"].get("src_uid")
        if src not in sight:
            continue
        ds = b["dataset"].replace("_blind", "")
        blind[ds][0].append(sc(b))
        blind[ds][1].append(sc(sight[src]))
        if ds == "charxiv":
            k = "descriptive" if b["meta"].get("split") == "descriptive" else "reasoning"
            cx_split[k][0].append(sc(b))
            cx_split[k][1].append(sc(sight[src]))

    CHANCE = {"ai2d": 0.25}
    out["blind"] = {
        ds: {"blind": sum(b) / len(b), "sighted": sum(s) / len(s),
             "vision_adds_pp": (sum(s) / len(s) - sum(b) / len(b)) * 100,
             "chance": CHANCE.get(ds, 0.0), "n": len(b)}
        for ds, (b, s) in blind.items()
    }
    out["blind_charxiv_split"] = {
        k: {"blind": sum(b) / len(b), "sighted": sum(s) / len(s), "n": len(b)}
        for k, (b, s) in cx_split.items()
    }

    # ---- one-page ablation (is the multi-page label real?) ------------------
    ev = {r["uid"]: token_f1(r["pred"], r["gold"])[1]
          for r in _dedup(RES / "slidevqa__haiku-4-5_think2000_native_r0.jsonl")}
    op = {r["meta"]["src_uid"]: token_f1(r["pred"], r["gold"])[1]
          for r in _dedup(RES / "control_onepage0.jsonl")}
    com = [k for k in op if k in ev]
    if com:
        out["onepage"] = {
            "n": len(com),
            "both_slides_f1": sum(ev[k] for k in com) / len(com) * 100,
            "one_slide_f1": sum(op[k] for k in com) / len(com) * 100,
            "collapse_f1": (sum(op[k] for k in com) - sum(ev[k] for k in com)) / len(com) * 100,
            "still_answerable_frac": sum(1 for k in com if op[k] >= 0.5) / len(com),
        }

    # ---- localization: precision falloff and the coordinate-emission tax ----
    ROWS, K = "ABCD", 4

    def cell(x, y):
        return f"{ROWS[min(int(y * K), K - 1)]}{min(int(x * K), K - 1) + 1}"

    sp = {r["uid"]: r for r in _dedup(RES / "screenspot_pro__haiku-4-5_think2000_native_r0.jsonl")}
    grids = {}
    for k_, label in ((2, "2x2"), (3, "3x3"), (4, "4x4"), (8, "8x8")):
        hit = 0
        for r in sp.values():
            px, py = r["pred"]
            x0, y0, x1, y1 = r["gold"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hit += (min(int(px * k_), k_ - 1), min(int(py * k_), k_ - 1)) == \
                   (min(int(cx * k_), k_ - 1), min(int(cy * k_), k_ - 1))
        grids[label] = {"acc": hit / len(sp), "chance": 1 / (k_ * k_)}
    exact = sum(point_in_bbox(r["pred"], r["gold"]) for r in sp.values()) / len(sp)
    area = sum(r["meta"]["target_area_frac"] for r in sp.values()) / len(sp)
    out["coarse_localization"] = {"grids": grids, "exact_click_in_bbox": exact,
                                  "mean_target_area_frac": area, "n": len(sp)}

    gr = {r["meta"]["src_uid"]: r for r in _dedup(RES / "control_grid4.jsonl")} \
        if (RES / "control_grid4.jsonl").exists() else {}
    pair = [k for k in gr if k in sp]
    if pair:
        def norm(p):
            m = re.search(r"([A-Da-d])\s*([1-4])", str(p))
            return (m.group(1).upper() + m.group(2)) if m else None
        named = sum(1 for k in pair if norm(gr[k]["pred"]) == gr[k]["gold"][0])
        clicked = sum(1 for k in pair if cell(*sp[k]["pred"]) == gr[k]["gold"][0])
        b = sum(1 for k in pair if norm(gr[k]["pred"]) == gr[k]["gold"][0] and cell(*sp[k]["pred"]) != gr[k]["gold"][0])
        c = sum(1 for k in pair if norm(gr[k]["pred"]) != gr[k]["gold"][0] and cell(*sp[k]["pred"]) == gr[k]["gold"][0])
        from math import comb
        n_ = b + c
        p = (sum(comb(n_, i) for i in range(min(b, c) + 1)) / 2 ** n_ * 2) if n_ else None
        out["grid_control"] = {
            "n": len(pair), "named_cell_acc": named / len(pair),
            "click_derived_cell_acc": clicked / len(pair),
            "delta_pp": (named - clicked) / len(pair) * 100,
            "chance": 1 / (K * K), "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p,
        }

    # ---- abstention: what happens when the thing simply is not there --------
    def isna(s):
        return str(s).strip().lower().rstrip(".") in ("not applicable", "n/a", "na", "none", "not available")

    cx = _dedup(RES / "charxiv__haiku-4-5_think2000_native_r0.jsonl")
    tp = fn = fp = tn = 0
    per = collections.defaultdict(lambda: [0, 0])
    for r in cx:
        g = any(isna(x) for x in r["gold"])
        p = isna(r["pred"])
        if g and p: tp += 1
        elif g: fn += 1
        elif p: fp += 1
        else: tn += 1
        if g:
            k = (r["meta"].get("qid"), (r["meta"].get("qlabel") or "")[:60])
            per[k][0] += 1
            per[k][1] += p
    out["abstention"] = {
        "gold_na_n": tp + fn, "correctly_abstained": tp / (tp + fn) if tp + fn else None,
        "invented_a_value": fn / (tp + fn) if tp + fn else None,
        "gold_value_n": fp + tn, "over_abstained": fp / (fp + tn) if fp + tn else None,
        "by_question": [{"qid": k[0], "qlabel": k[1], "n": v[0], "abstained": v[1] / v[0]}
                        for k, v in sorted(per.items(), key=lambda x: -x[1][0]) if v[0] >= 20],
    }

    # ---- format artifact: how much "failure" is the metric, not the model ---
    def artifact(rows, scorer):
        zero = [r for r in rows if scorer(r) == 0.0]
        nonex = [r for r in rows if scorer(r) < 1.0]
        return {"hard_zeros": len(zero),
                "hard_zeros_format_equivalent": sum(1 for r in zero if fmt_equiv(r["pred"], r["gold"])),
                "non_exact": len(nonex),
                "non_exact_format_equivalent": sum(1 for r in nonex if fmt_equiv(r["pred"], r["gold"]))}

    sv = _dedup(RES / "slidevqa__haiku-4-5_think2000_native_r0.jsonl")
    iv = _dedup(RES / "infographicvqa__haiku-4-5_think2000_native_r0.jsonl")
    out["format_artifact"] = {
        "slidevqa": artifact(sv, lambda r: token_f1(r["pred"], r["gold"])[1]),
        "infographicvqa": artifact(iv, lambda r: anls(r["pred"], r["gold"])),
        "charxiv": artifact(cx, lambda r: anls(r["pred"], r["gold"])),
        "note": "token-F1 scores '22%' against '22' as zero; ANLS (edit distance) barely notices",
    }

    # ---- is a wrong number a near miss, or the wrong element entirely? ------
    def numerr(rows, label):
        errs, ex, tot = [], 0, 0
        for r in rows:
            gn, pn = canon_num(r["gold"][0]), canon_num(r["pred"])
            if gn is None or pn is None or gn == 0:
                continue
            tot += 1
            if pn == gn:
                ex += 1
                continue
            errs.append(abs(pn - gn) / abs(gn))
        if not errs:
            return None
        return {"label": label, "n_numeric": tot, "exact_frac": ex / tot, "n_wrong": len(errs),
                "median_rel_error": st.median(errs),
                "within_10pct_frac": sum(1 for e in errs if e <= 0.10) / len(errs),
                "over_100pct_frac": sum(1 for e in errs if e > 1.0) / len(errs)}

    out["numeric_error"] = [x for x in (
        numerr([r for r in cx if r["meta"].get("split") == "descriptive"], "CharXiv descriptive"),
        numerr([r for r in cx if r["meta"].get("split") != "descriptive"], "CharXiv reasoning"),
        numerr(iv, "InfographicVQA"),
        numerr(sv, "SlideVQA"),
    ) if x]

    # ---- SlideVQA: the three costs, as-scored and format-corrected ----------
    def f1s(r, fix):
        f = token_f1(r["pred"], r["gold"])[1]
        return 1.0 if (fix and f < 1.0 and fmt_equiv(r["pred"], r["gold"])) else f

    ap = {r["uid"].rsplit(":", 1)[1]: r for r in _dedup(RES / "slidevqa_allpages__haiku-4-5_think2000_native_r0.jsonl")}
    evd = {r["uid"].rsplit(":", 1)[1]: r for r in sv}
    common = sorted(set(evd) & set(ap))
    costs = {}
    for fix, tag in ((False, "as_scored"), (True, "format_corrected")):
        mean = lambda rs: sum(f1s(r, fix) for r in rs) / len(rs) * 100
        mp = [r for r in sv if r["meta"].get("multi_page")]
        spg = [r for r in sv if not r["meta"].get("multi_page")]
        ar = [r for r in sv if r["meta"].get("arithmetic")]
        lo = [r for r in sv if not r["meta"].get("arithmetic")]
        costs[tag] = {
            "retrieval": mean([ap[k] for k in common]) - mean([evd[k] for k in common]),
            "integration": mean(mp) - mean(spg),
            "derivation": mean(ar) - mean(lo),
            "overall_f1": mean(sv), "lookup_f1": mean(lo), "arithmetic_f1": mean(ar),
            "single_page_f1": mean(spg), "multi_page_f1": mean(mp),
        }
    costs["paired_n"] = len(common)
    out["slidevqa_costs"] = costs
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s = summarize(DATASETS)
    s["controls"] = controls()
    s["generated"] = "2026-08-22"
    s["totals"]["questions"] = sum(d["n"] for d in s["datasets"].values() if d.get("n"))
    p = OUT / "summary.json"
    p.write_text(json.dumps(s, indent=1, default=str))
    print(f"wrote {p} ({p.stat().st_size/1024:.0f} KB)")
    for ds, d in s["datasets"].items():
        if d.get("acc") is not None:
            print(f"  {ds:20} {d['acc']*100:5.2f}%  n={d['n']}")
    print(f"  TOTAL questions {s['totals']['questions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
