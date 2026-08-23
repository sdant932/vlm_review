"""Results -> outputs/summary.json. Data only; no HTML.

Separating this from rendering means the numbers are independently checkable,
the report can be rebuilt in a second without re-scoring thousands of rows, and
a reviewer can diff two runs without parsing a web page.
"""

from __future__ import annotations

import glob
import json
import math
import collections
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from blindspot.core.adapters import load
from blindspot.core.scoring import score
from blindspot.core.failure_modes import classify as classify_failure, classify_point
from blindspot.core.taxonomy import LABELS, is_not_applicable, primitive_for

RESULTS = Path("results")
OUT = Path("outputs")


def load_rows(dataset: str, with_judge: bool = True) -> list[dict]:
    """All usable rows for a dataset, unioned across tag schemes.

    ScreenSpot-Pro results ended up split across two differently-tagged files
    mid-project; unioning here (best row per uid wins) is what stops the report
    silently counting half a dataset.
    """
    examples = {e.uid: e for e in load(dataset)}
    best: dict[str, dict] = {}
    for f in sorted(glob.glob(f"results/{dataset}__*.jsonl")):
        if ".judged" in f or "_excluded" in f or "__gtaudit" in f or "__equiv" in f:
            continue
        for line in open(f):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            prev = best.get(rec["uid"])
            if prev is None or prev.get("pred") is None or rec.get("pred") is not None:
                best[rec["uid"]] = rec

    judge = judged_scores(dataset) if with_judge else {}
    equiv = equiv_verdicts(dataset)
    rows = []
    for uid, rec in best.items():
        ex = examples.get(uid)
        if ex is None or rec.get("pred") is None:
            continue
        prim, prov = primitive_for(ex)
        row = dict(rec)
        row.update(score(ex, rec["pred"]))
        row.update({
            "_ex": ex, "question": ex.question, "primitive": prim,
            "provenance": prov, "not_applicable": is_not_applicable(ex),
        })
        # CharXiv's official grader is authoritative where it ran; keep the
        # string-match score alongside so the two remain comparable per item.
        if uid in judge:
            row["judge_score"] = judge[uid]
            row["string_score"] = row["score"]
            row["score"] = judge[uid]
            row["metric"] = "charxiv_official_judge"
        # Why it failed, not just that it did. Deterministic where the answer is
        # list-shaped; the LLM pass resolves the rest.
        if (row.get("score") or 0) < 0.5:
            if ex.answer_type == "point":
                mode = classify_point(ex.gold, rec.get("pred"))
            elif ex.answer_type == "choice":
                mode = "wrong_option"
            else:
                mode = classify_failure(ex.gold, rec.get("pred"))
            v = equiv.get(uid)
            if v:
                if v.get("equivalent"):
                    mode = "format_only"
                    row["meaning_equivalent"] = True
                elif mode == "unclassified":
                    mode = v.get("failure_mode", "unclassified")
                row["gold_looks_wrong"] = bool(v.get("gold_looks_wrong"))
            row["failure_mode"] = mode
        rows.append(row)
    return rows


def equiv_verdicts(dataset: str) -> dict[str, dict]:
    """uid -> meaning-equivalence + failure-mode verdict, when that pass has run."""
    out = {}
    p = RESULTS / f"{dataset}__equiv.jsonl"
    if p.exists():
        for line in open(p):
            try:
                v = json.loads(line)
            except Exception:
                continue
            if "error" not in v:
                out[v["uid"]] = v
    return out


def judged_scores(dataset: str) -> dict[str, float]:
    """uid -> official LLM-judge score, when a judge pass exists."""
    out = {}
    for f in glob.glob(f"results/{dataset}__*.judged.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("judge_score") is not None:
                out[r["uid"]] = float(r["judge_score"])
    return out


def wilson(vals: list[float]) -> tuple[float, float] | None:
    n = len(vals)
    if not n:
        return None
    p, z = sum(vals) / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0)) / d
    return max(0.0, c - m), min(1.0, c + m)


def cell(rows: list[dict], key: str = "score") -> dict | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    ci = wilson(vals)
    return {"acc": sum(vals) / len(vals), "n": len(vals),
            "ci_lo": ci[0] if ci else None, "ci_hi": ci[1] if ci else None}


def slice_by(rows: list[dict], keyfn: Callable[[dict], Any], min_n: int = 1) -> list[dict]:
    g = defaultdict(list)
    for r in rows:
        g[keyfn(r)].append(r)
    out = []
    for k, v in g.items():
        c = cell(v)
        if c and c["n"] >= min_n:
            out.append({"label": str(k), **c})
    return sorted(out, key=lambda d: d["acc"])


DATASETS = ["charxiv", "infographicvqa", "screenspot_pro", "ai2d"]


def summarize(datasets: list[str] | None = None) -> dict:
    datasets = datasets or DATASETS
    all_rows: dict[str, list[dict]] = {ds: load_rows(ds) for ds in datasets}
    judged = {ds: judged_scores(ds) for ds in datasets}

    summary: dict[str, Any] = {
        "model": "claude-haiku-4-5-20251001",
        "datasets": {},
        "primitives": {},
        "charxiv": {},
        "localization": {},
        "totals": {},
    }

    # ---- per dataset -----------------------------------------------------
    for ds, rows in all_rows.items():
        c = cell(rows) or {}
        fails = [r for r in rows if (r.get("score") or 0) < 0.5]
        modes = collections.Counter(r.get("failure_mode", "unclassified") for r in fails)
        n_equiv = sum(1 for r in fails if r.get("meaning_equivalent"))
        summary["datasets"][ds] = {
            **c,
            "errors": 0,
            "judged_n": len(judged.get(ds, {})),
            "failures": len(fails),
            "failure_modes": dict(modes),
            "meaning_equivalent": n_equiv,
            # Official metric stays the headline; this is the adjusted twin.
            "acc_meaning_adjusted": ((c.get("acc", 0) * c.get("n", 0) + n_equiv) / c["n"]
                                     if c.get("n") else None),
            "gold_looks_wrong": sum(1 for r in fails if r.get("gold_looks_wrong")),
        }

    # ---- primitive x dataset matrix (the headline) -----------------------
    # N/A-heavy cells are scored on the answerable subset as well, because a
    # pooled number there measures "can you tell this doesn't apply".
    for prim in LABELS:
        per_ds = {}
        for ds, rows in all_rows.items():
            sel = [r for r in rows if r["primitive"] == prim]
            if not sel:
                continue
            answerable = [r for r in sel if not r["not_applicable"]]
            entry = {"all": cell(sel), "answerable": cell(answerable),
                     "na_rate": 1 - len(answerable) / len(sel),
                     "provenance": sel[0]["provenance"]}
            per_ds[ds] = entry
        if per_ds:
            pooled = [r for rows in all_rows.values() for r in rows if r["primitive"] == prim]
            pooled_ans = [r for r in pooled if not r["not_applicable"]]
            summary["primitives"][prim] = {
                "label": LABELS[prim],
                "sources": per_ds,
                "pooled": cell(pooled),
                "pooled_answerable": cell(pooled_ans),
                "n_sources": len(per_ds),
            }

    # ---- CharXiv: descriptive vs reasoning, and judge vs string match -----
    cx = all_rows.get("charxiv", [])
    if cx:
        desc = [r for r in cx if r["_ex"].meta.get("split") == "descriptive"]
        reas = [r for r in cx if r["_ex"].meta.get("split") == "reasoning"]
        summary["charxiv"]["descriptive"] = cell(desc)
        summary["charxiv"]["reasoning"] = cell(reas)
        summary["charxiv"]["by_qlabel"] = slice_by(
            desc, lambda r: r["_ex"].meta.get("qlabel") or "?", min_n=5)
        summary["charxiv"]["by_qlabel_answerable"] = slice_by(
            [r for r in desc if not r["not_applicable"]],
            lambda r: r["_ex"].meta.get("qlabel") or "?", min_n=5)
        j = judged.get("charxiv", {})
        if j:
            # Compare against string_score: `score` now holds the judge verdict,
            # so using it here would compare the judge with itself (100% by
            # construction) and silently destroy the check.
            paired = [(r.get("string_score", r["score"]), j[r["uid"]])
                      for r in cx if r["uid"] in j]
            ours = [1.0 if s >= 0.5 else 0.0 for s, _ in paired]
            off = [jj for _, jj in paired]
            summary["charxiv"]["grader_comparison"] = {
                "n": len(paired),
                "string_match": sum(ours) / len(ours) if ours else None,
                "official_judge": sum(off) / len(off) if off else None,
                "agreement": sum(1 for a, b in zip(ours, off) if a == b) / len(paired) if paired else None,
            }

    # ---- localization: the resolution story -------------------------------
    sp = all_rows.get("screenspot_pro", [])
    if sp:
        def bucket(r):
            frac = r["_ex"].meta.get("target_area_frac", 0) or 0
            side = math.sqrt(frac * 1568 * 882)
            for lim, name in ((12, "<12px"), (20, "12-20px"), (32, "20-32px"), (56, "32-56px")):
                if side < lim:
                    return name
            return ">=56px"
        summary["localization"] = {
            "by_target_size": slice_by(sp, bucket, min_n=5),
            "by_ui_type": slice_by(sp, lambda r: r["_ex"].meta.get("ui_type") or "?", min_n=5),
            "by_application": slice_by(sp, lambda r: r["_ex"].meta.get("group") or "?", min_n=10),
        }

    # ---- benchmark quality: reported as a caveat, never as an adjustment ----
    # Scores stay on the official metric over the full official split. These
    # numbers tell a reader how much annotation noise sits underneath, which is
    # a different thing from correcting for it: filtering to the cells a model
    # does well on would be indistinguishable from cherry-picking.
    quality = {}
    for ds, rows in all_rows.items():
        p_aud = RESULTS / f"{ds}__gtaudit.jsonl"
        if not p_aud.exists():
            continue
        aud = []
        for line in open(p_aud):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" not in r:
                aud.append(r)
        if not aud:
            continue
        bad = [r for r in aud if r.get("gt_quality") != "unambiguous"]
        fails = [r for r in rows if (r.get("score") or 0) < 0.5]
        rate = len(bad) / len(aud)
        quality[ds] = {
            "audited": len(aud),
            "failures": len(fails),
            "total": len(rows),
            "bad_gt_in_failures": rate,
            # The number that actually characterises the benchmark: the audit
            # sampled failures, where bad GT is enriched because it causes them.
            "bad_gt_whole_set": rate * len(fails) / max(len(rows), 1),
            "headline_if_all_credited": ((sum(r.get("score") or 0 for r in rows)
                                          + rate * len(fails)) / len(rows)) if rows else None,
        }
    summary["benchmark_quality"] = quality

    # ---- AI2D blind control: how much of the score needs the image at all ----
    p_blind = RESULTS / "ai2d_blind_control.json"
    if p_blind.exists():
        blind = json.loads(p_blind.read_text())
        acc = collections.defaultdict(list)
        for r in blind:
            acc[r["qtype"]].append(r["pred"] == r["gold"])
        seen = {}
        for r in all_rows.get("ai2d", []):
            seen.setdefault(r["_ex"].meta.get("qtype"), []).append(r.get("score") or 0)
        summary["ai2d_blind_control"] = {
            k: {"blind": sum(v) / len(v), "blind_n": len(v),
                "with_image": (sum(seen[k]) / len(seen[k])) if seen.get(k) else None,
                "n": len(seen.get(k, []))}
            for k, v in acc.items()}

    n_all = sum(len(v) for v in all_rows.values())
    summary["totals"] = {
        "questions": n_all,
        "primitives_measured": len(summary["primitives"]),
        "multi_source_primitives": sum(1 for p in summary["primitives"].values()
                                       if p["n_sources"] > 1),
    }
    return summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s = summarize()
    p = OUT / "summary.json"
    p.write_text(json.dumps(s, indent=1, default=str))
    print(f"wrote {p}  ({p.stat().st_size/1024:.0f} KB)")
    print(f"  {s['totals']['questions']} questions | "
          f"{s['totals']['primitives_measured']} primitives | "
          f"{s['totals']['multi_source_primitives']} with >1 source")
    for ds, d in s["datasets"].items():
        if d.get("acc") is not None:
            print(f"  {ds:16} {d['acc']*100:5.1f}%  n={d['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
