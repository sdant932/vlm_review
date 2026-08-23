"""outputs/slidevqa.html -- the SlideVQA arm of the perception blind-spot study.

SlideVQA is the only multi-page source in this harness: every question ships a
20-slide business deck and carries annotated `evidence_pages`. That annotation
lets the same questions run under two conditions, so a single score never has to
stand in for two different abilities:

    evidence    only the 1-2 annotated slides are sent -- oracle retrieval.
                What is left is multi-hop reading and derivation.
    all_pages   the whole 20-slide deck is sent. The model must find the
                evidence itself before it can read it.

The paired gap between the two isolates retrieval. Everything on the rendered
page is recomputed here from the raw JSONL; nothing is transcribed from a
previous write-up.

One honest caveat, surfaced rather than buried: SlideVQA's official metrics are
exact match and token-level F1, both of which score "22%" against a gold of "22"
as a total miss. That is a formatting disagreement, not a perception failure, and
it is not evenly distributed -- arithmetic answers are bare numbers and take the
brunt of it. The report therefore carries every headline twice: as officially
scored, and with format-equivalent answers credited. Both are shown; neither is
presented alone.

Entry point:  python -m blindspot.reporting.slidevqa_report
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from blindspot.core.scoring import token_f1

ROOT = Path(__file__).resolve().parents[2]   # blindspot/reporting/x.py -> repo root
OUT = ROOT / "outputs"
ASSETS = OUT / "assets_slidevqa"
DATA = ROOT / "data" / "slidevqa"
RESULTS = ROOT / "results"

EVIDENCE_JSONL = RESULTS / "slidevqa__haiku-4-5_think2000_native_r0.jsonl"
ALLPAGES_JSONL = RESULTS / "slidevqa_allpages__haiku-4-5_think2000_native_r0.jsonl"
MANIFEST = DATA / "manifest.jsonl"

THUMB_W = 200
THUMB_Q = 70
FULL_EDGE = 1100
FULL_Q = 82

N_PAGES = 20


def esc(s) -> str:
    return html.escape(str(s), quote=True)


# ---------------------------------------------------------------------------
# Format equivalence.
#
# EM and token-F1 compare surface strings. A large share of this model's
# "failures" are the same value wearing a unit: 22% vs 22, $2,410 vs 2410,
# 3.3bn vs 3.3. Calling those perception errors would overstate the blind spot,
# so they are detected explicitly and counted separately.
#
# Deliberately conservative. Numeric comparison is sign-sensitive (-300 is NOT
# 300 -- that is a real sign error) and never falls back to substring matching,
# because "7" is a substring of "17". Text falls back to whole-word containment
# only, and only when the shorter side is >= 4 characters.
# ---------------------------------------------------------------------------
_UNITS = (r"(bn|billion|billions|million|millions|mn|thousand|k|usd|dollars?|euros?|"
          r"eur|gbp|percent|pct|tonnes?|tons?|units?|people|users?|trillion|crores?|"
          r"lakhs?|percentage\s*points?|points?|pts?)")


def canon(x) -> str:
    t = str(x).strip().lower().replace(",", "")
    for ch in "$€£%":
        t = t.replace(ch, "")
    t = re.sub(r"\b" + _UNITS + r"\b", "", t)
    t = re.sub(r"[^\w\s.+\-]", " ", t)
    return " ".join(t.split())


def as_float(x):
    m = re.fullmatch(r"[+\-]?\d*\.?\d+", canon(x))
    return float(m.group()) if m else None


def format_equivalent(pred, golds) -> bool:
    """True when pred and some gold are the same value in different clothes."""
    cp = canon(pred)
    if not cp:
        return False
    for g in golds:
        cg = canon(g)
        if not cg:
            continue
        if cp == cg:
            return True
        fp, fg = as_float(pred), as_float(g)
        if fp is not None and fg is not None:
            if abs(fp - fg) <= 1e-9 * max(1.0, abs(fg)):
                return True
            continue
        if fp is not None or fg is not None:
            continue  # one numeric, one not -- not a formatting difference
        short, long_ = (cp, cg) if len(cp) < len(cg) else (cg, cp)
        if len(short) >= 4 and re.search(r"(?<!\w)" + re.escape(short) + r"(?!\w)", long_):
            return True
    return False


# ---------------------------------------------------------------------------
# What the model looked at, according to its own trace.
# ---------------------------------------------------------------------------
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
        "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
        "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
        "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
        "twentieth": 20}
_ORD_RE = re.compile(r"\b(" + "|".join(_ORD) + r")\s+(?:slide|page|image)\b")
_NUM_RE = re.compile(r"\b(?:slide|page|image)\s*#?\s*(\d{1,2})\b")


def cited_slides(thinking, n_pages=N_PAGES) -> list[int]:
    """1-based slide indices the trace names explicitly.

    Only ~1 trace in 3 names a slide at all; when it does the reference is
    reliable. Absence is reported as unknown, never as zero.
    """
    t = str(thinking or "").lower()
    out = set()
    for m in _NUM_RE.finditer(t):
        v = int(m.group(1))
        if 1 <= v <= n_pages:
            out.add(v)
    for m in _ORD_RE.finditer(t):
        out.add(_ORD[m.group(1)])
    return sorted(out)


def trace_numbers(thinking) -> set[float]:
    return {float(x) for x in re.findall(r"\d+\.?\d*", str(thinking or "").replace(",", ""))}


_EXPR_RE = re.compile(r"([\d.]+)\s*([-+*/])\s*([\d.]+)")


def classify_arithmetic(row, expression):
    """exact | format_only | wrong_operand | wrong_operation | unparsed_expr.

    The operand/operation split is the informative one: it separates *misreading
    a number off the slide* (a perception failure) from *reading both numbers
    correctly and then computing wrong* (a derivation failure). Decided by
    checking whether both annotated operands appear in the model's own trace.
    """
    if row["em"] == 1:
        return "exact"
    if row["fmt_equiv"]:
        return "format_only"
    e = str(expression or "").replace(",", "")
    m = _EXPR_RE.fullmatch(e.strip())
    if not m:
        return "unparsed_expr"
    a, b = float(m.group(1)), float(m.group(3))
    seen = trace_numbers(row["thinking"])

    def near(v):
        return any(abs(v - x) < 0.005 for x in seen)

    return "wrong_operation" if (near(a) and near(b)) else "wrong_operand"


# ---------------------------------------------------------------------------
# Load + join
# ---------------------------------------------------------------------------
def jsonl(p):
    with open(p) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def uid_index(row) -> int:
    return int(row["uid"].rsplit(":", 1)[1])


def load():
    """Join results to the manifest and score every row.

    Join key: the trailing integer of `uid`. The adapter emits
    `uid=f"slidevqa:{condition}:{r.get('qa_id', i)}"`, so that integer is the
    manifest's own `qa_id`; in this manifest qa_id also happens to equal the row
    index, so the two candidate joins coincide. Verified rather than assumed --
    `verify_join` re-checks gold, deck name and evidence count on every row.
    """
    man = jsonl(MANIFEST)
    by_qa = {m["qa_id"]: m for m in man}
    ev = jsonl(EVIDENCE_JSONL)
    ap = jsonl(ALLPAGES_JSONL)
    for r in ev + ap:
        r["em"], r["f1"] = token_f1(r["pred"], r["gold"])
        r["fmt_equiv"] = format_equivalent(r["pred"], r["gold"])
        # Format-corrected twins: full credit when the answer is the same value
        # in different clothes. Never *removes* credit.
        r["emc"] = 1.0 if (r["em"] == 1 or r["fmt_equiv"]) else 0.0
        r["f1c"] = 1.0 if r["fmt_equiv"] else r["f1"]
        r["qa"] = uid_index(r)
    return man, by_qa, ev, ap


def verify_join(by_qa, rows) -> dict:
    ok = bad = 0
    for r in rows:
        m = by_qa.get(r["qa"])
        if (m and str(m["answer"]) == r["gold"][0]
                and m["deck_name"] == r["meta"]["deck"]
                and len(m["evidence_pages"]) == r["meta"]["n_evidence"]):
            ok += 1
        else:
            bad += 1
    return {"ok": ok, "bad": bad, "n": len(rows)}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def mean(rows, key):
    return 100.0 * sum(r[key] for r in rows) / len(rows) if rows else None


def stats(rows) -> dict:
    return {"f1": mean(rows, "f1"), "em": mean(rows, "em"),
            "f1c": mean(rows, "f1c"), "emc": mean(rows, "emc"), "n": len(rows)}


SLICES = [
    ("overall", "all questions", lambda r: True),
    ("single-page evidence", "answer sits on one slide", lambda r: not r["meta"]["multi_page"]),
    ("multi-page evidence", "answer spans 2+ slides", lambda r: r["meta"]["multi_page"]),
    ("plain lookup", "read a value off the slide", lambda r: not r["meta"]["arithmetic"]),
    ("needs arithmetic", "derive a value not printed anywhere", lambda r: r["meta"]["arithmetic"]),
]


def cell_name(r) -> str:
    return ("multi-page" if r["meta"]["multi_page"] else "single-page") + " / " + \
           ("arithmetic" if r["meta"]["arithmetic"] else "lookup")


def analyse(ev, ap) -> dict:
    E = {r["qa"]: r for r in ev}
    A = {r["qa"]: r for r in ap}
    common = sorted(set(E) & set(A))
    pe = [E[k] for k in common]
    pa = [A[k] for k in common]

    out = {
        "n_evidence_rows": len(ev), "n_allpages_rows": len(ap), "n_paired": len(common),
        "allpages_unpaired": len(set(A) - set(E)),
        "full": [{"slice": name, "note": note, **stats([r for r in ev if f(r)])}
                 for name, note, f in SLICES],
        "paired_overall": {"evidence": stats(pe), "allpages": stats(pa)},
        "arith_share_paired": sum(r["meta"]["arithmetic"] for r in pe) / len(pe),
        "arith_share_full": sum(r["meta"]["arithmetic"] for r in ev) / len(ev),
    }

    # per-cell paired gaps
    cells = []
    for mp in (True, False):
        for ar in (True, False):
            ks = [k for k in common if E[k]["meta"]["multi_page"] == mp
                  and E[k]["meta"]["arithmetic"] == ar]
            if not ks:
                continue
            a, b = stats([E[k] for k in ks]), stats([A[k] for k in ks])
            cells.append({
                "cell": ("multi-page" if mp else "single-page") + " / " + ("arithmetic" if ar else "lookup"),
                "evidence": a["f1"], "allpages": b["f1"], "gap": b["f1"] - a["f1"],
                "evidence_c": a["f1c"], "allpages_c": b["f1c"], "gap_c": b["f1c"] - a["f1c"],
                "n": len(ks)})
    cells.sort(key=lambda c: c["gap"])
    out["cells"] = cells

    # the three costs, all in F1 points
    by = {name: stats([r for r in ev if f(r)]) for name, _, f in SLICES}
    po = out["paired_overall"]
    out["costs"] = [
        {"name": "retrieval",
         "what": "find the evidence among 20 slides instead of being handed it",
         "cost": po["allpages"]["f1"] - po["evidence"]["f1"],
         "cost_c": po["allpages"]["f1c"] - po["evidence"]["f1c"],
         "basis": f"paired, n={len(common)}"},
        {"name": "integration",
         "what": "combine two slides instead of reading one",
         "cost": by["multi-page evidence"]["f1"] - by["single-page evidence"]["f1"],
         "cost_c": by["multi-page evidence"]["f1c"] - by["single-page evidence"]["f1c"],
         "basis": f"evidence condition, n={by['multi-page evidence']['n']} vs {by['single-page evidence']['n']}"},
        {"name": "derivation",
         "what": "compute on what was read instead of quoting it",
         "cost": by["needs arithmetic"]["f1"] - by["plain lookup"]["f1"],
         "cost_c": by["needs arithmetic"]["f1c"] - by["plain lookup"]["f1c"],
         "basis": f"evidence condition, n={by['needs arithmetic']['n']} vs {by['plain lookup']['n']}"},
    ]

    # divergence buckets on the paired set
    div = Counter()
    div_c = Counter()
    for k in common:
        div[("ev_ok" if E[k]["em"] == 1 else "ev_no") + "|" + ("ap_ok" if A[k]["em"] == 1 else "ap_no")] += 1
        div_c[("ev_ok" if E[k]["emc"] == 1 else "ev_no") + "|" + ("ap_ok" if A[k]["emc"] == 1 else "ap_no")] += 1
    out["divergence"] = dict(div)
    out["divergence_c"] = dict(div_c)

    # format-only artifact rate
    non_em = [r for r in ev if r["em"] == 0]
    zero_f1 = [r for r in ev if r["f1"] == 0]
    out["format"] = {
        "non_em": len(non_em),
        "non_em_fmt": sum(r["fmt_equiv"] for r in non_em),
        "zero_f1": len(zero_f1),
        "zero_f1_fmt": sum(r["fmt_equiv"] for r in zero_f1),
        "by_slice": [{"slice": name,
                      "rate": 100.0 * sum(1 for r in ev if f(r) and r["em"] == 0 and r["fmt_equiv"])
                              / max(1, sum(1 for r in ev if f(r))),
                      "n": sum(1 for r in ev if f(r))}
                     for name, _, f in SLICES],
    }

    # F1 histogram, both conditions, paired only so the comparison is fair
    edges = [0.0, 0.001, 0.2, 0.4, 0.6, 0.8, 0.999, 1.01]
    labels = ["0", "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-<1", "1.0"]

    def hist(rows, key="f1"):
        c = [0] * len(labels)
        for r in rows:
            v = r[key]
            for i in range(len(labels)):
                if edges[i] <= v < edges[i + 1]:
                    c[i] += 1
                    break
        return c

    out["hist"] = {"labels": labels, "evidence": hist(pe), "allpages": hist(pa),
                   "evidence_c": hist(pe, "f1c"), "allpages_c": hist(pa, "f1c"), "n": len(pe)}

    # F1 by number of evidence pages
    byn = []
    for n_ev in sorted({r["meta"]["n_evidence"] for r in ev}):
        rows = [r for r in ev if r["meta"]["n_evidence"] == n_ev]
        byn.append({"n_evidence": n_ev, **stats(rows)})
    out["by_n_evidence"] = byn

    # F1 by gold answer length in tokens
    bylen = []
    buckets = [(1, 1, "1 token"), (2, 2, "2 tokens"), (3, 4, "3-4 tokens"), (5, 99, "5+ tokens")]
    for lo, hi, lab in buckets:
        rows = [r for r in ev if lo <= len(str(r["gold"][0]).split()) <= hi]
        if rows:
            bylen.append({"label": lab, **stats(rows)})
    out["by_gold_len"] = bylen

    # cost of the extra 18 slides, in latency and tokens
    out["cost_tokens"] = {
        c: {"latency": sum(r["latency_s"] for r in R) / len(R),
            "in_tok": sum(r["usage"]["input_tokens"] for r in R) / len(R),
            "out_tok": sum(r["usage"]["output_tokens"] for r in R) / len(R)}
        for c, R in (("evidence", pe), ("allpages", pa))}

    return out, E, A, common


def analyse_arithmetic(ev, by_qa) -> dict:
    c = Counter()
    for r in ev:
        if not r["meta"]["arithmetic"]:
            continue
        m = by_qa[r["qa"]]
        r["arith_class"] = classify_arithmetic(r, m.get("arithmetic_expression"))
        c[r["arith_class"]] += 1
    # how often the annotated expression does not evaluate to the annotated gold
    bad_gold = 0
    for r in ev:
        if not r["meta"]["arithmetic"]:
            continue
        m = _EXPR_RE.fullmatch(str(by_qa[r["qa"]].get("arithmetic_expression") or "").replace(",", "").strip())
        if not m:
            continue
        a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
        want = {"-": a - b, "+": a + b, "*": a * b, "/": (a / b if b else None)}[op]
        g = as_float(r["gold"][0])
        if g is not None and want is not None and abs(g - want) > 0.02 * max(1.0, abs(want)):
            bad_gold += 1
    total = sum(c.values())
    return {"counts": dict(c), "n": total, "bad_gold": bad_gold}


# ---------------------------------------------------------------------------
# Example selection -- weighted to failures on purpose.
#
# A highlight reel of successes would say nothing. Every bucket below is a
# failure mode except the last, which exists so the failures have a baseline to
# be read against.
# ---------------------------------------------------------------------------
BUCKETS = [
    ("retrieval_fail", "Retrieval failure",
     "Correct when handed the evidence slide, wrong when made to find it among 20. "
     "This is the cleanest evidence of a retrieval blind spot."),
    ("allpages_only", "All-pages-only win",
     "Wrong on the evidence slides alone, right on the full deck. Mostly noise, "
     "but a few are cases where surrounding slides supplied a missing unit or label."),
    ("both_wrong", "Wrong in both conditions",
     "Retrieval was never the problem -- the model fails these with the evidence in hand."),
    ("wrong_operand", "Arithmetic: wrong operand",
     "The annotated expression's operands do not both appear in the trace. The model "
     "misread a number off the slide, then computed correctly on the wrong input."),
    ("wrong_operation", "Arithmetic: wrong operation",
     "Both operands appear in the trace, so the reading was right and the computation "
     "was not. A derivation failure with perception intact."),
    ("integration_fail", "Multi-page integration failure",
     "Two evidence slides were supplied and the answer is wrong. Where the trace names "
     "slides, check whether it ever mentions the second one."),
    ("format_only", "Format-only failure (metric artifact)",
     "Scored zero, semantically correct. \"22%\" against a gold of \"22\". These are not "
     "perception failures and are counted separately throughout this page."),
    ("clean_success", "Clean success (control)",
     "Correct in both conditions. Included so the failures above have a baseline."),
]

BUCKET_TARGET = 25


def select_examples(ev, ap, E, A, common, by_qa) -> list[dict]:
    picked = defaultdict(list)

    def add(bucket, rows, limit=BUCKET_TARGET):
        # Spread across decks so one deck cannot dominate a bucket.
        rows = sorted(rows, key=lambda r: (r["meta"]["deck"], r["qa"]))
        seen = Counter()
        rows.sort(key=lambda r: seen[r["meta"]["deck"]])
        out, per_deck = [], Counter()
        for r in rows:
            if per_deck[r["meta"]["deck"]] >= 3:
                continue
            per_deck[r["meta"]["deck"]] += 1
            out.append(r)
            if len(out) >= limit:
                break
        if len(out) < limit:  # relax the per-deck cap rather than under-fill
            for r in rows:
                if r not in out:
                    out.append(r)
                if len(out) >= limit:
                    break
        picked[bucket] = out

    add("retrieval_fail", [E[k] for k in common if E[k]["emc"] == 1 and A[k]["emc"] == 0])
    add("allpages_only", [E[k] for k in common if E[k]["emc"] == 0 and A[k]["emc"] == 1])
    add("both_wrong", [E[k] for k in common if E[k]["emc"] == 0 and A[k]["emc"] == 0])
    add("wrong_operand", [r for r in ev if r.get("arith_class") == "wrong_operand"])
    add("wrong_operation", [r for r in ev if r.get("arith_class") == "wrong_operation"])
    add("integration_fail", [r for r in ev if r["meta"]["multi_page"] and r["emc"] == 0
                             and not r["meta"]["arithmetic"]])
    add("format_only", [r for r in ev if r["em"] == 0 and r["fmt_equiv"]])
    add("clean_success", [E[k] for k in common if E[k]["em"] == 1 and A[k]["em"] == 1], 20)

    # one record per question, carrying every bucket it belongs to
    tags = defaultdict(list)
    for b, rows in picked.items():
        for r in rows:
            tags[r["qa"]].append(b)

    examples = []
    for qa in sorted(tags):
        e = E.get(qa)
        if e is None:
            continue
        m = by_qa[qa]
        a = A.get(qa)
        expr = m.get("arithmetic_expression")
        expr = None if expr in (None, "None", "") else expr
        ev_pages = [int(x) for x in m["evidence_pages"] if 1 <= int(x) <= N_PAGES]
        rec = {
            "qa": qa, "buckets": tags[qa], "deck": m["deck_name"], "deck_url": m.get("deck_url", ""),
            "q": m["question"], "gold": e["gold"][0], "expr": expr,
            "ev_pages": ev_pages, "cell": cell_name(e),
            "multi": bool(e["meta"]["multi_page"]), "arith": bool(e["meta"]["arithmetic"]),
            "arith_class": e.get("arith_class"),
            "ev": cond_record(e, ev_pages),
        }
        rec["ap"] = cond_record(a, ev_pages) if a else None
        examples.append(rec)
    return examples, {b: len(v) for b, v in picked.items()}


def cond_record(r, ev_pages) -> dict:
    cited = cited_slides(r["thinking"])
    return {
        "pred": r["pred"], "f1": round(r["f1"], 3), "em": r["em"],
        "f1c": round(r["f1c"], 3), "emc": r["emc"], "fmt": bool(r["fmt_equiv"]),
        "thinking": r["thinking"] or "",
        "cited": cited,
        "cited_ok": bool(cited) and bool(set(cited) & set(ev_pages)),
        "n_sent": r["meta"]["n_pages_sent"],
        "lat": round(r["latency_s"], 2),
        "in_tok": r["usage"]["input_tokens"], "out_tok": r["usage"]["output_tokens"],
    }


# ---------------------------------------------------------------------------
# Slide rendering.
#
# The manifest stores a private copy of all 20 slides per question, so the same
# deck is on disk 2-6 times over (44,300 files, 4.8 GB). The copies are
# byte-identical, so slides are keyed by (deck, page) and rendered once.
# ---------------------------------------------------------------------------
def deck_key(deck: str) -> str:
    return hashlib.md5(deck.encode()).hexdigest()[:10]


def asset_names(deck: str, page: int) -> tuple[str, str]:
    k = deck_key(deck)
    return f"{k}_{page:02d}_t.jpg", f"{k}_{page:02d}_f.jpg"


def _render_one(job):
    src, thumb, full = job
    from PIL import Image
    try:
        im = Image.open(src)
        im.load()
    except Exception as exc:  # a missing slide must not sink the whole page
        return f"{src}: {exc}"
    if im.mode != "RGB":
        im = im.convert("RGB")
    w, h = im.size
    if not Path(full).exists():
        sc = min(1.0, FULL_EDGE / max(w, h))
        f = im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.LANCZOS) if sc < 1 else im
        f.save(full, "JPEG", quality=FULL_Q, optimize=True, progressive=True)
    if not Path(thumb).exists():
        sc = min(1.0, THUMB_W / w)
        t = im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.LANCZOS) if sc < 1 else im
        t.save(thumb, "JPEG", quality=THUMB_Q, optimize=True)
    return None


def render_slides(examples, by_qa, workers=None):
    """One job per unique (deck, page). Pure CPU -- processes, not threads."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    jobs, seen = [], set()
    for rec in examples:
        m = by_qa[rec["qa"]]
        for p in range(1, N_PAGES + 1):
            rel = m.get(f"page_{p}")
            if not rel:
                continue
            key = (rec["deck"], p)
            if key in seen:
                continue
            seen.add(key)
            t, f = asset_names(rec["deck"], p)
            jobs.append((str(DATA / rel), str(ASSETS / t), str(ASSETS / f)))
    errs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for e in pool.map(_render_one, jobs, chunksize=8):
            if e:
                errs.append(e)
    return len(jobs), errs


# ---------------------------------------------------------------------------
# Charts. House style: recessive grid, thin marks, 4px rounded data-ends,
# a 2px surface gap between adjacent bars, legend whenever there are 2 series,
# and every value direct-labelled in ink -- never in the series colour.
#
# The two-hue categorical pair (--s1 blue / --s2 orange) was checked against the
# six-check validator in both modes: chroma >= 0.16 (floor 0.10), contrast 3.1-4.8
# vs surface (min 3.0), and adjacent OKLab dE of 24.7 protan / 31.7 deutan
# (target 8). It passes, so it is used unchanged.
# ---------------------------------------------------------------------------
def fmtv(v, d=1):
    return "&mdash;" if v is None else f"{v:.{d}f}"


def grouped_bars(title, sub, rows, series=("evidence", "all 20 slides"), vmax=100.0, unit=""):
    """rows: (label, v1, v2, n, tip). Two series, one axis, legend + direct labels."""
    body = []
    for lab, v1, v2, n, tip in rows:
        nlab = "" if n == "" else (n if isinstance(n, str) else f"n={n}")
        bars = ""
        for v, cls in ((v1, ""), (v2, " s2")):
            w = 0 if v is None else max(v / vmax * 100, 0.7)
            bars += (f'<div class="sb"><div class="b{cls}" style="width:{w:.2f}%"></div>'
                     f'<span class="t">{fmtv(v)}{unit}</span></div>')
        body.append(f'<div class="split" tabindex="0" data-tip="{esc(tip)}">'
                    f'<div class="rlab">{lab}<span class="nlab">{esc(nlab)}</span></div>'
                    f'<div class="splitbars">{bars}</div></div>')
    leg = (f'<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>{esc(series[0])}</span>'
           f'<span><i class="sw" style="background:var(--s2)"></i>{esc(series[1])}</span></div>')
    return (f'<div class="card"><h3>{title}</h3><p class="sub">{sub}</p>{leg}{"".join(body)}</div>')


def single_bars(title, sub, rows, vmax=100.0, unit="", tone=None):
    body = []
    for lab, v, n, tip in rows:
        w = 0 if v is None else max(v / vmax * 100, 0.7)
        cls = "" if tone is None else f" {tone(v)}"
        body.append(f'<div class="row" tabindex="0" data-tip="{esc(tip)}">'
                    f'<div class="rlab">{lab}</div>'
                    f'<div class="track"><div class="bar{cls}" style="width:{w:.2f}%"></div></div>'
                    f'<div class="rval">{fmtv(v)}{unit}<span class="nlab">n={n}</span></div></div>')
    return f'<div class="card"><h3>{title}</h3><p class="sub">{sub}</p>{"".join(body)}</div>'


def diverging_bars(title, sub, rows, unit=" F1"):
    """rows: (label, value, n, tip). Signed magnitude around a zero line.

    Two poles + a neutral zero: negative uses the `bad` status token, positive
    `good`. These are status, not identity -- the sign *means* worse/better.
    """
    lim = max([abs(v) for _, v, _, _ in rows] + [1.0]) * 1.15
    body = []
    for lab, v, n, tip in rows:
        w = abs(v) / lim * 50.0
        neg = v < 0
        style = (f"right:50%;width:{w:.2f}%;background:var(--bad);border-radius:4px 0 0 4px"
                 if neg else
                 f"left:50%;width:{w:.2f}%;background:var(--good);border-radius:0 4px 4px 0")
        body.append(
            f'<div class="row" tabindex="0" data-tip="{esc(tip)}">'
            f'<div class="rlab">{lab}<span class="nlab">n={n}</span></div>'
            f'<div class="track div"><div class="zero"></div>'
            f'<div class="dbar" style="{style}"></div></div>'
            f'<div class="rval {"neg" if neg else "pos"}">{v:+.1f}{unit}</div></div>')
    return (f'<div class="card"><h3>{title}</h3><p class="sub">{sub}</p>{"".join(body)}'
            f'<p class="axnote">worse with distractors &larr; &nbsp;0&nbsp; &rarr; better with distractors</p></div>')


def table(headers, rows, first_is_row_header=True):
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            tag = "th" if (i == 0 and first_is_row_header) else "td"
            scope = ' scope="row"' if tag == "th" else ""
            cells.append(f"<{tag}{scope}>{c}</{tag}>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def note(text):
    return f'<div class="note">{text}</div>'


def whatmeans(text):
    return f'<p class="wm"><span>What this means</span>{text}</p>'


# ---------------------------------------------------------------------------
# Styling.
#
# Every custom property is declared on :root. This is not decorative: custom
# properties inherit downward only, so declaring them on a wrapper class leaves
# `body` unable to see them and the page renders black-on-black in dark mode.
# Colour variables live on :root, on all three theme branches, always.
# ---------------------------------------------------------------------------
CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
 --muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
 --s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;--bad:#d03b3b;--warn:#fab219;
 --evring:#2a78d6;--shade:rgba(11,11,11,.05)}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
 --surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;
 --good:#3fbf3f;--bad:#e46060;--evring:#3987e5;--shade:rgba(255,255,255,.05)}}
:root[data-theme=dark]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;
 --ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
 --s1:#3987e5;--s2:#d95926;--good:#3fbf3f;--bad:#e46060;--evring:#3987e5;
 --shade:rgba(255,255,255,.05)}
*{box-sizing:border-box}
html,body{background:var(--page);color:var(--ink);margin:0;
 font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:30px 22px 90px}
header{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}
h1{font-size:26px;margin:0 0 6px}
.dek{color:var(--ink2);margin:0;max-width:74ch}
.crumb{font-size:12.5px;color:var(--muted);margin:0 0 14px}
.crumb a{color:var(--s1)}
h2{font-size:19px;margin:44px 0 4px;padding-top:20px;border-top:1px solid var(--grid)}
h2 .sub{display:block;font-size:13.5px;font-weight:400;color:var(--ink2);margin-top:5px;max-width:82ch}
button,select,input{font:inherit;font-size:13px;padding:6px 10px;border-radius:8px;
 border:1px solid var(--border);background:var(--surface);color:var(--ink)}
button{cursor:pointer;color:var(--ink2)}
button:hover{border-color:var(--axis)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:15px 16px}
.tlab{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.tval{font-size:31px;line-height:1.1;margin:7px 0 3px;font-variant-numeric:tabular-nums}
.tile.bad .tval{color:var(--bad)}.tile.good .tval{color:var(--good)}
.tnote{font-size:12.5px;color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
 padding:18px 20px 20px;margin:16px 0}
.card h3{font-size:15.5px;margin:0 0 3px}
.card .sub{font-size:13px;color:var(--ink2);margin:0 0 15px;max-width:84ch}
.row{display:grid;grid-template-columns:250px 1fr 118px;align-items:center;gap:12px;padding:5px 0}
.split{display:grid;grid-template-columns:250px 1fr;gap:12px;padding:7px 0;align-items:center}
.rlab{font-size:12.5px;color:var(--ink2);text-align:right;overflow-wrap:anywhere}
.nlab{display:block;font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}
.track{height:15px;background:var(--grid);border-radius:4px;position:relative}
.track.div{overflow:visible}
.zero{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--axis)}
.dbar{position:absolute;top:0;bottom:0}
.bar{height:100%;background:var(--s1);border-radius:0 4px 4px 0}
.bar.s2{background:var(--s2)}
.bar.good{background:var(--good)}.bar.bad{background:var(--bad)}
.splitbars{display:flex;flex-direction:column;gap:2px}
.sb{display:flex;align-items:center;gap:8px;height:14px}
.sb .b{height:100%;border-radius:0 4px 4px 0;min-width:2px;background:var(--s1)}
.sb .b.s2{background:var(--s2)}
.sb .t{font-size:11.5px;color:var(--ink2);font-variant-numeric:tabular-nums;white-space:nowrap}
.rval{font-size:13px;line-height:1.35;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;color:var(--ink)}
.rval.neg{color:var(--bad)}.rval.pos{color:var(--good)}
.axnote{font-size:11.5px;color:var(--muted);text-align:center;margin:12px 0 0}
.legend{display:flex;gap:18px;margin:0 0 12px 262px;font-size:12.5px;color:var(--ink2)}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
.note{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--s2);
 border-radius:9px;padding:13px 16px;font-size:13.5px;color:var(--ink2);margin:16px 0}
.note strong{color:var(--ink)}
.wm{font-size:13.5px;color:var(--ink2);max-width:88ch;margin:10px 0 0;
 border-left:2px solid var(--grid);padding-left:14px}
.wm span{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
 color:var(--muted);margin-bottom:3px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 16px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--grid);color:var(--ink)}
td{font-variant-numeric:tabular-nums}
th{color:var(--ink2);font-weight:600}
th[scope=row]{font-weight:400;color:var(--ink2)}
tr.hi td,tr.hi th{background:var(--shade);font-weight:600;color:var(--ink)}
a{color:var(--s1)}
code{font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--shade);
 padding:1px 5px;border-radius:4px;color:var(--ink)}

/* ---- explorer ---- */
.filters{display:flex;gap:10px;flex-wrap:wrap;align-items:center;position:sticky;top:0;z-index:60;
 padding:12px 14px;margin:16px 0;background:var(--surface);border:1px solid var(--border);
 border-radius:11px;box-shadow:0 1px 0 var(--border)}
.filters label{font-size:12px;color:var(--muted);display:flex;flex-direction:column;gap:4px}
.filters input[type=search]{min-width:220px}
.fcount{margin-left:auto;font-size:12.5px;color:var(--ink2);font-variant-numeric:tabular-nums}
.case{background:var(--surface);border:1px solid var(--border);border-radius:12px;
 padding:15px 16px;margin-bottom:14px}
.chd{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:9px}
.pill{font-size:11px;font-weight:600;padding:2px 9px;border-radius:999px;white-space:nowrap}
.ok{background:color-mix(in srgb,var(--good) 16%,var(--surface));color:var(--good);
 border:1px solid color-mix(in srgb,var(--good) 35%,transparent)}
.no{background:color-mix(in srgb,var(--bad) 16%,var(--surface));color:var(--bad);
 border:1px solid color-mix(in srgb,var(--bad) 35%,transparent)}
.part{background:color-mix(in srgb,var(--warn) 20%,var(--surface));color:var(--ink);
 border:1px solid color-mix(in srgb,var(--warn) 45%,transparent)}
.tag{font-size:11px;color:var(--ink2);border:1px solid var(--border);padding:2px 8px;
 border-radius:999px;background:var(--page)}
.tag.b{border-color:color-mix(in srgb,var(--s2) 45%,transparent);color:var(--ink)}
.qtext{font-size:14.5px;line-height:1.45;margin:2px 0 8px;color:var(--ink)}
.meta{font-size:12.5px;color:var(--ink2);margin:0 0 10px;display:flex;gap:16px;flex-wrap:wrap}
.meta b{color:var(--ink);font-weight:600}
.striphd{display:flex;align-items:center;gap:10px;margin:4px 0 6px}
.striphd .lab{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.strip{display:flex;gap:6px;overflow-x:auto;padding:4px 2px 10px;scroll-behavior:smooth}
.strip::-webkit-scrollbar{height:8px}
.strip::-webkit-scrollbar-thumb{background:var(--axis);border-radius:99px}
.sl{flex:0 0 auto;width:112px;border:2px solid transparent;border-radius:7px;padding:0;
 background:var(--grid);cursor:pointer;position:relative;overflow:hidden;line-height:0}
.sl img{width:100%;display:block;aspect-ratio:4/3;object-fit:contain;background:var(--page);
 filter:grayscale(.85) opacity(.5);transition:filter .12s}
.sl:hover img,.sl:focus-visible img{filter:none}
.sl .pn{position:absolute;left:3px;top:3px;font-size:10px;line-height:1;padding:2px 5px;
 border-radius:4px;background:rgba(0,0,0,.62);color:#fff;font-variant-numeric:tabular-nums}
.sl.ev{border-color:var(--evring);box-shadow:0 0 0 2px color-mix(in srgb,var(--evring) 28%,transparent)}
.sl.ev img{filter:none}
.sl.ev .pn{background:var(--evring)}
.sl .badge{position:absolute;right:3px;bottom:3px;font-size:9.5px;font-weight:700;
 letter-spacing:.04em;padding:2px 6px;border-radius:4px;background:var(--evring);color:#fff}
.sl:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
.cmp{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}
.cmp.one{grid-template-columns:1fr}
@media(max-width:820px){.cmp{grid-template-columns:1fr}}
.cond{border:1px solid var(--border);border-radius:10px;padding:11px 13px;background:var(--page)}
.cond h4{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
 display:flex;align-items:center;gap:8px}
.cond h4 .dot{width:9px;height:9px;border-radius:3px;flex:0 0 auto}
.pred{font-size:14px;margin:0 0 7px;overflow-wrap:anywhere;color:var(--ink)}
.pred .k{font-size:11px;color:var(--muted);display:block;margin-bottom:2px}
.sc{font-size:12px;color:var(--ink2);font-variant-numeric:tabular-nums;display:flex;gap:12px;flex-wrap:wrap}
.cite{font-size:12px;color:var(--ink2);margin-top:7px}
.cite b{color:var(--ink)}
details.think{margin-top:9px;border-top:1px solid var(--grid);padding-top:7px}
details.think summary{cursor:pointer;font-size:12px;color:var(--ink2);list-style:none}
details.think summary::-webkit-details-marker{display:none}
details.think summary::before{content:"\\25B8 ";color:var(--muted)}
details.think[open] summary::before{content:"\\25BE "}
.trace{font:12.5px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;
 background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 12px;
 margin-top:7px;max-height:230px;overflow:auto;color:var(--ink2)}
.trace mark{background:color-mix(in srgb,var(--warn) 45%,transparent);color:var(--ink);
 border-radius:3px;padding:0 2px}
.pager{display:flex;gap:8px;align-items:center;justify-content:center;margin:20px 0 0;
 font-size:13px;color:var(--ink2)}
.pager button[disabled]{opacity:.4;cursor:default}
.empty{padding:40px;text-align:center;color:var(--muted);font-size:14px}
.bhd{font-size:12.5px;color:var(--ink2);margin:26px 0 8px;padding:9px 12px;background:var(--surface);
 border:1px solid var(--border);border-left:3px solid var(--s2);border-radius:9px}
.bhd b{color:var(--ink)}

/* ---- lightbox ---- */
#lb{position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.94);display:none}
#lb.on{display:block}
#lb .stage{position:absolute;inset:0;overflow:hidden;cursor:grab;touch-action:none}
#lb .stage.grabbing{cursor:grabbing}
#lb img{position:absolute;top:0;left:0;transform-origin:0 0;will-change:transform;
 max-width:none;max-height:none;user-select:none;-webkit-user-drag:none}
#lb .ctrl{position:fixed;top:12px;left:50%;transform:translateX(-50%);display:flex;gap:6px;
 align-items:center;z-index:2;background:rgba(20,20,20,.9);padding:6px 8px;border-radius:10px;
 max-width:calc(100vw - 20px);flex-wrap:wrap;justify-content:center}
#lb .ctrl button{font:inherit;font-size:13px;line-height:1;min-width:34px;padding:8px 10px;
 border-radius:7px;border:1px solid rgba(255,255,255,.2);background:#242423;color:#eee;cursor:pointer}
#lb .ctrl button:hover{background:#343433}
#lb .ctrl button.jump{background:var(--s1);border-color:transparent;color:#fff;font-weight:600}
#lb .ctrl .lvl,#lb .ctrl .pos{color:#c3c2b7;font-size:12.5px;min-width:56px;text-align:center;
 font-variant-numeric:tabular-nums}
#lb .cap{position:fixed;top:62px;left:50%;transform:translateX(-50%);z-index:2;
 background:rgba(20,20,20,.9);color:#e4e3de;font-size:12.5px;padding:6px 13px;border-radius:999px;
 max-width:calc(100vw - 40px);text-align:center}
#lb .cap .evb{color:#fff;background:var(--s1);border-radius:4px;padding:1px 7px;
 font-weight:700;font-size:11px;margin-right:7px}
#lb .hint{position:fixed;bottom:12px;left:50%;transform:translateX(-50%);color:#a3a19b;
 font-size:12px;z-index:2;background:rgba(20,20,20,.85);padding:6px 13px;border-radius:999px;
 text-align:center;max-width:calc(100vw - 30px)}
#tip{position:fixed;z-index:99;pointer-events:none;opacity:0;transition:opacity .1s;
 background:var(--ink);color:var(--page);font-size:12px;padding:6px 9px;border-radius:6px;max-width:300px}
"""


# ---------------------------------------------------------------------------
# Client-side explorer.
#
# ~170 examples x 20 slides is 3,400 potential <img> nodes. Only the current
# page of 8 is ever in the DOM, and its thumbnails are lazy-loaded, so scrolling
# stays cheap and the browser never fetches a slide the user has not looked at.
# ---------------------------------------------------------------------------
JS = r"""
const PAGE_SIZE = 8;
const $ = s => document.querySelector(s);
const pad2 = n => String(n).padStart(2, '0');
const thumbSrc = (dk, p) => `assets_slidevqa/${dk}_${pad2(p)}_t.jpg`;
const fullSrc  = (dk, p) => `assets_slidevqa/${dk}_${pad2(p)}_f.jpg`;
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

/* ---------- theme ---------- */
const tbtn = $('button.theme');
tbtn.addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme === 'dark';
  document.documentElement.dataset.theme = dark ? 'light' : 'dark';
  tbtn.textContent = dark ? 'Dark mode' : 'Light mode';
});

/* ---------- chart tooltips ---------- */
const tip = $('#tip');
document.querySelectorAll('[data-tip]').forEach(el => {
  const show = () => { tip.innerHTML = el.dataset.tip; tip.style.opacity = 1;
    const r = el.getBoundingClientRect();
    tip.style.left = Math.min(innerWidth - 320, Math.max(8, r.left + 12)) + 'px';
    tip.style.top = Math.max(8, r.top - 40) + 'px'; };
  const hide = () => tip.style.opacity = 0;
  el.addEventListener('mouseenter', show); el.addEventListener('mouseleave', hide);
  el.addEventListener('focus', show); el.addEventListener('blur', hide);
});

/* ---------- outcome helpers ---------- */
function outcome(c) {
  if (!c) return 'na';
  if (c.em === 1) return 'correct';
  if (c.f1 > 0) return 'partial';
  return 'wrong';
}
function divergence(r) {
  if (!r.ap) return 'unpaired';
  const e = r.ev.emc === 1, a = r.ap.emc === 1;
  if (e && a) return 'both_ok';
  if (e && !a) return 'ret_fail';
  if (!e && a) return 'ap_only';
  return 'both_no';
}

/* ---------- filtering ---------- */
let filtered = [], page = 0;
function readFilters() {
  return {
    ev: $('#f-ev').value, kind: $('#f-kind').value, out: $('#f-out').value,
    div: $('#f-div').value, bucket: $('#f-bucket').value,
    q: $('#f-q').value.trim().toLowerCase(),
  };
}
function applyFilters() {
  const f = readFilters();
  filtered = EX.filter(r => {
    if (f.ev === 'single' && r.multi) return false;
    if (f.ev === 'multi' && !r.multi) return false;
    if (f.kind === 'lookup' && r.arith) return false;
    if (f.kind === 'arith' && !r.arith) return false;
    if (f.out !== 'all' && outcome(r.ev) !== f.out) return false;
    if (f.div !== 'all' && divergence(r) !== f.div) return false;
    if (f.bucket !== 'all' && !r.buckets.includes(f.bucket)) return false;
    if (f.q) {
      const hay = (r.q + ' ' + r.gold + ' ' + r.ev.pred + ' ' + (r.ap ? r.ap.pred : '')
                   + ' ' + r.deck).toLowerCase();
      if (!hay.includes(f.q)) return false;
    }
    return true;
  });
  page = 0;
  render();
}
document.querySelectorAll('.filters select').forEach(s => s.addEventListener('change', applyFilters));
$('#f-q').addEventListener('input', applyFilters);
$('#f-reset').addEventListener('click', () => {
  document.querySelectorAll('.filters select').forEach(s => s.value = 'all');
  $('#f-q').value = ''; applyFilters();
});

/* ---------- rendering ---------- */
function scorePills(c) {
  const o = outcome(c);
  const cls = o === 'correct' ? 'ok' : (o === 'partial' ? 'part' : 'no');
  const lab = o === 'correct' ? 'EM' : (o === 'partial' ? 'partial' : 'F1 0');
  let s = `<span class="pill ${cls}">${lab}</span>`;
  if (c.fmt && c.em !== 1) s += `<span class="pill part" title="same value, different formatting">format-only</span>`;
  return s;
}
function traceHTML(c, r) {
  if (!c.thinking) return '';
  const marked = esc(c.thinking)
    .replace(/\b((?:slide|page|image)\s*#?\s*\d{1,2})\b/gi, '<mark>$1</mark>')
    .replace(/\b((?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|last)\s+(?:slide|page|image))\b/gi, '<mark>$1</mark>');
  const n = c.thinking.length;
  return `<details class="think"><summary>thinking trace &middot; ${n} chars`
       + `${c.cited.length ? ' &middot; names slide ' + c.cited.join(', ') : ' &middot; names no slide explicitly'}`
       + `</summary><div class="trace">${marked}</div>`
       + `<button class="expand" type="button">expand / collapse box</button></details>`;
}
function condHTML(c, r, which) {
  if (!c) return '';
  const isEv = which === 'ev';
  const dot = isEv ? 'var(--s1)' : 'var(--s2)';
  const title = isEv ? `evidence only &mdash; ${c.n_sent} slide${c.n_sent > 1 ? 's' : ''} sent`
                     : `all pages &mdash; ${c.n_sent} slides sent`;
  let cite = '';
  if (!isEv) {
    if (c.cited.length) {
      let verdict;
      if (c.cited_ok) verdict = ' &middot; overlaps';
      else if (c.em === 1) verdict = ' &middot; no overlap, but answered correctly &mdash; the trace is probably naming a number printed on the slide, not a slide index';
      else verdict = ' &middot; <b style="color:var(--bad)">no overlap &mdash; read the wrong slide</b>';
      cite = `<div class="cite">trace names slide(s) <b>${c.cited.join(', ')}</b> &middot; evidence is <b>${r.ev_pages.join(', ')}</b>` + verdict + '</div>';
    } else {
      cite = `<div class="cite">trace names no slide explicitly &mdash; which slide it read is not recoverable</div>`;
    }
  }
  return `<div class="cond"><h4><span class="dot" style="background:${dot}"></span>${title}</h4>`
       + `<p class="pred"><span class="k">prediction</span>${esc(c.pred) || '<i>(empty)</i>'}</p>`
       + `<div class="sc"><span>F1 <b>${c.f1.toFixed(2)}</b></span><span>EM <b>${c.em.toFixed(0)}</b></span>`
       + `<span>${c.lat}s</span><span>${c.in_tok.toLocaleString()} in / ${c.out_tok} out</span></div>`
       + scorePills(c) + cite + traceHTML(c, r) + `</div>`;
}
function stripHTML(r) {
  const ev = new Set(r.ev_pages);
  let s = '';
  for (let p = 1; p <= 20; p++) {
    const isEv = ev.has(p);
    s += `<button class="sl${isEv ? ' ev' : ''}" data-dk="${r.dk}" data-p="${p}" data-qa="${r.qa}"`
       + ` title="slide ${p}${isEv ? ' — evidence' : ''}">`
       + `<img loading="lazy" src="${thumbSrc(r.dk, p)}" alt="slide ${p}">`
       + `<span class="pn">${p}</span>${isEv ? '<span class="badge">EVIDENCE</span>' : ''}</button>`;
  }
  return s;
}
function caseHTML(r) {
  const tags = r.buckets.map(b => `<span class="tag b">${esc(BUCKET_LABEL[b] || b)}</span>`).join('');
  const d = divergence(r);
  const dlab = {ret_fail: 'evidence right &rarr; all-pages wrong', both_ok: 'both right',
                both_no: 'both wrong', ap_only: 'all-pages right &rarr; evidence wrong',
                unpaired: 'evidence condition only'}[d];
  return `<article class="case" data-qa="${r.qa}">
    <div class="chd"><span class="tag">#${r.qa}</span><span class="tag">${esc(r.cell)}</span>
      <span class="tag">${esc(dlab)}</span>${tags}</div>
    <p class="qtext">${esc(r.q)}</p>
    <p class="meta"><span>gold <b>${esc(r.gold)}</b></span>
      ${r.expr ? `<span>intended computation <b><code>${esc(r.expr)}</code></b></span>` : ''}
      ${r.arith_class && r.arith_class !== 'exact' ? `<span>arithmetic error <b>${esc(r.arith_class.replace(/_/g, ' '))}</b></span>` : ''}
      <span>evidence slide(s) <b>${r.ev_pages.join(', ')}</b> of 20</span>
      <span>deck ${r.deck_url ? `<a href="${esc(r.deck_url)}" target="_blank" rel="noopener">source</a>` : esc(r.deck)}</span></p>
    <div class="striphd"><span class="lab">deck &mdash; 20 slides, evidence outlined</span>
      <button class="jump" type="button" data-qa="${r.qa}">Jump to evidence &rarr;</button>
      <button class="openev" type="button" data-qa="${r.qa}">Open evidence large</button></div>
    <div class="strip" id="strip-${r.qa}">${stripHTML(r)}</div>
    <div class="cmp${r.ap ? '' : ' one'}">${condHTML(r.ev, r, 'ev')}${condHTML(r.ap, r, 'ap')}</div>
  </article>`;
}
function render() {
  const host = $('#cases');
  const nPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  page = Math.min(page, nPages - 1);
  const slice = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  $('#fcount').textContent = `${filtered.length} of ${EX.length} examples`;
  host.innerHTML = slice.length ? slice.map(caseHTML).join('')
    : '<div class="empty">No examples match these filters.</div>';
  $('#pgnum').textContent = `page ${page + 1} of ${nPages}`;
  $('#prev').disabled = page === 0;
  $('#next').disabled = page >= nPages - 1;
  wire(host);
}
function wire(host) {
  host.querySelectorAll('.sl').forEach(b => b.addEventListener('click', () => {
    openLB(+b.dataset.qa, +b.dataset.p);
  }));
  host.querySelectorAll('button.jump').forEach(b => b.addEventListener('click', () => {
    const r = byQa[b.dataset.qa];
    const strip = document.getElementById('strip-' + r.qa);
    const el = strip.querySelector(`.sl[data-p="${r.ev_pages[0]}"]`);
    strip.scrollTo({left: el.offsetLeft - strip.clientWidth / 2 + el.clientWidth / 2, behavior: 'smooth'});
    el.focus({preventScroll: true});
  }));
  host.querySelectorAll('button.openev').forEach(b => b.addEventListener('click', () => {
    const r = byQa[b.dataset.qa]; openLB(r.qa, r.ev_pages[0]);
  }));
  host.querySelectorAll('button.expand').forEach(b => b.addEventListener('click', () => {
    const t = b.previousElementSibling;
    t.style.maxHeight = t.style.maxHeight === 'none' ? '230px' : 'none';
  }));
}
$('#prev').addEventListener('click', () => { page--; render(); scrollTo({top: $('#explorer').offsetTop - 10, behavior: 'smooth'}); });
$('#next').addEventListener('click', () => { page++; render(); scrollTo({top: $('#explorer').offsetTop - 10, behavior: 'smooth'}); });

/* ---------- lightbox: pan + zoom + deck navigation ----------
   Adapted from the annotation gallery's viewer. Slide text is small, so the
   zoom ceiling is deliberately high (40x fit) and the arrow keys walk the deck
   without leaving the viewer. */
(function () {
  const lb = $('#lb'), stage = lb.querySelector('.stage'), img = lb.querySelector('img'),
        lvl = lb.querySelector('.lvl'), pos = lb.querySelector('.pos'), cap = lb.querySelector('.cap');
  let s = 1, fit = 1, tx = 0, ty = 0, drag = false, lx = 0, ly = 0;
  let cur = null, curPage = 1, evIdx = 0;

  const apply = () => { img.style.transform = `translate(${tx}px,${ty}px) scale(${s})`;
                        lvl.textContent = Math.round(s / fit * 100) + '%'; };
  const fitView = () => { const r = stage.getBoundingClientRect();
    if (!img.naturalWidth) return;
    fit = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
    s = fit; tx = (r.width - img.naturalWidth * s) / 2; ty = (r.height - img.naturalHeight * s) / 2; apply(); };
  const zoomAt = (px, py, f) => { const ns = Math.min(fit * 40, Math.max(fit * 0.5, s * f));
    tx = px - (px - tx) * (ns / s); ty = py - (py - ty) * (ns / s); s = ns; apply(); };
  const centreZoom = f => { const r = stage.getBoundingClientRect(); zoomAt(r.width / 2, r.height / 2, f); };

  function show(p) {
    curPage = Math.max(1, Math.min(20, p));
    img.src = fullSrc(cur.dk, curPage);
    const isEv = cur.ev_pages.includes(curPage);
    pos.textContent = `${curPage} / 20`;
    cap.innerHTML = (isEv ? '<span class="evb">EVIDENCE</span>' : '')
      + `slide ${curPage} of 20 &middot; ${esc(cur.q)}`;
  }
  window.openLB = function (qa, p) {
    cur = byQa[qa]; evIdx = 0;
    lb.classList.add('on'); show(p);
    lb.querySelector('.jumpev').textContent = cur.ev_pages.length > 1
      ? `evidence: ${cur.ev_pages.join(' / ')}` : `evidence: slide ${cur.ev_pages[0]}`;
  };
  const close = () => { lb.classList.remove('on'); img.removeAttribute('src'); cur = null; };

  img.addEventListener('load', fitView);
  addEventListener('resize', () => { if (lb.classList.contains('on')) fitView(); });
  stage.addEventListener('wheel', e => { e.preventDefault(); const r = stage.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.18 : 1 / 1.18); }, {passive: false});
  stage.addEventListener('dblclick', e => { const r = stage.getBoundingClientRect();
    if (s > fit * 1.5) fitView(); else zoomAt(e.clientX - r.left, e.clientY - r.top, 5); });
  stage.addEventListener('mousedown', e => { drag = true; lx = e.clientX; ly = e.clientY;
    stage.classList.add('grabbing'); e.preventDefault(); });
  addEventListener('mousemove', e => { if (!drag) return;
    tx += e.clientX - lx; ty += e.clientY - ly; lx = e.clientX; ly = e.clientY; apply(); });
  addEventListener('mouseup', () => { drag = false; stage.classList.remove('grabbing'); });
  /* pinch to zoom */
  let pts = new Map(), pd = 0;
  stage.addEventListener('pointerdown', e => { pts.set(e.pointerId, e); });
  stage.addEventListener('pointermove', e => {
    if (!pts.has(e.pointerId)) return; pts.set(e.pointerId, e);
    if (pts.size === 2) {
      const [a, b] = [...pts.values()];
      const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      if (pd) { const r = stage.getBoundingClientRect();
        zoomAt((a.clientX + b.clientX) / 2 - r.left, (a.clientY + b.clientY) / 2 - r.top, d / pd); }
      pd = d;
    }
  });
  const up = e => { pts.delete(e.pointerId); if (pts.size < 2) pd = 0; };
  stage.addEventListener('pointerup', up); stage.addEventListener('pointercancel', up);

  lb.querySelector('.zin').onclick = () => centreZoom(1.5);
  lb.querySelector('.zout').onclick = () => centreZoom(1 / 1.5);
  lb.querySelector('.zfit').onclick = fitView;
  lb.querySelector('.zclose').onclick = close;
  lb.querySelector('.prevs').onclick = () => show(curPage - 1);
  lb.querySelector('.nexts').onclick = () => show(curPage + 1);
  lb.querySelector('.jumpev').onclick = () => {
    show(cur.ev_pages[evIdx % cur.ev_pages.length]); evIdx++;
  };
  addEventListener('keydown', e => {
    if (!lb.classList.contains('on')) return;
    if (e.key === 'Escape') close();
    else if (e.key === 'ArrowRight') { e.preventDefault(); show(curPage + 1); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); show(curPage - 1); }
    else if (e.key === '+' || e.key === '=') centreZoom(1.5);
    else if (e.key === '-') centreZoom(1 / 1.5);
    else if (e.key === '0') fitView();
    else if (e.key === 'e' || e.key === 'E') lb.querySelector('.jumpev').click();
  });
})();

const byQa = {}; EX.forEach(r => byQa[r.qa] = r);
applyFilters();
"""


LIGHTBOX_HTML = """
<div id="lb" role="dialog" aria-label="slide viewer">
 <div class="ctrl">
  <button class="prevs" title="previous slide (left arrow)">&#8249;</button>
  <span class="pos">1 / 20</span>
  <button class="nexts" title="next slide (right arrow)">&#8250;</button>
  <button class="jumpev jump" title="jump to the evidence slide (E)">evidence</button>
  <button class="zout" title="zoom out (-)">&minus;</button>
  <span class="lvl">100%</span>
  <button class="zin" title="zoom in (+)">+</button>
  <button class="zfit" title="fit to screen (0)">fit</button>
  <button class="zclose" title="close (Esc)">&times;</button>
 </div>
 <div class="cap"></div>
 <div class="stage"><img alt="slide, full size"></div>
 <span class="hint">scroll or pinch to zoom &middot; drag to pan &middot; &larr;/&rarr; walk the deck
  &middot; E jumps to evidence &middot; Esc closes</span>
</div>
"""


def tile(lab, val, note_, tone="") -> str:
    return (f'<div class="tile {tone}"><div class="tlab">{esc(lab)}</div>'
            f'<div class="tval">{val}</div><div class="tnote">{note_}</div></div>')


def build_html(a, arith, examples, bucket_counts, join_ok) -> str:
    po = a["paired_overall"]
    full = {r["slice"]: r for r in a["full"]}
    costs = {c["name"]: c for c in a["costs"]}
    fm = a["format"]

    # ---- tiles -------------------------------------------------------------
    tiles = "".join([
        tile("Evidence-condition F1", f'{full["overall"]["f1"]:.1f}',
             f'EM {full["overall"]["em"]:.1f} &middot; n={full["overall"]["n"]} &middot; '
             f'oracle retrieval'),
        tile("Retrieval cost", f'{costs["retrieval"]["cost"]:+.1f}',
             f'F1, paired on n={a["n_paired"]} &middot; 20 slides vs the right 1&ndash;2', "bad"),
        tile("Integration cost", f'{costs["integration"]["cost"]:+.1f}',
             "F1, multi-page evidence vs single-page", "bad"),
        tile("Derivation cost", f'{costs["derivation"]["cost"]:+.1f}',
             "F1, arithmetic vs plain lookup", "bad"),
        tile("Format-only failures", f'{100 * fm["zero_f1_fmt"] / fm["zero_f1"]:.0f}%',
             f'of the {fm["zero_f1"]} hard zeros are the right value, wrongly formatted', "bad"),
    ])

    # ---- table 1: full evidence condition ----------------------------------
    t1 = table(
        ["slice", "F1", "EM", "F1 (fmt-corrected)", "EM (fmt-corrected)", "n"],
        [[f'{esc(r["slice"])}<span class="nlab">{r["note"]}</span>',
          f'{r["f1"]:.1f}', f'{r["em"]:.1f}', f'{r["f1c"]:.1f}', f'{r["emc"]:.1f}', r["n"]]
         for r in a["full"]])

    # ---- costs chart -------------------------------------------------------
    cost_rows = [(f'{c["name"]}<span class="nlab">{c["what"]}</span>',
                  abs(c["cost"]), abs(c["cost_c"]), c["basis"],
                  f'{c["name"]}: {c["cost"]:+.1f} F1 as officially scored, '
                  f'{c["cost_c"]:+.1f} once format-equivalent answers are credited. {c["basis"]}.')
                 for c in a["costs"]]
    vmax = max(max(r[1], r[2]) for r in cost_rows) * 1.1
    costs_chart = grouped_bars(
        "The three costs, in F1 points lost",
        "How much each added demand costs the model. All three are losses; bars show magnitude.",
        [(lab, v1, v2, basis, tipt) for lab, v1, v2, basis, tipt in cost_rows],
        series=("as officially scored", "format-equivalent answers credited"),
        vmax=vmax, unit="")

    ratio = abs(costs["derivation"]["cost"]) / max(1e-9, abs(costs["retrieval"]["cost"]))
    ratio_c = abs(costs["derivation"]["cost_c"]) / max(1e-9, abs(costs["retrieval"]["cost_c"]))

    # ---- per-cell gaps -----------------------------------------------------
    cell_chart = diverging_bars(
        "Retrieval cost per cell, paired",
        "Same questions, both conditions. Negative means the model did worse when it had to "
        "find the evidence itself.",
        [(esc(c["cell"]), c["gap"], c["n"],
          f'{c["cell"]}: evidence {c["evidence"]:.1f} F1 &rarr; all-pages {c["allpages"]:.1f} F1 '
          f'({c["gap"]:+.1f}), n={c["n"]}')
         for c in a["cells"]])

    t2 = table(["condition", "F1", "EM", "F1 (fmt-corrected)", "EM (fmt-corrected)"],
               [["evidence only (oracle retrieval)", f'{po["evidence"]["f1"]:.1f}',
                 f'{po["evidence"]["em"]:.1f}', f'{po["evidence"]["f1c"]:.1f}', f'{po["evidence"]["emc"]:.1f}'],
                ["all 20 slides", f'{po["allpages"]["f1"]:.1f}', f'{po["allpages"]["em"]:.1f}',
                 f'{po["allpages"]["f1c"]:.1f}', f'{po["allpages"]["emc"]:.1f}'],
                ['<b>retrieval cost</b>',
                 f'<b>{po["allpages"]["f1"] - po["evidence"]["f1"]:+.1f}</b>',
                 f'<b>{po["allpages"]["em"] - po["evidence"]["em"]:+.1f}</b>',
                 f'<b>{po["allpages"]["f1c"] - po["evidence"]["f1c"]:+.1f}</b>',
                 f'<b>{po["allpages"]["emc"] - po["evidence"]["emc"]:+.1f}</b>']])

    # ---- histogram ---------------------------------------------------------
    H = a["hist"]
    hist_chart = grouped_bars(
        "Where the F1 mass sits, paired questions only",
        "Share of questions falling in each F1 band. The distribution is bimodal: the model is "
        "usually all right or all wrong, and partial credit is thin.",
        [(lab, 100 * H["evidence"][i] / H["n"], 100 * H["allpages"][i] / H["n"], H["n"],
          f'F1 {lab}: evidence {H["evidence"][i]}, all-pages {H["allpages"][i]} of {H["n"]}')
         for i, lab in enumerate(H["labels"])],
        vmax=max(max(H["evidence"]), max(H["allpages"])) / H["n"] * 110, unit="%")

    # ---- format artifact ---------------------------------------------------
    fmt_chart = single_bars(
        "Format-only failure rate by slice",
        "Share of all questions in the slice that scored EM=0 while being the same value as gold, "
        "differently written.",
        [(esc(r["slice"]), r["rate"], r["n"],
          f'{r["slice"]}: {r["rate"]:.1f}% of n={r["n"]} are format-only misses')
         for r in fm["by_slice"]],
        vmax=max(r["rate"] for r in fm["by_slice"]) * 1.15, unit="%",
        tone=lambda v: "bad")

    # ---- arithmetic breakdown ---------------------------------------------
    AC = arith["counts"]
    order = [("exact", "exact match"), ("format_only", "right value, wrong format"),
             ("wrong_operand", "wrong operand (misread the slide)"),
             ("wrong_operation", "wrong operation (read right, computed wrong)"),
             ("unparsed_expr", "expression not a simple binary op")]
    arith_chart = single_bars(
        "What actually goes wrong on the 194 arithmetic questions",
        "Decided deterministically: an answer is a wrong-operand error when the annotated "
        "expression's operands do not both appear in the model's own thinking trace, and a "
        "wrong-operation error when they do.",
        [(esc(lab), 100 * AC.get(k, 0) / arith["n"], AC.get(k, 0),
          f'{lab}: {AC.get(k, 0)} of {arith["n"]} arithmetic questions')
         for k, lab in order if AC.get(k, 0)],
        vmax=max(100 * v / arith["n"] for v in AC.values()) * 1.15, unit="%",
        tone=lambda v: "")

    # ---- n_evidence, gold length, cost -------------------------------------
    nev_chart = single_bars(
        "F1 by number of evidence slides",
        "The annotation says how many slides carry the answer. More slides, less accuracy &mdash; "
        "but the drop is modest.",
        [(f'{r["n_evidence"]} slide{"s" if r["n_evidence"] > 1 else ""}', r["f1"], r["n"],
          f'{r["n_evidence"]} evidence slides: F1 {r["f1"]:.1f}, EM {r["em"]:.1f}, n={r["n"]}')
         for r in a["by_n_evidence"]])
    len_chart = single_bars(
        "F1 by gold answer length",
        "Token F1 punishes long answers: every extra token the model does not produce costs recall.",
        [(esc(r["label"]), r["f1"], r["n"],
          f'{r["label"]}: F1 {r["f1"]:.1f}, EM {r["em"]:.1f}, n={r["n"]}')
         for r in a["by_gold_len"]])

    ct = a["cost_tokens"]
    cost_table = table(
        ["condition", "mean latency", "mean input tokens", "mean output tokens"],
        [["evidence only", f'{ct["evidence"]["latency"]:.2f}s',
          f'{ct["evidence"]["in_tok"]:,.0f}', f'{ct["evidence"]["out_tok"]:,.0f}'],
         ["all 20 slides", f'{ct["allpages"]["latency"]:.2f}s',
          f'{ct["allpages"]["in_tok"]:,.0f}', f'{ct["allpages"]["out_tok"]:,.0f}'],
         ["<b>ratio</b>", f'<b>{ct["allpages"]["latency"] / ct["evidence"]["latency"]:.1f}&times;</b>',
          f'<b>{ct["allpages"]["in_tok"] / ct["evidence"]["in_tok"]:.1f}&times;</b>',
          f'<b>{ct["allpages"]["out_tok"] / ct["evidence"]["out_tok"]:.1f}&times;</b>']])

    D, DC = a["divergence"], a["divergence_c"]
    div_table = table(
        ["outcome on the paired questions", "as scored", "format-corrected"],
        [["right in both conditions", D.get("ev_ok|ap_ok", 0), DC.get("ev_ok|ap_ok", 0)],
         ["<b>right on evidence, wrong on 20 slides</b> &mdash; retrieval failure",
          f'<b>{D.get("ev_ok|ap_no", 0)}</b>', f'<b>{DC.get("ev_ok|ap_no", 0)}</b>'],
         ["wrong on evidence, right on 20 slides", D.get("ev_no|ap_ok", 0), DC.get("ev_no|ap_ok", 0)],
         ["wrong in both conditions", D.get("ev_no|ap_no", 0), DC.get("ev_no|ap_no", 0)]])

    # ---- explorer ----------------------------------------------------------
    bucket_opts = "".join(
        f'<option value="{k}">{esc(lab)} ({bucket_counts.get(k, 0)})</option>'
        for k, lab, _ in BUCKETS)
    bucket_legend = "".join(
        f'<div class="bhd"><b>{esc(lab)}</b> &mdash; {desc} '
        f'<span class="nlab">{bucket_counts.get(k, 0)} shown</span></div>'
        for k, lab, desc in BUCKETS)

    ex_json = json.dumps(examples, ensure_ascii=False, separators=(",", ":"))
    labels_json = json.dumps({k: lab for k, lab, _ in BUCKETS}, ensure_ascii=False)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>SlideVQA &mdash; retrieval, integration and derivation on 20-slide decks</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style></head><body>
<div class="wrap">
<p class="crumb"><a href="report.html">&larr; blind-spot overview</a> &middot; SlideVQA</p>
<header><div>
<h1>SlideVQA: what does it cost to find the slide?</h1>
<p class="dek">Claude Haiku 4.5 (thinking, 2000-token budget) answering questions over 20-slide
business decks. Every question carries annotated evidence pages, so the same questions ran twice:
once with only the right 1&ndash;2 slides in the prompt, once with all 20. The gap between them is
retrieval, measured rather than assumed. All numbers are recomputed from the raw result files.</p>
</div><button class="theme" type="button">Dark mode</button></header>

{tiles and f'<div class="tiles">{tiles}</div>'}

{note(f'<strong>Setup.</strong> {a["n_evidence_rows"]} questions ran in the evidence condition and '
      f'{a["n_allpages_rows"]} in the all-pages condition; {a["n_paired"]} appear in both and carry '
      f'the paired comparison. The all-pages sample was stratified to oversample arithmetic '
      f'({100 * a["arith_share_paired"]:.0f}% of the paired subset vs '
      f'{100 * a["arith_share_full"]:.0f}% of the full set), so its absolute scores sit below the '
      f'headline. The <em>gap</em> is the valid quantity and it is paired &mdash; same questions, '
      f'same model, only the number of slides changes.')}

<h2>Reading the right slide<span class="sub">The evidence condition: the model is handed the
1&ndash;2 slides that contain the answer. Whatever it gets wrong here is not a retrieval
problem.</span></h2>
{t1}
{whatmeans(f'Handed the right slides, the model answers {full["overall"]["f1"]:.1f} F1 / '
           f'{full["overall"]["em"]:.1f} EM. The two structural splits behave very differently: '
           f'needing a second slide costs about {abs(costs["integration"]["cost"]):.0f} F1, while '
           f'needing to compute something costs about {abs(costs["derivation"]["cost"]):.0f}. The '
           f'last two columns credit answers that are the same value differently written &mdash; '
           f'"22%" against a gold of "22". That correction moves the arithmetic row by '
           f'{full["needs arithmetic"]["f1c"] - full["needs arithmetic"]["f1"]:.0f} F1 and the '
           f'lookup row by only {full["plain lookup"]["f1c"] - full["plain lookup"]["f1"]:.0f}, '
           f'which is the first sign that the arithmetic penalty is partly a metric artifact.')}

<h2>The three costs<span class="sub">Retrieval (find it), integration (combine two slides),
derivation (compute on it) &mdash; priced in F1 points.</span></h2>
{costs_chart}
{whatmeans(f'As officially scored, derivation costs {ratio:.1f}&times; more than retrieval: '
           f'{abs(costs["derivation"]["cost"]):.1f} F1 against {abs(costs["retrieval"]["cost"]):.1f}. '
           f'That is the headline, and it is real but overstated. Credit the format-equivalent '
           f'answers &mdash; the ones where the model said "22%" and gold said "22" &mdash; and the '
           f'ratio falls to {ratio_c:.1f}&times;: '
           f'{abs(costs["derivation"]["cost_c"]):.1f} against {abs(costs["retrieval"]["cost_c"]):.1f}. '
           f'Either way the ordering holds and the practical conclusion is the same: putting all 20 '
           f'slides in the prompt is close to free, and the model&rsquo;s real weakness is doing '
           f'arithmetic on what it has read. But the 6&times; version of that claim is roughly half '
           f'metric artifact, and should not be quoted without the correction.')}

<h2>Retrieval, paired<span class="sub">The same {a["n_paired"]} questions under both conditions.
Nothing changes except how many slides are in the prompt.</span></h2>
{t2}
{cell_chart}
{whatmeans(f'Making the model find the evidence among 20 slides costs '
           f'{abs(po["allpages"]["f1"] - po["evidence"]["f1"]):.1f} F1 overall. The per-cell view is '
           f'where it gets interesting. Every cell loses ground except '
           f'<b>{esc(a["cells"][-1]["cell"])}</b>, which is flat at '
           f'{a["cells"][-1]["gap"]:+.1f} F1 on n={a["cells"][-1]["n"]}. That is the tell: on '
           f'single-page arithmetic the model was going to fail the computation anyway, so 19 '
           f'distractor slides cost it nothing. Distractors only hurt when the model would '
           f'otherwise have succeeded.')}
{div_table}
{whatmeans(f'Counting outcomes rather than averaging scores: {D.get("ev_ok|ap_no", 0)} questions '
           f'flipped from right to wrong when the distractors were added, against '
           f'{D.get("ev_no|ap_ok", 0)} that flipped the other way. The net is small &mdash; about '
           f'{D.get("ev_ok|ap_no", 0) - D.get("ev_no|ap_ok", 0)} questions out of {a["n_paired"]}. '
           f'Retrieval over 20 slides is a real cost but a modest one; the {D.get("ev_no|ap_no", 0)} '
           f'questions that are wrong in both conditions are where the actual capability gap lives.')}

<h2>Distribution of scores<span class="sub">Averages hide shape. This is where the F1 mass actually
sits.</span></h2>
{hist_chart}
{whatmeans('The distribution is strongly bimodal: most questions score exactly 1.0 or exactly 0.0, '
           'and the middle bands are nearly empty. Token F1 is behaving almost like accuracy here, '
           'because SlideVQA answers are short &mdash; usually one token &mdash; so there is no '
           'room for partial overlap. That also means every formatting mismatch lands in the 0 bin '
           'rather than scoring 0.5, which is exactly why the artifact below is so large.')}

<h2>How much of this is the metric?<span class="sub">A formatting disagreement is not a perception
failure. It is counted separately, not quietly folded in.</span></h2>
{note(f'<strong>{fm["zero_f1_fmt"]} of the {fm["zero_f1"]} hard zeros '
      f'({100 * fm["zero_f1_fmt"] / fm["zero_f1"]:.0f}%) are semantically correct answers</strong> '
      f'that scored nothing: "22%" against a gold of "22", "$2,410" against "2410", "3.3bn" against '
      f'"3.3". Across all {fm["non_em"]} non-exact answers, {fm["non_em_fmt"]} '
      f'({100 * fm["non_em_fmt"] / fm["non_em"]:.0f}%) are format-equivalent. The detector is '
      f'deliberately conservative: numeric comparison is sign-sensitive, so "-300" is never treated '
      f'as "300", and it never falls back to substring matching on numbers.')}
{fmt_chart}
{whatmeans(f'The artifact is not evenly spread. Arithmetic answers take '
           f'{[r["rate"] for r in fm["by_slice"] if r["slice"] == "needs arithmetic"][0]:.0f}% '
           f'format-only misses against '
           f'{[r["rate"] for r in fm["by_slice"] if r["slice"] == "plain lookup"][0]:.0f}% for plain '
           f'lookups, because a derived answer is a bare number and the model habitually dresses it '
           f'with the unit it just read off the chart. So the metric penalises arithmetic roughly '
           f'twice as hard as lookup for reasons that have nothing to do with arithmetic. This is '
           f'the single most important caveat on this page.')}

<h2>Arithmetic, decomposed<span class="sub">The manifest ships the intended computation for every
arithmetic question, which makes the failures legible.</span></h2>
{arith_chart}
{whatmeans(f'Of {arith["n"]} arithmetic questions, {AC.get("exact", 0)} are exactly right and '
           f'another {AC.get("format_only", 0)} are right but wrongly formatted &mdash; so real '
           f'arithmetic accuracy is about '
           f'{100 * (AC.get("exact", 0) + AC.get("format_only", 0)) / arith["n"]:.0f}%, not the '
           f'{100 * AC.get("exact", 0) / arith["n"]:.0f}% the EM column reports. The genuine '
           f'failures split almost evenly: {AC.get("wrong_operand", 0)} wrong-operand (the model '
           f'misread a number off the slide, then computed correctly on it) against '
           f'{AC.get("wrong_operation", 0)} wrong-operation (both operands appear verbatim in the '
           f'trace, so the reading was fine and the computation was not). That is a useful split: '
           f'roughly half of what looks like an arithmetic blind spot is actually a perception '
           f'blind spot wearing arithmetic&rsquo;s clothes.')}
{note(f'<strong>Ground-truth caveat.</strong> On {arith["bad_gold"]} of these questions the '
      f'annotated expression does not evaluate to the annotated answer &mdash; e.g. an expression of '
      f'<code>220-50</code> with a gold of <code>17</code>. Those are dataset errors, and the model '
      f'is scored wrong on them no matter what it says.')}

<h2>Other slices<span class="sub">Things worth checking before drawing conclusions.</span></h2>
{nev_chart}
{len_chart}
{whatmeans('Answer length matters more than it should. Single-token golds score highest and long '
           'golds worst, which is partly genuine difficulty and partly token F1 punishing any '
           'answer the model phrases more fully than the annotation. Read the multi-token rows as a '
           'lower bound.')}
{cost_table}
{whatmeans(f'The all-pages condition costs '
           f'{ct["allpages"]["in_tok"] / ct["evidence"]["in_tok"]:.0f}&times; the input tokens and '
           f'{ct["allpages"]["latency"] / ct["evidence"]["latency"]:.1f}&times; the latency to buy '
           f'{po["allpages"]["f1"] - po["evidence"]["f1"]:+.1f} F1. If you have a retrieval step '
           f'that can find the right slide, use it &mdash; not because the model cannot cope with '
           f'20 slides, but because paying '
           f'{ct["allpages"]["in_tok"]:,.0f} input tokens per question to lose '
           f'{abs(po["allpages"]["f1"] - po["evidence"]["f1"]):.1f} F1 is a bad trade. Note also '
           f'that output tokens barely move: the model does not think noticeably longer when given '
           f'19 extra slides. It does not appear to search them so much as skim.')}

<h2 id="explorer">Browse the failures<span class="sub">{len(examples)} examples, weighted towards
failure on purpose. Click any thumbnail to open the deck viewer &mdash; scroll to zoom, drag to pan,
arrow keys walk the deck, <code>E</code> jumps to the evidence slide.</span></h2>

{bucket_legend}

<div class="filters">
 <label>evidence<select id="f-ev"><option value="all">any</option>
  <option value="single">single-page</option><option value="multi">multi-page</option></select></label>
 <label>question<select id="f-kind"><option value="all">any</option>
  <option value="lookup">lookup</option><option value="arith">arithmetic</option></select></label>
 <label>outcome (evidence)<select id="f-out"><option value="all">any</option>
  <option value="correct">correct (EM=1)</option><option value="partial">partial (0&lt;F1&lt;1)</option>
  <option value="wrong">wrong (F1=0)</option></select></label>
 <label>divergence<select id="f-div"><option value="all">any</option>
  <option value="ret_fail">evidence right, all-pages wrong</option>
  <option value="both_no">wrong in both</option>
  <option value="both_ok">right in both</option>
  <option value="ap_only">all-pages right, evidence wrong</option>
  <option value="unpaired">evidence condition only</option></select></label>
 <label>bucket<select id="f-bucket"><option value="all">all buckets</option>{bucket_opts}</select></label>
 <label>search<input id="f-q" type="search" placeholder="question, gold, prediction, deck"></label>
 <button id="f-reset" type="button">reset</button>
 <span class="fcount" id="fcount"></span>
</div>

<div id="cases"></div>
<div class="pager"><button id="prev" type="button">&larr; previous</button>
 <span id="pgnum"></span><button id="next" type="button">next &rarr;</button></div>

</div>
{LIGHTBOX_HTML}
<div id="tip"></div>
<script>
const EX = {ex_json};
const BUCKET_LABEL = {labels_json};
</script>
<script>{JS}</script>
</body></html>"""


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--out", default=str(OUT / "slidevqa.html"))
    ap_.add_argument("--workers", type=int, default=None,
                     help="process pool size for slide resizing (default: all cores)")
    ap_.add_argument("--skip-images", action="store_true",
                     help="rebuild the HTML only, reusing slides already on disk")
    args = ap_.parse_args()

    man, by_qa, ev, ap = load()
    v_ev, v_ap = verify_join(by_qa, ev), verify_join(by_qa, ap)
    print(f"join by uid trailing integer -> manifest qa_id: "
          f"evidence {v_ev['ok']}/{v_ev['n']} verified, all-pages {v_ap['ok']}/{v_ap['n']} verified")
    if v_ev["bad"] or v_ap["bad"]:
        raise SystemExit("join verification failed -- refusing to render a page on a bad join")

    a, E, A, common = analyse(ev, ap)
    arith = analyse_arithmetic(ev, by_qa)
    examples, bucket_counts = select_examples(ev, ap, E, A, common, by_qa)
    for rec in examples:
        rec["dk"] = deck_key(rec["deck"])

    print(f"selected {len(examples)} examples over "
          f"{len({r['deck'] for r in examples})} decks: " +
          ", ".join(f"{k}={v}" for k, v in bucket_counts.items()))

    if not args.skip_images:
        n_jobs, errs = render_slides(examples, by_qa, args.workers)
        print(f"rendered {n_jobs} unique slides (thumb {THUMB_W}px q{THUMB_Q} + "
              f"full {FULL_EDGE}px q{FULL_Q})")
        for e in errs[:10]:
            print("  image error:", e)

    OUT.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.write_text(build_html(a, arith, examples, {k: v for k, v in bucket_counts.items()},
                              (v_ev, v_ap)), encoding="utf-8")
    size = sum(p.stat().st_size for p in ASSETS.glob("*.jpg")) if ASSETS.exists() else 0
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB); "
          f"assets {size / 1e6:.1f} MB in {len(list(ASSETS.glob('*.jpg')))} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
