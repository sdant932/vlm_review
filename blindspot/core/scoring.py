"""Per-dataset scorers.

Metrics follow each benchmark's own convention so numbers stay comparable to
published work:

    InfographicVQA  ANLS against the best-matching gold (threshold 0.5)
    CharXiv         normalized match; numeric-aware where the answer is a value
    ScreenSpot/-Pro click-in-bbox accuracy

One honest caveat, surfaced rather than buried: CharXiv's official grader is an
LLM judge with per-question-type rubrics ("same term, different form scores 1").
This harness does not run a judge, so free-text descriptive types (title, axis
labels, legend contents, trend) are graded approximately and their scores are a
**lower bound** -- a correct answer phrased differently can be marked wrong.
`charxiv_grading_confidence()` reports which side of that line each question
falls on, and the report separates them instead of pooling them into one number.
"""

from __future__ import annotations

from typing import Any

# CharXiv descriptive question types whose answers are values, counts, or a
# closed set -- normalized matching is reliable here.
CHARXIV_STRICT_QIDS = {4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18, 19}
# ...and those whose answers are free text, where strict matching undercounts.
CHARXIV_FUZZY_QIDS = {1, 2, 3, 13, 16}

ANLS_THRESHOLD = 0.5


def charxiv_grading_confidence(qid: int | None) -> str:
    if qid is None:
        return "fuzzy"  # reasoning split: short free-text answers
    return "strict" if qid in CHARXIV_STRICT_QIDS else "fuzzy"


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _normalize(s: Any) -> str:
    return " ".join(str(s).strip().lower().replace(",", "").split())


def _as_float(s: Any) -> float | None:
    t = _normalize(s).rstrip("%").replace("$", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


def _anls_normalize(s: Any) -> str:
    """Lowercase + strip, exactly as the official DocVQA/InfographicVQA script.

    Deliberately NOT `_normalize`: that one also strips commas, which would score
    "1,000" against "1000" as a perfect match. That is more lenient than the
    official metric, and being more lenient than the benchmark is just as wrong
    as being stricter.
    """
    return str(s).lower().strip()


def anls(pred: str, golds: list[str], threshold: float = ANLS_THRESHOLD) -> float:
    """Official ANLS: 1 - NL against the best gold, zeroed below the threshold.

    Follows the published definition: NL = levenshtein / max(len); the score is
    1 - NL when NL < tau (tau = 0.5), otherwise 0; take the max over ground
    truths. Note the boundary is strict -- a normalized distance of exactly 0.5
    scores 0, not 0.5.
    """
    p = _anls_normalize(pred)
    best = 0.0
    for g in golds:
        g = _anls_normalize(g)
        if not p and not g:
            best = max(best, 1.0)
            continue
        denom = max(len(p), len(g))
        if not denom:
            continue
        nl = _levenshtein(p, g) / denom
        best = max(best, 1.0 - nl if nl < threshold else 0.0)
    return best


def numeric_or_text_match(pred: str, golds: list[str]) -> float:
    """Exact match after normalization, comparing numerically when both sides parse.

    Numeric comparison uses a relative tolerance so that 0.28 == 0.280 == "0.28 "
    without letting genuinely different values through.
    """
    pf = _as_float(pred)
    for g in golds:
        gf = _as_float(g)
        if pf is not None and gf is not None:
            if abs(pf - gf) <= 1e-6 * max(1.0, abs(gf)):
                return 1.0
        elif _normalize(pred) == _normalize(g):
            return 1.0
    return 0.0


def boolean_match(pred: Any, golds: list) -> float:
    """Exact yes/no match.

    A raw accuracy number is not sufficient for these families and the report
    must never show one alone: FlowLearn's arrow probes are balanced by
    construction (one matched positive and negative per figure), so a model that
    answers "yes" to everything scores ~50% while having perceived nothing. The
    runner records polarity so the report can show balanced accuracy and the
    yes-rate alongside.
    """
    p = str(pred).strip().lower()
    return 1.0 if any(p == str(g).strip().lower() for g in golds) else 0.0


def count_score(pred: Any, golds: list) -> dict:
    """Exact-count accuracy plus the signed error.

    Signed error is the interesting half: consistent undercounting as object
    count rises is a different failure from noisy counting, and only the sign
    distinguishes them.
    """
    try:
        p = int(pred)
        g = int(golds[0])
    except (TypeError, ValueError, IndexError):
        return {"score": 0.0, "abs_error": None, "signed_error": None}
    return {"score": float(p == g), "abs_error": abs(p - g), "signed_error": p - g}


def token_f1(pred: Any, golds: list) -> tuple[float, float]:
    """(EM, F1) over normalized tokens -- SlideVQA's official pair of metrics.

    Not ANLS: SlideVQA reports exact match and token-level F1, so using ANLS here
    would make the number incomparable to the published results.
    """
    import collections
    def toks(x):
        return [t for t in _anls_normalize(x).replace(",", " ").split() if t]
    p = toks(pred)
    best_em = best_f1 = 0.0
    for g in golds:
        gt = toks(g)
        best_em = max(best_em, float(p == gt))
        if not p or not gt:
            best_f1 = max(best_f1, float(p == gt))
            continue
        common = collections.Counter(p) & collections.Counter(gt)
        n = sum(common.values())
        if n:
            prec, rec = n / len(p), n / len(gt)
            best_f1 = max(best_f1, 2 * prec * rec / (prec + rec))
    return best_em, best_f1


def point_in_bbox(pred: tuple[float, float], bbox: list[float]) -> float:
    x, y = pred
    x0, y0, x1, y1 = bbox
    return 1.0 if (x0 <= x <= x1 and y0 <= y <= y1) else 0.0


def score(example, pred: Any) -> dict:
    """Score one prediction. Returns the value plus how it was obtained."""
    ds, meta = example.dataset, example.meta

    if example.answer_type == "point":
        correct = point_in_bbox(pred, example.gold)
        x, y = pred
        x0, y0, x1, y1 = example.gold
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return {
            "score": correct,
            "metric": "click_in_bbox",
            "grading_confidence": "strict",
            # Distance to target center, for asking "near miss or nowhere near?"
            "center_distance": ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5,
        }

    if example.answer_type == "bbox":
        # Scored as centre-of-predicted-box inside the gold box, so the number
        # is directly comparable to click-in-bbox rather than a new unit. IoU
        # would be a different metric and EVAL.md forbids mixing the two.
        x0, y0, x1, y1 = pred
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return {"score": point_in_bbox((cx, cy), example.gold),
                "metric": "bbox_centre_in_gold", "grading_confidence": "strict",
                "pred_centre": [cx, cy], "pred_area": abs((x1 - x0) * (y1 - y0))}

    if example.answer_type == "boolean":
        return {"score": boolean_match(pred, example.gold),
                "metric": "exact_yes_no", "grading_confidence": "strict",
                "polarity": meta.get("polarity"), "pred_yes": str(pred).strip().lower() == "yes"}

    if example.answer_type == "choice":
        # Record the chosen letter so position bias is checkable: AI2D's answer
        # key is near-uniform across A-D, so a model that favours one slot would
        # show up here rather than hiding inside the accuracy number.
        return {"score": 1.0 if str(pred).strip().upper() == str(example.gold[0]).strip().upper() else 0.0,
                "metric": "multiple_choice", "grading_confidence": "strict",
                "picked": str(pred).strip().upper()}

    if example.answer_type == "count":
        r = count_score(pred, example.gold)
        return {**r, "metric": "exact_count", "grading_confidence": "strict",
                "true_count": meta.get("true_count")}

    if ds == "charxiv":
        conf = charxiv_grading_confidence(meta.get("qid"))
        if conf == "strict":
            return {"score": numeric_or_text_match(pred, example.gold),
                    "metric": "normalized_match", "grading_confidence": "strict"}
        return {"score": anls(pred, example.gold),
                "metric": "anls", "grading_confidence": "fuzzy"}

    if ds == "svg_localization":
        # EVAL.md 3.7: EM and token-F1 reported side by side, no substring
        # containment -- the labels are short and "Close" must not be credited
        # for "Close Ledger". token_f1 compares whole normalized tokens, so it
        # already refuses containment; F1 is the headline, EM rides alongside.
        em, f1 = token_f1(pred, example.gold)
        return {"score": f1, "exact_match": em, "metric": "svgloc_token_f1",
                "grading_confidence": "strict"}

    if ds in ("slidevqa", "slidevqa_allpages"):
        em, f1 = token_f1(pred, example.gold)
        # F1 is the headline in SlideVQA's own reporting; EM is kept alongside.
        return {"score": f1, "exact_match": em, "metric": "slidevqa_f1",
                "grading_confidence": "strict"}

    # InfographicVQA and anything else span-shaped
    return {"score": anls(pred, example.gold), "metric": "anls", "grading_confidence": "strict"}
