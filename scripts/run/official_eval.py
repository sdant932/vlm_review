"""ScreenSpot / ScreenSpot-Pro under the OFFICIAL evaluation protocol.

Ports third_party/ScreenSpot-Pro-GUI-Grounding/models/gpt4x.py verbatim -- the
adapter behind the published GPT-4o numbers -- so our result is comparable to
published work instead of merely internally consistent.

Differences from blindspot/core/runner.py, all deliberate:

    output shape   bounding box [[x0,y0,x1,y1]], not a point
    range          0-1 floats, not 0-1000 integers
    reasoning      none ("Don't output any analysis"), temperature 0
    parsing        regex over free text, not structured outputs
    format failure counted as wrong AND reported as wrong_format_num

Scoring mirrors eval_screenspot_pro.py:115-152 -- action_acc / text_acc /
icon_acc, with wrong_format in the denominator.

    python scripts/run/official_eval.py --datasets screenspot_pro --max-spend 2
"""
from __future__ import annotations
import argparse, json, re, sys, threading, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from blindspot.core.adapters import load
from blindspot.core.prompts import encode_image, HAIKU_MAX_EDGE
from blindspot.core.runner import Budget, FatalBillingError, _is_billing_error, model_spec, short_name

RESULTS = Path("results")

# --- verbatim from gpt4x.py -------------------------------------------------
SYSTEM = ("You are an expert in using electronic devices and interacting with graphic "
          "interfaces. You should not call any external tools.")
# NB: the source concatenates two literals with no separating space, producing
# "...0 to 1.The instruction is:". Reproduced exactly -- prompt text is part of
# the protocol, and silently "fixing" it would make the run non-comparable.
USER_TMPL = ("You are asked to find the bounding box of an UI element in the given "
             "screenshot corresponding to a given instruction.\n"
             "Don't output any analysis. Output your result in the format of "
             "[[x0,y0,x1,y1]], with x and y ranging from 0 to 1."
             "The instruction is:\n{instruction}\n")

BBOX_RE = re.compile(r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]", re.DOTALL)
POINT_RE = re.compile(r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]", re.DOTALL)

# Lenient variants. The official parser is already model-accommodating -- it tries
# a bbox regex, then a point regex, then falls back to the bbox centre, all to fit
# what GPT-4o happens to emit. These extend the same courtesy to Haiku, which
# formats correctly but with whitespace after commas ("[[0.4, 0.15, 0.7, 0.25]]")
# and sometimes answers in pixels despite being asked for 0-1.
NUM = r"[-+]?\d*\.?\d+"
BBOX_LENIENT = re.compile(rf"\[?\[\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*\]\]?", re.DOTALL)
POINT_LENIENT = re.compile(rf"\[?\[\s*({NUM})\s*,\s*({NUM})\s*\]\]?", re.DOTALL)


def extract_lenient(text: str):
    """Return (bbox, point, kind) using the whitespace-tolerant patterns."""
    m = BBOX_LENIENT.search(text or "")
    if m:
        return [float(m.group(i)) for i in range(1, 5)], None, "bbox"
    m = POINT_LENIENT.search(text or "")
    if m:
        return None, [float(m.group(1)), float(m.group(2))], "point"
    return None, None, None


def to_unit(vals, size):
    """Rescale to 0-1 when the model answered in pixels. Returns (vals, violated).

    Scale-invariance means the divisor only matters if it is wrong: normalising
    by the image the model was actually sent is the defensible choice, and it is
    recorded per row so the decision stays auditable.
    """
    if not vals or max(abs(v) for v in vals) <= 1.0:
        return vals, False
    # Divide by the resolution the model actually SAW, not the one we uploaded.
    # The API downscales to ~1568px long edge first, and the model's pixel
    # estimates live in that space: measured on the pixel-valued subset,
    # 1568-capped scores 28.4% vs 16.0% for the sent size on ScreenSpot-v2.
    # Using the sent size manufactures misses on high-res screenshots.
    W, H = size
    sc = min(1.0, HAIKU_MAX_EDGE / max(W, H))
    W, H = W * sc, H * sc
    dims = (W, H, W, H) if len(vals) == 4 else (W, H)
    return [v / d for v, d in zip(vals, dims)], True


def extract_first_bounding_box(text: str):
    m = BBOX_RE.search(text or "")
    return [float(m.group(i)) for i in range(1, 5)] if m else None


def extract_first_point(text: str):
    m = POINT_RE.search(text or "")
    return [float(m.group(1)), float(m.group(2))] if m else None
# ---------------------------------------------------------------------------


def run_one(client, ex, budget, model, max_retries=3) -> dict:
    b64, media_type, size, shrunk = encode_image(ex.images[0], None)
    rec = {"uid": ex.uid, "dataset": ex.dataset, "gold": ex.gold, "meta": ex.meta,
           "protocol": "official_gpt4x", "model": model,
           "sent_image_sizes": [size], "preflight_downscaled": shrunk}
    for attempt in range(max_retries):
        try:
            t0 = time.monotonic()
            resp = client.messages.create(
                model=model,
                max_tokens=2048,
                # anthropic 1.0.0 does not expose `temperature` as a named arg;
                # extra_body puts it on the wire, which is what parity requires.
                extra_body={"temperature": 0.0},
                system=SYSTEM,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": media_type, "data": b64}},
                    {"type": "text", "text": USER_TMPL.format(instruction=ex.question)},
                ]}],
            )
            budget.add(resp.usage.input_tokens, resp.usage.output_tokens, model)
            text = "".join(b.text for b in resp.content if b.type == "text")
            bbox = extract_first_bounding_box(text)
            point = extract_first_point(text)
            if not point and bbox:
                point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            rec.update({"raw_response": text, "bbox": bbox, "point": point,
                        "latency_s": round(time.monotonic() - t0, 2),
                        "stop_reason": resp.stop_reason,
                        "usage": {"input_tokens": resp.usage.input_tokens,
                                  "output_tokens": resp.usage.output_tokens}})
            return rec
        except Exception as e:
            if _is_billing_error(e):
                raise FatalBillingError(str(e)) from e
            if attempt == max_retries - 1:
                rec.update({"point": None, "bbox": None, "error": f"{type(e).__name__}: {e}"})
                return rec
            time.sleep(2 ** attempt + random.random())
    return rec


def score_row(r: dict) -> dict:
    """Attach official + lenient verdicts to one raw row."""
    gold, size = r["gold"], tuple(r["sent_image_sizes"][0])
    out = {"official": None, "lenient": None, "range_violation": False, "kind": None}
    if r.get("error"):
        out["official"] = out["lenient"] = "call_error"
        return out

    text = r.get("raw_response") or ""
    bbox, point = extract_first_bounding_box(text), extract_first_point(text)
    if not point and bbox:
        point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    out["official"] = eval_sample(gold, point)

    lb, lp, kind = extract_lenient(text)
    out["kind"] = kind
    lb, v1 = to_unit(lb, size)
    lp, v2 = to_unit(lp, size)
    out["range_violation"] = bool(v1 or v2)
    if not lp and lb:
        lp = [(lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2]
    out["lenient"] = eval_sample(gold, lp)
    out["lenient_point"] = lp
    return out


def eval_sample(gold, point, errored=False) -> str:
    """eval_screenspot_pro.py:137-152 -- gold already normalised by adapters.py.

    `errored` is ours, not the official protocol's: a failed HTTP call is not a
    model format failure, and conflating the two lets a wholly broken run report
    itself as a clean 0.0% with 100% wrong_format.
    """
    if errored:
        return "call_error"
    if point is None:
        return "wrong_format"
    return "correct" if (gold[0] <= point[0] <= gold[2] and gold[1] <= point[1] <= gold[3]) else "wrong"


def report(ds: str, recs: list[dict]) -> dict:
    """Official metric set, plus the lenient companions, from one pass."""
    scored = [(r, score_row(r)) for r in recs]
    errs = [r for r, v in scored if v["official"] == "call_error"]
    if errs:
        print(f"  !! {len(errs)}/{len(scored)} calls ERRORED -- not a model result; "
              f"first: {str(errs[0].get('error'))[:110]}")
    scored = [(r, v) for r, v in scored if v["official"] != "call_error"]
    n = len(scored)
    if not n:
        return {"dataset": ds, "n": 0}

    def acc(key, rows=None):
        rows = rows if rows is not None else scored
        return sum(v[key] == "correct" for _, v in rows) / len(rows) if rows else float("nan")

    def by_type(key, t):
        rows = [(r, v) for r, v in scored if (r["meta"].get("ui_type") or "").lower() == t]
        return acc(key, rows), len(rows)

    m = {
        "dataset": ds, "n": n,
        "action_acc": acc("official"),
        "wrong_format": sum(v["official"] == "wrong_format" for _, v in scored),
        "action_acc_lenient": acc("lenient"),
        "wrong_format_lenient": sum(v["lenient"] == "wrong_format" for _, v in scored),
        "range_violation_rate": sum(v["range_violation"] for _, v in scored) / n,
    }
    for t in ("text", "icon"):
        m[f"{t}_acc"], m[f"{t}_n"] = by_type("official", t)
        m[f"{t}_acc_lenient"], _ = by_type("lenient", t)

    print(f"  official : action_acc {m['action_acc']*100:5.1f}%   wrong_format {m['wrong_format']:3d}")
    print(f"  lenient  : action_acc {m['action_acc_lenient']*100:5.1f}%   wrong_format {m['wrong_format_lenient']:3d}"
          f"   text {m['text_acc_lenient']*100:.1f}% (n={m['text_n']})"
          f"   icon {m['icon_acc_lenient']*100:.1f}% (n={m['icon_n']})")
    print(f"  answered in pixels despite being asked for 0-1: {m['range_violation_rate']*100:.1f}%")
    return m


def print_table(summary: list[dict]) -> None:
    print(f"\n{'dataset':16s} {'n':>4s} | {'official':>9s} {'wrongfmt':>8s} | "
          f"{'lenient':>8s} {'wrongfmt':>8s} {'text':>6s} {'icon':>6s} | {'px-range':>8s}")
    for m in summary:
        if not m.get("n"):
            continue
        print(f"{m['dataset']:16s} {m['n']:4d} | {m['action_acc']*100:8.1f}% {m['wrong_format']:8d} | "
              f"{m['action_acc_lenient']*100:7.1f}% {m['wrong_format_lenient']:8d} "
              f"{m['text_acc_lenient']*100:5.1f}% {m['icon_acc_lenient']*100:5.1f}% | "
              f"{m['range_violation_rate']*100:7.1f}%")
    print("\n  official = published protocol verbatim; lenient = whitespace-tolerant parse")
    print("  + pixel->0-1 rescale, the same courtesy the official parser already extends to GPT-4o")


def rescore(datasets, model) -> int:
    """Recompute every metric from saved raw responses -- no API calls."""
    summary = []
    for ds in datasets:
        f = RESULTS / f"{ds}__{short_name(model)}_official_r0.jsonl"
        if not f.exists():
            print(f"{ds}: no results at {f}"); continue
        recs = [json.loads(l) for l in open(f) if l.strip()]
        print(f"\n{ds}: rescoring {len(recs)} saved rows")
        summary.append(report(ds, recs))
    print_table(summary)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["screenspot", "screenspot_pro"])
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--match-uids-from", default="haiku-4-5_think2000_native_r0",
                    help="reuse the exact subset from an existing run, for a like-for-like comparison")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-spend", type=float, default=3.0)
    ap.add_argument("--full", action="store_true",
                    help="run the entire split instead of reusing a prior run's subset")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute metrics from saved raw responses; no API calls")
    a = ap.parse_args()

    if a.rescore:
        return rescore(a.datasets, a.model)

    client = anthropic.Anthropic()
    budget = Budget(a.max_spend)
    summary = []

    for ds in a.datasets:
        exs = {e.uid: e for e in load(ds)}
        ref = RESULTS / f"{ds}__{a.match_uids_from}.jsonl"
        uids = (list(exs) if a.full
                else [json.loads(l)["uid"] for l in open(ref) if l.strip()] if ref.exists()
                else list(exs))
        todo = [exs[u] for u in dict.fromkeys(uids) if u in exs]
        out = RESULTS / f"{ds}__{short_name(a.model)}_official_r0.jsonl"
        # Resume: a killed run must not re-pay for rows it already collected.
        done, recs = set(), []
        if out.exists():
            for line in open(out):
                if line.strip():
                    r = json.loads(line)
                    if not r.get("error"):
                        done.add(r["uid"]); recs.append(r)
        todo = [e for e in todo if e.uid not in done]
        print(f"\n{ds}: {len(done)} already done, {len(todo)} to run (official protocol) -> {out.name}")

        lock = threading.Lock()
        with open(out, "a") as fh, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            futs = {pool.submit(run_one, client, e, budget, a.model): e for e in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                with lock:
                    fh.write(json.dumps(r) + "\n"); fh.flush(); recs.append(r)
                    if i % 50 == 0 or i == len(todo):
                        print(f"  {i}/{len(todo)} | ${budget.spent:.3f}", flush=True)
                if budget.exhausted():
                    print("  !! spend cap reached"); break

        summary.append(report(ds, recs))

    print_table(summary)
    print(f"\n{budget.calls} calls | ${budget.spent:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
