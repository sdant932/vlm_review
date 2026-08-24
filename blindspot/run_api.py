"""Experiment drivers: the API-spending arms that the main runner does not cover.

Everything here is a thin wrapper over the shared request path in
`blindspot/core`. The heavy lifting -- request construction, retries, structured
output parsing, spend accounting -- lives there, and these subcommands call into
it through the same four names the production runner uses:

    run_one              blindspot.core   one example -> one result row
    Budget               blindspot.core   the hard spend ceiling
    short_name           blindspot.core   model id -> results/ filename tag
    FatalBillingError    blindspot.core   stop the whole run, do not retry

Each subcommand exists because `blindspot.core` cannot express its arm:
the runner runs *the* headline configuration over a dataset, and these need
either a different protocol, a manipulated input, a different model, or a
sampling scheme the runner has no flag for.

    official      ScreenSpot / ScreenSpot-Pro under the benchmark's OWN published
                  protocol, so the number is leaderboard-comparable. This is the
                  one arm that does not go through runner.run_one at all -- see
                  `cmd_official` for the full, deliberate list of differences.
    ablations     prompt-wording and answer-channel ablations on the
                  svg_localization point questions
    probe         harness sanity probe: a stronger model on byte-identical inputs
    derived       the svg_localization-derived counting / word-presence sets,
                  with their required paired blind arm
    controls      blind and one-page controls that isolate *why* an arm fails
    grid          does the model fail to locate, or fail to say where?
    coord-probe   a stronger model on the items Haiku hit and missed

    python -m blindspot.run_api <subcommand> --help

Every subcommand that spends money takes --max-spend and enforces it through
`Budget`. `official --rescore` is the only path here that makes no API calls.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from PIL import Image, ImageDraw, ImageFont

from blindspot.core import ADAPTERS, Example, load
from blindspot.core import POINT_INSTRUCTION, encode_image, HAIKU_MAX_EDGE
from blindspot.core import (Budget, FatalBillingError, MODEL, _is_billing_error,
                                   model_spec, run_one, short_name)
from blindspot.core import point_in_bbox

RESULTS = Path("results")

# The identical blind-control wording is used by both `derived` and `controls`,
# on purpose: it makes the blind arms of the two studies comparable rather than
# merely similar. Defining it once is what enforces that.
BLIND_NOTE = (
    "\n\nNo image is provided. Answer from the question alone, using your best "
    "guess. You must commit to a specific answer; do not say that you cannot tell."
)


# ============================================================ shared client

# Sentinel: ask for the SDK's own defaults instead of the pinned pair below.
SDK_DEFAULTS = object()


def make_client(max_retries=0, timeout=120.0):
    """The single Anthropic client construction point for every arm here.

    Five of the seven arms want the same pinned pair -- `max_retries=0` because
    they drive their own concurrency and want a failed row recorded rather than
    silently re-billed, and `timeout=120.0` so a hung request cannot stall a
    ThreadPoolExecutor for the SDK's default ten minutes.

    Two arms deliberately do NOT: `official` and `coord-probe` were written
    against a bare `anthropic.Anthropic()`. `official` in particular implements
    its own retry loop (`official_run_one`, max_retries=3 with backoff) layered
    on top of the SDK's, and the published numbers in results/ were collected
    that way. They pass SDK_DEFAULTS to say so explicitly rather than inheriting
    a retry policy they never had.
    """
    if max_retries is SDK_DEFAULTS or timeout is SDK_DEFAULTS:
        return anthropic.Anthropic()
    return anthropic.Anthropic(max_retries=max_retries, timeout=timeout)


# ============================================================ official
#
# Ports third_party/ScreenSpot-Pro-GUI-Grounding/models/gpt4x.py verbatim -- the
# adapter behind the published GPT-4o numbers -- so our result is comparable to
# published work instead of merely internally consistent.
#
# Differences from blindspot/core/runner.py, all deliberate:
#
#     output shape   bounding box [[x0,y0,x1,y1]], not a point
#     range          0-1 floats, not 0-1000 integers
#     reasoning      none ("Don't output any analysis"), temperature 0
#     parsing        regex over free text, not structured outputs
#     format failure counted as wrong AND reported as wrong_format_num
#
# Scoring mirrors eval_screenspot_pro.py:115-152 -- action_acc / text_acc /
# icon_acc, with wrong_format in the denominator.
#
# This is why `official` does not call runner.run_one: it has its own request
# function, `official_run_one`, below.

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


def official_run_one(client, ex, budget, model, max_retries=3) -> dict:
    """The official protocol's own request path.

    Named apart from `blindspot.core.run_one` (imported above and used by
    every other subcommand) precisely because it is NOT that function: no
    structured outputs, no thinking budget, temperature 0, regex parsing.
    """
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


def cmd_official(a) -> int:
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

        python -m blindspot.run_api official --datasets screenspot_pro --max-spend 2
    """
    if a.rescore:
        return rescore(a.datasets, a.model)

    client = make_client(SDK_DEFAULTS)
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
            futs = {pool.submit(official_run_one, client, e, budget, a.model): e for e in todo}
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


# ============================================================ ablations

CROPS = Path("cache/svgloc_crops")
RUNGS = ("small", "large")

CAREFUL = (
    "Locate the described UI element in the screenshot and return the point at "
    "its center.\n"
    "Use a normalized coordinate system where x=0 is the left edge, x=1000 the "
    "right edge, y=0 the top edge, and y=1000 the bottom edge.\n"
    "Be precise: the target is small, often under 1% of the image, so an answer "
    "that is merely in the right region will miss it. Read off the element's "
    "left and right edges and take the midpoint, then its top and bottom edges "
    "and take the midpoint.\n"
    "Always return your single best guess, even if you are uncertain.\n\n"
    "Element: ")

DESCRIBE = (
    "Locate the described UI element in the screenshot.\n"
    "First, in your reasoning, describe in words where it sits: which part of "
    "the image, what is immediately above, below, left and right of it, and how "
    "far across and down the frame it falls.\n"
    "Then convert that description into a point at the element's center, using "
    "a normalized coordinate system where x=0 is the left edge, x=1000 the "
    "right edge, y=0 the top edge, and y=1000 the bottom edge.\n"
    "Always return your single best guess, even if you are uncertain.\n\n"
    "Element: ")

CELL_THEN_POINT = (
    "Locate the described UI element in the screenshot.\n"
    "Work coarse to fine. First, in your reasoning, divide the image into a 4x4 "
    "grid of equal cells, columns A-D from the left and rows 1-4 from the top, "
    "and name the single cell that contains the element. Then divide that cell "
    "into a 4x4 grid and name the sub-cell. Then convert the sub-cell to a "
    "point.\n"
    "Report the point at the element's center in a normalized coordinate system "
    "where x=0 is the left edge, x=1000 the right edge, y=0 the top edge, and "
    "y=1000 the bottom edge.\n"
    "Always return your single best guess, even if you are uncertain.\n\n"
    "Element: ")

LANDMARK = (
    "Locate the described UI element in the screenshot.\n"
    "First, in your reasoning, find the largest nearby landmark you are "
    "confident about -- a title, an axis, a panel edge, a big labelled box -- "
    "and estimate its position. Then state where the target sits relative to "
    "that landmark, as a fraction of the image width and height. Then combine "
    "the two into an absolute position.\n"
    "Report the point at the element's center in a normalized coordinate system "
    "where x=0 is the left edge, x=1000 the right edge, y=0 the top edge, and "
    "y=1000 the bottom edge.\n"
    "Always return your single best guess, even if you are uncertain.\n\n"
    "Element: ")

BBOX_INSTRUCTION = (
    "Locate the described UI element in the screenshot and return its bounding "
    "box.\n"
    "Use a normalized coordinate system where x=0 is the left edge, x=1000 the "
    "right edge, y=0 the top edge, and y=1000 the bottom edge. Report x0,y0 as "
    "the top-left corner and x1,y1 as the bottom-right corner of the element.\n"
    "Always return your single best guess, even if you are uncertain.\n\n"
    "Element: ")

QUADRANT_LETTERS = ["top-left", "top-right", "bottom-left", "bottom-right"]

ARMS = ["repeat", "quadrant_mc", "crop", "bbox",
        "careful", "describe", "cell_then_point", "landmark"]


def abl_sample(n: int, seed: int) -> list[Example]:
    """Deterministic, balanced across rung x area tertile."""
    pts = [e for e in load("svg_localization")
           if e.answer_type == "point" and e.meta["resolution"] in RUNGS]
    per_rung = n // len(RUNGS)
    out = []
    for rung in RUNGS:
        rows = sorted((e for e in pts if e.meta["resolution"] == rung),
                      key=lambda e: (e.meta["target_area_frac"], e.uid))
        for t in range(3):
            band = rows[t * len(rows) // 3:(t + 1) * len(rows) // 3]
            want = per_rung // 3 + (1 if t < per_rung % 3 else 0)
            out += random.Random(f"{seed}:{rung}:{t}").sample(band, min(want, len(band)))
    return out


def quadrant_of(box) -> int:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return (0 if cy < 0.5 else 2) + (0 if cx < 0.5 else 1)


def make_crop(e: Example) -> Example | None:
    """Crop to the half-width, half-height quadrant containing the target.

    The search field shrinks 4x; the target keeps its absolute pixel size, so
    precision demands are unchanged in pixels while the field is smaller. Gold
    is remapped into the crop's own normalized frame.
    """
    CROPS.mkdir(parents=True, exist_ok=True)
    src = Path(e.images[0])
    b = e.gold
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    x0 = min(max(cx - 0.25, 0.0), 0.5)
    y0 = min(max(cy - 0.25, 0.0), 0.5)
    x1, y1 = x0 + 0.5, y0 + 0.5
    if not (x0 <= b[0] and b[2] <= x1 and y0 <= b[1] and b[3] <= y1):
        return None                     # target straddles the crop edge; skip
    out = CROPS / f"{e.uid.replace(':', '_')}.png"
    if not out.exists():
        im = Image.open(src)
        W, H = im.size
        im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))).save(out)
    g = [(b[0] - x0) / 0.5, (b[1] - y0) / 0.5, (b[2] - x0) / 0.5, (b[3] - y0) / 0.5]
    return Example(uid=e.uid, dataset=e.dataset, images=[str(out)], question=e.question,
                   answer_type="point", gold=g, meta={**e.meta, "arm": "crop"})


def abl_build(arm: str, exs: list[Example]) -> list[Example]:
    if arm in ("baseline", "repeat"):
        return exs
    if arm == "quadrant_mc":
        out = []
        for e in exs:
            q = quadrant_of(e.gold)
            out.append(Example(uid=e.uid, dataset=e.dataset, images=e.images,
                               question=f"Which quarter of the image contains {e.question}?",
                               answer_type="choice", gold=["ABCD"[q]],
                               meta={**e.meta, "options": QUADRANT_LETTERS, "arm": arm}))
        return out
    if arm == "crop":
        return [c for c in (make_crop(e) for e in exs) if c]
    if arm == "bbox":
        return [Example(uid=e.uid, dataset=e.dataset, images=e.images, question=e.question,
                        answer_type="bbox", gold=e.gold,
                        meta={**e.meta, "arm": arm,
                              "prompt_override": BBOX_INSTRUCTION + e.question})
                for e in exs]
    text = {"careful": CAREFUL, "describe": DESCRIBE,
            "cell_then_point": CELL_THEN_POINT, "landmark": LANDMARK}[arm]
    return [Example(uid=e.uid, dataset=e.dataset, images=e.images, question=e.question,
                    answer_type="point", gold=e.gold,
                    meta={**e.meta, "arm": arm, "prompt_override": text + e.question})
            for e in exs]


def cmd_ablations(a) -> int:
    """Diagnostic and prompt-wording ablations for svg_localization point questions.

    Two questions, one shared sample so every arm is paired against every other.

    DIAGNOSTIC -- is the deficit knowledge or expression?
      repeat        the identical request a second time. Thinking pins temperature
                    to 1, so the spread between two runs is the in-set noise floor
                    the localization set otherwise lacks, and it separates "noisy
                    estimate" from "stable but wrong".
      quadrant_mc   which quadrant is it in, as a 4-way letter choice. Same
                    granularity as the 2x2 rung of the precision curve, but with the
                    continuous channel removed entirely. This is the grid-control
                    idea WITHOUT drawing on the image -- the dataset forbids an
                    overlay, which occluded the gold text in 31% of a trial build.
      crop          the same question on an image cropped to the quadrant holding
                    the target. Shrinks the search field without changing precision
                    demands, separating "could not find it" from "could not aim".
      bbox          ask for the element's box instead of its centre, scored as
                    centre-of-predicted-box inside gold so it stays comparable.

    WORDING -- can we just ask better? Each arm keeps the image, the schema, the
    model and the thinking budget byte-identical and varies only the text, so any
    difference is the wording and nothing else. All four are built on the finding
    that coarse spatial ability is intact (~68%) while precision is not, so they
    scaffold coarse-to-fine rather than simply asking harder.
    """
    exs = abl_sample(a.n, a.seed)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "svgloc_ablation_uids.json").write_text(
        json.dumps([e.uid for e in exs], indent=1))
    print(f"shared sample: {len(exs)} point questions "
          f"({sum(1 for e in exs if e.meta['resolution']=='small')} small / "
          f"{sum(1 for e in exs if e.meta['resolution']=='large')} large)")

    client = make_client()
    budget = Budget(a.max_spend)
    tag = f"{short_name(a.model)}_think{a.thinking_budget}"
    for arm in a.arms:
        todo = abl_build(arm, exs)
        out = RESULTS / f"svgloc_abl_{arm}__{tag}.jsonl"
        have = set()
        if out.exists():
            for line in open(out):
                if line.strip():
                    r = json.loads(line)
                    if r.get("pred") is not None:
                        have.add(r["uid"])
        todo = [e for e in todo if e.uid not in have]
        if not todo:
            print(f"  {arm:16s} already complete"); continue
        n_ok = 0
        with open(out, "a") as fh, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            futs = [pool.submit(run_one, client, e, budget, a.thinking_budget, None, a.model)
                    for e in todo]
            for f in as_completed(futs):
                rec = f.result()
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                n_ok += rec.get("pred") is not None
        print(f"  {arm:16s} {n_ok}/{len(todo)} usable  (${budget.spent:.2f})")
        if budget.exhausted():
            print("  !! budget ceiling reached"); break
    print(f"done | ${budget.spent:.3f} | {budget.calls} calls")
    return 0


# ============================================================ probe

DS = "svg_localization"


def probe_sample_rung(rung: str, n: int, seed: int) -> list:
    """n point questions from one rung, stratified across area tertiles.

    Same deterministic scheme as probe_sample(), restricted to a single resolution
    so a rung can be measured on its own without paying for the other two.
    """
    pts = [e for e in load(DS)
           if e.answer_type == "point" and e.meta["resolution"] == rung]
    rows = sorted(pts, key=lambda e: (e.meta["target_area_frac"], e.uid))
    total, out = len(rows), []
    for t in range(3):
        band = rows[t * total // 3:(t + 1) * total // 3]
        want = n // 3 + (1 if t < n % 3 else 0)
        rng = random.Random(f"{seed}:{rung}:{t}")
        out += rng.sample(band, min(want, len(band)))
    return out


def probe_sample(per_cell: int, seed: int) -> list:
    """Deterministic, stratified by (resolution x target-area tertile).

    Area tertiles are cut *within* a resolution: target_area_frac is a fraction
    of the image, so the same target occupies the same fraction at every rung.
    Cutting globally would therefore just reproduce the resolution split.
    """
    pts = [e for e in load(DS) if e.answer_type == "point"]
    by_res = defaultdict(list)
    for e in pts:
        by_res[e.meta["resolution"]].append(e)
    out = []
    for res in sorted(by_res):
        rows = sorted(by_res[res], key=lambda e: (e.meta["target_area_frac"], e.uid))
        n = len(rows)
        for t in range(3):
            band = rows[t * n // 3:(t + 1) * n // 3]
            rng = random.Random(f"{seed}:{res}:{t}")
            out += rng.sample(band, min(per_cell, len(band)))
    return out


def probe_arm(client, examples, model, max_edge, budget, thinking, concurrency,
              suffix: str = "") -> Path:
    cond = "native" if max_edge is None else f"edge{max_edge}"
    out = RESULTS / f"{DS}__probe{suffix}_{short_name(model)}_think{thinking}_{cond}_r0.jsonl"
    with open(out, "w") as fh, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(run_one, client, e, budget, thinking, max_edge, model)
                for e in examples]
        for f in futs:
            fh.write(json.dumps(f.result()) + "\n")
    return out


def summarise(path: Path, examples: list) -> dict:
    golds = {e.uid: e.gold for e in examples}
    meta = {e.uid: e.meta for e in examples}
    rows = [json.loads(l) for l in open(path) if l.strip()]
    usable = [r for r in rows if r.get("pred") is not None]
    per_res: Counter = Counter()
    per_res_n: Counter = Counter()
    for r in usable:
        ok = point_in_bbox(tuple(r["pred"]), golds[r["uid"]])
        res = meta[r["uid"]]["resolution"]
        per_res[res] += ok
        per_res_n[res] += 1
    hits = sum(point_in_bbox(tuple(r["pred"]), golds[r["uid"]]) for r in usable)
    return {"file": path.name, "rows": len(rows), "usable": len(usable),
            "unusable": len(rows) - len(usable),
            "hits": int(hits), "acc": hits / max(len(usable), 1),
            "by_res": {k: (per_res[k] / per_res_n[k], per_res_n[k]) for k in sorted(per_res_n)}}


def cmd_probe(a) -> int:
    """Harness sanity probe for data/svg_localization (EVAL.md 3.9).

    A near-zero localization score has two explanations: the model cannot do it, or
    the harness is broken. On a dataset that has never been run against any model
    that ambiguity is sharper than it was on ScreenSpot-Pro, not weaker -- there is
    no prior run to sanity-check against.

    So: run a stronger model (Sonnet) on byte-identical inputs before reporting
    anything. If Sonnet lands in the boxes, the pipeline is sound and a low Haiku
    score is a capability result. If both models score near zero, suspect the
    dataset or the harness and check verify/index.html before writing a conclusion.

    This does NOT reuse the `coord-probe` subcommand: that one stratifies by which
    items Haiku already hit or missed, which presupposes the Haiku run this probe is
    meant to precede. Sampling here is from the manifest alone.

    Three arms on the SAME uids:
        haiku   native      -- the target of the study
        sonnet  native       -- is the pipeline sound?
        sonnet  edge1568     -- ...or was that just Sonnet's bigger image budget?

    Sonnet 5's image ceiling (~2576px) is higher than Haiku's (~1568px), and `large`
    is 3000x1900, so the two models would not otherwise receive the same thing.

        python -m blindspot.run_api probe --per-cell 10 --max-spend 4
    """
    exs = probe_sample_rung(a.rung, a.n, a.seed) if a.rung else probe_sample(a.per_cell, a.seed)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{DS}__probe{'_'+a.rung if a.rung else ''}_uids.json").write_text(
        json.dumps([e.uid for e in exs], indent=1))
    print(f"probe sample: {len(exs)} point questions "
          f"({Counter(e.meta['resolution'] for e in exs)})")

    client = make_client()
    budget = Budget(a.max_spend)
    all_arms = {"target-native": (a.target_model, None),
                "strong-native": (a.strong_model, None),
                "strong-edge1568": (a.strong_model, 1568)}
    plan = [all_arms[k.strip()] for k in a.arms.split(",") if k.strip() in all_arms]
    out = []
    for model, max_edge in plan:
        p = probe_arm(client, exs, model, max_edge, budget, a.thinking_budget, a.concurrency,
                      suffix=f"_{a.rung}" if a.rung else "")
        s = summarise(p, exs)
        s["model"], s["max_edge"] = model, max_edge
        out.append(s)
        by = "  ".join(f"{k} {v[0]*100:.0f}% (n={v[1]})" for k, v in s["by_res"].items())
        print(f"  {model:32s} {'native' if max_edge is None else f'edge{max_edge}':9s} "
              f"click-in-bbox {s['acc']*100:5.1f}%  ({s['hits']}/{s['usable']}, "
              f"{s['unusable']} unusable)   {by}   ${budget.spent:.2f}")
    (RESULTS / f"{DS}__probe{'_'+a.rung if a.rung else ''}_summary.json").write_text(
        json.dumps(out, indent=1))
    print(f"total ${budget.spent:.3f}")
    return 0


# ============================================================ derived


def blind_of(e: Example) -> Example:
    return Example(uid=f"blind:{e.uid}", dataset=f"{e.dataset}_blind", images=[],
                   question=e.question + BLIND_NOTE, answer_type=e.answer_type,
                   gold=e.gold, meta={**e.meta, "control": "blind", "src_uid": e.uid})


def done_uids(path: Path) -> set[str]:
    out = set()
    if not path.exists():
        return out
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("pred") is not None and not r.get("error") and not r.get("parse_error"):
            out.add(r["uid"])
    return out


def derived_run(client, examples, out: Path, budget: Budget, thinking: int, concurrency: int,
                model: str) -> None:
    have = done_uids(out)
    todo = [e for e in examples if e.uid not in have]
    if not todo:
        print(f"  {out.name}: already complete ({len(have)} rows)")
        return
    lock = threading.Lock()
    written = failed = 0
    with open(out, "a") as fh, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(run_one, client, e, budget, thinking, None, model): e for e in todo}
        for f in as_completed(futs):
            try:
                rec = f.result()
            except FatalBillingError:
                print("  !! billing error -- stopping")
                break
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                written += 1
                failed += rec.get("pred") is None
                if written % 100 == 0:
                    print(f"    {written}/{len(todo)} | ${budget.spent:.3f} | "
                          f"no-pred {failed}", flush=True)
            if budget.exhausted():
                print(f"  !! budget ceiling ${budget.limit} reached")
                break
    print(f"  {out.name}: +{written} rows (${budget.spent:.3f} total, {failed} without a prediction)")


def cmd_derived(a) -> int:
    """Run the svg_localization-derived sets (counting, word_mc) on selected rungs.

    A thin wrapper over the production request path rather than a new one: the same
    `runner.run_one`, the same schema, the same scorer. The only thing this adds is
    rung selection and a paired blind arm, because both EVAL.md files require the
    blind control and neither the runner nor the `controls` subcommand can express
    "these rungs only".

        python -m blindspot.run_api derived --datasets svg_counting svg_word_mc \
            --rungs small large --max-spend 12

    Resumes: rows already present with a usable prediction are skipped, so a killed
    run can be restarted without paying twice.
    """
    RESULTS.mkdir(exist_ok=True)
    client = make_client()
    budget = Budget(a.max_spend)
    tag = f"{short_name(a.model)}_think{a.thinking_budget}_native_r0"
    rungs = set(a.rungs)

    for ds in a.datasets:
        exs = [e for e in ADAPTERS[ds]() if e.meta["resolution"] in rungs]
        print(f"{ds}: {len(exs)} questions at rungs {sorted(rungs)}")
        derived_run(client, exs, RESULTS / f"{ds}__{tag}.jsonl", budget,
                    a.thinking_budget, a.concurrency, a.model)
        if not a.skip_blind:
            # Both EVAL.md files require this: whatever survives with the image
            # withheld was never a perception task.
            derived_run(client, [blind_of(e) for e in exs], RESULTS / f"{ds}__blind_{tag}.jsonl",
                        budget, a.thinking_budget, a.concurrency, a.model)
    print(f"done | ${budget.spent:.3f} | {budget.calls} calls")
    return 0


# ============================================================ controls


def blind_examples(dataset: str, n: int, seed: int = 0) -> list[Example]:
    """Same question, same schema, image stripped."""
    pool = [e for e in ADAPTERS[dataset]() if e.answer_type != "point"]
    random.Random(seed).shuffle(pool)
    out = []
    for e in pool[:n]:
        out.append(Example(
            uid=f"blind:{e.uid}", dataset=f"{e.dataset}_blind",
            images=[],                      # the manipulation
            question=e.question + BLIND_NOTE,
            answer_type=e.answer_type,
            gold=e.gold, meta={**e.meta, "control": "blind", "src_uid": e.uid},
        ))
    return out


def onepage_examples(which: int = 0) -> list[Example]:
    """SlideVQA multi-evidence questions, given only one of their evidence slides."""
    out = []
    for e in ADAPTERS["slidevqa"]():
        if not e.meta.get("multi_page"):
            continue
        idx = which if which < len(e.images) else len(e.images) - 1
        out.append(Example(
            uid=f"onepage{which}:{e.uid}", dataset="slidevqa_onepage",
            images=[e.images[idx]],         # the manipulation
            question=e.question, answer_type=e.answer_type, gold=e.gold,
            meta={**e.meta, "control": f"onepage{which}", "src_uid": e.uid,
                  "n_pages_sent": 1},
        ))
    return out


def cmd_controls(a) -> int:
    """Controls that isolate *why* Haiku fails, run as ablations on the main eval.

    Three experiments, all reusing the production request path (`runner.run_one`) so
    that the only thing that differs from the headline run is the manipulation
    itself -- same model, same thinking budget, same schema, same scorer.

      blind      Ask the identical question with NO image. Anything still answered
                 correctly was never a perception task: it was recoverable from the
                 question text plus world knowledge. This is the ceiling that must be
                 subtracted from every accuracy number before calling it "vision".

      onepage    SlideVQA multi-evidence questions with only the FIRST evidence slide.
                 If accuracy holds, the question never genuinely spanned pages and the
                 "multi-hop" label is a dataset artifact. If it collapses, the span is
                 real and the earlier -4.0 F1 multi-page cost is a true integration cost.

      grid       ScreenSpot-Pro with a labelled 4x4 grid drawn over the screenshot;
                 the model names the CELL rather than emitting coordinates. Click
                 accuracy conflates "did not see it" with "saw it, cannot say where".
                 Grid accuracy >> click accuracy would prove the deficit is expression.
                 (That arm lives in the `grid` subcommand.)

    Nothing here overwrites an existing result file; every arm writes its own
    results/control_<arm>*.jsonl.
    """
    if a.arm == "blind":
        exs, out = [], RESULTS / "control_blind.jsonl"
        for d in a.datasets:
            exs += blind_examples(d, a.n)
    else:
        exs, out = onepage_examples(a.which), RESULTS / f"control_onepage{a.which}.jsonl"

    done = set()
    if out.exists():
        for line in out.open():
            try: done.add(json.loads(line)["uid"])
            except Exception: pass
    exs = [e for e in exs if e.uid not in done]
    print(f"{a.arm}: {len(exs)} to run ({len(done)} already done) -> {out}", flush=True)
    if not exs:
        return 0

    budget = Budget(a.max_spend)
    client = make_client()
    n = 0
    with out.open("a") as f, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futs = {pool.submit(run_one, client, e, budget, 2000, None): e for e in exs}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except FatalBillingError as e:
                print("FATAL BILLING:", e); break
            f.write(json.dumps(rec) + "\n"); n += 1
            if n % 100 == 0:
                f.flush(); print(f"  {n}/{len(exs)} | ${budget.spent:.3f}", flush=True)
    print(f"done | {n} calls | ${budget.spent:.3f}", flush=True)
    return 0


# ============================================================ grid

K = 4
GRID_DIR = Path("cache/grid")
ROWS = "ABCD"

INSTR = (
    f"The screenshot is divided into a {K}x{K} grid drawn in magenta. Rows are "
    f"labelled A (top) to {ROWS[K-1]} (bottom); columns are numbered 1 (left) to {K} (right). "
    "Each cell is labelled in its top-left corner.\n"
    "Reply with ONLY the label of the single cell that contains the described element, "
    "for example B3. Always commit to one cell even if you are unsure.\n\n"
    "Element: "
)


def gridded(src: str) -> str:
    """Draw the labelled grid once per screenshot; cache by source path."""
    # In the standalone grid_control.py this mkdir ran at import time. Here it is
    # at first use instead, so that merely running `--help` (or any other
    # subcommand) does not create cache/grid as a side effect.
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    out = GRID_DIR / (re.sub(r"[^A-Za-z0-9]+", "_", src)[-120:] + f".k{K}.jpg")
    if out.exists():
        return str(out)
    with Image.open(src) as im:
        im = im.convert("RGB")
        if max(im.size) > 1568:                    # what the model sees anyway
            s = 1568 / max(im.size)
            im = im.resize((round(im.width*s), round(im.height*s)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        w, h = im.size
        try: font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", max(14, w//55))
        except Exception: font = ImageFont.load_default()
        for i in range(1, K):
            d.line([(w*i/K, 0), (w*i/K, h)], fill=(255, 0, 255), width=2)
            d.line([(0, h*i/K), (w, h*i/K)], fill=(255, 0, 255), width=2)
        for r in range(K):
            for c in range(K):
                lab = f"{ROWS[r]}{c+1}"
                x, y = w*c/K + 5, h*r/K + 4
                d.rectangle([x-3, y-2, x+w//22, y+h//30], fill=(255, 0, 255))
                d.text((x, y), lab, fill=(255, 255, 255), font=font)
        im.save(out, quality=88)
    return str(out)


def cell_of(x, y):
    return f"{ROWS[min(int(y*K), K-1)]}{min(int(x*K), K-1)+1}"


def cmd_grid(a) -> int:
    """Does Haiku fail to LOCATE, or fail to SAY where?

    Click-in-bbox conflates two very different deficits. This arm removes coordinate
    emission entirely: a labelled 4x4 grid is drawn over the screenshot and the model
    names the cell (e.g. "B3") containing the target.

    The comparison is apples-to-apples because the click predictions from the main run
    can be reduced to the same 4x4 cell. Same items, same granularity, only the answer
    FORMAT differs:

        click -> derived cell   31.2%   (model emitted x,y; we bucketed it)
        named cell              ????    (model names the cell directly)

    If naming beats clicking, coordinate emission is lossy and the perception was
    partly intact. If they match, the ceiling is perceptual and no output format saves it.
    """
    pool = list(ADAPTERS["screenspot_pro"]())
    random.Random(0).shuffle(pool)
    exs = []
    for e in pool[:a.n]:
        x0, y0, x1, y1 = e.gold
        exs.append(Example(
            uid=f"grid:{e.uid}", dataset="screenspot_pro_grid",
            images=[gridded(e.images[0])],
            question=INSTR + e.question,
            answer_type="span",
            gold=[cell_of((x0+x1)/2, (y0+y1)/2)],
            meta={**e.meta, "control": "grid4", "src_uid": e.uid, "bbox": e.gold},
        ))
    out = Path("results/control_grid4.jsonl")
    done = set()
    if out.exists():
        done = {json.loads(l)["uid"] for l in out.open() if l.strip()}
    exs = [e for e in exs if e.uid not in done]
    print(f"grid: {len(exs)} to run -> {out}", flush=True)
    budget = Budget(a.max_spend)
    client = make_client()
    n = 0
    with out.open("a") as f, ThreadPoolExecutor(max_workers=a.concurrency) as pl:
        futs = [pl.submit(run_one, client, e, budget, 2000, None) for e in exs]
        for fut in as_completed(futs):
            try: rec = fut.result()
            except FatalBillingError as e: print("FATAL:", e); break
            f.write(json.dumps(rec)+"\n"); n += 1
            if n % 50 == 0: f.flush(); print(f"  {n}/{len(exs)} | ${budget.spent:.3f}", flush=True)
    print(f"done | {n} calls | ${budget.spent:.3f}", flush=True)
    return 0


# ============================================================ coord-probe

HAIKU_TAG = "haiku-4-5_think2000_native_r0"


def haiku_rows(ds: str) -> dict[str, dict]:
    p = RESULTS / f"{ds}__{HAIKU_TAG}.jsonl"
    out = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            out[r["uid"]] = r
    return out


def pick(ds: str, n_hit: int, n_miss: int) -> list[str]:
    """Stratified, deterministic: some Haiku hits, some misses, smallest targets first."""
    exs = {e.uid: e for e in load(ds)}
    hits, misses = [], []
    for uid, r in sorted(haiku_rows(ds).items()):
        e = exs.get(uid)
        if e is None or not r.get("pred"):
            continue
        (hits if point_in_bbox(tuple(r["pred"]), e.gold) else misses).append(uid)
    misses.sort(key=lambda u: exs[u].meta.get("target_area_frac", 1.0))
    return hits[:n_hit] + misses[:n_miss]


def cmd_coord_probe(a) -> int:
    """Sonnet control probe: does a stronger model land in the gold boxes?

    Purpose is a harness sanity check, not a model comparison. If Sonnet hits boxes
    that Haiku misses on byte-identical inputs, the coordinate pipeline is sound and
    Haiku's low score is a capability result rather than a bug.

    Two conditions, because Sonnet 5's image ceiling (~2576px) is higher than Haiku's
    (~1568px): a native win could be resolution rather than model. Running Sonnet
    handicapped to Haiku's pixel budget separates those.

        python -m blindspot.run_api coord-probe --max-spend 1
    """
    plan = {"screenspot": pick("screenspot", 2, 3), "screenspot_pro": pick("screenspot_pro", 1, 4)}
    sel = {ds: {e.uid: e for e in load(ds) if e.uid in set(u)} for ds, u in plan.items()}

    print(f"probe model: {a.model}")
    for ds, uids in plan.items():
        print(f"  {ds}: {len(uids)} examples")
    (RESULTS).mkdir(exist_ok=True)
    Path("results/probe_uids.json").write_text(json.dumps(plan, indent=2))

    client = make_client(SDK_DEFAULTS)
    budget = Budget(a.max_spend)

    for cond, max_edge in [("native", None), ("edge1568", 1568)]:
        for ds, uids in plan.items():
            tag = f"{short_name(a.model)}_think{a.thinking_budget}_{cond}"
            out = RESULTS / f"{ds}__{tag}_r0.jsonl"
            todo = [sel[ds][u] for u in uids]
            with open(out, "w") as fh, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
                futs = [pool.submit(run_one, client, e, budget, a.thinking_budget, max_edge, a.model)
                        for e in todo]
                for f in futs:
                    fh.write(json.dumps(f.result()) + "\n")
            print(f"  wrote {out.name}  (${budget.spent:.3f} so far)")

    print(f"\ndone | {budget.calls} calls | ${budget.spent:.3f}")
    return 0


# ============================================================ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="blindspot.run_api",
        description="Experiment drivers over the shared runner in blindspot/core.")
    sub = ap.add_subparsers(dest="cmd", metavar="SUBCOMMAND", required=True)

    p = sub.add_parser("official", help="ScreenSpot / ScreenSpot-Pro under the published protocol")
    p.add_argument("--datasets", nargs="+", default=["screenspot", "screenspot_pro"])
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--match-uids-from", default="haiku-4-5_think2000_native_r0",
                   help="reuse the exact subset from an existing run, for a like-for-like comparison")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-spend", type=float, default=3.0)
    p.add_argument("--full", action="store_true",
                   help="run the entire split instead of reusing a prior run's subset")
    p.add_argument("--rescore", action="store_true",
                   help="recompute metrics from saved raw responses; no API calls")
    p.set_defaults(fn=cmd_official)

    p = sub.add_parser("ablations", help="prompt-wording / answer-channel ablations on point questions")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--thinking-budget", type=int, default=2000)
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--max-spend", type=float, default=16.0)
    p.set_defaults(fn=cmd_ablations)

    p = sub.add_parser("probe", help="harness sanity probe: a stronger model on identical inputs")
    p.add_argument("--per-cell", type=int, default=10)
    p.add_argument("--rung", choices=["small", "medium", "large"], default=None,
                   help="restrict to one resolution rung (use with --n)")
    p.add_argument("--n", type=int, default=100, help="sample size when --rung is given")
    p.add_argument("--arms", default="strong-native",
                   help="comma-separated subset of: target-native,strong-native,strong-edge1568")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--strong-model", default="claude-sonnet-5")
    p.add_argument("--target-model", default="claude-haiku-4-5-20251001")
    p.add_argument("--thinking-budget", type=int, default=2000)
    p.add_argument("--max-spend", type=float, default=4.0)
    p.add_argument("--concurrency", type=int, default=8)
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("derived", help="the svg_localization-derived counting / word-presence sets")
    p.add_argument("--datasets", nargs="+", default=["svg_counting", "svg_word_mc"])
    p.add_argument("--rungs", nargs="+", default=["small", "large"])
    p.add_argument("--model", default=MODEL)
    p.add_argument("--thinking-budget", type=int, default=2000)
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--max-spend", type=float, default=12.0)
    p.add_argument("--skip-blind", action="store_true")
    p.set_defaults(fn=cmd_derived)

    p = sub.add_parser("controls", help="blind / one-page controls that isolate why an arm fails")
    p.add_argument("--arm", required=True, choices=["blind", "onepage"])
    p.add_argument("--datasets", nargs="*", default=["charxiv", "infographicvqa", "slidevqa", "ai2d"])
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--which", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--max-spend", type=float, default=4.0)
    p.set_defaults(fn=cmd_controls)

    p = sub.add_parser("grid", help="name the grid cell instead of emitting coordinates")
    p.add_argument("--n", type=int, default=350)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-spend", type=float, default=2.0)
    p.set_defaults(fn=cmd_grid)

    p = sub.add_parser("coord-probe", help="a stronger model on the items Haiku hit and missed")
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--thinking-budget", type=int, default=2000)
    p.add_argument("--max-spend", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=5)
    p.set_defaults(fn=cmd_coord_probe)

    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
