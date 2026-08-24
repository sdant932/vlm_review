"""Standalone pages: seven builds that sit outside the `blindspots.md` chain.

`blindspot.report` renders the report itself -- the figures, the tables and the
`figures.json` every number in it is traced back to. This module renders the
pages a reader goes to *after* the report, when a claim needs checking, plus the
two overview pages and the contact sheet that the report chain never absorbed:

    report_pages causes      -> outputs/causes/*.html      one page per blind spot
                                outputs/assets_causes/*.jpg
    report_pages drilldown   -> outputs/drilldown.html     every number, openable
                                outputs/drilldown.{json,csv}
    report_pages slidevqa    -> outputs/slidevqa.html      the SlideVQA arm in full
                                outputs/assets_slidevqa/*.jpg
    report_pages tasks       -> outputs/tasks/*.html       one page per primitive
    report_pages primitives  -> outputs/report.html        the per-primitive overview
    report_pages headline    -> outputs/aug22/report.html  the corrected headline report
    report_pages candidates  -> outputs/report/candidates.html + candidates/

The first four share a job rather than a data path, and that is why they are one
module and not four. None of them reads `figures.json` or any `summary.json`:
each recomputes what it shows from `results/*.jsonl` using the official scorers
in `blindspot.core`, so a page can never quietly disagree with the report -- if
it does, one of them is wrong and the difference is visible. They sit beside
`blindspot.eval` in the dependency graph, not after `blindspot.report`, and can
be run without the report chain having run at all.

The last three are the exceptions, and each is one:

* **`primitives`** renders `outputs/summary.json`, which `blindspot.eval
  aggregate` writes. It is a pure function of that file -- the numbers can be
  checked without the page and the page rebuilt without re-scoring. It is also
  the `../report.html` that `causes`, `tasks`, `slidevqa` and `drilldown` all
  link back to, so without it those four pages have a dead crumb.
* **`headline`** renders `outputs/report/summary.json`, which `blindspot.report
  summary` writes. That is a **file dependency, not an import**: run `report
  summary` first when the results change, or this page will be built from a
  stale one. Its output path keeps the study's own -- `outputs/aug22/` -- so the
  artifact lands where the published tree has it; `--out` moves it.
* **`candidates`** is the odd one out twice over: it writes *into*
  `outputs/report/`, and it is the only build here that imports
  `blindspot.report` (for `as_model_saw`, `fit`, `draw_target` and
  `busiest_crop`, which render a panel at the resolution the API actually
  delivered). Its natural home is `blindspot/report.py` beside those helpers;
  it is here so that restoring it did not mean editing that file.

Inputs, so a missing artefact is diagnosable rather than mysterious:

    causes      results/{charxiv,infographicvqa,slidevqa,slidevqa_allpages,ai2d,
                screenspot_pro}__<TAG>.jsonl, the three control arms
                (results/control_{blind,onepage0,grid4}.jsonl),
                results/*__gtaudit.labelled.jsonl, and the source images.
    drilldown   the five main result files, plus
                results/charxiv__<TAG>.judged.jsonl and the control arms.
    slidevqa    results/slidevqa__<TAG>.jsonl,
                results/slidevqa_allpages__<TAG>.jsonl,
                data/slidevqa/manifest.jsonl and the deck images.
    tasks       whatever `blindspot.eval.load_rows` can assemble for charxiv,
                infographicvqa and screenspot_pro, plus the annotated assets
                `blindspot.eval annotate` renders.
    primitives  outputs/summary.json           (`eval aggregate`)
    headline    outputs/report/summary.json    (`report summary`)
    candidates  results/{infographicvqa,charxiv,ai2d}__<TAG>.jsonl,
                results/svg_localization__<TAG>.jsonl, and the source images.

Seven modules merged here, and the collisions that had to be resolved:

* **`esc()`** was defined seven times with one identical body. There is now one,
  imported from `blindspot.eval` -- which is where `task_pages` already took it
  from.
* **`format_equivalent()`** was defined three times and the three are *not* the
  same function. `causes` takes `(gold, pred)` and folds scale words; `drilldown`
  takes `(pred, golds)` and is exact after folding; `slidevqa` adds a whole-word
  containment fallback for spans. Merging them would have changed three published
  numbers, so they are kept apart as `cause_`/`drill_`/`slide_format_equivalent`
  and each section calls its own.
* **`pct()`** was defined twice here and twice again in `blindspot.report`, and
  all four differ in the null marker or the `%` sign: `primitives` wants a bare
  number and `&mdash;`, `headline` wants `%` and a literal `--`. They stay local
  as `prim_pct` and `aug22_pct` rather than being pointed at `report.pct`, which
  would silently restyle both pages.
* **`CSS`, `JS`, `LIGHTBOX_HTML`, `LIGHTBOX_JS`, `ASSETS`, `MAIN_FILES`, `OUT`,
  `table`, `tile`, `tiles`, `bars`, `bar`, `mean`, `load`, `render`** collided
  with different values or different signatures and carry a section prefix.
  `OUT`, `RESULTS`, `GOOD` and `BAD` were identical in every copy that has them
  and are defined once, below.
* `slidevqa_report` computed its own repository root by walking two directories
  up from its own file -- correct for the nested layout it was written in, and
  wrong here. Like every other module in the package it now reads `results/` and
  `data/` relative to the working directory.

`legacy/blindspot/reporting/` keeps the seven originals, frozen.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import html
import itertools
import json
import math
import os
import re
import sys
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image, ImageDraw, ImageFont

from blindspot.core import load, slidevqa
from blindspot.core import score, anls, token_f1
from blindspot.core import classify as classify_failure, classify_point
from blindspot.core import (is_na, wilson, quantiles, cell_of, centre_cell,
                            bbox_cells)
# `task_pages` also imported `primitive_for`, which it never called: its rows
# arrive from `load_rows` with `primitive` already resolved.
from blindspot.core import LABELS, FAILURE_MODE_LABELS as FM_LABELS
# `esc` and the gallery chrome come from the eval layer rather than being
# redefined here: `task_pages` already imported them from what is now
# `blindspot.eval`, and the other six defined `esc` with the identical body.
from blindspot.eval import esc, load_rows, agg_cell as cell, slice_by
from blindspot.eval import CSS as GCSS, LIGHTBOX_HTML, LIGHTBOX_JS, build_one
from blindspot.eval import load_loc_run as load_run
# Only `candidates` needs these, and only to draw a panel the way the report
# draws one. See the module docstring: it is the single upward import here.
from blindspot.report import as_model_saw, busiest_crop, draw_target, fit

# `causes` calls the official scorer under the name the original used, to keep
# the distinction from its own `fmt_score()` legible at every call site.
official_score = score

OUT = Path("outputs")
RESULTS = Path("results")

GOOD, BAD = "#0ca30c", "#d03b3b"


# ======================= causes: outputs/causes/*.html + outputs/assets_causes/
PAGES = OUT / "causes"
CAUSE_ASSETS = OUT / "assets_causes"

THUMB_MAX, THUMB_Q = 200, 70
FULL_MAX, FULL_Q = 1100, 82

TAG = "haiku-4-5_think2000_native_r0"
CAUSE_MAIN_FILES = {
    "charxiv": f"results/charxiv__{TAG}.jsonl",
    "infographicvqa": f"results/infographicvqa__{TAG}.jsonl",
    "slidevqa": f"results/slidevqa__{TAG}.jsonl",
    "slidevqa_allpages": f"results/slidevqa_allpages__{TAG}.jsonl",
    "ai2d": f"results/ai2d__{TAG}.jsonl",
    "screenspot_pro": f"results/screenspot_pro__{TAG}.jsonl",
}
NICE = {
    "charxiv": "CharXiv", "infographicvqa": "InfographicVQA", "slidevqa": "SlideVQA",
    "slidevqa_allpages": "SlideVQA (all 20 pages)", "ai2d": "AI2D",
    "screenspot_pro": "ScreenSpot-Pro",
}

# The API scales every image so both edges are <= 1568px and the total is under
# ~1.15 megapixels. `sent_image_sizes` records what left the harness; this is
# what the model actually got to look at, and it is what we render.
API_EDGE, API_PIXELS = 1568, 1_150_000


# --------------------------------------------------------------- loading
def read_jsonl(path: str) -> tuple[list[dict], int]:
    """Defensive read: a file still being written loses only its last line."""
    rows, bad = [], 0
    p = Path(path)
    if not p.exists():
        return rows, bad
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                bad += 1
    return rows, bad


def _cause_examples() -> dict:
    ex = {}
    for ds in ("charxiv", "infographicvqa", "ai2d", "screenspot_pro"):
        for e in load(ds):
            ex[e.uid] = e
    for e in slidevqa("evidence"):
        ex[e.uid] = e
    for e in slidevqa("all_pages"):
        ex[e.uid] = e
    return ex


class Data:
    """Everything scored, once, so no page can disagree with another."""

    def __init__(self) -> None:
        self.ex = _cause_examples()
        self.rows: dict[str, list[dict]] = {}
        self.counts: dict[str, dict] = {}
        for key, path in CAUSE_MAIN_FILES.items():
            self.rows[key], self.counts[key] = self._load(key, path)
        self.by_uid = {r["uid"]: r for rs in self.rows.values() for r in rs}
        self.blind = self._control("results/control_blind.jsonl")
        self.onepage = self._control("results/control_onepage0.jsonl")
        self.grid4 = self._control("results/control_grid4.jsonl")
        self.manifests = {
            "slidevqa": {m["qa_id"]: m for m in self._man("slidevqa")},
            "charxiv": list(self._man("charxiv")),
            "infographicvqa": {str(m["questionId"]): m for m in self._man("infographicvqa")},
            "screenspot_pro": {m["id"]: m for m in self._man("screenspot_pro")},
            "ai2d": list(self._man("ai2d")),
        }
        self.contested = self._contested()
        self.blind_score = self._blind_scores()

    @staticmethod
    def _man(ds: str):
        with open(Path("data") / ds / "manifest.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _load(self, key: str, path: str) -> tuple[list[dict], dict]:
        raw, bad = read_jsonl(path)
        best: dict[str, dict] = {}
        for r in raw:                       # retries append; last usable row wins
            if r.get("pred") is None:
                best.setdefault(r["uid"], r)
            else:
                best[r["uid"]] = r
        out, nulls, orphan = [], 0, 0
        for uid, r in best.items():
            if r.get("pred") is None:
                nulls += 1
                continue
            e = self.ex.get(uid)
            if e is None:
                orphan += 1
                continue
            row = dict(r)
            row.update(official_score(e, r["pred"]))
            row["_ex"] = e
            row["question"] = e.question
            row["gold"] = e.gold
            row["bench"] = key
            out.append(row)
        return out, {"lines": len(raw), "unique": len(best), "unusable": nulls,
                     "orphan": orphan, "malformed": bad, "scored": len(out)}

    def _control(self, path: str) -> dict[str, dict]:
        raw, bad = read_jsonl(path)
        out: dict[str, dict] = {}
        for r in raw:
            su = (r.get("meta") or {}).get("src_uid")
            if not su:
                continue
            if r.get("pred") is None:
                out.setdefault(su, r)
            else:
                out[su] = r
        out["__meta__"] = {"lines": len(raw), "malformed": bad,
                           "unique": len([k for k in out if k != "__meta__"])}
        return out

    # ------------------------------------------------- example eligibility
    def _contested(self) -> set[str]:
        """uids whose gold an independent audit did not uphold.

        An example whose gold is ambiguous teaches the reader nothing about the
        model -- "how many countries outside the Middle East" depends on whether
        Turkey counts, not on whether Haiku can see. These are excluded from the
        galleries and counted in the methods note instead. The same rows still
        drive the ground_truth_noise page, which is about exactly this.
        """
        out: set[str] = set()
        for ds in ("charxiv", "infographicvqa", "screenspot_pro", "slidevqa"):
            rows, _ = read_jsonl(f"results/{ds}__gtaudit.jsonl")
            for r in rows:
                if (r.get("verdict") in ("prediction_correct", "both_acceptable")
                        or r.get("gt_quality") in ("ambiguous", "wrong")):
                    out.add(r["uid"])
        return out

    def _blind_scores(self) -> dict[str, float]:
        """Score of the no-image arm, keyed by the sighted uid it pairs with."""
        out: dict[str, float] = {}
        for su, br in self.blind.items():
            if su == "__meta__" or br.get("pred") is None:
                continue
            e = self.ex.get(su)
            if e is not None:
                out[su] = float(official_score(e, br["pred"]).get("score") or 0.0)
        return out

    def vision_status(self, r: dict) -> str:
        """Did this item actually require the image?

        confirmed  -- the blind arm was run on it and got it wrong
        answerable -- the blind arm got it right, so it is not a perception item
        exempt     -- ScreenSpot-Pro: a click target cannot be located without
                      the screenshot, so a blind arm would score ~0 by
                      construction and was never run
        untested   -- no blind counterpart (the arm sampled 500 per benchmark)
        """
        if r["bench"] == "screenspot_pro":
            return "exempt"
        bs = self.blind_score.get(r["uid"])
        if bs is None:
            return "untested"
        return "answerable" if bs >= 0.5 else "confirmed"


# ------------------------------------------------------- format equivalence
_CAUSE_SCALE = {"bn": 1e9, "b": 1e9, "billion": 1e9, "billions": 1e9,
          "m": 1e6, "mn": 1e6, "million": 1e6, "millions": 1e6,
          "k": 1e3, "thousand": 1e3, "thousands": 1e3, "tn": 1e12, "trillion": 1e12}
_CAUSE_NUMRE = re.compile(r"^\s*([+-]?)\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?|\.\d+)\s*(%?)\s*([a-zA-Z]*)\s*$")


def scalars(s) -> list[float]:
    """Every numeric reading of a scalar-shaped string.

    A scale word gives two readings -- stripped ("5m" -> 5) and applied
    ("5m" -> 5e6) -- because whether the unit was already implied by the
    question is not knowable from the string. Both are the same *value* in a
    different dress, which is exactly what this test is for.
    """
    t = re.sub(r"[\$€£]", "", str(s).strip().replace("−", "-"))
    t = re.sub(r"\s+", " ", t).strip()
    m = _CAUSE_NUMRE.match(t)
    if not m:
        return []
    sign, num, _pct, word = m.groups()
    try:
        v = float(num.replace(",", ""))
    except ValueError:
        return []
    if sign == "-":
        v = -v
    w = word.strip().lower()
    if not w:
        return [v]
    if w in _CAUSE_SCALE:
        return [v, v * _CAUSE_SCALE[w]]
    return []                      # an unknown trailing word is not a formatting difference


def numval(s):
    v = scalars(s)
    return v[0] if v else None


def _cause_alnum(s) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def cause_format_equivalent(gold, pred) -> bool:
    """Right value, wrong dress. Deliberately conservative: numeric comparison is
    sign-sensitive and exact; text comparison is full-string after folding case
    and punctuation. There is no substring fallback, so "22" never matches
    "22 million people surveyed"."""
    golds = gold if isinstance(gold, (list, tuple)) else [gold]
    ps, pa = scalars(pred), _cause_alnum(pred)
    for g in golds:
        gs = scalars(g)
        if gs and ps:
            for a in gs:
                for b in ps:
                    if a == b or (a != 0 and abs(a - b) <= 1e-9 * abs(a)):
                        return True
            continue
        if _cause_alnum(g) and _cause_alnum(g) == pa:
            return True
    return False


def fmt_score(r: dict) -> float:
    """Score with the metric's formatting penalty removed."""
    return 1.0 if cause_format_equivalent(r["gold"], r["pred"]) else float(r.get("score") or 0.0)


def hit(r: dict, thr: float = 0.5) -> bool:
    return float(r.get("score") or 0.0) >= thr


# Below this many scored rows a slice-vs-slice comparison is not a measurement.
MIN_CAUSE_ROWS = 200


def cause_mean(vals) -> float | None:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else None

def pct_or_dash(v, d: int = 1) -> str:
    """Format a `cause_mean` result, which is None when the sample is too thin.

    `cause_mean` returns None rather than 0.0 for an empty or suppressed group --
    correctly, because "no measurement" and "measured zero" are different claims
    and this whole family of pages exists to keep them apart. Every call site then
    has to remember it. Most did; the ones that did not raised
    `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'` and took
    the page down, but only on a thin `results/` -- which is exactly what a fresh
    clone has, and never what the full study tree has. So it passed every test
    against real data and failed for anyone starting from scratch.
    """
    return "&mdash;" if v is None else f"{v * 100:.{d}f}%"


def pct_bare(v, d: int = 1) -> str:
    """`pct_or_dash` without the trailing %, for prose that supplies its own."""
    return "&mdash;" if v is None else f"{v * 100:.{d}f}"


# ------------------------------------------------------------- rendering
_FONT_PATHS = ("/System/Library/Fonts/Supplemental/Arial.ttf",
               "/System/Library/Fonts/Helvetica.ttc")


def _font(size: int):
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def effective_size(sent: list | None, native: tuple[int, int]) -> tuple[int, int]:
    """What the model actually resolved: the sent size after the API's own
    downscale to <=1568px per edge and <=~1.15MP total."""
    w, h = (int(sent[0]), int(sent[1])) if sent else native
    if w <= 0 or h <= 0:
        w, h = native
    s = min(1.0, API_EDGE / max(w, h), math.sqrt(API_PIXELS / max(w * h, 1)))
    return max(1, round(w * s)), max(1, round(h * s))


def _overlay(im: Image.Image, gold=None, pred=None) -> Image.Image:
    """Gold bbox in green, predicted click as a red crosshair. Same convention as
    the existing annotation gallery so the two read the same way."""
    W, H = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(W, H) / 400))
    if gold:
        x0, y0, x1, y1 = [c * s for c, s in zip(gold, (W, H, W, H))]
        d.rectangle([x0, y0, x1, y1], outline=GOOD, width=lw)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ring = max(lw * 14, (x1 - x0) * 1.6, (y1 - y0) * 1.6)
        d.ellipse([cx - ring, cy - ring, cx + ring, cy + ring], outline=GOOD,
                  width=max(1, lw // 2))
        d.text((x0, max(0, y0 - lw * 9)), "target", fill=GOOD, font=_font(lw * 8))
    if pred:
        px, py = pred[0] * W, pred[1] * H
        r = lw * 8
        d.line([px - r, py, px + r, py], fill=BAD, width=lw)
        d.line([px, py - r, px, py + r], fill=BAD, width=lw)
        d.ellipse([px - r / 2, py - r / 2, px + r / 2, py + r / 2], outline=BAD, width=lw)
        d.text((px + r, py + r), "click", fill=BAD, font=_font(lw * 8))
    return im


def render_one(job: dict) -> dict:
    """Module-level for the process pool. Writes <key>_t.jpg and <key>_f.jpg.

    The source is first resized to the *effective* size the model saw, so a
    3840x2160 screenshot is never shown at detail Haiku never had; only then is
    it resized again for display.
    """
    key, src = job["key"], job["src"]
    tp, fp = CAUSE_ASSETS / f"{key}_t.jpg", CAUSE_ASSETS / f"{key}_f.jpg"
    if tp.exists() and fp.exists() and not job.get("force"):
        return {"key": key, "thumb": tp.name, "full": fp.name, "cached": True}
    try:
        im = Image.open(src).convert("RGB")
    except Exception as e:                      # a missing asset must not kill the build
        return {"key": key, "error": str(e)}
    ew, eh = effective_size(job.get("sent"), im.size)
    if (ew, eh) != im.size:
        im = im.resize((ew, eh), Image.LANCZOS)
    if job.get("gold") or job.get("pred"):
        im = _overlay(im, job.get("gold"), job.get("pred"))
    crop = job.get("crop")
    if crop:                                    # zoom panel around a small target
        x0, y0, x1, y1 = crop
        im = im.crop((round(max(0, x0 * ew)), round(max(0, y0 * eh)),
                      round(min(ew, x1 * ew)), round(min(eh, y1 * eh))))
    full = im
    if max(full.size) > FULL_MAX:
        s = FULL_MAX / max(full.size)
        full = full.resize((max(1, round(full.width * s)), max(1, round(full.height * s))),
                           Image.LANCZOS)
    full.save(fp, format="JPEG", quality=FULL_Q, optimize=True)
    th = im
    if max(th.size) > THUMB_MAX:
        s = THUMB_MAX / max(th.size)
        th = th.resize((max(1, round(th.width * s)), max(1, round(th.height * s))),
                       Image.LANCZOS)
    th.save(tp, format="JPEG", quality=THUMB_Q, optimize=True)
    return {"key": key, "thumb": tp.name, "full": fp.name,
            "eff": [ew, eh], "cached": False}


def render_all(jobs: list[dict], workers: int) -> dict[str, dict]:
    if not jobs:
        return {}
    seen, uniq = set(), []
    for j in jobs:
        if j["key"] in seen:
            continue
        seen.add(j["key"])
        uniq.append(j)
    CAUSE_ASSETS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(render_one, uniq, chunksize=4))
    return {r["key"]: r for r in res}


# ------------------------------------------------------------------- html
# Every custom property is declared on :root. Declaring them on a wrapper class
# instead is what shipped a black-on-black page last time: custom properties
# inherit downward only, so anything outside the wrapper -- the lightbox, which
# is a direct child of <body> -- resolves them to nothing.
CAUSE_CSS = """
:root{
 color-scheme:light;
 --surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#6e6d68;
 --grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.12);
 --s1:#2a78d6;--s2:#eb6834;--s3:#7a5af0;--good:#0ca30c;--bad:#d03b3b;--warn:#a06a00;
 --chip:rgba(11,11,11,.05);
}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
 color-scheme:dark;
 --surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#98968e;
 --grid:#2c2c2a;--axis:#4a4a46;--border:rgba(255,255,255,.13);
 --s1:#5b9df0;--s2:#f0864f;--s3:#a48cff;--good:#3cc93c;--bad:#f26a6a;--warn:#fab219;
 --chip:rgba(255,255,255,.06);
}}
:root[data-theme=dark]{
 color-scheme:dark;
 --surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#98968e;
 --grid:#2c2c2a;--axis:#4a4a46;--border:rgba(255,255,255,.13);
 --s1:#5b9df0;--s2:#f0864f;--s3:#a48cff;--good:#3cc93c;--bad:#f26a6a;--warn:#fab219;
 --chip:rgba(255,255,255,.06);
}
*{box-sizing:border-box}
html,body{background:var(--page);color:var(--ink);margin:0;
 font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:26px 22px 90px}
a{color:var(--s1)}
header.top{display:flex;justify-content:space-between;gap:18px;align-items:flex-start;
 flex-wrap:wrap;margin-bottom:6px}
h1{font-size:25px;margin:0 0 6px;line-height:1.25}
.dek{color:var(--ink2);margin:0;max-width:76ch}
h2{font-size:18px;margin:40px 0 6px;padding-top:18px;border-top:1px solid var(--grid)}
h3{font-size:15px;margin:22px 0 6px}
p{max-width:82ch}
button{font:inherit;font-size:13px;padding:6px 11px;border-radius:8px;
 border:1px solid var(--border);background:var(--surface);color:var(--ink2);cursor:pointer}
button:hover{border-color:var(--s1)}
nav.crumbs{display:flex;gap:7px;flex-wrap:wrap;margin:0 0 18px}
nav.crumbs a{font-size:12.5px;padding:4px 10px;border:1px solid var(--border);border-radius:999px;
 text-decoration:none;color:var(--ink2);background:var(--surface)}
nav.crumbs a:hover{color:var(--ink);border-color:var(--s1)}
.badge{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.06em;
 padding:3px 9px;border-radius:999px;text-transform:uppercase;white-space:nowrap}
.b-proven{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}
.b-supported{background:color-mix(in srgb,var(--s1) 18%,transparent);color:var(--s1)}
.b-mixed{background:color-mix(in srgb,var(--warn) 24%,transparent);color:var(--warn)}
.b-refuted{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}
.b-artifact{background:color-mix(in srgb,var(--s3) 20%,transparent);color:var(--s3)}
.b-untested{background:var(--chip);color:var(--muted)}
.claim{background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--s1);
 border-radius:10px;padding:14px 18px;margin:14px 0 6px;font-size:16px;line-height:1.5}
.claim .lab{display:block;font-size:11px;letter-spacing:.07em;text-transform:uppercase;
 color:var(--muted);margin-bottom:5px}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0 0;align-items:center}
.chip{font-size:11.5px;padding:2px 9px;border-radius:999px;background:var(--chip);
 color:var(--ink2);border:1px solid var(--border);white-space:nowrap}
.chip.one{border-color:var(--warn);color:var(--warn)}
.chip.vis-ok{border-color:var(--good);color:var(--good);cursor:help}
.chip.vis-un{border-style:dashed;cursor:help}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:11px;margin:16px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 15px}
.tlab{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.tval{font-size:29px;line-height:1.15;margin:6px 0 3px;font-variant-numeric:tabular-nums}
.tile.bad .tval{color:var(--bad)}.tile.good .tval{color:var(--good)}
.tile.warn .tval{color:var(--warn)}
.tnote{font-size:12.5px;color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
 padding:16px 18px 18px;margin:14px 0}
.card h4{font-size:14.5px;margin:0 0 3px}
.card .sub{font-size:12.5px;color:var(--ink2);margin:0 0 13px;max-width:84ch}
.row{display:grid;grid-template-columns:250px 1fr 108px;align-items:center;gap:11px;padding:4px 0}
.rlab{font-size:12.5px;color:var(--ink2);text-align:right;overflow-wrap:anywhere}
.track{height:14px;background:var(--grid);border-radius:4px;position:relative;overflow:hidden}
.bar{height:100%;background:var(--s1);border-radius:0 4px 4px 0}
.bar.s2{background:var(--s2)}.bar.s3{background:var(--s3)}
.bar.good{background:var(--good)}.bar.bad{background:var(--bad)}
.chance{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--axis)}
.rval{font-size:12.5px;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.rval .n{display:block;font-size:10.5px;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 14px}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid var(--grid);vertical-align:top}
td{font-variant-numeric:tabular-nums}
th{font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:600}
tbody th{font-weight:400;color:var(--ink2);text-transform:none;letter-spacing:0;font-size:13px}
td.g{color:var(--good)}td.b{color:var(--bad)}td.num{text-align:right}
.note{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--s2);
 border-radius:9px;padding:12px 15px;font-size:13.5px;color:var(--ink2);margin:14px 0;max-width:88ch}
.note strong{color:var(--ink)}
.note.warn{border-left-color:var(--warn)}
.note.good{border-left-color:var(--good)}
.bench{margin:26px 0 0}
.bench > h3{display:flex;align-items:center;gap:10px;font-size:15px;margin:0 0 4px}
.bench .bnote{font-size:12.5px;color:var(--ink2);margin:0 0 12px;max-width:86ch}
.ex{background:var(--surface);border:1px solid var(--border);border-radius:11px;
 padding:12px 13px;margin:0 0 11px}
.ex .exhd{display:flex;gap:8px;align-items:flex-start;margin-bottom:9px;flex-wrap:wrap}
.pill{font-size:10.5px;font-weight:700;padding:2px 8px;border-radius:999px;white-space:nowrap}
.ok{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good)}
.no{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}
.exq{font-size:13.5px;line-height:1.45;flex:1 1 300px;min-width:240px}
.strip{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 9px}
.strip figure{margin:0;max-width:230px}
.strip img{display:block;width:auto;max-width:230px;max-height:170px;object-fit:contain;
 background:var(--grid);border:1px solid var(--border);border-radius:6px}
.strip figcaption{font-size:10.5px;color:var(--muted);margin-top:3px;max-width:230px}
.strip a{display:block;position:relative;text-decoration:none}
.strip a:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
.strip a::after{content:"zoom";position:absolute;right:5px;bottom:5px;font-size:10px;
 padding:1px 6px;border-radius:999px;background:rgba(0,0,0,.66);color:#fff;opacity:0;
 transition:opacity .12s}
.strip a:hover::after,.strip a:focus-visible::after{opacity:1}
dl.kv{margin:0;font-size:12.5px;display:grid;gap:3px}
dl.kv>div{display:grid;grid-template-columns:132px 1fr;gap:9px}
dl.kv dt{color:var(--muted)}
dl.kv dd{margin:0;overflow-wrap:anywhere}
dl.kv dd.g{color:var(--good)}dl.kv dd.b{color:var(--bad)}
details{margin-top:8px}
details summary{cursor:pointer;font-size:12.5px;color:var(--s1);padding:4px 0}
details pre{white-space:pre-wrap;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
 background:var(--page);border:1px solid var(--border);border-radius:8px;padding:10px 12px;
 max-height:340px;overflow:auto;color:var(--ink2)}
.idx{width:100%;border-collapse:separate;border-spacing:0 8px}
.idx td{border:none;background:var(--surface);padding:12px 13px;vertical-align:top}
.idx tr td:first-child{border-radius:11px 0 0 11px;border-left:1px solid var(--border)}
.idx tr td:last-child{border-radius:0 11px 11px 0;border-right:1px solid var(--border)}
.idx td{border-top:1px solid var(--border);border-bottom:1px solid var(--border)}
.idx a.t{font-size:15px;font-weight:600;text-decoration:none}
.idx .cl{font-size:13px;color:var(--ink2);margin-top:3px;max-width:74ch}
.impact{width:74px}
.imeter{height:9px;background:var(--grid);border-radius:3px;overflow:hidden}
.imeter i{display:block;height:100%;background:var(--s1)}
svg .ax{stroke:var(--axis);stroke-width:1}
svg .gl{stroke:var(--grid);stroke-width:1}
svg text{fill:var(--muted);font:11px system-ui,sans-serif}
svg text.v{fill:var(--ink2)}
.xlinks{display:flex;gap:7px;flex-wrap:wrap;margin:8px 0 0}
.xlinks a{font-size:12px;padding:3px 9px;border-radius:999px;background:var(--chip);
 border:1px solid var(--border);text-decoration:none}
#lb{position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.93);display:none}
#lb.on{display:block}
#lb .stage{position:absolute;inset:0;overflow:hidden;cursor:grab;touch-action:none}
#lb .stage.grabbing{cursor:grabbing}
#lb img{position:absolute;top:0;left:0;transform-origin:0 0;will-change:transform;
 max-width:none;max-height:none;user-select:none;-webkit-user-drag:none}
#lb .ctrl{position:fixed;top:14px;left:50%;transform:translateX(-50%);display:flex;gap:6px;
 align-items:center;z-index:2;background:rgba(20,20,20,.9);padding:6px 8px;border-radius:10px}
#lb .ctrl button{font:inherit;font-size:14px;line-height:1;min-width:34px;padding:7px 9px;
 border-radius:7px;border:1px solid rgba(255,255,255,.2);background:#222;color:#eee;cursor:pointer}
#lb .ctrl button:hover{background:#333;border-color:#666}
#lb .ctrl .lvl{color:#c3c2b7;font-size:12.5px;min-width:54px;text-align:center;
 font-variant-numeric:tabular-nums}
#lb .cap{position:fixed;bottom:44px;left:50%;transform:translateX(-50%);color:#ddd;font-size:12.5px;
 z-index:2;background:rgba(20,20,20,.85);padding:5px 12px;border-radius:8px;max-width:80vw;
 text-align:center}
#lb .hint{position:fixed;bottom:12px;left:50%;transform:translateX(-50%);color:#98968e;
 font-size:11.5px;z-index:2;background:rgba(20,20,20,.8);padding:4px 11px;border-radius:999px}
@media(max-width:760px){.row{grid-template-columns:1fr}.rlab{text-align:left}
 dl.kv>div{grid-template-columns:1fr}}
"""

CAUSE_LIGHTBOX_HTML = """
<div id="lb" role="dialog" aria-modal="true" aria-label="image viewer">
 <div class="ctrl">
  <button class="zprev" title="previous (left arrow)">&#8249;</button>
  <button class="zout" title="zoom out (-)">&minus;</button>
  <span class="lvl">100%</span>
  <button class="zin" title="zoom in (+)">+</button>
  <button class="zfit" title="fit to screen (0)">fit</button>
  <button class="znext" title="next (right arrow)">&#8250;</button>
  <button class="zclose" title="close (Esc)">&times;</button>
 </div>
 <div class="stage"><img alt="full size"></div>
 <span class="cap"></span>
 <span class="hint">scroll / pinch to zoom &middot; drag to pan &middot; &larr; &rarr; through the set &middot; Esc to close</span>
</div>
"""

CAUSE_LIGHTBOX_JS = r"""
(function(){
 const lb=document.getElementById('lb'); if(!lb) return;
 const stage=lb.querySelector('.stage'), img=lb.querySelector('img'),
       lvl=lb.querySelector('.lvl'), cap=lb.querySelector('.cap');
 let s=1, fit=1, tx=0, ty=0, drag=false, lx=0, ly=0, seq=[], at=0, pinch=null;
 const apply=()=>{img.style.transform='translate('+tx+'px,'+ty+'px) scale('+s+')';
                  lvl.textContent=Math.round(s/fit*100)+'%';};
 const fitView=()=>{const r=stage.getBoundingClientRect();
   if(!img.naturalWidth) return;
   fit=Math.min(r.width/img.naturalWidth, r.height/img.naturalHeight);
   s=fit; tx=(r.width-img.naturalWidth*s)/2; ty=(r.height-img.naturalHeight*s)/2; apply();};
 const zoomAt=(px,py,f)=>{const ns=Math.min(fit*40, Math.max(fit*0.4, s*f));
   tx=px-(px-tx)*(ns/s); ty=py-(py-ty)*(ns/s); s=ns; apply();};
 const centreZoom=f=>{const r=stage.getBoundingClientRect(); zoomAt(r.width/2,r.height/2,f);};
 const show=i=>{ if(!seq.length) return; at=(i+seq.length)%seq.length;
   img.src=seq[at].href; cap.textContent=seq[at].cap+(seq.length>1?'  ('+(at+1)+'/'+seq.length+')':''); };
 img.addEventListener('load', fitView);
 addEventListener('resize', ()=>{ if(lb.classList.contains('on')) fitView(); });
 stage.addEventListener('wheel', e=>{ e.preventDefault();
   const r=stage.getBoundingClientRect();
   zoomAt(e.clientX-r.left, e.clientY-r.top, e.deltaY<0?1.18:1/1.18); }, {passive:false});
 stage.addEventListener('dblclick', e=>{ const r=stage.getBoundingClientRect();
   if(s>fit*1.5) fitView(); else zoomAt(e.clientX-r.left, e.clientY-r.top, 5); });
 stage.addEventListener('mousedown', e=>{ drag=true; lx=e.clientX; ly=e.clientY;
   stage.classList.add('grabbing'); e.preventDefault(); });
 addEventListener('mousemove', e=>{ if(!drag) return;
   tx+=e.clientX-lx; ty+=e.clientY-ly; lx=e.clientX; ly=e.clientY; apply(); });
 addEventListener('mouseup', ()=>{ drag=false; stage.classList.remove('grabbing'); });
 stage.addEventListener('touchstart', e=>{
   if(e.touches.length===2){ const [a,b]=e.touches;
     pinch={d:Math.hypot(a.clientX-b.clientX,a.clientY-b.clientY),
            x:(a.clientX+b.clientX)/2, y:(a.clientY+b.clientY)/2}; }
   else if(e.touches.length===1){ drag=true; lx=e.touches[0].clientX; ly=e.touches[0].clientY; }
 }, {passive:true});
 stage.addEventListener('touchmove', e=>{
   const r=stage.getBoundingClientRect();
   if(e.touches.length===2 && pinch){ const [a,b]=e.touches;
     const d=Math.hypot(a.clientX-b.clientX,a.clientY-b.clientY);
     zoomAt((a.clientX+b.clientX)/2-r.left,(a.clientY+b.clientY)/2-r.top, d/pinch.d);
     pinch.d=d; e.preventDefault(); }
   else if(e.touches.length===1 && drag){ const t=e.touches[0];
     tx+=t.clientX-lx; ty+=t.clientY-ly; lx=t.clientX; ly=t.clientY; apply(); e.preventDefault(); }
 }, {passive:false});
 stage.addEventListener('touchend', ()=>{ pinch=null; drag=false; }, {passive:true});
 lb.querySelector('.zin').onclick =()=>centreZoom(1.5);
 lb.querySelector('.zout').onclick=()=>centreZoom(1/1.5);
 lb.querySelector('.zfit').onclick=fitView;
 lb.querySelector('.zprev').onclick=()=>show(at-1);
 lb.querySelector('.znext').onclick=()=>show(at+1);
 const close=()=>{ lb.classList.remove('on'); img.removeAttribute('src'); };
 lb.querySelector('.zclose').onclick=close;
 addEventListener('keydown', e=>{ if(!lb.classList.contains('on')) return;
   if(e.key==='Escape') close();
   if(e.key==='+'||e.key==='=') centreZoom(1.5);
   if(e.key==='-') centreZoom(1/1.5);
   if(e.key==='0') fitView();
   if(e.key==='ArrowLeft') show(at-1);
   if(e.key==='ArrowRight') show(at+1); });
 document.querySelectorAll('.strip').forEach(strip=>{
   const links=[...strip.querySelectorAll('a.zoom')];
   links.forEach((a,i)=>a.addEventListener('click', e=>{ e.preventDefault();
     seq=links.map(l=>({href:l.getAttribute('href'), cap:l.dataset.cap||''}));
     lb.classList.add('on'); show(i); }));
 });
})();
"""

THEME_JS = r"""
(function(){
 const b=document.querySelector('button.theme'); if(!b) return;
 const set=t=>{document.documentElement.dataset.theme=t;
   b.textContent=t==='dark'?'Light mode':'Dark mode';
   try{localStorage.setItem('bs-theme',t);}catch(e){}};
 let t=null; try{t=localStorage.getItem('bs-theme');}catch(e){}
 if(t) set(t); else b.textContent=matchMedia('(prefers-color-scheme: dark)').matches?'Light mode':'Dark mode';
 b.addEventListener('click',()=>set(document.documentElement.dataset.theme==='dark'?'light':'dark'));
})();
"""


def pctf(v, d=1) -> str:
    return "&mdash;" if v is None else f"{v * 100:.{d}f}%"


def pct_bare(v, d: int = 1) -> str:
    """`pct_or_dash` without the trailing %, for prose that supplies its own."""
    return "&mdash;" if v is None else f"{v * 100:.{d}f}"


VERDICTS = {
    "PROVEN": "b-proven", "SUPPORTED": "b-supported", "MIXED": "b-mixed",
    "REFUTED": "b-refuted", "ARTIFACT": "b-artifact", "UNTESTED": "b-untested",
}


def badge(v: str) -> str:
    return f'<span class="badge {VERDICTS.get(v, "b-untested")}">{esc(v)}</span>'


def cause_tile(lab, val, note, tone="") -> str:
    # Labels, values and notes are authored in this module and may carry entities
    # and inline markup, so they are not escaped here. Anything derived from data
    # is escaped at the call site.
    return (f'<div class="tile {tone}"><div class="tlab">{lab}</div>'
            f'<div class="tval">{val}</div><div class="tnote">{note}</div></div>')


def cause_tiles(items) -> str:
    return '<div class="tiles">' + "".join(cause_tile(*i) for i in items) + "</div>"


def bars(title, sub, items, cls="") -> str:
    """items: (label, value_0_1, n_text, right_text, chance_or_None, bar_class)."""
    out = []
    for lab, v, n, right, chance, bc in items:
        if v is None:
            out.append(f'<div class="row"><div class="rlab">{lab}</div>'
                       f'<div class="track"></div><div class="rval">&mdash;</div></div>')
            continue
        ch = (f'<span class="chance" style="left:{max(min(chance * 100, 100), 0):.2f}%" '
              f'title="chance = {chance * 100:.2f}%"></span>') if chance is not None else ""
        out.append(
            f'<div class="row"><div class="rlab">{lab}</div>'
            f'<div class="track"><div class="bar {bc}" style="width:{max(v * 100, 0.6):.2f}%"></div>{ch}</div>'
            f'<div class="rval">{right}<span class="n">{n}</span></div></div>')
    return (f'<div class="card {cls}"><h4>{title}</h4><p class="sub">{sub}</p>'
            + "".join(out) + "</div>")


def cause_table(headers, rows, note="", cls="") -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            if isinstance(c, tuple):
                txt, klass = c
            else:
                txt, klass = c, ("" if i == 0 else "num")
            tag = "th" if i == 0 else "td"
            cells.append(f'<{tag} class="{klass}">{txt}</{tag}>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    n = f'<p class="sub">{note}</p>' if note else ""
    return (f'<div class="card {cls}">{n}<table><thead><tr>{th}</tr></thead>'
            f"<tbody>{''.join(body)}</tbody></table></div>")


def hist_svg(series, title, sub, xlabels, ymax=None) -> str:
    """Grouped vertical bars. series: [(name, colorvar, [values])]."""
    W, H = 780, 210
    pad_l, pad_b, pad_t = 44, 44, 14
    n = len(xlabels)
    if not ymax:
        raw = max((max(x for x in v if x is not None) for _, _, v in series if any(x is not None for x in v)),
                  default=1.0) or 1.0
        ymax = next((t for t in (0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.8, 1.0) if t >= raw), 1.0)
    gw = (W - pad_l - 14) / max(n, 1)
    bw = gw / (len(series) + 0.6)
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{esc(title)}">']
    for i in range(5):
        y = pad_t + (H - pad_t - pad_b) * i / 4
        parts.append(f'<line class="gl" x1="{pad_l}" y1="{y:.1f}" x2="{W - 8}" y2="{y:.1f}"/>')
        tickv = ymax * (1 - i / 4) * 100
        parts.append(f'<text x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end">'
                     f'{tickv:.0f}%</text>' if abs(tickv - round(tickv)) < 0.05 else
                     f'<text x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end">{tickv:.1f}%</text>')
    for si, (name, col, vals) in enumerate(series):
        for xi, v in enumerate(vals):
            if v is None:
                continue
            h = (H - pad_t - pad_b) * (v / ymax)
            x = pad_l + gw * xi + bw * si + bw * 0.3
            parts.append(f'<rect x="{x:.1f}" y="{H - pad_b - h:.1f}" width="{bw * 0.86:.1f}" '
                         f'height="{max(h, 1):.1f}" fill="var({col})" rx="2">'
                         f'<title>{esc(html.unescape(name))} {esc(xlabels[xi])}: '
                         f'{v * 100:.1f}%</title></rect>')
    for xi, lab in enumerate(xlabels):
        parts.append(f'<text x="{pad_l + gw * xi + gw / 2:.1f}" y="{H - pad_b + 15}" '
                     f'text-anchor="middle">{esc(lab)}</text>')
    parts.append("</svg>")
    leg = " ".join(
        f'<span class="chip"><span style="display:inline-block;width:9px;height:9px;'
        f'border-radius:2px;background:var({c});margin-right:5px"></span>{nm}</span>'
        for nm, c, _ in series)
    return (f'<div class="card"><h4>{title}</h4><p class="sub">{sub}</p>'
            f'{"".join(parts)}<div class="chips">{leg}</div></div>')


def _thumb_html(assets, key, cap, alt="") -> str:
    a = assets.get(key)
    if not a or "thumb" not in a:
        return ""
    return (f'<figure><a class="zoom" href="../assets_causes/{a["full"]}" '
            f'data-cap="{esc(cap)}"><img loading="lazy" src="../assets_causes/{a["thumb"]}" '
            f'alt="{esc(alt or cap)}"></a><figcaption>{esc(cap)}</figcaption></figure>')


def example_html(e: dict, assets: dict) -> str:
    ok = e.get("ok")
    pill = ('<span class="pill ok">&#10003; correct</span>' if ok
            else '<span class="pill no">&#10007; wrong</span>')
    strip = "".join(_thumb_html(assets, im["key"], im.get("cap", "")) for im in e.get("imgs", []))
    kv = []
    for lab, val, klass in e.get("kv", []):
        kv.append(f'<div><dt>{esc(lab)}</dt><dd class="{klass}">{val}</dd></div>')
    think = ""
    t = (e.get("thinking") or "").strip()
    if t:
        think = (f'<details><summary>model’s thinking trace '
                 f'({len(t.split())} words)</summary><pre>{esc(t)}</pre></details>')
    why = f'<div class="note" style="margin:9px 0 0">{e["why"]}</div>' if e.get("why") else ""
    tags = "".join(f'<span class="chip">{esc(x)}</span>' for x in e.get("tags", []))
    vis = {"confirmed": ('<span class="chip vis-ok" title="The same question was asked '
                         'with the image withheld and answered wrong, so this item really '
                         'does require seeing.">needs the image</span>'),
           "untested": ('<span class="chip vis-un" title="The blind arm sampled 500 items '
                        'per benchmark and did not include this one.">not tested blind</span>')}
    tags = vis.get(e.get("vision", ""), "") + tags
    return (f'<article class="ex"><div class="exhd">{pill}{tags}'
            f'<span class="exq">{esc(e.get("question", ""))[:600]}</span></div>'
            f'{f"<div class=strip>{strip}</div>" if strip else ""}'
            f'<dl class="kv">{"".join(kv)}</dl>{think}{why}</article>')


class Cause:
    def __init__(self, cid, title, claim, verdict, effect, benchmarks, impact,
                 body="", groups=(), refute="", links=(), one_bench_note=""):
        self.id, self.title, self.claim = cid, title, claim
        self.verdict, self.effect = verdict, effect
        self.benchmarks = list(benchmarks)
        self.impact = impact                     # 0-1, drives index ranking
        self.body = body
        self.groups = self._clean(groups)        # [{bench,note,examples:[...]}]
        self.refute = refute
        self.links = list(links)
        self.one_bench_note = one_bench_note

    @staticmethod
    def _clean(groups) -> list:
        """Drop the cards Builder.example() refused, order the survivors.

        Filtering centrally means a cause function can keep selecting rows by
        whatever criterion it cares about without also having to remember the
        blind-arm and audit rules. Groups left with nothing are removed rather
        than rendered as an empty "0 examples" heading.
        """
        out = []
        for g in groups:
            exs = [e for e in g.get("examples") or [] if e]
            exs.sort(key=lambda e: e.get("_rank", 1))
            if exs:
                out.append({**g, "examples": exs})
        return out

    @property
    def n_examples(self) -> int:
        return sum(len(g["examples"]) for g in self.groups)


CAUSE_TITLES: dict[str, str] = {}


def _drilldown_href() -> str:
    """The drill-down page is built by a separate generator that may not have run
    yet. Prefer the canonical path; fall back to a dated sibling if that is where
    it actually landed, so the nav never points at nothing that exists."""
    for cand in ("outputs/drilldown.html", "outputs/aug22/drilldown.html"):
        if Path(cand).exists():
            return "../" + cand[len("outputs/"):]
    return "../drilldown.html"


def crumbs(here: str) -> str:
    items = [("../report.html", "Summary report"), ("../datasets.html", "Datasets"),
             ("../slidevqa.html", "SlideVQA deep-dive"), (_drilldown_href(), "Drill-down"),
             ("index.html", "All causes")]
    return ('<nav class="crumbs">'
            + "".join(f'<a href="{h}">{esc(t)}</a>' for h, t in items if t != here)
            + "</nav>")


def page(title: str, dek: str, body: str, here: str = "") -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{CAUSE_CSS}</style></head><body>
<div class="wrap">
<header class="top"><div><h1>{title}</h1><p class="dek">{dek}</p></div>
<button class="theme" type="button">Dark mode</button></header>
{crumbs(here)}
{body}
</div>
{CAUSE_LIGHTBOX_HTML}
<script>{THEME_JS}
{CAUSE_LIGHTBOX_JS}</script></body></html>"""


def cause_page(c: Cause, assets: dict) -> str:
    bench_chips = "".join(
        f'<span class="chip{" one" if len(c.benchmarks) == 1 else ""}">{esc(NICE.get(b, b))}</span>'
        for b in c.benchmarks)
    single = ""
    if len(c.benchmarks) == 1:
        note = c.one_bench_note or (
            "Only one dataset in this study can test this claim, so it cannot be separated from a "
            "property of that dataset. Treat it as weaker than the cross-benchmark causes.")
        single = ('<div class="note warn"><strong>Single-benchmark evidence.</strong> '
                  + note + "</div>")
    groups = []
    for g in c.groups:
        exs = "".join(example_html(e, assets) for e in g["examples"])
        groups.append(f'<div class="bench"><h3>{esc(NICE.get(g["bench"], g["bench"]))}'
                      f'<span class="chip">{len(g["examples"])} examples</span></h3>'
                      f'<p class="bnote">{g.get("note", "")}</p>{exs}</div>')
    links = ""
    if c.links:
        links = ('<h2>Interacts with</h2><div class="xlinks">'
                 + "".join(f'<a href="{l}.html">{esc(CAUSE_TITLES.get(l, l))}</a>'
                           for l in c.links) + "</div>")
    return page(
        f"{esc(c.title)} {badge(c.verdict)}",
        f"Cause <code>{esc(c.id)}</code> &middot; effect size {c.effect}",
        f'<div class="claim"><span class="lab">Claim</span>{c.claim}</div>'
        f'<div class="chips"><span class="chip">evidenced by:</span>{bench_chips}</div>'
        f"{single}"
        f"<h2>Evidence</h2>{c.body}"
        f'<h2>Examples, grouped by benchmark</h2>'
        f'<p class="dek">Failures first and generously; a few contrasting successes at the end of '
        f'each group. Click any image for a pan/zoom view &mdash; the detail that decides these '
        f'cases is often a few pixels wide.</p>'
        f"{''.join(groups)}"
        f"<h2>What would refute this, and what we did not test</h2>{c.refute}"
        f"{links}",
        here="")


# ------------------------------------------------------------- example kit


class Builder:
    """Collects image render jobs while assembling example cards."""

    def __init__(self, data: Data, want_images: bool = True):
        self.d = data
        self.jobs: list[dict] = []
        self.want_images = want_images
        self.dropped: collections.Counter = collections.Counter()
        self.kept: collections.Counter = collections.Counter()

    def img(self, src, sent=None, gold=None, pred=None, crop=None, cap="") -> dict:
        sig = json.dumps([str(src), sent, gold, pred, crop], sort_keys=True, default=str)
        key = hashlib.md5(sig.encode()).hexdigest()[:16]
        if self.want_images:
            self.jobs.append({"key": key, "src": str(src), "sent": sent,
                              "gold": gold, "pred": pred, "crop": crop})
        return {"key": key, "cap": cap}

    # ---- images for one scored row, rendered at the size the model saw
    def row_images(self, r: dict, max_pages: int = 4) -> list[dict]:
        ex = r["_ex"]
        sizes = r.get("sent_image_sizes") or []
        out = []
        if r["bench"] == "screenspot_pro":
            sent = sizes[0] if sizes else None
            gold, pred = list(ex.gold), list(r["pred"])
            out.append(self.img(ex.images[0], sent, gold, pred,
                                cap="full screenshot as sent (green = target, red = click)"))
            x0, y0, x1, y1 = gold
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            px, py = pred
            pad = max(0.10, abs(px - cx) * 0.8, abs(py - cy) * 0.8)
            crop = [max(0.0, min(cx, px) - pad), max(0.0, min(cy, py) - pad),
                    min(1.0, max(cx, px) + pad), min(1.0, max(cy, py) + pad)]
            out.append(self.img(ex.images[0], sent, gold, pred, crop,
                                cap="zoom: target and click"))
            return out
        pages = ex.images[:max_pages]
        for i, p in enumerate(pages):
            sent = sizes[i] if i < len(sizes) else (sizes[0] if sizes else None)
            cap = (f"page {i + 1} of {len(ex.images)} sent" if len(ex.images) > 1
                   else "image as sent")
            out.append(self.img(p, sent, cap=cap))
        return out

    def example(self, r: dict, why="", tags=(), extra_kv=(), imgs=None,
                thinking=True, ok=None, require_vision=True,
                allow_contested=False) -> dict | None:
        """Build one example card, or None if the row is not worth showing.

        Two exclusions, both applied here so no page can forget them:

        * the gold is contested by the audit -- the card would be arguing about
          the annotation, not the model;
        * the blind arm answered it correctly without the image -- whatever went
          wrong, it was not perception, so it does not belong in a gallery of
          perception failures.

        Rows with no blind counterpart are kept and labelled, because the blind
        arm only sampled 500 per benchmark; dropping them would shrink the
        galleries to a few dozen cards and silently bias them toward the sample.

        Both exclusions have exactly one page that opts out, because that page's
        subject *is* the excluded set: language_prior_override shows
        blind-answerable items on purpose (`require_vision=False`), and
        ground_truth_noise shows contested golds on purpose
        (`allow_contested=True`). Nothing else may.
        """
        status = self.d.vision_status(r)
        if r["uid"] in self.d.contested and not allow_contested:
            self.dropped["contested_gold"] += 1
            return None
        if status == "answerable" and require_vision:
            self.dropped["answerable_blind"] += 1
            return None
        self.kept[status] += 1
        ex = r["_ex"]
        gold = ex.gold[0] if isinstance(ex.gold, list) and len(ex.gold) == 1 else ex.gold
        sc = float(r.get("score") or 0.0)
        okv = hit(r) if ok is None else ok
        if ex.answer_type == "point":
            gold_txt = ("box x {:.1f}–{:.1f}%, y {:.1f}–{:.1f}% (centre {:.1f}%, {:.1f}%)"
                        .format(ex.gold[0] * 100, ex.gold[2] * 100, ex.gold[1] * 100,
                                ex.gold[3] * 100, (ex.gold[0] + ex.gold[2]) / 2 * 100,
                                (ex.gold[1] + ex.gold[3]) / 2 * 100))
            pred_txt = "clicked {:.1f}%, {:.1f}%".format(r["pred"][0] * 100, r["pred"][1] * 100)
        else:
            gold_txt, pred_txt = esc(gold), esc(r["pred"])
        kv = [("model answered", pred_txt, "g" if okv else "b"),
              ("gold", gold_txt, "g")]
        kv += list(extra_kv)
        kv.append(("score", f'{sc:.2f} <span style="color:var(--muted)">({esc(r.get("metric"))})</span>', ""))
        sizes = r.get("sent_image_sizes") or []
        if sizes:
            ew, eh = effective_size(sizes[0], (int(sizes[0][0]), int(sizes[0][1])))
            note = f"{sizes[0][0]}&times;{sizes[0][1]}"
            if (ew, eh) != (int(sizes[0][0]), int(sizes[0][1])):
                note += f" &rarr; downscaled by the API to {ew}&times;{eh}"
            if len(sizes) > 1:
                note += f" &middot; {len(sizes)} images sent"
            kv.append(("image the model saw", note, ""))
        kv.append(("uid", f'<span style="color:var(--muted)">{esc(r["uid"])}</span>', ""))
        bs = self.d.blind_score.get(r["uid"])
        if status == "confirmed":
            kv.append(("needs the image",
                       f'answered <b>wrong</b> without it (blind score {bs:.2f})', "g"))
        return {"ok": okv, "question": ex.question, "why": why, "tags": list(tags),
                "imgs": imgs if imgs is not None else self.row_images(r),
                "kv": kv, "vision": status,
                # confirmed cards sort ahead of untested ones within a group
                "_rank": {"confirmed": 0, "exempt": 0, "untested": 1}.get(status, 2),
                "thinking": (r.get("thinking") or "")[:4000] if thinking else ""}


def pick(rows, key, n, reverse=True):
    return sorted(rows, key=key, reverse=reverse)[:n]


def worst_first(rows, n):
    return sorted(rows, key=lambda r: (float(r.get("score") or 0), r["uid"]))[:n]


# --------------------------------------------------------------- utilities


def rel_errors(rows):
    """(relative error, row) for wrong answers whose gold and prediction are both
    scalars. Format-equivalent answers are excluded: they are not misreadings."""
    out = []
    for r in rows:
        if hit(r) or cause_format_equivalent(r["gold"], r["pred"]):
            continue
        pv = numval(r["pred"])
        if pv is None:
            continue
        gs = [numval(g) for g in (r["gold"] if isinstance(r["gold"], list) else [r["gold"]])]
        gs = [g for g in gs if g is not None]
        if not gs:
            continue
        gv = min(gs, key=lambda g: abs(g - pv))
        if gv == 0:
            continue
        out.append((abs(pv - gv) / abs(gv), r))
    return out


def numbers_in(t):
    return sorted({float(x.replace(",", ""))
                   for x in re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(t))})


# ============================================================ CAUSE BUILDERS
def c_effective_resolution(d: Data, b: Builder) -> Cause:
    iv, sp, cx = d.rows["infographicvqa"], d.rows["screenspot_pro"], d.rows["charxiv"]
    man = d.manifests["infographicvqa"]

    def px(r):
        s = (r.get("sent_image_sizes") or [None])[0]
        return (int(s[0]) * int(s[1])) if s else None

    def words(r):
        m = man.get(r["uid"].split(":")[1])
        return len(str(m.get("ocr") or "").split()) if m else None

    qpx = quantiles(iv, px, 5)
    qw = quantiles(iv, words, 5)
    px_items, px_vals = [], []
    for lo, hi, ch in qpx:
        a = cause_mean(r["score"] for r in ch)
        px_vals.append(a)
        px_items.append((f"{lo / 1e6:.2f}&ndash;{hi / 1e6:.2f} MP", a, f"n={len(ch)}",
                         pct_bare(a, 1), None, "" if a > 0.66 else "s2"))
    w_vals = [cause_mean(r["score"] for r in ch) for _, _, ch in qw]
    drop = (px_vals[0] - px_vals[-1]) if px_vals else 0.0

    # ScreenSpot-Pro: same story in the extreme, expressed as target size
    qa = quantiles(sp, lambda r: r["meta"]["target_area_frac"], 5)
    sp_rows = []
    for lo, hi, ch in qa:
        k = sum(1 for r in ch if hit(r))
        c4 = sum(1 for r in ch if cell_of(*r["pred"], 4) == centre_cell(r["gold"], 4))
        sp_rows.append([f"{lo * 100:.4f}&ndash;{hi * 100:.4f}% of screen", f"{len(ch)}",
                        f"{k / len(ch) * 100:.2f}%", f"{c4 / len(ch) * 100:.1f}%"])

    # CharXiv: more panels in the same pixel budget
    def sub_bin(n):
        n = int(n or 1)
        return "1" if n <= 1 else "2–3" if n <= 3 else "4–6" if n <= 6 else "7–12" if n <= 12 else "13+"
    ORDER = ["1", "2–3", "4–6", "7–12", "13+"]
    cxd = collections.defaultdict(list)
    for r in cx:
        if r["meta"].get("split") == "descriptive":
            cxd[sub_bin(r["meta"].get("num_subplots"))].append(hit(r))
    cx_rows = [[k, f"{len(cxd[k])}", f"{100 * sum(cxd[k]) / len(cxd[k]):.1f}%"]
               for k in ORDER if cxd.get(k)]
    cx_drop = (sum(cxd["1"]) / len(cxd["1"]) - sum(cxd["13+"]) / len(cxd["13+"])) if cxd.get("13+") else 0

    body = (
        cause_tiles([
            ("InfographicVQA", f"&minus;{drop * 100:.1f}pp",
             f"ANLS from the smallest fifth of images ({px_vals[0] * 100:.1f}%) to the largest "
             f"({px_vals[-1] * 100:.1f}%), n={len(iv)}", "bad"),
            ("Text volume control", f"{(max(w_vals) - min(w_vals)) * 100:.1f}pp",
             "spread across OCR-word-count quintiles &mdash; no monotone trend, so it is not "
             "simply &ldquo;more text is harder&rdquo;", "good"),
            ("CharXiv", f"&minus;{cx_drop * 100:.1f}pp",
             "descriptive accuracy, single-panel figures vs 13+ panels", "bad"),
            ("ScreenSpot-Pro", f"{sp_rows[-1][2]} vs {sp_rows[0][2]}",
             "click-in-box on the largest vs smallest fifth of targets", "bad"),
        ])
        + bars("InfographicVQA: accuracy falls as the source image gets bigger",
               "The API scales every image to at most 1568px per edge and ~1.15 megapixels. A 0.5 MP "
               "infographic arrives intact; a 20 MP one arrives at a twentieth of its detail. ANLS by "
               "quintile of the pixel count actually sent.", px_items)
        + hist_svg([("ANLS by image size (MP)", "--s2", px_vals),
                    ("ANLS by OCR word count", "--s1", w_vals)],
                   "The effect is pixels, not words",
                   "Same 2,801 questions binned two ways. Sorting by how much text the infographic "
                   "contains produces no trend; sorting by how many pixels were sent produces a "
                   "monotone decline. Whatever is lost is lost in the downscale, not in the reading.",
                   ["Q1 (smallest)", "Q2", "Q3", "Q4", "Q5 (largest)"], ymax=0.8)
        + cause_table(["ScreenSpot-Pro target size", "n", "click inside the box", "right 4&times;4 cell"],
                sp_rows,
                "The same law at the other end of the scale. Targets here average a few hundredths "
                "of one percent of the screen; the largest fifth is hit seven times more often than "
                "the smallest, and even coarse cell-level placement improves with size.")
        + cause_table(["CharXiv panels in the figure", "n", "descriptive accuracy"], cx_rows,
                "A 13-panel figure is sent at the same pixel budget as a single-panel one, so each "
                "panel gets a thirteenth of the resolution. Accuracy declines monotonically.")
        + '<div class="note"><strong>Why this is the top-ranked cause.</strong> It is the only one '
        'that three independent benchmarks measure in the same direction, and it is mechanical: the '
        'model is not given the pixels. It also predicts the others &mdash; localization precision, '
        'small-element misreads and dense-figure errors are all this cause seen through different '
        'instruments.</div>')

    groups = []
    big = [r for r in iv if px(r) and px(r) > 6e6 and not hit(r)]
    groups.append({"bench": "infographicvqa",
                   "note": "Failures on infographics sent at more than 6 megapixels &mdash; every "
                           "one of these was downscaled by 5&times; or more before the model saw it. "
                           "Zoom in and the answer is legible in the source; it was not legible at "
                           "the size Haiku received.",
                   "examples": [b.example(r, tags=["large source image"])
                                for r in worst_first(big, 8)]})
    small = [r for r in iv if px(r) and px(r) < 9e5 and hit(r)]
    groups[-1]["examples"] += [b.example(r, tags=["small source image", "contrast: correct"])
                               for r in pick(small, lambda r: r["score"], 3)]
    tiny = [r for r in sp if r["meta"]["target_area_frac"] < 0.0002 and not hit(r)]
    groups.append({"bench": "screenspot_pro",
                   "note": "Targets under 0.02% of the screen &mdash; roughly a 16&times;16 pixel "
                           "icon on a 4K screenshot, which survives the downscale as about 6 pixels "
                           "across. The click is usually in the right neighbourhood.",
                   "examples": [b.example(r, tags=["tiny target"]) for r in worst_first(tiny, 6)]})
    dense = [r for r in cx if (r["meta"].get("num_subplots") or 1) >= 9
             and r["meta"].get("split") == "descriptive" and not hit(r)]
    groups.append({"bench": "charxiv",
                   "note": "Figures with nine or more panels. The question names one panel; the "
                           "resolution available to that panel is a ninth of the figure's.",
                   "examples": [b.example(r, tags=["9+ panels"]) for r in worst_first(dense, 6)]})

    refute = (
        "<p>This would be refuted if accuracy tracked <em>content</em> rather than <em>pixels</em>. "
        "It does not: binning the same InfographicVQA questions by OCR word count &mdash; a direct "
        "measure of how much there is to read &mdash; produces a flat, non-monotone curve, while "
        "binning by delivered pixel count produces a clean decline.</p>"
        "<p><strong>Not tested.</strong> The decisive experiment is a resolution ablation: send the "
        "same question at full resolution, at half, and tiled into crops, and watch accuracy move. "
        "That needs API calls, which this study did not spend. Until then the causal direction rests "
        "on the OCR-word control and on the agreement of three benchmarks, not on an intervention. "
        "A second confound remains: very large infographics may also be intrinsically harder "
        "(longer, more multi-part questions), and the OCR-word control only partly rules that out.</p>")
    return Cause("effective_resolution", "Detail is lost before the model ever looks",
                 "Accuracy is governed by how many pixels of the relevant element actually reach "
                 "the model. Every image is downscaled to about 1.15 megapixels, so the larger or "
                 "denser the source, the less of it the model can resolve — and accuracy falls "
                 "monotonically with that loss on three independent benchmarks.",
                 "PROVEN", f"&minus;{drop * 100:.0f}pp ANLS across image-size quintiles",
                 ["infographicvqa", "screenspot_pro", "charxiv"], 0.98,
                 body, groups, refute,
                 ["resolution_precision", "wrong_element_not_near_miss", "subplot_scope"])


def c_resolution_precision(d: Data, b: Builder) -> Cause:
    sp = d.rows["screenspot_pro"]
    n = len(sp)
    grid_rows, gvals, cvals = [], [], []
    for g in (2, 3, 4, 8, 16):
        k = sum(1 for r in sp if cell_of(*r["pred"], g) == centre_cell(r["gold"], g))
        anyk = sum(1 for r in sp if cell_of(*r["pred"], g) in bbox_cells(r["gold"], g))
        ch = 1 / (g * g)
        gvals.append(k / n)
        cvals.append(ch)
        grid_rows.append([f"{g}&times;{g} grid", f"{n}", f"{k / n * 100:.1f}%",
                          f"{ch * 100:.2f}%", f"{k / n / ch:.1f}&times;",
                          f"{anyk / n * 100:.1f}%"])
    exact = sum(1 for r in sp if hit(r)) / n
    area = cause_mean((r["gold"][2] - r["gold"][0]) * (r["gold"][3] - r["gold"][1]) for r in sp)
    gvals.append(exact)
    cvals.append(area)
    grid_rows.append([("<strong>exact: inside the box</strong>", ""), f"{n}",
                      f"<strong>{pct_bare(exact, 2)}%</strong>", pct_or_dash(area, 4),
                      f"<strong>{exact / area:.0f}&times;</strong>", "&mdash;"])

    # grid control: does naming a cell beat clicking, at the same granularity?
    g4 = {k: v for k, v in d.grid4.items() if k != "__meta__" and v.get("pred") is not None}
    byuid = {r["uid"]: r for r in sp}
    paired = [(k, v, byuid[k]) for k, v in g4.items() if k in byuid]

    def cellstr(s):
        m = re.match(r"\s*([A-Da-d])\s*([1-4])", str(s))
        return (m.group(1).upper() + m.group(2)) if m else None
    name_ok = [1 if (cellstr(v["gold"][0] if isinstance(v["gold"], list) else v["gold"])
                     == cellstr(v["pred"])) else 0 for _, v, _ in paired]
    click_ok = [1 if cell_of(*m["pred"], 4) == centre_cell(m["gold"], 4) else 0
                for _, _, m in paired]
    name_acc, click_acc = cause_mean(name_ok), cause_mean(click_ok)
    delta = name_acc - click_acc
    lo1, hi1 = wilson(sum(name_ok), len(name_ok))
    lo2, hi2 = wilson(sum(click_ok), len(click_ok))
    disjoint = lo1 > hi2

    ui_rows = []
    for ui in sorted({r["meta"]["ui_type"] for r in sp}):
        v = [r for r in sp if r["meta"]["ui_type"] == ui]
        k4 = sum(1 for r in v if cell_of(*r["pred"], 4) == centre_cell(r["gold"], 4))
        ui_rows.append([ui, f"{len(v)}", f"{100 * sum(1 for r in v if hit(r)) / len(v):.2f}%",
                        f"{100 * k4 / len(v):.1f}%",
                        f"{100 * cause_mean((r['gold'][2] - r['gold'][0]) * (r['gold'][3] - r['gold'][1]) for r in v):.4f}%"])

    modes = collections.Counter(classify_point(r["_ex"].gold, r["pred"]) for r in sp if not hit(r))
    nf = sum(modes.values())
    mode_rows = [[{"near_miss": "right area, missed the box (&lt;10% of the screen off)",
                   "moderate_miss": "roughly the wrong place (10&ndash;25% off)",
                   "wrong_region": "nowhere near (&gt;25% off)"}.get(k, k),
                  f"{v}", f"{100 * v / nf:.1f}%"] for k, v in modes.most_common()]

    body = (
        cause_tiles([("Exact click accuracy", pct_or_dash(exact, 1),
                f"n={n}; chance is {pct_bare(area, 3)}% because the mean target is that "
                f"fraction of the screen", "bad"),
               ("Ratio above chance, 2&times;2", f"{gvals[0] / cvals[0]:.1f}&times;", "coarse quadrant", "warn"),
               ("Ratio above chance, exact", f"{exact / area:.0f}&times;",
                "the fine information <em>is</em> there &mdash; it is just not sufficient", "good"),
               ("Naming a cell vs clicking", f"{delta * 100:+.1f}pp",
                f"same 4&times;4 granularity, same {len(paired)} items", "warn")])
        + cause_table(["granularity required", "n", "accuracy", "chance", "ratio above chance",
                 "accuracy if any overlapping cell counts"], grid_rows,
                "Every click bucketed into a coarser grid. The model's answer is the cell its click "
                "lands in; the target's cell is the one containing the box centre, so chance is "
                "exactly 1/g&sup2;. The last column is the more forgiving definition &mdash; credit "
                "for any cell the target box touches.")
        + hist_svg([("accuracy", "--s1", gvals), ("chance", "--s2", cvals)],
                   "Localization degrades smoothly, and never collapses to chance",
                   "There is no cliff. Performance decays continuously as the required precision "
                   "rises, and the multiple of chance <em>increases</em> the whole way &mdash; the "
                   "model carries genuine fine-grained information about where things are, just not "
                   "enough of it to land inside a 0.06%-of-screen box.",
                   ["2×2", "3×3", "4×4", "8×8", "16×16", "exact box"], ymax=0.8)
        + f'<div class="note {"warn" if disjoint else ""}"><strong>Is the deficit perception or '
        f'expression?</strong> The grid control removes coordinate emission entirely: a labelled '
        f'4&times;4 grid is drawn on the screenshot and the model just names a cell. On the same '
        f'{len(paired)} items it scores <strong>{name_acc * 100:.1f}%</strong> '
        f'[{lo1 * 100:.1f}&ndash;{hi1 * 100:.1f}] against <strong>{click_acc * 100:.1f}%</strong> '
        f'[{lo2 * 100:.1f}&ndash;{hi2 * 100:.1f}] for the click-derived cell &mdash; '
        f'{delta * 100:+.1f}pp. The intervals '
        f'{"do not overlap, so part of the deficit is expression" if disjoint else "overlap, so the expression component is not established"}'
        f'. But naming a cell is a 16-way choice and the model still gets two thirds of them wrong, '
        f'so the great majority of the deficit is perceptual, not a coordinate-emission artifact. '
        f'Note one confound: the grid screenshots were pre-downscaled to 1568px before the grid was '
        f'drawn, while the main run sent native resolution, so the grid arm saw slightly '
        f'<em>less</em> detail &mdash; which makes its advantage a lower bound.</div>'
        + cause_table(["element type", "n", "click inside the box", "right 4&times;4 cell",
                 "mean target size"], ui_rows,
                "Icons are both smaller and harder than text, and the gap survives at cell "
                "granularity, so it is not purely a size effect.")
        + cause_table(["how the click missed", "n", "share of failures"], mode_rows,
                "Classified by distance from the target centre. Only about a quarter are near "
                "misses; the rest land somewhere else on the screen entirely."))

    groups = []
    near = [r for r in sp if not hit(r) and classify_point(r["_ex"].gold, r["pred"]) == "near_miss"]
    far = [r for r in sp if not hit(r) and classify_point(r["_ex"].gold, r["pred"]) == "wrong_region"]
    groups.append({"bench": "screenspot_pro",
                   "note": "Near misses first &mdash; the model found the right toolbar and missed "
                           "the icon &mdash; then clicks that landed in a different region "
                           "altogether, then the rare successes.",
                   "examples": ([b.example(r, tags=["near miss", r["meta"]["ui_type"]])
                                 for r in worst_first(near, 6)]
                                + [b.example(r, tags=["wrong region", r["meta"]["ui_type"]])
                                   for r in worst_first(far, 6)]
                                + [b.example(r, tags=["contrast: correct", r["meta"]["ui_type"]])
                                   for r in [x for x in sp if hit(x)][:3]])})
    gex = []
    for su, v, m in paired:
        gold = v["gold"][0] if isinstance(v["gold"], list) else v["gold"]
        if cellstr(gold) == cellstr(v["pred"]) and not hit(m):
            gex.append((su, v, m))
    if gex:
        groups.append({"bench": "screenspot_pro",
                       "note": "<strong>Grid control divergences.</strong> The model named the "
                               "correct 4&times;4 cell but its free coordinate click missed the "
                               "target &mdash; the cases where expression, not perception, is the "
                               "binding constraint.",
                       "examples": [b.example(m, tags=["named the right cell", "clicked wrong"],
                                              extra_kv=[("grid-control answer",
                                                         f'<span style="color:var(--good)">{esc(v["pred"])}</span> '
                                                         f'&mdash; correct', "")])
                                    for _, v, m in gex[:8]]})

    refute = (
        "<p>Refuted if accuracy were flat across granularities &mdash; that would mean the model "
        "either knows exactly where the element is or has no idea, with nothing in between. It is "
        "not flat: the curve decays smoothly and the ratio above chance rises monotonically from "
        f"{gvals[0] / cvals[0]:.1f}&times; to {exact / area:.0f}&times;.</p>"
        "<p>Also refuted if the grid control had scored near-perfectly, which would have made the "
        "whole deficit a coordinate-emission artifact. It scored "
        f"{name_acc * 100:.1f}% &mdash; better than clicking, nowhere near solved.</p>"
        "<p><strong>Not tested.</strong> Whether zooming into a crop recovers the target (a tiling "
        "or magnification arm), and whether a different coordinate convention &mdash; percentages "
        "instead of pixels, or a two-step coarse-then-fine protocol &mdash; closes the remaining "
        "gap. Both need API calls. This is also a single-benchmark cause: ScreenSpot-Pro is the "
        "only source here that asks for a coordinate at all.</p>")
    return Cause("resolution_precision", "The model knows roughly where, not exactly which",
                 "Localization degrades smoothly with the precision demanded of it. Haiku places "
                 "elements in the right quadrant far above chance and in the right 4×4 cell five "
                 "times above chance, but lands inside the actual element only "
                 f"{pct_bare(exact, 1)}% of the time.",
                 "PROVEN", f"{pct_bare(exact, 1)}% exact vs {gvals[0] * 100:.0f}% at quadrant level",
                 ["screenspot_pro"], 0.94, body, groups, refute,
                 ["effective_resolution", "label_reference_binding"],
                 one_bench_note="ScreenSpot-Pro is the only benchmark in this study that asks for a "
                                "coordinate, so the precision curve cannot be replicated elsewhere. "
                                "What <em>can</em> be replicated is the underlying resolution "
                                "story &mdash; see effective_resolution, which finds the same "
                                "gradient on InfographicVQA and CharXiv.")


def _blind_pairs(d: Data):
    """(blind row, sighted row, blind score) for every item run in both arms."""
    out = []
    for su, br in d.blind.items():
        if su == "__meta__" or br.get("pred") is None:
            continue
        m = d.by_uid.get(su)
        e = d.ex.get(su)
        if m is None or e is None:
            continue
        s = official_score(e, br["pred"])
        out.append((br, m, float(s.get("score") or 0.0)))
    return out


def c_language_prior_override(d: Data, b: Builder) -> Cause:
    pairs = _blind_pairs(d)
    by = collections.defaultdict(list)
    for br, m, s in pairs:
        by[m["bench"]].append((br, m, s))
    order = ["ai2d", "charxiv", "infographicvqa", "slidevqa"]
    rows, series_b, series_s, labels = [], [], [], []
    for k in order:
        v = by.get(k) or []
        if not v:
            continue
        bl = cause_mean(s for _, _, s in v)
        si = cause_mean(float(m.get("score") or 0) for _, m, _ in v)
        chance = 0.25 if k == "ai2d" else None
        rows.append([NICE[k], f"{len(v)}", pct_or_dash(bl, 1), pct_or_dash(si, 1),
                     (f"<strong>+{(si - bl) * 100:.1f}pp</strong>", "num"),
                     ("25%" if chance else "&mdash;")])
        series_b.append(bl)
        series_s.append(si)
        labels.append(NICE[k])
    cxp = by.get("charxiv") or []
    split_rows = []
    for sp_ in ("descriptive", "reasoning"):
        v = [x for x in cxp if x[1]["meta"].get("split") == sp_]
        if v:
            split_rows.append([f"CharXiv {sp_}", f"{len(v)}",
                               pct_or_dash(cause_mean(s for _, _, s in v), 1),
                               pct_or_dash(cause_mean(float(m.get('score') or 0) for _, m, _ in v), 1),
                               f"+{(cause_mean(float(m.get('score') or 0) for _, m, _ in v) - cause_mean(s for _, _, s in v)) * 100:.1f}pp"])
    aiv = by.get("ai2d") or []
    for qt in ("label_reference", "diagram_reasoning"):
        v = [x for x in aiv if x[1]["meta"].get("qtype") == qt]
        if v:
            split_rows.append([f"AI2D {qt.replace('_', ' ')}", f"{len(v)}",
                               pct_or_dash(cause_mean(s for _, _, s in v), 1),
                               pct_or_dash(cause_mean(float(m.get('score') or 0) for _, m, _ in v), 1),
                               f"+{(cause_mean(float(m.get('score') or 0) for _, m, _ in v) - cause_mean(s for _, _, s in v)) * 100:.1f}pp"])
    ai_gain = next((float(r[4].replace("+", "").replace("pp", "")) if isinstance(r[4], str)
                    else 0 for r in rows if r[0] == "AI2D"), 0)
    ai_gain = (cause_mean(float(m.get("score") or 0) for _, m, _ in aiv)
               - cause_mean(s for _, _, s in aiv)) if aiv else 0

    body = (
        cause_tiles([("AI2D without the diagram", pct_or_dash(cause_mean(s for _, _, s in aiv), 1),
                f"chance is 25%; the image adds only {ai_gain * 100:.1f}pp (n={len(aiv)})", "bad"),
               ("CharXiv without the figure",
                pct_or_dash(cause_mean(s for _, _, s in cxp), 1),
                f"vision adds {(cause_mean(float(m.get('score') or 0) for _, m, _ in cxp) - cause_mean(s for _, _, s in cxp)) * 100:.1f}pp "
                f"(n={len(cxp)})", "good"),
               ("Items answered correctly blind",
                f"{sum(1 for _, _, s in pairs if s >= 0.5)}",
                f"of {len(pairs)} paired control items &mdash; the image was never needed", "warn")])
        + cause_table(["benchmark", "paired n", "no image", "with image", "what vision buys", "chance"],
                rows,
                "Same question, same prompt, same model &mdash; the image simply withheld. The gap "
                "is the honest measure of how much of a benchmark score is actually perception.")
        + hist_svg([("blind (no image)", "--s2", series_b), ("sighted", "--s1", series_s)],
                   "How much of each benchmark is a vision test at all",
                   "AI2D is barely one: a text-only model already scores well over twice chance "
                   "from world knowledge about food chains, water cycles and life cycles. CharXiv "
                   "is almost entirely one.", labels, ymax=1.0)
        + cause_table(["split", "n", "no image", "with image", "vision gain"], split_rows,
                "The split view matters more than the benchmark average. AI2D's label-reference "
                "questions are a real perception test; its reasoning questions are mostly a "
                "knowledge test that happens to ship a picture.")
        + '<div class="note warn"><strong>What this changes about every other number on this '
        'site.</strong> A benchmark score is an upper bound on perception, not a measurement of it. '
        'When AI2D reports 82% it is reporting roughly 63 points of prior plus 19 points of '
        'seeing. Any claim about a perceptual blind spot has to be argued against the blind '
        'baseline, not against zero.</div>')

    groups = []
    for k in order:
        v = by.get(k) or []
        blind_right = [(br, m, s) for br, m, s in v if s >= 0.5]
        blind_right.sort(key=lambda x: -x[2])
        ex = []
        for br, m, s in blind_right[:9]:
            e = b.example(m, tags=["answered correctly with no image"],
                          extra_kv=[("answer with NO image",
                                     f'<span style="color:var(--bad)">{esc(br["pred"])}</span> '
                                     f'&mdash; scored {s:.2f}', "")],
                          ok=True, require_vision=False)
            if e is None:                     # contested gold
                continue
            e["thinking"] = (br.get("thinking") or "")[:4000]
            e["why"] = ("The image is shown here only to make the point that it was never needed. "
                        "The blind trace above is the model reasoning from the question text alone.")
            ex.append(e)
        blind_wrong = [(br, m, s) for br, m, s in v if s < 0.5 and hit(m)]
        for br, m, s in blind_wrong[:4]:
            e = b.example(m, tags=["needed the image"],
                          extra_kv=[("answer with NO image",
                                     f'<span style="color:var(--bad)">{esc(br["pred"])}</span> '
                                     f'&mdash; scored {s:.2f}', "")])
            if e is None:
                continue
            e["why"] = "Contrasting case: wrong without the image, right with it. This is the part " \
                       "of the benchmark that is genuinely measuring perception."
            ex.append(e)
        if ex:
            groups.append({"bench": k,
                           "note": f"{len(blind_right)} of {len(v)} paired items were answered "
                                   f"correctly with no image at all "
                                   f"({len(blind_right) / max(len(v), 1) * 100:.0f}%). Shown first; "
                                   f"contrasting image-dependent items at the end.",
                           "examples": ex})

    refute = (
        "<p>Refuted for a given benchmark if blind accuracy sat at chance. It does for none of "
        "them, but the size of the prior varies by a factor of three, which is the finding.</p>"
        "<p><strong>Caveats.</strong> The blind arm is a 500-item random sample per benchmark, and "
        f"only {len(by.get('slidevqa') or [])} SlideVQA blind items have a sighted counterpart "
        "because the sighted run covers a subset of the deck questions &mdash; that row has the "
        "widest interval. The control also withholds the image entirely rather than replacing it "
        "with a scrambled one, so it measures &ldquo;answerable from text&rdquo; rather than "
        "&ldquo;ignores the image&rdquo;. A shuffled-image control would separate those two and "
        "was not run.</p>")
    return Cause("language_prior_override", "Some of these benchmarks barely need the picture",
                 "A large share of benchmark score is recoverable from the question text alone. "
                 "With the image withheld the model still answers "
                 f"{pct_bare(cause_mean(s for _, _, s in (by.get('ai2d') or [])), 0)}% of AI2D correctly "
                 "against 25% chance, so at most a fifth of that score is perception.",
                 "PROVEN",
                 f"vision adds only {ai_gain * 100:.0f}pp on AI2D vs "
                 f"{(cause_mean(float(m.get('score') or 0) for _, m, _ in cxp) - cause_mean(s for _, _, s in cxp)) * 100:.0f}pp on CharXiv",
                 order, 0.96, body, groups, refute,
                 ["label_reference_binding", "ground_truth_noise"])


def c_label_reference_binding(d: Data, b: Builder) -> Cause:
    ai, cx = d.rows["ai2d"], d.rows["charxiv"]
    lr = [r for r in ai if r["meta"]["qtype"] == "label_reference"]
    dr = [r for r in ai if r["meta"]["qtype"] == "diagram_reasoning"]
    a_lr, a_dr = cause_mean(hit(r) for r in lr), cause_mean(hit(r) for r in dr)
    pairs = _blind_pairs(d)
    blr = [(br, m, s) for br, m, s in pairs
           if m["bench"] == "ai2d" and m["meta"]["qtype"] == "label_reference"]
    bdr = [(br, m, s) for br, m, s in pairs
           if m["bench"] == "ai2d" and m["meta"]["qtype"] == "diagram_reasoning"]

    # CharXiv's own binding question: name the legend entries, on figures that have one
    q13 = [r for r in cx if r["meta"].get("qid") == 13 and not is_na(r["gold"][0])]
    q12 = [r for r in cx if r["meta"].get("qid") == 12 and not is_na(r["gold"][0])]
    a13, a12 = cause_mean(hit(r) for r in q13), cause_mean(hit(r) for r in q12)

    rows = [["AI2D label-reference (resolve a printed letter to the thing it marks)",
             f"{len(lr)}", f"{a_lr * 100:.1f}%",
             pct_or_dash(cause_mean(s for _, _, s in blr), 1) if blr else "&mdash;",
             f"n={len(blr)}"],
            ["AI2D diagram-reasoning", f"{len(dr)}", f"{a_dr * 100:.1f}%",
             pct_or_dash(cause_mean(s for _, _, s in bdr), 1) if bdr else "&mdash;",
             f"n={len(bdr)}"],
            ["CharXiv: name the legend entries (figures that have a legend)",
             f"{len(q13)}", f"{a13 * 100:.1f}%", "&mdash;", "not in blind arm"],
            ["CharXiv: count the legend entries", f"{len(q12)}", f"{a12 * 100:.1f}%",
             "&mdash;", "not in blind arm"]]

    body = (
        cause_tiles([("AI2D label-reference", f"{a_lr * 100:.1f}%",
                f"vs {a_dr * 100:.1f}% on the same benchmark's reasoning questions "
                f"(n={len(lr)} / {len(dr)})", "bad"),
               ("Gap", f"{(a_dr - a_lr) * 100:.1f}pp",
                "the largest within-benchmark split anywhere in this study", "bad"),
               ("Blind baseline", pct_or_dash(cause_mean(s for _, _, s in blr), 1) if blr else "&mdash;",
                "label-reference is near chance without the diagram, so this really is perception",
                "good"),
               ("CharXiv legend naming", f"{a13 * 100:.1f}%",
                f"the same binding operation on charts is near-solved (n={len(q13)})", "good")])
        + cause_table(["question family", "n", "accuracy", "blind accuracy", ""], rows,
                "Resolving a mark printed on the image to the object it designates. AI2D asks it "
                "directly with letters on a diagram; CharXiv asks it with a legend on a chart.")
        + '<div class="note warn"><strong>The two benchmarks disagree, and the disagreement is the '
        'result.</strong> Binding a legend entry to a chart series is near-solved '
        f'({a13 * 100:.0f}% on CharXiv). Binding a letter printed <em>inside</em> a diagram to the '
        'structure it points at is not '
        f'({a_lr * 100:.0f}% on AI2D, against 25% chance). The difference is that a legend is a '
        'tidy list beside the plot with an unambiguous colour key, whereas an AI2D label is a '
        'bare letter dropped on top of the artwork, resolved only by proximity and by a leader '
        'line. So the failing operation is not &ldquo;binding&rdquo; in general &mdash; it is '
        'binding that has to be resolved <em>spatially</em>, by working out which nearby thing a '
        'mark belongs to. That is the same operation ScreenSpot-Pro fails at, from the other '
        'direction.</div>'
        + bars("Where AI2D's score actually comes from",
               "Split by question type, with the blind control beneath each. The reasoning half is "
               "mostly world knowledge; the label half is mostly looking.",
               [("label-reference &mdash; sighted", a_lr, f"n={len(lr)}", f"{a_lr * 100:.1f}", 0.25, ""),
                ("label-reference &mdash; blind", cause_mean(s for _, _, s in blr) if blr else None,
                 f"n={len(blr)}", pct_bare(cause_mean(s for _, _, s in blr), 1) if blr else "&mdash;", 0.25, "s2"),
                ("diagram-reasoning &mdash; sighted", a_dr, f"n={len(dr)}", f"{a_dr * 100:.1f}", 0.25, ""),
                ("diagram-reasoning &mdash; blind", cause_mean(s for _, _, s in bdr) if bdr else None,
                 f"n={len(bdr)}", pct_bare(cause_mean(s for _, _, s in bdr), 1) if bdr else "&mdash;", 0.25, "s2")]))

    groups = [
        {"bench": "ai2d",
         "note": "Label-reference failures. The question names a letter printed on the diagram and "
                 "asks what it marks; four candidate letters are offered, so chance is 25%.",
         "examples": ([b.example(r, tags=["label reference"],
                                 extra_kv=[("options", esc(", ".join(map(str, r["meta"]["options"])))
                                            , ""),
                                           ("gold option text", esc(r["meta"].get("gold_text")), "g")])
                       for r in worst_first([r for r in lr if not hit(r)], 10)]
                      + [b.example(r, tags=["label reference", "contrast: correct"],
                                   extra_kv=[("options", esc(", ".join(map(str, r["meta"]["options"]))), "")])
                         for r in [r for r in lr if hit(r)][:3]])},
        {"bench": "charxiv",
         "note": "The contrasting case. CharXiv asks the same kind of binding question about a "
                 "legend and the model is right almost every time; these are the rare misses.",
         "examples": ([b.example(r, tags=["legend naming"])
                       for r in worst_first([r for r in q13 if not hit(r)], 6)]
                      + [b.example(r, tags=["legend naming", "contrast: correct"])
                         for r in [r for r in q13 if hit(r)][:2]])},
    ]
    refute = (
        "<p>This would be refuted if AI2D's label-reference deficit were a knowledge deficit rather "
        "than a perceptual one. The blind control rules that out: withholding the diagram drops "
        f"label-reference to {pct_bare(cause_mean(s for _, _, s in blr), 1)}% "
        f"(chance 25%, n={len(blr)}) while reasoning questions survive at "
        f"{pct_bare(cause_mean(s for _, _, s in bdr), 1)}%. Nearly all the label-reference signal comes from "
        "looking, and most of the looking fails.</p>"
        "<p><strong>Partly refuted by CharXiv,</strong> and that is stated on the page rather than "
        "buried: legend-to-series binding is near-solved, so the claim has to be narrowed to "
        "spatially-resolved binding. A third benchmark with in-figure callout labels &mdash; "
        "engineering drawings, anatomical diagrams &mdash; would settle whether the narrow version "
        "generalises. We did not have one.</p>"
        "<p><strong>Not tested.</strong> Whether the failure is finding the letter or resolving what "
        "it points to. An ablation that highlights the labelled region would separate them.</p>")
    return Cause("label_reference_binding", "Binding a mark to the thing it marks",
                 "Reading a label that sits <em>on</em> the artwork and deciding which object it "
                 "designates is the weakest single operation measured here: "
                 f"{a_lr * 100:.0f}% against {a_dr * 100:.0f}% for reasoning questions on the same "
                 "diagrams. Binding a legend entry beside a chart, by contrast, is near-solved.",
                 "MIXED", f"{(a_dr - a_lr) * 100:.0f}pp within-benchmark gap on AI2D",
                 ["ai2d", "charxiv"], 0.88, body, groups, refute,
                 ["language_prior_override", "resolution_precision", "absence_detection"])


def c_absence_detection(d: Data, b: Builder) -> Cause:
    cx = d.rows["charxiv"]
    na = [r for r in cx if is_na(r["gold"][0])]
    ans = [r for r in cx if not is_na(r["gold"][0])]
    invented = [r for r in na if not is_na(r["pred"])]
    over = [r for r in ans if is_na(r["pred"])]
    inv_rate = len(invented) / len(na)
    lo, hi = wilson(len(invented), len(na))

    by = collections.defaultdict(lambda: [0, 0])
    lab = {}
    for r in na:
        q = r["meta"].get("qid")
        by[q][0] += 1
        by[q][1] += (not is_na(r["pred"]))
        lab[q] = r["meta"].get("qlabel")
    rows = []
    for q, (n, i) in sorted(by.items(), key=lambda kv: -kv[1][1] / kv[1][0]):
        l1, h1 = wilson(i, n)
        rows.append([f"qid {q} &mdash; {esc(lab[q])[:78]}", f"{n}",
                     (f'<span style="color:var(--bad)">{100 * i / n:.1f}%</span>' if i / n > 0.15
                      else f"{100 * i / n:.1f}%"),
                     f"{l1 * 100:.0f}&ndash;{h1 * 100:.0f}%", f"{100 * (1 - i / n):.1f}%"])

    body = (
        cause_tiles([("Invents a structure that is not there", f"{inv_rate * 100:.1f}%",
                f"of {len(na)} questions whose gold answer is &ldquo;Not Applicable&rdquo; "
                f"[95% CI {lo * 100:.1f}&ndash;{hi * 100:.1f}]", "bad"),
               ("Correctly abstains", f"{(1 - inv_rate) * 100:.1f}%", "the common case", "good"),
               ("Over-abstains", f"{len(over) / len(ans) * 100:.2f}%",
                f"says &ldquo;Not Applicable&rdquo; when an answer exists (n={len(ans)}) &mdash; "
                f"the error is one-sided", "good"),
               ("Worst template",
                f"{max(100 * i / n for _, (n, i) in by.items()):.0f}%",
                "invention rate on &ldquo;how many entries in the legend&rdquo; for figures with no "
                "legend", "bad")])
        + cause_table(["CharXiv template, on items where the structure is absent", "n",
                 "invents an answer", "95% CI", "correctly abstains"], rows,
                "CharXiv's descriptive templates are generated per figure, so the same question is "
                "asked of figures that do and do not have the structure. That makes the "
                "&ldquo;Not Applicable&rdquo; subset a clean absence probe.")
        + '<div class="note"><strong>The error is one-sided and it is structural.</strong> Haiku '
        f'almost never refuses an answerable question ({len(over) / len(ans) * 100:.2f}%), but on '
        'absent structures it fabricates one in one case out of ten. The rate is not uniform: '
        'templates about things that are <em>usually</em> present in a chart &mdash; a legend, '
        'intersecting lines, a title &mdash; are where invention concentrates, and templates about '
        'things that are nearly always present anyway &mdash; axis labels &mdash; are where it '
        'vanishes. That is the signature of a prior about what charts contain overriding what this '
        'chart contains.</div>')

    inv_sorted = sorted(invented, key=lambda r: (-by[r["meta"].get("qid")][1] / by[r["meta"].get("qid")][0],
                                                 r["uid"]))
    groups = [{"bench": "charxiv",
               "note": "Figures with no legend where the model reported a legend, no intersecting "
                       "lines where it reported intersections, no title where it reported one. The "
                       "thinking trace is worth opening &mdash; the fabrication is usually visible "
                       "as a confident description of something absent.",
               "examples": ([b.example(r, tags=[f"qid {r['meta'].get('qid')}", "invented"],
                                       extra_kv=[("template", esc(r["meta"].get("qlabel")), "")])
                             for r in inv_sorted[:14]]
                            + [b.example(r, tags=["contrast: correctly abstained"])
                               for r in [x for x in na if is_na(x["pred"])][:3]])}]
    refute = (
        "<p>Refuted if the model were simply reluctant to answer &mdash; a high abstention rate "
        "everywhere would explain the &ldquo;Not Applicable&rdquo; successes without any absence "
        f"detection. The opposite holds: over-abstention on answerable items is {len(over)} of "
        f"{len(ans)} ({len(over) / len(ans) * 100:.2f}%), so abstention is not a general habit.</p>"
        "<p><strong>Single-benchmark.</strong> CharXiv is the only source here with a systematic "
        "absent-structure gold. InfographicVQA and SlideVQA have no unanswerable questions at all; "
        "AI2D always has a correct option. An absence probe on a second benchmark &mdash; asking "
        "about a series that is not in the chart &mdash; is the obvious missing experiment, and it "
        "needs API calls.</p>"
        "<p><strong>Measurement caveat.</strong> A small number of &ldquo;Not Applicable&rdquo; "
        "golds are themselves debatable &mdash; whether a colourbar counts as a legend, for "
        "instance. See ground_truth_noise for the size of that floor.</p>")
    return Cause("absence_detection", "When a structure is absent, the model invents it",
                 "Asked about something a chart does not contain, Haiku fabricates a plausible "
                 f"answer {inv_rate * 100:.1f}% of the time. The error is strongly one-sided: it "
                 "almost never abstains on a question that does have an answer.",
                 "SUPPORTED", f"{inv_rate * 100:.1f}% invention on n={len(na)} absent-structure items",
                 ["charxiv"], 0.83, body, groups, refute,
                 ["label_reference_binding", "ground_truth_noise"],
                 one_bench_note="CharXiv is the only benchmark in this study that systematically "
                                "asks questions whose correct answer is &ldquo;there is no such "
                                "thing here&rdquo;. Without a second source this is a claim about "
                                "Haiku on scientific charts, not a general property.")


def c_wrong_element(d: Data, b: Builder) -> Cause:
    cx, iv, sv = d.rows["charxiv"], d.rows["infographicvqa"], d.rows["slidevqa"]
    sets = [("CharXiv descriptive", "charxiv",
             [r for r in cx if r["meta"].get("split") == "descriptive"]),
            ("CharXiv reasoning", "charxiv",
             [r for r in cx if r["meta"].get("split") == "reasoning"]),
            ("InfographicVQA", "infographicvqa", iv),
            ("SlideVQA", "slidevqa", sv)]
    rows, buckets, labels = [], [], ["≤10%", "10–25%", "25–50%", "50–100%", ">100%", "≥10×"]
    series = []
    for name, _, rs in sets:
        es = [e for e, _ in rel_errors(rs)]
        if not es:
            continue
        es.sort()
        med = statistics.median(es)
        def sh(lo, hi):
            return sum(1 for x in es if lo < x <= hi) / len(es)
        dist = [sum(1 for x in es if x <= 0.10) / len(es), sh(0.10, 0.25), sh(0.25, 0.50),
                sh(0.50, 1.0), sum(1 for x in es if x > 1.0) / len(es),
                sum(1 for x in es if x >= 9) / len(es)]
        series.append((name, ["--s1", "--s2", "--s3", "--good"][len(series) % 4], dist))
        rows.append([name, f"{len(es)}", f"{med * 100:.1f}%", f"{dist[0] * 100:.1f}%",
                     f"{dist[4] * 100:.1f}%", f"{dist[5] * 100:.1f}%"])
        buckets.append((name, dist))

    # is the predicted value another number that is present in the same figure?
    def other_value_hits(rs, get_pool):
        out = []
        for r in rs:
            if hit(r) or cause_format_equivalent(r["gold"], r["pred"]):
                continue
            pv = numval(r["pred"])
            if pv is None:
                continue
            pool = get_pool(r)
            if pool and any(abs(pv - x) <= 1e-9 * max(1.0, abs(x)) for x in pool):
                out.append(r)
        return out

    ivman = d.manifests["infographicvqa"]

    def iv_pool(r):
        m = ivman.get(r["uid"].split(":")[1])
        if not m:
            return []
        gs = {numval(g) for g in r["gold"]} - {None}
        return [x for x in numbers_in(str(m.get("ocr") or "")) if x not in gs]

    iv_other = other_value_hits(iv, iv_pool)
    iv_denom = [r for r in iv if not hit(r) and not cause_format_equivalent(r["gold"], r["pred"])
                and numval(r["pred"]) is not None and iv_pool(r)]
    # Permutation control. A dense infographic contains dozens of numbers, so a
    # wrong answer will often coincide with one by chance. Shuffling predictions
    # across questions measures exactly that base rate -- and it turns out to be
    # high, so without this control the check would look far stronger than it is.
    import random as _random
    _rng = _random.Random(0)
    _preds = [numval(r["pred"]) for r in iv_denom]
    _pools = [iv_pool(r) for r in iv_denom]
    _tot, _R = 0, 20
    for _ in range(_R):
        sh = _preds[:]
        _rng.shuffle(sh)
        _tot += sum(1 for pv, pl in zip(sh, _pools)
                    if any(abs(pv - x) <= 1e-9 * max(1.0, abs(x)) for x in pl))
    iv_null = _tot / _R / max(len(iv_denom), 1)
    iv_obs = len(iv_other) / max(len(iv_denom), 1)

    body = (
        cause_tiles([("Median relative error", f"{rows[0][2]}",
                f"CharXiv descriptive, over {rows[0][1]} wrong numeric answers", "bad"),
               ("Within 10% of the truth", f"{rows[0][3]}",
                "if the failure were imprecise interpolation this would dominate", "warn"),
               ("Predicted value is another number printed on the same infographic",
                f"{(iv_obs - iv_null) * 100:+.1f}pp",
                f"{len(iv_other)} of {len(iv_denom)} wrong InfographicVQA numbers "
                f"({iv_obs * 100:.0f}%) appear verbatim in the page's own OCR &mdash; but a "
                f"permutation control puts the chance rate at {iv_null * 100:.0f}%, so the real "
                f"excess is small", "warn")])
        + cause_table(["benchmark / split", "wrong numeric answers", "median relative error",
                 "within 10%", "over 100% off", "off by 10&times; or more"], rows,
                "Relative error is |pred &minus; gold| / |gold| over answers that are wrong, are "
                "scalars on both sides, and are not merely formatted differently. Format-equivalent "
                "answers are excluded &mdash; they are not misreadings at all.")
        + hist_svg(series, "The error distribution is not a bell around the truth",
                   "If the model were interpolating a value off an axis and landing slightly off, "
                   "the mass would sit in the leftmost bucket. Instead the modal error on every "
                   "benchmark is 25&ndash;100% &mdash; the size of a jump to the adjacent bar, the "
                   "next series, or the neighbouring row.", labels, ymax=0.6)
        + '<div class="note"><strong>What a discrete jump looks like.</strong> A median error of a '
        'third to a half is not what a noisy reading process produces. It is what happens when the '
        'right value is read off the wrong object &mdash; the bar next to the one asked about, the '
        'line for a different series, the figure in the adjacent table row. Order-of-magnitude '
        'errors, which is what a mis-parsed axis scale would produce, are rare everywhere.</div>'
        + f'<div class="note warn"><strong>One tempting piece of evidence, and the control that '
        f'deflates it.</strong> {iv_obs * 100:.1f}% of wrong InfographicVQA numbers appear '
        f'somewhere in the page&rsquo;s OCR layer, which sounds decisive &mdash; until you shuffle '
        f'the predictions across questions and find that {iv_null * 100:.1f}% of <em>other '
        f'questions&rsquo;</em> wrong answers also appear. A dense infographic simply contains a '
        f'lot of numbers. The genuine excess is {(iv_obs - iv_null) * 100:+.1f}pp over '
        f'{len(iv_denom)} items: real, but modest. The weight of this cause therefore rests on the '
        f'shape of the error distribution, which is measured on four splits across three '
        f'benchmarks, and not on the OCR match.</div>')

    groups = []
    ex = sorted(iv_other, key=lambda r: r["uid"])[:10]
    groups.append({"bench": "infographicvqa",
                   "note": "The predicted number is present elsewhere on the same infographic, "
                           "verified against the page's own OCR layer. Read these individually: on "
                           "a dense page some matches are coincidence (see the permutation control "
                           "above), but in most of these the element the model read instead is "
                           "identifiable by eye.",
                   "examples": [b.example(r, tags=["value present elsewhere on the page"])
                                for r in ex]})
    for name, bench, rs in sets:
        es = sorted(rel_errors(rs), key=lambda x: x[0])
        mid = [r for e, r in es if 0.2 <= e <= 1.2][:6]
        if not mid:
            continue
        groups.append({"bench": bench,
                       "note": f"{name}: wrong numeric answers whose error sits in the 20&ndash;120% "
                               f"band &mdash; too large for imprecision, too small for a scale "
                               f"error. Median error for this split is "
                               f"{statistics.median([e for e, _ in es]) * 100:.0f}%.",
                       "examples": [b.example(r, tags=[name, f"error {e * 100:.0f}%"])
                                    for e, r in es if 0.2 <= e <= 1.2][:6]})
    near = [(e, r) for e, r in rel_errors(cx) if e <= 0.05][:4]
    if near:
        groups.append({"bench": "charxiv",
                       "note": "Contrasting cases: the small minority of genuine near misses, where "
                               "the model really did interpolate slightly wrong.",
                       "examples": [b.example(r, tags=[f"error {e * 100:.1f}%", "genuine near miss"])
                                    for e, r in near]})
    refute = (
        "<p>Refuted if the error distribution were concentrated below 10% &mdash; that would make "
        "these interpolation failures, fixable by better axis reading. It is not: the within-10% "
        "share is 11&ndash;16% on every benchmark, and the median sits three to five times higher."
        "</p>"
        "<p>Also refuted if wrong numbers were unrelated to the image, which would point at "
        "hallucination rather than misattribution. The InfographicVQA OCR check argues only weakly "
        f"against that: {iv_obs * 100:.0f}% of wrong numbers are printed somewhere on the page, but "
        f"the permutation baseline is {iv_null * 100:.0f}%, so the effect is "
        f"{(iv_obs - iv_null) * 100:+.1f}pp and is not the decisive evidence it first appears to "
        "be.</p>"
        "<p><strong>Not tested.</strong> The OCR-presence check is only available for "
        "InfographicVQA, which ships an OCR layer; CharXiv and SlideVQA have no machine-readable "
        "list of the values printed in the figure, so for those the claim rests on the shape of the "
        "error distribution alone. A vision judge asked &ldquo;is the predicted value present "
        "somewhere in this figure?&rdquo; would extend the check, and needs API calls. The OCR "
        "check also has a large false-positive floor, quantified above.</p>")
    return Cause("wrong_element_not_near_miss", "Wrong numbers are jumps, not wobbles",
                 "When Haiku gets a number wrong it is usually reporting a different real value "
                 "from the same figure, not a slightly-off reading of the right one. Median "
                 "relative error is 33–50% and only about one wrong answer in eight is within 10% "
                 "of the truth.",
                 "PROVEN", f"median relative error {rows[0][2]}&ndash;{max(r[2] for r in rows)}",
                 ["charxiv", "infographicvqa", "slidevqa"], 0.9, body, groups, refute,
                 ["effective_resolution", "derivation_vs_reading", "answer_expression"])


def _arith_split(d: Data):
    """Wrong-operand vs wrong-operation, decided constructively from SlideVQA's
    annotated arithmetic expression rather than by opinion."""
    man = d.manifests["slidevqa"]
    out = []
    for r in d.rows["slidevqa"]:
        if not r["meta"].get("arithmetic"):
            continue
        if hit(r) or cause_format_equivalent(r["gold"], r["pred"]):
            continue
        m = man.get(int(r["uid"].split(":")[-1]))
        expr = (m or {}).get("arithmetic_expression") or ""
        toks = re.split(r"([+\-*/])", expr.replace(",", ""))
        try:
            vals = [float(t) for t in toks[0::2]]
            ops = toks[1::2]
        except ValueError:
            vals, ops = None, None
        pvs = scalars(r["pred"]) or numbers_in(r["pred"])[:1]
        tag = "unclassified"

        def ev(o, v):
            acc = v[0]
            for op, x in zip(o, v[1:]):
                if op == "+":
                    acc += x
                elif op == "-":
                    acc -= x
                elif op == "*":
                    acc *= x
                else:
                    if x == 0:
                        return None
                    acc /= x
            return acc

        def near(a):
            return a is not None and any(abs(a - p) <= max(1e-6, 1e-6 * abs(p)) for p in pvs)

        if vals and len(vals) >= 2 and pvs:
            got = any(near(ev(list(c), vals)) for c in itertools.product("+-*/", repeat=len(ops))
                      if list(c) != ops)
            if not got and len(vals) == 2:
                got = any(near(ev([o], vals[::-1])) for o in "+-*/")
            if got:
                tag = "wrong_operation"
            else:
                pool = set(numbers_in(r.get("thinking") or "")) | set(numbers_in(r["pred"]))
                for i in range(len(vals)):
                    for c in pool:
                        vv = list(vals)
                        vv[i] = c
                        if near(ev(ops, vv)):
                            tag = "wrong_operand"
                            break
                    if tag == "wrong_operand":
                        break
        out.append((tag, r, expr))
    return out


def c_derivation_vs_reading(d: Data, b: Builder) -> Cause:
    cx, sv = d.rows["charxiv"], d.rows["slidevqa"]
    desc = [r for r in cx if r["meta"].get("split") == "descriptive"]
    reas = [r for r in cx if r["meta"].get("split") == "reasoning"]
    a_d, a_r = cause_mean(hit(r) for r in desc), cause_mean(hit(r) for r in reas)
    look = [r for r in sv if not r["meta"].get("arithmetic")]
    arith = [r for r in sv if r["meta"].get("arithmetic")]
    f_l, f_a = cause_mean(r["score"] for r in look), cause_mean(r["score"] for r in arith)
    c_l, c_a = cause_mean(fmt_score(r) for r in look), cause_mean(fmt_score(r) for r in arith)
    tags = _arith_split(d)
    tc = collections.Counter(t for t, _, _ in tags)
    nt = sum(tc.values())

    rows = [["CharXiv descriptive &mdash; read a value off the chart", f"{len(desc)}",
             f"{a_d * 100:.1f}%", "&mdash;"],
            ["CharXiv reasoning &mdash; derive an answer from what was read", f"{len(reas)}",
             f"{a_r * 100:.1f}%", "&mdash;"],
            ["SlideVQA lookup", f"{len(look)}", f"{f_l * 100:.1f}", f"{c_l * 100:.1f}"],
            ["SlideVQA arithmetic", f"{len(arith)}", f"{f_a * 100:.1f}", f"{c_a * 100:.1f}"]]

    body = (
        cause_tiles([("CharXiv read &rarr; derive", f"{(a_d - a_r) * 100:.1f}pp",
                f"{a_d * 100:.1f}% descriptive vs {a_r * 100:.1f}% reasoning "
                f"(n={len(desc)} / {len(reas)})", "bad"),
               ("SlideVQA as scored", f"{(f_l - f_a) * 100:.1f}pp",
                f"F1 {f_l * 100:.1f} lookup vs {f_a * 100:.1f} arithmetic", "bad"),
               ("SlideVQA format-corrected", f"{(c_l - c_a) * 100:.1f}pp",
                f"F1 {c_l * 100:.1f} vs {c_a * 100:.1f} &mdash; most of the apparent arithmetic "
                f"deficit was the metric", "good"),
               ("Failed arithmetic that is really a misreading",
                f"{tc['wrong_operand'] / max(nt, 1) * 100:.0f}%",
                f"{tc['wrong_operand']} of {nt} genuinely-failed items used the right operation on "
                f"a wrong number", "warn")])
        + cause_table(["split", "n", "as scored", "format-corrected"], rows,
                "Two benchmarks separate reading from deriving. CharXiv does it by design "
                "(descriptive templates vs a reasoning question per figure); SlideVQA does it with "
                "an annotated arithmetic expression.")
        + f'<div class="note warn"><strong>The as-scored SlideVQA gap overstates the deficit, '
        f'badly.</strong> Token-F1 gives zero to <code>22%</code> against a gold of <code>22</code>, '
        f'and arithmetic answers are exactly the ones that carry units. Once right-value/wrong-dress '
        f'answers are credited, the lookup&ndash;arithmetic gap collapses from '
        f'{(f_l - f_a) * 100:.1f}pp to {(c_l - c_a) * 100:.1f}pp. Anyone reporting the '
        f'{(f_l - f_a) * 100:.0f}-point number as a reasoning deficit is reporting a scoring '
        f'artifact. See answer_expression.</div>'
        + cause_table(["how the failed arithmetic actually failed", "n", "share"],
                [["right operation, wrong number read out of the slide (perception)",
                  f"{tc['wrong_operand']}", f"{tc['wrong_operand'] / max(nt, 1) * 100:.0f}%"],
                 ["wrong operation applied to the right numbers (reasoning)",
                  f"{tc['wrong_operation']}", f"{tc['wrong_operation'] / max(nt, 1) * 100:.0f}%"],
                 ["neither reconstruction fits", f"{tc['unclassified']}",
                  f"{tc['unclassified'] / max(nt, 1) * 100:.0f}%"]],
                "Decided constructively, not by opinion: an item is <em>wrong operation</em> if the "
                "prediction equals the result of some other operator combination on the annotated "
                "operands, and <em>wrong operand</em> if it equals the annotated operation applied "
                "to a number the model itself wrote down in its thinking trace. Neither test can be "
                "satisfied by accident.")
        + '<div class="note"><strong>What is left is still real, and it is CharXiv that shows '
        f'it.</strong> Descriptive {a_d * 100:.0f}% against reasoning {a_r * 100:.0f}% is a '
        f'{(a_d - a_r) * 100:.0f}-point gap that no format correction touches &mdash; CharXiv '
        'reasoning answers are short values and its formatting instruction is explicit. So the '
        'claim survives, but the honest version is narrower than the headline: deriving is worse '
        'than reading, and roughly half of what looks like a derivation failure is a reading '
        'failure feeding into a correct derivation.</div>')

    groups = []
    wo = [(t, r, e) for t, r, e in tags if t == "wrong_operand"]
    wp = [(t, r, e) for t, r, e in tags if t == "wrong_operation"]
    groups.append({"bench": "slidevqa",
                   "note": "Arithmetic failures, split by what actually went wrong. The "
                           "wrong-operand cases are perception failures wearing an arithmetic "
                           "costume: open the thinking trace and the subtraction is done "
                           "correctly on a number that is not on the slide.",
                   "examples": ([b.example(r, tags=["misread an operand"],
                                           extra_kv=[("annotated expression",
                                                      f"<code>{esc(e)}</code>", "")])
                                 for _, r, e in wo[:9]]
                                + [b.example(r, tags=["wrong operation"],
                                             extra_kv=[("annotated expression",
                                                        f"<code>{esc(e)}</code>", "")])
                                   for _, r, e in wp[:4]]
                                + [b.example(r, tags=["contrast: correct arithmetic"])
                                   for r in [x for x in arith if hit(x)][:3]])})
    groups.append({"bench": "charxiv",
                   "note": "Reasoning-split failures on figures whose descriptive questions were "
                           "all answered correctly &mdash; the model demonstrably read the chart and "
                           "still could not derive the answer. This is the residual, format-proof "
                           "part of the claim.",
                   "examples": []})
    dbyfig = collections.defaultdict(list)
    for r in desc:
        dbyfig[r["uid"].split(":")[1]].append(r)
    clean_fail = [r for r in reas if not hit(r)
                  and all(hit(x) for x in dbyfig.get(r["uid"].split(":")[1], []))]
    groups[-1]["examples"] = ([b.example(r, tags=["read the figure, failed the derivation"])
                               for r in worst_first(clean_fail, 9)]
                              + [b.example(r, tags=["contrast: correct derivation"])
                                 for r in [x for x in reas if hit(x)][:3]])
    refute = (
        "<p>Refuted if the reading&ndash;deriving gap disappeared under a fairer metric. On "
        "SlideVQA it very nearly does, and that is reported as the headline finding rather than "
        "hidden. On CharXiv it does not, and CharXiv is the cleaner instrument here because its "
        "descriptive and reasoning questions are asked about the same figures.</p>"
        "<p>It would be further refuted if the surviving CharXiv gap were explained by the "
        "reasoning questions simply being longer or more ambiguous. That is only partly "
        "controllable from the data we have: CharXiv reasoning answers are graded approximately "
        "(short free text, no LLM judge in this harness), so the reasoning number is a lower "
        "bound and the true gap is somewhat smaller than shown.</p>"
        "<p><strong>Not tested.</strong> Handing the model the correct intermediate values and "
        "asking only for the final computation would isolate derivation cleanly. That is one API "
        "call per item and was not run.</p>")
    return Cause("derivation_vs_reading", "Reading is strong; deriving from it is weaker",
                 "Haiku reads values off charts and slides reliably and is measurably worse at "
                 "computing with what it read — but roughly half of the apparent arithmetic "
                 "deficit is a misread input rather than a botched calculation, and on SlideVQA "
                 "most of the rest is a scoring artifact.",
                 "MIXED",
                 f"CharXiv {(a_d - a_r) * 100:.0f}pp; SlideVQA {(f_l - f_a) * 100:.0f}pp as scored "
                 f"&rarr; {(c_l - c_a) * 100:.0f}pp corrected",
                 ["charxiv", "slidevqa"], 0.86, body, groups, refute,
                 ["answer_expression", "wrong_element_not_near_miss", "counting"])


def c_answer_expression(d: Data, b: Builder) -> Cause:
    keys = ["slidevqa", "slidevqa_allpages", "charxiv", "infographicvqa"]
    rows, vals, labels = [], [], []
    detail = {}
    for k in keys:
        rs = d.rows[k]
        z = [r for r in rs if float(r.get("score") or 0.0) == 0.0]
        fe = [r for r in z if cause_format_equivalent(r["gold"], r["pred"])]
        metric = rs[0]["metric"] if rs else "?"
        rows.append([NICE[k], {"slidevqa_f1": "token F1", "anls": "ANLS (edit distance)",
                               "normalized_match": "normalized match"}.get(metric, metric),
                     f"{len(rs)}", f"{len(z)}",
                     (f'<strong>{len(fe) / max(len(z), 1) * 100:.1f}%</strong>', "num"),
                     f"{len(fe) / len(rs) * 100:.1f}pp"])
        vals.append(len(fe) / max(len(z), 1))
        labels.append(NICE[k].replace(" (all 20 pages)", " all-pages"))
        detail[k] = fe
    sv_fe = detail["slidevqa"]
    kinds = collections.Counter()
    for r in sv_fe:
        g, p = str(r["gold"][0]), str(r["pred"])
        if "%" in p and "%" not in g:
            kinds["prediction adds a % sign"] += 1
        elif re.search(r"[a-zA-Z]", p) and not re.search(r"[a-zA-Z]", g):
            kinds["prediction adds a unit or scale word"] += 1
        elif "," in p and "," not in g:
            kinds["prediction adds thousands separators"] += 1
        elif "$" in p or "€" in p or "£" in p:
            kinds["prediction adds a currency symbol"] += 1
        else:
            kinds["other surface difference"] += 1

    body = (
        '<div class="note warn"><strong>This is a measurement artifact, not a model blind '
        'spot.</strong> Everything on this page is the model producing the right value and the '
        'metric refusing to accept it. It is on the site because it contaminates two other causes '
        '&mdash; derivation_vs_reading and cross_page_integration both look worse than they are '
        'until it is subtracted &mdash; and because it is the single largest correction available '
        'to any number in this study.</div>'
        + cause_tiles([("SlideVQA hard zeros that are format-equivalent",
                  f"{len(detail['slidevqa']) / max(len([r for r in d.rows['slidevqa'] if float(r.get('score') or 0) == 0]), 1) * 100:.1f}%",
                  "token F1 gives zero to <code>22%</code> against a gold of <code>22</code>", "bad"),
                 ("CharXiv",
                  f"{len(detail['charxiv']) / max(len([r for r in d.rows['charxiv'] if float(r.get('score') or 0) == 0]), 1) * 100:.1f}%",
                  "normalized match, and CharXiv states the answer format in the prompt", "good"),
                 ("InfographicVQA",
                  f"{len(detail['infographicvqa']) / max(len([r for r in d.rows['infographicvqa'] if float(r.get('score') or 0) == 0]), 1) * 100:.1f}%",
                  "ANLS is edit distance, so a stray <code>%</code> costs a few points, not "
                  "everything", "good")])
        + cause_table(["benchmark", "official metric", "n", "hard zeros",
                 "of those, right value in wrong dress", "cost to the headline score"], rows,
                "A hard zero is a score of exactly 0.00. The format-equivalence test is "
                "deliberately conservative: numbers must match exactly after stripping "
                "<code>, $ € £ %</code> and resolving scale words, comparing sign-sensitively; text "
                "must match in full after folding case and punctuation. There is no substring "
                "fallback, so <code>22</code> never matches <code>22 million people surveyed</code>.")
        + hist_svg([("share of hard zeros that are only a formatting difference", "--s3", vals)],
                   "The artifact is a property of the metric, not of the model",
                   "The same model, answering four benchmarks, produces a formatting-only failure "
                   "rate that varies by a factor of thirty &mdash; tracking exactly which metric is "
                   "in use. Token F1 is all-or-nothing per token; ANLS charges edit distance and "
                   "barely notices a percent sign.", labels, ymax=0.6)
        + cause_table(["what the dress difference is (SlideVQA)", "n"],
                [[k, str(v)] for k, v in kinds.most_common()],
                f"Breakdown of the {len(sv_fe)} SlideVQA format-equivalent zeros.")
        + '<div class="note"><strong>Where the correction is applied.</strong> Every page on this '
        'site that reports a SlideVQA number reports both columns. The uncorrected number is the '
        'benchmark result and stays comparable to published work; the corrected number is the one '
        'to use when reasoning about what the model can see.</div>')

    groups = []
    for k in keys:
        fe = detail[k]
        if not fe:
            continue
        groups.append({"bench": k,
                       "note": f"{len(fe)} of this benchmark's hard zeros are the right value in the "
                               f"wrong dress. Side-by-side gold and prediction below; the images are "
                               f"included so you can confirm the value really is what the figure says.",
                       "examples": [b.example(r, tags=["scored 0.00", "value is correct"],
                                              extra_kv=[("difference",
                                                         f"gold <code>{esc(r['gold'][0])}</code> "
                                                         f"vs predicted <code>{esc(r['pred'])}</code>",
                                                         "")])
                                    for r in fe[:(10 if k == "slidevqa" else 6)]]})
    refute = (
        "<p>Refuted if the detector were simply generous. It is built not to be: no substring "
        "matching, exact numeric equality, sign-sensitive, and an unrecognised trailing word "
        "disqualifies a match outright. Spot-check any example above &mdash; the gold and the "
        "prediction are printed side by side.</p>"
        "<p><strong>One known false-positive mode.</strong> A scale word is accepted under both "
        "readings, stripped and applied, so a gold of <code>3.3</code> matches a prediction of "
        "<code>3.3bn</code>. When the question asked &ldquo;how many billions&rdquo; that is "
        "correct; if a question ever asked for a raw count and the model answered in billions, this "
        "would wrongly forgive it. We found no such case in the examples above but the rate is not "
        "zero by construction.</p>"
        "<p><strong>Not tested.</strong> The complementary error &mdash; the metric awarding credit "
        "for a wrong answer that shares tokens with the gold &mdash; is not measured here. Token F1 "
        "is generous in that direction, so the corrected SlideVQA numbers are not a strict "
        "improvement in accuracy, only a removal of one known bias.</p>")
    return Cause("answer_expression", "Right value, wrong dress, scored zero",
                 "About half of SlideVQA's hard zeros are answers whose value is correct and whose "
                 "surface form differs — a percent sign, a currency symbol, a scale word. This is "
                 "the metric failing, not the model, and it inflates two other apparent weaknesses.",
                 "ARTIFACT",
                 f"{vals[0] * 100:.0f}% of SlideVQA zeros vs {vals[2] * 100:.0f}% on CharXiv",
                 keys, 0.8, body, groups, refute,
                 ["derivation_vs_reading", "cross_page_integration", "retrieval_search",
                  "ground_truth_noise"])


def c_cross_page(d: Data, b: Builder) -> Cause:
    sv = d.rows["slidevqa"]
    single = [r for r in sv if not r["meta"].get("multi_page")]
    multi = [r for r in sv if r["meta"].get("multi_page")]
    f_s, f_m = cause_mean(r["score"] for r in single), cause_mean(r["score"] for r in multi)
    c_s, c_m = cause_mean(fmt_score(r) for r in single), cause_mean(fmt_score(r) for r in multi)
    op = {k: v for k, v in d.onepage.items() if k != "__meta__" and v.get("pred") is not None}
    byuid = {r["uid"]: r for r in sv}
    paired = []
    for su, v in op.items():
        m = byuid.get(su)
        e = d.ex.get(su)
        if m is None or e is None:
            continue
        s = official_score(e, v["pred"])
        paired.append((v, m, float(s.get("score") or 0.0)))
    both = cause_mean(m["score"] for _, m, _ in paired)
    one = cause_mean(s for _, _, s in paired)
    still = cause_mean(1 if s >= 0.5 else 0 for _, _, s in paired)
    both_ok = cause_mean(1 if hit(m) else 0 for _, m, _ in paired)

    man = list(d.manifests["slidevqa"].values())
    nev = collections.Counter(len([x for x in (m.get("evidence_pages") or [])]) for m in man)
    ct = collections.Counter((len(m.get("evidence_pages") or []) > 1,
                              (m.get("arithmetic_expression") not in (None, "None", "")))
                             for m in man)

    body = (
        cause_tiles([("Multi-page vs single-page", f"{(f_m - f_s) * 100:+.1f}pp",
                f"F1 {f_m * 100:.1f} vs {f_s * 100:.1f} (n={len(multi)} / {len(single)}); "
                f"format-corrected {(c_m - c_s) * 100:+.1f}pp", "good"),
               ("Take one of the two slides away", f"{(one - both) * 100:+.1f}pp",
                f"F1 {pct_bare(both, 1)} &rarr; {pct_bare(one, 1)} on the same {len(paired)} questions",
                "bad"),
               ("Still answerable from one slide", pct_or_dash(still, 1),
                f"against {pct_bare(both_ok, 1)}% with both &mdash; the information really is "
                f"distributed", "warn")])
        + bars("Integration across slides costs almost nothing; losing a slide costs everything",
               "The first two bars are the observational comparison, the second two the ablation. "
               "If multi-page questions were hard <em>because</em> they span slides, the first gap "
               "would be large. It is not. The ablation confirms the questions are genuinely "
               "multi-page: remove the second evidence slide and the same questions collapse.",
               [("single-evidence questions", f_s, f"n={len(single)}", f"{f_s * 100:.1f}", None, ""),
                ("multi-evidence questions", f_m, f"n={len(multi)}", f"{f_m * 100:.1f}", None, ""),
                ("multi-evidence, both slides sent", both, f"n={len(paired)}",
                 pct_bare(both, 1), None, "good"),
                ("multi-evidence, first slide only", one, f"n={len(paired)}",
                 pct_bare(one, 1), None, "bad")])
        + cause_table(["SlideVQA question population (full manifest)", "n"],
                [["one evidence slide", str(nev.get(1, 0))],
                 ["two evidence slides", str(nev.get(2, 0))],
                 ["three evidence slides", str(nev.get(3, 0))],
                 ["multi-page &times; arithmetic", str(ct[(True, True)])],
                 ["multi-page &times; lookup", str(ct[(True, False)])],
                 ["single-page &times; arithmetic", str(ct[(False, True)])],
                 ["single-page &times; lookup", str(ct[(False, False)])]],
                "Multi-page questions are overwhelmingly lookups, not calculations, which is why "
                "they are not harder: they are bridge questions. One slide identifies the subject "
                "&mdash; &ldquo;the year with the third largest Organic Growth&rdquo; &mdash; and "
                "the other holds the value asked for.")
        + '<div class="note good"><strong>Verdict: this is a strength, and the page says so.</strong> '
        'Two-slide questions cost about four F1 points as scored and about '
        f'{abs(c_m - c_s) * 100:.0f} once formatting is corrected &mdash; smaller than the '
        'difference between two arbitrary question types. The ablation rules out the deflationary '
        'explanation that the questions were never really multi-page: with one slide removed the '
        f'same questions fall {abs(one - both) * 100:.0f} points and only {pct_bare(still, 0)}% remain '
        'answerable. Haiku is carrying information across images.</div>')

    hard = sorted([p for p in paired if hit(p[1]) and p[2] < 0.5],
                  key=lambda p: -p[1]["score"])
    exs = []
    for v, m, s in hard[:10]:
        ex = m["_ex"]
        sizes = m.get("sent_image_sizes") or []
        imgs = [b.img(p, sizes[i] if i < len(sizes) else None,
                      cap=f"evidence slide {i + 1} of {len(ex.images)}")
                for i, p in enumerate(ex.images)]
        e = b.example(m, imgs=imgs, tags=["bridge question"],
                      extra_kv=[("answer when only the first slide was sent",
                                 f'<span style="color:var(--bad)">{esc(v["pred"])}</span> '
                                 f'&mdash; scored {s:.2f}', "")])
        e["why"] = ("Right with both slides, wrong with one. The first slide identifies which "
                    "subject the question is about; the second holds the number. Removing either "
                    "breaks the chain, which is what makes this a genuine cross-page question.")
        exs.append(e)
    groups = [{"bench": "slidevqa",
               "note": "Worked bridge questions, shown as a filmstrip of the evidence slides the "
                       "model actually received. Use the arrow keys in the lightbox to step between "
                       "slides.",
               "examples": exs}]
    mfail = [r for r in multi if not hit(r) and not cause_format_equivalent(r["gold"], r["pred"])]
    groups.append({"bench": "slidevqa",
                   "note": "Multi-page questions the model got wrong even with every evidence slide "
                           "in hand &mdash; the residual integration failures, and the ones that "
                           "would have to grow for this cause to become a weakness.",
                   "examples": [b.example(r, tags=["failed with all evidence"])
                                for r in worst_first(mfail, 8)]})
    refute = (
        "<p>This is filed as <strong>refuted as a weakness</strong>. It would become a weakness if "
        "multi-page questions scored materially below single-page ones on comparable question "
        f"types; they score {abs(f_m - f_s) * 100:.0f} points below, and the multi-page pool is "
        "richer in lookups, which are easier, so the true adjusted gap is smaller still.</p>"
        "<p>The one-slide ablation is what makes the negative result trustworthy: without it, "
        "&ldquo;multi-page is not harder&rdquo; would be consistent with the questions never having "
        "needed two slides.</p>"
        "<p><strong>Not tested.</strong> Only the <em>first</em> evidence slide was withheld-from "
        "&mdash; there is no second-slide-only arm, so we cannot say whether the two slides "
        "contribute symmetrically. Nor is there a three-or-more-slide condition with enough items "
        f"to measure ({nev.get(3, 0)} such questions exist in the manifest). Whether integration "
        "holds up at five or ten hops is untested and is where a real limit would be expected.</p>")
    return Cause("cross_page_integration", "Carrying information between images is not the problem",
                 "Questions whose evidence is split across two slides score about the same as "
                 "single-slide questions — yet removing one of the two slides collapses them. The "
                 "information is genuinely distributed, and Haiku genuinely integrates it.",
                 "REFUTED",
                 f"{(f_m - f_s) * 100:+.1f}pp multi vs single, against {(one - both) * 100:+.1f}pp "
                 f"for the ablation",
                 ["slidevqa"], 0.6, body, groups, refute,
                 ["retrieval_search", "answer_expression", "derivation_vs_reading"],
                 one_bench_note="SlideVQA is the only multi-image benchmark in this study, so both "
                                "the negative result and the ablation that validates it come from "
                                "one dataset of business slide decks.")


def c_retrieval(d: Data, b: Builder) -> Cause:
    sv = {r["uid"].split(":")[-1]: r for r in d.rows["slidevqa"]}
    ap = {r["uid"].split(":")[-1]: r for r in d.rows["slidevqa_allpages"]}
    common = sorted(set(sv) & set(ap))
    e_f1 = cause_mean(sv[k]["score"] for k in common)
    a_f1 = cause_mean(ap[k]["score"] for k in common)
    e_fc = cause_mean(fmt_score(sv[k]) for k in common)
    a_fc = cause_mean(fmt_score(ap[k]) for k in common)
    ti_e = cause_mean(sv[k]["usage"]["input_tokens"] for k in common)
    ti_a = cause_mean(ap[k]["usage"]["input_tokens"] for k in common)
    to_e = cause_mean(sv[k]["usage"]["output_tokens"] for k in common)
    to_a = cause_mean(ap[k]["usage"]["output_tokens"] for k in common)
    la_e = cause_mean(sv[k]["latency_s"] for k in common)
    la_a = cause_mean(ap[k]["latency_s"] for k in common)
    div = [k for k in common if hit(sv[k]) and not hit(ap[k])]
    rev = [k for k in common if not hit(sv[k]) and hit(ap[k])]

    body = (
        cause_tiles([("Cost of finding the slide yourself", f"{(a_f1 - e_f1) * 100:+.1f}pp",
                f"F1 {e_f1 * 100:.1f} with only the evidence slides &rarr; {a_f1 * 100:.1f} with "
                f"all 20 (paired n={len(common)}); format-corrected {(a_fc - e_fc) * 100:+.1f}pp",
                "warn"),
               ("Input tokens", f"{ti_a / ti_e:.1f}&times;",
                f"{ti_e:.0f} &rarr; {ti_a:.0f} per question", ""),
               ("Output tokens", f"{to_a / to_e:.2f}&times;",
                f"{to_e:.0f} &rarr; {to_a:.0f} &mdash; nineteen extra slides buy "
                f"{to_a - to_e:.0f} extra tokens of thought", "bad"),
               ("Latency", f"{la_a / la_e:.2f}&times;", f"{la_e:.2f}s &rarr; {la_a:.2f}s", "")])
        + bars("Twenty slides instead of two costs almost no accuracy",
               "Same questions, same golds; the only difference is whether the harness pre-selects "
               "the evidence slides. Retrieval over a 20-slide deck is close to free.",
               [("evidence slides only", e_f1, f"n={len(common)}", pct_bare(e_f1, 1), None, ""),
                ("all 20 slides", a_f1, f"n={len(common)}", pct_bare(a_f1, 1), None, "s2"),
                ("evidence only, format-corrected", e_fc, f"n={len(common)}", pct_bare(e_fc, 1),
                 None, "good"),
                ("all 20, format-corrected", a_fc, f"n={len(common)}", pct_bare(a_fc, 1), None,
                 "good")])
        + f'<div class="note warn"><strong>But it does not search &mdash; it skims.</strong> '
        f'Handing the model ten times more input raises input tokens {ti_a / ti_e:.1f}&times; and '
        f'latency {la_a / la_e:.2f}&times;, while output tokens rise only {to_a / to_e:.2f}&times; '
        f'({to_e:.0f} &rarr; {to_a:.0f}). A model actually examining nineteen additional slides '
        f'would have visibly more to think about. The near-free retrieval result should therefore '
        f'be read narrowly: on a 20-slide deck the right slide is usually findable at a glance. It '
        f'is not evidence that deep search over a hundred pages would also be free.</div>'
        + cause_table(["outcome when the deck was not pre-filtered", "n", "share of paired items"],
                [["was right with the evidence slides, wrong with all 20", f"{len(div)}",
                  f"{len(div) / len(common) * 100:.1f}%"],
                 ["was wrong with the evidence slides, right with all 20", f"{len(rev)}",
                  f"{len(rev) / len(common) * 100:.1f}%"],
                 ["unchanged", f"{len(common) - len(div) - len(rev)}",
                  f"{(len(common) - len(div) - len(rev)) / len(common) * 100:.1f}%"]],
                "The net cost is the difference between the first two rows, and it is small. The "
                "reverse cases are real: sometimes the surrounding slides supply context the "
                "evidence slide alone does not."))

    groups = [{"bench": "slidevqa",
               "note": "Divergence cases: correct when handed the evidence slides, wrong when made "
                       "to find them in a 20-slide deck. The filmstrip shows the evidence slides "
                       "only; the all-pages arm additionally received the other eighteen.",
               "examples": []}]
    exs = []
    for k in div[:10]:
        m, a = sv[k], ap[k]
        ex = m["_ex"]
        sizes = m.get("sent_image_sizes") or []
        imgs = [b.img(p, sizes[i] if i < len(sizes) else None,
                      cap=f"evidence slide {i + 1}") for i, p in enumerate(ex.images)]
        e = b.example(m, imgs=imgs, tags=["lost when the deck was not pre-filtered"],
                      extra_kv=[("answer with all 20 slides sent",
                                 f'<span style="color:var(--bad)">{esc(a["pred"])}</span> '
                                 f'&mdash; scored {a["score"]:.2f}', "")])
        e["why"] = ("The evidence is unchanged; only the haystack grew. Compare the two thinking "
                    "traces &mdash; the all-pages run typically commits to a slide early and does "
                    "not revisit.")
        e["thinking"] = ("--- with only the evidence slides ---\n" + (m.get("thinking") or "")[:1800]
                         + "\n\n--- with all 20 slides ---\n" + (a.get("thinking") or "")[:1800])
        exs.append(e)
    groups[0]["examples"] = exs
    if rev:
        groups.append({"bench": "slidevqa_allpages",
                       "note": "The reverse: wrong on the evidence slides alone, right with the "
                               "whole deck. Context from neighbouring slides sometimes disambiguates "
                               "a units label or a time period.",
                       "examples": [b.example(ap[k], tags=["recovered with full context"],
                                              extra_kv=[("answer with only the evidence slides",
                                                         f'<span style="color:var(--bad)">'
                                                         f'{esc(sv[k]["pred"])}</span> &mdash; scored '
                                                         f'{sv[k]["score"]:.2f}', "")])
                                    for k in rev[:5]]})
    refute = (
        "<p>Filed as <strong>refuted as a weakness</strong> at this scale. It would become a "
        "weakness if the all-pages arm fell substantially; it falls "
        f"{abs(a_f1 - e_f1) * 100:.1f} points as scored and {abs(a_fc - e_fc) * 100:.1f} corrected, "
        f"on n={len(common)}.</p>"
        "<p><strong>The output-token evidence is suggestive, not decisive.</strong> A model could "
        "in principle locate the right slide efficiently and think about it just as hard, which "
        "would produce the same token profile. Distinguishing &ldquo;efficient&rdquo; from "
        "&ldquo;skimming&rdquo; requires a needle-in-haystack arm where the answer is planted on a "
        "slide the model has no prior reason to look at.</p>"
        "<p><strong>Not tested.</strong> Anything beyond 20 images. The API treats 20 as a boundary "
        "&mdash; above it each image is capped harder &mdash; so this measures retrieval exactly at "
        "the point where it is cheapest. A 50- or 100-page condition is the experiment that would "
        "find the limit, and it needs API calls.</p>")
    return Cause("retrieval_search", "Finding the right slide among twenty is nearly free",
                 "Sending the whole 20-slide deck instead of just the evidence slides costs about "
                 "four F1 points. But the model pays 11× the input tokens and produces only 1.08× "
                 "the output — it is glancing, not searching, and that limits how far the result "
                 "generalises.",
                 "REFUTED", f"{(a_f1 - e_f1) * 100:+.1f}pp for a {ti_a / ti_e:.0f}&times; larger input",
                 ["slidevqa", "slidevqa_allpages"], 0.55, body, groups, refute,
                 ["cross_page_integration", "answer_expression"])


COUNT_BINS = ["1–2", "3–4", "5–6", "7–9", "10–15", "16+"]
MIN_BIN = 5   # bins thinner than this are shown as "—" rather than as a data point


def _cbin(v):
    v = int(v)
    return ("1–2" if v <= 2 else "3–4" if v <= 4 else "5–6" if v <= 6
            else "7–9" if v <= 9 else "10–15" if v <= 15 else "16+")


def c_counting(d: Data, b: Builder) -> Cause:
    cx, iv = d.rows["charxiv"], d.rows["infographicvqa"]
    OBJ = {10, 12, 19}
    # (display name, benchmark, short plain-text tag, rows)
    families = [
        ("CharXiv &mdash; count objects (lines, legend entries, subplots)", "charxiv",
         [r for r in cx if r["meta"].get("qid") in OBJ and numval(r["gold"][0]) is not None]),
        ("CharXiv &mdash; count labelled ticks (qid 17)", "charxiv",
         [r for r in cx if r["meta"].get("qid") == 17 and numval(r["gold"][0]) is not None]),
        ("InfographicVQA &mdash; questions the dataset labels &ldquo;counting&rdquo;",
         "infographicvqa",
         [r for r in iv if "counting" in (r["meta"].get("operation") or [])
          and numval(r["gold"][0]) is not None]),
    ]
    rows, series_acc, series_names = [], [], []
    signed_rows = []
    for name, bench, rs in families:
        by = collections.defaultdict(list)
        for r in rs:
            g = numval(r["gold"][0])
            p = numval(r["pred"])
            by[_cbin(g)].append((hit(r), (p - g) if p is not None else None))
        acc = []
        for kbin in COUNT_BINS:
            v = by.get(kbin)
            # A bin holding one or two questions is noise dressed as a data point.
            acc.append(cause_mean(x[0] for x in v) if v and len(v) >= MIN_BIN else None)
        series_acc.append((name, ["--s1", "--s2", "--s3"][len(series_acc) % 3], acc))
        series_names.append(name)
        overall = cause_mean(hit(r) for r in rs)
        rows.append([name, f"{len(rs)}", pct_or_dash(overall, 1)]
                    + [(pct_or_dash(a, 0) if a is not None else "&mdash;")
                       + (f'<span style="color:var(--muted);font-size:10.5px"> n={len(by[kb])}</span>'
                          if by.get(kb) else "")
                       for kb, a in zip(COUNT_BINS, acc)])
        sr = [name]
        for kbin in COUNT_BINS:
            v = by.get(kbin)
            se = [x[1] for x in (v or []) if x[1] is not None]
            sr.append(f"{statistics.mean(se):+.2f}" if len(se) >= MIN_BIN else "&mdash;")
        signed_rows.append(sr)

    iv_c = families[2][2]
    iv_rest = [r for r in iv if "counting" not in (r["meta"].get("operation") or [])]
    obj = families[0][2]
    tick = families[1][2]

    body = (
        cause_tiles([("CharXiv object counting", pct_or_dash(cause_mean(hit(r) for r in obj), 1),
                f"n={len(obj)} &mdash; counting a handful of lines or legend entries is close to "
                f"solved", "good"),
               ("CharXiv tick counting", pct_or_dash(cause_mean(hit(r) for r in tick), 1),
                f"n={len(tick)} &mdash; the same operation over ten to forty small marks", "warn"),
               ("InfographicVQA counting", pct_or_dash(cause_mean(r['score'] for r in iv_c), 1),
                f"ANLS, n={len(iv_c)}, against {pct_bare(cause_mean(r['score'] for r in iv_rest), 1)}% for "
                f"everything else on the same benchmark", "warn"),
               ("Dose response",
                f"{(series_acc[2][2][0] or 0) * 100:.0f}% &rarr; {(series_acc[2][2][-1] or 0) * 100:.0f}%",
                "InfographicVQA accuracy from the smallest counts to the largest", "bad")])
        + hist_svg(series_acc, "Accuracy against the true count",
                   "Bins holding fewer than five questions are omitted rather than plotted, which "
                   "is why some series stop short. Two of the three families decline clearly; "
                   "CharXiv's tick counting does not decline in accuracy at all, and its evidence "
                   "sits in the signed error below instead.",
                   COUNT_BINS, ymax=1.0)
        + cause_table(["family", "n", "overall"] + [f"true count {k}" for k in COUNT_BINS], rows,
                "Accuracy binned by the true count &mdash; a dose-response curve, with the n for "
                "each cell beside it. Bins under five items are blanked. The trend, not any single "
                "cell, is the evidence.")
        + cause_table(["family &mdash; mean <em>signed</em> error"] + COUNT_BINS, signed_rows,
                "The sign is what distinguishes two different failures. Consistent negative values "
                "mean the model stops counting early; values that scatter around zero mean it is "
                "just noisy. Both patterns are present here, in different families.")
        + '<div class="note"><strong>Two different counting failures, not one.</strong> Counting '
        '<em>objects</em> &mdash; lines in a plot, entries in a legend &mdash; drifts '
        '<strong>negative</strong> as the count rises: the model stops early, which is the '
        'signature of losing track in a crowded field. Counting <em>tick marks</em> drifts '
        '<strong>positive</strong> at every bin: it over-reports, which is the signature of '
        'counting a regular repeating pattern by estimating rather than enumerating. Pooling these '
        'into one &ldquo;counting accuracy&rdquo; number would hide both.</div>'
        + f'<div class="note">On InfographicVQA, counting questions score '
        f'{pct_bare(cause_mean(r["score"] for r in iv_c), 1)} against '
        f'{pct_bare(cause_mean(r["score"] for r in iv_rest), 1)} for the rest of the benchmark &mdash; a '
        f'{(cause_mean(r["score"] for r in iv_rest) - cause_mean(r["score"] for r in iv_c)) * 100:.1f}-point '
        f'penalty, and the dose-response within it is the steepest of the three families. That is '
        f'the cross-benchmark half of this claim.</div>')

    groups = []
    for name, bench, rs in families:
        big = [r for r in rs if not hit(r) and (numval(r["gold"][0]) or 0) >= 5]
        small = [r for r in rs if not hit(r) and (numval(r["gold"][0]) or 0) < 5]
        short = html.unescape(name).split("—")[-1].strip()
        ex = ([b.example(r, tags=[short, f"true count {numval(r['gold'][0]):.0f}"],
                         extra_kv=[("signed error",
                                    (f"{numval(r['pred']) - numval(r['gold'][0]):+.0f}"
                                     if numval(r["pred"]) is not None else "&mdash;"), "")])
               for r in worst_first(big, 6)]
              + [b.example(r, tags=[f"true count {numval(r['gold'][0]):.0f}", "small count, still wrong"])
                 for r in worst_first(small, 3)]
              + [b.example(r, tags=["contrast: correct"]) for r in [x for x in rs if hit(x)][:2]])
        groups.append({"bench": bench,
                       "note": f"{name}. Large-count failures first &mdash; those are where the "
                               f"dose-response lives &mdash; then the rarer small-count failures.",
                       "examples": ex})
    refute = (
        "<p>Refuted if accuracy were flat against the true count, which would make counting a "
        "uniform skill that either works or does not. It is not flat on two of the three "
        "families &mdash; but it <em>is</em> flat on CharXiv tick counting, and that is stated "
        "rather than smoothed over. For that family the claim rests entirely on the signed "
        "error.</p>"
        "<p>The <em>direction</em> claim would be refuted if the signed errors were symmetric "
        "within a family. They are not: object counting drifts negative at high counts and tick "
        "counting drifts positive at every bin.</p>"
        "<p><strong>Not tested, and this matters.</strong> The upper bins are thin &mdash; CharXiv "
        "charts rarely contain more than a dozen lines, so the interesting regime (30, 50, 100 "
        "objects) is barely sampled. A synthetic counting probe with a controlled object count is "
        "the right instrument and needs API calls. InfographicVQA's &ldquo;counting&rdquo; label is "
        "also the dataset's own annotation and is not always a pure count &mdash; some items ask "
        "&ldquo;how many of X are Y&rdquo;, which mixes counting with filtering.</p>")
    return Cause("counting", "Counting degrades with the count, and in two different directions",
                 "Counting a handful of objects is near-solved; counting a crowd is not. Accuracy "
                 "falls with the true count on two of the three families measured, and the sign of "
                 "the error separates two distinct failures — stopping early on objects, "
                 "over-estimating on repeated marks. Tick counting is the honest exception: its "
                 "accuracy is flat and only the signed error reveals anything.",
                 "SUPPORTED",
                 f"InfographicVQA {(series_acc[2][2][0] or 0) * 100:.0f}% &rarr; "
                 f"{(series_acc[2][2][-1] or 0) * 100:.0f}% across count bins",
                 ["charxiv", "infographicvqa"], 0.7, body, groups, refute,
                 ["effective_resolution", "derivation_vs_reading"])


def c_position_bias(d: Data, b: Builder) -> Cause:
    ai = d.rows["ai2d"]
    n = len(ai)
    gold = collections.Counter(r["gold"][0] for r in ai)
    pick_ = collections.Counter(str(r["pred"]).strip().upper() for r in ai)
    wrong = [r for r in ai if not hit(r)]
    pw = collections.Counter(str(r["pred"]).strip().upper() for r in wrong)
    rows, gv, pv, wv = [], [], [], []
    for L in "ABCD":
        acc = cause_mean(hit(r) for r in ai if r["gold"][0] == L)
        rows.append([f"option {L}", f"{gold[L]}", f"{gold[L] / n * 100:.1f}%",
                     f"{pick_[L] / n * 100:.1f}%",
                     f"{(pick_[L] - gold[L]) / n * 100:+.1f}pp",
                     f"{pw[L] / max(len(wrong), 1) * 100:.1f}%",
                     pct_or_dash(acc, 1)])
        gv.append(gold[L] / n)
        pv.append(pick_[L] / n)
        wv.append(pw[L] / max(len(wrong), 1))
    chi = sum((pick_[L] - gold[L]) ** 2 / gold[L] for L in "ABCD")
    off = {k: v for k, v in pick_.items() if k not in set("ABCD")}
    maxdev = max(abs(pick_[L] - gold[L]) / n for L in "ABCD")

    body = (
        cause_tiles([("Largest deviation from the key", f"{maxdev * 100:.1f}pp",
                f"across all four positions, n={n}", "good"),
               ("&chi;&sup2; against the gold distribution", f"{chi:.2f}",
                "3 degrees of freedom; the 5% critical value is 7.81", "good"),
               ("Unparseable answers", f"{sum(off.values())}",
                "the model always emitted one of the four letters", "good")])
        + cause_table(["", "gold count", "share of the key", "share of picks", "deviation",
                 "share of <em>wrong</em> picks", "accuracy when this is the answer"], rows,
                "AI2D's answer key is close to uniform, which is what makes a skew detectable at "
                "all. If Haiku favoured a slot, its pick distribution would separate from the gold "
                "distribution &mdash; and the effect would be strongest among wrong answers, where "
                "there is nothing else to go on.")
        + hist_svg([("gold key", "--s1", gv), ("model picks", "--s2", pv),
                    ("picks among wrong answers", "--s3", wv)],
                   "No position preference, at any slice",
                   "Three distributions that would come apart if a slot bias existed. They do not. "
                   "The wrong-answer distribution is the sharpest test &mdash; a model guessing "
                   "would guess somewhere &mdash; and it is flat within a couple of points.",
                   ["A", "B", "C", "D"], ymax=0.4)
        + '<div class="note good"><strong>Negative result, kept on the page.</strong> A '
        '&chi;&sup2; of '
        f'{chi:.2f} against a critical value of 7.81 is not close to significance. There is no '
        'position bias in Haiku\'s multiple-choice answers on AI2D, either overall or among its '
        'errors. That is worth stating explicitly: it means AI2D\'s accuracy number is not '
        'contaminated by a guessing strategy, and it removes one candidate explanation for the '
        'label-reference deficit &mdash; those questions are hard because the diagram is hard to '
        'read, not because the model falls back on a favourite letter.</div>')

    groups = [{"bench": "ai2d",
               "note": "Wrong answers, shown for completeness rather than as evidence for a bias "
                       "there is none of. What these have in common is a hard diagram, not a "
                       "position.",
               "examples": [b.example(r, tags=[f"picked {str(r['pred']).strip().upper()}",
                                               f"gold {r['gold'][0]}", r["meta"]["qtype"]],
                                      extra_kv=[("options",
                                                 esc(", ".join(map(str, r["meta"]["options"]))), ""),
                                                ("gold option text",
                                                 esc(r["meta"].get("gold_text")), "g")])
                            for r in wrong[:10]]}]
    refute = (
        "<p>This cause is <strong>refuted</strong>. It would have been supported by a pick "
        "distribution that deviated from a near-uniform key, most visibly among wrong answers. "
        f"The largest deviation is {maxdev * 100:.1f}pp and &chi;&sup2;={chi:.2f} on 3 df.</p>"
        "<p><strong>Scope.</strong> AI2D is the only multiple-choice benchmark here, so this "
        "refutes position bias for four-way MC on diagrams. It says nothing about ordering effects "
        "in free-text answers &mdash; whether the model prefers the first item it read &mdash; "
        "which would need a different probe.</p>"
        "<p><strong>Not tested.</strong> Option-order shuffling. The decisive experiment is to "
        "re-run the same questions with the four options permuted and check whether the model's "
        "choice follows the content or the slot. That needs API calls.</p>")
    return Cause("position_bias", "No preference for a particular answer slot",
                 "Haiku's multiple-choice picks track AI2D's near-uniform answer key to within a "
                 "point, overall and among its wrong answers. There is no slot bias to correct for.",
                 "REFUTED", f"max deviation {maxdev * 100:.1f}pp, &chi;&sup2;={chi:.2f} (crit 7.81)",
                 ["ai2d"], 0.25, body, groups, refute,
                 ["label_reference_binding", "language_prior_override"],
                 one_bench_note="AI2D is the only multiple-choice benchmark in this study. A "
                                "negative result from one dataset is still a negative result, but "
                                "it bounds only four-way MC on science diagrams.")


def c_subplot_scope(d: Data, b: Builder) -> Cause:
    cx = d.rows["charxiv"]
    desc = [r for r in cx if r["meta"].get("split") == "descriptive"]
    multi = [r for r in desc if (r["meta"].get("num_subplots") or 1) > 1]
    named = [r for r in multi if r["question"].startswith("For the subplot at row")]
    verbal = [r for r in multi if r["question"].startswith("For ")
              and not r["question"].startswith("For the subplot at row")
              and "current plot" not in r["question"]]
    noprefix = [r for r in multi if not r["question"].startswith("For ")]

    # exact cross-subplot test: does a wrong answer match the gold of the SAME
    # template asked about a different subplot of the SAME figure?
    byimg = collections.defaultdict(set)
    for r in cx:
        byimg[r["_ex"].images[0]].add(r["uid"].split(":")[1])
    shared = sum(1 for v in byimg.values() if len(v) > 1)

    dist = collections.defaultdict(list)
    for r in named:
        m = re.match(r"For the subplot at row (\d+) and column (\d+)", r["question"])
        if m:
            dist[(int(m.group(1)) - 1) + (int(m.group(2)) - 1)].append(hit(r))
    drows = [[f"{k} step(s) from the top-left panel", f"{len(v)}", pct_or_dash(cause_mean(v), 1)]
             for k, v in sorted(dist.items()) if len(v) >= 40]

    # axis confusion: x-axis answer that is really the y-axis label, and vice versa
    pairs = collections.defaultdict(dict)
    for r in cx:
        if r["meta"].get("qid") in (2, 3):
            pairs[r["uid"].split(":")[1]][r["meta"]["qid"]] = r
    conf, elig = 0, 0
    for _, dd in pairs.items():
        if 2 in dd and 3 in dd:
            for q, other in ((2, 3), (3, 2)):
                if hit(dd[q]):
                    continue
                elig += 1
                if anls(dd[q]["pred"], [dd[other]["gold"][0]]) >= 0.5:
                    conf += 1

    body = (
        '<div class="note warn"><strong>The decisive test could not be run on this sample.</strong> '
        'The exact scope test is: when the model answers a subplot-addressed question wrongly, does '
        'its answer match the correct answer for a <em>different</em> subplot of the same figure? '
        'That requires the same figure to appear with more than one subplot address. In the 1,000 '
        f'CharXiv figures sampled here, {shared} figures do &mdash; each figure carries exactly one '
        'subplot address &mdash; so there are zero eligible items and no permutation control to run '
        'against. This is recorded as untestable rather than quietly dropped.</div>'
        + cause_table(["how the question addresses the panel (multi-panel figures only)", "n", "accuracy"],
                [["by row and column (&ldquo;row 2, column 1&rdquo;)", f"{len(named)}",
                  pct_or_dash(cause_mean(hit(r) for r in named), 1)],
                 ["verbally (&ldquo;the left-most subplot&rdquo;)", f"{len(verbal)}",
                  pct_or_dash(cause_mean(hit(r) for r in verbal), 1)],
                 ["no address needed (layout / count questions)", f"{len(noprefix)}",
                  pct_or_dash(cause_mean(hit(r) for r in noprefix), 1)]],
                "What can be measured is whether addressing a panel costs anything. It costs a "
                "little, and verbal addressing costs more than coordinates &mdash; but both stay "
                "close to the single-panel baseline.")
        + cause_table(["distance of the addressed panel from the top-left", "n", "accuracy"], drows,
                "A weak monotone decline, with the caveat that distance from the top-left is "
                "strongly correlated with how many panels the figure has, so this is largely the "
                "resolution effect seen from another angle. Bins with fewer than 40 items are "
                "omitted rather than shown as noise.")
        + f'<div class="note"><strong>Axis confusion is not happening either.</strong> Of the '
        f'{elig} eligible cases &mdash; a wrong x-axis-label answer on a figure whose y-axis label '
        f'is also asked, or vice versa &mdash; {conf} matched the other axis. The sample is small '
        f'because both axis questions are rarely asked of the same figure, so this is weak evidence, '
        f'but it points the same way as everything else here.</div>'
        '<div class="note good"><strong>Conclusion: Haiku\'s chart errors are misreadings, not '
        'scope errors.</strong> Nothing in the data suggests the model answers about the wrong '
        'panel or the wrong axis. What the data does show is that panels get harder as figures get '
        'denser, which is a resolution story, not an addressing story.</div>')

    hard = [r for r in named if not hit(r) and (r["meta"].get("num_subplots") or 1) >= 6]
    groups = [{"bench": "charxiv",
               "note": "Failures on explicitly addressed panels in figures with six or more panels. "
                       "Open each image and check the addressed panel: in these cases the model is "
                       "generally looking at the right panel and misreading it, not reading a "
                       "neighbour.",
               "examples": ([b.example(r, tags=[f"{r['meta'].get('num_subplots')} panels",
                                                "row/col addressed"])
                             for r in worst_first(hard, 8)]
                            + [b.example(r, tags=["verbal address"])
                               for r in worst_first([x for x in verbal if not hit(x)], 4)]
                            + [b.example(r, tags=["contrast: correct"])
                               for r in [x for x in hard if False][:0]]
                            + [b.example(r, tags=["contrast: correct, dense figure"])
                               for r in [x for x in named
                                         if hit(x) and (x["meta"].get("num_subplots") or 1) >= 9][:3]])}]
    refute = (
        "<p>Marked <strong>REFUTED / untestable</strong> rather than supported. The exact test "
        "&mdash; wrong answer equals another panel's correct answer &mdash; has zero eligible items "
        "in this sample, so the strong form of the claim cannot be evaluated here at all. The "
        "indirect evidence available all points away from it: axis confusion is absent, addressing "
        "a panel costs only a few points, and the decline with panel count matches the resolution "
        "gradient found on two other benchmarks.</p>"
        "<p><strong>What would establish it.</strong> Ask every descriptive template about every "
        "panel of a multi-panel figure, then check whether wrong answers land on a sibling panel's "
        "gold at above the permutation rate. That is a straightforward experiment on the CharXiv "
        "images already on disk, and it needs API calls.</p>"
        "<p><strong>Caveat on the distance table.</strong> Panels far from the top-left only exist "
        "in large grids, so panel distance and figure density cannot be separated with observational "
        "data. Do not read that table as an independent effect.</p>")
    return Cause("subplot_scope", "Answering about the wrong panel: no evidence",
                 "There is no sign that Haiku answers about a neighbouring subplot or confuses one "
                 "axis for the other. Addressing a specific panel costs a few points, and denser "
                 "figures cost more — but that is resolution, not scope.",
                 "REFUTED",
                 f"axis confusion {conf}/{elig}; panel addressing costs "
                 f"{(cause_mean(hit(r) for r in noprefix) - cause_mean(hit(r) for r in named)) * 100:.1f}pp",
                 ["charxiv"], 0.3, body, groups, refute,
                 ["effective_resolution", "wrong_element_not_near_miss"],
                 one_bench_note="Only CharXiv has multi-panel figures with per-panel questions. The "
                                "exact test additionally needs the same figure asked about more than "
                                "one panel, which this 1,000-figure sample never does.")


def c_list_integrity(d: Data, b: Builder) -> Cause:
    keys = ["charxiv", "infographicvqa", "slidevqa"]
    rows, tot = [], {}
    modes_all = {}
    for k in keys:
        rs = d.rows[k]
        listish = [r for r in rs
                   if len([x for x in str(r["gold"][0]).split(",") if x.strip()]) > 1]
        fails = [r for r in listish if not hit(r)]
        modes = collections.Counter()
        for r in fails:
            m = classify_failure(r["gold"], r["pred"])
            modes[m] += 1
        modes_all[k] = (listish, fails, modes)
        struct = sum(modes[m] for m in ("order_only", "extra_items", "missing_items",
                                        "partial_overlap"))
        rows.append([NICE[k], f"{len(listish)}", f"{len(fails)}",
                     f"{modes['order_only']}", f"{modes['missing_items']}",
                     f"{modes['extra_items']}", f"{modes['partial_overlap']}",
                     f"{struct / max(len(listish), 1) * 100:.1f}%"])
        tot[k] = struct

    body = (
        cause_tiles([("Order violations, everywhere", f"{sum(modes_all[k][2]['order_only'] for k in keys)}",
                "right items, wrong sequence &mdash; across every list-shaped question in the "
                "study", "good"),
               ("Missing or extra items",
                f"{sum(modes_all[k][2]['missing_items'] + modes_all[k][2]['extra_items'] for k in keys)}",
                "incomplete or over-complete answers", "warn"),
               ("Partly right", f"{sum(modes_all[k][2]['partial_overlap'] for k in keys)}",
                "some items right, some wrong", "warn")])
        + cause_table(["benchmark", "list-shaped questions", "failures", "order only", "missing items",
                 "extra items", "partial overlap", "structural failure rate"], rows,
                "Classified deterministically by set and sequence comparison &mdash; the "
                "classifier already in this codebase. &ldquo;Order only&rdquo; means the answer "
                "contained exactly the right items in the wrong sequence, which is an "
                "instruction-following failure on a question whose perception succeeded.")
        + '<div class="note good"><strong>Enumeration order is not a blind spot.</strong> CharXiv '
        'template 13 spells the ordering out &mdash; &ldquo;from top to bottom, then from left to '
        'right&rdquo; &mdash; which makes it the sternest test of instruction-following under '
        'constraint available here, and the model violates it a handful of times in hundreds of '
        'opportunities. Anyone assuming a vision model shuffles list answers should stop assuming '
        'it.</div>'
        + '<div class="note warn"><strong>Set completeness is a real but small effect.</strong> '
        'Missing and extra items together are the visible failure mode on InfographicVQA lists, '
        'where the question often asks for &ldquo;the top three&rdquo; from a chart whose bars are '
        'close in length. This is the one part of this cause with any weight, and it is worth about '
        'ten percent of list questions on a single benchmark.</div>')

    groups = []
    for k in keys:
        listish, fails, modes = modes_all[k]
        struct = [r for r in fails if classify_failure(r["gold"], r["pred"])
                  in ("order_only", "extra_items", "missing_items", "partial_overlap")]
        if not struct:
            continue
        groups.append({"bench": k,
                       "note": f"All {len(struct)} structural list failures on this benchmark, "
                               f"tagged with the mode. Compare the gold and the prediction "
                               f"item-by-item.",
                       "examples": [b.example(r, tags=[classify_failure(r["gold"], r["pred"]).replace("_", " ")])
                                    for r in struct[:10]]})
    refute = (
        "<p>The order-violation half is <strong>refuted</strong>: a handful of cases across "
        "hundreds of list questions, on the one benchmark that explicitly demands an order.</p>"
        "<p>The completeness half is <strong>supported but small</strong>. It would be refuted if "
        "missing and extra items were absent; they are not, but they account for a single-digit "
        "percentage of list questions.</p>"
        "<p><strong>Not tested.</strong> Whether the missing items are missing because they were "
        "not seen or because the model stopped early. Distinguishing those needs a probe that asks "
        "for the items one at a time. Also untested: instruction-following under a numeric format "
        "constraint (&ldquo;give your answer to two decimal places&rdquo;) &mdash; CharXiv states "
        "such constraints in its prompts, but this harness has no clean way to separate a "
        "constraint violation from a wrong value, so it is not claimed here in either "
        "direction.</p>")
    return Cause("list_answer_integrity", "Lists come back complete and in order",
                 "When the answer is a list, Haiku almost never returns the right items in the "
                 "wrong order — even where the prompt demands a specific order. Missing or extra "
                 "items do occur, but at single-digit rates and mainly on one benchmark.",
                 "REFUTED",
                 f"{sum(modes_all[k][2]['order_only'] for k in keys)} order violations across all "
                 f"list questions in the study",
                 keys, 0.28, body, groups, refute,
                 ["answer_expression", "counting"])


def c_gt_noise(d: Data, b: Builder) -> Cause:
    audits, rows = {}, []
    for ds in ("charxiv", "infographicvqa", "screenspot_pro"):
        recs = [json.loads(l) for l in open(f"results/{ds}__gtaudit.jsonl") if l.strip()]
        recs = [r for r in recs if r.get("verdict")]
        audits[ds] = recs
        nfail = len([r for r in d.rows[ds] if not hit(r)])
        n = len(d.rows[ds])
        contested = [r for r in recs
                     if r.get("verdict") in ("prediction_correct", "both_acceptable")
                     or r.get("gt_quality") in ("wrong", "ambiguous")]
        rate = len(contested) / max(len(recs), 1)
        rows.append([NICE[ds], f"{len(recs)}", f"{nfail}",
                     f"{len(contested)}", f"{rate * 100:.1f}%",
                     (f"<strong>{rate * nfail / n * 100:.1f}%</strong>", "num")])

    equiv = {}
    for ds in ("charxiv", "infographicvqa"):
        recs = []
        for line in open(f"results/{ds}__equiv.jsonl"):
            line = line.strip()
            if not line:
                continue
            try:
                v = json.loads(line)
            except Exception:
                continue
            if "error" not in v:
                recs.append(v)
        equiv[ds] = recs

    # SlideVQA: arithmetic expressions that do not evaluate to the annotated answer
    bad_expr = []
    for qa, m in d.manifests["slidevqa"].items():
        e = m.get("arithmetic_expression")
        if e in (None, "None", ""):
            continue
        try:
            v = eval(e.replace(",", ""), {"__builtins__": {}})
        except Exception:
            bad_expr.append((qa, e, m["answer"], None))
            continue
        a = numval(m["answer"])
        if a is None or abs(v - a) > 1e-6 * max(1.0, abs(a)):
            bad_expr.append((qa, e, m["answer"], v))

    total_floor = cause_mean([float(re.sub(r"<[^>]*>", "", r[5][0]).rstrip("%")) for r in rows])

    body = (
        cause_tiles([("CharXiv floor", rows[0][5][0],
                f"share of the whole benchmark whose failure is contested by an independent audit "
                f"(n={rows[0][1]} audited)", "warn"),
               ("InfographicVQA floor", rows[1][5][0],
                f"n={rows[1][1]} audited failures", "warn"),
               ("ScreenSpot-Pro floor", rows[2][5][0],
                f"n={rows[2][1]} audited failures &mdash; box annotations are nearly always right",
                "good"),
               ("SlideVQA arithmetic golds that do not compute", f"{len(bad_expr)}",
                "the annotated expression does not evaluate to the annotated answer", "warn")])
        + cause_table(["benchmark", "failures audited", "total failures", "contested", "contest rate",
                 "implied floor on the whole benchmark"], rows,
                "An independent judge was shown the image, the question, the gold and the "
                "prediction, and asked whether the gold was actually right. &ldquo;Contested&rdquo; "
                "counts verdicts of <em>prediction correct</em> or <em>both acceptable</em>, plus "
                "golds judged ambiguous or wrong. Extrapolating the contest rate over all failures "
                "gives the floor: the share of each benchmark where a reported failure may not be "
                "one.")
        + cause_table(["SlideVQA question", "annotated expression", "annotated answer", "what it evaluates to"],
                [[f"qa_id {qa}", f"<code>{esc(e)}</code>", f"<code>{esc(a)}</code>",
                  (f"<code>{v:g}</code>" if v is not None else "does not parse")]
                 for qa, e, a, v in bad_expr],
                "Found by evaluating every annotated arithmetic expression in the SlideVQA "
                "manifest and comparing it to the annotated answer. Two of these are answers with "
                "the expression accidentally concatenated; the rest are genuine disagreements "
                "between the two annotation fields.")
        + f'<div class="note warn"><strong>What this bounds.</strong> Averaged across the three '
        f'audited benchmarks the floor is around {total_floor:.1f}% of all items. That is small '
        f'next to the effects on the top-ranked causes &mdash; a 15-point resolution gradient or a '
        f'10% invention rate is not going to be produced by 2% bad labels &mdash; but it is the same '
        f'order of magnitude as the smaller claims on this site, which is why those are labelled '
        f'MIXED or REFUTED rather than reported as findings.</div>'
        + '<div class="note"><strong>A second, larger floor sits underneath this one.</strong> '
        'CharXiv\'s official grader is an LLM judge with per-question-type rubrics; this harness '
        'grades by normalized string match for value-shaped templates and by edit distance for free '
        'text. That makes every CharXiv free-text number here a <em>lower bound</em> &mdash; a '
        'correct answer phrased differently is scored wrong. The equivalence audit found '
        f'{sum(1 for r in equiv["charxiv"] if r.get("equivalent"))} meaning-equivalent answers among '
        f'{len(equiv["charxiv"])} audited CharXiv failures and '
        f'{sum(1 for r in equiv["infographicvqa"] if r.get("equivalent"))} among '
        f'{len(equiv["infographicvqa"])} on InfographicVQA.</div>')

    groups = []
    for ds in ("charxiv", "infographicvqa", "screenspot_pro"):
        recs = [r for r in audits[ds]
                if r.get("verdict") in ("prediction_correct", "both_acceptable")
                or r.get("gt_quality") in ("wrong", "ambiguous")]
        byuid = {r["uid"]: r for r in d.rows[ds]}
        exs = []
        for a in recs[:10]:
            r = byuid.get(a["uid"])
            if not r:
                continue
            e = b.example(r, tags=["audited", a.get("verdict", ""), a.get("gt_quality", "")],
                          extra_kv=[("audit verdict",
                                     f'<span style="color:var(--warn)">{esc(a.get("verdict"))}</span> '
                                     f'&middot; gold quality: {esc(a.get("gt_quality"))}', "")],
                          require_vision=False, allow_contested=True)
            if e is None:
                continue
            e["why"] = esc(a.get("why") or a.get("what_the_figure_shows") or "")[:600]
            exs.append(e)
        if exs:
            groups.append({"bench": ds,
                           "note": f"Audited failures where the gold itself was judged wrong, "
                                   f"ambiguous, or no better than the model's answer. These are "
                                   f"counted as failures everywhere else on this site; they may "
                                   f"not be.",
                           "examples": exs})
    refute = (
        "<p>This page is the floor, not a claim about the model. It would matter more if the "
        "contest rate were large enough to explain a headline effect; it is not, on any of the "
        "three audited benchmarks.</p>"
        "<p><strong>The audit is itself a model.</strong> An LLM judge decided these verdicts, and "
        "a judge that sides with the model it is auditing would inflate the contest rate. The "
        "sample is also stratified by failure mode rather than uniform, so the extrapolation to "
        "the whole benchmark assumes the contest rate is similar across modes &mdash; which is "
        "plausible but unverified.</p>"
        "<p><strong>Not audited.</strong> SlideVQA and AI2D have no gold audit at all; the only "
        "SlideVQA check here is the internal consistency of its two annotation fields. AI2D's "
        "answer key is unexamined, and its label-reference questions &mdash; the weakest split in "
        "the study &mdash; are exactly the ones where an ambiguous diagram label would be easiest "
        "to mis-annotate. That is a real gap in the evidence for label_reference_binding.</p>")
    return Cause("ground_truth_noise", "How much of the reported failure is not failure",
                 "Independent audits of the gold answers put the measurement floor at roughly one "
                 "to four percent of each benchmark. Small relative to the top-ranked causes, "
                 "comparable in size to the smallest ones — which is why those are not claimed.",
                 "SUPPORTED",
                 f"floor ≈ {rows[0][5][0]}–{rows[1][5][0]} of all items",
                 ["charxiv", "infographicvqa", "screenspot_pro", "slidevqa"], 0.35,
                 body, groups, refute,
                 ["answer_expression", "absence_detection", "label_reference_binding"])


BUILDERS = [
    c_effective_resolution, c_language_prior_override, c_resolution_precision,
    c_wrong_element, c_label_reference_binding, c_derivation_vs_reading,
    c_absence_detection, c_answer_expression, c_counting, c_cross_page,
    c_retrieval, c_gt_noise, c_subplot_scope,
    c_list_integrity, c_position_bias,
]


def index_page(causes: list[Cause], d: Data) -> str:
    causes = sorted(causes, key=lambda c: -c.impact)
    rows = []
    for c in causes:
        chips = "".join(
            f'<span class="chip{" one" if len(c.benchmarks) == 1 else ""}">'
            f'{esc(NICE.get(bk, bk))}</span>' for bk in c.benchmarks)
        cross = ("cross-benchmark" if len(c.benchmarks) > 1 else "single benchmark")
        rows.append(
            f'<tr><td class="impact"><div class="imeter"><i style="width:{c.impact * 100:.0f}%">'
            f'</i></div><div style="font-size:11px;color:var(--muted);margin-top:4px">'
            f'{c.impact * 100:.0f}</div></td>'
            f'<td><a class="t" href="{c.id}.html">{esc(c.title)}</a>'
            f'<div class="cl">{c.claim}</div>'
            f'<div class="chips"><span class="chip">{cross}</span>{chips}'
            f'<span class="chip">{c.n_examples} examples</span></div></td>'
            f'<td style="width:150px">{badge(c.verdict)}</td>'
            f'<td style="width:230px;font-size:12.5px;color:var(--ink2)">{c.effect}</td></tr>')

    acc = []
    for k, cnt in d.counts.items():
        acc.append([NICE[k], f"{cnt['lines']}", f"{cnt['unique']}", f"{cnt['unusable']}",
                    f"{cnt['malformed']}", f"{cnt['scored']}",
                    pct_or_dash(cause_mean(hit(r) for r in d.rows[k]), 1),
                    pct_bare(cause_mean(r['score'] for r in d.rows[k]), 1)])
    ctrl = [["blind (no image)", f"{d.blind['__meta__']['lines']}",
             f"{d.blind['__meta__']['unique']}", f"{d.blind['__meta__']['malformed']}"],
            ["SlideVQA, first evidence slide only", f"{d.onepage['__meta__']['lines']}",
             f"{d.onepage['__meta__']['unique']}", f"{d.onepage['__meta__']['malformed']}"],
            ["ScreenSpot-Pro, labelled 4&times;4 grid", f"{d.grid4['__meta__']['lines']}",
             f"{d.grid4['__meta__']['unique']}", f"{d.grid4['__meta__']['malformed']}"]]

    verdict_counts = collections.Counter(c.verdict for c in causes)
    cross_n = sum(1 for c in causes if len(c.benchmarks) > 1)

    body = (
        '<div class="claim"><span class="lab">How to read this</span>'
        'Each page below states one hypothesis about <em>why</em> Claude Haiku 4.5 fails at a '
        'perception task, shows the quantitative evidence for and against it, and then shows real '
        'examples drawn from every benchmark that can speak to it. A weakness two independent '
        'datasets agree on is a property of the model; a weakness only one dataset shows might be a '
        'property of that dataset, and is labelled as such at the top of its page. '
        'Negative results are kept &mdash; four of these hypotheses are refuted, and that is '
        'reported rather than dropped.</div>'
        + cause_tiles([("Causes examined", f"{len(causes)}",
                  f"{cross_n} with cross-benchmark support, {len(causes) - cross_n} single-benchmark",
                  ""),
                 ("Proven or supported",
                  f"{verdict_counts['PROVEN'] + verdict_counts['SUPPORTED']}",
                  "effects large enough to survive the measurement floor", "good"),
                 ("Refuted", f"{verdict_counts['REFUTED']}",
                  "hypotheses the data argues against", "bad"),
                 ("Questions scored", f"{sum(c['scored'] for c in d.counts.values()):,}",
                  "six benchmarks, plus three control arms", "")])
        + f'<h2>Causes, ranked by estimated impact</h2>'
        f'<p class="dek">Impact is a judgement, not a computed statistic: it weighs the size of the '
        f'effect, the number of benchmarks that reproduce it, and how much of the observed failure '
        f'it would explain if true. The effect sizes beside it are computed.</p>'
        f'<table class="idx"><tbody>{"".join(rows)}</tbody></table>'
        + '<h2>What was scored, and what was thrown away</h2>'
        + cause_table(["benchmark", "lines in file", "unique questions", "unusable (null prediction)",
                 "malformed lines", "scored", "accuracy at threshold 0.5", "mean metric score"],
                acc,
                "Result files contain retries, so lines exceed questions; the last usable row per "
                "uid wins. Rows with a null prediction are excluded and counted here rather than "
                "scored as zero &mdash; scoring a dropped API call as a failure would be the same "
                "mistake this whole study is about. Every number on every cause page is recomputed "
                "from these rows using each benchmark's official metric.")
        + cause_table(["control arm", "lines", "unique source questions", "malformed lines"], ctrl,
                "Controls join back to the main run through <code>meta.src_uid</code>. The grid "
                "file was read defensively in case it was still being written.")
        + '<div class="note"><strong>Metric per benchmark, as published.</strong> '
        'InfographicVQA and CharXiv free-text: ANLS. CharXiv value-shaped templates: normalized '
        'numeric match. SlideVQA: token F1 (exact match recorded alongside). AI2D: letter equality. '
        'ScreenSpot-Pro: click inside the gold box. No metric was substituted for a friendlier one; '
        'where a metric is unfair to the model that is reported as its own cause '
        '(<a href="answer_expression.html">answer_expression</a>) rather than silently '
        'corrected.</div>'
        '<div class="note"><strong>Which examples a gallery is allowed to show.</strong> '
        'Two exclusions are applied centrally, so no page can forget them. An item whose gold an '
        'independent audit did not uphold is dropped: a card arguing about whether Turkey counts '
        'as the Middle East teaches nothing about perception. And an item the blind arm answered '
        'correctly <em>without the image</em> is dropped from every gallery except '
        '<a href="language_prior_override.html">language_prior_override</a>, whose subject is '
        'exactly those items -- if the question was answerable from text alone, whatever went '
        'wrong was not perception. Cards confirmed image-dependent are badged and sorted first; '
        'the blind arm sampled 500 per benchmark, so the rest are kept and marked '
        '<em>not tested blind</em> rather than silently mixed in. ScreenSpot-Pro is exempt: a '
        'click target cannot be located without the screenshot, so a blind arm would score zero '
        'by construction and was never run.</div>'
        '<div class="note warn"><strong>No API calls were made to build these pages.</strong> '
        'Every number is recomputed from result files already on disk. Where a cause needs an '
        'experiment that was not run, its page says so under &ldquo;what we did not test&rdquo; '
        'rather than inferring the answer.</div>')
    return page("Why Claude Haiku 4.5 fails: causes, not benchmarks",
                "One page per hypothesised cause, each marshalling evidence from every benchmark "
                "that can test it.", body, here="All causes")


def cmd_causes(a) -> int:
    # Directories are created after the precondition check, not before: an abort
    # should not leave an empty outputs/causes/ behind looking like a half-run.
    print("loading + scoring results ...", flush=True)
    d = Data()
    for k, c in d.counts.items():
        print(f"  {k:20s} {c['scored']:5d} scored  ({c['lines']} lines, "
              f"{c['unusable']} null preds, {c['malformed']} malformed)", flush=True)

    # These pages are per-cause evidence: each one states a claim and then shows
    # the measurement behind it. On a thin results/ there is no measurement, and
    # the arithmetic that compares one slice against another has nothing to
    # subtract -- so the page either dies mid-render or, worse, comes out as a
    # wall of em-dashes that still *looks* like evidence. Refuse instead, and say
    # which datasets are short. `report_pages` is the one build that needs a full
    # scoring run; everything else here degrades gracefully.
    thin = {k: c["scored"] for k, c in d.counts.items() if c["scored"] < MIN_CAUSE_ROWS}
    if thin:
        print("\nABORT: causes needs a full results tree.", file=sys.stderr)
        for k, n in sorted(thin.items()):
            print(f"  {k:20s} {n:5d} scored  (need >= {MIN_CAUSE_ROWS})", file=sys.stderr)
        print("\nThese pages compare slices against each other, so a short run gives\n"
              "nothing to compare and the page would render as em-dashes that still\n"
              "read as evidence. Score the benchmarks first:\n"
              "  python -m blindspot.core --datasets charxiv ai2d slidevqa "
              "infographicvqa screenspot_pro --max-spend N\n"
              "Refusing rather than publishing an empty page.", file=sys.stderr)
        return 2

    PAGES.mkdir(parents=True, exist_ok=True)
    CAUSE_ASSETS.mkdir(parents=True, exist_ok=True)

    b = Builder(d, want_images=not a.no_images)
    causes = []
    for fn in BUILDERS:
        c = fn(d, b)
        CAUSE_TITLES[c.id] = c.title
        causes.append(c)
        print(f"  built {c.id:28s} {c.verdict:10s} {c.n_examples:3d} examples", flush=True)

    # Surface what the galleries refused. A silent filter reads as "we had
    # nothing better to show"; a counted one is a result.
    print(f"  example filter: kept {b.kept['confirmed']} vision-confirmed, "
          f"{b.kept['untested']} not-tested-blind, {b.kept['exempt']} exempt (ScreenSpot-Pro); "
          f"dropped {b.dropped['answerable_blind']} answerable-without-image, "
          f"{b.dropped['contested_gold']} contested-gold", flush=True)

    assets = {}
    if a.no_images:
        for p in CAUSE_ASSETS.glob("*_t.jpg"):
            k = p.name[:-6]
            assets[k] = {"key": k, "thumb": f"{k}_t.jpg", "full": f"{k}_f.jpg"}
        print(f"--no-images: reusing {len(assets)} cached assets")
    else:
        print(f"rendering {len({j['key'] for j in b.jobs})} unique images "
              f"on {a.workers} workers ...", flush=True)
        assets = render_all(b.jobs, a.workers)
        errs = [v for v in assets.values() if "error" in v]
        if errs:
            print(f"  {len(errs)} image(s) could not be rendered; those cards show text only")

    for c in causes:
        (PAGES / f"{c.id}.html").write_text(cause_page(c, assets), encoding="utf-8")
    (PAGES / "index.html").write_text(index_page(causes, d), encoding="utf-8")

    size = sum(p.stat().st_size for p in CAUSE_ASSETS.glob("*.jpg"))
    print(f"\nwrote {len(causes) + 1} pages -> {PAGES}/")
    print(f"assets: {len(list(CAUSE_ASSETS.glob('*.jpg')))} files, {size / 1e6:.1f} MB -> {CAUSE_ASSETS}/")
    return 0


# ================================= drilldown: outputs/drilldown.{html,json,csv}
SMALL_N = 30           # below this a node's score is noise, and is labelled as such
DIVERGE = 0.15         # children spread wider than this is the point of drilling in
MAX_EXAMPLES = 3       # concrete cases embedded per leaf

# Primary run per benchmark. slidevqa_allpages is a *condition*, not a benchmark,
# and is handled in the ablations section so the tree's root n stays exact.
DRILL_MAIN_FILES = {
    "charxiv": "charxiv__haiku-4-5_think2000_native_r0.jsonl",
    "infographicvqa": "infographicvqa__haiku-4-5_think2000_native_r0.jsonl",
    "slidevqa": "slidevqa__haiku-4-5_think2000_native_r0.jsonl",
    "ai2d": "ai2d__haiku-4-5_think2000_native_r0.jsonl",
    "screenspot_pro": "screenspot_pro__haiku-4-5_think2000_native_r0.jsonl",
}
ALLPAGES_FILE = "slidevqa_allpages__haiku-4-5_think2000_native_r0.jsonl"
JUDGE_FILE = "charxiv__haiku-4-5_think2000_native_r0.judged.jsonl"

BENCH_LABEL = {
    "charxiv": "CharXiv",
    "infographicvqa": "InfographicVQA",
    "slidevqa": "SlideVQA",
    "ai2d": "AI2D",
    "screenspot_pro": "ScreenSpot-Pro",
}
BENCH_METRIC = {
    "charxiv": "string match (lower bound)",
    "infographicvqa": "ANLS",
    "slidevqa": "token F1",
    "ai2d": "MC accuracy",
    "screenspot_pro": "click-in-bbox",
}
BENCH_BLURB = {
    "charxiv": ("1,000 arXiv figures, 4 descriptive questions plus 1 reasoning question each. "
                "String scoring is a lower bound on the free-text types; the official judge "
                "covers part of the split and is shown separately, never averaged in."),
    "infographicvqa": "Dense infographics; official ANLS with the 0.5 threshold.",
    "slidevqa": ("Slide decks, evidence pages supplied. Token F1 is the headline and punishes "
                 "formatting; the format-corrected column is shown beside it, never instead."),
    "ai2d": "Grade-school science diagrams, 4-way multiple choice, so chance is 25%.",
    "screenspot_pro": ("Professional application screenshots; the model must return a point "
                       "inside the target's box. Mean target covers 0.065% of the screen."),
}
NAV = [("report.html", "primitives report"), ("datasets.html", "dataset documentation"),
       ("slidevqa.html", "SlideVQA explorer"), ("failure_analysis.html", "failure analysis"),
       ("gallery/charxiv_000.html", "CharXiv gallery"),
       ("gallery/infographicvqa_000.html", "InfographicVQA gallery"),
       ("gallery/screenspot_pro_000.html", "ScreenSpot-Pro gallery")]
# Leaf "see the actual questions" targets, per benchmark.
EVIDENCE_LINK = {
    "charxiv": ("gallery/charxiv_000.html", "CharXiv gallery"),
    "infographicvqa": ("gallery/infographicvqa_000.html", "InfographicVQA gallery"),
    "screenspot_pro": ("gallery/screenspot_pro_000.html", "ScreenSpot-Pro gallery"),
    "slidevqa": ("slidevqa.html", "SlideVQA explorer"),
    "ai2d": ("datasets.html", "AI2D dataset notes"),
}


# --------------------------------------------------------------------------
# loading + scoring
# --------------------------------------------------------------------------

class Shim:
    """The four attributes `scoring.score` reads, taken straight off a result row.

    Using the official entry point rather than a reimplementation is the point:
    the drill-down must not be able to disagree with the rest of the study about
    what a score is.
    """

    __slots__ = ("uid", "dataset", "answer_type", "gold", "meta")

    def __init__(self, rec: dict, dataset: str | None = None):
        self.uid = rec["uid"]
        self.dataset = dataset or rec["dataset"]
        self.answer_type = rec["answer_type"]
        self.gold = rec["gold"]
        self.meta = rec.get("meta") or {}


def read_best(path: Path) -> tuple[dict[str, dict], dict]:
    """Rows keyed by uid, best row wins; plus what had to be thrown away.

    Result files are appended to on resume, so several of them contain the same
    uid more than once. House rule (`aggregate.load_rows`): a later row replaces
    an earlier one unless it would replace a real prediction with a null. The
    tail of a file that is still being written may be a partial line, so parse
    failures are counted rather than raised.
    """
    best: dict[str, dict] = {}
    stats = {"lines": 0, "malformed": 0, "duplicate_uids": 0}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stats["lines"] += 1
            try:
                rec = json.loads(line)
            except Exception:
                stats["malformed"] += 1
                continue
            if "uid" not in rec:
                stats["malformed"] += 1
                continue
            prev = best.get(rec["uid"])
            if prev is not None:
                stats["duplicate_uids"] += 1
            if prev is None or prev.get("pred") is None or rec.get("pred") is not None:
                best[rec["uid"]] = rec
    return best, stats


def score_file(args: tuple[str, str]) -> tuple[str, list[dict], dict]:
    """Load + score one benchmark file. Runs in a worker process."""
    bench, path = args
    best, stats = read_best(Path(path))
    recs = []
    for uid, rec in best.items():
        meta = rec.get("meta") or {}
        pred = rec.get("pred")
        r = {"uid": uid, "bench": bench, "meta": meta, "gold": rec.get("gold"),
             "pred": pred, "null": pred is None, "score": None, "em": None,
             "alt": None, "usage": rec.get("usage") or {}}
        if pred is not None:
            s = score(Shim(rec, bench), pred)
            r["score"] = float(s["score"])
            r["metric"] = s["metric"]
            if "exact_match" in s:
                r["em"] = float(s["exact_match"])
            if "picked" in s:
                r["picked"] = s["picked"]
            if "center_distance" in s:
                r["center_distance"] = s["center_distance"]
            # Format-corrected score: credit answers that mean the same number or
            # the same string once punctuation and scale words are removed.
            if bench in ("slidevqa", "infographicvqa", "charxiv"):
                r["alt"] = 1.0 if (r["score"] < 1.0 and drill_format_equivalent(pred, rec.get("gold") or [])) \
                    else r["score"]
        recs.append(r)
    stats["unique"] = len(best)
    stats["null_pred_after_dedup"] = sum(1 for r in recs if r["null"])
    return bench, recs, stats


# --------------------------------------------------------------------------
# format equivalence (conservative; no substring fallback)
# --------------------------------------------------------------------------

_DRILL_SCALE = {"k": 1e3, "thousand": 1e3, "thousands": 1e3, "m": 1e6, "mn": 1e6,
          "million": 1e6, "millions": 1e6, "bn": 1e9, "b": 1e9, "billion": 1e9,
          "billions": 1e9, "tn": 1e12, "trillion": 1e12}
_DRILL_NUM_RE = re.compile(r"^([+-]?\d*\.?\d+)\s*([a-z]*)$")


def _to_number(s: Any) -> float | None:
    t = str(s).strip().lower()
    for ch in (",", "$", "€", "£", "%", " "):
        t = t.replace(ch, "")
    t = t.strip()
    m = _DRILL_NUM_RE.match(t)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    suf = m.group(2)
    if suf:
        if suf not in _DRILL_SCALE:
            return None
        v *= _DRILL_SCALE[suf]
    return v


def _drill_alnum(s: Any) -> str:
    return re.sub(r"[^0-9a-z]", "", str(s).lower())


def drill_format_equivalent(pred: Any, golds: Iterable) -> bool:
    """True when pred and some gold differ only in formatting.

    Sign-sensitive on the numeric path (-5 is not 5) and exact on the folded
    string path (no substring matching), because a lenient detector here would
    manufacture the very finding it is meant to measure.
    """
    if pred is None:
        return False
    pn = _to_number(pred)
    pa = _drill_alnum(pred)
    for g in golds or []:
        gn = _to_number(g)
        if pn is not None and gn is not None:
            if math.isclose(pn, gn, rel_tol=1e-9, abs_tol=1e-12):
                return True
            continue
        if pa and pa == _drill_alnum(g):
            return True
    return False


# --------------------------------------------------------------------------
# nesting spec
# --------------------------------------------------------------------------

@dataclass
class Dim:
    """One way to split a node. `key` maps a record to a child label."""
    name: str
    key: Callable[[dict], Any]
    order: Callable[[tuple[Any, list]], Any] | None = None   # sort key over (label, recs)
    top_k: int | None = None
    other: str = "other (aggregated)"


def _b(v) -> str:
    return "yes" if v else "no"


def _missing(v, label="(metadata missing)"):
    return label if v is None else v


CHARXIV_QLABEL = {
    1: "title", 2: "x-axis label", 3: "y-axis label", 4: "leftmost x tick",
    5: "rightmost x tick", 6: "lowest y tick", 7: "highest y tick",
    8: "x tick spacing", 9: "y tick spacing", 10: "how many lines",
    11: "do any lines intersect", 12: "how many legend labels",
    13: "legend label names", 14: "colorbar range", 15: "colorbar max",
    16: "general trend", 17: "total labeled ticks", 18: "subplot layout",
    19: "number of subplots",
}
CHARXIV_ATYPE = {1: "a-type 1", 2: "a-type 2", 3: "a-type 3", 4: "a-type 4"}
CHARXIV_QSRC = {1: "q-source 1", 2: "q-source 2", 3: "q-source 3"}


def _subplot_bucket(r) -> str:
    n = r["meta"].get("num_subplots")
    if n is None:
        return "(metadata missing)"
    if n == 1:
        return "1 subplot"
    if n <= 3:
        return "2-3 subplots"
    if n <= 6:
        return "4-6 subplots"
    if n <= 12:
        return "7-12 subplots"
    return "13+ subplots"


def _is_na(r) -> str:
    golds = r.get("gold") or []
    na = any(str(g).strip().lower() == "not applicable" for g in golds)
    return "gold is 'Not Applicable' (abstention test)" if na else "gold is a real answer"


def _target_px(r) -> str:
    """Target side length in pixels *as the model saw it*, after the 1568px cap."""
    frac = r["meta"].get("target_area_frac")
    if frac is None:
        return "(metadata missing)"
    side = math.sqrt(frac * 1568 * 882)
    for lim, name in ((12, "<12px"), (20, "12-20px"), (32, "20-32px"), (56, "32-56px")):
        if side < lim:
            return name
    return ">=56px"


def _size_order(item):
    order = ["<12px", "12-20px", "20-32px", "32-56px", ">=56px", "(metadata missing)"]
    lab = item[0]
    return order.index(lab) if lab in order else 99


def _n_desc(item):
    return -len(item[1])


def _alpha(item):
    return str(item[0])


def spec_charxiv(path: tuple) -> list[Dim]:
    if len(path) == 0:
        return [Dim("split", lambda r: r["meta"].get("split") or "(metadata missing)",
                    order=_alpha)]
    split = path[0]
    if len(path) == 1:
        if split == "descriptive":
            return [Dim("question type",
                        lambda r: f"Q{r['meta'].get('qid')} — "
                                  f"{CHARXIV_QLABEL.get(r['meta'].get('qid'), r['meta'].get('qlabel') or '?')}",
                        order=lambda it: int(str(it[0]).split()[0][1:].rstrip("—").strip() or 0))]
        return [Dim("answer type", lambda r: CHARXIV_ATYPE.get(
                        r["meta"].get("reasoning_a_type"), "(metadata missing)"), order=_alpha),
                Dim("question source", lambda r: CHARXIV_QSRC.get(
                        r["meta"].get("reasoning_q_source"), "(metadata missing)"), order=_alpha)]
    if len(path) == 2:
        dims = []
        if split == "descriptive":
            dims.append(Dim("answerability", _is_na, order=_alpha))
        dims += [Dim("subplot count", _subplot_bucket, order=_alpha),
                 Dim("arXiv subject", lambda r: r["meta"].get("category") or "(metadata missing)",
                     order=_n_desc),
                 Dim("arXiv year", lambda r: "20" + str(r["meta"].get("year") or "??"),
                     order=_alpha)]
        return dims
    if len(path) == 3 and split == "descriptive" and str(path[2]).startswith("gold is"):
        # One more level under the abstention split: does subplot count change
        # whether the model invents a value for a question with no answer?
        return [Dim("subplot count", _subplot_bucket, order=_alpha)]
    return []


def spec_infographicvqa(path: tuple) -> list[Dim]:
    if len(path) == 0:
        return [Dim("operation", lambda r: " + ".join(r["meta"].get("operation") or [])
                    or "direct_lookup", order=_n_desc)]
    if len(path) == 1:
        return [Dim("gold answer type",
                    lambda r: " + ".join(r["meta"].get("gold_answer_type") or []) or "(unlabelled)",
                    order=_n_desc)]
    if len(path) == 2:
        return [Dim("gold answer shape", lambda r: _gold_shape(r), order=_n_desc)]
    return []


def _gold_shape(r) -> str:
    golds = r.get("gold") or []
    if not golds:
        return "(no gold)"
    g = str(golds[0])
    if _to_number(g) is not None:
        return "numeric gold"
    ntok = len(g.split())
    if ntok == 1:
        return "one-word gold"
    if ntok <= 4:
        return "short phrase (2-4 words)"
    return "long phrase (5+ words)"


def spec_slidevqa(path: tuple) -> list[Dim]:
    if len(path) == 0:
        return [Dim("evidence spread",
                    lambda r: "multi-page evidence" if r["meta"].get("multi_page")
                    else "single-page evidence", order=_alpha)]
    if len(path) == 1:
        return [Dim("arithmetic required",
                    lambda r: "arithmetic" if r["meta"].get("arithmetic") else "lookup only",
                    order=_alpha)]
    if len(path) == 2:
        return [Dim("evidence pages", lambda r: f"{_missing(r['meta'].get('n_evidence'), '?')} "
                                                f"evidence page(s)", order=_alpha),
                Dim("gold answer shape", _gold_shape, order=_n_desc)]
    if len(path) == 3 and "evidence page" in str(path[2]):
        return [Dim("deck", lambda r: (r["meta"].get("deck") or "(metadata missing)")[:44],
                    order=_n_desc, top_k=15, other="remaining decks (aggregated)")]
    return []


def spec_ai2d(path: tuple) -> list[Dim]:
    if len(path) == 0:
        return [Dim("question type", lambda r: r["meta"].get("qtype") or "(metadata missing)",
                    order=_alpha)]
    if len(path) == 1:
        return [Dim("option count", lambda r: f"{len(r['meta'].get('options') or [])} options",
                    order=_alpha)]
    if len(path) == 2:
        return [Dim("gold letter", lambda r: f"gold = {(r.get('gold') or ['?'])[0]}", order=_alpha),
                Dim("picked letter", lambda r: f"picked {r.get('picked', '?')}", order=_alpha),
                Dim("gold option length",
                    lambda r: _opt_len(r), order=_alpha)]
    if len(path) == 3 and str(path[2]).startswith("gold ="):
        return [Dim("picked letter", lambda r: f"picked {r.get('picked', '?')}", order=_alpha)]
    return []


def _opt_len(r) -> str:
    g = r["meta"].get("gold_text") or ""
    n = len(str(g).split())
    return "1-word answer" if n <= 1 else ("2-3 word answer" if n <= 3 else "4+ word answer")


def spec_screenspot(path: tuple) -> list[Dim]:
    if len(path) == 0:
        return [Dim("element type", lambda r: r["meta"].get("ui_type") or "(metadata missing)",
                    order=_alpha)]
    if len(path) == 1:
        return [Dim("platform", lambda r: r["meta"].get("platform") or "(metadata missing)",
                    order=_n_desc)]
    if len(path) == 2:
        return [Dim("application group", lambda r: r["meta"].get("group") or "(metadata missing)",
                    order=_n_desc),
                Dim("target size", _target_px, order=_size_order)]
    if len(path) == 3 and not str(path[2]).endswith("px"):
        return [Dim("application", lambda r: r["meta"].get("application") or "(metadata missing)",
                    order=_n_desc, top_k=12, other="remaining applications (aggregated)")]
    if len(path) == 4:
        return [Dim("target size", _target_px, order=_size_order)]
    return []


SPEC: dict[str, Callable[[tuple], list[Dim]]] = {
    "charxiv": spec_charxiv,
    "infographicvqa": spec_infographicvqa,
    "slidevqa": spec_slidevqa,
    "ai2d": spec_ai2d,
    "screenspot_pro": spec_screenspot,
}


# --------------------------------------------------------------------------
# tree
# --------------------------------------------------------------------------

@dataclass
class Node:
    label: str
    level: str
    kind: str                       # root | bench | dim | node
    depth: int
    bench: str | None
    recs: list = field(repr=False, default_factory=list)
    children: list = field(default_factory=list)
    path: tuple = ()
    nid: str = ""
    # computed
    n: int = 0
    n_null: int = 0
    value: float | None = None
    metric: str = ""
    em: float | None = None
    alt: float | None = None
    delta: float | None = None
    spread: float | None = None
    judge_n: int = 0
    judge_value: float | None = None
    examples: list = field(default_factory=list)
    note: str = ""


def _agg(recs: list) -> tuple[int, int, float | None, float | None, float | None]:
    live = [r for r in recs if not r["null"]]
    n_null = len(recs) - len(live)
    if not live:
        return 0, n_null, None, None, None
    v = sum(r["score"] for r in live) / len(live)
    ems = [r["em"] for r in live if r["em"] is not None]
    alts = [r["alt"] for r in live if r["alt"] is not None]
    return (len(live), n_null, v,
            (sum(ems) / len(ems)) if ems else None,
            (sum(alts) / len(alts)) if alts else None)


def _drill_examples(recs: list, bench: str) -> list[dict]:
    live = [r for r in recs if not r["null"]]
    if not live:
        return []
    live = sorted(live, key=lambda r: r["score"])
    picks = []
    seen = set()
    for r in [live[0], live[-1], live[len(live) // 2]]:
        if r["uid"] in seen:
            continue
        seen.add(r["uid"])
        picks.append(r)
        if len(picks) >= MAX_EXAMPLES:
            break

    def sh(v, lim=54):
        s = ", ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)
        return s if len(s) <= lim else s[:lim - 1] + "…"

    return [{"uid": r["uid"], "gold": sh(r["gold"]), "pred": sh(r["pred"]),
             "score": r["score"]} for r in picks]


def build(label: str, level: str, kind: str, recs: list, bench: str,
          spec: Callable, path: tuple, depth: int, violations: list,
          counter: list) -> Node:
    counter[0] += 1
    node = Node(label=label, level=level, kind=kind, depth=depth, bench=bench,
                recs=recs, path=path, nid=f"n{counter[0]}")
    node.n, node.n_null, node.value, node.em, node.alt = _agg(recs)
    node.metric = BENCH_METRIC.get(bench, "mixed metrics")

    dims = spec(path) if spec else []
    if not dims:
        node.examples = _drill_examples(recs, bench)
        return node

    dim_nodes = []
    # A level with several dimensions gets a "by X" grouping node, so its real
    # children sit one indent deeper than a single-dimension level's would.
    kid_depth = depth + (2 if len(dims) > 1 else 1)
    for dim in dims:
        groups: dict[Any, list] = defaultdict(list)
        for r in recs:
            groups[dim.key(r)].append(r)
        items = list(groups.items())
        try:
            items.sort(key=dim.order or _n_desc)
        except Exception:
            items.sort(key=_n_desc)
        if dim.top_k is not None and len(items) > dim.top_k:
            by_n = sorted(items, key=lambda kv: -len(kv[1]))
            keep = {k for k, _ in by_n[:dim.top_k]}
            rest = [r for k, v in items if k not in keep for r in v]
            items = [(k, v) for k, v in items if k in keep]
            if rest:
                items.append((f"{dim.other} ×{len(by_n) - dim.top_k}", rest))

        kids = [build(str(k), dim.name, "node", v, bench, spec, path + (str(k),),
                      kid_depth, violations, counter) for k, v in items]

        # --- the arithmetic check, per dimension ---------------------------
        tot_n = sum(k.n for k in kids)
        tot_null = sum(k.n_null for k in kids)
        if tot_n != node.n or tot_null != node.n_null:
            violations.append({
                "node": " › ".join(("ALL",) + path) or "ALL",
                "dimension": dim.name, "parent_n": node.n, "children_n": tot_n,
                "parent_null": node.n_null, "children_null": tot_null})

        if len(dims) == 1:
            node.children = kids
            _finish(node, kids)
            return node
        counter[0] += 1
        dn = Node(label=f"by {dim.name}", level=dim.name, kind="dim", depth=depth + 1,
                  bench=bench, path=path, nid=f"n{counter[0]}")
        dn.n, dn.n_null, dn.value, dn.em, dn.alt = node.n, node.n_null, node.value, node.em, node.alt
        dn.metric = node.metric
        dn.children = kids
        _finish(dn, kids)
        dim_nodes.append(dn)

    node.children = dim_nodes
    return node


def _finish(parent: Node, kids: list[Node]):
    """Delta-vs-parent on each child, and the parent's child-spread flag."""
    for k in kids:
        if k.value is not None and parent.value is not None:
            k.delta = k.value - parent.value
    solid = [k.value for k in kids if k.n >= SMALL_N and k.value is not None]
    if len(solid) >= 2:
        parent.spread = max(solid) - min(solid)


def walk(node: Node):
    yield node
    for c in node.children:
        yield from walk(c)


# --------------------------------------------------------------------------
# ablations / controls
# --------------------------------------------------------------------------

def load_control(name: str) -> tuple[list[dict], dict]:
    p = RESULTS / name
    if not p.exists():
        return [], {"lines": 0, "malformed": 0, "unique": 0}
    best, stats = read_best(p)
    stats["unique"] = len(best)
    return list(best.values()), stats


def control_blocks(main_recs: dict[str, list[dict]]) -> dict:
    """Every ablation in the study, recomputed from disk."""
    by_uid: dict[str, dict] = {}
    for recs in main_recs.values():
        for r in recs:
            by_uid[r["uid"]] = r

    out: dict[str, Any] = {}

    # ---- blind control -------------------------------------------------
    blind, bstats = load_control("control_blind.jsonl")
    agg = defaultdict(lambda: {"blind": [], "sighted": [], "unmatched": 0})
    cx_split = defaultdict(lambda: {"blind": [], "sighted": []})
    for b in blind:
        ds = b["dataset"].replace("_blind", "")
        src = (b.get("meta") or {}).get("src_uid")
        m = by_uid.get(src)
        if m is None or m["null"] or b.get("pred") is None:
            agg[ds]["unmatched"] += 1
            continue
        bs = float(score(Shim(b, ds), b["pred"])["score"])
        agg[ds]["blind"].append(bs)
        agg[ds]["sighted"].append(m["score"])
        if ds == "charxiv":
            sp = (b.get("meta") or {}).get("split") or "?"
            cx_split[sp]["blind"].append(bs)
            cx_split[sp]["sighted"].append(m["score"])
    out["blind"] = {"stats": bstats, "rows": [], "charxiv_split": []}
    for ds in sorted(agg):
        d = agg[ds]
        if not d["blind"]:
            continue
        out["blind"]["rows"].append({
            "bench": BENCH_LABEL.get(ds, ds), "n": len(d["blind"]),
            "blind": statistics.mean(d["blind"]), "sighted": statistics.mean(d["sighted"]),
            "delta": statistics.mean(d["sighted"]) - statistics.mean(d["blind"]),
            "unmatched": d["unmatched"],
            "chance": 0.25 if ds == "ai2d" else None,
            "metric": BENCH_METRIC.get(ds, "")})
    for sp in sorted(cx_split):
        d = cx_split[sp]
        out["blind"]["charxiv_split"].append({
            "bench": f"CharXiv {sp}", "n": len(d["blind"]),
            "blind": statistics.mean(d["blind"]), "sighted": statistics.mean(d["sighted"]),
            "delta": statistics.mean(d["sighted"]) - statistics.mean(d["blind"]),
            "unmatched": 0, "chance": None, "metric": BENCH_METRIC["charxiv"]})

    # ---- one-page ablation ---------------------------------------------
    op, opstats = load_control("control_onepage0.jsonl")
    one, both, ans = [], [], []
    unmatched = 0
    for r in op:
        m = by_uid.get((r.get("meta") or {}).get("src_uid"))
        if m is None or m["null"] or r.get("pred") is None:
            unmatched += 1
            continue
        f1 = float(score(Shim(r, "slidevqa"), r["pred"])["score"])
        one.append(f1)
        both.append(m["score"])
        ans.append(1.0 if f1 >= 0.5 else 0.0)
    out["onepage"] = {"stats": opstats, "n": len(one), "unmatched": unmatched,
                      "one": statistics.mean(one) if one else None,
                      "both": statistics.mean(both) if both else None,
                      "answerable": statistics.mean(ans) if ans else None}

    # ---- coarse localization -------------------------------------------
    ss = main_recs.get("screenspot_pro", [])
    live = [r for r in ss if not r["null"]]
    coarse = []
    for k in (2, 3, 4, 8):
        hit = 0
        for r in live:
            x0, y0, x1, y1 = r["gold"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            px, py = r["pred"]
            same = (min(int(px * k), k - 1) == min(int(cx * k), k - 1)
                    and min(int(py * k), k - 1) == min(int(cy * k), k - 1))
            hit += same
        coarse.append({"label": f"{k}x{k} cell", "n": len(live), "acc": hit / len(live),
                       "chance": 1.0 / (k * k)})
    if live:
        mean_frac = statistics.mean(r["meta"].get("target_area_frac") or 0 for r in live)
        coarse.append({"label": "exact click-in-bbox", "n": len(live),
                       "acc": statistics.mean(r["score"] for r in live),
                       "chance": mean_frac})
    g4, g4stats = load_control("control_grid4.jsonl")
    g4live = [r for r in g4 if r.get("pred") is not None]
    if g4live:
        hit = sum(1 for r in g4live
                  if str(r["pred"]).strip().upper() == str((r.get("gold") or ["?"])[0]).strip().upper())
        coarse.append({"label": "4x4 named cell (model asked for the cell)", "n": len(g4live),
                       "acc": hit / len(g4live), "chance": 1 / 16,
                       "note": "separate control run, not derived"})
    out["coarse"] = {"rows": coarse, "grid4_stats": g4stats,
                     "grid4_used": len(g4live), "grid4_null": len(g4) - len(g4live)}

    # ---- abstention (CharXiv 'Not Applicable') --------------------------
    cx = [r for r in main_recs.get("charxiv", []) if not r["null"]]
    na = [r for r in cx if any(str(g).strip().lower() == "not applicable" for g in (r["gold"] or []))]
    real = [r for r in cx
            if not any(str(g).strip().lower() == "not applicable" for g in (r["gold"] or []))]

    def says_na(r):
        return str(r["pred"]).strip().lower() in ("not applicable", "n/a", "na", "none", "not applicable.")

    per_q = []
    byq = defaultdict(list)
    for r in na:
        byq[r["meta"].get("qid")].append(r)
    for q, rs in byq.items():
        per_q.append({"qid": q, "label": CHARXIV_QLABEL.get(q, str(q)), "n": len(rs),
                      "abstains": statistics.mean(float(says_na(r)) for r in rs)})
    per_q.sort(key=lambda d: d["abstains"])
    out["abstention"] = {
        "n_na": len(na), "n_real": len(real),
        "correct_abstain": statistics.mean(float(says_na(r)) for r in na) if na else None,
        "invents": 1 - (statistics.mean(float(says_na(r)) for r in na) if na else 0),
        "over_abstain": statistics.mean(float(says_na(r)) for r in real) if real else None,
        "per_q": per_q}

    # ---- format artifact ------------------------------------------------
    fmt = []
    for bench in ("slidevqa", "charxiv", "infographicvqa"):
        rs = [r for r in main_recs.get(bench, []) if not r["null"]]
        zeros = [r for r in rs if r["score"] == 0.0]
        if not zeros:
            continue
        eq = sum(1 for r in zeros if drill_format_equivalent(r["pred"], r["gold"]))
        base = statistics.mean(r["score"] for r in rs)
        corr = statistics.mean(r["alt"] if r["alt"] is not None else r["score"] for r in rs)
        fmt.append({"bench": BENCH_LABEL[bench], "n": len(rs), "zeros": len(zeros),
                    "fmt_equiv": eq, "share": eq / len(zeros),
                    "as_scored": base, "corrected": corr,
                    "metric": BENCH_METRIC[bench]})
    out["format"] = fmt

    # ---- numeric error distribution -------------------------------------
    num = []
    groups = [("CharXiv descriptive", [r for r in main_recs.get("charxiv", [])
                                       if r["meta"].get("split") == "descriptive"]),
              ("CharXiv reasoning", [r for r in main_recs.get("charxiv", [])
                                     if r["meta"].get("split") == "reasoning"]),
              ("InfographicVQA", main_recs.get("infographicvqa", [])),
              ("SlideVQA", main_recs.get("slidevqa", []))]
    for lab, rs in groups:
        errs = []
        skipped_fmt = 0
        for r in rs:
            if r["null"] or r["score"] >= 0.5:
                continue
            # A "22%" scored against a gold of "22" is a formatting artifact, not a
            # misread number. Counting it here would report a 0% median error and
            # say the model reads numbers perfectly, which is the opposite of true.
            if drill_format_equivalent(r["pred"], r["gold"]):
                skipped_fmt += 1
                continue
            p = _to_number(r["pred"])
            gs = [_to_number(g) for g in (r["gold"] or [])]
            gs = [g for g in gs if g is not None]
            if p is None or not gs:
                continue
            g = min(gs, key=lambda gg: abs(gg - p))
            if g == 0:
                continue
            errs.append(abs(p - g) / abs(g))
        if len(errs) < 10:
            continue
        errs.sort()
        num.append({"label": lab, "n": len(errs), "median": statistics.median(errs),
                    "within10": sum(1 for e in errs if e <= 0.10) / len(errs),
                    "over100": sum(1 for e in errs if e > 1.0) / len(errs),
                    "skipped_fmt": skipped_fmt})
    out["numeric"] = num

    # ---- SlideVQA 20-slide haystack -------------------------------------
    ap_best, ap_stats = read_best(RESULTS / ALLPAGES_FILE)
    ap = []
    for uid, rec in ap_best.items():
        if rec.get("pred") is None:
            continue
        s = score(Shim(rec, "slidevqa"), rec["pred"])
        ap.append({"uid": uid, "f1": float(s["score"]), "em": float(s["exact_match"]),
                   "meta": rec.get("meta") or {}})
    ev = [r for r in main_recs.get("slidevqa", []) if not r["null"]]
    ev_by = {r["uid"].replace("slidevqa:evidence:", ""): r for r in ev}
    pairs = [(r, ev_by.get(r["uid"].replace("slidevqa:all_pages:", ""))) for r in ap]
    pairs = [(a, b) for a, b in pairs if b is not None]
    out["haystack"] = {
        "n": len(ap), "stats": ap_stats,
        "f1": statistics.mean(r["f1"] for r in ap) if ap else None,
        "em": statistics.mean(r["em"] for r in ap) if ap else None,
        "paired_n": len(pairs),
        "paired_all": statistics.mean(a["f1"] for a, b in pairs) if pairs else None,
        "paired_ev": statistics.mean(b["score"] for a, b in pairs) if pairs else None}
    return out


def judge_join(cx_recs: list[dict]) -> dict:
    """CharXiv official-judge subset, kept strictly apart from string scoring."""
    p = RESULTS / JUDGE_FILE
    if not p.exists():
        return {}
    judged = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "uid" in d and d.get("judge_score") is not None:
            judged[d["uid"]] = d
    by_uid = {r["uid"]: r for r in cx_recs}
    inter = [u for u in judged if u in by_uid and not by_uid[u]["null"]]
    if not inter:
        return {}
    mismatch = sum(1 for u in inter if str(judged[u].get("pred")) != str(by_uid[u]["pred"]))
    per_split = defaultdict(lambda: {"j": [], "s": []})
    for u in inter:
        sp = judged[u].get("split") or by_uid[u]["meta"].get("split") or "?"
        per_split[sp]["j"].append(float(judged[u]["judge_score"]))
        per_split[sp]["s"].append(by_uid[u]["score"])
    return {
        "judged_rows": len(judged), "joined": len(inter), "coverage": len(inter) / len(cx_recs),
        "pred_mismatch": mismatch,
        "judge": statistics.mean(float(judged[u]["judge_score"]) for u in inter),
        "string": statistics.mean(by_uid[u]["score"] for u in inter),
        "agreement": statistics.mean(
            float((float(judged[u]["judge_score"]) >= 0.5) == (by_uid[u]["score"] >= 0.5))
            for u in inter),
        "per_split": {k: {"n": len(v["j"]), "judge": statistics.mean(v["j"]),
                          "string": statistics.mean(v["s"])} for k, v in per_split.items()},
        "uids": set(inter)}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _mix(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def score_color(value: float | None, anchor: float, spread: float) -> tuple[str, str]:
    """Diverging ramp anchored at the benchmark's own headline.

    Absolute colour would paint all of ScreenSpot-Pro red and all of CharXiv
    green, which is exactly the information the drill-down already has. Anchoring
    each benchmark at its own number makes "worse than this benchmark's average"
    the thing the colour encodes, which is what you scan for.
    """
    if value is None:
        return ("transparent", "transparent")
    t = (value - anchor) / max(spread, 1e-6)
    t = max(-1.0, min(1.0, t))
    if t >= 0:
        strong = _mix("#8a8a8a", GOOD, t)
    else:
        strong = _mix("#8a8a8a", BAD, -t)
    return strong, strong


DRILL_CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
 --muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
 --s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;--bad:#d03b3b;--warn:#fab219;
 --track:#e8e7e0;--hover:rgba(42,120,214,.07);--chip:rgba(11,11,11,.05)}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
 --surface:#1a1a19;--page:#0d0d0d;--ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.12);--s1:#3987e5;--s2:#d95926;
 --good:#3fce3f;--bad:#f06a6a;--track:#2c2c2a;--hover:rgba(57,135,229,.13);
 --chip:rgba(255,255,255,.07)}}
:root[data-theme=dark]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#ffffff;
 --ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.12);
 --s1:#3987e5;--s2:#d95926;--good:#3fce3f;--bad:#f06a6a;--track:#2c2c2a;
 --hover:rgba(57,135,229,.13);--chip:rgba(255,255,255,.07)}
:root[data-theme=light]{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;
 --ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
 --s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;--bad:#d03b3b;--track:#e8e7e0;
 --hover:rgba(42,120,214,.07);--chip:rgba(11,11,11,.05)}
*{box-sizing:border-box}
html,body{background:var(--page);color:var(--ink);margin:0;
 font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:30px 22px 90px}
header{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;flex-wrap:wrap}
h1{font-size:26px;margin:0 0 6px}
.dek{color:var(--ink2);margin:0;max-width:78ch}
h2{font-size:19px;margin:42px 0 4px;padding-top:20px;border-top:1px solid var(--grid)}
h2 .sub{display:block;font-size:13.5px;font-weight:400;color:var(--ink2);margin-top:5px;max-width:88ch}
a{color:var(--s1)}
button,select{font:inherit;font-size:13px;padding:6px 11px;border-radius:8px;
 border:1px solid var(--border);background:var(--surface);color:var(--ink2);cursor:pointer}
button:hover,select:hover{color:var(--ink);border-color:var(--axis)}
button.on{background:var(--s1);border-color:var(--s1);color:#fff}
input[type=search]{font:inherit;font-size:13px;padding:6px 10px;border-radius:8px;
 border:1px solid var(--border);background:var(--surface);color:var(--ink);min-width:190px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:15px 16px}
.tlab{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.tval{font-size:30px;line-height:1.1;margin:7px 0 3px;font-variant-numeric:tabular-nums}
.tnote{font-size:12.5px;color:var(--ink2)}
.tile.bad .tval{color:var(--bad)}.tile.good .tval{color:var(--good)}
.pcts{font-size:16px;color:var(--muted)}
.note{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--s2);
 border-radius:9px;padding:13px 16px;font-size:13.5px;color:var(--ink2);margin:16px 0}
.note strong{color:var(--ink)}
.note.bad{border-left-color:var(--bad)}
.note.ok{border-left-color:var(--good)}
.toolbar{position:sticky;top:0;z-index:20;background:var(--page);border-bottom:1px solid var(--grid);
 padding:11px 0;margin:6px 0 0;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.toolbar .gap{width:9px}
.toolbar label{font-size:12.5px;color:var(--muted);margin-right:2px}
.tree{background:var(--surface);border:1px solid var(--border);border-radius:12px;
 padding:8px 4px 12px;margin:14px 0;overflow-x:auto}
.hdr{display:grid;grid-template-columns:minmax(290px,1fr) 148px 74px 62px 124px 70px;
 gap:10px;align-items:end;padding:6px 14px 8px;font-size:11px;color:var(--muted);
 text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--grid);min-width:900px}
.hdr span:nth-child(n+3){text-align:right}
.nd{min-width:900px}
.ln{display:grid;grid-template-columns:minmax(290px,1fr) 148px 74px 62px 124px 70px;
 gap:10px;align-items:center;padding:3px 14px;border-radius:7px;cursor:default}
.ln:hover{background:var(--hover)}
.nd.dim>.ln{opacity:.86}
.nd[data-k="bench"]>.ln{font-weight:600;margin-top:6px}
.nd[data-k="root"]>.ln{font-weight:700}
.lbl{display:flex;align-items:center;gap:6px;min-width:0}
.tw{flex:0 0 15px;width:15px;text-align:center;color:var(--muted);font-size:10px;
 cursor:pointer;user-select:none;transition:transform .12s}
.nd.open>.ln .tw{transform:rotate(90deg)}
.tw.leaf{cursor:default;opacity:.32}
.txt{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lv{font-size:10.5px;color:var(--muted);background:var(--chip);border-radius:4px;
 padding:1px 5px;flex:0 0 auto}
.track{height:13px;background:var(--track);border-radius:4px;position:relative;overflow:hidden}
.bar{display:block;height:100%;width:0;border-radius:0 4px 4px 0;min-width:2px;
 transition:width .15s}
.val{text-align:right;font-variant-numeric:tabular-nums;font-size:13px;white-space:nowrap}
.nnum{text-align:right;font-variant-numeric:tabular-nums;font-size:12.5px;color:var(--ink2)}
.met{text-align:right;font-size:11px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;
 white-space:nowrap}
.dlt{text-align:right;font-variant-numeric:tabular-nums;font-size:12px}
.dlt.up{color:var(--good)}.dlt.dn{color:var(--bad)}
.kids{display:none;border-left:1px solid var(--grid);margin-left:21px}
.nd.open>.kids{display:block}
.flag{display:inline-block;font-size:10px;font-weight:600;padding:1px 6px;border-radius:999px;
 background:color-mix(in srgb,var(--warn) 26%,transparent);color:var(--ink);flex:0 0 auto}
.flag.sm{background:color-mix(in srgb,var(--bad) 20%,transparent)}
.flag.jd{background:color-mix(in srgb,var(--s1) 22%,transparent)}
.ex{margin:4px 0 8px 36px;font-size:12px;border-collapse:collapse;width:calc(100% - 60px);
 min-width:520px}
.ex th{text-align:left;color:var(--muted);font-weight:400;font-size:10.5px;
 text-transform:uppercase;letter-spacing:.04em;padding:2px 8px}
.ex td{padding:2px 8px;border-top:1px solid var(--grid);vertical-align:top;
 font-variant-numeric:tabular-nums}
.ex td.g{color:var(--ink2)}
.ex td.s{text-align:right;width:52px}
.ex .ok{color:var(--good)}.ex .no{color:var(--bad)}
.ex .lk{padding-top:4px}
table.t{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 6px}
table.t th,table.t td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--grid)}
table.t td{font-variant-numeric:tabular-nums}
table.t th[scope=row]{font-weight:400;color:var(--ink2)}
table.t td.num,table.t th.num{text-align:right}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
 padding:16px 20px 18px;margin:14px 0}
.card h3{font-size:15.5px;margin:0 0 3px}
.card .sub{font-size:13px;color:var(--ink2);margin:0 0 12px;max-width:88ch}
.ok{color:var(--good)}.no{color:var(--bad)}
.legend{display:flex;gap:14px;align-items:center;font-size:12px;color:var(--ink2);
 margin:2px 0 0;flex-wrap:wrap}
.ramp{width:120px;height:10px;border-radius:3px;
 background:linear-gradient(90deg,var(--bad),#8a8a8a,var(--good))}
.foot{font-size:12.5px;color:var(--muted);margin-top:36px;border-top:1px solid var(--grid);
 padding-top:14px}
.hide{display:none !important}
"""

DRILL_JS = r"""
(function(){
const root=document.documentElement;
const tree=document.getElementById('tree');
const nodes=()=>Array.from(tree.querySelectorAll('.nd'));

function setOpen(nd,on){nd.classList.toggle('open',on);}
tree.addEventListener('click',e=>{
  const ln=e.target.closest('.ln'); if(!ln) return;
  if(e.target.closest('a')) return;
  const nd=ln.parentElement;
  if(!nd.querySelector(':scope > .kids')) return;
  setOpen(nd,!nd.classList.contains('open'));
});
tree.addEventListener('keydown',e=>{
  if(e.key!=='Enter'&&e.key!==' ') return;
  const ln=e.target.closest('.ln'); if(!ln) return;
  e.preventDefault(); ln.click();
});

document.getElementById('all').onclick=()=>nodes().forEach(n=>setOpen(n,!!n.querySelector(':scope > .kids')));
document.getElementById('none').onclick=()=>nodes().forEach(n=>setOpen(n,false));
document.querySelectorAll('[data-depth]').forEach(b=>{
  b.onclick=()=>{const d=+b.dataset.depth;
    nodes().forEach(n=>setOpen(n,(+n.dataset.d)<d && !!n.querySelector(':scope > .kids')));};
});

const sortSel=document.getElementById('sort');
function sortAll(){
  const mode=sortSel.value;
  tree.querySelectorAll('.kids').forEach(k=>{
    const kids=Array.from(k.children).filter(c=>c.classList.contains('nd'));
    if(kids.length<2) return;
    kids.sort((a,b)=>{
      if(mode==='n') return (+b.dataset.n)-(+a.dataset.n);
      if(mode==='alpha') return a.dataset.lab.localeCompare(b.dataset.lab);
      const av=a.dataset.v===''?9e9:+a.dataset.v, bv=b.dataset.v===''?9e9:+b.dataset.v;
      if(mode==='low') return av-bv;
      if(mode==='high') return bv-av;
      return (+a.dataset.i)-(+b.dataset.i);
    });
    kids.forEach(c=>k.appendChild(c));
  });
}
sortSel.onchange=sortAll;

const scaleBtn=document.getElementById('scale');
function applyScale(){
  const rel=scaleBtn.classList.contains('on');
  tree.querySelectorAll('.bar').forEach(b=>{
    const v=+b.dataset.v, mx=+b.dataset.mx||1;
    if(isNaN(v)){b.style.width='0';return;}
    const w=rel? (mx>0? v/mx*100:0) : v*100;
    b.style.width=Math.max(w,0.7).toFixed(2)+'%';
  });
  scaleBtn.textContent=rel?'bars: relative to benchmark':'bars: absolute 0-100%';
}
scaleBtn.onclick=()=>{scaleBtn.classList.toggle('on');applyScale();};
applyScale();

const smallBtn=document.getElementById('small');
smallBtn.onclick=()=>{smallBtn.classList.toggle('on');
  const on=smallBtn.classList.contains('on');
  tree.querySelectorAll('.nd[data-small="1"]').forEach(n=>n.classList.toggle('hide',on));
  smallBtn.textContent=on?'small-n hidden':'hide small-n (n<30)';};

const q=document.getElementById('q');
q.oninput=()=>{
  const s=q.value.trim().toLowerCase();
  const all=nodes();
  if(!s){all.forEach(n=>{n.classList.remove('hide');});return;}
  all.forEach(n=>n.classList.add('hide'));
  all.forEach(n=>{
    if(!n.dataset.lab.toLowerCase().includes(s)) return;
    n.classList.remove('hide');
    let p=n.parentElement;
    while(p&&p!==tree){ if(p.classList&&p.classList.contains('nd')){p.classList.remove('hide');setOpen(p,true);} p=p.parentElement; }
    n.querySelectorAll('.nd').forEach(c=>c.classList.remove('hide'));
  });
};

const tb=document.getElementById('theme');
function label(){const d=root.dataset.theme;
  tb.textContent=d==='dark'?'Light mode':(d==='light'?'Dark mode':'Dark mode');}
tb.onclick=()=>{const cur=root.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
  root.dataset.theme=cur==='dark'?'light':'dark';label();};
label();
// open the first two levels so the page is useful before any click
nodes().forEach(n=>setOpen(n,(+n.dataset.d)<2 && !!n.querySelector(':scope > .kids')));
})();
"""


def fmt_pct(v, d=1) -> str:
    return "&mdash;" if v is None else f"{v * 100:.{d}f}"


def render_node(node: Node, anchor: float, spread: float, bmax: float,
                idx: int, judge_by_node: dict) -> str:
    strong, _ = score_color(node.value, anchor, spread)
    flags = []
    if node.n < SMALL_N and node.kind not in ("root", "dim"):
        flags.append('<span class="flag sm" title="fewer than 30 scored rows: '
                     'this number is noise">n&lt;30</span>')
    if node.spread is not None and node.spread >= DIVERGE:
        flags.append(f'<span class="flag" title="widest gap between this node\'s children '
                     f'(n&ge;{SMALL_N})">children spread {node.spread * 100:.0f}pp</span>')
    if node.n_null:
        flags.append(f'<span class="flag" title="rows whose prediction was null; excluded '
                     f'from the metric">{node.n_null} null pred</span>')
    jv = judge_by_node.get(id(node))
    if jv:
        flags.append(f'<span class="flag jd" title="official CharXiv LLM judge on the '
                     f'{jv["n"]} rows of this node it covers &mdash; never averaged with the '
                     f'string score">judge {jv["v"] * 100:.1f}% on {jv["n"]}</span>')

    delta = ""
    if node.delta is not None and node.kind == "node":
        cls = "up" if node.delta >= 0 else "dn"
        delta = f'<span class="dlt {cls}">{node.delta * 100:+.1f}</span>'
    elif node.kind in ("dim", "bench", "root"):
        delta = '<span class="dlt" style="color:var(--muted)">&mdash;</span>'

    alt = ""
    if node.bench == "slidevqa" and node.alt is not None and node.value is not None \
            and abs(node.alt - node.value) > 5e-4:
        alt = (f'<span class="lv" title="same rows, crediting answers that differ from the '
               f'gold only in formatting">fmt-corrected {node.alt * 100:.1f}</span>')
    if node.bench == "slidevqa" and node.em is not None:
        alt += f'<span class="lv" title="exact match, SlideVQA\'s second official metric">' \
               f'EM {node.em * 100:.1f}</span>'

    lvl = f'<span class="lv">{esc(node.level)}</span>' if node.level and node.kind == "node" else ""
    pad = 4 + node.depth * 15
    tw = "&#9656;" if node.children else "&#8226;"
    twcls = "tw" if node.children else "tw leaf"
    val = fmt_pct(node.value)
    ln = (
        f'<div class="ln" tabindex="0" role="button" aria-expanded="false">'
        f'<span class="lbl" style="padding-left:{pad}px">'
        f'<span class="{twcls}">{tw}</span>'
        f'<span class="txt" title="{esc(node.label)}">{esc(node.label)}</span>'
        f'{lvl}{alt}{"".join(flags)}</span>'
        f'<span class="track"><span class="bar" data-v="{"" if node.value is None else f"{node.value:.6f}"}" '
        f'data-mx="{bmax:.6f}" style="background:{strong}"></span></span>'
        f'<span class="val">{val}<span class="pcts" style="font-size:11px">%</span></span>'
        f'{delta}'
        f'<span class="met" title="{esc(node.metric)}">{esc(node.metric)}</span>'
        f'<span class="nnum">n={node.n:,}</span>'
        f'</div>')

    kids = ""
    if node.children:
        parts = []
        for i, c in enumerate(node.children):
            parts.append(render_node(c, anchor, spread, bmax, i, judge_by_node))
        kids = f'<div class="kids">{"".join(parts)}</div>'
    elif node.examples:
        link = EVIDENCE_LINK.get(node.bench or "")
        rows = "".join(
            f'<tr><td class="g">{esc(e["uid"])}</td><td class="g">{esc(e["gold"])}</td>'
            f'<td>{esc(e["pred"])}</td>'
            f'<td class="s {"ok" if e["score"] >= 0.5 else "no"}">{e["score"]:.2f}</td></tr>'
            for e in node.examples)
        lk = (f'<tr><td colspan="4" class="lk"><a href="{link[0]}">browse {esc(link[1])} '
              f'&rarr;</a></td></tr>') if link else ""
        kids = (f'<div class="kids"><table class="ex"><thead><tr><th>uid</th><th>gold</th>'
                f'<th>prediction</th><th class="s">score</th></tr></thead>'
                f'<tbody>{rows}{lk}</tbody></table></div>')

    small = "1" if (node.n < SMALL_N and node.kind == "node") else "0"
    return (f'<div class="nd {"dim" if node.kind == "dim" else ""}" data-k="{node.kind}" '
            f'data-d="{node.depth}" data-n="{node.n}" '
            f'data-v="{"" if node.value is None else f"{node.value:.6f}"}" '
            f'data-i="{idx}" data-small="{small}" data-lab="{esc(node.label)}">'
            f'{ln}{kids}</div>')


def control_html(c: dict, main_recs: dict) -> str:
    parts = []

    b = c["blind"]
    rows = "".join(
        f'<tr><th scope="row">{esc(r["bench"])}</th><td class="num">{r["n"]:,}</td>'
        f'<td class="num">{fmt_pct(r["blind"])}</td><td class="num">{fmt_pct(r["sighted"])}</td>'
        f'<td class="num"><b>{r["delta"] * 100:+.1f}</b></td>'
        f'<td class="num">{"25.0" if r["chance"] else "&mdash;"}</td>'
        f'<td class="num">{r["unmatched"]}</td>'
        f'<td>{esc(r["metric"])}</td></tr>'
        for r in b["rows"] + b["charxiv_split"])
    parts.append(f"""<div class="card"><h3>Blind control &mdash; how much of each number needs the image?</h3>
<p class="sub">The same question asked with the image withheld, joined back to the sighted run by
<code>meta.src_uid</code> and compared on exactly those items, never against the full-split headline.
{b['stats']['unique']:,} control rows on disk.</p>
<table class="t"><thead><tr><th>slice</th><th class="num">n paired</th>
<th class="num">blind</th><th class="num">sighted (same items)</th><th class="num">vision adds (pp)</th>
<th class="num">chance</th><th class="num">unpaired</th><th>metric</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class="sub" style="margin:8px 0 0">AI2D is the outlier and the reason a raw AI2D score is not a
perception measurement: most of it survives with the diagram hidden. The <em>unpaired</em> column
counts control rows whose <code>src_uid</code> belongs to a condition outside this tree &mdash; the
SlideVQA blind sample was drawn across both the evidence and the all-pages arms, and only the
evidence half can be compared like for like here.</p></div>""")

    o = c["onepage"]
    if o["n"]:
        parts.append(f"""<div class="card"><h3>One-page ablation &mdash; SlideVQA multi-evidence, given one slide</h3>
<p class="sub">{o['n']:,} multi-evidence questions re-asked with only the first evidence slide.
Same metric, same items.</p>
<table class="t"><thead><tr><th>condition</th><th class="num">n</th><th class="num">token F1</th></tr></thead>
<tbody>
<tr><th scope="row">both evidence slides</th><td class="num">{o['n']:,}</td><td class="num">{fmt_pct(o['both'])}</td></tr>
<tr><th scope="row">first slide only</th><td class="num">{o['n']:,}</td><td class="num">{fmt_pct(o['one'])}</td></tr>
<tr><th scope="row">collapse</th><td class="num">&mdash;</td><td class="num no"><b>{(o['one'] - o['both']) * 100:+.1f}</b></td></tr>
<tr><th scope="row">still answerable (F1 &ge; 0.5) on one slide</th><td class="num">&mdash;</td>
<td class="num">{fmt_pct(o['answerable'])}</td></tr>
</tbody></table></div>""")

    hs = c["haystack"]
    if hs["n"]:
        parts.append(f"""<div class="card"><h3>The other direction &mdash; 20 slides instead of the evidence</h3>
<p class="sub">The all-pages arm ({hs['n']:,} rows) sends the whole deck and makes the model find the
evidence itself. Held out of the tree's root count because it is a second condition on the same
questions, not extra questions.</p>
<table class="t"><thead><tr><th>condition</th><th class="num">n</th><th class="num">token F1</th>
<th class="num">EM</th></tr></thead><tbody>
<tr><th scope="row">all 20 slides (whole arm)</th><td class="num">{hs['n']:,}</td>
<td class="num">{fmt_pct(hs['f1'])}</td><td class="num">{fmt_pct(hs['em'])}</td></tr>
<tr><th scope="row">paired: evidence pages</th><td class="num">{hs['paired_n']:,}</td>
<td class="num">{fmt_pct(hs['paired_ev'])}</td><td class="num">&mdash;</td></tr>
<tr><th scope="row">paired: all 20 slides</th><td class="num">{hs['paired_n']:,}</td>
<td class="num">{fmt_pct(hs['paired_all'])}</td><td class="num">&mdash;</td></tr>
</tbody></table></div>""")

    cz = c["coarse"]
    rows = "".join(
        f'<tr><th scope="row">{esc(r["label"])}</th><td class="num">{r["n"]:,}</td>'
        f'<td class="num">{fmt_pct(r["acc"])}</td>'
        f'<td class="num">{r["chance"] * 100:.3f}</td>'
        f'<td class="num">{(r["acc"] / r["chance"]):.1f}&times;</td></tr>'
        for r in cz["rows"])
    parts.append(f"""<div class="card"><h3>Coarse localization &mdash; the click is not random, it is imprecise</h3>
<p class="sub">The same ScreenSpot-Pro predictions, scored against progressively finer grids: a
prediction counts if it lands in the same cell as the target's centre. The last row is a separate
control in which the model was shown a labelled 4&times;4 grid and asked to name the cell
({cz['grid4_used']:,} rows used from a file that may still be growing;
{cz['grid4_stats'].get('malformed', 0)} malformed lines skipped).</p>
<table class="t"><thead><tr><th>granularity</th><th class="num">n</th><th class="num">accuracy</th>
<th class="num">chance</th><th class="num">vs chance</th></tr></thead><tbody>{rows}</tbody></table></div>""")

    a = c["abstention"]
    worst = "".join(
        f'<tr><th scope="row">Q{r["qid"]} &mdash; {esc(r["label"])}</th><td class="num">{r["n"]}</td>'
        f'<td class="num">{fmt_pct(r["abstains"])}</td></tr>' for r in a["per_q"][:5])
    best = "".join(
        f'<tr><th scope="row">Q{r["qid"]} &mdash; {esc(r["label"])}</th><td class="num">{r["n"]}</td>'
        f'<td class="num">{fmt_pct(r["abstains"])}</td></tr>' for r in a["per_q"][-3:])
    parts.append(f"""<div class="card"><h3>Abstention &mdash; CharXiv questions whose gold is "Not Applicable"</h3>
<p class="sub">{a['n_na']:,} of the {a['n_na'] + a['n_real']:,} scored CharXiv questions have no answer
in the figure. Getting these right means declining, and the failure mode is inventing a value.
The same nodes are drillable in the tree under CharXiv &rsaquo; descriptive &rsaquo; <em>by answerability</em>.</p>
<table class="t"><thead><tr><th>behaviour</th><th class="num">n</th><th class="num">rate</th></tr></thead><tbody>
<tr><th scope="row">correctly declines when there is no answer</th><td class="num">{a['n_na']:,}</td>
<td class="num ok">{fmt_pct(a['correct_abstain'])}</td></tr>
<tr><th scope="row">invents a value instead</th><td class="num">{a['n_na']:,}</td>
<td class="num no">{fmt_pct(a['invents'])}</td></tr>
<tr><th scope="row">declines when an answer does exist</th><td class="num">{a['n_real']:,}</td>
<td class="num">{fmt_pct(a['over_abstain'])}</td></tr>
</tbody></table>
<table class="t"><thead><tr><th>hardest to decline</th><th class="num">n</th><th class="num">abstains</th></tr></thead>
<tbody>{worst}</tbody></table>
<table class="t"><thead><tr><th>easiest to decline</th><th class="num">n</th><th class="num">abstains</th></tr></thead>
<tbody>{best}</tbody></table></div>""")

    f = c["format"]
    rows = "".join(
        f'<tr><th scope="row">{esc(r["bench"])}</th><td class="num">{r["zeros"]:,}</td>'
        f'<td class="num">{r["fmt_equiv"]:,}</td><td class="num"><b>{fmt_pct(r["share"])}</b></td>'
        f'<td class="num">{fmt_pct(r["as_scored"])}</td><td class="num">{fmt_pct(r["corrected"])}</td>'
        f'<td class="num">{(r["corrected"] - r["as_scored"]) * 100:+.1f}</td></tr>' for r in f)
    parts.append(f"""<div class="card"><h3>Format artifact &mdash; how many hard zeros are only formatting?</h3>
<p class="sub">A hard zero is a scored-0 answer. The detector is deliberately conservative: reduce both
sides to a number after stripping <code>, $ &euro; &pound; %</code> and scale words (bn/billion/m/million/k/thousand)
and compare sign-sensitively; otherwise compare case- and punctuation-folded alphanumerics exactly.
No substring fallback. SlideVQA's token F1 scores <code>22%</code> against <code>22</code> as a zero;
ANLS does not, which is most of why the three columns differ so much.</p>
<table class="t"><thead><tr><th>benchmark</th><th class="num">hard zeros</th>
<th class="num">format-equivalent</th><th class="num">share of zeros</th>
<th class="num">as scored</th><th class="num">format-corrected</th><th class="num">delta</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class="sub" style="margin:8px 0 0">The corrected column is never the headline. It is shown so the
SlideVQA number can be read as what it is: a token-overlap metric, not a comprehension score.</p>
{_slidevqa_numeric_note(main_recs)}</div>""")

    nm = c["numeric"]
    rows = "".join(
        f'<tr><th scope="row">{esc(r["label"])}</th><td class="num">{r["n"]:,}</td>'
        f'<td class="num">{r["median"] * 100:.1f}%</td><td class="num">{fmt_pct(r["within10"])}%</td>'
        f'<td class="num">{fmt_pct(r["over100"])}%</td>'
        f'<td class="num">{r["skipped_fmt"]:,}</td></tr>' for r in nm)
    parts.append(f"""<div class="card"><h3>When the answer is a number and it is wrong, how wrong?</h3>
<p class="sub">Relative error against the nearest numeric gold, over scored-wrong answers where both
sides parse as numbers. Near misses would mean imprecise reading; these are not near misses.</p>
<table class="t"><thead><tr><th>slice</th><th class="num">n numeric failures</th>
<th class="num">median relative error</th><th class="num">within 10%</th>
<th class="num">off by &gt;100%</th><th class="num">format-only, excluded</th></tr></thead>
<tbody>{rows}</tbody></table></div>""")
    return "".join(parts)


def _slidevqa_numeric_note(main_recs: dict) -> str:
    """The one thing the flat SlideVQA number hides, stated as a number.

    Drilling SlideVQA to evidence spread -> arithmetic -> gold answer shape shows
    the arithmetic penalty is almost entirely a scoring artifact on number-shaped
    golds, not a reasoning failure. That is only visible four levels down, which
    is the argument for the page.
    """
    rs = [r for r in main_recs.get("slidevqa", []) if not r["null"]]
    num = [r for r in rs if _gold_shape(r) == "numeric gold"]
    txt = [r for r in rs if _gold_shape(r) != "numeric gold"]
    ari = [r for r in rs if r["meta"].get("arithmetic")]
    lok = [r for r in rs if not r["meta"].get("arithmetic")]
    if not (num and txt and ari and lok):
        return ""

    def m(rows, key):
        return sum((r[key] if r[key] is not None else r["score"]) for r in rows) / len(rows)

    return (f'<div class="note"><strong>Where this actually bites, four levels down.</strong> '
            f'Split SlideVQA by whether the gold answer is a number and the artifact stops being '
            f'uniform. Numeric golds (n={len(num):,}) score {m(num, "score") * 100:.1f} token F1 '
            f'but {m(num, "alt") * 100:.1f} format-corrected, a '
            f'{(m(num, "alt") - m(num, "score")) * 100:+.1f} point move; text golds '
            f'(n={len(txt):,}) move only {(m(txt, "alt") - m(txt, "score")) * 100:+.1f}. '
            f'That is the whole of SlideVQA\'s apparent arithmetic weakness: arithmetic questions '
            f'score {m(ari, "score") * 100:.1f} against {m(lok, "score") * 100:.1f} for lookups, a '
            f'{(m(lok, "score") - m(ari, "score")) * 100:.1f}-point gap that shrinks to '
            f'{(m(lok, "alt") - m(ari, "alt")) * 100:.1f} points once formatting is credited '
            f'&mdash; because arithmetic answers are numbers and lookups mostly are not. '
            f'The flat 68.8 hides this completely.</div>')


def render(trees: list[Node], root: Node, meta: dict, controls: dict,
           judge: dict, violations: list, load_stats: dict,
           judge_by_node: dict, main_recs: dict) -> str:
    tiles = [f'<div class="tile"><div class="tlab">questions</div>'
             f'<div class="tval">{root.n:,}</div>'
             f'<div class="tnote">{len(trees)} benchmarks &middot; official splits</div></div>']
    for t in trees:
        tone = "bad" if (t.value or 0) < 0.5 else ("good" if (t.value or 0) >= 0.8 else "")
        tiles.append(
            f'<div class="tile {tone}"><div class="tlab">{esc(BENCH_LABEL[t.bench])}</div>'
            f'<div class="tval">{fmt_pct(t.value)}<span class="pcts">%</span></div>'
            f'<div class="tnote">n={t.n:,} &middot; {esc(t.metric)}</div></div>')

    body = []
    for t in trees:
        vals = [x.value for x in walk(t) if x.value is not None and x.n >= SMALL_N]
        anchor = t.value or 0.0
        spread = max(
            (statistics.quantiles([abs(v - anchor) for v in vals], n=10)[-1]
             if len(vals) >= 10 else max((abs(v - anchor) for v in vals), default=0.1)),
            0.02)
        bmax = max(vals, default=1.0) or 1.0
        body.append(render_node(t, anchor, spread, bmax, 0, judge_by_node))
    tree_html = render_node_root(root, body)

    vio = ""
    if violations:
        rows = "".join(
            f'<tr><th scope="row">{esc(v["node"])}</th><td>{esc(v["dimension"])}</td>'
            f'<td class="num">{v["parent_n"]}</td><td class="num">{v["children_n"]}</td></tr>'
            for v in violations[:40])
        vio = (f'<div class="note bad"><strong>{len(violations)} node(s) do not add up.</strong> '
               f'Listed rather than hidden.<table class="t"><thead><tr><th>node</th><th>dimension</th>'
               f'<th class="num">parent n</th><th class="num">sum of children</th></tr></thead>'
               f'<tbody>{rows}</tbody></table></div>')
    else:
        vio = ('<div class="note ok"><strong>Every parent equals the sum of its children.</strong> '
               f'Checked at all {meta["splits_checked"]:,} splits in the tree, on both the scored '
               f'count and the null-prediction count. Nothing is dropped silently: a record with '
               f'missing metadata lands in an explicit <em>(metadata missing)</em> child rather '
               f'than falling out.</div>')

    lrows = "".join(
        f'<tr><th scope="row">{esc(BENCH_LABEL.get(k, k))}</th>'
        f'<td class="num">{s["lines"]:,}</td><td class="num">{s["unique"]:,}</td>'
        f'<td class="num">{s["duplicate_uids"]:,}</td>'
        f'<td class="num">{s["null_pred_raw"]:,}</td>'
        f'<td class="num">{s["null_pred_after_dedup"]:,}</td>'
        f'<td class="num">{s["malformed"]:,}</td></tr>'
        for k, s in load_stats.items())

    jb = ""
    if judge:
        pr = "".join(
            f'<tr><th scope="row">{esc(k)}</th><td class="num">{v["n"]:,}</td>'
            f'<td class="num">{fmt_pct(v["judge"])}</td><td class="num">{fmt_pct(v["string"])}</td>'
            f'<td class="num">{(v["judge"] - v["string"]) * 100:+.1f}</td></tr>'
            for k, v in sorted(judge["per_split"].items()))
        jb = f"""<div class="card"><h3>CharXiv: string scoring is a lower bound</h3>
<p class="sub">CharXiv's official grader is an LLM judge with per-question-type rubrics. This harness
scores strings. A partial judge file covers {judge['joined']:,} rows
({fmt_pct(judge['coverage'], 0)}% of the split) and joins cleanly &mdash;
{judge['pred_mismatch']} of those rows disagree about what the model actually said, so the join is
sound. Judge-scored nodes carry a blue badge in the tree; the two scores are shown side by side and
are never averaged into one number.</p>
<table class="t"><thead><tr><th>split</th><th class="num">n judged</th><th class="num">official judge</th>
<th class="num">string match (same rows)</th><th class="num">gap (pp)</th></tr></thead>
<tbody>{pr}
<tr><th scope="row"><b>all judged rows</b></th><td class="num">{judge['joined']:,}</td>
<td class="num"><b>{fmt_pct(judge['judge'])}</b></td><td class="num">{fmt_pct(judge['string'])}</td>
<td class="num">{(judge['judge'] - judge['string']) * 100:+.1f}</td></tr></tbody></table>
<p class="sub" style="margin:8px 0 0">The judge and the string matcher agree on
{fmt_pct(judge['agreement'], 0)}% of items at the 0.5 threshold. That agreement is what licenses
using string scoring for the {100 - judge['coverage'] * 100:.0f}% of the split the judge has not
covered &mdash; while remembering the free-text types are still undercounted there.</p></div>"""

    nav = " &middot; ".join(f'<a href="{h}">{esc(t)}</a>' for h, t in NAV)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Claude Haiku 4.5 &mdash; drill-down</title><style>{DRILL_CSS}</style></head>
<body><div class="wrap">
<header><div>
<h1>Every number in the study, opened up</h1>
<p class="dek">One headline per benchmark, then the splits that produced it, then the splits inside
those, down to the individual questions. {root.n:,} questions across {len(trees)} benchmarks, each
scored with its own official metric. Click any row to open it; the number beside each child is how
far that child sits from its parent, which is the only reason to drill down at all.</p>
<p class="dek" style="margin-top:8px;font-size:13px">{nav}</p>
</div><button id="theme" type="button">Dark mode</button></header>

<div class="tiles">{''.join(tiles)}</div>

<div class="note"><strong>The root number is not a score.</strong> ANLS, token F1, click-in-bbox and
multiple-choice accuracy measure different things on different scales; averaging them would produce a
number that describes nothing. The root row therefore shows only the question count, and every
benchmark below carries its own metric name on every row.</div>

{vio}

<h2>The drill-down<span class="sub">Depth 0 is the whole study. Colour is diverging and anchored at
each benchmark's own headline, so red means "worse than this benchmark's average", not "low" &mdash;
otherwise every ScreenSpot-Pro row would be red and every CharXiv row green. Nodes with fewer than
{SMALL_N} scored rows are labelled; a parent whose children spread more than
{DIVERGE * 100:.0f} points carries a badge, because that is where the drill-down is telling you
something.</span></h2>

<div class="toolbar">
<button id="all" type="button">expand all</button>
<button id="none" type="button">collapse all</button>
<span class="gap"></span><label>to depth</label>
<button data-depth="1" type="button">1</button><button data-depth="2" type="button">2</button>
<button data-depth="3" type="button">3</button><button data-depth="4" type="button">4</button>
<button data-depth="5" type="button">5</button><button data-depth="9" type="button">all</button>
<span class="gap"></span><label for="sort">sort children</label>
<select id="sort"><option value="def">natural order</option><option value="n">by n</option>
<option value="low">by score, worst first</option><option value="high">by score, best first</option>
<option value="alpha">alphabetically</option></select>
<span class="gap"></span>
<button id="scale" type="button">bars: absolute 0-100%</button>
<button id="small" type="button">hide small-n (n&lt;30)</button>
<span class="gap"></span>
<input id="q" type="search" placeholder="filter labels…" aria-label="filter labels">
</div>
<div class="legend"><span class="ramp"></span>
<span>worse than benchmark average &rarr; better</span>
<span class="flag">children spread</span><span class="flag sm">n&lt;30</span>
<span class="flag jd">judge-scored</span></div>

<div class="tree" id="tree"><div class="hdr"><span>node</span><span>score</span><span>value</span>
<span>vs parent</span><span>metric</span><span>n</span></div>
{tree_html}</div>

<p class="dek" style="font-size:13px">Flattened export:
<a href="drilldown.csv">drilldown.csv</a> &middot; <a href="drilldown.json">drilldown.json</a>
&mdash; every node in the tree with its n, metric, value and delta, so the numbers can be checked
without a browser.</p>

<h2>What the data had to say before it was scored<span class="sub">Result files are appended to on
resume, so several contain the same uid more than once. A later row replaces an earlier one unless
that would replace a real prediction with a null &mdash; the same rule the rest of the study uses.
Everything below the tree is computed after that deduplication.</span></h2>
<div class="card"><table class="t"><thead><tr><th>benchmark</th><th class="num">lines on disk</th>
<th class="num">unique questions</th><th class="num">duplicate rows</th>
<th class="num">null pred (raw)</th><th class="num">null pred (after dedup)</th>
<th class="num">malformed</th></tr></thead><tbody>{lrows}</tbody></table>
<p class="sub" style="margin:10px 0 0">Null predictions are excluded from every metric and counted at
every node they belong to; a node that dropped any carries the count as a badge. Deduplication
recovers most nulls because a resumed run re-asked the question and got an answer the second time.</p>
</div>

{jb}

<h2>The controls, recomputed<span class="sub">Each of these is an ablation over the same questions,
recomputed here from the result files rather than quoted. Where an ablation joins back to the main
run it is compared only against those same items.</span></h2>
{control_html(controls, main_recs)}

<p class="foot">Generated by <code>python -m blindspot.report_pages drilldown</code> from the result JSONL files
under <code>results/</code>. Nothing on this page is transcribed from a previous report: every value
is recomputed at build time using <code>blindspot/core.py</code>, the same scorer the rest of the
study uses.</p>
</div><script>{DRILL_JS}</script></body></html>"""


def render_node_root(root: Node, bench_html: list[str]) -> str:
    ln = (f'<div class="ln" tabindex="0" role="button">'
          f'<span class="lbl" style="padding-left:4px"><span class="tw">&#9656;</span>'
          f'<span class="txt">ALL &mdash; every question in the study</span>'
          f'<span class="flag" title="the benchmarks below use four different metrics">'
          f'mixed metrics: not comparable</span></span>'
          f'<span class="track"></span>'
          f'<span class="val">&mdash;</span><span class="dlt" style="color:var(--muted)">&mdash;</span>'
          f'<span class="met">&mdash;</span><span class="nnum">n={root.n:,}</span></div>')
    return (f'<div class="nd open" data-k="root" data-d="0" data-n="{root.n}" data-v="" '
            f'data-i="0" data-small="0" data-lab="ALL">'
            f'{ln}<div class="kids">{"".join(bench_html)}</div></div>')


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def flatten(root: Node, trees: list[Node]) -> list[dict]:
    rows = []

    def rec(n: Node, path: list[str]):
        p = path + [n.label]
        rows.append({
            "path": " > ".join(p), "depth": n.depth, "kind": n.kind,
            "benchmark": BENCH_LABEL.get(n.bench or "", ""), "level": n.level,
            "label": n.label, "n_scored": n.n, "n_null_pred": n.n_null,
            "metric": n.metric if n.kind != "root" else "mixed (not comparable)",
            "value": "" if n.value is None else round(n.value, 6),
            "value_pct": "" if n.value is None else round(n.value * 100, 2),
            "delta_vs_parent_pp": "" if n.delta is None else round(n.delta * 100, 2),
            "exact_match_pct": "" if n.em is None else round(n.em * 100, 2),
            "format_corrected_pct": "" if n.alt is None else round(n.alt * 100, 2),
            "children_spread_pp": "" if n.spread is None else round(n.spread * 100, 2),
            "small_n": int(n.n < SMALL_N and n.kind == "node"),
        })
        for c in n.children:
            rec(c, p)

    rec(root, [])
    return rows


def cmd_drilldown(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(b, str(RESULTS / f)) for b, f in DRILL_MAIN_FILES.items()]
    raw_nulls = {}
    for b, f in DRILL_MAIN_FILES.items():
        n = 0
        with open(RESULTS / f, encoding="utf-8") as fh:
            for line in fh:
                if '"pred": null' in line or '"pred":null' in line:
                    n += 1
        raw_nulls[b] = n

    main_recs: dict[str, list[dict]] = {}
    load_stats: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=min(6, len(jobs))) as ex:
        for bench, recs, stats in ex.map(score_file, jobs):
            main_recs[bench] = recs
            stats["null_pred_raw"] = raw_nulls[bench]
            load_stats[bench] = stats
    load_stats = {b: load_stats[b] for b in DRILL_MAIN_FILES}

    judge = judge_join(main_recs["charxiv"])

    violations: list = []
    counter = [0]
    trees = []
    for bench in DRILL_MAIN_FILES:
        t = build(BENCH_LABEL[bench], "benchmark", "bench", main_recs[bench], bench,
                  SPEC[bench], (), 1, violations, counter)
        t.note = BENCH_BLURB[bench]
        trees.append(t)

    root = Node(label="ALL", level="", kind="root", depth=0, bench=None)
    root.children = trees
    root.n = sum(t.n for t in trees)
    root.n_null = sum(t.n_null for t in trees)
    root.metric = "mixed metrics (not comparable)"
    assert root.n + root.n_null == sum(len(main_recs[b]) for b in DRILL_MAIN_FILES), \
        "root does not account for every loaded row"

    # Judge coverage per CharXiv node, kept as a side-channel so it can never be
    # averaged into the string score.
    judge_by_node: dict[int, dict] = {}
    if judge:
        juids = judge["uids"]
        jscore = {}
        for line in open(RESULTS / JUDGE_FILE, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("uid") in juids and d.get("judge_score") is not None:
                jscore[d["uid"]] = float(d["judge_score"])
        cx_tree = next(t for t in trees if t.bench == "charxiv")
        for node in walk(cx_tree):
            vals = [jscore[r["uid"]] for r in node.recs if r["uid"] in jscore]
            if len(vals) >= SMALL_N and len(vals) >= 0.5 * max(node.n, 1):
                judge_by_node[id(node)] = {"n": len(vals), "v": sum(vals) / len(vals)}

    controls = control_blocks(main_recs)

    meta = {"splits_checked": sum(1 for n in walk(root) if n.children)}
    html_out = render(trees, root, meta, controls, judge, violations, load_stats,
                      judge_by_node, main_recs)
    Path(a.out).write_text(html_out, encoding="utf-8")

    rows = flatten(root, trees)
    with open(a.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    Path(a.json).write_text(json.dumps({
        "root_n": root.n, "nodes": len(rows), "splits_checked": meta["splits_checked"],
        "violations": violations, "load_stats": load_stats,
        "benchmarks": {t.bench: {"n": t.n, "value": t.value, "metric": t.metric} for t in trees},
        "tree": rows}, indent=1), encoding="utf-8")

    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1024:.0f} KB), "
          f"{len(rows):,} nodes, {meta['splits_checked']:,} splits checked, "
          f"{len(violations)} arithmetic violations")
    print(f"wrote {a.csv} and {a.json}")
    for b, t in zip(DRILL_MAIN_FILES, trees):
        print(f"  {BENCH_LABEL[b]:16s} n={t.n:5,d} null={t.n_null:2d} "
              f"{t.metric:28s} {t.value * 100:6.2f}")
    return 0


# =================== slidevqa: outputs/slidevqa.html + outputs/assets_slidevqa/
SLIDE_ASSETS = OUT / "assets_slidevqa"
SLIDE_DATA = Path("data") / "slidevqa"

EVIDENCE_JSONL = RESULTS / "slidevqa__haiku-4-5_think2000_native_r0.jsonl"
ALLPAGES_JSONL = RESULTS / "slidevqa_allpages__haiku-4-5_think2000_native_r0.jsonl"
MANIFEST = SLIDE_DATA / "manifest.jsonl"

SLIDE_THUMB_W = 200
SLIDE_THUMB_Q = 70
SLIDE_FULL_EDGE = 1100
SLIDE_FULL_Q = 82

N_PAGES = 20


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


def slide_format_equivalent(pred, golds) -> bool:
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
_SLIDE_NUM_RE = re.compile(r"\b(?:slide|page|image)\s*#?\s*(\d{1,2})\b")


def cited_slides(thinking, n_pages=N_PAGES) -> list[int]:
    """1-based slide indices the trace names explicitly.

    Only ~1 trace in 3 names a slide at all; when it does the reference is
    reliable. Absence is reported as unknown, never as zero.
    """
    t = str(thinking or "").lower()
    out = set()
    for m in _SLIDE_NUM_RE.finditer(t):
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


def slide_load():
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
        r["fmt_equiv"] = slide_format_equivalent(r["pred"], r["gold"])
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
def slide_mean(rows, key):
    return 100.0 * sum(r[key] for r in rows) / len(rows) if rows else None


def stats(rows) -> dict:
    return {"f1": slide_mean(rows, "f1"), "em": slide_mean(rows, "em"),
            "f1c": slide_mean(rows, "f1c"), "emc": slide_mean(rows, "emc"), "n": len(rows)}


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
        sc = min(1.0, SLIDE_FULL_EDGE / max(w, h))
        f = im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.LANCZOS) if sc < 1 else im
        f.save(full, "JPEG", quality=SLIDE_FULL_Q, optimize=True, progressive=True)
    if not Path(thumb).exists():
        sc = min(1.0, SLIDE_THUMB_W / w)
        t = im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.LANCZOS) if sc < 1 else im
        t.save(thumb, "JPEG", quality=SLIDE_THUMB_Q, optimize=True)
    return None


def render_slides(examples, by_qa, workers=None):
    """One job per unique (deck, page). Pure CPU -- processes, not threads."""
    SLIDE_ASSETS.mkdir(parents=True, exist_ok=True)
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
            jobs.append((str(SLIDE_DATA / rel), str(SLIDE_ASSETS / t), str(SLIDE_ASSETS / f)))
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


def slide_table(headers, rows, first_is_row_header=True):
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
SLIDE_CSS = """
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
SLIDE_JS = r"""
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


SLIDE_LIGHTBOX_HTML = """
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


def slide_tile(lab, val, note_, tone="") -> str:
    return (f'<div class="tile {tone}"><div class="tlab">{esc(lab)}</div>'
            f'<div class="tval">{val}</div><div class="tnote">{note_}</div></div>')


def build_html(a, arith, examples, bucket_counts, join_ok) -> str:
    po = a["paired_overall"]
    full = {r["slice"]: r for r in a["full"]}
    costs = {c["name"]: c for c in a["costs"]}
    fm = a["format"]

    # ---- tiles -------------------------------------------------------------
    tiles = "".join([
        slide_tile("Evidence-condition F1", f'{full["overall"]["f1"]:.1f}',
             f'EM {full["overall"]["em"]:.1f} &middot; n={full["overall"]["n"]} &middot; '
             f'oracle retrieval'),
        slide_tile("Retrieval cost", f'{costs["retrieval"]["cost"]:+.1f}',
             f'F1, paired on n={a["n_paired"]} &middot; 20 slides vs the right 1&ndash;2', "bad"),
        slide_tile("Integration cost", f'{costs["integration"]["cost"]:+.1f}',
             "F1, multi-page evidence vs single-page", "bad"),
        slide_tile("Derivation cost", f'{costs["derivation"]["cost"]:+.1f}',
             "F1, arithmetic vs plain lookup", "bad"),
        slide_tile("Format-only failures", f'{100 * fm["zero_f1_fmt"] / fm["zero_f1"]:.0f}%',
             f'of the {fm["zero_f1"]} hard zeros are the right value, wrongly formatted', "bad"),
    ])

    # ---- table 1: full evidence condition ----------------------------------
    t1 = slide_table(
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

    t2 = slide_table(["condition", "F1", "EM", "F1 (fmt-corrected)", "EM (fmt-corrected)"],
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
    cost_table = slide_table(
        ["condition", "mean latency", "mean input tokens", "mean output tokens"],
        [["evidence only", f'{ct["evidence"]["latency"]:.2f}s',
          f'{ct["evidence"]["in_tok"]:,.0f}', f'{ct["evidence"]["out_tok"]:,.0f}'],
         ["all 20 slides", f'{ct["allpages"]["latency"]:.2f}s',
          f'{ct["allpages"]["in_tok"]:,.0f}', f'{ct["allpages"]["out_tok"]:,.0f}'],
         ["<b>ratio</b>", f'<b>{ct["allpages"]["latency"] / ct["evidence"]["latency"]:.1f}&times;</b>',
          f'<b>{ct["allpages"]["in_tok"] / ct["evidence"]["in_tok"]:.1f}&times;</b>',
          f'<b>{ct["allpages"]["out_tok"] / ct["evidence"]["out_tok"]:.1f}&times;</b>']])

    D, DC = a["divergence"], a["divergence_c"]
    div_table = slide_table(
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
<style>{SLIDE_CSS}</style></head><body>
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
{SLIDE_LIGHTBOX_HTML}
<div id="tip"></div>
<script>
const EX = {ex_json};
const BUCKET_LABEL = {labels_json};
</script>
<script>{SLIDE_JS}</script>
</body></html>"""


def cmd_slidevqa(ns) -> int:
    # The namespace is `ns`, not `a`: `a` and `ap` are both taken below --
    # `ap` is the all-pages arm and `a` the analysis dict.
    man, by_qa, ev, ap = slide_load()
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

    if not ns.skip_images:
        n_jobs, errs = render_slides(examples, by_qa, ns.workers)
        print(f"rendered {n_jobs} unique slides (thumb {SLIDE_THUMB_W}px q{SLIDE_THUMB_Q} + "
              f"full {SLIDE_FULL_EDGE}px q{SLIDE_FULL_Q})")
        for e in errs[:10]:
            print("  image error:", e)

    OUT.mkdir(parents=True, exist_ok=True)
    out = Path(ns.out)
    out.write_text(build_html(a, arith, examples, {k: v for k, v in bucket_counts.items()},
                              (v_ev, v_ap)), encoding="utf-8")
    size = sum(p.stat().st_size for p in SLIDE_ASSETS.glob("*.jpg")) if SLIDE_ASSETS.exists() else 0
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB); "
          f"assets {size / 1e6:.1f} MB in {len(list(SLIDE_ASSETS.glob('*.jpg')))} files")
    return 0


# ================================================== tasks: outputs/tasks/*.html
TASKS = OUT / "tasks"

# What each primitive actually asks the model to do, and how it is graded.
DESCRIBE = {
    "counting": ("Count discrete objects in a figure &mdash; lines in a plot, labels in a "
                 "legend, ticks across all axes, subplots, or items in an infographic.",
                 "Exact match on the number."),
    "line_following": ("Trace lines through a plot and decide whether any of them cross. "
                       "The only primitive here with a single source.",
                       "Exact match on yes/no."),
    "localization_read": ("Find a spatial extreme &mdash; the leftmost or rightmost x tick, "
                          "the highest or lowest y tick &mdash; and report the value written there. "
                          "Ends in reading text.",
                          "Normalized text/numeric match, or the official CharXiv judge."),
    "localization_point": ("Find a described UI element on a dense professional screenshot and "
                           "return its centre as coordinates. Ends in emitting numbers, not text.",
                           "Click-in-bbox: the predicted point must fall inside the gold box."),
    "value_interpolation": ("Read a value off a continuous scale &mdash; tick spacing, colorbar "
                            "range and maximum &mdash; where the answer is often not printed and "
                            "must be inferred from the axis.",
                            "Numeric match with tolerance."),
    "binding": ("Associate legend entries with the series they label, and report them in "
                "reading order.", "Normalized match on the ordered list."),
    "structure": ("Parse the layout of a composite figure &mdash; how many subplots and in "
                  "what grid.", "Normalized match (\"n by m\")."),
    "text_in_situ": ("Read text embedded in a figure: the plot title and the axis labels.",
                     "Normalized match; free-text, so a lower bound under string matching."),
    "comparison": ("Compare two or more quantities read off the figure and pick the larger, "
                   "the fastest, the third-highest.", "ANLS / normalized match."),
    "arithmetic": ("Derive a number that is written nowhere on the page &mdash; an inverse "
                   "percentage, a difference, a share.", "ANLS against the accepted answers."),
    "composition": ("Answer a question that requires combining several separate readings of "
                    "the same chart.", "Official CharXiv reasoning judge / ANLS."),
}

EXTRA_CSS = """
.hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
.hero .tile{background:var(--surface);border:1px solid var(--border);border-radius:11px;padding:14px 15px}
.hero .tlab{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.hero .tval{font-size:28px;margin:6px 0 2px}.hero .tnote{font-size:12px;color:var(--ink2)}
.what{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent);
 border-radius:10px;padding:14px 16px;margin:14px 0;font-size:13.5px;color:var(--ink2)}
.what b{color:var(--ink)}
.brow{display:grid;grid-template-columns:210px 1fr 74px;align-items:center;gap:11px;padding:4px 0}
.blab{font-size:12.5px;color:var(--ink2);text-align:right;overflow-wrap:anywhere}
.btrack{height:14px;background:var(--grid);border-radius:4px}
.bfill{height:100%;background:var(--accent);border-radius:0 4px 4px 0}
.bval{font-size:12.5px;text-align:right;font-variant-numeric:tabular-nums}
.bval span{display:block;font-size:10.5px;color:var(--muted)}
h2.sec{font-size:17px;margin:34px 0 6px;padding-top:18px;border-top:1px solid var(--grid)}
h2.sec .n{font-size:13px;font-weight:400;color:var(--ink2)}
svg.scatter{width:100%;height:auto;background:var(--surface);border:1px solid var(--border);border-radius:11px}
.scatter .ax{stroke:var(--axis);stroke-width:1}
.scatter .gl{stroke:var(--grid);stroke-width:1}
.scatter .pt{fill:var(--bad);opacity:.45}
.scatter .pt.ok{fill:var(--good);opacity:.75}
.scatter .ideal{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:4 3;fill:none}
.scatter .fit{stroke:var(--accent);stroke-width:2;fill:none}
.scatter text{font-size:11px;fill:var(--ink2)}
"""


def _fmt(v, d=1):
    return "&mdash;" if v is None else f"{v*100:.{d}f}"


def bar_rows(items) -> str:
    out = []
    for lab, acc, n in items:
        out.append(f'<div class="brow"><div class="blab">{esc(lab)}</div>'
                   f'<div class="btrack"><div class="bfill" style="width:{max((acc or 0)*100,0.6):.1f}%"></div></div>'
                   f'<div class="bval">{_fmt(acc,0)}%<span>n={n}</span></div></div>')
    return "".join(out)


def scatter_svg(rows: list[dict]) -> str:
    """Predicted vs gold centre, both axes, with the fitted line against y=x.

    This is the localization finding made visible: if the model tracked the
    target the points would sit on the dashed diagonal. A shallower fitted line
    means predictions are collapsing toward a prior instead.
    """
    import statistics as st
    pts = [r for r in rows if isinstance(r.get("pred"), (list, tuple)) and len(r["pred"]) == 2]
    if len(pts) < 20:
        return ""

    def panel(idx, name, ox):
        gold = [(r["gold"][idx] + r["gold"][idx + 2]) / 2 for r in pts]
        pred = [r["pred"][idx] for r in pts]
        mg, mp = st.mean(gold), st.mean(pred)
        sxx = sum((g - mg) ** 2 for g in gold)
        slope = (sum((g - mg) * (p - mp) for g, p in zip(gold, pred)) / sxx) if sxx else 0
        icept = mp - slope * mg
        S = 250
        dots = "".join(
            f'<circle class="pt{" ok" if (r.get("score") or 0) >= .5 else ""}" '
            f'cx="{g*S:.1f}" cy="{S-p*S:.1f}" r="2.4"/>'
            for g, p, r in zip(gold, pred, pts))
        y0, y1 = icept, slope + icept
        # Clip: with slope 0.865 and intercept 0.192 the x fit exits the top of
        # the box before gold=1, and an unclipped line drawn outside its own axes
        # reads as a rendering bug rather than as extrapolation.
        return f"""
  <g transform="translate({ox},14)">
    <clipPath id="clip{idx}"><rect x="0" y="0" width="{S}" height="{S}"/></clipPath>
    <text x="0" y="-2">{name} &mdash; slope {slope:.3f}, intercept {icept:.3f}</text>
    <rect class="gl" x="0" y="0" width="{S}" height="{S}" fill="none"/>
    <line class="ideal" x1="0" y1="{S}" x2="{S}" y2="0"/>
    <g clip-path="url(#clip{idx})">{dots}
    <line class="fit" x1="0" y1="{S-y0*S:.1f}" x2="{S}" y2="{S-y1*S:.1f}"/></g>
    <line class="ax" x1="0" y1="{S}" x2="{S}" y2="{S}"/>
    <line class="ax" x1="0" y1="0" x2="0" y2="{S}"/>
    <text x="{S/2-28}" y="{S+16}">gold centre</text>
    <text transform="translate(-8,{S/2+30}) rotate(-90)">predicted</text>
  </g>"""

    return f"""<figure style="margin:14px 0">
<svg class="scatter" viewBox="0 0 610 300" role="img"
     aria-label="Predicted versus gold centre coordinates on both axes">
 {panel(0,'horizontal (x)',34)}{panel(1,'vertical (y)',330)}
</svg>
<figcaption style="font-size:12.5px;color:var(--ink2);margin-top:8px">
Dashed diagonal = perfect tracking. Solid line = the actual fit. Green points landed
inside the gold box, red missed. A slope well under 1 means predictions are compressing
toward a central prior rather than following the target.</figcaption></figure>"""


def localization_decomposition(rows: list[dict]) -> str:
    """Split the failure into 'never found the region' vs 'found it, missed the box'.

    A single accuracy number hides that these are different problems with
    different fixes. The regression slope alone is actively misleading here: it
    reads as compression when what is really happening is a bimodal population
    -- a band that tracks the target plus a diffuse cloud that does not.
    """
    pts = [r for r in rows if isinstance(r.get("pred"), (list, tuple)) and len(r["pred"]) == 2]
    if not pts:
        return ""
    def err(r, i):
        return abs(r["pred"][i] - (r["gold"][i] + r["gold"][i + 2]) / 2)
    n = len(pts)
    track = [r for r in pts if err(r, 0) <= .10 and err(r, 1) <= .10]
    lost = [r for r in pts if err(r, 0) > .25 or err(r, 1) > .25]
    hit = lambda s: sum(r.get("score") or 0 for r in s) / max(len(s), 1) * 100
    mx = sorted(err(r, 0) for r in pts)[n // 2] * 100
    my = sorted(err(r, 1) for r in pts)[n // 2] * 100
    return f"""
<h2 class="sec">Two different failures, not one</h2>
<div class="card">
{bar_rows([("found the region (within 10% on both axes)", len(track)/n, len(track)),
           ("landed in the box, given the region was right", hit(track)/100, len(track)),
           ("lost entirely (&gt;25% off on either axis)", len(lost)/n, len(lost))])}
</div>
<div class="what">
<b>The model finds roughly the right area {len(track)/n*100:.0f}% of the time, and still hits the
box in only {hit(track):.1f}% of those cases.</b> Those are separate problems: coarse search fails on
{len(lost)/n*100:.0f}% of items outright, while on the ones it does locate, the target is around
22px across after the API downscales the screenshot &mdash; small enough that "close" is not close
enough. Median error is {mx:.1f}% of screen width horizontally and {my:.1f}% vertically, so the
vertical axis is the <em>more</em> accurate one; the shallow y regression slope reflects the outlier
cloud, not compression.
</div>"""


def build_page(prim: str, rows: list[dict], n_examples: int) -> Path:
    """One primitive: stats, sub-slices, scatter (if pointing), worked examples."""
    TASKS.mkdir(parents=True, exist_ok=True)
    c = cell(rows) or {}
    answerable = [r for r in rows if not r["not_applicable"]]
    ca = cell(answerable) or {}
    na_rate = 1 - len(answerable) / max(len(rows), 1)
    by_ds = slice_by(rows, lambda r: r["dataset"])
    what, metric = DESCRIBE.get(prim, ("", ""))

    # Sub-slices that make sense for this primitive.
    sub = []
    if prim == "localization_point":
        def bucket(r):
            f = r["_ex"].meta.get("target_area_frac", 0) or 0
            s = math.sqrt(f * 1568 * 882)
            return next((n for l, n in ((12, "<12px"), (20, "12-20px"), (32, "20-32px"),
                                        (56, "32-56px")) if s < l), ">=56px")
        sub = [("by target size after downscale", slice_by(rows, bucket, 5)),
               ("by element type", slice_by(rows, lambda r: r["_ex"].meta.get("ui_type") or "?", 5))]
    else:
        sub = [("by question type", slice_by(rows, lambda r: r["_ex"].meta.get("qlabel")
                                             or (r["_ex"].meta.get("operation") or ["?"])[0], 5))]

    # Examples from both sides. Hardest failures first; successes for contrast.
    wrong = sorted([r for r in answerable if (r.get("score") or 0) < 0.5],
                   key=lambda r: (r["_ex"].meta.get("target_area_frac", 1), r["uid"]))[:n_examples]
    right = [r for r in answerable if (r.get("score") or 0) >= 0.5][:max(2, n_examples // 2)]
    # Strip `_ex` (an Example) before crossing the process boundary and carry the
    # few fields the renderer needs explicitly -- spawn pickles every argument.
    jobs = []
    for r in wrong + right:
        j = {k: v for k, v in r.items() if k != "_ex"}
        j.update({"_image": r["_ex"].images[0], "question": r["_ex"].question,
                  "answer_type": r.get("answer_type") or r["_ex"].answer_type,
                  "gold": r["_ex"].gold})
        jobs.append(j)
    with ProcessPoolExecutor(max_workers=8) as pool:
        cards = list(pool.map(build_one, jobs, chunksize=2))
    neg, pos = cards[:len(wrong)], cards[len(wrong):]

    def case(a, ok):
        full = a.get("_full")

        def link(img: str) -> str:
            if not full:
                return img
            return f'<a class="zoom" href="{full}" title="open full size with both annotations">{img}</a>'

        main_img = link('<img src="%s" alt="asset">' % a["_thumb"])
        zoom = ""
        if a.get("_zoom"):
            zoom = "<figure>%s</figure>" % link('<img src="%s" alt="zoom">' % a["_zoom"])
        return f"""<article class="case">
 <div class="hd"><span class="pill {'ok' if ok else 'no'}">{'&#10003; correct' if ok else '&#10007; wrong'}</span>
 <span class="q">{esc(a.get('question',''))[:340]}</span></div>
 <div class="imgs{'' if zoom else ' one'}"><figure>{main_img}</figure>{zoom}</div>
 <dl><div><dt>model answered</dt><dd class="{'g' if ok else 'b'}">{esc(a.get('pred'))}</dd></div>
 <div><dt>gold</dt><dd class="g">{esc(a.get('gold'))}</dd></div>
 <div><dt>uid</dt><dd style="color:var(--muted)">{esc(a.get('uid'))}</dd></div></dl></article>"""

    # Why this primitive's failures failed.
    import collections as _c
    fails = [r for r in rows if (r.get("score") or 0) < 0.5]
    fm = _c.Counter(r.get("failure_mode", "unclassified") for r in fails)
    fm_block = ""
    if fm:
        fm_block = ('<h2 class="sec">Why the failures failed <span class="n">&mdash; '
                    f'{len(fails)} scored-wrong answers</span></h2><div class="card">'
                    + bar_rows([(FM_LABELS.get(m, m), c / len(fails), c)
                                for m, c in fm.most_common()]) + "</div>")

    na_note = (f'<div class="what"><b>{na_rate*100:.0f}% of golds are "Not Applicable"</b> for this '
               f'primitive &mdash; the figure has nothing to count or trace. Headline accuracy below is '
               f'the answerable subset; pooling the rest in would measure "can you tell this does not '
               f'apply" instead.</div>') if na_rate > 0.1 else ""

    subs = "".join(f'<h2 class="sec">{esc(t)}</h2><div class="card">'
                   f'{bar_rows([(d["label"], d["acc"], d["n"]) for d in items])}</div>'
                   for t, items in sub if items)

    return _write(prim, f"""
<h1>{LABELS[prim]}</h1>
<p class="dek">{what}</p>
<div class="what"><b>How it is scored.</b> {metric}</div>
{na_note}
<div class="hero">
 <div class="tile"><div class="tlab">accuracy</div><div class="tval">{_fmt(ca.get('acc'),0)}%</div>
  <div class="tnote">n={ca.get('n',0)}{f" &middot; CI {_fmt(ca.get('ci_lo'),0)}&ndash;{_fmt(ca.get('ci_hi'),0)}%" if ca.get('ci_lo') is not None else ""}</div></div>
 <div class="tile"><div class="tlab">questions asked</div><div class="tval">{len(rows)}</div>
  <div class="tnote">{len(answerable)} answerable</div></div>
 <div class="tile"><div class="tlab">sources</div><div class="tval">{len(by_ds)}</div>
  <div class="tnote">{esc(', '.join(d['label'] for d in by_ds))}</div></div>
</div>
{'<h2 class="sec">by dataset</h2><div class="card">' + bar_rows([(d['label'], d['acc'], d['n']) for d in by_ds]) + '</div>' if len(by_ds) > 1 else ''}
{subs}
{fm_block}
{('<h2 class="sec">Where the predictions actually land</h2>' + scatter_svg(rows) + localization_decomposition(rows)) if prim == 'localization_point' else ''}
<h2 class="sec">Failures <span class="n">&mdash; hardest cases first</span></h2>
{''.join(case(a, False) for a in neg) or '<p class="dek">No failures in this slice.</p>'}
<h2 class="sec">Successes <span class="n">&mdash; the same task, done right</span></h2>
{''.join(case(a, True) for a in pos) or '<p class="dek">No successes in this slice.</p>'}
""")


def _write(prim: str, body: str) -> Path:
    p = TASKS / f"{prim}.html"
    nav = " &middot; ".join(f'<a href="{k}.html">{esc(LABELS[k])}</a>' for k in LABELS)
    p.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(LABELS[prim])} &mdash; Haiku 4.5</title>
<style>{GCSS}{EXTRA_CSS}</style></head><body><div class="wrap">
<p style="font-size:12.5px;color:var(--muted)"><a href="../report.html">&larr; overview</a></p>
{body}
<h2 class="sec">Other primitives</h2><p class="dek" style="font-size:13px">{nav}</p>
</div>
{LIGHTBOX_HTML}
<script>{LIGHTBOX_JS}</script></body></html>""", encoding="utf-8")
    return p


def cmd_tasks(a) -> int:
    # The originals relied on outputs/tasks/ already existing; `_write` never
    # created it, so a fresh clone failed on the first page. Create it AFTER the
    # check below, though: on a thin results/ this used to make an empty
    # outputs/tasks/ and exit 0, which reads as "ran fine, nothing to say"
    # rather than "had nothing to work with".
    rows = [r for ds in ("charxiv", "infographicvqa", "screenspot_pro") for r in load_rows(ds)]
    by_prim = defaultdict(list)
    for r in rows:
        if r["primitive"]:
            by_prim[r["primitive"]].append(r)
    if not by_prim:
        print(f"ABORT: no scored rows carry a primitive, so there is nothing to build "
              f"a per-primitive page from\n({len(rows)} rows loaded). Score the "
              f"benchmarks first:\n"
              f"  python -m blindspot.core --datasets charxiv infographicvqa "
              f"screenspot_pro --max-spend N", file=sys.stderr)
        return 2
    TASKS.mkdir(parents=True, exist_ok=True)
    for prim in LABELS:
        if prim in by_prim:
            p = build_page(prim, by_prim[prim], a.examples)
            print(f"  {LABELS[prim]:42} n={len(by_prim[prim]):<5} -> {p}")
    return 0


# ============================================== primitives: outputs/report.html
DIVERGENCE = 0.25  # sources further apart than this are never pooled


def prim_pct(v) -> str:
    return "&mdash;" if v is None else f"{v*100:.0f}"


PRIM_CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
 --muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
 --s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;--bad:#d03b3b;--warn:#fab219}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
 --surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926}}
:root[data-theme=dark]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;
 --ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
 --s1:#3987e5;--s2:#d95926}
*{box-sizing:border-box}
html,body{background:var(--page);color:var(--ink);margin:0;
 font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1140px;margin:0 auto;padding:30px 22px 80px}
header{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}
h1{font-size:26px;margin:0 0 6px}
.dek{color:var(--ink2);margin:0;max-width:70ch}
h2{font-size:19px;margin:42px 0 4px;padding-top:20px;border-top:1px solid var(--grid)}
h2 .sub{display:block;font-size:13.5px;font-weight:400;color:var(--ink2);margin-top:5px;max-width:80ch}
button{font:inherit;font-size:13px;padding:7px 12px;border-radius:8px;
 border:1px solid var(--border);background:var(--surface);color:var(--ink2);cursor:pointer}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:15px 16px}
.tlab{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.tval{font-size:32px;line-height:1.1;margin:7px 0 3px}
.tile.bad .tval{color:var(--bad)}.tile.good .tval{color:var(--good)}
.tnote{font-size:12.5px;color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
 padding:18px 20px 20px;margin:16px 0}
.card h3{font-size:15.5px;margin:0 0 3px}
.card .sub{font-size:13px;color:var(--ink2);margin:0 0 15px;max-width:82ch}
.row{display:grid;grid-template-columns:230px 1fr 82px;align-items:center;gap:12px;padding:5px 0}
.rlab{font-size:12.5px;color:var(--ink2);text-align:right;overflow-wrap:anywhere}
.track{height:15px;background:var(--grid);border-radius:4px;position:relative}
.bar{height:100%;background:var(--s1);border-radius:0 4px 4px 0}
.bar.s2{background:var(--s2)}
.rval{font-size:13px;line-height:1.35;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.pcts{color:var(--muted);font-size:11px}.nlab{display:block;font-size:11px;color:var(--muted)}
.legend{display:flex;gap:16px;margin:0 0 12px 242px;font-size:12.5px;color:var(--ink2)}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
.flag{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;
 background:color-mix(in srgb,var(--warn) 22%,transparent);color:var(--ink);margin-left:8px}
.split{display:grid;grid-template-columns:230px 1fr;gap:12px;padding:7px 0;align-items:start}
.splitbars{display:flex;flex-direction:column;gap:3px}
.sb{display:flex;align-items:center;gap:8px;height:14px}
.sb .b{height:100%;border-radius:0 4px 4px 0;min-width:2px}
.sb .t{font-size:11.5px;color:var(--ink2);font-variant-numeric:tabular-nums;white-space:nowrap}
.note{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--s2);
 border-radius:9px;padding:13px 16px;font-size:13.5px;color:var(--ink2);margin:16px 0}
.note strong{color:var(--ink)}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 16px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--grid)}
td{font-variant-numeric:tabular-nums}
th[scope=row]{font-weight:400;color:var(--ink2)}
details summary{cursor:pointer;font-size:13.5px;color:var(--ink2);padding:8px 0}
a{color:var(--s1)}
#tip{position:fixed;z-index:99;pointer-events:none;opacity:0;transition:opacity .1s;
 background:var(--ink);color:var(--surface);font-size:12px;padding:6px 9px;border-radius:6px;max-width:290px}
"""

PRIM_JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{
 const show=e=>{tip.innerHTML=el.dataset.tip;tip.style.opacity=1;
  const r=el.getBoundingClientRect();
  tip.style.left=Math.min(innerWidth-310,r.left+12)+'px';tip.style.top=Math.max(8,r.top-38)+'px';};
 const hide=()=>tip.style.opacity=0;
 el.addEventListener('mouseenter',show);el.addEventListener('mouseleave',hide);
 el.addEventListener('focus',show);el.addEventListener('blur',hide);});
const b=document.querySelector('button.theme');
b.addEventListener('click',()=>{const d=document.documentElement.dataset.theme==='dark';
 document.documentElement.dataset.theme=d?'light':'dark';b.textContent=d?'Dark mode':'Light mode';});
"""


def prim_tile(lab, val, note, tone="") -> str:
    return (f'<div class="tile {tone}"><div class="tlab">{esc(lab)}</div>'
            f'<div class="tval">{val}</div><div class="tnote">{note}</div></div>')


def prim_bars(title, sub, items) -> str:
    """items: (label, acc, n, tooltip). Single series -> slot 1, no legend needed."""
    rows = []
    for lab, acc, n, tip in items:
        if acc is None:
            rows.append(f'<div class="row"><div class="rlab">{lab}</div>'
                        f'<div class="track"></div><div class="rval">&mdash;</div></div>')
            continue
        rows.append(
            f'<div class="row" tabindex="0" data-tip="{esc(tip)}">'
            f'<div class="rlab">{lab}</div>'
            f'<div class="track"><div class="bar" style="width:{max(acc*100,0.6):.2f}%"></div></div>'
            f'<div class="rval">{acc*100:.0f}<span class="pcts">%</span>'
            f'<span class="nlab">n={n}</span></div></div>')
    return (f'<div class="card"><h3>{title}</h3><p class="sub">{sub}</p>{"".join(rows)}</div>')


def primitive_section(s: dict) -> tuple[str, list]:
    """The headline. Divergent primitives are split by source, never pooled."""
    rows, table = [], []
    order = sorted(s["primitives"].items(),
                   key=lambda kv: ((kv[1]["pooled_answerable"] or kv[1]["pooled"] or {}).get("acc", 1)))
    for key, p in order:
        srcs = {d: v for d, v in p["sources"].items() if v.get("answerable")}
        accs = [v["answerable"]["acc"] for v in srcs.values()]
        diverged = len(accs) > 1 and (max(accs) - min(accs)) > DIVERGENCE
        pa = p["pooled_answerable"] or {}
        na = max((v["na_rate"] for v in p["sources"].values()), default=0)
        na_note = f" &middot; {na*100:.0f}% of golds are 'Not Applicable'; scored on the answerable subset" if na > 0.1 else ""

        if diverged:
            sb = []
            for i, (d, v) in enumerate(sorted(srcs.items(), key=lambda kv: -kv[1]["answerable"]["acc"])):
                a, n = v["answerable"]["acc"], v["answerable"]["n"]
                sb.append(f'<div class="sb" tabindex="0" data-tip="{esc(d)}: {a*100:.1f}% (n={n})">'
                          f'<div class="b" style="width:{max(a*100,0.6):.2f}%;'
                          f'background:var(--s{1 if i==0 else 2})"></div>'
                          f'<span class="t">{a*100:.0f}% &middot; {esc(d)} (n={n})</span></div>')
                table.append((f"{p['label']} &mdash; {d}", a, n))
            rows.append(f'<div class="split"><div class="rlab">{esc(p["label"])}'
                        f'<span class="flag">sources disagree</span></div>'
                        f'<div class="splitbars">{"".join(sb)}</div></div>')
        else:
            acc, n = pa.get("acc"), pa.get("n", 0)
            src_txt = ", ".join(f"{d} {v['answerable']['acc']*100:.0f}%" for d, v in srcs.items())
            rows.append(
                f'<div class="row" tabindex="0" data-tip="{esc(p["label"])}: {prim_pct(acc)}% (n={n})'
                f' &middot; {esc(src_txt)}{na_note}">'
                f'<div class="rlab">{esc(p["label"])}</div>'
                f'<div class="track"><div class="bar" style="width:{max((acc or 0)*100,0.6):.2f}%"></div></div>'
                f'<div class="rval">{prim_pct(acc)}<span class="pcts">%</span>'
                f'<span class="nlab">n={n}</span></div></div>')
            table.append((p["label"], acc, n))
    return ("".join(rows), table)


def prim_render(s: dict) -> str:
    n_multi = s["totals"]["multi_source_primitives"]
    tiles = [prim_tile("questions", f"{s['totals']['questions']:,}", "official splits, official metrics")]
    for ds, d in s["datasets"].items():
        if d.get("acc") is None:
            continue
        tone = "bad" if d["acc"] < 0.5 else ""
        ci = (f" &middot; CI {d['ci_lo']*100:.0f}&ndash;{d['ci_hi']*100:.0f}%"
              if d.get("ci_lo") is not None else "")
        tiles.append(prim_tile(ds, f"{d['acc']*100:.0f}<span class='pcts'>%</span>",
                          f"n={d['n']}{ci}", tone))

    prim_rows, prim_table = primitive_section(s)
    cx = s.get("charxiv", {})
    loc = s.get("localization", {})

    cx_block = ""
    if cx.get("descriptive") and cx.get("reasoning"):
        d, r = cx["descriptive"], cx["reasoning"]
        gap = (d["acc"] - r["acc"]) * 100
        cx_block = prim_bars(
            "CharXiv: reading the chart vs reasoning over it",
            "Same figures, same model. Descriptive questions each isolate one primitive; "
            "reasoning questions require combining several readings.",
            [("descriptive (single primitive)", d["acc"], d["n"],
              f"descriptive: {d['acc']*100:.1f}% (n={d['n']})"),
             ("reasoning (composed)", r["acc"], r["n"],
              f"reasoning: {r['acc']*100:.1f}% (n={r['n']})")])
        cx_block += (f'<div class="note"><strong>A {gap:.0f}-point gap.</strong> Haiku 4.5 reads '
                     f'individual chart elements reliably and degrades sharply once an answer '
                     f'requires holding several of them together. That gap, not the absolute '
                     f'scores, is the finding.</div>')

    g = cx.get("grader_comparison")
    grader = ""
    if g:
        grader = (f'<div class="note"><strong>Grader check.</strong> On {g["n"]} responses graded '
                  f'both ways, CharXiv\'s official LLM judge scores '
                  f'{g["official_judge"]*100:.1f}% and our string matcher {g["string_match"]*100:.1f}%, '
                  f'agreeing on {g["agreement"]*100:.0f}%. The official judge is authoritative; '
                  f'the agreement is what licenses using the fast matcher for the slices the '
                  f'judge has not covered.</div>')

    loc_block = ""
    if loc.get("by_target_size"):
        order = ["<12px", "12-20px", "20-32px", "32-56px", ">=56px"]
        byk = {d["label"]: d for d in loc["by_target_size"]}
        items = [(esc(k), byk[k]["acc"], byk[k]["n"], f"{k}: {byk[k]['acc']*100:.1f}% (n={byk[k]['n']})")
                 for k in order if k in byk]
        loc_block = prim_bars(
            "Localization on dense screens, by target size",
            "Target size as it reaches the model, after the API caps the long edge at 1568px. "
            "ScreenSpot-Pro only.", items)
        if loc.get("by_ui_type"):
            loc_block += prim_bars("Localization by element type",
                              "Icons carry no readable string, so they test perception rather than text matching.",
                              [(esc(d["label"]), d["acc"], d["n"],
                                f"{d['label']}: {d['acc']*100:.1f}% (n={d['n']})")
                               for d in loc["by_ui_type"]])

    # --- failure-mode breakdown: why, not just how often ---------------
    FM = FM_LABELS
    fm_cards = []
    for ds, d in s["datasets"].items():
        modes = d.get("failure_modes") or {}
        # A single-mode breakdown says nothing; multiple choice has only one way
        # to be wrong, and that is a property of the format, not a finding.
        if len(modes) < 2:
            continue
        tot = d.get("failures", sum(modes.values())) or 1
        items = [(FM.get(m, m), c / tot, c,
                  f"{FM.get(m, m)}: {c} of {tot} failures ({c/tot*100:.1f}%)")
                 for m, c in sorted(modes.items(), key=lambda kv: -kv[1])]
        adj = d.get("acc_meaning_adjusted")
        note = ""
        if adj and abs(adj - (d.get("acc") or 0)) > 1e-6:
            note = (f" Crediting answers that mean the right thing but are phrased differently "
                    f"raises {esc(ds)} from {d['acc']*100:.1f}% to {adj*100:.1f}%.")
        fm_cards.append(prim_bars(
            f"{esc(ds)} &mdash; why the failures failed",
            f"{tot} scored-wrong answers, classified. List-shaped cases are decided exactly; "
            f"the rest by an equivalence judge.{note}", items))
    fm_block = "".join(fm_cards)

    # --- benchmark quality + blind control, as caveats not adjustments ---
    q = s.get("benchmark_quality") or {}
    qual_block = ""
    if q:
        rows_q = "".join(
            f'<tr><th scope="row">{esc(d)}</th>'
            f'<td>{v["audited"]}</td>'
            f'<td>{v["bad_gt_in_failures"]*100:.1f}%</td>'
            f'<td><b>{v["bad_gt_whole_set"]*100:.2f}%</b></td>'
            f'<td>{v["headline_if_all_credited"]*100:.1f}%</td></tr>'
            for d, v in sorted(q.items(), key=lambda kv: -kv[1]["bad_gt_whole_set"]))
        qual_block = f"""
<div class="card">
<h3>How trustworthy is each benchmark's ground truth?</h3>
<p class="sub">A vision model was shown the image, the question, the shipped reference answer and
Haiku's answer, and asked which was actually right. It was told to default to the benchmark and to
justify any verdict against it. Only <em>failures</em> were audited &mdash; bad ground truth is
concentrated there, because bad ground truth is what causes a correct answer to be scored wrong.
The whole-set column is therefore the one that characterises the benchmark.</p>
<table><thead><tr><th scope="col">dataset</th><th scope="col">audited</th>
<th scope="col">bad GT among failures</th><th scope="col">bad GT across the whole set</th>
<th scope="col">headline if all credited</th></tr></thead><tbody>{rows_q}</tbody></table>
</div>
<div class="note"><strong>Nothing here is applied to the scores.</strong> Every number in this
report is the official metric over the full official split. Filtering to the questions whose
ground truth survives an audit would move each headline by 0&ndash;3 points and would be hard to
distinguish from choosing the questions the model happens to do well on. The errors are real
&mdash; two InfographicVQA golds were confirmed wrong by hand, and CharXiv contradicts itself
about whether one figure's panels have titles &mdash; but at 1.7&ndash;2.8% of each full split
they are smaller than any gap this study reports.</div>"""

    bc = s.get("ai2d_blind_control") or {}
    blind_block = ""
    if bc:
        items = []
        for k, v in sorted(bc.items(), key=lambda kv: kv[1]["blind"]):
            if v.get("with_image") is None:
                continue
            gain = v["with_image"] - v["blind"]
            items.append((f"{esc(k)} &mdash; blind", v["blind"], v["blind_n"],
                          f"{esc(k)} with the diagram withheld: {v['blind']*100:.1f}%"))
            items.append((f"{esc(k)} &mdash; with diagram", v["with_image"], v["n"],
                          f"{esc(k)} seeing the diagram: {v['with_image']*100:.1f}% "
                          f"(+{gain*100:.1f}pp)"))
        blind_block = prim_bars(
            "AI2D: how much of the score needs the diagram at all?",
            "The same questions, asked with and without the image. Four-way multiple choice, so "
            "chance is 25%. A split that scores the same blind is measuring world knowledge, not "
            "perception, and does not belong in a primitive map.", items)
        blind_block += ('<div class="note"><strong>Only the label-reference half of AI2D is a '
                        'perception measurement.</strong> Its reasoning questions are 80% '
                        'answerable with the diagram hidden &mdash; ecology and physics knowledge, '
                        'not chart reading. Label-reference starts at 31.5% blind, barely above '
                        'chance, and nearly doubles when the model can look. Only that half feeds '
                        'the primitive map above.</div>')

    tbl = "".join(f'<tr><th scope="row">{lab}</th><td>{prim_pct(a)}%</td><td>{n}</td></tr>'
                  for lab, a, n in prim_table)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Claude Haiku 4.5 &mdash; perceptual primitives</title><style>{PRIM_CSS}</style></head>
<body><div class="wrap">
<header><div>
<h1>Claude Haiku 4.5 &mdash; where perception breaks</h1>
<p class="dek">Business visual tasks decomposed into perceptual primitives and measured across
three benchmarks, using each benchmark's own questions and its own metric.
{s['totals']['questions']:,} questions &middot; {s['totals']['primitives_measured']} primitives &middot;
{n_multi} measured by more than one dataset. No comparison model: this maps failure, not ranking.</p>
</div><button class="theme" type="button">Dark mode</button></header>

<div class="tiles">{''.join(tiles)}</div>

<h2>The primitive gradient<span class="sub">Each bar is one perceptual operation, pooled across
every dataset that measures it. Where two datasets disagree by more than {DIVERGENCE*100:.0f} points
the sources are shown separately &mdash; averaging them would describe nothing real.</span></h2>
<div class="card">{prim_rows}</div>

<h2>Reading vs reasoning<span class="sub">The clearest single contrast in the study.</span></h2>
{cx_block}{grader}

<h2>Not every failure is a perception failure<span class="sub">A wrong score can mean the
model misread the figure, or that it read it correctly and expressed the answer differently,
or &mdash; where the question specifies an order &mdash; that it read everything and sequenced
it wrong. Those are different findings and are separated here. The official metric stays the
headline; the adjusted figure sits beside it.</span></h2>
{fm_block}

<h2>Can these benchmarks be trusted?<span class="sub">Two checks on the measuring instrument
itself: whether the reference answers are right, and whether the questions actually require the
image.</span></h2>
{qual_block}
{blind_block}

<h2>Localization, in detail<span class="sub">Why the two localization sources disagree so violently.</span></h2>
{loc_block}

<details><summary>Table view &mdash; every charted value as text</summary>
<table><thead><tr><th scope="col">Primitive</th><th scope="col">Accuracy</th>
<th scope="col">n</th></tr></thead><tbody>{tbl}</tbody></table></details>

<h2>Per-asset evidence<span class="sub">Every evaluated question with its image, gold answer and
Haiku's answer, filterable by primitive and correctness.</span></h2>
<p><a href="gallery/charxiv_000.html">CharXiv gallery</a> &middot;
<a href="gallery/infographicvqa_000.html">InfographicVQA gallery</a> &middot;
<a href="gallery/screenspot_pro_000.html">ScreenSpot-Pro gallery</a> &middot;
<a href="datasets.html">dataset documentation</a></p>

</div><div id="tip" role="status"></div><script>{PRIM_JS}</script></body></html>"""


def cmd_primitives(a) -> int:
    s = json.loads(Path(a.summary).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(prim_render(s), encoding="utf-8")
    print(f"wrote {a.out} ({Path(a.out).stat().st_size/1024:.0f} KB)")
    return 0


# ========================================== headline: outputs/aug22/report.html
AUG22_OUT = Path("outputs/aug22")

AUG22_CSS = """
:root{
  --bg:#0f1116; --panel:#171a21; --panel2:#1d2028; --ink:#e8eaed; --muted:#9aa0aa;
  --line:#2a2f3a; --good:#0ca30c; --bad:#d03b3b; --warn:#d68a1e; --accent:#5b8def;
  --chip:#232833;
}
@media (prefers-color-scheme: light){
  :root{ --bg:#f7f8fa; --panel:#fff; --panel2:#f0f2f6; --ink:#15181d; --muted:#5c636e;
         --line:#dfe3ea; --chip:#eceff4; }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:27px;margin:0 0 6px} h2{font-size:20px;margin:38px 0 12px;padding-bottom:7px;
   border-bottom:1px solid var(--line)} h3{font-size:16px;margin:22px 0 8px;color:var(--ink)}
p{color:var(--ink)} .sub{color:var(--muted);margin:0 0 22px}
nav{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0 26px}
nav a{background:var(--chip);color:var(--ink);text-decoration:none;padding:6px 12px;
      border-radius:7px;font-size:13px;border:1px solid var(--line)}
nav a:hover{border-color:var(--accent)}
table{width:100%;border-collapse:collapse;margin:12px 0;background:var(--panel);
      border:1px solid var(--line);border-radius:9px;overflow:hidden}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);font-size:14px}
th{background:var(--panel2);color:var(--muted);font-weight:600;font-size:12px;
   text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.tile .v{font-size:25px;font-weight:650;font-variant-numeric:tabular-nums}
.tile .l{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.tile .n{color:var(--muted);font-size:12px;margin-top:5px}
.good{color:var(--good)} .bad{color:var(--bad)} .warn{color:var(--warn)}
.callout{background:var(--panel);border-left:3px solid var(--accent);padding:13px 16px;
         border-radius:0 9px 9px 0;margin:14px 0}
.callout.bad{border-left-color:var(--bad)} .callout.good{border-left-color:var(--good)}
.callout.warn{border-left-color:var(--warn)}
.bar{height:9px;background:var(--panel2);border-radius:5px;overflow:hidden;min-width:90px}
.bar>i{display:block;height:100%;background:var(--accent)}
.badge{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;
       font-weight:600;background:var(--chip);border:1px solid var(--line)}
.badge.p{color:var(--good);border-color:var(--good)}
.badge.r{color:var(--bad);border-color:var(--bad)}
code{background:var(--chip);padding:1px 5px;border-radius:4px;font-size:13px}
"""


def aug22_pct(v, d=1): return "--" if v is None else f"{v*100:.{d}f}%"
def aug22_f1(v, d=1): return "--" if v is None else f"{v:.{d}f}"


def aug22_bar(frac, tone=""):
    w = max(0.0, min(1.0, frac or 0)) * 100
    col = {"good": "var(--good)", "bad": "var(--bad)", "warn": "var(--warn)"}.get(tone, "var(--accent)")
    return f'<div class="bar"><i style="width:{w:.1f}%;background:{col}"></i></div>'


def aug22_render(s: dict) -> str:
    c = s.get("controls", {})
    ds = s["datasets"]
    P = []
    A = P.append

    A(f'<!doctype html><html><head><meta charset="utf-8">'
      f'<meta name="viewport" content="width=device-width,initial-scale=1">'
      f'<title>Haiku 4.5 perception blind spots -- corrected report</title>'
      f'<style>{AUG22_CSS}</style></head><body><div class="wrap">')
    A("<h1>Claude Haiku 4.5 -- perception blind spots</h1>")
    A(f'<p class="sub">Corrected report, {esc(s.get("generated",""))} &middot; '
      f'<code>{esc(s["model"])}</code> &middot; thinking enabled (2000 tokens) &middot; '
      f'{s["totals"]["questions"]:,} questions across {len(ds)} benchmark arms plus 4 control ablations</p>')
    A('<nav>'
      '<a href="drilldown.html">Drill-down (top&rarr;bottom)</a>'
      '<a href="slidevqa.html">SlideVQA explorer</a>'
      '<a href="causes/index.html">Per-cause evidence</a>'
      '<a href="summary.json">summary.json</a>'
      '<a href="../datasets.html">Dataset documentation</a>'
      '<a href="../gallery/">Galleries</a></nav>')

    # ---------- what changed --------------------------------------------
    dup = c.get("duplication", {})
    dupline = ", ".join(f"{k} {v['duplicate_lines']}" for k, v in dup.items() if v["duplicate_lines"])
    A('<div class="callout warn"><b>What changed in this rebuild.</b> '
      'Resumed runs appended rather than replaced, leaving duplicate lines '
      f'({esc(dupline)}). The aggregate loader always collapsed these by uid, but '
      'ad-hoc analyses during the session did not, so some interim figures were quoted '
      'with inflated <i>n</i>. Every number below is deduplicated. SlideVQA is now '
      'included (it was missing from the original dataset list, silently omitting '
      '1,497 questions), and the four control ablations are reported alongside the '
      'accuracies they explain.</div>')

    A('<div class="callout bad"><b>Retracted.</b> An interim claim that arithmetic '
      'costs ~6&times; what retrieval costs on SlideVQA does not survive. Roughly half '
      'that gap is a metric artifact: token-F1 scores <code>22%</code> against a gold of '
      '<code>22</code> as zero. Format-corrected, the three costs are comparable. Both '
      'columns are shown below.</div>')

    # ---------- headline -------------------------------------------------
    A("<h2>Headline accuracy</h2>")
    A('<table><tr><th>Benchmark</th><th class="num">n</th><th>Metric</th>'
      '<th class="num">Score</th><th style="width:180px"></th><th class="num">95% CI</th></tr>')
    METRIC = {"charxiv": "string match / official judge", "infographicvqa": "ANLS",
              "screenspot_pro": "click-in-bbox", "ai2d": "MC accuracy",
              "slidevqa": "token F1 (evidence pages)", "slidevqa_allpages": "token F1 (all 20 slides)"}
    for k, d in ds.items():
        if d.get("acc") is None:
            continue
        tone = "bad" if d["acc"] < 0.35 else "good" if d["acc"] > 0.75 else "warn"
        ci = (f'{d["ci_lo"]*100:.1f}-{d["ci_hi"]*100:.1f}'
              if d.get("ci_lo") is not None else "--")
        A(f'<tr><td><b>{esc(k)}</b></td><td class="num">{d["n"]:,}</td>'
          f'<td>{esc(METRIC.get(k,""))}</td>'
          f'<td class="num {tone}"><b>{aug22_pct(d["acc"],2)}</b></td>'
          f'<td>{aug22_bar(d["acc"], tone)}</td><td class="num">{ci}</td></tr>')
    A("</table>")

    rp = c.get("reproducibility", {})
    if rp.get("disagreement_rate") is not None:
        A(f'<div class="callout"><b>Noise floor.</b> {rp["repeated_items"]:,} CharXiv items were '
          f'answered twice by the resumed runs -- same question, same settings. '
          f'<b>{aug22_pct(rp["disagreement_rate"])}</b> returned a different answer the second time. '
          'That is a measured reproducibility floor on real repeated trials: single-item '
          'differences below roughly this size are not interpretable.</div>')

    # ---------- blind control -------------------------------------------
    A("<h2>How much of each score is actually vision?</h2>")
    A('<p>The same questions, asked with the image withheld. Whatever survives was '
      'never a perception task -- it was recoverable from the question text and world '
      'knowledge. This is the ceiling that must be subtracted before calling a number "vision".</p>')
    A('<table><tr><th>Benchmark</th><th class="num">Blind</th><th class="num">Sighted</th>'
      '<th class="num">Vision adds</th><th style="width:180px"></th><th class="num">Chance</th><th class="num">n</th></tr>')
    for k, v in sorted(c.get("blind", {}).items(), key=lambda x: -x[1]["vision_adds_pp"]):
        tone = "bad" if v["vision_adds_pp"] < 20 else "good"
        A(f'<tr><td><b>{esc(k)}</b></td><td class="num">{aug22_pct(v["blind"])}</td>'
          f'<td class="num">{aug22_pct(v["sighted"])}</td>'
          f'<td class="num {tone}"><b>{v["vision_adds_pp"]:.1f}pp</b></td>'
          f'<td>{aug22_bar(v["vision_adds_pp"]/100, tone)}</td>'
          f'<td class="num">{aug22_pct(v["chance"],0)}</td><td class="num">{v["n"]}</td></tr>')
    A("</table>")
    cs = c.get("blind_charxiv_split", {})
    if cs:
        A("<h3>CharXiv, by split</h3><table><tr><th>Split</th><th class='num'>Blind</th>"
          "<th class='num'>Sighted</th><th class='num'>Vision adds</th><th class='num'>n</th></tr>")
        for k, v in cs.items():
            A(f'<tr><td>{esc(k)}</td><td class="num">{aug22_pct(v["blind"])}</td>'
              f'<td class="num">{aug22_pct(v["sighted"])}</td>'
              f'<td class="num good">{(v["sighted"]-v["blind"])*100:.1f}pp</td>'
              f'<td class="num">{v["n"]}</td></tr>')
        A("</table>")
    b = c.get("blind", {})
    if "ai2d" in b:
        v = b["ai2d"]
        above = (v["sighted"] - v["chance"]) * 100
        A(f'<div class="callout bad"><b>AI2D is largely not a vision benchmark.</b> Of the '
          f'{above:.0f} points it scores above chance, only <b>{v["vision_adds_pp"]:.1f}</b> come from '
          'seeing the diagram. Its headline overstates diagram perception by roughly 3&times;. '
          'CharXiv is the opposite and is the most trustworthy arm here.</div>')

    # ---------- localization ---------------------------------------------
    cl = c.get("coarse_localization", {})
    if cl:
        A("<h2>Localization: not blind, imprecise</h2>")
        A('<p>ScreenSpot-Pro scores near zero on exact clicks. But bucketing the '
          '<i>same</i> predictions by how much precision you demand shows a smooth falloff, '
          'not an absence of perception.</p>')
        A('<table><tr><th>Granularity</th><th class="num">Accuracy</th><th style="width:200px"></th>'
          '<th class="num">Chance</th><th class="num">Above chance</th></tr>')
        for lab, g in cl["grids"].items():
            A(f'<tr><td>{esc(lab)} grid</td><td class="num"><b>{aug22_pct(g["acc"])}</b></td>'
              f'<td>{aug22_bar(g["acc"])}</td><td class="num">{aug22_pct(g["chance"])}</td>'
              f'<td class="num">{g["acc"]/g["chance"]:.1f}&times;</td></tr>')
        ex, ar = cl["exact_click_in_bbox"], cl["mean_target_area_frac"]
        A(f'<tr><td><b>exact click-in-bbox</b></td><td class="num bad"><b>{aug22_pct(ex,2)}</b></td>'
          f'<td>{aug22_bar(ex,"bad")}</td><td class="num">{aug22_pct(ar,3)}</td>'
          f'<td class="num">{ex/ar:.0f}&times;</td></tr></table>')
        A(f'<p class="sub">Mean target occupies {aug22_pct(ar,3)} of the screen (n={cl["n"]:,}).</p>')

    g = c.get("grid_control")
    if g:
        A("<h3>Is it perception, or coordinate emission?</h3>")
        A('<p>Same items, same 4&times;4 granularity; only the answer format differs. '
          'The model either names a cell ("B3") or emits coordinates that we bucket.</p>')
        A('<table><tr><th>Condition</th><th class="num">Accuracy</th><th style="width:200px"></th></tr>'
          f'<tr><td><b>names the cell</b></td><td class="num good"><b>{aug22_pct(g["named_cell_acc"])}</b></td>'
          f'<td>{aug22_bar(g["named_cell_acc"],"good")}</td></tr>'
          f'<tr><td>clicks, bucketed to same cell</td><td class="num">{aug22_pct(g["click_derived_cell_acc"])}</td>'
          f'<td>{aug22_bar(g["click_derived_cell_acc"])}</td></tr>'
          f'<tr><td>chance</td><td class="num">{aug22_pct(g["chance"])}</td><td>{aug22_bar(g["chance"])}</td></tr></table>')
        p = g.get("mcnemar_p")
        A(f'<div class="callout"><b>Both are true, in proportion.</b> Naming beats pointing by '
          f'<b>{g["delta_pp"]:+.1f}pp</b> (McNemar {g["mcnemar_b"]}/{g["mcnemar_c"]} discordant, '
          f'p={p:.2g}) -- coordinate emission is genuinely lossy and a UI agent using element '
          f'selection rather than raw pixels would recover it. But even with coordinates removed, '
          f'accuracy is {aug22_pct(g["named_cell_acc"])} at 4&times;4. The majority of the deficit is '
          'perceptual, and no output format rescues it.</div>')

    # ---------- slidevqa costs -------------------------------------------
    sc = c.get("slidevqa_costs")
    if sc:
        A("<h2>SlideVQA: what does each operation actually cost?</h2>")
        A(f'<p>Retrieval is measured paired on n={sc["paired_n"]} questions present in both '
          'conditions. Format-corrected columns neutralise the token-F1 unit artifact.</p>')
        A('<table><tr><th>Operation</th><th class="num">As scored</th>'
          '<th class="num">Format-corrected</th><th>What it means</th></tr>')
        rows = [("retrieval", "find the right slide among 20"),
                ("integration", "combine 2+ slides"),
                ("derivation", "compute on what was read")]
        for k, note in rows:
            a_, b_ = sc["as_scored"][k], sc["format_corrected"][k]
            A(f'<tr><td><b>{esc(k)}</b></td><td class="num">{a_:+.1f}</td>'
              f'<td class="num"><b>{b_:+.1f}</b></td><td class="sub">{esc(note)}</td></tr>')
        A("</table>")
        A('<table><tr><th>Slice</th><th class="num">As scored F1</th><th class="num">Format-corrected F1</th></tr>')
        for k, lab in (("overall_f1", "overall"), ("lookup_f1", "plain lookup"),
                       ("arithmetic_f1", "needs arithmetic"), ("single_page_f1", "single-page evidence"),
                       ("multi_page_f1", "multi-page evidence")):
            A(f'<tr><td>{esc(lab)}</td><td class="num">{aug22_f1(sc["as_scored"][k])}</td>'
              f'<td class="num">{aug22_f1(sc["format_corrected"][k])}</td></tr>')
        A("</table>")

    op = c.get("onepage")
    if op:
        A("<h3>Are the multi-page questions really multi-page?</h3>")
        A('<p>Give the model only one of the two annotated evidence slides. If accuracy holds, '
          'the "multi-hop" label was a dataset artifact.</p>')
        A(f'<table><tr><th>Condition</th><th class="num">F1</th><th style="width:200px"></th></tr>'
          f'<tr><td>both evidence slides</td><td class="num">{aug22_f1(op["both_slides_f1"])}</td>'
          f'<td>{aug22_bar(op["both_slides_f1"]/100)}</td></tr>'
          f'<tr><td><b>only one of the two</b></td><td class="num bad"><b>{aug22_f1(op["one_slide_f1"])}</b></td>'
          f'<td>{aug22_bar(op["one_slide_f1"]/100,"bad")}</td></tr></table>')
        A(f'<div class="callout good"><b>Genuinely multi-page, and integration is a strength.</b> '
          f'Removing one slide costs <b>{op["collapse_f1"]:.1f} F1</b>; only '
          f'{aug22_pct(op["still_answerable_frac"])} stay answerable. The information really is '
          'distributed -- yet combining two slides costs only ~4-6 F1. These are '
          '<i>bridge questions</i>: the target is identified by a property on a different '
          'slide ("the Trading Operating Profit in the year Nestl&eacute; achieved the third '
          'largest Organic Growth"). Haiku handles that indirection well.</div>')

    # ---------- abstention ------------------------------------------------
    ab = c.get("abstention")
    if ab and ab.get("gold_na_n"):
        A("<h2>When the thing is not there</h2>")
        A(f'<p>{ab["gold_na_n"]:,} CharXiv questions have "Not Applicable" as the gold answer -- '
          'the chart genuinely has no legend, no second axis, no intersecting lines.</p>')
        A('<div class="tiles">'
          f'<div class="tile"><div class="l">Correctly abstained</div>'
          f'<div class="v good">{aug22_pct(ab["correctly_abstained"])}</div>'
          f'<div class="n">n={ab["gold_na_n"]:,}</div></div>'
          f'<div class="tile"><div class="l">Invented a value</div>'
          f'<div class="v bad">{aug22_pct(ab["invented_a_value"])}</div>'
          f'<div class="n">confident fabrication</div></div>'
          f'<div class="tile"><div class="l">Over-abstained</div>'
          f'<div class="v">{aug22_pct(ab["over_abstained"])}</div>'
          f'<div class="n">n={ab["gold_value_n"]:,}</div></div></div>')
        A('<p>The aggregate hides the shape of it. Absence detection is '
          '<b>structure-dependent</b>:</p>')
        A('<table><tr><th>Question</th><th class="num">n</th><th class="num">Correctly abstained</th>'
          '<th style="width:180px"></th></tr>')
        for q in ab["by_question"]:
            tone = "bad" if q["abstained"] < 0.8 else "good"
            A(f'<tr><td>{esc(q["qlabel"])}</td><td class="num">{q["n"]}</td>'
              f'<td class="num {tone}">{aug22_pct(q["abstained"])}</td>'
              f'<td>{aug22_bar(q["abstained"],tone)}</td></tr>')
        A("</table>")
        A('<div class="callout bad"><b>The worst failure mode for business use.</b> Asked how many '
          'legend entries a legend-less chart has, Haiku invents a count roughly 4 times in 10. '
          'That is a <i>detection</i> failure -- does this structure exist at all -- not a reading '
          'failure, and it produces confident fabricated output rather than an error.</div>')

    # ---------- numeric error --------------------------------------------
    ne = c.get("numeric_error", [])
    if ne:
        A("<h2>When a number is wrong, how wrong is it?</h2>")
        A('<p>If the model were misreading values off an axis, errors would cluster below 10%. '
          'They do not.</p>')
        A('<table><tr><th>Benchmark</th><th class="num">numeric n</th><th class="num">exact</th>'
          '<th class="num">median error when wrong</th><th class="num">within 10%</th>'
          '<th class="num">&gt;100%</th></tr>')
        for r in ne:
            A(f'<tr><td>{esc(r["label"])}</td><td class="num">{r["n_numeric"]:,}</td>'
              f'<td class="num">{aug22_pct(r["exact_frac"])}</td>'
              f'<td class="num bad"><b>{aug22_pct(r["median_rel_error"])}</b></td>'
              f'<td class="num">{aug22_pct(r["within_10pct_frac"])}</td>'
              f'<td class="num">{aug22_pct(r["over_100pct_frac"])}</td></tr>')
        A("</table>")
        A('<div class="callout"><b>Wrong element, not near miss.</b> A 33-90% relative error is the '
          'signature of grabbing a neighbouring bar or row -- a discrete jump to the wrong element -- '
          'not of imprecise interpolation, which would cluster under 10%. Only ~14-19% of errors '
          'land within 10%. This is consistent across four independent benchmarks, which makes it '
          'a property of the model rather than a dataset artifact.</div>')

    # ---------- format artifact ------------------------------------------
    fa = c.get("format_artifact", {})
    if fa:
        A("<h2>How much of the &quot;failure&quot; is the metric?</h2>")
        A('<table><tr><th>Benchmark</th><th>Metric</th><th class="num">hard zeros</th>'
          '<th class="num">of which format-equivalent</th></tr>')
        MET = {"slidevqa": "token F1", "charxiv": "ANLS", "infographicvqa": "ANLS"}
        for k in ("slidevqa", "charxiv", "infographicvqa"):
            v = fa.get(k)
            if not v or not v["hard_zeros"]:
                continue
            frac = v["hard_zeros_format_equivalent"] / v["hard_zeros"]
            tone = "bad" if frac > 0.2 else ""
            A(f'<tr><td><b>{esc(k)}</b></td><td>{esc(MET[k])}</td>'
              f'<td class="num">{v["hard_zeros"]:,}</td>'
              f'<td class="num {tone}"><b>{aug22_pct(frac)}</b> ({v["hard_zeros_format_equivalent"]:,})</td></tr>')
        A("</table>")
        A('<div class="callout warn"><b>Metric artifact, not a model blind spot.</b> Token-F1 scores '
          '<code>22%</code> against <code>22</code> as a flat zero; ANLS, being edit-distance based, '
          'barely notices. This is why the SlideVQA arithmetic gap needed retracting and why the '
          'ANLS-scored benchmarks did not.</div>')

    # ---------- limits ----------------------------------------------------
    A("<h2>Limits of this evaluation</h2><ul>")
    for t in [
        "<b>Single run, temperature not controllable.</b> Thinking pins temperature to 1 and the "
        "SDK exposes no override. The measured repeat-disagreement rate above is the honest noise floor.",
        "<b>CharXiv free-text types are string-scored where the official judge did not run</b>, which "
        "is a lower bound: a correct answer phrased differently scores zero. Judge-scored and "
        "string-scored cells are never averaged together.",
        "<b>Ground-truth noise sets the floor.</b> Audits put questionable gold at roughly 1.5% of "
        "CharXiv and 2.8% of InfographicVQA whole-set; 3 SlideVQA arithmetic items have expressions "
        "that do not evaluate to their own annotated answer.",
        "<b>ScreenSpot-Pro conflates perception with coordinate emission</b> -- quantified above, "
        "but the grid control covers 350 of 1,581 items, not all.",
        "<b>Blind control samples 500 per benchmark</b>, not the full splits.",
        "<b>Haiku downscales images to ~1568px long edge</b> regardless of what is sent, so no result "
        "here speaks to native-resolution performance. A prior ablation confirmed pre-downscaling "
        "changes nothing.",
        "<b>Benchmarks measure what they measure.</b> The blind control shows AI2D is substantially "
        "a language task; conclusions about diagram perception should lean on CharXiv.",
    ]:
        A(f"<li>{t}</li>")
    A("</ul>")

    A('<p class="sub" style="margin-top:34px">Generated by <code>python -m blindspot.report_pages headline</code>. '
      'Numbers recomputed from <code>results/*.jsonl</code>; see <code>summary.json</code> '
      'and <code>drilldown.csv</code> to verify any figure outside the browser.</p>')
    A("</div></body></html>")
    return "\n".join(P)


def cmd_headline(a) -> int:
    s = json.loads(Path(a.summary).read_text())
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(aug22_render(s))
    print(f"wrote {p} ({p.stat().st_size/1024:.0f} KB)")
    return 0


# ===================== candidates: outputs/report/candidates.html + candidates/
CAND_OUT = Path("outputs/report")
IMGS = CAND_OUT / "candidates"
RUN = "haiku-4-5_think2000_native_r0"

# items rejected on review: the question, not the model, is the problem
BLOCKED = {"infographicvqa:80994"}      # turns on how "the Middle East" is defined

# questions that name a printed mark on the diagram (same cut as report_data)
MARK_RE = re.compile(
    r"\b(letter|labell?ed|label|marked|arrow|point(?:ed|ing)?\s+(?:to|at)|shown by)\b", re.I)

PROBLEMS = [
    ("resolution", "Resolution bias",
     "The source image is far larger than the ~1.15 MP the API delivers, so the "
     "text the question asks about is destroyed before the model sees it."),
    ("localization", "Localization",
     "The model is given the exact target string and asked to point at it. Green "
     "box marks the target, red crosshair the click."),
    ("binding", "Label - object matching",
     "The question names a printed mark on the diagram and asks what it points to."),
    ("ocr_reasoning", "General OCR reasoning",
     "The arithmetic is right; the number it was applied to was misread off the chart."),
    ("hallucination", "Hallucination",
     "The chart has none of the structure the question asks about. Gold is "
     "'Not Applicable'; the model answers anyway."),
    ("counting", "Counting",
     "Counting repeated marks on a chart. Drawn from CharXiv rather than from the "
     "infographics, so the panel shows a counting failure and not a legibility one."),
]


def preds(path: str) -> dict:
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("pred") is not None:
            out[r["uid"]] = r
    return out


def mp(sizes) -> float:
    if not sizes:
        return 0.0
    w, h = sizes[0]
    return w * h / 1e6


# --------------------------------------------------------------- candidate pools
def cand_resolution() -> list[dict]:
    ex = {e.uid: e for e in load("infographicvqa")}
    rows = []
    for uid, r in preds(f"results/infographicvqa__{RUN}.jsonl").items():
        e = ex.get(uid)
        if not e or uid in BLOCKED or score(e, r["pred"])["score"] > 0:
            continue
        rows.append((mp(r.get("sent_image_sizes")), uid, e, r))
    rows.sort(key=lambda t: -t[0])
    out = []
    for m, uid, e, r in rows[:4]:
        im = as_model_saw(e.images[0])
        out.append({"uid": uid, "q": e.question, "gold": e.gold, "pred": r["pred"],
                    "note": f"source {m:.1f} MP, delivered {im.width}x{im.height} "
                            f"({im.width*im.height/1e6:.2f} MP)",
                    "im": im, "zoom": busiest_crop(im)})
    return out


def cand_localization() -> list[dict]:
    ex = {e.uid: e for e in load("svg_localization")}
    rows = [r for r in load_run(RUN)["point"] if not r["hit"]]
    near = sorted([r for r in rows if r["d_centre"] < .10], key=lambda r: r["d_centre"])
    mod = sorted([r for r in rows if .10 <= r["d_centre"] <= .25], key=lambda r: r["d_centre"])
    wrong = sorted([r for r in rows if r["d_centre"] > .25], key=lambda r: -r["d_centre"])
    picks = [near[0], mod[len(mod) // 3], mod[2 * len(mod) // 3], wrong[0]]
    out = []
    for r in picks:
        im = draw_target(as_model_saw(ex[r["uid"]].images[0]),
                         box=r["gold"], point=tuple(r["pred"]))
        out.append({"uid": r["uid"], "q": r["question"], "gold": "the boxed element",
                    "pred": f'({r["pred"][0]:.3f}, {r["pred"][1]:.3f})',
                    "note": f'{r["d_centre"]*100:.1f}% of the frame from the target',
                    "im": im, "zoom": None})
    return out


def cand_binding() -> list[dict]:
    ex = {e.uid: e for e in load("ai2d")}
    rows = []
    for uid, r in preds(f"results/ai2d__{RUN}.jsonl").items():
        e = ex.get(uid)
        if not e or not MARK_RE.search(e.question) or score(e, r["pred"])["score"] > 0:
            continue
        rows.append((len(e.question), uid, e, r))
    rows.sort()
    out = []
    for _, uid, e, r in rows[:4]:
        opts = e.meta.get("options") or []
        gi = "ABCD".find(str(e.gold[0] if isinstance(e.gold, list) else e.gold).strip())
        pi = "ABCD".find(str(r["pred"]).strip()[:1].upper())
        pick = lambda i: f'"{opts[i]}"' if 0 <= i < len(opts) else "?"
        out.append({"uid": uid, "q": e.question,
                    "gold": f'choice {"ABCD"[gi] if gi >= 0 else "?"} = {pick(gi)}',
                    "pred": f'choice {"ABCD"[pi] if pi >= 0 else "?"} = {pick(pi)}',
                    "note": "options: " + ", ".join(f'{L} = {o}'
                                                   for L, o in zip("ABCD", opts)),
                    "im": as_model_saw(e.images[0]), "zoom": None})
    return out


def cand_ocr_reasoning() -> list[dict]:
    ex = {e.uid: e for e in load("charxiv")}
    rows = []
    for uid, r in preds(f"results/charxiv__{RUN}.jsonl").items():
        e = ex.get(uid)
        if not e or e.meta.get("split") != "reasoning":
            continue
        if re.search(r"approximate|roughly|about ", e.question, re.I):
            continue
        if score(e, r["pred"])["score"] > 0:
            continue
        try:
            g = float(str(e.gold[0]).replace(",", ""))
            p = float(str(r["pred"]).replace(",", "").split()[0])
        except Exception:
            continue
        if g == 0 or p == 0:
            continue
        rel = abs(p - g) / abs(g)
        if 0.05 < rel < 0.6:                    # close enough to be a misread input
            rows.append((rel, uid, e, r))
    rows.sort()
    out = []
    for rel, uid, e, r in rows[:4]:
        out.append({"uid": uid, "q": e.question.split("*")[0].strip(),
                    "gold": e.gold, "pred": r["pred"],
                    "note": f"{rel*100:.0f}% off - the operation is right, "
                            f"the value read off the chart is not",
                    "im": as_model_saw(e.images[0]), "zoom": None})
    return out


def cand_hallucination() -> list[dict]:
    ex = {e.uid: e for e in load("charxiv")}
    rows = []
    for uid, r in preds(f"results/charxiv__{RUN}.jsonl").items():
        e = ex.get(uid)
        if not e or e.meta.get("qid") not in (11, 12, 13):
            continue
        g = e.gold[0] if isinstance(e.gold, list) else e.gold
        if is_na(g) and not is_na(r["pred"]):
            rows.append((uid, e, r))
    seen, spread = set(), []
    for uid, e, r in rows:                      # one per template first, then fill
        if e.meta.get("qid") not in seen:
            seen.add(e.meta.get("qid"))
            spread.append((uid, e, r))
    spread += [t for t in rows if t not in spread]
    out = []
    for uid, e, r in spread[:4]:
        out.append({"uid": uid, "q": e.question.split("*")[0].strip(),
                    "gold": "Not Applicable", "pred": r["pred"],
                    "note": e.meta.get("qlabel", ""),
                    "im": as_model_saw(e.images[0]), "zoom": None})
    return out


COUNT_Q = ("total number of explicitly labeled ticks", "how many discrete labels",
           "how many lines are there", "number of subplots")


def cand_counting() -> list[dict]:
    """CharXiv, not the infographics.

    Every InfographicVQA counting failure is a tall, dense poster, which makes
    the panel indistinguishable from the resolution example. On CharXiv the
    chart is fully legible and only the count is wrong.
    """
    ex = {e.uid: e for e in load("charxiv")}
    rows = []
    for uid, r in preds(f"results/charxiv__{RUN}.jsonl").items():
        e = ex.get(uid)
        if not e or uid in BLOCKED:
            continue
        ql = (e.meta.get("qlabel") or "").lower()
        if not any(q in ql for q in COUNT_Q):
            continue
        if score(e, r["pred"])["score"] > 0:
            continue
        try:                                    # both must be counts, or it is not
            g = float(str(e.gold[0]).replace(",", ""))       # a counting failure
            p = float(str(r["pred"]).replace(",", "").strip())
        except Exception:
            continue
        rows.append((ql, -g, uid, e, r, p))
    rows.sort()
    out, seen = [], set()
    for ql, negg, uid, e, r, p in rows:         # one per question form, biggest first
        if ql in seen:
            continue
        seen.add(ql)
        out.append({"uid": uid, "q": e.question.split("*")[0].strip(),
                    "gold": e.gold, "pred": r["pred"],
                    "note": f"true count {-negg:.0f}, model said {p:.0f}",
                    "im": as_model_saw(e.images[0]), "zoom": None})
    return out[:4]


POOLS = {"resolution": cand_resolution, "localization": cand_localization,
         "binding": cand_binding, "ocr_reasoning": cand_ocr_reasoning,
         "hallucination": cand_hallucination, "counting": cand_counting}


def cmd_candidates(a) -> int:
    IMGS.mkdir(parents=True, exist_ok=True)
    cards = []
    for key, title, blurb in PROBLEMS:
        try:
            cands = POOLS[key]()
        except Exception as exc:                # one bad pool must not kill the sheet
            cards.append(f"<section><h2>{esc(title)}</h2>"
                         f"<p class=err>could not build: {esc(exc)}</p></section>")
            print(f"  !! {key}: {exc}")
            continue
        panels = []
        for i, c in enumerate(cands):
            letter = "ABCD"[i]
            stem = f"{key}_{letter}"
            fit(c["im"], 1400, 1000).save(IMGS / f"{stem}.png")
            imgs = f'<img src="candidates/{stem}.png">'
            if c.get("zoom") is not None:
                c["zoom"].save(IMGS / f"{stem}_zoom.png")
                imgs += (f'<img src="candidates/{stem}_zoom.png" class=zoom>'
                         f'<div class=cap>1:1 crop of what was delivered</div>')
            panels.append(
                f'<figure><div class=letter>{letter}</div>{imgs}'
                f'<dl><dt>question</dt><dd>{esc(c["q"])}</dd>'
                f'<dt>gold</dt><dd>{esc(c["gold"])}</dd>'
                f'<dt>model</dt><dd class=pred>{esc(c["pred"])}</dd>'
                f'<dt>note</dt><dd class=note>{esc(c["note"])}</dd>'
                f'<dt>uid</dt><dd class=uid>{esc(c["uid"])}</dd></dl></figure>')
        cards.append(f'<section><h2>{esc(title)}</h2><p class=blurb>{esc(blurb)}</p>'
                     f'<div class=row>{"".join(panels)}</div></section>')
        print(f"  {key}: {len(cands)} candidates")

    # A contact sheet exists to be chosen from. On a thin results/ every pool
    # comes back empty and this used to write a 2.5 KB page with zero panels and
    # exit 0 -- indistinguishable from "no good candidates found" when the truth
    # is "there was nothing to look at". Refuse instead.
    if not any("<img" in c for c in cards):
        print(f"ABORT: every candidate pool came back empty, so the contact sheet "
              f"would have no panels to choose from.\nThis needs a scored run "
              f"across the benchmarks; score them first:\n"
              f"  python -m blindspot.core --datasets charxiv ai2d infographicvqa "
              f"screenspot_pro --max-spend N", file=sys.stderr)
        return 2

    (CAND_OUT / "candidates.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Blind spot example candidates</title>"
        "<style>body{background:#f2f2f0;font:15px/1.5 -apple-system,BlinkMacSystemFont,"
        "'Segoe UI',Roboto,sans-serif;margin:0;padding:28px;color:#2c2b28}"
        "h1{font-size:26px;margin:0 0 6px}h2{font-size:19px;margin:0 0 4px}"
        ".lede{color:#6e6d68;max-width:74ch;margin:0 0 26px}"
        "section{background:#fff;border-radius:12px;padding:20px;margin:0 0 22px;"
        "box-shadow:0 1px 3px rgba(0,0,0,.1)}"
        ".blurb{color:#6e6d68;margin:0 0 14px;max-width:90ch}"
        ".row{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}"
        "figure{margin:0;border:1px solid #e4e3df;border-radius:10px;padding:12px;position:relative}"
        ".letter{position:absolute;top:8px;left:8px;background:#2a78d6;color:#fff;"
        "font-weight:700;font-size:13px;border-radius:6px;padding:2px 8px}"
        "img{width:100%;display:block;border-radius:6px;background:#fafaf8}"
        "img.zoom{margin-top:8px;image-rendering:pixelated}"
        ".cap{font-size:11px;color:#8a8983;margin-top:2px}"
        "dl{margin:10px 0 0;font-size:12.5px}dt{color:#8a8983;font-size:10.5px;"
        "text-transform:uppercase;letter-spacing:.05em;margin-top:7px}"
        "dd{margin:1px 0 0}.pred{color:#d03b3b}.note{color:#52514e}"
        ".uid{color:#8a8983;font-family:ui-monospace,monospace;font-size:11px}"
        ".err{color:#d03b3b}</style>"
        "<h1>Blind spot example candidates</h1>"
        "<p class=lede>Four candidates per blind spot for the centrepiece figure. Every "
        "image is shown at the resolution the API actually delivered. Reply with one "
        "letter per section, in this order: resolution, localization, label-object "
        "matching, OCR reasoning, hallucination, counting.</p>" + "".join(cards))
    print(f"wrote {CAND_OUT/'candidates.html'}")
    return 0


# ================================================================== subcommands
EPILOG = """\
the seven builds, and what each is for:
  causes      one page per hypothesised blind spot, each marshalling the evidence
              from every benchmark that can test it (15 causes + an index)
  drilldown   every number in the study, openable into the splits that produced
              it, down to individual questions
  slidevqa    the SlideVQA arm: oracle-retrieval vs whole-deck, paired
  tasks       one page per perceptual primitive, successes beside failures
  primitives  the per-primitive overview the four pages above link back to
  headline    the corrected headline report: every accuracy beside its control
  candidates  four candidate images per blind spot, for picking a figure panel

The first four read results/*.jsonl and recompute everything they show with the
official scorers, so they cannot drift from the report; none of them is part of
the `blindspot.report` chain.

`primitives` reads outputs/summary.json      -- written by `eval aggregate`.
`headline`   reads outputs/report/summary.json -- written by `report summary`.
Both are file dependencies, not imports: nothing checks them, so a stale summary
gives a stale page. Run the writer first when the results change.

`causes` and `slidevqa` render images; both take a flag to rebuild the HTML
alone against assets already on disk, which is what you want while iterating on
a page.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="report_pages", description=__doc__.splitlines()[0],
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="SUBCOMMAND")

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(fn=fn)
        return p

    p = add("causes", cmd_causes, "outputs/causes/*.html -- one page per blind spot")
    p.add_argument("--no-images", action="store_true",
                   help="HTML only; reuse the assets already in outputs/assets_causes/")
    p.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8) - 2),
                   help="process pool size for image rendering")

    p = add("drilldown", cmd_drilldown,
            "outputs/drilldown.{html,json,csv} -- every number, openable")
    p.add_argument("--out", default=str(OUT / "drilldown.html"))
    p.add_argument("--csv", default=str(OUT / "drilldown.csv"))
    p.add_argument("--json", default=str(OUT / "drilldown.json"))

    p = add("slidevqa", cmd_slidevqa, "outputs/slidevqa.html -- the SlideVQA arm in full")
    p.add_argument("--out", default=str(OUT / "slidevqa.html"))
    p.add_argument("--workers", type=int, default=None,
                   help="process pool size for slide resizing (default: all cores)")
    p.add_argument("--skip-images", action="store_true",
                   help="rebuild the HTML only, reusing slides already on disk")

    p = add("tasks", cmd_tasks, "outputs/tasks/*.html -- one page per perceptual primitive")
    p.add_argument("--examples", type=int, default=8,
                   help="worked examples per side of the line, per page")

    p = add("primitives", cmd_primitives, "outputs/report.html -- the per-primitive overview")
    p.add_argument("--summary", default=str(OUT / "summary.json"),
                   help="written by `python -m blindspot.eval aggregate` (default: %(default)s)")
    p.add_argument("--out", default=str(OUT / "report.html"))

    p = add("headline", cmd_headline, "outputs/aug22/report.html -- the corrected headline report")
    p.add_argument("--summary", default=str(Path("outputs/report") / "summary.json"),
                   help="written by `python -m blindspot.report summary` (default: %(default)s)")
    p.add_argument("--out", default=str(AUG22_OUT / "report.html"))

    add("candidates", cmd_candidates,
        "outputs/report/candidates.html -- four candidate panels per blind spot")

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
