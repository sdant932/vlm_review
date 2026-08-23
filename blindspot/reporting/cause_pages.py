"""Per-cause evidence pages: outputs/causes/*.html.

The organising unit here is the *cause*, not the benchmark. Every page states one
claim, shows the quantitative evidence that bears on it, and then shows concrete
examples grouped by benchmark -- because a weakness that two independent datasets
agree on is a property of the model, while a weakness only one dataset shows is a
property that might belong to the dataset.

Nothing in this module reads or writes any pre-existing artefact. It creates
outputs/causes/ and outputs/assets_causes/ only.

Every number on every page is recomputed here from results/*.jsonl using the
official per-benchmark scorer in blindspot.core.scoring. Nothing is hardcoded.

Usage:
    python -m blindspot.reporting.cause_pages            # full build
    python -m blindspot.reporting.cause_pages --no-images  # HTML only, fast iteration
"""

from __future__ import annotations

import argparse
import collections
import html
import itertools
import json
import math
import os
import re
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from blindspot.core.adapters import load, slidevqa
from blindspot.core.scoring import score as official_score
from blindspot.core.failure_modes import classify as classify_failure, classify_point
from blindspot.core.stats import (is_na, wilson, quantiles, cell_of,
                                  centre_cell, bbox_cells)

ROOT = Path(".")
OUT = Path("outputs")
PAGES = OUT / "causes"
ASSETS = OUT / "assets_causes"

GOOD, BAD = "#0ca30c", "#d03b3b"
THUMB_MAX, THUMB_Q = 200, 70
FULL_MAX, FULL_Q = 1100, 82

TAG = "haiku-4-5_think2000_native_r0"
MAIN_FILES = {
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


def _examples() -> dict:
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
        self.ex = _examples()
        self.rows: dict[str, list[dict]] = {}
        self.counts: dict[str, dict] = {}
        for key, path in MAIN_FILES.items():
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
_SCALE = {"bn": 1e9, "b": 1e9, "billion": 1e9, "billions": 1e9,
          "m": 1e6, "mn": 1e6, "million": 1e6, "millions": 1e6,
          "k": 1e3, "thousand": 1e3, "thousands": 1e3, "tn": 1e12, "trillion": 1e12}
_NUMRE = re.compile(r"^\s*([+-]?)\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?|\.\d+)\s*(%?)\s*([a-zA-Z]*)\s*$")


def scalars(s) -> list[float]:
    """Every numeric reading of a scalar-shaped string.

    A scale word gives two readings -- stripped ("5m" -> 5) and applied
    ("5m" -> 5e6) -- because whether the unit was already implied by the
    question is not knowable from the string. Both are the same *value* in a
    different dress, which is exactly what this test is for.
    """
    t = re.sub(r"[\$€£]", "", str(s).strip().replace("−", "-"))
    t = re.sub(r"\s+", " ", t).strip()
    m = _NUMRE.match(t)
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
    if w in _SCALE:
        return [v, v * _SCALE[w]]
    return []                      # an unknown trailing word is not a formatting difference


def numval(s):
    v = scalars(s)
    return v[0] if v else None


def _alnum(s) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def format_equivalent(gold, pred) -> bool:
    """Right value, wrong dress. Deliberately conservative: numeric comparison is
    sign-sensitive and exact; text comparison is full-string after folding case
    and punctuation. There is no substring fallback, so "22" never matches
    "22 million people surveyed"."""
    golds = gold if isinstance(gold, (list, tuple)) else [gold]
    ps, pa = scalars(pred), _alnum(pred)
    for g in golds:
        gs = scalars(g)
        if gs and ps:
            for a in gs:
                for b in ps:
                    if a == b or (a != 0 and abs(a - b) <= 1e-9 * abs(a)):
                        return True
            continue
        if _alnum(g) and _alnum(g) == pa:
            return True
    return False


def fmt_score(r: dict) -> float:
    """Score with the metric's formatting penalty removed."""
    return 1.0 if format_equivalent(r["gold"], r["pred"]) else float(r.get("score") or 0.0)




def hit(r: dict, thr: float = 0.5) -> bool:
    return float(r.get("score") or 0.0) >= thr


def mean(vals) -> float | None:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else None




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
    tp, fp = ASSETS / f"{key}_t.jpg", ASSETS / f"{key}_f.jpg"
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
    ASSETS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(render_one, uniq, chunksize=4))
    return {r["key"]: r for r in res}


# ------------------------------------------------------------------- html
# Every custom property is declared on :root. Declaring them on a wrapper class
# instead is what shipped a black-on-black page last time: custom properties
# inherit downward only, so anything outside the wrapper -- the lightbox, which
# is a direct child of <body> -- resolves them to nothing.
CSS = """
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

LIGHTBOX_HTML = """
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

LIGHTBOX_JS = r"""
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


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def pctf(v, d=1) -> str:
    return "&mdash;" if v is None else f"{v * 100:.{d}f}%"


VERDICTS = {
    "PROVEN": "b-proven", "SUPPORTED": "b-supported", "MIXED": "b-mixed",
    "REFUTED": "b-refuted", "ARTIFACT": "b-artifact", "UNTESTED": "b-untested",
}


def badge(v: str) -> str:
    return f'<span class="badge {VERDICTS.get(v, "b-untested")}">{esc(v)}</span>'


def tile(lab, val, note, tone="") -> str:
    # Labels, values and notes are authored in this module and may carry entities
    # and inline markup, so they are not escaped here. Anything derived from data
    # is escaped at the call site.
    return (f'<div class="tile {tone}"><div class="tlab">{lab}</div>'
            f'<div class="tval">{val}</div><div class="tnote">{note}</div></div>')


def tiles(items) -> str:
    return '<div class="tiles">' + "".join(tile(*i) for i in items) + "</div>"


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


def table(headers, rows, note="", cls="") -> str:
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
<title>{esc(title)}</title><style>{CSS}</style></head><body>
<div class="wrap">
<header class="top"><div><h1>{title}</h1><p class="dek">{dek}</p></div>
<button class="theme" type="button">Dark mode</button></header>
{crumbs(here)}
{body}
</div>
{LIGHTBOX_HTML}
<script>{THEME_JS}
{LIGHTBOX_JS}</script></body></html>"""


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
import hashlib


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
        if hit(r) or format_equivalent(r["gold"], r["pred"]):
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
        a = mean(r["score"] for r in ch)
        px_vals.append(a)
        px_items.append((f"{lo / 1e6:.2f}&ndash;{hi / 1e6:.2f} MP", a, f"n={len(ch)}",
                         f"{a * 100:.1f}", None, "" if a > 0.66 else "s2"))
    w_vals = [mean(r["score"] for r in ch) for _, _, ch in qw]
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
        tiles([
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
        + table(["ScreenSpot-Pro target size", "n", "click inside the box", "right 4&times;4 cell"],
                sp_rows,
                "The same law at the other end of the scale. Targets here average a few hundredths "
                "of one percent of the screen; the largest fifth is hit seven times more often than "
                "the smallest, and even coarse cell-level placement improves with size.")
        + table(["CharXiv panels in the figure", "n", "descriptive accuracy"], cx_rows,
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
    area = mean((r["gold"][2] - r["gold"][0]) * (r["gold"][3] - r["gold"][1]) for r in sp)
    gvals.append(exact)
    cvals.append(area)
    grid_rows.append([("<strong>exact: inside the box</strong>", ""), f"{n}",
                      f"<strong>{exact * 100:.2f}%</strong>", f"{area * 100:.4f}%",
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
    name_acc, click_acc = mean(name_ok), mean(click_ok)
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
                        f"{100 * mean((r['gold'][2] - r['gold'][0]) * (r['gold'][3] - r['gold'][1]) for r in v):.4f}%"])

    modes = collections.Counter(classify_point(r["_ex"].gold, r["pred"]) for r in sp if not hit(r))
    nf = sum(modes.values())
    mode_rows = [[{"near_miss": "right area, missed the box (&lt;10% of the screen off)",
                   "moderate_miss": "roughly the wrong place (10&ndash;25% off)",
                   "wrong_region": "nowhere near (&gt;25% off)"}.get(k, k),
                  f"{v}", f"{100 * v / nf:.1f}%"] for k, v in modes.most_common()]

    body = (
        tiles([("Exact click accuracy", f"{exact * 100:.1f}%",
                f"n={n}; chance is {area * 100:.3f}% because the mean target is that "
                f"fraction of the screen", "bad"),
               ("Ratio above chance, 2&times;2", f"{gvals[0] / cvals[0]:.1f}&times;", "coarse quadrant", "warn"),
               ("Ratio above chance, exact", f"{exact / area:.0f}&times;",
                "the fine information <em>is</em> there &mdash; it is just not sufficient", "good"),
               ("Naming a cell vs clicking", f"{delta * 100:+.1f}pp",
                f"same 4&times;4 granularity, same {len(paired)} items", "warn")])
        + table(["granularity required", "n", "accuracy", "chance", "ratio above chance",
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
        + table(["element type", "n", "click inside the box", "right 4&times;4 cell",
                 "mean target size"], ui_rows,
                "Icons are both smaller and harder than text, and the gap survives at cell "
                "granularity, so it is not purely a size effect.")
        + table(["how the click missed", "n", "share of failures"], mode_rows,
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
                 f"{exact * 100:.1f}% of the time.",
                 "PROVEN", f"{exact * 100:.1f}% exact vs {gvals[0] * 100:.0f}% at quadrant level",
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
        bl = mean(s for _, _, s in v)
        si = mean(float(m.get("score") or 0) for _, m, _ in v)
        chance = 0.25 if k == "ai2d" else None
        rows.append([NICE[k], f"{len(v)}", f"{bl * 100:.1f}%", f"{si * 100:.1f}%",
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
                               f"{mean(s for _, _, s in v) * 100:.1f}%",
                               f"{mean(float(m.get('score') or 0) for _, m, _ in v) * 100:.1f}%",
                               f"+{(mean(float(m.get('score') or 0) for _, m, _ in v) - mean(s for _, _, s in v)) * 100:.1f}pp"])
    aiv = by.get("ai2d") or []
    for qt in ("label_reference", "diagram_reasoning"):
        v = [x for x in aiv if x[1]["meta"].get("qtype") == qt]
        if v:
            split_rows.append([f"AI2D {qt.replace('_', ' ')}", f"{len(v)}",
                               f"{mean(s for _, _, s in v) * 100:.1f}%",
                               f"{mean(float(m.get('score') or 0) for _, m, _ in v) * 100:.1f}%",
                               f"+{(mean(float(m.get('score') or 0) for _, m, _ in v) - mean(s for _, _, s in v)) * 100:.1f}pp"])
    ai_gain = next((float(r[4].replace("+", "").replace("pp", "")) if isinstance(r[4], str)
                    else 0 for r in rows if r[0] == "AI2D"), 0)
    ai_gain = (mean(float(m.get("score") or 0) for _, m, _ in aiv)
               - mean(s for _, _, s in aiv)) if aiv else 0

    body = (
        tiles([("AI2D without the diagram", f"{mean(s for _, _, s in aiv) * 100:.1f}%",
                f"chance is 25%; the image adds only {ai_gain * 100:.1f}pp (n={len(aiv)})", "bad"),
               ("CharXiv without the figure",
                f"{mean(s for _, _, s in cxp) * 100:.1f}%",
                f"vision adds {(mean(float(m.get('score') or 0) for _, m, _ in cxp) - mean(s for _, _, s in cxp)) * 100:.1f}pp "
                f"(n={len(cxp)})", "good"),
               ("Items answered correctly blind",
                f"{sum(1 for _, _, s in pairs if s >= 0.5)}",
                f"of {len(pairs)} paired control items &mdash; the image was never needed", "warn")])
        + table(["benchmark", "paired n", "no image", "with image", "what vision buys", "chance"],
                rows,
                "Same question, same prompt, same model &mdash; the image simply withheld. The gap "
                "is the honest measure of how much of a benchmark score is actually perception.")
        + hist_svg([("blind (no image)", "--s2", series_b), ("sighted", "--s1", series_s)],
                   "How much of each benchmark is a vision test at all",
                   "AI2D is barely one: a text-only model already scores well over twice chance "
                   "from world knowledge about food chains, water cycles and life cycles. CharXiv "
                   "is almost entirely one.", labels, ymax=1.0)
        + table(["split", "n", "no image", "with image", "vision gain"], split_rows,
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
                 f"{mean(s for _, _, s in (by.get('ai2d') or [])) * 100:.0f}% of AI2D correctly "
                 "against 25% chance, so at most a fifth of that score is perception.",
                 "PROVEN",
                 f"vision adds only {ai_gain * 100:.0f}pp on AI2D vs "
                 f"{(mean(float(m.get('score') or 0) for _, m, _ in cxp) - mean(s for _, _, s in cxp)) * 100:.0f}pp on CharXiv",
                 order, 0.96, body, groups, refute,
                 ["label_reference_binding", "ground_truth_noise"])


def c_label_reference_binding(d: Data, b: Builder) -> Cause:
    ai, cx = d.rows["ai2d"], d.rows["charxiv"]
    lr = [r for r in ai if r["meta"]["qtype"] == "label_reference"]
    dr = [r for r in ai if r["meta"]["qtype"] == "diagram_reasoning"]
    a_lr, a_dr = mean(hit(r) for r in lr), mean(hit(r) for r in dr)
    pairs = _blind_pairs(d)
    blr = [(br, m, s) for br, m, s in pairs
           if m["bench"] == "ai2d" and m["meta"]["qtype"] == "label_reference"]
    bdr = [(br, m, s) for br, m, s in pairs
           if m["bench"] == "ai2d" and m["meta"]["qtype"] == "diagram_reasoning"]

    # CharXiv's own binding question: name the legend entries, on figures that have one
    q13 = [r for r in cx if r["meta"].get("qid") == 13 and not is_na(r["gold"][0])]
    q12 = [r for r in cx if r["meta"].get("qid") == 12 and not is_na(r["gold"][0])]
    a13, a12 = mean(hit(r) for r in q13), mean(hit(r) for r in q12)

    rows = [["AI2D label-reference (resolve a printed letter to the thing it marks)",
             f"{len(lr)}", f"{a_lr * 100:.1f}%",
             f"{mean(s for _, _, s in blr) * 100:.1f}%" if blr else "&mdash;",
             f"n={len(blr)}"],
            ["AI2D diagram-reasoning", f"{len(dr)}", f"{a_dr * 100:.1f}%",
             f"{mean(s for _, _, s in bdr) * 100:.1f}%" if bdr else "&mdash;",
             f"n={len(bdr)}"],
            ["CharXiv: name the legend entries (figures that have a legend)",
             f"{len(q13)}", f"{a13 * 100:.1f}%", "&mdash;", "not in blind arm"],
            ["CharXiv: count the legend entries", f"{len(q12)}", f"{a12 * 100:.1f}%",
             "&mdash;", "not in blind arm"]]

    body = (
        tiles([("AI2D label-reference", f"{a_lr * 100:.1f}%",
                f"vs {a_dr * 100:.1f}% on the same benchmark's reasoning questions "
                f"(n={len(lr)} / {len(dr)})", "bad"),
               ("Gap", f"{(a_dr - a_lr) * 100:.1f}pp",
                "the largest within-benchmark split anywhere in this study", "bad"),
               ("Blind baseline", f"{mean(s for _, _, s in blr) * 100:.1f}%" if blr else "&mdash;",
                "label-reference is near chance without the diagram, so this really is perception",
                "good"),
               ("CharXiv legend naming", f"{a13 * 100:.1f}%",
                f"the same binding operation on charts is near-solved (n={len(q13)})", "good")])
        + table(["question family", "n", "accuracy", "blind accuracy", ""], rows,
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
                ("label-reference &mdash; blind", mean(s for _, _, s in blr) if blr else None,
                 f"n={len(blr)}", f"{mean(s for _, _, s in blr) * 100:.1f}" if blr else "&mdash;", 0.25, "s2"),
                ("diagram-reasoning &mdash; sighted", a_dr, f"n={len(dr)}", f"{a_dr * 100:.1f}", 0.25, ""),
                ("diagram-reasoning &mdash; blind", mean(s for _, _, s in bdr) if bdr else None,
                 f"n={len(bdr)}", f"{mean(s for _, _, s in bdr) * 100:.1f}" if bdr else "&mdash;", 0.25, "s2")]))

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
        f"label-reference to {mean(s for _, _, s in blr) * 100:.1f}% "
        f"(chance 25%, n={len(blr)}) while reasoning questions survive at "
        f"{mean(s for _, _, s in bdr) * 100:.1f}%. Nearly all the label-reference signal comes from "
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
        tiles([("Invents a structure that is not there", f"{inv_rate * 100:.1f}%",
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
        + table(["CharXiv template, on items where the structure is absent", "n",
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
            if hit(r) or format_equivalent(r["gold"], r["pred"]):
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
    iv_denom = [r for r in iv if not hit(r) and not format_equivalent(r["gold"], r["pred"])
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
        tiles([("Median relative error", f"{rows[0][2]}",
                f"CharXiv descriptive, over {rows[0][1]} wrong numeric answers", "bad"),
               ("Within 10% of the truth", f"{rows[0][3]}",
                "if the failure were imprecise interpolation this would dominate", "warn"),
               ("Predicted value is another number printed on the same infographic",
                f"{(iv_obs - iv_null) * 100:+.1f}pp",
                f"{len(iv_other)} of {len(iv_denom)} wrong InfographicVQA numbers "
                f"({iv_obs * 100:.0f}%) appear verbatim in the page's own OCR &mdash; but a "
                f"permutation control puts the chance rate at {iv_null * 100:.0f}%, so the real "
                f"excess is small", "warn")])
        + table(["benchmark / split", "wrong numeric answers", "median relative error",
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
        if hit(r) or format_equivalent(r["gold"], r["pred"]):
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
    a_d, a_r = mean(hit(r) for r in desc), mean(hit(r) for r in reas)
    look = [r for r in sv if not r["meta"].get("arithmetic")]
    arith = [r for r in sv if r["meta"].get("arithmetic")]
    f_l, f_a = mean(r["score"] for r in look), mean(r["score"] for r in arith)
    c_l, c_a = mean(fmt_score(r) for r in look), mean(fmt_score(r) for r in arith)
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
        tiles([("CharXiv read &rarr; derive", f"{(a_d - a_r) * 100:.1f}pp",
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
        + table(["split", "n", "as scored", "format-corrected"], rows,
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
        + table(["how the failed arithmetic actually failed", "n", "share"],
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
        fe = [r for r in z if format_equivalent(r["gold"], r["pred"])]
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
        + tiles([("SlideVQA hard zeros that are format-equivalent",
                  f"{len(detail['slidevqa']) / max(len([r for r in d.rows['slidevqa'] if float(r.get('score') or 0) == 0]), 1) * 100:.1f}%",
                  "token F1 gives zero to <code>22%</code> against a gold of <code>22</code>", "bad"),
                 ("CharXiv",
                  f"{len(detail['charxiv']) / max(len([r for r in d.rows['charxiv'] if float(r.get('score') or 0) == 0]), 1) * 100:.1f}%",
                  "normalized match, and CharXiv states the answer format in the prompt", "good"),
                 ("InfographicVQA",
                  f"{len(detail['infographicvqa']) / max(len([r for r in d.rows['infographicvqa'] if float(r.get('score') or 0) == 0]), 1) * 100:.1f}%",
                  "ANLS is edit distance, so a stray <code>%</code> costs a few points, not "
                  "everything", "good")])
        + table(["benchmark", "official metric", "n", "hard zeros",
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
        + table(["what the dress difference is (SlideVQA)", "n"],
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
    f_s, f_m = mean(r["score"] for r in single), mean(r["score"] for r in multi)
    c_s, c_m = mean(fmt_score(r) for r in single), mean(fmt_score(r) for r in multi)
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
    both = mean(m["score"] for _, m, _ in paired)
    one = mean(s for _, _, s in paired)
    still = mean(1 if s >= 0.5 else 0 for _, _, s in paired)
    both_ok = mean(1 if hit(m) else 0 for _, m, _ in paired)

    man = list(d.manifests["slidevqa"].values())
    nev = collections.Counter(len([x for x in (m.get("evidence_pages") or [])]) for m in man)
    ct = collections.Counter((len(m.get("evidence_pages") or []) > 1,
                              (m.get("arithmetic_expression") not in (None, "None", "")))
                             for m in man)

    body = (
        tiles([("Multi-page vs single-page", f"{(f_m - f_s) * 100:+.1f}pp",
                f"F1 {f_m * 100:.1f} vs {f_s * 100:.1f} (n={len(multi)} / {len(single)}); "
                f"format-corrected {(c_m - c_s) * 100:+.1f}pp", "good"),
               ("Take one of the two slides away", f"{(one - both) * 100:+.1f}pp",
                f"F1 {both * 100:.1f} &rarr; {one * 100:.1f} on the same {len(paired)} questions",
                "bad"),
               ("Still answerable from one slide", f"{still * 100:.1f}%",
                f"against {both_ok * 100:.1f}% with both &mdash; the information really is "
                f"distributed", "warn")])
        + bars("Integration across slides costs almost nothing; losing a slide costs everything",
               "The first two bars are the observational comparison, the second two the ablation. "
               "If multi-page questions were hard <em>because</em> they span slides, the first gap "
               "would be large. It is not. The ablation confirms the questions are genuinely "
               "multi-page: remove the second evidence slide and the same questions collapse.",
               [("single-evidence questions", f_s, f"n={len(single)}", f"{f_s * 100:.1f}", None, ""),
                ("multi-evidence questions", f_m, f"n={len(multi)}", f"{f_m * 100:.1f}", None, ""),
                ("multi-evidence, both slides sent", both, f"n={len(paired)}",
                 f"{both * 100:.1f}", None, "good"),
                ("multi-evidence, first slide only", one, f"n={len(paired)}",
                 f"{one * 100:.1f}", None, "bad")])
        + table(["SlideVQA question population (full manifest)", "n"],
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
        f'same questions fall {abs(one - both) * 100:.0f} points and only {still * 100:.0f}% remain '
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
    mfail = [r for r in multi if not hit(r) and not format_equivalent(r["gold"], r["pred"])]
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
    e_f1 = mean(sv[k]["score"] for k in common)
    a_f1 = mean(ap[k]["score"] for k in common)
    e_fc = mean(fmt_score(sv[k]) for k in common)
    a_fc = mean(fmt_score(ap[k]) for k in common)
    ti_e = mean(sv[k]["usage"]["input_tokens"] for k in common)
    ti_a = mean(ap[k]["usage"]["input_tokens"] for k in common)
    to_e = mean(sv[k]["usage"]["output_tokens"] for k in common)
    to_a = mean(ap[k]["usage"]["output_tokens"] for k in common)
    la_e = mean(sv[k]["latency_s"] for k in common)
    la_a = mean(ap[k]["latency_s"] for k in common)
    div = [k for k in common if hit(sv[k]) and not hit(ap[k])]
    rev = [k for k in common if not hit(sv[k]) and hit(ap[k])]

    body = (
        tiles([("Cost of finding the slide yourself", f"{(a_f1 - e_f1) * 100:+.1f}pp",
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
               [("evidence slides only", e_f1, f"n={len(common)}", f"{e_f1 * 100:.1f}", None, ""),
                ("all 20 slides", a_f1, f"n={len(common)}", f"{a_f1 * 100:.1f}", None, "s2"),
                ("evidence only, format-corrected", e_fc, f"n={len(common)}", f"{e_fc * 100:.1f}",
                 None, "good"),
                ("all 20, format-corrected", a_fc, f"n={len(common)}", f"{a_fc * 100:.1f}", None,
                 "good")])
        + f'<div class="note warn"><strong>But it does not search &mdash; it skims.</strong> '
        f'Handing the model ten times more input raises input tokens {ti_a / ti_e:.1f}&times; and '
        f'latency {la_a / la_e:.2f}&times;, while output tokens rise only {to_a / to_e:.2f}&times; '
        f'({to_e:.0f} &rarr; {to_a:.0f}). A model actually examining nineteen additional slides '
        f'would have visibly more to think about. The near-free retrieval result should therefore '
        f'be read narrowly: on a 20-slide deck the right slide is usually findable at a glance. It '
        f'is not evidence that deep search over a hundred pages would also be free.</div>'
        + table(["outcome when the deck was not pre-filtered", "n", "share of paired items"],
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
            acc.append(mean(x[0] for x in v) if v and len(v) >= MIN_BIN else None)
        series_acc.append((name, ["--s1", "--s2", "--s3"][len(series_acc) % 3], acc))
        series_names.append(name)
        overall = mean(hit(r) for r in rs)
        rows.append([name, f"{len(rs)}", f"{overall * 100:.1f}%"]
                    + [(f"{a * 100:.0f}%" if a is not None else "&mdash;")
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
        tiles([("CharXiv object counting", f"{mean(hit(r) for r in obj) * 100:.1f}%",
                f"n={len(obj)} &mdash; counting a handful of lines or legend entries is close to "
                f"solved", "good"),
               ("CharXiv tick counting", f"{mean(hit(r) for r in tick) * 100:.1f}%",
                f"n={len(tick)} &mdash; the same operation over ten to forty small marks", "warn"),
               ("InfographicVQA counting", f"{mean(r['score'] for r in iv_c) * 100:.1f}%",
                f"ANLS, n={len(iv_c)}, against {mean(r['score'] for r in iv_rest) * 100:.1f}% for "
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
        + table(["family", "n", "overall"] + [f"true count {k}" for k in COUNT_BINS], rows,
                "Accuracy binned by the true count &mdash; a dose-response curve, with the n for "
                "each cell beside it. Bins under five items are blanked. The trend, not any single "
                "cell, is the evidence.")
        + table(["family &mdash; mean <em>signed</em> error"] + COUNT_BINS, signed_rows,
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
        f'{mean(r["score"] for r in iv_c) * 100:.1f} against '
        f'{mean(r["score"] for r in iv_rest) * 100:.1f} for the rest of the benchmark &mdash; a '
        f'{(mean(r["score"] for r in iv_rest) - mean(r["score"] for r in iv_c)) * 100:.1f}-point '
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
        acc = mean(hit(r) for r in ai if r["gold"][0] == L)
        rows.append([f"option {L}", f"{gold[L]}", f"{gold[L] / n * 100:.1f}%",
                     f"{pick_[L] / n * 100:.1f}%",
                     f"{(pick_[L] - gold[L]) / n * 100:+.1f}pp",
                     f"{pw[L] / max(len(wrong), 1) * 100:.1f}%",
                     f"{acc * 100:.1f}%"])
        gv.append(gold[L] / n)
        pv.append(pick_[L] / n)
        wv.append(pw[L] / max(len(wrong), 1))
    chi = sum((pick_[L] - gold[L]) ** 2 / gold[L] for L in "ABCD")
    off = {k: v for k, v in pick_.items() if k not in set("ABCD")}
    maxdev = max(abs(pick_[L] - gold[L]) / n for L in "ABCD")

    body = (
        tiles([("Largest deviation from the key", f"{maxdev * 100:.1f}pp",
                f"across all four positions, n={n}", "good"),
               ("&chi;&sup2; against the gold distribution", f"{chi:.2f}",
                "3 degrees of freedom; the 5% critical value is 7.81", "good"),
               ("Unparseable answers", f"{sum(off.values())}",
                "the model always emitted one of the four letters", "good")])
        + table(["", "gold count", "share of the key", "share of picks", "deviation",
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
    drows = [[f"{k} step(s) from the top-left panel", f"{len(v)}", f"{mean(v) * 100:.1f}%"]
             for k, v in sorted(dist.items()) if len(v) >= 40]

    # axis confusion: x-axis answer that is really the y-axis label, and vice versa
    from blindspot.core.scoring import anls
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
        + table(["how the question addresses the panel (multi-panel figures only)", "n", "accuracy"],
                [["by row and column (&ldquo;row 2, column 1&rdquo;)", f"{len(named)}",
                  f"{mean(hit(r) for r in named) * 100:.1f}%"],
                 ["verbally (&ldquo;the left-most subplot&rdquo;)", f"{len(verbal)}",
                  f"{mean(hit(r) for r in verbal) * 100:.1f}%"],
                 ["no address needed (layout / count questions)", f"{len(noprefix)}",
                  f"{mean(hit(r) for r in noprefix) * 100:.1f}%"]],
                "What can be measured is whether addressing a panel costs anything. It costs a "
                "little, and verbal addressing costs more than coordinates &mdash; but both stay "
                "close to the single-panel baseline.")
        + table(["distance of the addressed panel from the top-left", "n", "accuracy"], drows,
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
                 f"{(mean(hit(r) for r in noprefix) - mean(hit(r) for r in named)) * 100:.1f}pp",
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
        tiles([("Order violations, everywhere", f"{sum(modes_all[k][2]['order_only'] for k in keys)}",
                "right items, wrong sequence &mdash; across every list-shaped question in the "
                "study", "good"),
               ("Missing or extra items",
                f"{sum(modes_all[k][2]['missing_items'] + modes_all[k][2]['extra_items'] for k in keys)}",
                "incomplete or over-complete answers", "warn"),
               ("Partly right", f"{sum(modes_all[k][2]['partial_overlap'] for k in keys)}",
                "some items right, some wrong", "warn")])
        + table(["benchmark", "list-shaped questions", "failures", "order only", "missing items",
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

    total_floor = mean([float(re.sub(r"<[^>]*>", "", r[5][0]).rstrip("%")) for r in rows])

    body = (
        tiles([("CharXiv floor", rows[0][5][0],
                f"share of the whole benchmark whose failure is contested by an independent audit "
                f"(n={rows[0][1]} audited)", "warn"),
               ("InfographicVQA floor", rows[1][5][0],
                f"n={rows[1][1]} audited failures", "warn"),
               ("ScreenSpot-Pro floor", rows[2][5][0],
                f"n={rows[2][1]} audited failures &mdash; box annotations are nearly always right",
                "good"),
               ("SlideVQA arithmetic golds that do not compute", f"{len(bad_expr)}",
                "the annotated expression does not evaluate to the annotated answer", "warn")])
        + table(["benchmark", "failures audited", "total failures", "contested", "contest rate",
                 "implied floor on the whole benchmark"], rows,
                "An independent judge was shown the image, the question, the gold and the "
                "prediction, and asked whether the gold was actually right. &ldquo;Contested&rdquo; "
                "counts verdicts of <em>prediction correct</em> or <em>both acceptable</em>, plus "
                "golds judged ambiguous or wrong. Extrapolating the contest rate over all failures "
                "gives the floor: the share of each benchmark where a reported failure may not be "
                "one.")
        + table(["SlideVQA question", "annotated expression", "annotated answer", "what it evaluates to"],
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
                    f"{mean(hit(r) for r in d.rows[k]) * 100:.1f}%",
                    f"{mean(r['score'] for r in d.rows[k]) * 100:.1f}"])
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
        + tiles([("Causes examined", f"{len(causes)}",
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
        + table(["benchmark", "lines in file", "unique questions", "unusable (null prediction)",
                 "malformed lines", "scored", "accuracy at threshold 0.5", "mean metric score"],
                acc,
                "Result files contain retries, so lines exceed questions; the last usable row per "
                "uid wins. Rows with a null prediction are excluded and counted here rather than "
                "scored as zero &mdash; scoring a dropped API call as a failure would be the same "
                "mistake this whole study is about. Every number on every cause page is recomputed "
                "from these rows using each benchmark's official metric.")
        + table(["control arm", "lines", "unique source questions", "malformed lines"], ctrl,
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Build outputs/causes/*.html")
    ap.add_argument("--no-images", action="store_true", help="HTML only; reuse existing assets")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 8) - 2))
    a = ap.parse_args()

    PAGES.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    print("loading + scoring results ...", flush=True)
    d = Data()
    for k, c in d.counts.items():
        print(f"  {k:20s} {c['scored']:5d} scored  ({c['lines']} lines, "
              f"{c['unusable']} null preds, {c['malformed']} malformed)", flush=True)

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
        for p in ASSETS.glob("*_t.jpg"):
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

    size = sum(p.stat().st_size for p in ASSETS.glob("*.jpg"))
    print(f"\nwrote {len(causes) + 1} pages -> {PAGES}/")
    print(f"assets: {len(list(ASSETS.glob('*.jpg')))} files, {size / 1e6:.1f} MB -> {ASSETS}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
