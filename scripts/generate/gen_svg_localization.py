"""Generate a synthetic text-localization benchmark from procedural charts and diagrams.

Why this exists
---------------
ScreenSpot-Pro measures localization but confounds three things:

  1. how large the *target* is,
  2. how large the *image* is (and so how much survives the API downscale to
     <=1568px per edge and ~1.15MP),
  3. whether the model can *see* the target versus *emit a coordinate* for it.

Here the layout is generated, so each of those is a knob rather than a property
of a scraped screenshot, and the gold box is measured off the actual raster --
there is no ground-truth noise floor to subtract.

How correctness is kept
-----------------------
Every builder emits one list of drawing primitives. That single list feeds both
outputs: PIL rasterizes it, and the SVG writer serializes it. Nothing is drawn
by one path and not the other, so the vector source and the PNG cannot drift
apart per chart type.

Text placement is pinned in both directions:

  * vertical -- an explicit alphabetic baseline is computed from PIL's ink box,
    rather than relying on SVG `dominant-baseline`, which centres on font
    metrics and lands 2-3px away for labels with descenders;
  * horizontal -- `textLength` + `lengthAdjust="spacingAndGlyphs"` force any
    renderer onto the advance width PIL measured.

Two validity rules are enforced before a text may become a question target:

  * its ink box must not overlap any other text's ink box, and
  * its string must be unique in the scene,

otherwise "where is X" would have more than one defensible answer. Rejected
targets are counted and reported rather than silently dropped.

Question types deliberately span the perception/expression axis:

  point / bbox  emit coordinates      (the ScreenSpot-Pro condition)
  grid          name a 4x4 cell       (the naming condition)
  relation      name a neighbour      (no coordinates at all)
  reverse       name text at a point  (inverse localization)

Usage
-----
    python scripts/generate/gen_svg_localization.py --count 10
    python scripts/generate/gen_svg_localization.py --count 500 --seed 7
    python scripts/generate/gen_svg_localization.py --list-types
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------- constants

API_EDGE, API_PIXELS = 1568, 1_150_000      # matches blindspot.reporting.cause_pages
BASE_W, BASE_H = 1500, 950
MARGIN = 58
TITLE_BAND = 76
BASE_FONT = 22
# Nothing may render below this at the smallest scale. 7px was legible only
# in the sense that a crop could be squinted at; the ask is that a reader can
# actually read every label on the `small` variant.
MIN_LEGIBLE_PX = 10

DEFAULT_SCALES = {"small": 0.6, "medium": 1.0, "large": 2.0}

FONT_DIRS = ["/System/Library/Fonts/Supplemental", "/System/Library/Fonts",
             "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
             "C:/Windows/Fonts"]

# (file, index, svg font-family). Mixed serif / sans / mono / condensed so the
# set does not train a model on one typeface.
FONT_CHOICES = [
    ("Arial.ttf", 0, "Arial, Helvetica, sans-serif"),
    ("Helvetica.ttc", 0, "Helvetica, Arial, sans-serif"),
    ("Georgia.ttf", 0, "Georgia, 'Times New Roman', serif"),
    ("Verdana.ttf", 0, "Verdana, Geneva, sans-serif"),
    ("Trebuchet MS.ttf", 0, "'Trebuchet MS', sans-serif"),
    ("Tahoma.ttf", 0, "Tahoma, Geneva, sans-serif"),
    ("Courier New.ttf", 0, "'Courier New', Courier, monospace"),
    ("Menlo.ttc", 0, "Menlo, Monaco, monospace"),
    ("Palatino.ttc", 0, "Palatino, 'Palatino Linotype', serif"),
    ("Futura.ttc", 0, "Futura, 'Century Gothic', sans-serif"),
    ("DejaVuSans.ttf", 0, "'DejaVu Sans', sans-serif"),
    ("LiberationSans-Regular.ttf", 0, "'Liberation Sans', Arial, sans-serif"),
    ("arial.ttf", 0, "Arial, Helvetica, sans-serif"),
]


@dataclass
class Theme:
    name: str
    bg: str
    ink: str            # primary text
    muted: str          # secondary text / axes
    stroke: str         # shape outlines
    grid: str | None    # background rule colour, None for no grid
    fills: list[str]    # series / node fills
    radius: int         # corner radius
    sw: float           # stroke width in base units
    dark: bool = False
    pattern: str = "none"   # none | grid | dots


THEMES = [
    Theme("paper", "#ffffff", "#12161c", "#5c636e", "#39414e", "#eceff4",
          ["#dfe9f7", "#e2f0e6", "#faeedd", "#ece5f6", "#dff0f3", "#fae4e4"], 4, 1.6),
    Theme("slate-dark", "#151a21", "#f2f5f8", "#9aa6b4", "#5b6675", "#222a34",
          ["#25405e", "#204a3c", "#5a4425", "#3e2f5a", "#1e4650", "#5c2b30"], 4, 1.6,
          dark=True),
    Theme("blueprint", "#0d2137", "#e8f2ff", "#8fb4d9", "#4a7aa8", "#173a56",
          ["#1b3d5c", "#1d4a5e", "#26445e", "#1a3550", "#22506b", "#154059"], 2, 1.5,
          dark=True, pattern="grid"),
    Theme("cream", "#fbf7ef", "#2b2418", "#7a6c56", "#5f5340", "#efe7d6",
          ["#f3e6cd", "#e6ecdb", "#f7e2d5", "#e8e3f0", "#dceaea", "#f5dede"], 6, 1.7),
    Theme("mono-print", "#ffffff", "#000000", "#555555", "#000000", "#dddddd",
          ["#ffffff", "#eeeeee", "#dddddd", "#f6f6f6", "#e6e6e6", "#cccccc"], 0, 1.9),
    Theme("mint", "#f2fbf7", "#0f2b22", "#4a6b60", "#2f5b4c", "#dcefe6",
          ["#cfe9dc", "#d9ecc9", "#c9e6ea", "#eae2d0", "#e3dcef", "#f2dcdc"], 8, 1.5),
    Theme("carbon", "#1b1b1f", "#f5f5f7", "#9a9aa4", "#63636e", "#2a2a31",
          ["#33333d", "#2c3f38", "#40372a", "#3a2f40", "#263a44", "#452c30"], 3, 1.6,
          dark=True, pattern="dots"),
    Theme("sun", "#fffdf5", "#241f10", "#77704f", "#5a5330", "#f0ead2",
          ["#fdf0c4", "#e9f2cf", "#fde3c8", "#e8e6f5", "#d6eef0", "#fadcd6"], 5, 1.6),
    Theme("ice", "#f5f9ff", "#0c1b2e", "#54677f", "#33506e", "#e2ebf6",
          ["#d8e6f7", "#d5ecef", "#e4e2f7", "#dff0e4", "#f7e8d8", "#f7dde2"], 4, 1.5),
    Theme("high-contrast", "#ffffff", "#000000", "#333333", "#000000", None,
          ["#ffe680", "#b3e6b3", "#ffb3b3", "#b3d9ff", "#e6ccff", "#ffd9b3"], 0, 2.4),
]

# ----------------------------------------------------------------- vocabulary

DOMAINS = [
    ("Payment Authorization", ["Card", "Payer", "Ledger", "Fraud", "Refund", "Invoice",
                               "Chargeback", "Settlement", "Merchant", "Wallet"]),
    ("Build Pipeline", ["Artifact", "Commit", "Runner", "Cache", "Bundle", "Manifest",
                        "Release", "Coverage", "Linter", "Container"]),
    ("Triage Protocol", ["Intake", "Vitals", "Symptom", "Consult", "Referral", "Chart",
                         "Dosage", "Discharge", "Allergy", "Specimen"]),
    ("Telemetry Ingest", ["Stream", "Envelope", "Partition", "Offset", "Schema", "Sink",
                          "Replica", "Window", "Checkpoint", "Backlog"]),
    ("Returns Workflow", ["Parcel", "Label", "Inspection", "Restock", "Credit", "Courier",
                          "Warehouse", "Dispute", "Pallet", "Receipt"]),
    ("Access Review", ["Principal", "Grant", "Policy", "Session", "Token", "Directory",
                       "Approval", "Revocation", "Scope", "Audit"]),
    ("Claims Handling", ["Adjuster", "Estimate", "Coverage", "Deductible", "Payout",
                         "Appraisal", "Reserve", "Endorsement", "Salvage", "Liability"]),
    ("Content Moderation", ["Report", "Queue", "Reviewer", "Appeal", "Strike", "Signal",
                            "Escalation", "Takedown", "Reinstate", "Classifier"]),
    ("Fleet Logistics", ["Depot", "Route", "Driver", "Manifest", "Fuel", "Dispatch",
                         "Trailer", "Dock", "Waybill", "Yard"]),
    ("Grid Operations", ["Feeder", "Substation", "Breaker", "Load", "Outage", "Meter",
                         "Relay", "Transformer", "Reserve", "Curtailment"]),
]

VERBS = ["Validate", "Enrich", "Normalize", "Dispatch", "Reconcile", "Archive", "Score",
         "Quarantine", "Aggregate", "Publish", "Rehydrate", "Throttle", "Annotate",
         "Partition", "Verify", "Escalate", "Compact", "Resolve", "Sample", "Merge",
         "Flush", "Rebalance", "Snapshot", "Deduplicate", "Index", "Backfill", "Route",
         "Cluster", "Rotate", "Purge"]

TERMINALS = ["Accept", "Reject", "Hold", "Retry", "Expire", "Close", "Defer", "Approve"]
EDGE_LABELS = ["ok", "fail", "retry", "stale", "match", "miss", "over cap", "signed",
               "expired", "dup", "clean", "flagged", "partial", "queued"]
REGIONS = ["North", "South", "East", "West", "Central", "Nordics", "Iberia", "Andes",
           "Pacific", "Baltic", "Rhine", "Sahel", "Cascadia", "Levant"]
QUARTERS = ["Q1 FY22", "Q2 FY22", "Q3 FY22", "Q4 FY22", "Q1 FY23", "Q2 FY23",
            "Q3 FY23", "Q4 FY23", "Q1 FY24", "Q2 FY24"]
METRICS = ["Throughput", "Latency", "Backlog", "Yield", "Uptime", "Churn", "Margin",
           "Coverage", "Defects", "Retention"]
ROLES = ["Director", "Lead", "Analyst", "Engineer", "Architect", "Manager", "Auditor",
         "Planner", "Specialist", "Coordinator"]
ACTORS = ["Client", "Gateway", "Broker", "Worker", "Registry", "Store", "Scheduler",
          "Notifier", "Cache", "Auditor"]


# ----------------------------------------------------------------- primitives
# One list of dicts; PIL draws it and the SVG writer serializes it.

def rect(x, y, w, h, fill=None, stroke=None, sw=1.5, r=0):
    return {"k": "rect", "x": x, "y": y, "w": w, "h": h, "fill": fill,
            "stroke": stroke, "sw": sw, "r": r}


def line(x0, y0, x1, y1, stroke, sw=1.5, arrow=False, dash=False):
    return {"k": "line", "x0": x0, "y0": y0, "x1": x1, "y1": y1, "stroke": stroke,
            "sw": sw, "arrow": arrow, "dash": dash}


def circle(cx, cy, r, fill=None, stroke=None, sw=1.5):
    return {"k": "circle", "cx": cx, "cy": cy, "r": r, "fill": fill,
            "stroke": stroke, "sw": sw}


def poly(pts, fill=None, stroke=None, sw=1.5):
    return {"k": "poly", "pts": pts, "fill": fill, "stroke": stroke, "sw": sw}


def wedge(cx, cy, r, a0, a1, fill=None, stroke=None, sw=1.5):
    return {"k": "wedge", "cx": cx, "cy": cy, "r": r, "a0": a0, "a1": a1,
            "fill": fill, "stroke": stroke, "sw": sw}


def text(x, y, s, size, fill, anchor="mm", role="label", target=True):
    return {"k": "text", "x": x, "y": y, "s": s, "size": size, "fill": fill,
            "anchor": anchor, "role": role, "target": target}


@dataclass
class Scene:
    gid: int
    title: str
    ctype: str
    theme: Theme
    font_file: str
    font_index: int
    font_family: str
    prims: list = field(default_factory=list)
    w: int = BASE_W
    h: int = BASE_H
    margin: float = MARGIN
    title_band: float = TITLE_BAND
    complexity: int = 1

    facts: dict = field(default_factory=dict)

    def add(self, *p):
        self.prims.extend(p)

    def tidx(self) -> int:
        """Index the next text prim will get, so facts can point at gold boxes."""
        return sum(1 for p in self.prims if p["k"] == "text")

    def fact(self, key, value):
        """Record scene semantics.

        The SVG carries geometry and strings but not relationships -- which value
        belongs to which category, which node an edge leaves. Future question
        types (read-off, aggregation, topology) need that, and it cannot be
        recovered from the rendered output, so it is captured at build time.
        """
        self.facts[key] = value

    @property
    def texts(self):
        return [p for p in self.prims if p["k"] == "text"]


# ----------------------------------------------------------------- fonts

_FONT_CACHE: dict = {}


def find_font(name: str) -> str | None:
    for d in FONT_DIRS:
        p = Path(d) / name
        if p.exists():
            return str(p)
    return None


def available_fonts(verbose: bool = False) -> list[tuple[str, int, str]]:
    """Fonts present *and* able to rasterize at the smallest size we will use.

    Some macOS faces (Helvetica.ttc among them) raise a FreeType division-by-zero
    below ~8px. A font that cannot draw the `small` variant would fail halfway
    through a long run, so each candidate is probed here instead.
    """
    probe = ImageDraw.Draw(Image.new("RGB", (32, 32)))
    sample = "Wg| 0123 source: internal"
    out, rejected = [], []
    for name, idx, fam in FONT_CHOICES:
        p = find_font(name)
        if not p:
            continue
        try:
            for px in (5, 7, 9):
                f = ImageFont.truetype(p, px, index=idx)
                probe.text((4, 4), sample, font=f, anchor="lt")
                probe.textbbox((4, 4), sample, font=f, anchor="lt")
        except Exception as e:
            rejected.append((name, str(e)))
            continue
        out.append((p, idx, fam))
    if verbose and rejected:
        for n, why in rejected:
            print(f"  - font {n} unusable at small sizes ({why}); skipped")
    if not out:
        raise SystemExit("No usable fonts found. Pass --font with a TTF path.")
    return out


def load(path: str, idx: int, size: int) -> ImageFont.FreeTypeFont:
    key = (path, idx, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, max(5, size), index=idx)
    return _FONT_CACHE[key]


def effective_size(w: int, h: int) -> tuple[int, int]:
    s = min(1.0, API_EDGE / max(w, h), math.sqrt(API_PIXELS / max(w * h, 1)))
    return max(1, round(w * s)), max(1, round(h * s))


# ----------------------------------------------------------------- readability
# WCAG 2.1 relative luminance and contrast ratio. A label is only usable as a
# question target if a reader could actually read it, so contrast is measured
# rather than assumed -- see `render`, which samples the true background from a
# text-free pass instead of trusting the theme's nominal background colour. A
# label sitting on a bar, a wedge or a table stripe has a background the theme
# never names.

WCAG_NORMAL, WCAG_LARGE = 4.5, 3.0     # AA thresholds
LARGE_PX = 24                          # >=24px counts as "large text" under AA


def _rgb(c) -> tuple[float, float, float]:
    if isinstance(c, (tuple, list)):
        return tuple(v / 255 for v in c[:3])
    c = str(c).lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))


def luminance(c) -> float:
    def lin(u):
        return u / 12.92 if u <= 0.03928 else ((u + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(v) for v in _rgb(c))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a, b) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def required_contrast(font_px: int) -> float:
    return WCAG_LARGE if font_px >= LARGE_PX else WCAG_NORMAL


def _median_rgb(im: Image.Image, box) -> tuple[int, int, int]:
    """Median colour of a region -- robust to a stray gridline crossing it."""
    import numpy as np
    x0, y0, x1, y1 = (max(0, int(box[0])), max(0, int(box[1])),
                      min(im.width, int(math.ceil(box[2]))),
                      min(im.height, int(math.ceil(box[3]))))
    if x1 <= x0 or y1 <= y0:
        return (255, 255, 255)
    a = np.asarray(im.crop((x0, y0, x1, y1)).convert("RGB")).reshape(-1, 3)
    if not a.size:
        return (255, 255, 255)
    return tuple(int(v) for v in np.median(a, axis=0))


def readable_on(col: str, bg: str, need: float = WCAG_NORMAL) -> str:
    """Nudge `col` away from `bg` until it clears `need`.

    Series and point colours are derived from theme fills, and a pale fill that
    reads well as a *shape* can fail badly as *text*. Rather than hand-tuning
    every palette, the colour is stepped until it measures.
    """
    if contrast_ratio(col, bg) >= need:
        return col
    darker = luminance(bg) > 0.5
    cur = col
    for _ in range(30):
        cur = _shade(cur, -0.035 if darker else 0.035)
        if contrast_ratio(cur, bg) >= need:
            return cur
    return "#000000" if darker else "#ffffff"


def _shade(hex_c: str, dl: float) -> str:
    hex_c = hex_c.lstrip("#")
    r, g, b = (int(hex_c[i:i + 2], 16) / 255 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(h, min(1, max(0, l + dl)), s)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


# ----------------------------------------------------------------- helpers

def measure(draw, font, s, anchor):
    """Ink box of `s` drawn at the origin with `anchor`, plus its baseline offset."""
    bb = draw.textbbox((0, 0), s, font=font, anchor=anchor)
    ls = draw.textbbox((0, 0), s, font=font, anchor="ls")
    return bb, ls


def text_w(draw, font, s) -> float:
    return font.getlength(s)


def uniq_labels(rng, nouns, n, maxlen=18):
    out, seen, guard = [], set(), 0
    while len(out) < n and guard < n * 300:
        guard += 1
        lab = (rng.choice(TERMINALS) if rng.random() < 0.18
               else f"{rng.choice(VERBS)} {rng.choice(nouns)}")
        if len(lab) > maxlen or lab in seen:
            continue
        seen.add(lab)
        out.append(lab)
    return out


def sample_uniq(rng, pool, n):
    return rng.sample(pool, min(n, len(pool)))


# ----------------------------------------------------------------- builders
# Each returns prims on a BASE_W x BASE_H canvas. `sc.theme` supplies colour.

def _title(sc, draw, base_font):
    t = sc.theme
    sc.add(text(sc.margin, sc.margin * 0.62, sc.title, round(BASE_FONT * 1.36), t.ink,
                anchor="lt", role="title", target=False))


def b_flowchart(sc, rng, draw, F):
    t, fs = sc.theme, BASE_FONT
    dom, nouns = sc.domain
    vertical = rng.random() < 0.6
    c = sc.complexity
    nl = rng.randint(4, min(8, 4 + c))
    hi = min(6, 3 + c)
    widths = [rng.randint(1, 2) if i in (0, nl - 1) else rng.randint(2, hi)
              for i in range(nl)]
    labs = uniq_labels(rng, nouns, sum(widths), 17)
    if len(labs) < sum(widths):
        return None
    nodes, i = [], 0
    layers = []
    # Cap each layer to what the canvas can hold. Without this, a wide layer at
    # high complexity runs off the right edge and its labels are clipped -- they
    # get rejected as out-of-bounds targets, so the gold stays valid, but the
    # scene looks broken and the targets are wasted.
    max_w = max((text_w(draw, F, l) + 30 for l in labs), default=120)
    gap = 46 if vertical else 54
    fit = max(1, int((sc.w - 2 * sc.margin + gap) // (max_w + gap)))
    widths = [min(w, fit) for w in widths]
    for li, w in enumerate(widths):
        row = []
        for _ in range(w):
            if i >= len(labs):
                break
            lab = labs[i]; i += 1
            tw = text_w(draw, F, lab)
            row.append({"lab": lab, "w": tw + 30, "h": fs + 22})
        if not row:
            continue
        rng.shuffle(row)
        layers.append(row)
        nodes += row
    nl = len(layers)
    if nl < 2:
        return None
    if vertical:
        top, bot = sc.margin + sc.title_band, sc.h - sc.margin - 20
        for li, row in enumerate(layers):
            cy = top + (bot - top) * li / max(nl - 1, 1)
            span = sum(n["w"] for n in row) + 46 * (len(row) - 1)
            x = (sc.w - span) / 2
            for n in row:
                n["cx"], n["cy"] = x + n["w"] / 2, cy
                x += n["w"] + 46
    else:
        widest = max(n["w"] for n in nodes)
        for li, row in enumerate(layers):
            cx = sc.margin + widest / 2 + (sc.w - 2 * sc.margin - widest) * li / max(nl - 1, 1)
            span = sum(n["h"] for n in row) + 54 * (len(row) - 1)
            y = sc.margin + sc.title_band + (sc.h - sc.margin - sc.title_band - sc.margin - span) / 2
            for n in row:
                n["cx"], n["cy"] = cx, y + n["h"] / 2
                y += n["h"] + 54
    edges = []
    for li in range(1, nl):
        for dst in layers[li]:
            src = rng.choice(layers[li - 1])
            lab = rng.choice(EDGE_LABELS) if rng.random() < .4 else ""
            _edge(sc, src, dst, lab, F, draw)
            edges.append({"from": src["lab"], "to": dst["lab"], "label": lab})
    for _ in range((sc.complexity - 1) * 2):        # skip and back edges
        a, b = rng.randrange(nl), rng.randrange(nl)
        if a == b:
            continue
        src, dst = rng.choice(layers[a]), rng.choice(layers[b])
        lab = rng.choice(EDGE_LABELS) if rng.random() < .35 else ""
        _edge(sc, src, dst, lab, F, draw)
        edges.append({"from": src["lab"], "to": dst["lab"], "label": lab})
    sc.fact("nodes", [n["lab"] for n in nodes])
    sc.fact("layers", [[n["lab"] for n in row] for row in layers])
    sc.fact("edges", edges)
    sc.fact("flow", "vertical" if vertical else "horizontal")
    for n in nodes:
        sc.add(rect(n["cx"] - n["w"] / 2, n["cy"] - n["h"] / 2, n["w"], n["h"],
                    rng.choice(t.fills), t.stroke, t.sw, t.radius),
               text(n["cx"], n["cy"], n["lab"], fs, t.ink))
    return True


def _edge(sc, a, b, lab, F, draw):
    t = sc.theme
    dx, dy = b["cx"] - a["cx"], b["cy"] - a["cy"]
    n = math.hypot(dx, dy) or 1
    ux, uy = dx / n, dy / n
    def clip(nd, sx, sy):
        tx = (nd["w"] / 2) / abs(sx) if sx else 1e9
        ty = (nd["h"] / 2) / abs(sy) if sy else 1e9
        k = min(tx, ty)
        return nd["cx"] + sx * k, nd["cy"] + sy * k
    x0, y0 = clip(a, ux, uy)
    x1, y1 = clip(b, -ux, -uy)
    sc.add(line(x0, y0, x1, y1, t.muted, t.sw * 0.9, arrow=True))
    if lab:
        sc.add(text((x0 + x1) / 2, (y0 + y1) / 2 - 11, lab, round(BASE_FONT * .68),
                    readable_on(t.muted, t.bg), role="edge"))


def b_bar(sc, rng, draw, F):
    t, fs = sc.theme, BASE_FONT
    cats = sample_uniq(rng, REGIONS, min(14, rng.randint(5, 8) + sc.complexity))
    vals = [rng.randint(12, 98) for _ in cats]
    sc.fact("series_name", None)
    sc.fact("bars", [{"category": c, "value": v} for c, v in zip(cats, vals)])
    x0, y1 = sc.margin + 110, sc.h - sc.margin - 62
    y0 = sc.margin + sc.title_band + 16
    x1 = sc.w - sc.margin - 30
    horiz = rng.random() < 0.35
    sc.add(line(x0, y0, x0, y1, t.muted, t.sw), line(x0, y1, x1, y1, t.muted, t.sw))
    mx = max(vals)
    for gi in range(5):
        gv = round(mx * gi / 4)
        gy = y1 - (y1 - y0) * gi / 4
        if t.grid and gi:
            sc.add(line(x0, gy, x1, gy, t.grid, t.sw * .7))
        sc.add(text(x0 - 12, gy, str(gv), round(fs * .8), readable_on(t.muted, t.bg),
                    anchor="rm", role="tick"))
    n = len(cats)
    slot = (x1 - x0) / n
    bw = slot * rng.uniform(0.52, 0.72)
    for i, (c, v) in enumerate(zip(cats, vals)):
        cx = x0 + slot * (i + 0.5)
        bh = (y1 - y0) * v / mx
        sc.add(rect(cx - bw / 2, y1 - bh, bw, bh, t.fills[i % len(t.fills)], t.stroke,
                    t.sw, min(t.radius, 4)))
        cfs = round(fs * .86)
        while cfs > 9 and text_w(draw, load(sc.font_file, sc.font_index, cfs), c) > slot * .94:
            cfs -= 1                      # keep neighbouring categories apart
        sc.add(text(cx, y1 + 20, c, cfs, t.ink, role="category"))
        sc.add(text(cx, y1 - bh - 15, f"{v}%", round(fs * .82), t.ink, role="value"))
    axis = rng.choice(METRICS)
    sc.fact("series_name", axis)
    sc.add(text(sc.margin + 22, (y0 + y1) / 2, axis, round(fs * .9),
                t.muted, anchor="mm", role="axis"))
    return True


def b_line(sc, rng, draw, F):
    t, fs = sc.theme, BASE_FONT
    xs = sample_uniq(rng, QUARTERS, min(10, rng.randint(5, 7) + sc.complexity))
    xs.sort(key=QUARTERS.index)
    series = sample_uniq(rng, METRICS, min(5, rng.randint(2, 3) + sc.complexity // 2))
    x0, y1 = sc.margin + 96, sc.h - sc.margin - 58
    y0, x1 = sc.margin + sc.title_band + 20, sc.w - sc.margin - 150
    sc.add(line(x0, y0, x0, y1, t.muted, t.sw), line(x0, y1, x1, y1, t.muted, t.sw))
    for gi in range(5):
        gy = y1 - (y1 - y0) * gi / 4
        if t.grid and gi:
            sc.add(line(x0, gy, x1, gy, t.grid, t.sw * .7))
        sc.add(text(x0 - 12, gy, str(gi * 25), round(fs * .78), t.muted, anchor="rm",
                    role="tick"))
    for i, xv in enumerate(xs):
        cx = x0 + (x1 - x0) * i / max(len(xs) - 1, 1)
        sc.add(text(cx, y1 + 20, xv, round(fs * .78), t.ink, role="category"))
    series_pts = {}
    for si, s in enumerate(series):
        col = readable_on(_shade(t.fills[si % len(t.fills)],
                                 -0.34 if not t.dark else 0.30), t.bg)
        pts, v, vseq = [], rng.randint(25, 70), []
        for i in range(len(xs)):
            v = max(6, min(98, v + rng.randint(-16, 20)))
            vseq.append(v)
            pts.append((x0 + (x1 - x0) * i / max(len(xs) - 1, 1), y1 - (y1 - y0) * v / 100))
        series_pts[s] = dict(zip(xs, vseq))
        for a, b in zip(pts, pts[1:]):
            sc.add(line(a[0], a[1], b[0], b[1], col, t.sw * 1.7))
        for p in pts:
            sc.add(circle(p[0], p[1], t.sw * 2.6, col, t.bg, t.sw * .8))
        sc.add(text(x1 + 14, pts[-1][1], s, round(fs * .84), col, anchor="lm",
                    role="series"))
    sc.fact("x_labels", xs)
    sc.fact("series", series_pts)
    return True


def b_scatter(sc, rng, draw, F):
    t, fs = sc.theme, BASE_FONT
    x0, y1 = sc.margin + 96, sc.h - sc.margin - 56
    y0, x1 = sc.margin + sc.title_band + 18, sc.w - sc.margin - 40
    sc.add(line(x0, y0, x0, y1, t.muted, t.sw), line(x0, y1, x1, y1, t.muted, t.sw))
    pts = sample_uniq(rng, REGIONS, min(14, rng.randint(6, 9) + sc.complexity))
    placed = []
    sc.fact("points", list(pts))
    for i, lab in enumerate(pts):
        for _ in range(60):
            px = rng.uniform(x0 + 60, x1 - 90)
            py = rng.uniform(y0 + 30, y1 - 40)
            if all(math.hypot(px - a, py - b) > 130 for a, b in placed):
                placed.append((px, py)); break
        else:
            placed.append((px, py))
        col = readable_on(_shade(t.fills[i % len(t.fills)],
                                 -0.32 if not t.dark else 0.28), t.bg)
        sc.add(circle(px, py, rng.uniform(7, 15), col, t.stroke, t.sw * .8))
        sc.add(text(px, py - 24, lab, round(fs * .82), t.ink, role="point"))
    ax = rng.choice(METRICS)
    sc.fact("x_axis", ax)
    sc.add(text((x0 + x1) / 2, y1 + 30, ax, round(fs * .88), t.muted, role="axis"))
    return True


def b_pie(sc, rng, draw, F):
    t, fs = sc.theme, BASE_FONT
    labs = sample_uniq(rng, REGIONS, rng.randint(4, 6))
    vals = [rng.randint(8, 40) for _ in labs]
    tot = sum(vals)
    sc.fact("wedges", [{"label": l, "value": v, "pct": v * 100 // tot}
                       for l, v in zip(labs, vals)])
    cx, cy = sc.w * 0.40, (sc.margin + sc.title_band + sc.h - sc.margin) / 2
    R = min(sc.h - sc.margin - sc.title_band, 520) * 0.42
    donut = rng.random() < 0.45
    a = -90.0
    for i, (l, v) in enumerate(zip(labs, vals)):
        sweep = 360 * v / tot
        col = t.fills[i % len(t.fills)]
        sc.add(wedge(cx, cy, R, a, a + sweep, col, t.stroke, t.sw))
        # Percentages sit inside the wedge. Outside with a leader line, they
        # collided with the legend once the canvas got narrow (dashboard panels),
        # and every collision costs two targets to the overlap rule.
        mid = math.radians(a + sweep / 2)
        rr = R * (0.62 if not donut else 0.78)
        lx, ly = cx + math.cos(mid) * rr, cy + math.sin(mid) * rr
        sc.add(text(lx, ly, f"{v * 100 // tot}%", round(fs * .8),
                    readable_on(t.ink, col), role="value"))
        a += sweep
    if donut:
        sc.add(circle(cx, cy, R * 0.46, t.bg, None, 0))
    lx = sc.w - sc.margin - 300
    ly = sc.margin + sc.title_band + 40
    for i, l in enumerate(labs):
        sc.add(rect(lx, ly + i * 44 - 10, 22, 22, t.fills[i % len(t.fills)], t.stroke, t.sw, 3))
        sc.add(text(lx + 36, ly + i * 44, l, round(fs * .88), t.ink, anchor="lm",
                    role="legend"))
    return True


def b_table(sc, rng, draw, F):
    t, fs = sc.theme, round(BASE_FONT * .84)
    cols = ["Region"] + sample_uniq(rng, METRICS, min(6, rng.randint(3, 4) + sc.complexity // 2))
    rows = sample_uniq(rng, REGIONS, min(14, rng.randint(5, 7) + sc.complexity * 2))
    nC, nR = len(cols), len(rows)
    x0, y0 = sc.margin + 30, sc.margin + sc.title_band + 10
    tw = sc.w - 2 * sc.margin - 60
    ch = min(58, (sc.h - sc.margin - y0) / (nR + 1))
    cw = tw / nC
    hdr = _shade(t.fills[0], -0.08 if not t.dark else 0.06)
    sc.add(rect(x0, y0, tw, ch, hdr, t.stroke, t.sw, t.radius))
    for j, c in enumerate(cols):
        sc.add(text(x0 + cw * (j + .5), y0 + ch / 2, c, fs, t.ink, role="header"))
    cells = {}
    for i, r in enumerate(rows):
        ry = y0 + ch * (i + 1)
        if i % 2 and t.grid:
            sc.add(rect(x0, ry, tw, ch, t.grid, None, 0, 0))
        sc.add(text(x0 + cw * .5, ry + ch / 2, r, fs, t.ink, role="cell"))
        for j in range(1, nC):
            v = f"{rng.randint(10, 99)}.{rng.randint(0, 9)}"
            cells[f"{r}|{cols[j]}"] = v
            sc.add(text(x0 + cw * (j + .5), ry + ch / 2, v, fs, t.ink, role="cell"))
    sc.fact("columns", cols)
    sc.fact("row_headers", rows)
    sc.fact("cells", cells)
    for j in range(nC + 1):
        sc.add(line(x0 + cw * j, y0, x0 + cw * j, y0 + ch * (nR + 1), t.stroke, t.sw * .7))
    for i in range(nR + 2):
        sc.add(line(x0, y0 + ch * i, x0 + tw, y0 + ch * i, t.stroke, t.sw * .7))
    return True


def b_org(sc, rng, draw, F):
    t, fs = sc.theme, round(BASE_FONT * .92)
    _, nouns = sc.domain
    names = [f"{r} {n}" for r, n in zip(sample_uniq(rng, ROLES, 9),
                                        sample_uniq(rng, nouns, 9))]
    levels = [[names[0]], names[1:1 + rng.randint(2, 3)]]
    rest = names[1 + len(levels[1]):]
    levels.append(rest[:rng.randint(3, 5)])
    top, bot = sc.margin + sc.title_band + 20, sc.h - sc.margin - 40
    pos = {}
    sc.fact("levels", [list(r) for r in levels])
    for li, row in enumerate(levels):
        if not row:
            continue
        cy = top + (bot - top) * li / max(len(levels) - 1, 1)
        slot = sc.w / (len(row) + 1)
        for i, lab in enumerate(row):
            cx = sc.w * (i + 1) / (len(row) + 1)
            nfs = fs
            # Shrink to the slot rather than letting siblings collide; the box is
            # the hit target now, so overlapping boxes would make the question
            # ambiguous, not merely untidy.
            while nfs > 9 and text_w(draw, load(sc.font_file, sc.font_index, nfs),
                                     lab) + 26 > slot * 0.94:
                nfs -= 1
            w = text_w(draw, load(sc.font_file, sc.font_index, nfs), lab) + 26
            pos[lab] = (cx, cy, w, nfs + 24, nfs)
    for li in range(1, len(levels)):
        for lab in levels[li]:
            if not levels[li - 1]:
                continue
            par = levels[li - 1][min(len(levels[li - 1]) - 1,
                                     (levels[li].index(lab) * len(levels[li - 1]))
                                     // max(len(levels[li]), 1))]
            a, b = pos[par][:4], pos[lab][:4]
            my = (a[1] + a[3] / 2 + b[1] - b[3] / 2) / 2
            sc.add(line(a[0], a[1] + a[3] / 2, a[0], my, t.muted, t.sw),
                   line(a[0], my, b[0], my, t.muted, t.sw),
                   line(b[0], my, b[0], b[1] - b[3] / 2, t.muted, t.sw, arrow=True))
    for i, (lab, (cx, cy, w, h, nfs)) in enumerate(pos.items()):
        sc.add(rect(cx - w / 2, cy - h / 2, w, h, t.fills[i % len(t.fills)], t.stroke,
                    t.sw, t.radius),
               text(cx, cy, lab, nfs, t.ink))
    return True


def b_network(sc, rng, draw, F):
    t, fs = sc.theme, round(BASE_FONT * .9)
    _, nouns = sc.domain
    labs = uniq_labels(rng, nouns, min(16, rng.randint(7, 10) + sc.complexity), 15)
    cx, cy = sc.w / 2, (sc.margin + sc.title_band + sc.h - sc.margin) / 2
    R = min(sc.w, sc.h) * 0.33
    pts = []
    for i, l in enumerate(labs):
        a = 2 * math.pi * i / len(labs) + rng.uniform(-.12, .12)
        rr = R * rng.uniform(.82, 1.06)
        pts.append((cx + math.cos(a) * rr * 1.45, cy + math.sin(a) * rr))
    seen = set()
    sc.fact("nodes", list(labs))
    for i in range(len(labs)):
        for _ in range(rng.randint(1, 2)):
            j = rng.randrange(len(labs))
            if i == j or (min(i, j), max(i, j)) in seen:
                continue
            seen.add((min(i, j), max(i, j)))
            sc.add(line(pts[i][0], pts[i][1], pts[j][0], pts[j][1], t.grid or t.muted,
                        t.sw * .9))
    sc.fact("edges", [{"a": labs[i], "b": labs[j]} for i, j in sorted(seen)])
    for i, (l, (px, py)) in enumerate(zip(labs, pts)):
        w = text_w(draw, load(sc.font_file, sc.font_index, fs), l) + 26
        sc.add(rect(px - w / 2, py - (fs + 18) / 2, w, fs + 18,
                    t.fills[i % len(t.fills)], t.stroke, t.sw, t.radius),
               text(px, py, l, fs, t.ink))
    return True


def b_timeline(sc, rng, draw, F):
    t, fs = sc.theme, round(BASE_FONT * .86)
    _, nouns = sc.domain
    labs = uniq_labels(rng, nouns, rng.randint(5, 7), 16)
    dates = sample_uniq(rng, QUARTERS, len(labs))
    dates.sort(key=QUARTERS.index)
    sc.fact("milestones", [{"label": l, "date": d} for l, d in zip(labs, dates)])
    y = (sc.margin + sc.title_band + sc.h - sc.margin) / 2
    x0, x1 = sc.margin + 60, sc.w - sc.margin - 60
    sc.add(line(x0, y, x1, y, t.muted, t.sw * 2.2))
    for i, (l, dt) in enumerate(zip(labs, dates)):
        px = x0 + (x1 - x0) * i / max(len(labs) - 1, 1)
        up = i % 2 == 0
        oy = -1 if up else 1
        sc.add(circle(px, y, t.sw * 4.4, t.fills[i % len(t.fills)], t.stroke, t.sw))
        sc.add(line(px, y + oy * 14, px, y + oy * 62, t.muted, t.sw))
        sc.add(text(px, y + oy * 84, l, fs, t.ink, role="milestone"))
        sc.add(text(px, y + oy * 84 + oy * 26, dt, round(fs * .82), t.muted, role="date"))
    return True


def b_gantt(sc, rng, draw, F):
    t, fs = sc.theme, round(BASE_FONT * .84)
    _, nouns = sc.domain
    tasks = uniq_labels(rng, nouns, min(12, rng.randint(5, 7) + sc.complexity), 16)
    x0 = sc.margin + 250
    x1 = sc.w - sc.margin - 40
    y0 = sc.margin + sc.title_band + 34
    rh = min(64, (sc.h - sc.margin - y0) / len(tasks))
    per = (x1 - x0) / 8
    for c in range(9):
        gx = x0 + per * c
        if t.grid:
            sc.add(line(gx, y0 - 22, gx, y0 + rh * len(tasks), t.grid, t.sw * .8))
        if c < 8:
            sc.add(text(gx + per / 2, y0 - 32, f"W{c + 1}", round(fs * .82),
                        readable_on(t.muted, t.bg), role="tick"))
    bars = []
    for i, tk in enumerate(tasks):
        cy = y0 + rh * (i + .5)
        s = rng.randint(0, 4)
        d = rng.randint(2, 8 - s) if 8 - s > 2 else 2
        bars.append({"task": tk, "start_week": s + 1, "duration_weeks": d})
        sc.add(text(x0 - 22, cy, tk, fs, t.ink, anchor="rm", role="task"))
        sc.add(rect(x0 + per * s, cy - rh * .28, per * d, rh * .56,
                    t.fills[i % len(t.fills)], t.stroke, t.sw, min(t.radius, 5)))
        sc.add(text(x0 + per * s + per * d / 2, cy, f"{d}w", round(fs * .8), t.ink,
                    role="value"))
    sc.fact("tasks", bars)
    return True


def b_sequence(sc, rng, draw, F):
    t, fs = sc.theme, round(BASE_FONT * .84)
    actors = sample_uniq(rng, ACTORS, rng.randint(3, 5))
    y0 = sc.margin + sc.title_band + 24
    yb = sc.h - sc.margin - 24
    for i, a in enumerate(actors):
        cx = sc.w * (i + 1) / (len(actors) + 1)
        w = text_w(draw, load(sc.font_file, sc.font_index, fs), a) + 30
        sc.add(rect(cx - w / 2, y0, w, fs + 22, t.fills[i % len(t.fills)], t.stroke,
                    t.sw, t.radius),
               text(cx, y0 + (fs + 22) / 2, a, fs, t.ink, role="actor"))
        sc.add(line(cx, y0 + fs + 22, cx, yb, t.muted, t.sw * .8, dash=True))
    msgs = []
    n = rng.randint(4, 6)
    for m in range(n):
        i, j = rng.sample(range(len(actors)), 2)
        ax = sc.w * (i + 1) / (len(actors) + 1)
        bx = sc.w * (j + 1) / (len(actors) + 1)
        my = y0 + fs + 60 + m * ((yb - y0 - fs - 80) / max(n - 1, 1))
        sc.add(line(ax, my, bx, my, t.ink, t.sw, arrow=True))
        lab = rng.choice(EDGE_LABELS)
        msgs.append({"from": actors[i], "to": actors[j], "label": lab, "order": m + 1})
        sc.add(text((ax + bx) / 2, my - 16, lab, round(fs * .84),
                    readable_on(t.muted, t.bg), role="message"))
    sc.fact("actors", actors)
    sc.fact("messages", msgs)
    return True


def b_treemap(sc, rng, draw, F):
    t, fs = sc.theme, round(BASE_FONT * .88)
    labs = sample_uniq(rng, REGIONS, rng.randint(5, 8))
    vals = [rng.randint(10, 60) for _ in labs]
    x0, y0 = sc.margin + 20, sc.margin + sc.title_band + 10
    W, H = sc.w - 2 * sc.margin - 40, sc.h - sc.margin - y0
    items = sorted(zip(labs, vals), key=lambda z: -z[1])
    sc.fact("blocks", [{"label": l, "value": v} for l, v in items])
    tot = sum(vals)
    cx, cy, cw, ch = x0, y0, W, H
    horiz = True
    for i, (l, v) in enumerate(items):
        last = i == len(items) - 1
        frac = v / tot if tot else 0
        if last:
            bw, bh = cw, ch
        elif horiz:
            bw, bh = cw * (v / sum(z[1] for z in items[i:])), ch
        else:
            bw, bh = cw, ch * (v / sum(z[1] for z in items[i:]))
        sc.add(rect(cx + 3, cy + 3, bw - 6, bh - 6, t.fills[i % len(t.fills)],
                    t.stroke, t.sw, min(t.radius, 4)))
        if bw > 90 and bh > 40:
            sc.add(text(cx + bw / 2, cy + bh / 2 - 9, l, fs, t.ink, role="cell"))
            sc.add(text(cx + bw / 2, cy + bh / 2 + 15, str(v), round(fs * .82),
                        readable_on(t.muted, t.fills[i % len(t.fills)]), role="value"))
        if not last:
            if horiz:
                cx += bw; cw -= bw
            else:
                cy += bh; ch -= bh
            horiz = not horiz
    return True


def b_quadrant(sc, rng, draw, F):
    t, fs = sc.theme, round(BASE_FONT * .86)
    x0, y0 = sc.margin + 120, sc.margin + sc.title_band + 20
    x1, y1 = sc.w - sc.margin - 60, sc.h - sc.margin - 60
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    sc.add(rect(x0, y0, x1 - x0, y1 - y0, None, t.muted, t.sw))
    sc.add(line(mx, y0, mx, y1, t.muted, t.sw), line(x0, my, x1, my, t.muted, t.sw))
    quads = ["Invest", "Sustain", "Divest", "Review"]
    sc.fact("quadrants", quads)
    for qi, q in enumerate(quads):
        qx = x0 + (x1 - x0) * (.25 if qi % 2 == 0 else .75)
        qy = y0 + (y1 - y0) * (.08 if qi < 2 else .92)
        sc.add(text(qx, qy, q, round(fs * .95), t.muted, role="quadrant"))
    plotted = []
    for i, l in enumerate(sample_uniq(rng, REGIONS, rng.randint(5, 8))):
        px = rng.uniform(x0 + 70, x1 - 70)
        py = rng.uniform(y0 + 60, y1 - 60)
        plotted.append({"label": l,
                        "quadrant": quads[(0 if px < mx else 1) + (0 if py < my else 2)]})
        sc.add(circle(px, py, 11, t.fills[i % len(t.fills)], t.stroke, t.sw))
        sc.add(text(px, py - 24, l, round(fs * .84), t.ink, role="point"))
    sc.fact("items", plotted)
    sc.add(text((x0 + x1) / 2, y1 + 32, rng.choice(METRICS), round(fs * .9), t.muted,
                role="axis"))
    return True


def b_mindmap(sc, rng, draw, F):
    t, fs = sc.theme, round(BASE_FONT * .86)
    dom, nouns = sc.domain
    labs = uniq_labels(rng, nouns, rng.randint(6, 8), 15)
    cx, cy = sc.w / 2, (sc.margin + sc.title_band + sc.h - sc.margin) / 2
    root = dom.split()[0]
    sc.fact("root", root)
    sc.fact("branches", list(labs))
    for i, l in enumerate(labs):
        a = 2 * math.pi * i / len(labs) - math.pi / 2
        rr = min(sc.w, sc.h) * 0.30
        px, py = cx + math.cos(a) * rr * 1.55, cy + math.sin(a) * rr
        sc.add(line(cx, cy, px, py, t.grid or t.muted, t.sw * 1.4))
        w = text_w(draw, load(sc.font_file, sc.font_index, fs), l) + 26
        sc.add(rect(px - w / 2, py - (fs + 18) / 2, w, fs + 18,
                    t.fills[i % len(t.fills)], t.stroke, t.sw, 14),
               text(px, py, l, fs, t.ink))
    rw = text_w(draw, load(sc.font_file, sc.font_index, round(fs * 1.2)), root) + 44
    sc.add(circle(cx, cy, max(rw / 2, 58), _shade(t.fills[0], -.12 if not t.dark else .12),
                  t.stroke, t.sw * 1.4),
           text(cx, cy, root, round(fs * 1.2), t.ink, role="root"))
    return True


def b_state(sc, rng, draw, F):
    t, fs = sc.theme, round(BASE_FONT * .86)
    _, nouns = sc.domain
    labs = uniq_labels(rng, nouns, rng.randint(4, 6), 14)
    y = (sc.margin + sc.title_band + sc.h - sc.margin) / 2
    n = len(labs)
    pos = []
    for i, l in enumerate(labs):
        px = sc.margin + 110 + (sc.w - 2 * sc.margin - 220) * i / max(n - 1, 1)
        py = y + (70 if i % 2 else -70)
        pos.append((px, py))
        r = max(text_w(draw, load(sc.font_file, sc.font_index, fs), l) / 2 + 22, 52)
        sc.add(circle(px, py, r, t.fills[i % len(t.fills)], t.stroke, t.sw))
        sc.add(text(px, py, l, fs, t.ink))
    trans = []
    for i in range(n - 1):
        a, b = pos[i], pos[i + 1]
        sc.add(line(a[0] + 46, a[1], b[0] - 46, b[1], t.muted, t.sw, arrow=True))
        lab = rng.choice(EDGE_LABELS)
        trans.append({"from": labs[i], "to": labs[i + 1], "label": lab})
        sc.add(text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 14, lab,
                    round(fs * .78), readable_on(t.muted, t.bg), role="transition"))
    sc.fact("states", labs)
    sc.fact("transitions", trans)
    return True


def _xform(prims, k, dx, dy, text_k=None):
    """Scale and translate a primitive list into a sub-rectangle.

    Prims are pure geometry, so any builder can be composed into a panel without
    knowing it is being composed -- which is what makes `dashboard` reuse the
    other fourteen rather than reimplement them.
    """
    out = []
    for p in prims:
        q = dict(p)
        if "sw" in q and q["sw"]:
            q["sw"] = p["sw"] * k
        kk = p["k"]
        if kk == "rect":
            q.update(x=p["x"] * k + dx, y=p["y"] * k + dy, w=p["w"] * k,
                     h=p["h"] * k, r=p.get("r", 0) * k)
        elif kk == "line":
            q.update(x0=p["x0"] * k + dx, y0=p["y0"] * k + dy,
                     x1=p["x1"] * k + dx, y1=p["y1"] * k + dy)
        elif kk in ("circle", "wedge"):
            q.update(cx=p["cx"] * k + dx, cy=p["cy"] * k + dy, r=p["r"] * k)
        elif kk == "poly":
            q["pts"] = [(x * k + dx, y * k + dy) for x, y in p["pts"]]
        elif kk == "text":
            # Text scales on its own factor. Shrinking labels by the same ratio as
            # the geometry pushes them under the legibility floor at `small`,
            # which is how dashboards lost every target at that rung; real
            # dashboards enlarge type relative to the panel for the same reason.
            q.update(x=p["x"] * k + dx, y=p["y"] * k + dy,
                     size=p["size"] * (text_k if text_k is not None else k))
        out.append(q)
    return out


def b_dashboard(sc, rng, draw, F):
    """Several panels on one canvas -- the density ScreenSpot-Pro screenshots have.

    Each panel is built on a Scene sized to the panel, so its builder lays out
    for the space it actually has and its type stays at natural size. Building
    full-canvas and shrinking instead pushed every label under the legibility
    floor at the `small` rung, which cost the scene all of its targets there.
    """
    t = sc.theme
    sub = [k for k in BUILDERS if k != "dashboard"]
    n = 4 if sc.complexity >= 3 else 2
    cols, rowsn = (2, 2) if n == 4 else (2, 1)
    pw = (sc.w - 2 * sc.margin - 22 * (cols - 1)) / cols
    ph = (sc.h - sc.margin - sc.title_band - sc.margin - 22 * (rowsn - 1)) / rowsn
    panels = []
    for idx in range(n):
        ct = sub[rng.randrange(len(sub))]
        tmp = Scene(gid=sc.gid, title="", ctype=ct, theme=t, font_file=sc.font_file,
                    font_index=sc.font_index, font_family=sc.font_family,
                    w=int(pw), h=int(ph), margin=22, title_band=26)
        tmp.domain = sc.domain
        tmp.complexity = 1                     # panels stay readable, not dense
        if BUILDERS[ct](tmp, rng, draw, F) is None:
            continue
        cx = sc.margin + (idx % cols) * (pw + 22)
        cy = sc.margin + sc.title_band + (idx // cols) * (ph + 22)
        sc.add(rect(cx, cy, pw, ph, None, t.stroke, t.sw * .9, t.radius))
        sc.add(text(cx + 12, cy + 9, ct.replace("_", " ").title(),
                    round(BASE_FONT * .72), t.muted, anchor="lt",
                    role="panel_title", target=False))
        panels.append({"panel": idx, "chart_type": ct,
                       "origin": [round(cx, 1), round(cy, 1)],
                       "size": [round(pw, 1), round(ph, 1)],
                       "facts": tmp.facts})
        sc.prims.extend(_xform(tmp.prims, 1.0, cx, cy))
    sc.fact("panels", panels)
    return True


def _decorate(sc, rng, draw):
    """Decoy text at higher complexity: captions, badges, footnotes.

    These are never targets. They exist so the scene contains text that looks
    like a target and is not one, which is what makes a localization question
    non-trivial.
    """
    t = sc.theme
    if sc.complexity < 3:
        return
    fs = round(BASE_FONT * .62)
    sc.add(text(MARGIN, BASE_H - MARGIN * .5,
                f"source: internal | n={rng.randint(120, 9800)} | "
                f"rev {rng.randint(2, 19)}", fs, t.muted, anchor="lt",
                role="footnote", target=False))
    sc.add(text(BASE_W - MARGIN, MARGIN * .62,
                rng.choice(["DRAFT", "INTERNAL", "CONFIDENTIAL", "PREVIEW"]),
                round(fs * 1.05), t.muted, anchor="rt", role="badge", target=False))
    if sc.complexity >= 4:
        sc.add(text(BASE_W - MARGIN, BASE_H - MARGIN * .5,
                    f"generated {rng.randint(1, 28)}/{rng.randint(1, 12)}", fs,
                    t.muted, anchor="rt", role="footnote", target=False))


BUILDERS = {
    "flowchart": b_flowchart, "bar_chart": b_bar, "line_chart": b_line,
    "scatter": b_scatter, "pie_chart": b_pie, "table": b_table, "org_chart": b_org,
    "network": b_network, "timeline": b_timeline, "gantt": b_gantt,
    "sequence": b_sequence, "treemap": b_treemap, "quadrant": b_quadrant,
    "mindmap": b_mindmap, "state_machine": b_state, "dashboard": b_dashboard,
}


# ----------------------------------------------------------------- rendering

# ------------------------------------------------------- hit boxes (the "button")
# ScreenSpot-Pro's gold is the *widget* box -- the region you could click -- not
# the glyph outline. Scoring against a tight ink box would be a harder and
# different task, and would leave nothing widget-shaped to measure distance
# against. So every label gets a hit box: the shape that encloses it where one
# exists (a node rectangle, a state circle), otherwise the ink box grown by
# button-like padding.

HIT_PAD_X, HIT_PAD_Y = 0.42, 0.34      # of font size, per side
MAX_HIT_FRAC = 0.06                    # a shape bigger than this is scenery,
                                       # not a button (panel frames, plot areas)


def _enclosing(prims, upto, ink, W, H, s):
    """Smallest already-drawn shape that fully contains the label's ink box.

    Containment must be total, not just of the centre: a shape that clips its own
    label would give a hit box the gold text pokes out of, and the invariant
    "the text is inside the box you are asked to click" is the whole point.
    """
    best, ba = None, float("inf")
    for p in prims[:upto]:
        if p["k"] == "rect":
            x0, y0 = p["x"] * s, p["y"] * s
            x1, y1 = (p["x"] + p["w"]) * s, (p["y"] + p["h"]) * s
            a = (x1 - x0) * (y1 - y0)
            box = [x0, y0, x1, y1]
        elif p["k"] == "circle":
            ccx, ccy, r = p["cx"] * s, p["cy"] * s, p["r"] * s
            a = math.pi * r * r
            box = [ccx - r, ccy - r, ccx + r, ccy + r]
        else:
            continue                    # wedges/lines are not button-shaped
        if not (box[0] <= ink[0] and box[1] <= ink[1]
                and ink[2] <= box[2] and ink[3] <= box[3]):
            continue
        if a > MAX_HIT_FRAC * W * H or a >= ba:
            continue
        best, ba = box, a
    return best


def hit_box(prims, upto, ink, fs, W, H, s):
    """The scored target: enclosing widget if there is one, else padded ink."""
    box = _enclosing(prims, upto, ink, W, H, s)
    src = "shape"
    if box is None:
        px, py = HIT_PAD_X * fs, HIT_PAD_Y * fs
        box = [ink[0] - px, ink[1] - py, ink[2] + px, ink[3] + py]
        src = "padded_text"
    box = [min(box[0], ink[0]), min(box[1], ink[1]),
           max(box[2], ink[2]), max(box[3], ink[3])]
    box = [max(0.0, box[0]), max(0.0, box[1]),
           min(float(W), box[2]), min(float(H), box[3])]
    return [round(v, 2) for v in box], src


def enforce_legibility(sc: Scene, min_scale: float,
                       min_px: int = MIN_LEGIBLE_PX) -> int:
    """Grow any text that would render below `min_px` at the smallest scale.

    Applied to *every* label, not only targets: a scene where the axis ticks are
    unreadable is a bad stimulus even if the asked-about label happens to be
    large. Sizes are set in base units, so the floor is `min_px / min_scale`.
    """
    floor = min_px / max(min_scale, 1e-6)
    n = 0
    for p in sc.prims:
        # Decorative text (footnotes, badges, panel titles) is excluded: it is
        # never a question target, and enlarging it collides with real content.
        if p["k"] == "text" and p["target"] and p["size"] < floor:
            p["size"] = floor
            n += 1
    return n


def occlusion(full: Image.Image, over: Image.Image, box) -> float:
    """Fraction of a label's ink box that something later drew over.

    Measured, not inferred from draw order. `over` is the same scene with every
    label composited last, so it is what the text would look like if nothing
    covered it; any pixel that differs inside the box is a pixel some shape took.
    """
    import numpy as np
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1 = min(full.width, int(math.ceil(box[2])))
    y1 = min(full.height, int(math.ceil(box[3])))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    a = np.asarray(full.crop((x0, y0, x1, y1)).convert("RGB"), dtype=np.int16)
    b = np.asarray(over.crop((x0, y0, x1, y1)).convert("RGB"), dtype=np.int16)
    return float((np.abs(a - b).max(axis=2) > 12).mean())


def _bg(sc, d, s):
    t = sc.theme
    W, H = round(sc.w * s), round(sc.h * s)
    d.rectangle([0, 0, W, H], fill=t.bg)
    if t.pattern == "grid" and t.grid:
        step = 48 * s
        x = step
        while x < W:
            d.line([(x, 0), (x, H)], fill=t.grid, width=1); x += step
        y = step
        while y < H:
            d.line([(0, y), (W, y)], fill=t.grid, width=1); y += step
    elif t.pattern == "dots" and t.grid:
        step = 40 * s
        y = step
        while y < H:
            x = step
            while x < W:
                d.ellipse([x - 1.4 * s, y - 1.4 * s, x + 1.4 * s, y + 1.4 * s], fill=t.grid)
                x += step
            y += step


def _arrowhead(d, x0, y0, x1, y1, col, s, sw):
    ang = math.atan2(y1 - y0, x1 - x0)
    L, W = 11 * s, 5.0 * s
    d.polygon([(x1, y1),
               (x1 - L * math.cos(ang) + W * math.sin(ang),
                y1 - L * math.sin(ang) - W * math.cos(ang)),
               (x1 - L * math.cos(ang) - W * math.sin(ang),
                y1 - L * math.sin(ang) + W * math.cos(ang))], fill=col)


def render(sc: Scene, s: float, skip_text: bool = False):
    """Rasterize at scale `s`; return the image and every text's measured ink box.

    Called twice per scale: once with `skip_text` to obtain the true background
    behind each label, then once normally so contrast can be measured against
    what is actually underneath rather than against the theme's page colour.
    """
    W, H = round(sc.w * s), round(sc.h * s)
    im = Image.new("RGB", (W, H), sc.theme.bg)
    d = ImageDraw.Draw(im)
    _bg(sc, d, s)
    under = over = None
    if not skip_text:
        under, _ = render(sc, s, skip_text=True)
        over = under.copy()                     # same scene, labels composited last
        od = ImageDraw.Draw(over)
        for q in sc.prims:
            if q["k"] == "text":
                qf = load(sc.font_file, sc.font_index, max(5, round(q["size"] * s)))
                od.text((q["x"] * s, q["y"] * s), q["s"], font=qf,
                        fill=q["fill"], anchor=q["anchor"])
    gold = {}
    ti = 0
    for pi, p in enumerate(sc.prims):
        k = p["k"]
        sw = max(1, round(p.get("sw", 1.5) * s)) if p.get("stroke") else 0
        if k == "rect":
            box = [p["x"] * s, p["y"] * s, (p["x"] + p["w"]) * s, (p["y"] + p["h"]) * s]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            r = p.get("r", 0) * s
            if r >= 1:
                d.rounded_rectangle(box, radius=r, fill=p["fill"], outline=p["stroke"],
                                    width=sw)
            else:
                d.rectangle(box, fill=p["fill"], outline=p["stroke"], width=sw)
        elif k == "line":
            xy = [p["x0"] * s, p["y0"] * s, p["x1"] * s, p["y1"] * s]
            w = max(1, round(p["sw"] * s))
            if p.get("dash"):
                _dashed(d, *xy, p["stroke"], w, s)
            else:
                d.line(xy, fill=p["stroke"], width=w)
            if p.get("arrow"):
                _arrowhead(d, *xy, p["stroke"], s, w)
        elif k == "circle":
            box = [(p["cx"] - p["r"]) * s, (p["cy"] - p["r"]) * s,
                   (p["cx"] + p["r"]) * s, (p["cy"] + p["r"]) * s]
            d.ellipse(box, fill=p["fill"], outline=p["stroke"], width=sw)
        elif k == "poly":
            d.polygon([(x * s, y * s) for x, y in p["pts"]], fill=p["fill"],
                      outline=p["stroke"])
        elif k == "wedge":
            box = [(p["cx"] - p["r"]) * s, (p["cy"] - p["r"]) * s,
                   (p["cx"] + p["r"]) * s, (p["cy"] + p["r"]) * s]
            d.pieslice(box, p["a0"], p["a1"], fill=p["fill"], outline=p["stroke"],
                       width=sw)
        elif k == "text":
            fs = max(5, round(p["size"] * s))
            f = load(sc.font_file, sc.font_index, fs)
            x, y = p["x"] * s, p["y"] * s
            if skip_text:
                continue
            try:
                d.text((x, y), p["s"], font=f, fill=p["fill"], anchor=p["anchor"])
                bb = d.textbbox((x, y), p["s"], font=f, anchor=p["anchor"])
            except OSError as e:
                raise SystemExit(
                    f"font {Path(sc.font_file).name} failed to render "
                    f"{p['s']!r} at {fs}px ({e}). Some faces raise on rare glyphs "
                    f"at small ppem; keep generated text ASCII.") from e
            bg = _median_rgb(under, bb) if under is not None else _rgb(sc.theme.bg)
            cr = contrast_ratio(p["fill"], bg)
            hb, hsrc = hit_box(sc.prims, pi, bb, fs, W, H, s)
            gold[ti] = {"text": p["s"], "role": p["role"], "eligible": p["target"],
                        "bbox": hb, "hit_source": hsrc,
                        "ink_bbox": [round(v, 2) for v in bb],
                        "center": [round((hb[0] + hb[2]) / 2, 2),
                                   round((hb[1] + hb[3]) / 2, 2)],
                        "ink_center": [round((bb[0] + bb[2]) / 2, 2),
                                       round((bb[1] + bb[3]) / 2, 2)],
                        "area_frac": round(max(0, hb[2] - hb[0]) * max(0, hb[3] - hb[1])
                                           / (W * H), 8),
                        "ink_area_frac": round(max(0, bb[2] - bb[0]) * max(0, bb[3] - bb[1])
                                               / (W * H), 8),
                        "font_px": fs,
                        "occluded_frac": round(occlusion(im, over, bb), 4)
                        if over is not None else 0.0,
                        "contrast": round(cr, 2),
                        "required_contrast": required_contrast(fs),
                        "bg_rgb": list(bg) if under is not None else None,
                        "in_bounds": bb[0] >= 0 and bb[1] >= 0 and bb[2] <= W and bb[3] <= H}
            ti += 1
    return im, gold


def _dashed(d, x0, y0, x1, y1, col, w, s):
    n = math.hypot(x1 - x0, y1 - y0)
    if n < 1:
        return
    step = 9 * s
    k = 0.0
    while k < n:
        a = k / n
        b = min(1.0, (k + step * .55) / n)
        d.line([x0 + (x1 - x0) * a, y0 + (y1 - y0) * a,
                x0 + (x1 - x0) * b, y0 + (y1 - y0) * b], fill=col, width=w)
        k += step


# ----------------------------------------------------------------- SVG

def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def emit_svg(sc: Scene, draw: ImageDraw.ImageDraw) -> str:
    t = sc.theme
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{sc.w}" height="{sc.h}" '
         f'viewBox="0 0 {sc.w} {sc.h}">',
         f'<rect width="{sc.w}" height="{sc.h}" fill="{t.bg}"/>',
         '<defs><marker id="ah" markerWidth="11" markerHeight="8" refX="10" refY="4" '
         f'orient="auto" markerUnits="userSpaceOnUse">'
         f'<path d="M0,0 L11,4 L0,8 z" fill="{t.muted}"/></marker></defs>']
    if t.pattern == "grid" and t.grid:
        for x in range(48, sc.w, 48):
            o.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{sc.h}" stroke="{t.grid}" stroke-width="1"/>')
        for y in range(48, sc.h, 48):
            o.append(f'<line x1="0" y1="{y}" x2="{sc.w}" y2="{y}" stroke="{t.grid}" stroke-width="1"/>')
    elif t.pattern == "dots" and t.grid:
        for y in range(40, sc.h, 40):
            for x in range(40, sc.w, 40):
                o.append(f'<circle cx="{x}" cy="{y}" r="1.4" fill="{t.grid}"/>')

    ti = 0
    for p in sc.prims:
        k = p["k"]
        st = f' stroke="{p["stroke"]}" stroke-width="{p.get("sw", 1.5)}"' if p.get("stroke") else ""
        fl = f'fill="{p["fill"]}"' if p.get("fill") else 'fill="none"'
        if k == "rect":
            o.append(f'<rect x="{p["x"]:.2f}" y="{p["y"]:.2f}" width="{p["w"]:.2f}" '
                     f'height="{p["h"]:.2f}" rx="{p.get("r", 0)}" {fl}{st}/>')
        elif k == "line":
            dash = ' stroke-dasharray="5 4"' if p.get("dash") else ""
            mk = ' marker-end="url(#ah)"' if p.get("arrow") else ""
            o.append(f'<line x1="{p["x0"]:.2f}" y1="{p["y0"]:.2f}" x2="{p["x1"]:.2f}" '
                     f'y2="{p["y1"]:.2f}" stroke="{p["stroke"]}" '
                     f'stroke-width="{p["sw"]:.2f}"{dash}{mk}/>')
        elif k == "circle":
            o.append(f'<circle cx="{p["cx"]:.2f}" cy="{p["cy"]:.2f}" r="{p["r"]:.2f}" {fl}{st}/>')
        elif k == "poly":
            pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in p["pts"])
            o.append(f'<polygon points="{pts}" {fl}{st}/>')
        elif k == "wedge":
            a0, a1 = math.radians(p["a0"]), math.radians(p["a1"])
            x0 = p["cx"] + p["r"] * math.cos(a0); y0 = p["cy"] + p["r"] * math.sin(a0)
            x1 = p["cx"] + p["r"] * math.cos(a1); y1 = p["cy"] + p["r"] * math.sin(a1)
            large = 1 if (p["a1"] - p["a0"]) % 360 > 180 else 0
            o.append(f'<path d="M{p["cx"]:.2f},{p["cy"]:.2f} L{x0:.2f},{y0:.2f} '
                     f'A{p["r"]:.2f},{p["r"]:.2f} 0 {large} 1 {x1:.2f},{y1:.2f} Z" {fl}{st}/>')
        elif k == "text":
            f = load(sc.font_file, sc.font_index, max(5, round(p["size"])))
            ls = draw.textbbox((0, 0), p["s"], font=f, anchor="ls")
            bb = draw.textbbox((0, 0), p["s"], font=f, anchor=p["anchor"])
            # Explicit alphabetic baseline: SVG's dominant-baseline centres on font
            # metrics, PIL's anchor centres on ink, and they disagree by 2-3px.
            base_y = p["y"] + (bb[1] - ls[1])
            ha = {"m": "middle", "l": "start", "r": "end"}[p["anchor"][0]]
            adv = f.getlength(p["s"])
            o.append(f'<text id="t{ti}" data-role="{p["role"]}" '
                     f'data-target="{str(bool(p["target"])).lower()}" '
                     f'x="{p["x"]:.2f}" y="{base_y:.2f}" '
                     f'font-family="{sc.font_family}" font-size="{p["size"]}" '
                     f'fill="{p["fill"]}" text-anchor="{ha}" '
                     f'textLength="{adv:.2f}" lengthAdjust="spacingAndGlyphs">'
                     f'{_esc(p["s"])}</text>')
            ti += 1
    o.append("</svg>")
    return "\n".join(o)


# ----------------------------------------------------------------- questions

def _overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def eligible_targets(gold: dict, min_contrast: float | None = None,
                     min_font_px: int = 7) -> tuple[list[int], dict]:
    """Filter to targets a question can fairly be asked about.

    A target is rejected when it is ambiguous (a duplicate string, or ink
    overlapping another label), unreadable (below the WCAG AA ratio for its size,
    or rendered too small to resolve), or clipped by the canvas. Every rejection
    is counted so the manifest reports what was excluded instead of quietly
    shrinking the pool.
    """
    counts = {}
    for g in gold.values():
        counts[g["text"]] = counts.get(g["text"], 0) + 1
    ok = []
    rej = {"overlap": 0, "duplicate": 0, "ineligible": 0, "tiny": 0,
           "low_contrast": 0, "out_of_bounds": 0, "font_too_small": 0, "occluded": 0,
           "hitbox_contains_other": 0}
    for i, g in gold.items():
        if not g["eligible"]:
            rej["ineligible"] += 1; continue
        if counts[g["text"]] > 1:
            rej["duplicate"] += 1; continue
        if g["ink_bbox"][2] - g["ink_bbox"][0] < 6 or g["ink_bbox"][3] - g["ink_bbox"][1] < 5:
            rej["tiny"] += 1; continue
        if not g.get("in_bounds", True):
            rej["out_of_bounds"] += 1; continue
        if g["font_px"] < min_font_px:
            rej["font_too_small"] += 1; continue
        if g.get("occluded_frac", 0.0) > 0.005:
            rej["occluded"] += 1; continue
        need = min_contrast if min_contrast is not None else g["required_contrast"]
        if g.get("contrast", 99) < need:
            rej["low_contrast"] += 1; continue
        if any(j != i and _overlaps(g["ink_bbox"], h["ink_bbox"]) for j, h in gold.items()):
            rej["overlap"] += 1; continue
        hb = g["bbox"]
        if any(j != i and _overlaps(hb, h["ink_bbox"]) for j, h in gold.items()):
            rej["hitbox_contains_other"] += 1; continue
        ok.append(i)
    return ok, rej


def audit_readability(gold: dict) -> list[dict]:
    """Every label failing AA, target or not -- a styling bug, not just a filter."""
    out = []
    for i, g in gold.items():
        if g.get("contrast") is None:
            continue
        if g["contrast"] < g["required_contrast"]:
            out.append({"text": g["text"], "role": g["role"],
                        "contrast": g["contrast"], "required": g["required_contrast"],
                        "font_px": g["font_px"], "bg_rgb": g.get("bg_rgb")})
    return out


def _neighbour(gold, ids, i, direction):
    a = gold[i]["center"]
    ah = gold[i]["bbox"][3] - gold[i]["bbox"][1]
    best, bd = None, 1e18
    for j in ids:
        if j == i:
            continue
        b = gold[j]["center"]
        dx, dy = b[0] - a[0], b[1] - a[1]
        if direction == "left" and not (dx < -4 and abs(dy) < ah * 1.1):
            continue
        if direction == "right" and not (dx > 4 and abs(dy) < ah * 1.1):
            continue
        if direction == "above" and not (dy < -4 and abs(dx) < 90):
            continue
        if direction == "below" and not (dy > 4 and abs(dx) < 90):
            continue
        dist = abs(dx) if direction in ("left", "right") else abs(dy)
        if dist < bd:
            best, bd = j, dist
    return best


def make_questions(sc, gold, ids, W, H, rng, per):
    qs = []
    picks = ids[:]
    rng.shuffle(picks)
    for i in picks[:per]:
        g = gold[i]
        bb, ctr = g["bbox"], g["center"]
        # ScreenSpot-Pro convention, so these are scorable by the same code path
        # and directly comparable to that arm: the element is described, the
        # harness prepends POINT_INSTRUCTION, and the model answers in a 0-1000
        # normalized space. Pixel coordinates would inject a coordinate-space
        # error the model cannot avoid -- it never sees the native resolution,
        # because the API downscales before the image reaches it.
        ink = g["ink_bbox"]
        nb = [round(bb[0] / W, 6), round(bb[1] / H, 6),
              round(bb[2] / W, 6), round(bb[3] / H, 6)]
        nink = [round(ink[0] / W, 6), round(ink[1] / H, 6),
                round(ink[2] / W, 6), round(ink[3] / H, 6)]
        qs.append({"qtype": "point", "target_idx": i, "target_text": g["text"],
                   "target_role": g["role"], "answer_type": "point",
                   "prompt_style": "screenspot_pro",
                   "question": f'the text "{g["text"]}"',
                   "answer": {"x": round(ctr[0] / W * 1000),
                              "y": round(ctr[1] / H * 1000)},
                   "gold_bbox_norm": nb,
                   "gold_center_norm": [round(ctr[0] / W, 6), round(ctr[1] / H, 6)],
                   "gold_bbox_px": bb, "gold_center_px": ctr,
                   "hit_source": g["hit_source"],
                   "text_ink_bbox_norm": nink, "text_ink_bbox_px": ink,
                   "scoring": "point_in_bbox"})
    made = 0
    for i in picks:
        if made >= max(2, per // 2):
            break
        dirn = rng.choice(["left", "right", "above", "below"])
        j = _neighbour(gold, ids, i, dirn)
        if j is None:
            continue
        phrase = {"left": "to the left of", "right": "to the right of",
                  "above": "above", "below": "below"}[dirn]
        qs.append({"qtype": "relation", "target_idx": j, "target_text": gold[j]["text"],
                   "target_role": gold[j]["role"],
                   "question": (f'In this {sc.ctype.replace("_", " ")}, which text label is '
                                f'immediately {phrase} "{gold[i]["text"]}"? Answer with '
                                f'the text only.'),
                   "answer": gold[j]["text"], "anchor_text": gold[i]["text"],
                   "direction": dirn, "gold_bbox_px": gold[j]["bbox"],
                   "scoring": "exact_match"})
        made += 1
    for i in picks[:max(2, per // 2)]:
        ctr = gold[i]["center"]
        qs.append({"qtype": "reverse", "target_idx": i, "target_text": gold[i]["text"],
                   "target_role": gold[i]["role"],
                   "question": (f'What text appears at pixel coordinates '
                                f'({round(ctr[0])}, {round(ctr[1])}) in this image? '
                                f'Answer with the text only.'),
                   "answer": gold[i]["text"],
                   "probe_point_px": [round(ctr[0]), round(ctr[1])],
                   "gold_bbox_px": gold[i]["bbox"], "scoring": "exact_match"})
    return qs


# ----------------------------------------------------------------- driver

def build(out: Path, count: int, seed: int, scales: dict, per: int,
          types: list[str], images: bool = True, min_contrast: float | None = None,
          min_font_px: int = MIN_LEGIBLE_PX, complexity: int = 1):
    (out / "svg").mkdir(parents=True, exist_ok=True)
    if images:
        (out / "images").mkdir(parents=True, exist_ok=True)
    fonts = available_fonts(verbose=True)
    probe = Image.new("RGB", (8, 8))
    pdraw = ImageDraw.Draw(probe)

    rows, skipped, unreadable, scene_rows = [], [], [], []
    for gid in range(count):
        rng = random.Random(seed * 100003 + gid * 7919)
        ctype = types[gid % len(types)] if len(types) > 1 else types[0]
        theme = THEMES[rng.randrange(len(THEMES))]
        fpath, fidx, fam = fonts[rng.randrange(len(fonts))]
        dom, nouns = rng.choice(DOMAINS)
        sc = Scene(gid=gid, title=f"{dom} - {ctype.replace('_', ' ')} {gid:04d}",
                   ctype=ctype, theme=theme, font_file=fpath, font_index=fidx,
                   font_family=fam)
        sc.domain = (dom, nouns)
        sc.complexity = complexity
        F = load(fpath, fidx, BASE_FONT)
        _title(sc, pdraw, F)
        if BUILDERS[ctype](sc, rng, pdraw, F) is None:
            skipped.append((gid, ctype, "builder declined"))
            continue
        _decorate(sc, rng, pdraw)
        enforce_legibility(sc, min(scales.values()))

        (out / "svg" / f"g{gid:04d}.svg").write_text(emit_svg(sc, pdraw))
        scene_rows.append({
            "graph_id": gid, "chart_type": ctype, "theme": theme.name,
            "font_family": fam, "title": sc.title, "complexity": complexity,
            "domain": dom, "svg": f"svg/g{gid:04d}.svg",
            "canvas": [sc.w, sc.h],
            "images": {sn: f"images/g{gid:04d}_{sn}.png" for sn in scales},
            # Text prims in draw order; index i is <text id="ti"> in the SVG and
            # `target_idx` in manifest.jsonl, so the three files join cleanly.
            "texts": [{"idx": i, "text": p["s"], "role": p["role"],
                       "targetable": bool(p["target"])}
                      for i, p in enumerate(sc.texts)],
            "facts": sc.facts,
        })

        for sname, s in scales.items():
            im, gold = render(sc, s)
            W, H = im.size
            ids, rej = eligible_targets(gold, min_contrast, min_font_px)
            for bad in audit_readability(gold):
                unreadable.append({"graph_id": gid, "chart_type": ctype,
                                   "theme": theme.name, "resolution": sname, **bad})
            if not ids:
                skipped.append((gid, ctype, f"no eligible target at {sname}"))
                continue
            rel = f"images/g{gid:04d}_{sname}.png"
            if images:
                im.save(out / rel, "PNG", optimize=True)
            ew, eh = effective_size(W, H)
            qrng = random.Random(seed * 31 + gid)
            for qi, q in enumerate(make_questions(sc, gold, ids, W, H, qrng, per)):
                rows.append({
                    "uid": f"svgloc:{gid:04d}:{sname}:{qi:02d}",
                    "graph_id": gid, "chart_type": ctype, "theme": theme.name,
                    "font_family": fam, "resolution": sname, "scale": s,
                    "image": rel, "svg": f"svg/g{gid:04d}.svg",
                    "image_px": [W, H], "effective_px": [ew, eh],
                    "downscaled_by_api": [W, H] != [ew, eh],
                    "title": sc.title, "complexity": complexity,
                    "n_texts": len(gold),
                    "n_eligible_targets": len(ids), "rejected_targets": rej,
                    **q,
                    "target_area_frac": gold[q["target_idx"]]["area_frac"],
                    "target_ink_area_frac": gold[q["target_idx"]]["ink_area_frac"],
                    "hit_source": gold[q["target_idx"]]["hit_source"],
                    "font_px": gold[q["target_idx"]]["font_px"],
                    "target_contrast": gold[q["target_idx"]]["contrast"],
                    "target_occluded_frac": gold[q["target_idx"]]["occluded_frac"],
                    "target_bg_rgb": gold[q["target_idx"]].get("bg_rgb"),
                })
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "scenes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scene_rows))
    (out / "readability_report.json").write_text(json.dumps(
        {"unreadable_labels": unreadable, "skipped_scenes": skipped,
         "min_contrast": min_contrast, "min_font_px": min_font_px}, indent=1))
    return rows, skipped, unreadable


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=Path, default=Path("data/svg_localization"))
    ap.add_argument("--questions-per-graph", type=int, default=4)
    ap.add_argument("--scales", default="small=0.6,medium=1.0,large=2.0")
    ap.add_argument("--types", default="all", help="comma list, or 'all'")
    ap.add_argument("--list-types", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--min-contrast", type=float, default=None,
                    help="override the WCAG AA threshold (4.5 normal / 3.0 large)")
    ap.add_argument("--complexity", type=int, default=1, choices=[1, 2, 3, 4, 5],
                    help="scene density: more nodes, more edges, decoy text")
    ap.add_argument("--min-font-px", type=int, default=MIN_LEGIBLE_PX,
                    help="reject targets rendered below this pixel size")
    a = ap.parse_args(argv)

    if a.list_types:
        for k in BUILDERS:
            print(k)
        return 0

    scales = {}
    for part in a.scales.split(","):
        k, _, v = part.partition("=")
        scales[k.strip()] = float(v)
    types = list(BUILDERS) if a.types == "all" else [t.strip() for t in a.types.split(",")]
    bad = [t for t in types if t not in BUILDERS]
    if bad:
        raise SystemExit(f"unknown type(s): {bad}. Try --list-types")

    rows, skipped, unreadable = build(a.out, a.count, a.seed, scales,
                                      a.questions_per_graph, types,
                                      images=not a.no_images,
                                      min_contrast=a.min_contrast,
                                      min_font_px=a.min_font_px,
                                      complexity=a.complexity)
    byt, byr, byth = {}, {}, {}
    for r in rows:
        byt[r["chart_type"]] = byt.get(r["chart_type"], 0) + 1
        byr[r["resolution"]] = byr.get(r["resolution"], 0) + 1
        byth[r["theme"]] = byth.get(r["theme"], 0) + 1
    print(f"{len(rows)} questions over {len({r['graph_id'] for r in rows})} scenes "
          f"-> {a.out}")
    print(f"  chart types : {len(byt)}  {', '.join(sorted(byt))}")
    print(f"  themes      : {len(byth)}  {', '.join(sorted(byth))}")
    print(f"  fonts       : {len({r['font_family'] for r in rows})}")
    for k, v in byr.items():
        print(f"  {k:8s} {v:5d} questions")
    if rows:
        cs = sorted(r["target_contrast"] for r in rows)
        print(f"  target contrast: min {cs[0]:.2f}  median {cs[len(cs) // 2]:.2f}  "
              f"max {cs[-1]:.2f}  (all targets meet WCAG AA for their size)")
    if unreadable:
        print(f"  ! {len(unreadable)} label(s) below AA were excluded from targets; "
              f"see {a.out / 'readability_report.json'}")
        seen = set()
        for u in unreadable:
            k = (u["chart_type"], u["theme"], u["role"])
            if k in seen:
                continue
            seen.add(k)
            print(f"      {u['chart_type']}/{u['theme']}: {u['role']} "
                  f"\"{u['text'][:28]}\" {u['contrast']:.2f} < {u['required']}")
    for gid, ct, why in skipped:
        print(f"  ! skipped g{gid:04d} ({ct}): {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
