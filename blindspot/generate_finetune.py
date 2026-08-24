"""Build the finetuning data: the resolution ladder, exact ink boxes, SFT records.

Why this exists
---------------
Part 2 measured a localization defect. This module builds the data that would
fix it. It constructs datasets; it never trains, and it never calls the API.

Three subcommands, one file, because they share three things that must not
drift apart: the canvas ladder, the scene reconstruction, and the box measured
off the raster.

    python -m blindspot.generate_finetune ladder  --out data/svgloc_mr
    python -m blindspot.generate_finetune samples --n 20 --seed 0 --out FILE
    python -m blindspot.generate_finetune audit   --dataset data/svgloc_mr

`ladder` -- Part 2's set is one canvas shape (1500x950) at three scales, of which
only the smallest reaches the model untouched; the other two are downscaled by
the API, so they measure the same delivered pixels twice. As training data that
teaches one frame shape and one size, and the defect being fixed is precisely
that the model does not treat coordinates as independent of its frame. So: six
aspect ratios x four sizes, every one of them delivered exactly as rendered.

`ExactInk` -- the manifest's `text_ink_bbox` is PIL's `ImageDraw.textbbox`, the
font's *layout* box (advance widths, ascent, descent), not the pixels the
rasteriser paints. Glyphs overhang their advance width, so the recorded box
clips the last glyph and the bottom of the text. Harmless for Part 2, which
scored a click against a much wider hit box. Not harmless when the box becomes
a regression target: at these sizes a 30x9px box grown one pixel per side falls
to 0.77 IoU. part3.md quotes median overlap 0.899 over 400 items with 18.5%
below 0.75; `ExactInk.verify` re-measures it on the committed set and reports
0.908 with a much thinner tail (see its docstring).

`samples` -- one record per line: image, question, and the box the model should
return. The supervision target is a box, not a point: any point inside the
target is equally correct, so the point in Part 2's manifest is one arbitrary
choice out of thousands and the model is penalised for picking a different,
equally valid one. A box has one correct answer and overlap with it is a graded
signal.

`audit` -- five checks over a ladder dataset, plus an eyeball gallery, because a
box that is arithmetically valid can still sit two pixels off the glyphs.

Nothing here re-implements the generator: scenes are rebuilt by calling
`blindspot.generate.gen_svg_localization` with the arguments the committed dataset
was built with, so a change to a builder cannot make the training boxes and the
evaluation boxes describe different pictures.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from blindspot import generate as G

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------- the ladder

# The caps the API applies before the model sees anything. Taken from the
# generator rather than restated, so there is one definition in the repo.
MAX_EDGE, MAX_PIXELS = G.API_EDGE, G.API_PIXELS      # 1568 px, ~1.15 MP

# Four sizes per aspect. r100 is the largest the API delivers untouched; the
# rest are fractions of it, so every rung is the *same scene* at a different
# delivered resolution and nothing else varies with size.
RUNGS = {"r55": 0.55, "r70": 0.70, "r85": 0.85, "r100": 1.00}

# Six shapes, widest first. `standard` is Part 2's own canvas, kept so the
# ladder contains the shape every published number was measured on. Portrait
# matters because an entirely landscape set never requires the model to
# normalise against a taller-than-wide frame.
ASPECTS = [
    ("ultrawide", 7 / 3),                    # 21:9  -- the edge cap binds here
    ("wide", 16 / 9),
    ("standard", G.BASE_W / G.BASE_H),       # 1500x950, Part 2's canvas
    ("classic", 4 / 3),
    ("square", 1.0),
    ("portrait", 3 / 4),
]

# Scene construction. The committed set was generated with these; the ladder
# reuses them so the two are the same scenes at different frame shapes.
SEED, COMPLEXITY = 17, 4

SOURCE_REL = "data/svg_localization"        # what `samples` reads
MR_REL = "data/svgloc_mr"                   # what `ladder` writes, `audit` reads


def canvas_for(ratio: float) -> tuple[int, int]:
    """Largest canvas at `ratio` that survives both API caps untouched.

    Not simply `sqrt(MAX_PIXELS * ratio)`: the scene is laid out at the
    generator's base height and *then* scaled to the caps, so the integral base
    canvas carries its own rounding into the result -- 21:9 lands on 1568x671
    rather than the 1568x672 the continuous solution would give. Truncating
    rather than rounding is what keeps the result strictly inside both caps:
    rounding up on the long edge would hand the API a 1569px image and it would
    downscale the whole set, which is the one thing this dataset must not do.

    Whichever cap binds first is honoured exactly rather than through the scale
    factor, because `int(w * (MAX_EDGE / w))` can land a pixel short of
    MAX_EDGE on a float that rounds down.
    """
    w, h = round(G.BASE_H * ratio), G.BASE_H
    by_edge, by_pixels = MAX_EDGE / max(w, h), math.sqrt(MAX_PIXELS / (w * h))
    s = min(by_edge, by_pixels)
    W, H = int(w * s), int(h * s)
    if by_edge <= by_pixels:
        W, H = (MAX_EDGE, H) if w >= h else (W, MAX_EDGE)
    return W, H


def build_scene(gid: int, ctype: str, fonts: list, seed: int, canvas: tuple[int, int],
                complexity: int, min_scale: float):
    """Rebuild one scene exactly as `gen_svg_localization.build` would.

    Every random draw is taken in the same order from the same seed, so a scene
    rebuilt here is identical to the one that produced the committed images --
    which is what lets `ExactInk` measure boxes for a dataset whose pixels it
    did not write. Returns None when the builder declines the scene.
    """
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    rng = random.Random(seed * 100003 + gid * 7919)
    theme = G.THEMES[rng.randrange(len(G.THEMES))]
    fpath, fidx, fam = fonts[rng.randrange(len(fonts))]
    dom, nouns = rng.choice(G.DOMAINS)
    sc = G.Scene(gid=gid, title=f"{dom} - {ctype.replace('_', ' ')} {gid:04d}",
                 ctype=ctype, theme=theme, font_file=fpath, font_index=fidx,
                 font_family=fam, w=canvas[0], h=canvas[1])
    sc.domain = (dom, nouns)
    sc.complexity = complexity
    F = G.load(fpath, fidx, G.BASE_FONT)
    G._title(sc, probe, F)
    if G.BUILDERS[ctype](sc, rng, probe, F) is None:
        return None
    G._decorate(sc, rng, probe)
    # Sizes are in base units, so the floor is set by the *smallest* rung: a
    # label legible at r100 and unreadable at r55 is a bad stimulus twice.
    G.enforce_legibility(sc, min_scale)
    return sc


# ----------------------------------------------------------------- exact ink

class ExactInk:
    """The pixel-exact painted extent of one label, by difference of two renders.

    Render the scene twice -- once with no text and once with a single label --
    and take the pixels that differ. That avoids the font's layout box, which
    clips the final glyph, and intensity thresholding, which absorbs connector
    lines running under the text. There is no threshold to tune and no
    stopping rule to justify: a pixel either changed when the label was drawn
    or it did not.

    Library only, no CLI. `ladder` uses it to measure targets on scenes it just
    built; `samples` uses it to re-measure targets in the committed set.
    """

    # The values data/svg_localization was generated with. `verify` below is what
    # re-renders those scenes to check the boxes against the shipped manifest, so
    # these have to stay in step with `blindspot.pipelines`'s generate step -- a
    # mismatched seed or complexity silently measures a different picture.
    DATASET = SOURCE_REL
    SEED, COUNT, COMPLEXITY = 17, 200, 4
    SCALES = {"small": 0.6, "medium": 1.0, "large": 2.0}

    SOURCE = "exact painted extent, measured by scene difference"
    METHOD = "render-with-label minus render-without-label"

    def __init__(self, dataset: str | Path = None, seed: int = None,
                 complexity: int = None, scales: dict | None = None):
        self.dataset = Path(dataset or ROOT / self.DATASET)
        self.seed = self.SEED if seed is None else seed
        self.complexity = self.COMPLEXITY if complexity is None else complexity
        self.scales = scales or self.SCALES
        self.min_scale = min(self.scales.values())
        self._fonts = None
        self._scenes: dict[int, object] = {}
        self._bases: dict[tuple[int, float], Image.Image] = {}

    # ------------------------------------------------------------- scenes

    @property
    def fonts(self):
        if self._fonts is None:
            self._fonts = G.available_fonts()
        return self._fonts

    def scene(self, gid: int):
        """The committed scene `gid`, rebuilt and cached."""
        if gid not in self._scenes:
            types = list(G.BUILDERS)
            self._scenes[gid] = build_scene(
                gid, types[gid % len(types)], self.fonts, self.seed,
                (G.BASE_W, G.BASE_H), self.complexity, self.min_scale)
        return self._scenes[gid]

    def forget(self) -> None:
        """Drop the cached renders. One scene's worth of images is a few MB and
        the ladder walks 96 of them; the scenes themselves are cheap to keep."""
        self._bases.clear()

    def base(self, sc, scale: float) -> Image.Image:
        """The scene with every label withheld -- the `without` half of the diff."""
        key = (sc.gid, scale)
        if key not in self._bases:
            self._bases[key] = G.render(sc, scale, skip_text=True)[0]
        return self._bases[key]

    # -------------------------------------------------------------- boxes

    def measure(self, sc, scale: float, target_idx: int,
                layout: list[float] | None = None) -> dict | None:
        """Exact box for one text prim, plus how far it escapes the layout box.

        `layout` overrides the recomputed layout box. `samples` passes the value
        recorded in the manifest, because that is the box a naive pipeline would
        have trained on -- and because Pillow's `textbbox` has shifted by up to a
        pixel between versions, so recomputing it here would describe a box the
        committed dataset does not contain. The painted pixels are unaffected.

        Returns None when the label paints nothing (fully occluded, or clipped
        off-canvas): a target with no ink has no box to supervise.
        """
        p = sc.texts[target_idx]
        base = self.base(sc, scale)
        im = base.copy()
        d = ImageDraw.Draw(im)
        # Drawn exactly as `render` draws it: same size rounding, same anchor.
        fs = max(5, round(p["size"] * scale))
        font = G.load(sc.font_file, sc.font_index, fs)
        xy = (p["x"] * scale, p["y"] * scale)
        d.text(xy, p["s"], font=font, fill=p["fill"], anchor=p["anchor"])
        lay = list(layout) if layout is not None else \
            [round(v, 2) for v in d.textbbox(xy, p["s"], font=font, anchor=p["anchor"])]

        changed = np.any(np.asarray(im) != np.asarray(base), axis=2)
        ys, xs = np.nonzero(changed)
        if len(xs) == 0:
            return None
        # Half-open bounds, so the box covers the painted pixels the way a
        # rectangle covers them: [x0, x1) x [y0, y1).
        exact = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        return {"box_px": exact, "layout_box_px": lay,
                "grew_px_lrtb": self.growth(lay, exact),
                "method": self.METHOD, "image_px": list(im.size)}

    @staticmethod
    def growth(layout: list[float], exact: list[float]) -> list[float]:
        """Per-side growth, positive = the ink escapes the layout box outward.

        Ordered like the box itself (x0, y0, x1, y1), so it reads straight
        against `layout_box_px` beside it.
        """
        return [round(layout[0] - exact[0], 1), round(layout[1] - exact[1], 1),
                round(exact[2] - layout[2], 1), round(exact[3] - layout[3], 1)]

    @staticmethod
    def overlap(a: list[float], b: list[float]) -> float:
        """IoU. The metric the layout box is bad at, so the metric to report it in."""
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------- verify

    def verify(self, n: int = 400, seed: int = 17, quiet: bool = False) -> dict:
        """Re-measure `n` committed targets and report how wrong the layout box is.

        Measured over the population that can actually become a training target
        -- point questions on the render the API delivers untouched -- because a
        box measured on a render the model never sees says nothing about the
        supervision. This reports median overlap ~0.91 on the committed set;
        part3.md quotes 0.899 with 18.5% below 0.75 from a sample whose
        definition did not survive, so treat the tail figure as unverified.

        It doubles as the regression test for this class: if the generator
        drifts, the rebuilt scenes stop matching the shipped pixels and this
        number moves.
        """
        rows = [json.loads(x) for x in
                (self.dataset / "manifest.jsonl").read_text().splitlines() if x.strip()]
        pool = [r for r in rows if r["qtype"] == "point" and not r["downscaled_by_api"]]
        random.Random(seed).shuffle(pool)
        ious = []
        for r in pool[:n]:
            sc = self.scene(r["graph_id"])
            if sc is None:
                continue
            m = self.measure(sc, r["scale"], r["target_idx"], r["text_ink_bbox_px"])
            if m:
                ious.append(self.overlap(m["layout_box_px"], m["box_px"]))
        ious.sort()
        out = {"n": len(ious),
               "median_overlap": round(ious[len(ious) // 2], 4) if ious else None,
               "frac_below_0.75": round(sum(i < 0.75 for i in ious) / len(ious), 4)
               if ious else None}
        if not quiet:
            print(f"layout box vs painted extent over {out['n']} targets: "
                  f"median overlap {out['median_overlap']}, "
                  f"{out['frac_below_0.75']:.1%} below 0.75")
        return out


# ---------------------------------------------------------------- 1. ladder

def cmd_ladder(a) -> int:
    """Six aspect ratios x four sizes, every image delivered untouched."""
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    fonts = G.available_fonts(verbose=True)
    types = list(G.BUILDERS)
    ink = ExactInk(scales=RUNGS)
    min_scale = min(RUNGS.values())

    rows, scenes, images, downscaled = [], 0, 0, 0
    delivered: dict[str, set] = {name: set() for name, _ in ASPECTS}
    for ai, (aspect, ratio) in enumerate(ASPECTS):
        canvas = canvas_for(ratio)
        for i in range(a.scenes_per_aspect):
            gid = ai * a.scenes_per_aspect + i
            # Chart type cycles within the aspect, not across the whole run, so
            # every aspect gets the same spread of chart types.
            ctype = types[i % len(types)]
            sc = build_scene(gid, ctype, fonts, SEED, canvas, COMPLEXITY, min_scale)
            if sc is None:
                print(f"  ! skipped g{gid:04d} ({ctype}): builder declined")
                continue
            scenes += 1
            for rung, k in RUNGS.items():
                im, gold = G.render(sc, k)
                W, H = im.size
                ids, _rej = G.eligible_targets(gold, None, G.MIN_LEGIBLE_PX)
                if not ids:
                    print(f"  ! g{gid:04d} {aspect}/{rung}: no eligible target")
                    continue
                rel = f"images/g{gid:04d}_{aspect}_{rung}.png"
                im.save(out / rel, "PNG", optimize=True)
                images += 1
                delivered[aspect].add((W, H))
                ew, eh = G.effective_size(W, H)
                downscaled += [W, H] != [ew, eh]

                base = ink.base(sc, k)          # measured once, reused per target
                qrng = random.Random(SEED * 31 + gid * 4 + list(RUNGS).index(rung))
                qs = [q for q in G.make_questions(sc, gold, ids, W, H, qrng, 4)
                      if q["qtype"] == "point"]
                for qi, q in enumerate(qs):
                    ti = q["target_idx"]
                    m = ink.measure(sc, k, ti)
                    # A label that paints nothing, or paints a sliver, cannot be
                    # a regression target. Counted by the shortfall in the
                    # printed total rather than dropped silently.
                    if m is None or m["box_px"][2] - m["box_px"][0] < 2 \
                            or m["box_px"][3] - m["box_px"][1] < 2:
                        continue
                    box = m["box_px"]
                    g = gold[ti]
                    rows.append({
                        "uid": f"mr:{gid:04d}:{aspect}:{rung}:{qi:02d}",
                        "graph_id": gid, "chart_type": ctype, "theme": sc.theme.name,
                        "font_family": sc.font_family,
                        "aspect": aspect, "aspect_ratio": round(ratio, 4),
                        "rung": rung, "scale": k, "canvas_px": [sc.w, sc.h],
                        "image": rel, "image_px": [W, H], "effective_px": [ew, eh],
                        "downscaled_by_api": [W, H] != [ew, eh],
                        "target_idx": ti, "target_text": q["target_text"],
                        "target_role": q["target_role"], "question": q["question"],
                        "box_px": box,
                        "box_norm": [round(box[0] / W, 6), round(box[1] / H, 6),
                                     round(box[2] / W, 6), round(box[3] / H, 6)],
                        "box_area_frac": round((box[2] - box[0]) * (box[3] - box[1])
                                               / (W * H), 8),
                        "hit_box_px": g["bbox"], "hit_source": g["hit_source"],
                        # The hit box is the whole enclosing node, which answers
                        # "where is the widget", not "where is the text".
                        "hit_box_answers_the_question": g["hit_source"] != "shape",
                        "font_px": g["font_px"], "contrast": g["contrast"],
                        "bg_rgb": g["bg_rgb"],
                    })
            ink.forget()                        # one scene's renders, then gone

    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} targets -> {out / 'manifest.jsonl'}")
    print(f"  scenes: {scenes}   images: {images}")
    print(f"  downscaled by the API (must be 0): {downscaled}")
    print()
    print("  delivered sizes")
    for aspect, _ in ASPECTS:
        n = sum(1 for r in rows if r["aspect"] == aspect)
        sizes = "  ".join(f"{w}x{h}" for w, h in sorted(delivered[aspect]))
        print(f"    {aspect:<10s} n={n:4d}  {sizes}")
    print()
    if rows:
        areas = [r["box_area_frac"] for r in rows]
        held = sum(1 for r in rows if r["hit_source"] == "shape")
        print(f"  target area: {min(areas) * 100:.4f}% - {max(areas) * 100:.4f}% of image")
        print(f"  targets held inside a shape: {held}/{len(rows)}")
        print(f"  chart types: {len({r['chart_type'] for r in rows})}   "
              f"themes: {len({r['theme'] for r in rows})}")
    return 0


# --------------------------------------------------------------- 2. samples

# Smallest band first. Weights invert the measured accuracy-by-size gradient:
# the model scores 1.3% on the smallest size bin and 12.3% on the largest, so
# the smallest band earns the most supervision and the largest the least.
WEIGHTS = [6, 5, 4, 3, 2]
N_BANDS = len(WEIGHTS)

PROMPT = ('Return the bounding box of {q}.\n'
          'Answer with JSON only: {{"box": [x0, y0, x1, y1]}}\n'
          'Coordinates are fractions of the image between 0 and 1, with [0,0] at '
          'the top-left corner and [1,1] at the bottom-right. x0 < x1 and y0 < y1.')

ACCEPT_SOURCE = "gold_bbox -- Part 2 click-in-bbox region"


def area_bands(rows: list[dict], n_bands: int = N_BANDS) -> list[list[dict]]:
    """Cut into equal-count bands by target area, smallest first."""
    ordered = sorted(rows, key=lambda r: r["target_area_frac"])
    n = len(ordered)
    return [ordered[i * n // n_bands:(i + 1) * n // n_bands] for i in range(n_bands)]


def allocate(n: int, weights: list[int] = WEIGHTS) -> list[int]:
    """Split `n` records across the bands, every band represented.

    Weights are 6/5/4/3/2 smallest-first, so `n=20` gives exactly that. Below
    one-per-weight the weighting is meaningless and the bands are simply filled
    from the smallest end, which is where the model is worst anyway.
    """
    k = len(weights)
    if n <= k:
        return [1] * n + [0] * (k - n)
    alloc = [1] * k
    rest = n - k
    share = [rest * w / sum(weights) for w in weights]
    alloc = [a + int(s) for a, s in zip(alloc, share)]
    # Largest-remainder, so the total is exactly n rather than n-ish.
    order = sorted(range(k), key=lambda i: (-(share[i] - int(share[i])), i))
    for i in order[:n - sum(alloc)]:
        alloc[i] += 1
    return alloc


def stable_hash(seed: int, uid: str) -> int:
    """Deterministic across runs, machines and Python versions.

    `hash()` is salted per process, so a selection built on it is not
    reproducible from a seed -- which is the one thing --seed promises.
    """
    return int(hashlib.sha1(f"{seed}:{uid}".encode()).hexdigest(), 16)


def pick_spread(band: list[dict], want: int, seen: Counter, seed: int) -> list[dict]:
    """Take `want` records from `band`, maximising spread over the axes that matter.

    Greedy on the least-represented (chart type, theme, target role) so far --
    `seen` is carried across bands, so twenty records do not turn out to be
    twenty tables. Ties break on a stable hash of (seed, uid), which makes the
    selection reproducible without making it alphabetical: sorting by uid alone
    would bias every sample toward low graph ids.
    """
    pool = sorted(band, key=lambda r: (stable_hash(seed, r["uid"]), r["uid"]))
    out = []
    for _ in range(min(want, len(pool))):
        best = min(pool, key=lambda r: (seen[r["chart_type"]] + seen[r["theme"]]
                                        + seen[r["target_role"]]))
        pool.remove(best)
        for axis in ("chart_type", "theme", "target_role"):
            seen[best[axis]] += 1
        out.append(best)
    return out


def to_record(r: dict, band: int, band_label: str, exact: dict, polarity: str) -> dict:
    """One training record: image, question, and the box to return."""
    W, H = r["image_px"]
    box = exact["box_px"]
    norm = [round(box[0] / W, 6), round(box[1] / H, 6),
            round(box[2] / W, 6), round(box[3] / H, 6)]
    # Measured on the normalised box the model is asked for, not on the pixel
    # box, so `area_frac` describes the answer being supervised. The two differ
    # in the eighth decimal, which matters only because the curriculum bands and
    # the ladder manifest must stay comparable record for record.
    area = round((norm[2] - norm[0]) * (norm[3] - norm[1]), 8)
    return {
        "uid": r["uid"],
        "task": "localize_bbox",
        "image": {"path": f"{SOURCE_REL}/{r['image']}", "px": [W, H],
                  "delivered_untouched": not r["downscaled_by_api"]},
        "prompt": PROMPT.format(q=r["question"]),
        "completion": json.dumps({"box": norm}),
        "target": {
            "box_norm": norm, "box_px": box, "source": ExactInk.SOURCE,
            "area_frac": area,
            "correction": {"layout_box_px": exact["layout_box_px"],
                           "grew_px_lrtb": exact["grew_px_lrtb"],
                           "method": exact["method"]},
        },
        # Kept beside the target, never as the target: for shape-held labels this
        # is the whole node, which answers a question about the widget and not
        # about the text. Flagged rather than dropped so a reader can see the
        # difference the flag is making.
        "accept_region": {
            "box_norm": r["gold_bbox_norm"], "source": ACCEPT_SOURCE,
            "area_frac": r["target_area_frac"], "hit_source": r["hit_source"],
            "answers_the_question": r["hit_source"] != "shape",
            "times_larger_than_target": round(r["target_area_frac"] / area, 1)
            if area else None,
        },
        "curriculum": {"area_band": band, "area_band_label": band_label,
                       "font_px": r["font_px"], "contrast": r["target_contrast"],
                       "polarity": polarity},
        "meta": {"chart_type": r["chart_type"], "theme": r["theme"],
                 "target_text": r["target_text"], "target_role": r["target_role"],
                 "question": r["question"], "graph_id": r["graph_id"]},
    }


def cmd_samples(a) -> int:
    """SFT records from the committed set, curriculum-weighted toward small targets."""
    src = ROOT / SOURCE_REL
    rows = [json.loads(x) for x in
            (src / "manifest.jsonl").read_text().splitlines() if x.strip()]
    # Point questions only -- a box target needs a box question -- and only on
    # renders the API leaves alone, so the pixels supervised are the pixels the
    # model is shown.
    cands = [r for r in rows if r["qtype"] == "point" and not r["downscaled_by_api"]]
    print(f"candidates: {len(cands)} point questions on the untouched render")

    dark = {t.name: t.dark for t in G.THEMES}
    ink = ExactInk(dataset=src)
    bands = area_bands(cands)
    alloc = allocate(a.n)
    seen: Counter = Counter()
    out_rows = []
    for i, (band, want) in enumerate(zip(bands, alloc)):
        lo = min(r["target_area_frac"] for r in band) * 100
        hi = max(r["target_area_frac"] for r in band) * 100
        label = f"{lo:.3f}-{hi:.3f}% of image"
        for r in pick_spread(band, want, seen, a.seed):
            sc = ink.scene(r["graph_id"])
            if sc is None:
                continue
            # The recorded layout box is passed in deliberately: the correction
            # has to describe the box this dataset ships, not one recomputed
            # against whatever Pillow is installed today.
            exact = ink.measure(sc, r["scale"], r["target_idx"], r["text_ink_bbox_px"])
            if exact is None:
                print(f"  ! {r['uid']}: label paints nothing; skipped")
                continue
            out_rows.append(to_record(r, i + 1, label, exact,
                                      "dark" if dark[r["theme"]] else "light"))

    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(r) + "\n" for r in out_rows))
    print(f"wrote {len(out_rows)} records -> {dest}")
    print(f"  band allocation (smallest first): {alloc}")
    print(f"  chart types: {len({r['meta']['chart_type'] for r in out_rows})}")
    print(f"  themes:      {len({r['meta']['theme'] for r in out_rows})}")
    print(f"  polarity:    {dict(Counter(r['curriculum']['polarity'] for r in out_rows))}")
    if out_rows:
        areas = [r["target"]["area_frac"] for r in out_rows]
        shape = sum(1 for r in out_rows if r["accept_region"]["hit_source"] == "shape")
        print(f"  target area: {min(areas) * 100:.4f}% - {max(areas) * 100:.4f}% of image")
        print(f"  accept region is the enclosing shape, not the text: "
              f"{shape}/{len(out_rows)}")
    return 0


# ----------------------------------------------------------------- 3. audit

# Ink is any pixel that is not exactly the background measured under this label.
# A tolerance is tempting -- `gen_svg_localization.occlusion` uses 12 -- but it
# is wrong here: at a 10px font the outermost row of a glyph sits one or two
# units off the background, so `tight` starts reporting correct boxes as loose
# (measured on the shipped set: 21 of 400 fail at a tolerance of 2, 70 at 12).
# Zero is also the honest threshold, because the box came from a difference of
# two renders and that is the same test.
INK_DELTA = 0


def _ink_mask(im: Image.Image, bg) -> np.ndarray:
    a = np.asarray(im.convert("RGB"), dtype=np.int16)
    return np.abs(a - np.asarray(bg, dtype=np.int16)).max(axis=2) > INK_DELTA


def _checks(r: dict, im: Image.Image) -> dict[str, bool]:
    """Five ways a box can be wrong, each checked against the actual pixels."""
    W, H = r["image_px"]
    x0, y0, x1, y1 = r["box_px"]
    px = im.size
    crop = im.crop((int(x0), int(y0), int(x1), int(y1)))
    mask = _ink_mask(crop, r["bg_rgb"]) if crop.width and crop.height else np.zeros((0, 0), bool)
    return {
        # A box outside the frame cannot be normalised into [0,1].
        "in_bounds": 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H and list(px) == [W, H],
        # One pixel wide is not a box; IoU against it is a coin flip.
        "non_degenerate": (x1 - x0) >= 2 and (y1 - y0) >= 2,
        # The whole point of the ladder: what is rendered is what is delivered.
        "undownscaled": list(G.effective_size(*px)) == list(px)
        and not r["downscaled_by_api"],
        # Something is actually painted where the box says the text is.
        "has_ink": bool(mask.any()),
        # ... and it reaches all four edges, so the box is tight rather than
        # merely containing. A box two pixels too big passes `has_ink` and would
        # still teach the model to overshoot.
        "tight": bool(mask.size and mask[0].any() and mask[-1].any()
                      and mask[:, 0].any() and mask[:, -1].any()),
    }


def _gallery(records: list[dict], dataset: Path, dest: Path) -> None:
    """Full frame plus a zoom, because a 0.02% target is invisible at page scale."""
    cards = []
    for i, (r, im) in enumerate(records, 1):
        x0, y0, x1, y1 = r["box_px"]
        figs = []
        for kind in ("full", "zoom"):
            shot = im.copy()
            d = ImageDraw.Draw(shot)
            # Outline drawn strictly OUTSIDE the box: PIL renders a multi-pixel
            # rectangle inward, painting over the glyph rows at the box edge and
            # making a correct box look like it clips the text.
            d.rectangle([x0 - 1, y0 - 1, x1, y1], outline="#e0245e", width=1)
            if kind == "zoom":
                pad = max(24, int(max(x1 - x0, y1 - y0)))
                shot = shot.crop((max(0, int(x0) - pad), max(0, int(y0) - pad),
                                  min(shot.width, int(x1) + pad),
                                  min(shot.height, int(y1) + pad)))
                shot = shot.resize((shot.width * 3, shot.height * 3), Image.NEAREST)
            buf = io.BytesIO()
            shot.save(buf, "PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode()
            cap = (f"full image &mdash; {im.width}&times;{im.height}, delivered untouched"
                   if kind == "full" else "zoom on the target")
            figs.append(f"<figure><img src='data:image/png;base64,{b64}' alt='{kind}'>"
                        f"<figcaption>{cap}</figcaption></figure>")
        tags = "".join(f"<span class='tag'>{t}</span>" for t in (
            r["chart_type"], r["theme"], r["aspect"], r["rung"],
            f"{r['box_area_frac'] * 100:.3f}% of image", f"{r['font_px']}px font"))
        cards.append(
            f"<div class='card'><div class='hd'><span class='q'>{i}. {r['question']}"
            f"</span><span class='uid'>{r['uid']}</span></div>"
            f"<div class='tags'>{tags}</div><div class='imgs'>{''.join(figs)}</div>"
            f"<details><summary>ground-truth record</summary>"
            f"<pre>{json.dumps(r, indent=2)}</pre></details></div>")
    css = ("body{background:#fcfcfb;color:#1b1b19;font:14px/1.5 system-ui,sans-serif;"
           "margin:0;padding:24px}.wrap{max-width:1100px;margin:0 auto}"
           ".card{background:#fff;border:1px solid #e4e4e0;border-radius:8px;"
           "padding:14px;margin:18px 0}.hd{display:flex;justify-content:space-between;"
           "gap:12px}.q{font-weight:600}.uid{color:#6a6a66;font-family:ui-monospace,monospace}"
           ".tags{margin:8px 0}.tag{background:#f4f4f1;border-radius:99px;padding:2px 9px;"
           "margin-right:6px;font-size:12px;color:#6a6a66}"
           ".imgs{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}"
           "img{max-width:100%;border:1px solid #e4e4e0}"
           "figcaption{color:#6a6a66;font-size:12px;margin-top:4px}"
           "pre{background:#f4f4f1;padding:10px;overflow:auto;font-size:12px}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(f"<!doctype html><meta charset='utf-8'>"
                    f"<title>Multi-resolution ground truth</title><style>{css}</style>"
                    f"<div class='wrap'><h1>Multi-resolution ground truth</h1>"
                    f"<p class='sub'>{len(records)} records from {dataset}</p>"
                    f"{''.join(cards)}</div>")


def cmd_audit(a) -> int:
    """Check every box against the pixels, then show a sample of them."""
    ds = Path(a.dataset)
    rows = [json.loads(x) for x in
            (ds / "manifest.jsonl").read_text().splitlines() if x.strip()]
    print(f"{len(rows)} records in {ds}")
    print()

    names = ["in_bounds", "non_degenerate", "undownscaled", "has_ink", "tight"]
    fails = {k: 0 for k in names}
    bad: list[tuple[dict, str]] = []
    cache: dict[str, Image.Image] = {}
    for r in rows:
        if r["image"] not in cache:
            cache.clear()               # rows are grouped by image; one is enough
            cache[r["image"]] = Image.open(ds / r["image"]).convert("RGB")
        res = _checks(r, cache[r["image"]])
        for k, ok in res.items():
            if not ok:
                fails[k] += 1
                bad.append((r, k))

    print("checks (count failing / total)")
    for k in names:
        print(f"  {k:<20s} {fails[k]}/{len(rows)}   {'ok' if not fails[k] else 'FAIL'}")
    for r, k in bad[:10]:
        print(f"    ! {r['uid']} failed {k}")
    print()

    if a.gallery_n:
        picks = rows[:]
        random.Random(a.seed).shuffle(picks)
        picks = picks[:a.gallery_n]
        loaded = [(r, Image.open(ds / r["image"]).convert("RGB")) for r in picks]
        # The original hardcoded finetune/samples/gallery_multires.html and
        # ignored --dataset, so auditing a throwaway set overwrote the shipped
        # gallery. The path now follows the dataset being audited.
        dest = Path(a.out) if a.out else ROOT / "outputs" / "finetune" / f"gallery_{ds.name}.html"
        _gallery(loaded, ds, dest)
        kb = dest.stat().st_size // 1024
        print(f"gallery: {len(loaded)} records -> {dest} ({kb} KB)")
    return 1 if any(fails.values()) else 0


# -------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ladder", help="generate the aspect/resolution ladder")
    p.add_argument("--out", default=MR_REL)
    p.add_argument("--scenes-per-aspect", type=int, default=16)
    p.set_defaults(fn=cmd_ladder)

    p = sub.add_parser("samples", help="build SFT records with pixel-exact boxes")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    # No default: the committed sft_bbox_20.jsonl is a reference artefact and a
    # defaulted --out would overwrite it on a bare run.
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_samples)

    p = sub.add_parser("audit", help="check a ladder dataset against its pixels")
    p.add_argument("--dataset", default=MR_REL)
    p.add_argument("--gallery-n", type=int, default=24)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out", default=None,
                   help="gallery destination; defaults to outputs/finetune/gallery_<dataset>.html")
    p.set_defaults(fn=cmd_audit)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
