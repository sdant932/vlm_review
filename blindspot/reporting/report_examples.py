"""Real-image example figures for the report.

A report about perception that argues from bar charts is not credible. These
figures show the actual pictures the model was given, at the resolution it
actually received them, with its actual answer beside them — so a reader can
judge the failure rather than take a percentage on trust.

Every panel renders the image through `effective_size()` first, so nothing here
displays detail the model never had. Overlays are drawn from the manifest, not
from the code that produced the manifest, so a wrong gold box would be visible as
a box that misses its target rather than cancelling itself out.

    python -m blindspot.reporting.report_examples
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from blindspot.core.adapters import load

RUN = "haiku-4-5_think2000_native_r0"

OUT = Path("outputs/report/figures")
W = 2600                                   # 2x of the 1300-unit diagram width
SURFACE = (252, 252, 251)
INK, INK2, MUTED = (11, 11, 11), (82, 81, 78), (138, 137, 131)
GOOD, CRIT, S1, S2 = (12, 163, 12), (208, 59, 59), (42, 120, 214), (235, 104, 52)
RULE = (228, 227, 223)

_FONTS = ["/System/Library/Fonts/Helvetica.ttc",
          "/System/Library/Fonts/Supplemental/Arial.ttf",
          "/Library/Fonts/Arial.ttf"]


def font(size: int, bold: bool = False):
    for p in _FONTS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size, index=1 if bold else 0)
            except Exception:
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    continue
    return ImageFont.load_default()


def effective(w: int, h: int) -> tuple[int, int]:
    """What the API actually delivers: 1568px long edge, ~1.15 MP total."""
    s = min(1.0, 1568 / max(w, h), math.sqrt(1_150_000 / max(w * h, 1)))
    return max(1, round(w * s)), max(1, round(h * s))


def as_model_saw(path: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im.resize(effective(*im.size), Image.LANCZOS)


def fit(im: Image.Image, bw: int, bh: int) -> Image.Image:
    r = min(bw / im.width, bh / im.height)
    return im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)


def draw_target(im: Image.Image, box=None, point=None, ring=True) -> Image.Image:
    """Green box (+ locator ring) for the target, red crosshair for the click."""
    im = im.copy()
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(im.size) / 400))
    if box:
        x0, y0, x1, y1 = [v * s for v, s in zip(box, (im.width, im.height) * 2)]
        d.rectangle([x0, y0, x1, y1], outline=GOOD, width=lw)
        if ring:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            r = max(lw * 13, (x1 - x0) * 1.5, (y1 - y0) * 1.5)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=GOOD, width=max(1, lw // 2))
    if point:
        px, py = point[0] * im.width, point[1] * im.height
        r = lw * 8
        d.line([px - r, py, px + r, py], fill=CRIT, width=lw)
        d.line([px, py - r, px, py + r], fill=CRIT, width=lw)
        d.ellipse([px - r / 2, py - r / 2, px + r / 2, py + r / 2], outline=CRIT, width=lw)
    return im


class Panel:
    """A figure canvas with a title, then image+text cards stacked or in a row."""

    def __init__(self, title: str, sub: str = "", h: int = 1200):
        self.im = Image.new("RGB", (W, h), SURFACE)
        self.d = ImageDraw.Draw(self.im)
        self.d.text((80, 60), title, INK, font=font(46, True))
        if sub:
            self.d.text((80, 126), sub, INK2, font=font(28))
        self.y = 190 if sub else 150

    def text(self, x, y, s, size=26, fill=INK2, bold=False, wrap=None):
        f = font(size, bold)
        lines = textwrap.wrap(s, wrap) if wrap else [s]
        for i, ln in enumerate(lines):
            self.d.text((x, y + i * int(size * 1.35)), ln, fill, font=f)
        return y + len(lines) * int(size * 1.35)

    def rule(self, y):
        self.d.line([80, y, W - 80, y], fill=RULE, width=2)

    def paste(self, im, x, y, bw, bh, border=True):
        im = fit(im, bw, bh)
        self.im.paste(im, (int(x), int(y)))
        if border:
            self.d.rectangle([x, y, x + im.width, y + im.height], outline=RULE, width=2)
        return im.width, im.height

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        self.im.crop((0, 0, W, min(self.y + 60, self.im.height))).save(
            OUT / f"{name}@2x.png", quality=92)
        return name


# ------------------------------------------------------------------ selection
def _preds(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("pred") is not None:
                out[r["uid"]] = r["pred"]
    return out


def e05_instrument():
    """The variety the generator covers: eight scenes, one per chart type."""
    import json as _j
    base = Path("data/svg_localization/images")
    scenes = [_j.loads(l) for l in open("data/svg_localization/scenes.jsonl") if l.strip()]
    want, seen = [], set()
    for sc in scenes:
        if sc["chart_type"] in seen:
            continue
        if not (base / f'g{sc["graph_id"]:04d}_small.png').exists():
            continue
        seen.add(sc["chart_type"])
        want.append(sc)
        if len(want) == 8:
            break

    p = Panel("Generated scenes: eight of the sixteen chart types",
              "200 scenes in total, each with its own chart type, colour theme, font and "
              "domain vocabulary.", h=1500)
    y = p.y
    colw, roww = 620, 470
    for i, sc in enumerate(want):
        x = 80 + (i % 4) * colw
        yy = y + (i // 4) * roww
        f = base / f'g{sc["graph_id"]:04d}_small.png'
        _, ih = p.paste(as_model_saw(str(f)), x, yy, colw - 40, 350)
        t2 = p.text(x, yy + ih + 14, sc["chart_type"].replace("_", " "), 25, INK, True)
        p.text(x, t2 + 2, f'{sc["theme"]}  ·  {sc["domain"]}', 22, MUTED)
    p.y = y + 2 * roww - 40
    p.d.line([80, p.y, W - 80, p.y], fill=RULE, width=2)
    p.text(80, p.y + 22, "Every scene is rendered at two resolutions that reach the model "
                         "differently:", 27, INK, True)
    p.text(80, p.y + 60, "small  900x570, delivered untouched, label text 10-14px", 26, INK2)
    p.text(80, p.y + 94, "large  3000x1900, downscaled by the API to 1348x853", 26, INK2)
    p.text(1180, p.y + 60, "The target occupies the same fraction of the image at both, so "
                           "the contrast isolates absolute resolution rather than target "
                           "size.", 26, INK2, wrap=58)
    p.y += 140
    return p.save("e05_instrument")


def e08_generation_pipeline():
    """Block diagram of the procedural generation, from seed to scored questions."""
    import json as _j, collections
    rows = [_j.loads(l) for l in open("data/svg_localization/manifest.jsonl") if l.strip()]
    rej, seen = collections.Counter(), set()
    for r in rows:
        k = (r["graph_id"], r["resolution"])
        if k in seen:
            continue
        seen.add(k)
        for cat, n in (r.get("rejected_targets") or {}).items():
            rej[cat] += n

    p = Panel("How a scene and its ground truth are generated",
              "Every draw is seeded, so the whole corpus is reproducible from one integer.",
              h=1650)

    def block(x, y, w, h, head, body, fill=(255, 255, 255), accent=S1, hs=27, bs=23):
        p.d.rounded_rectangle([x, y, x + w, y + h], 10, fill=fill, outline=RULE, width=2)
        p.d.rectangle([x, y, x + 8, y + h], fill=accent)
        p.text(x + 26, y + 16, head, hs, INK, True)
        if body:
            p.text(x + 26, y + 16 + int(hs * 1.5), body, bs, INK2, wrap=int(w / (bs * 0.52)))

    def arrow(x1, y1, x2, y2):
        p.d.line([x1, y1, x2, y2], fill=MUTED, width=3)
        if y2 > y1:
            p.d.polygon([(x2 - 8, y2 - 12), (x2 + 8, y2 - 12), (x2, y2)], fill=MUTED)
        else:
            p.d.polygon([(x2 - 12, y2 - 8), (x2 - 12, y2 + 8), (x2, y2)], fill=MUTED)

    y = p.y
    block(80, y, 420, 78, "seed = 17", "one integer fixes every draw below", accent=(11, 11, 11))

    # the four independent random draws, shown as parallel blocks
    y2 = y + 132
    draws = [("chart type", "16 options", "flowchart, bar, line, scatter, table, network, "
              "pie, org chart, timeline, gantt, mindmap, dashboard, quadrant, sequence, "
              "treemap, state machine"),
             ("colour theme", "10 options", "paper, cream, ice, mint, sun, mono-print, "
              "high-contrast, and 3 dark: slate-dark, carbon, blueprint"),
             ("font family", "9 options", "Arial, Verdana, Tahoma, Trebuchet, Georgia, "
              "Palatino, Futura, Menlo, Courier"),
             ("domain vocabulary", "10 options", "Triage Protocol, Payment Authorization, "
              "Access Review, Fleet Logistics, Returns Workflow, Build Pipeline, Claims "
              "Handling, Grid Operations, Telemetry Ingest, Content Moderation")]
    bw = 590
    for i, (name, n, opts) in enumerate(draws):
        x = 80 + (i % 2) * (bw + 40)
        yy = y2 + (i // 2) * 190
        block(x, yy, bw, 172, f"{name}  —  {n}", opts, accent=S2, hs=26, bs=22)
        arrow(290, y + 78, 290, y2 - 6) if i == 0 else None

    y3 = y2 + 400
    steps = [("lay out primitives",
              "rects, circles, wedges, lines and text into one shared list; complexity 4 "
              "sets how many nodes, bars, rows or slices"),
             ("enforce legibility",
              "grow any targetable text that would fall below 10px at the smaller size"),
             ("render at each scale, twice",
              "once with text suppressed, so the true background behind every label can be "
              "measured rather than assumed"),
             ("measure gold off the raster",
              "ink box from the rendered glyphs; hit box is the enclosing widget, or the ink "
              "grown by button padding"),
             ("filter eligible targets",
              "nine rules: unique, non-overlapping, unoccluded, above AA contrast, at least "
              "10px, hit box must not swallow a neighbour"),
             ("emit questions",
              "point, relation and reverse, drawn only from surviving targets")]
    for i, (head, body) in enumerate(steps):
        yy = y3 + i * 142
        block(80, yy, 1180, 124, f"{i+1}.  {head}", body, accent=S1, hs=27, bs=22)
        if i:
            arrow(670, yy - 22, 670, yy - 2)
    arrow(670, y2 + 362, 670, y3 - 4)

    # what the filter threw away
    x2 = 1340
    p.text(x2, y3 - 40, "WHAT THE TARGET FILTER REJECTED", 22, MUTED, True)
    yy = y3
    top = rej.most_common(7)
    mx = max((n for _, n in top), default=1)
    for cat, n in top:
        w = 560 * n / mx
        p.d.rounded_rectangle([x2, yy, x2 + max(w, 3), yy + 26], 4, fill=S2)
        p.text(x2 + max(w, 3) + 14, yy + 1, f"{n:,}", 24, INK, True)
        p.text(x2, yy + 32, cat.replace("_", " "), 22, INK2)
        yy += 78
    p.text(x2, yy + 8, f"{sum(rej.values()):,} candidate labels rejected across 600 "
                       f"scenes and resolutions. What survives is the only thing a question is ever "
                       f"asked about.", 24, INK2, wrap=54)
    p.y = max(y3 + len(steps) * 142, yy + 110)
    return p.save("e08_generation_pipeline")




# --------------------------------------------------------------- the two figures
def e09_bad_gold():
    """One item where the benchmark, not the model, is wrong.

    The infographic never states a percentage: the model returned exactly what
    the page says and was scored zero, because the gold answer applies a unit
    conversion the image does not contain.
    """
    uid = "infographicvqa:94919"
    e = {x.uid: x for x in load("infographicvqa")}[uid]
    pred = _preds(f"results/infographicvqa__{RUN}.jsonl")[uid]
    p = Panel("A scored failure where the gold answer is the problem",
              "One of the audited items. The model's answer and the gold answer say "
              "the same thing; only the surface form differs.", h=1500)
    y = p.y + 10
    iw, ih = p.paste(as_model_saw(e.images[0]), 80, y, 900, 1050)
    x = 80 + iw + 70
    yy = p.text(x, y + 10, "question", 24, MUTED, True)
    yy = p.text(x, yy + 6, e.question, 32, INK, True, wrap=44)
    yy = p.text(x, yy + 40, "gold answer", 24, MUTED, True)
    yy = p.text(x, yy + 6, ", ".join(str(g) for g in e.gold), 34, INK, True)
    yy = p.text(x, yy + 34, "model answer", 24, MUTED, True)
    yy = p.text(x, yy + 6, str(pred), 34, CRIT, True)
    yy = p.text(x, yy + 34, "scored", 24, MUTED, True)
    yy = p.text(x, yy + 6, "0.0 ANLS - counted as a perception failure", 30, CRIT, True)
    yy = p.text(x, yy + 44,
                "An audit would not defend 16.8% of InfographicVQA's scored failures, "
                "or 16.3% of CharXiv's. Contested items occur only among errors, so "
                "CharXiv's 14.7% error rate is nearer 12.3% model error and 2.4% label "
                "error.", 27, INK2, wrap=52)
    p.y = max(y + ih, yy) + 40
    return p.save("e09_bad_gold")


# (heading, uid, explanation, zoom) chosen from outputs/report/candidates.html.
# `zoom` adds a 1:1 crop of the delivered image beside the whole scene, for the two
# rows whose argument is about detail the shrunk-to-fit view cannot show.
PICKS = [
    ("Resolution bias", "infographicvqa:80736",
     "The source is 42.7 megapixels; the API delivers 1.15. What the question asks "
     "about is destroyed before the model looks at it.", "busiest"),
    ("Localization", "svgloc:0034:small:03",
     "Given the exact target string and asked to point at it, the model returns a "
     "coordinate in the wrong region of the frame.", None),
    ("Label - object matching", "ai2d:01458",
     "The question names a printed mark on the diagram. Resolving which object that "
     "mark sits on is where the answer goes wrong.", None),
    ("General OCR reasoning", "charxiv:00052:r",
     "The arithmetic is right and the operation is right. The value it was applied "
     "to was misread off the chart.", None),
    ("Hallucination", "charxiv:00772:d4",
     "The plot has no legend at all. Asked how many entries it has, the model "
     "answers with a number rather than abstaining.", None),
    ("Counting", "charxiv:00655:d2",
     "The legend is printed at full size and is plainly legible. The model still "
     "counts 7 of its 11 entries.", (0.0, 0.0, 1.0, 0.24)),
]


def _pick_panel(uid: str):
    """(image, question, gold, model) for one chosen example, whatever it came from."""
    ds = uid.split(":")[0]
    if ds == "svgloc":
        from blindspot.analysis.svgloc_eval import load_run
        ex = {e.uid: e for e in load("svg_localization")}
        r = next(x for x in load_run(RUN)["point"] if x["uid"] == uid)
        im = draw_target(as_model_saw(ex[uid].images[0]),
                         box=r["gold"], point=tuple(r["pred"]))
        return (im, f'point at {r["question"]}', "inside the green box",
                f'({r["pred"][0]:.2f}, {r["pred"][1]:.2f}) - '
                f'{r["d_centre"]*100:.0f}% of the frame away')
    name = {"infographicvqa": "infographicvqa", "ai2d": "ai2d", "charxiv": "charxiv"}[ds]
    e = {x.uid: x for x in load(name)}[uid]
    pred = _preds(f"results/{name}__{RUN}.jsonl")[uid]
    q = e.question.split("*")[0].strip()
    gold = e.gold[0] if isinstance(e.gold, list) else e.gold
    if ds == "ai2d":                            # letters mean nothing without the text
        opts = e.meta.get("options") or []
        letter = lambda v: opts["ABCD".find(str(v).strip()[:1].upper())] \
            if 0 <= "ABCD".find(str(v).strip()[:1].upper()) < len(opts) else str(v)
        gold, pred = f'{gold} = "{letter(gold)}"', f'{pred} = "{letter(pred)}"'
    return as_model_saw(e.images[0]), q, str(gold), str(pred)


def _zoom_of(im, spec):
    """A 1:1 window on the delivered image: either a fixed box or the busiest one."""
    if spec is None:
        return None
    if isinstance(spec, tuple):
        x0, y0, x1, y1 = spec
        return im.crop((int(x0 * im.width), int(y0 * im.height),
                        int(x1 * im.width), int(y1 * im.height)))
    from blindspot.reporting.report_candidates import busiest_crop
    return busiest_crop(im, 900, 620)


def f06_problems():
    """The six blind spots, one real picture each.

    Deliberately plain: a name, the evidence, and a sentence. No verdict chips
    and no percentages - those live in the table beside it, and repeating them
    here would just be a second, worse table.
    """
    ROW, IMGW, IMGH = 540, 940, 470
    p = Panel("Six candidate blind spots, one scored item each",
              "Every picture is shown at the resolution the API actually delivered.",
              h=260 + ROW * len(PICKS))
    y = p.y + 10
    for head, uid, why, zspec in PICKS:
        im, q, gold, pred = _pick_panel(uid)
        p.rule(y - 18)
        yy = p.text(80, y + 8, head, 38, INK, True, wrap=22)
        p.text(80, yy + 14, why, 26, INK2, wrap=34)
        zoom = _zoom_of(im, zspec)
        if zoom is None:
            _, bot = p.paste(im, 700, y, IMGW, IMGH)
        else:                       # whole scene small, then the detail at 1:1
            fw, bot = p.paste(im, 700, y, 330, IMGH)
            zx = 700 + fw + 30
            _, zh = p.paste(zoom, zx, y, IMGW - fw - 30, IMGH)
            p.text(zx, y + zh + 8, "1:1 crop of the delivered pixels", 21, MUTED)
            bot = max(bot, zh + 34)
        x = 700 + IMGW + 60
        ty = p.text(x, y + 8, "asked", 22, MUTED, True)
        ty = p.text(x, ty + 4, q, 27, INK, wrap=42)
        ty = p.text(x, ty + 18, "gold", 22, MUTED, True)
        ty = p.text(x, ty + 4, gold, 27, INK, wrap=42)
        ty = p.text(x, ty + 18, "model", 22, MUTED, True)
        ty = p.text(x, ty + 4, pred, 27, CRIT, True, wrap=42)
        p.text(x, ty + 20, uid, 21, MUTED)
        last = max(y + bot, ty + 50)            # rows are as tall as their content
        y = max(y + ROW, last + 60)
    p.y = last
    return p.save("f06_problems")


EXAMPLES = [
    (e09_bad_gold,
     "A scored failure the audit would not defend: the model's answer and the gold "
     "answer differ only in surface form.",
     "One of 410 audited InfographicVQA failures."),
    (f06_problems,
     "The six candidate blind spots, each with one real item the model was scored on.",
     "Pictures only; the measured results are in the table."),
    (e08_generation_pipeline,
     "The deterministic scene-generation pipeline and what the target filter rejects.",
     "Counts are per scene and resolution, deduplicated across questions."),
    (e05_instrument,
     "Eight of the sixteen generated chart types, with the two delivered resolutions.",
     "Scenes are shown at the smaller of the two."),
]


def main() -> int:
    names = []
    for fn, cap, strip in EXAMPLES:
        try:
            names.append((fn(), cap, strip))
            print(f"  built {names[-1][0]}")
        except Exception as e:
            print(f"  !! {fn.__name__}: {type(e).__name__}: {e}")
    (Path("outputs/report") / "examples_index.json").write_text(
        json.dumps([{"name": n, "caption": c, "strip": s} for n, c, s in names], indent=1))
    print(f"wrote {len(names)} example figures -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
