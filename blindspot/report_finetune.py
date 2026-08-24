"""Render the Part 3 finetuning artefacts: the sample gallery, the two
explanatory figures, and the example strip that goes into `part3.md`.

Three renderers live in one module because they share one drawing rule and one
set of colours, and because splitting them was what let the rule drift between
them.

The rule
--------
The outline is drawn strictly OUTSIDE the box. PIL renders a multi-pixel
`rectangle` outline inward, which paints over the glyph rows at the box edge and
makes a correct box look like it clips the text. An earlier version did exactly
that and the boxes were wrongly blamed. `_outline` is the only place a box is
ever stroked; every caller goes through it.

Only ONE box is ever drawn: the supervision target. The wider `accept_region`
is deliberately not drawn -- it is the whole enclosing node for shape-held
targets, and drawing it invites it to be read as the answer to "where is the
text", which it is not. It stays in the JSON, flagged, where it cannot be
mistaken for ground truth.

Usage
-----
    python -m blindspot.report_finetune gallery  [--records FILE] [--out FILE]
    python -m blindspot.report_finetune figures  [--out-dir DIR] [--dataset DIR]
    python -m blindspot.report_finetune examples [--dataset DIR] [--out-dir DIR]
                                               [--md FILE] [--inject]
"""

from __future__ import annotations

import argparse
import base64
import copy
import html
import io
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from blindspot import generate as G
from blindspot import generate_finetune as GF

REPO = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ constants

TARGET_GREEN = (34, 160, 90)      # --tgt in the gallery CSS: the supervision box
ACCENT_RED = (214, 48, 48)        # the box in the figures and the example strip
DIFF_BLUE = (24, 128, 216)        # pixels that one label changed
INK = (26, 26, 24)                # figure titles
MUTED = (122, 122, 118)           # figure subtitles

DEFAULT_RECORDS = Path("data/sft_bbox/sft_bbox_20.jsonl")
DEFAULT_GALLERY = Path("outputs/finetune/gallery.html")
DEFAULT_DATASET = Path("data/svgloc_mr")
DEFAULT_ASSETS = Path("outputs/part3/assets")
DEFAULT_MD = Path("outputs/part3/part3.md")

# gallery layout
SHOWN_W = 460                     # rendered width of both figures in a card
ZOOM_PAD_FRAC = 0.10              # zoom window = target box grown by 10% of the frame

# example strip layout
EX_FRAME_BOX = (430, 250)         # the whole frame is fitted into this
EX_ZOOM_WIN = (150, 125)          # source pixels around the target ...
EX_ZOOM_SCALE = 2                 # ... shown at 2x, so 300x250
EX_GAP = 14                       # white gutter between the two panels
EX_COUNT = 9
EX_SEED = 17                      # the generator's seed, reused for the strip

# The size/shape ladder: taken from the module that builds it, never restated.
# A second, hand-copied copy of these numbers is exactly how `_scene` came to
# rebuild a scene the ladder never shipped -- see its docstring.
RUNGS = GF.RUNGS
ASPECTS = [(name, *GF.canvas_for(ratio)) for name, ratio in GF.ASPECTS]
RUNG_INK = {                      # darkest = the largest frame
    "r55": (210, 210, 222),
    "r70": (168, 168, 180),
    "r85": (126, 126, 138),
    "r100": (84, 84, 96),
}

# Chart types are prose in a caption, not identifiers. `str.replace("_", " ")`
# would also mangle anything that legitimately carries an underscore, so the
# rewrites are listed rather than inferred.
CHART_LABEL = {
    "bar_chart": "bar chart",
    "pie_chart": "pie chart",
    "line_chart": "line chart",
    "org_chart": "org chart",
    "state_machine": "state machine",
}

# The record the shipped figure is drawn from. Pinned rather than sampled: the
# figure is an explanation, and it should not change shape because the dataset
# was regenerated. It is a preference, not a requirement -- see `_fig_row`,
# which picks a substitute from whatever dataset it is handed. Graph 89 does not
# exist in a 24-scene dev ladder, and requiring it made the workflow the docs
# recommend die on `rows[0]` with an IndexError.
FIG_UID = "mr:0089:portrait:r70:03"

FIG_PANEL = (250, 140)            # one panel of fig_box_extraction
FIG_GAP = 20
FIG_HEAD = 44                     # height of the title/subtitle band
FIG_FIT_PAD = 30                  # clearance the target must have inside a panel

FRAME_CELL = (263, 223)           # one aspect's cell in fig_frames
FRAME_ORIGIN = (29, 33)           # shared corner of the nested frames, in-cell
FRAME_SCALE = 212 / 1568          # the widest frame draws 212px across


# ------------------------------------------------------------------- drawing

def _outline(draw: ImageDraw.ImageDraw, box, colour, width: int = 1) -> None:
    """Stroke `box` with the line sitting entirely outside it.

    PIL draws a `rectangle` outline inward from the coordinates it is given, so
    a 3px stroke on a 14px-tall box eats three of the fourteen rows -- including
    the glyph rows that define the box. Growing the rectangle by the stroke
    width first puts the whole line outside, and the box then reads as what it
    is: the exact painted extent.
    """
    x0, y0, x1, y1 = box
    draw.rectangle([x0 - width, y0 - width, x1 + width, y1 + width],
                   outline=colour, width=width)


def _font(size: int):
    """A real TTF at `size`, via the generator's own font probe."""
    path, idx, _ = G.available_fonts()[0]
    return G.load(path, idx, size)


def _png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _kb(path: Path) -> int:
    return round(path.stat().st_size / 1024)


# =========================================================================
# 1. gallery
# =========================================================================

GALLERY_CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a19;--mut:#6b6b68;--line:#e3e3e0;--card:#fff;
      --tgt:#22a05a;--acc:#d68f1e;--code:#f5f5f3}
@media (prefers-color-scheme:dark){
 :root{--bg:#161614;--fg:#eeeeec;--mut:#a0a09c;--line:#2e2e2b;--card:#1e1e1c;
       --code:#232320}}
*{box-sizing:border-box}
body{margin:0;padding:32px;background:var(--bg);color:var(--fg);
     font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:22px;margin:0 0 6px}
.sub{color:var(--mut);margin:0 0 22px}
.legend{display:flex;gap:20px;flex-wrap:wrap;padding:12px 16px;background:var(--card);
        border:1px solid var(--line);border-radius:8px;margin-bottom:22px}
.key{display:flex;align-items:center;gap:8px;font-size:13px}
.sw{width:26px;height:0;border-top:3px solid}
.sw.t{border-color:var(--tgt)}
.sw.a{border-top-style:dashed;border-color:var(--acc)}
.stats{display:flex;gap:26px;flex-wrap:wrap;margin-bottom:26px;font-size:13px;
       color:var(--mut)}
.stats b{color:var(--fg);font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
      padding:18px;margin-bottom:18px}
.hd{display:flex;justify-content:space-between;align-items:baseline;gap:14px;
    margin-bottom:12px;flex-wrap:wrap}
.uid{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mut)}
.q{font-weight:600}
.tags{display:flex;gap:6px;flex-wrap:wrap}
.tag{font-size:11px;padding:2px 8px;border:1px solid var(--line);border-radius:99px;
     color:var(--mut)}
.imgs{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:12px}
.imgs figure{margin:0}
.imgs img{border:1px solid var(--line);border-radius:6px;display:block;max-width:100%}
figcaption{font-size:12px;color:var(--mut);margin-top:5px}
details{margin-top:8px}
summary{cursor:pointer;font-size:13px;color:var(--mut)}
pre{background:var(--code);border:1px solid var(--line);border-radius:6px;
    padding:12px;overflow-x:auto;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
    margin:8px 0 0}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--acc);
      border-radius:6px;padding:14px 16px;margin-bottom:22px;font-size:14px}
"""

GALLERY_NOTE = (
    "<b>One box only.</b> Part 2's wider accepted region is not drawn. For a "
    "target sitting inside a shape it is the entire node, so showing it next to "
    "the answer invites it to be read as the ground truth for &ldquo;where is "
    "the text&rdquo;, which it is not. It stays in the record below, flagged "
    "with <code>answers_the_question</code>. The outline is drawn just outside "
    "the box: a thick outline drawn on the box covers the glyph rows at its own "
    "edge and makes a correct box look like it clips the text."
)


def load_records(path: Path) -> list[dict]:
    """One JSON record per line, in file order -- the gallery does not sort."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _gallery_views(rec: dict) -> tuple[Image.Image, Image.Image]:
    """The two figures of a card: the whole frame, and a zoom on the target.

    Both are needed. At 900x570 a target occupying 0.02% of the frame is a few
    pixels and invisible at page scale, so the zoom is what makes the ground
    truth checkable by eye; the whole frame is what says the image reached the
    model untouched.

    The box is stroked on the full frame *before* it is shrunk -- at 0.51x a
    one-pixel line all but disappears, which is the honest depiction: that is
    how much of the frame the target actually occupies. On the zoom it is
    stroked after, so it stays one crisp pixel.
    """
    im = Image.open(REPO / rec["image"]["path"]).convert("RGB")
    W, H = im.size
    nx0, ny0, nx1, ny1 = rec["target"]["box_norm"]
    px0, py0, px1, py1 = nx0 * W, ny0 * H, nx1 * W, ny1 * H

    full = im.copy()
    _outline(ImageDraw.Draw(full), (px0, py0, px1, py1), TARGET_GREEN)
    full = full.resize((SHOWN_W, int(H * SHOWN_W / W)), Image.LANCZOS)

    # The zoom window is the box grown by a tenth of the frame on every side,
    # clipped to the frame. The window is kept as floats so the box can be
    # placed against the same origin the crop was taken from.
    padx, pady = ZOOM_PAD_FRAC * W, ZOOM_PAD_FRAC * H
    wx0, wy0 = max(0.0, px0 - padx), max(0.0, py0 - pady)
    wx1, wy1 = min(float(W), px1 + padx), min(float(H), py1 + pady)
    crop = im.crop((int(wx0), int(wy0), int(wx1), int(wy1)))
    s = SHOWN_W / crop.width
    zoom = crop.resize((int(crop.width * s), int(crop.height * s)), Image.LANCZOS)
    _outline(ImageDraw.Draw(zoom),
             ((px0 - wx0) * s, (py0 - wy0) * s, (px1 - wx0) * s, (py1 - wy0) * s),
             TARGET_GREEN)
    return full, zoom


def _card(i: int, rec: dict) -> str:
    """One record, rendered exactly as it was written to the JSONL."""
    meta, cur, tgt = rec["meta"], rec["curriculum"], rec["target"]
    W, H = rec["image"]["px"]
    full, zoom = _gallery_views(rec)

    tags = [
        meta["chart_type"],
        f"{meta['theme']} ({cur['polarity']})",
        f"role: {meta['target_role']}",
        f"band {cur['area_band']}",
        f"{tgt['area_frac'] * 100:.3f}% of image",
        f"{cur['font_px']}px font",
    ]
    def img(im, alt, cap):
        b64 = base64.b64encode(_png_bytes(im)).decode("ascii")
        return (f"<figure><img src='data:image/png;base64,{b64}' alt='{alt}'>"
                f"<figcaption>{cap}</figcaption></figure>")

    return (
        "<div class='card'>"
        f"<div class='hd'><span class='q'>{i}. {html.escape(meta['question'])}</span>"
        f"<span class='uid'>{rec['uid']}</span></div>"
        "<div class='tags'>"
        + "".join(f"<span class='tag'>{html.escape(t)}</span>" for t in tags)
        + "</div><div class='imgs'>"
        + img(full, "full image",
              f"full image &mdash; {W}&times;{H}, delivered untouched")
        + img(zoom, "zoom on target", "zoom on the target")
        + "</div><details><summary>ground-truth record</summary><pre>"
        + html.escape(json.dumps(rec, indent=2))
        + "</pre></details></div>"
    )


def _stats(records: list[dict]) -> str:
    areas = [r["target"]["area_frac"] * 100 for r in records]
    dark = sum(1 for r in records if r["curriculum"]["polarity"] == "dark")
    return (
        "<div class='stats'>"
        f"<span><b>{len(records)}</b> records</span>"
        f"<span><b>{len({r['meta']['chart_type'] for r in records})}</b> chart types</span>"
        f"<span><b>{len({r['meta']['theme'] for r in records})}</b> themes</span>"
        f"<span><b>{dark}</b> dark / <b>{len(records) - dark}</b> light</span>"
        f"<span>target area <b>{min(areas):.3f}%</b>&ndash;<b>{max(areas):.3f}%</b>"
        " of image</span></div>"
    )


def render_gallery(records: list[dict], out: Path) -> Path:
    """Write one self-contained HTML file: no assets, no network, no build."""
    parts = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Bounding-box training samples</title><style>",
        GALLERY_CSS,
        "</style><body><div class='wrap'>"
        "<h1>Bounding-box training samples</h1>"
        f"<p class='sub'>{len(records)} supervised records built from the "
        "synthetic chart set, shown with the ground truth from the manifest.</p>"
        "<div class='legend'><span class='key'><span class='sw t'></span>"
        "supervision target &mdash; exact painted extent of the text</span></div>"
        f"<div class='note'>{GALLERY_NOTE}</div>",
        _stats(records),
    ]
    parts += [_card(i, rec) for i, rec in enumerate(records, 1)]
    parts.append("</div></body></html>")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    return out


def cmd_gallery(args) -> int:
    records = load_records(args.records)
    out = render_gallery(records, args.out)
    print(f"{len(records)} records -> {out}  ({_kb(out)} KB)")
    return 0


# =========================================================================
# 2. figures
# =========================================================================

def _manifest(dataset: Path) -> list[dict]:
    with open(dataset / "manifest.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _scene(row: dict):
    """Rebuild the ladder scene that manifest `row` was measured on.

    The ladder stores images and a manifest but not the scenes, so a figure that
    needs a second render of the same scene has to reconstruct it. It does that
    by calling the ladder's own constructor, not by restating its recipe: an
    earlier version reimplemented the recipe here and drifted from it (chart
    type by global position instead of position within the aspect, canvas by a
    hard-coded block of sixteen, and `complexity = 1` against the ladder's 4).
    The scenes it produced were not the shipped ones, so `target_idx` indexed a
    different list of labels -- 268 of the 1513 shipped rows out of range and a
    further 159 pointing at a different string, with no error raised anywhere.

    Every argument comes from the row itself, so a scene rebuilt here is the one
    that row describes even if the ladder's own layout of gids ever changes.
    Returns None when the builder declines the scene.
    """
    return GF.build_scene(row["graph_id"], row["chart_type"], G.available_fonts(),
                          GF.SEED, tuple(row["canvas_px"]), GF.COMPLEXITY,
                          min(RUNGS.values()))


def _fig_row(rows: list[dict]) -> dict:
    """The record `fig_box_extraction` draws, given the rows it was handed.

    `FIG_UID` when the dataset has it, so the shipped figure is stable across
    regenerations of the committed ladder. When it does not -- a dev ladder
    built with `--scenes-per-aspect 4` has 24 scenes and no graph 89 -- a
    substitute is chosen rather than the first row blindly taken, because the
    first row is whatever the ladder happened to emit first and its target may
    not even fit in a panel.

    The substitute is the largest rung available (bigger delivered glyphs, so
    the four panels are readable at figure scale) whose target clears the panel
    edges, tie-broken on uid. Deterministic: the same dataset always yields the
    same figure.
    """
    pinned = next((r for r in rows if r["uid"] == FIG_UID), None)
    if pinned is not None:
        return pinned
    if not rows:
        raise SystemExit("dataset manifest is empty: nothing to draw")

    def fits(r: dict) -> bool:
        x0, y0, x1, y1 = r["box_px"]
        return (x1 - x0 <= FIG_PANEL[0] - 2 * FIG_FIT_PAD
                and y1 - y0 <= FIG_PANEL[1] - 2 * FIG_FIT_PAD)

    candidates = [r for r in rows if fits(r)] or rows
    return min(candidates, key=lambda r: (-r["scale"], r["uid"]))


def _panel(im: Image.Image, box, size=FIG_PANEL) -> tuple[Image.Image, tuple[int, int]]:
    """A `size` window of `im`, centred on `box`, clipped to the frame."""
    w, h = size
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    x0 = min(max(0, int(cx - w / 2)), max(0, im.width - w))
    y0 = min(max(0, int(cy - h / 2)), max(0, im.height - h))
    out = Image.new("RGB", size, "white")
    out.paste(im.crop((x0, y0, x0 + w, y0 + h)), (0, 0))
    return out, (x0, y0)


def fig_box_extraction(out_path: Path, dataset: str = "data/svgloc_mr") -> Path:
    """Four panels: the delivered image, the same scene with no text, the pixels
    one label changed, and the box those pixels imply.

    This is not an illustration of the algorithm, it is the algorithm: the scene
    is re-rendered twice and the difference taken, exactly as `exact_ink` does
    it. The prose version of this ("render twice and subtract") is much harder
    to follow than the picture.
    """
    ds = REPO / dataset if not Path(dataset).is_absolute() else Path(dataset)
    rows = _manifest(ds)
    row = _fig_row(rows)

    sc = _scene(row)
    if sc is None:                                   # builder declined this seed
        raise SystemExit(f"could not rebuild scene {row['graph_id']}")
    # The rebuilt scene must be the one the row was measured on, or `target_idx`
    # below indexes a different list of labels and the figure boxes the wrong
    # string while captioning it as the delivered image. Checked rather than
    # assumed: that failure is silent, and it shipped once already.
    texts = sc.texts
    ti = row["target_idx"]
    got = texts[ti]["s"] if ti < len(texts) else None
    if got != row["target_text"]:
        raise SystemExit(
            f"{row['uid']}: rebuilt scene does not match the manifest -- "
            f"target {ti} of {len(texts)} is {got!r}, manifest says "
            f"{row['target_text']!r}")
    scale = RUNGS[row["rung"]]

    # Three renders of one scene: everything, nothing, and this label alone.
    # All four panels come from the same pass so they line up pixel for pixel;
    # cropping one of them from the shipped PNG instead would show the delivered
    # frame beside a re-render and invite the reader to compare the wrong things.
    whole = G.render(sc, scale)[0].convert("RGB")
    blank = G.render(sc, scale, skip_text=True)[0].convert("RGB")
    solo = copy.copy(sc)
    keep = texts[ti]
    solo.prims = [p for p in sc.prims if p["k"] != "text" or p is keep]
    only = G.render(solo, scale)[0].convert("RGB")

    box = row["box_px"]
    shown, (ox, oy) = _panel(whole, box)
    plain, _ = _panel(blank, box)

    # The difference of the two renders is this label's ink and nothing else:
    # no layout box, so no clipped final glyph, and no intensity threshold, so
    # no connector line running under the text gets absorbed into the box.
    win = (ox, oy, ox + FIG_PANEL[0], oy + FIG_PANEL[1])
    a, b = only.crop(win), blank.crop(win)
    diff = Image.new("RGB", FIG_PANEL, "white")
    dpx, apx, bpx = diff.load(), a.load(), b.load()
    xs, ys = [], []
    for y in range(FIG_PANEL[1]):
        for x in range(FIG_PANEL[0]):
            if apx[x, y] != bpx[x, y]:
                dpx[x, y] = DIFF_BLUE
                xs.append(x)
                ys.append(y)

    boxed = shown.copy()
    ink = ((min(xs), min(ys), max(xs), max(ys)) if xs else
           (box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy))
    _outline(ImageDraw.Draw(boxed), ink, ACCENT_RED, width=3)

    titles = [
        ("1. the image", "as delivered to the model"),
        ("2. same scene, no text", "re-rendered from the seed"),
        ("3. pixels this label changed", "difference of the two"),
        ("4. the box", "extent of those pixels"),
    ]
    panels = [shown, plain, diff, boxed]
    W = len(panels) * FIG_PANEL[0] + (len(panels) - 1) * FIG_GAP
    fig = Image.new("RGB", (W, FIG_HEAD + FIG_PANEL[1]), "white")
    d = ImageDraw.Draw(fig)
    head, sub = _font(13), _font(10)
    for i, (panel, (t, s)) in enumerate(zip(panels, titles)):
        x = i * (FIG_PANEL[0] + FIG_GAP)
        d.text((x, 4), t, font=head, fill=INK, anchor="lt")
        d.text((x, 22), s, font=sub, fill=MUTED, anchor="lt")
        fig.paste(panel, (x, FIG_HEAD))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(_png_bytes(fig))
    return out_path


def fig_frames(out_path: Path) -> Path:
    """The size/shape ladder: six aspect ratios, four sizes each, to scale.

    Drawn from a common corner so the four rungs of one aspect nest, and the six
    aspects can be compared against each other without measuring anything.
    """
    cols, rows = 3, 2
    cw, ch = FRAME_CELL
    fig = Image.new("RGB", (cols * cw, rows * ch), "white")
    d = ImageDraw.Draw(fig)
    title, sub = _font(13), _font(10)

    for i, (name, W, H) in enumerate(ASPECTS):
        cx = (i % cols) * cw
        cy = (i // cols) * ch
        ox, oy = cx + FRAME_ORIGIN[0], cy + FRAME_ORIGIN[1]
        d.text((cx + 26, cy + 5), name, font=title, fill=INK, anchor="lt")
        d.text((cx + 26, cy + 22), f"{W}x{H} max, {len(RUNGS)} sizes",
               font=sub, fill=MUTED, anchor="lt")
        # largest first, so the smaller rungs stay legible on top of it
        for rung, scale in sorted(RUNGS.items(), key=lambda kv: -kv[1]):
            w = round(W * scale * FRAME_SCALE)
            h = round(H * scale * FRAME_SCALE)
            _outline(d, (ox, oy, ox + w, oy + h), RUNG_INK[rung],
                     width=3 if scale == 1.0 else 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(_png_bytes(fig))
    return out_path


def cmd_figures(args) -> int:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    made = [
        fig_box_extraction(out_dir / "fig_box_extraction.png", str(args.dataset)),
        fig_frames(out_dir / "fig_frames.png"),
    ]
    for p in made:
        rel = p.relative_to(REPO) if p.is_absolute() and REPO in p.parents else p
        print(f"  {rel}  ({_kb(p)} KB)")
    return 0


# =========================================================================
# 3. examples
# =========================================================================

def _pretty_chart(ctype: str) -> str:
    return CHART_LABEL.get(ctype, ctype)


def caption(row: dict) -> str:
    W, H = row["image_px"]
    return (f'the text "{row["target_text"]}" — {row["aspect"]} '
            f'{W}×{H}, {_pretty_chart(row["chart_type"])}, '
            f'{row["box_area_frac"] * 100:.3f}% of the image')


def pick_examples(rows: list[dict], n: int = EX_COUNT) -> list[dict]:
    """One record per aspect-ratio/size cell, biased to the small end of it.

    Selecting per cell is the point: ten views of the same shape would say
    nothing about whether the model can normalise coordinates against the frame
    it is given, which is the defect the ladder exists to expose. Within a cell
    the pool is the smallest third, because that is where the model fails and a
    typical example would be an easy one.

    The cell order is shuffled from a fixed seed so the strip does not read as
    an ordered sweep from widest to tallest, which would suggest a gradient the
    data does not have. Fixed seed, so the strip is reproducible.
    """
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        cells[(row["aspect"], row["rung"])].append(row)

    rng = random.Random(EX_SEED)
    keys = sorted(cells)
    rng.shuffle(keys)

    picked = []
    for key in keys[:n]:
        pool = sorted(cells[key], key=lambda r: r["box_area_frac"])
        pool = pool[: max(1, len(pool) // 3)]
        picked.append(rng.choice(pool))
    return picked


def compose_example(row: dict, dataset: Path) -> Image.Image:
    """The whole frame with the target boxed, and a zoom beside it.

    One image rather than two, so the markdown needs one link per example and
    the pair can never be separated by a reflow.
    """
    im = Image.open(dataset / row["image"]).convert("RGB")
    W, H = im.size
    x0, y0, x1, y1 = row["box_px"]

    # left: the whole frame, box stroked before the shrink. At this scale the
    # line is nearly invisible, which is the honest depiction of the target.
    framed = im.copy()
    _outline(ImageDraw.Draw(framed), (x0, y0, x1, y1), ACCENT_RED)
    s = min(EX_FRAME_BOX[0] / W, EX_FRAME_BOX[1] / H)
    left = framed.resize((round(W * s), round(H * s)), Image.LANCZOS)

    # right: a fixed window of source pixels around the target, at 2x
    ww, wh = EX_ZOOM_WIN
    cx0 = min(max(0, int((x0 + x1) / 2 - ww / 2)), max(0, W - ww))
    cy0 = min(max(0, int((y0 + y1) / 2 - wh / 2)), max(0, H - wh))
    k = EX_ZOOM_SCALE
    right = im.crop((cx0, cy0, cx0 + ww, cy0 + wh)).resize(
        (ww * k, wh * k), Image.LANCZOS)
    _outline(ImageDraw.Draw(right),
             ((x0 - cx0) * k, (y0 - cy0) * k, (x1 - cx0) * k, (y1 - cy0) * k),
             ACCENT_RED, width=2)

    out = Image.new("RGB", (left.width + EX_GAP + right.width, wh * k), "white")
    out.paste(left, (0, (out.height - left.height) // 2))
    out.paste(right, (left.width + EX_GAP, 0))
    return out


def inject_markdown(md_path: Path, block: str, heading: str = "#### Examples") -> None:
    """Splice `block` into `md_path` under `heading`, replacing what is there.

    The section is machine-generated, so it is replaced wholesale rather than
    appended to; the surrounding prose is hand-written and is left alone.
    """
    lines = md_path.read_text(encoding="utf-8").splitlines()
    try:
        i = next(n for n, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        raise SystemExit(f"{md_path}: no '{heading}' heading to inject under")

    j = i + 1
    while j < len(lines) and not lines[j].lstrip().startswith("#"):
        j += 1
    body = ["", *block.splitlines()]
    while body and not body[-1].strip():
        body.pop()
    tail = lines[j:]
    if tail:
        body.append("")
    md_path.write_text("\n".join(lines[: i + 1] + body + tail) + "\n",
                       encoding="utf-8")


def cmd_examples(args) -> int:
    dataset = args.dataset
    rows = pick_examples(_manifest(dataset))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    listing, md = [], []
    for i, row in enumerate(rows, 1):
        name = f"ex{i:02d}.png"
        cap = caption(row)
        (args.out_dir / name).write_bytes(_png_bytes(compose_example(row, dataset)))
        listing.append(f"  {name}  {cap}")
        md.append(f"![{cap}](assets/{name})")

    block = "\n\n".join(md)
    print("\n".join(listing))
    print()
    print(block)
    if args.inject:
        inject_markdown(args.md, block)
    return 0


# =========================================================================
# CLI
# =========================================================================

def _path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO / q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gallery", help="the bounding-box sample gallery")
    g.add_argument("--records", type=_path, default=REPO / DEFAULT_RECORDS)
    g.add_argument("--out", type=_path, default=REPO / DEFAULT_GALLERY)
    g.set_defaults(func=cmd_gallery)

    f = sub.add_parser("figures", help="the two explanatory figures")
    f.add_argument("--out-dir", type=_path, default=REPO / DEFAULT_ASSETS)
    f.add_argument("--dataset", type=_path, default=REPO / DEFAULT_DATASET)
    f.set_defaults(func=cmd_figures)

    e = sub.add_parser("examples", help="the example strip for part3.md")
    e.add_argument("--dataset", type=_path, default=REPO / DEFAULT_DATASET)
    e.add_argument("--out-dir", type=_path, default=REPO / DEFAULT_ASSETS)
    e.add_argument("--md", type=_path, default=REPO / DEFAULT_MD)
    e.add_argument("--inject", action="store_true",
                   help="also splice the markdown into --md")
    e.set_defaults(func=cmd_examples)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
