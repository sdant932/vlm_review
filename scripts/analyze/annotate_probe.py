"""Annotate GT vs prediction directly onto the screenshots.

Self-describing: each image carries its own labels, so a PNG pulled out of the
folder still explains itself without the surrounding page.

    green box  = ground truth (the element the instruction refers to)
    blue cross = where Haiku 4.5 actually clicked
    dashed line between them = the miss

    python scripts/analyze/annotate_probe.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from blindspot.core.adapters import load
from blindspot.core.prompts import prompt_text
from blindspot.core.scoring import point_in_bbox

R, OUT = Path("results"), Path("outputs/probe")
TAG = "haiku-4-5_think2000_native_r0"
GT_C, PR_C = (27, 175, 122), (42, 120, 214)      # palette slots 3 and 1
HALO = (0, 0, 0)

FONT_CANDIDATES = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                   "/System/Library/Fonts/Helvetica.ttc",
                   "/Library/Fonts/Arial Bold.ttf"]


def font(px: int):
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            try:
                return ImageFont.truetype(f, px)
            except Exception:
                pass
    return ImageFont.load_default()


def tag(d, xy, text, colour, fnt, size, avoid=None, prefer="above"):
    """Filled label chip, kept inside the image and off the thing it labels.

    Targets sit anywhere -- including flush against an edge (ScreenSpot has boxes
    at y0 = 0.0 exactly) -- so a chip placed blindly above the box gets clipped
    off-canvas. Try the preferred side, fall back to the opposite, then clamp.
    """
    W, H = size
    x, y = xy
    l, t, r, b = d.textbbox((0, 0), text, font=fnt)
    w, h = r - l, b - t
    pad = max(4, h // 3)
    bw, bh = w + 2 * pad, h + 2 * pad

    if avoid:
        ax0, ay0, ax1, ay1 = avoid
        gap = pad
        above, below = ay0 - bh - gap, ay1 + gap
        y = above if (prefer == "above" and above >= 0) else below
        if y + bh > H:                      # no room below either -> go above
            y = max(0, above)
        x = ax0
    x = min(max(0, x), W - bw)
    y = min(max(0, y), H - bh)
    d.rounded_rectangle([x, y, x + bw, y + bh], radius=pad,
                        fill=colour + (235,), outline=(255, 255, 255, 210), width=1)
    d.text((x + pad - l, y + pad - t), text, fill=(255, 255, 255), font=fnt)
    return (x, y, x + bw, y + bh)


def wrap(d, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if d.textlength(t, font=fnt) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def banner(im: Image.Image, instruction: str, hit: bool, sub: str,
           prompt: str = "", answer: str = "") -> Image.Image:
    """Caption strip above the screenshot.

    Extending the canvas rather than overlaying keeps the instruction from
    covering the very UI the reader needs to judge the click against.
    """
    W = im.width
    fs = max(17, round(W / 66))
    f_main, f_sub = font(fs), font(max(13, round(fs * 0.72)))
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    pad = round(fs * 0.85)
    chip = "HIT" if hit else "MISS"
    chip_w = tmp.textlength(chip, font=f_sub) + pad * 1.6
    lines = wrap(tmp, f'"{instruction}"', f_main, W - pad * 3 - chip_w)
    # The metadata line is long and narrow zoom crops are common, so it wraps too
    # -- otherwise it silently runs off the right edge.
    subs = wrap(tmp, sub, f_sub, W - pad * 2)
    # Verbatim prompt, so the image shows what the model was actually asked. The
    # ask is a POINT in 0-1000 space, which is why the mark is a crosshair and
    # not a box -- the official benchmark protocol asks for a bbox instead.
    f_mono = font(max(12, round(fs * 0.66)))
    plines = []
    for para in (prompt or "").split("\n"):
        plines.extend(wrap(tmp, para, f_mono, W - pad * 3.2) if para.strip() else [""])
    prow = round(fs * 0.9)
    bh = (pad * 2 + len(lines) * round(fs * 1.32)
          + len(subs) * round(fs * 0.95)
          + (round(fs * 1.5) + len(plines) * prow + (prow if answer else 0) if plines else 0))

    out = Image.new("RGB", (W, im.height + bh), (250, 250, 248))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, W, bh], fill=(250, 250, 248))
    d.line([0, bh - 1, W, bh - 1], fill=(210, 209, 204), width=2)

    y = pad
    for ln in lines:
        d.text((pad, y), ln, fill=(11, 11, 11), font=f_main)
        y += round(fs * 1.32)
    for sl in subs:
        d.text((pad, y + 2), sl, fill=(90, 89, 85), font=f_sub)
        y += round(fs * 0.95)

    if plines:
        y += round(fs * 0.55)
        d.text((pad, y), "PROMPT SENT TO THE MODEL", fill=(120, 119, 114), font=f_mono)
        y += prow
        top = y - 2
        for pl in plines:
            d.text((pad + round(fs * 0.55), y), pl, fill=(60, 59, 56), font=f_mono)
            y += prow
        if answer:
            d.text((pad + round(fs * 0.55), y), f"model returned: {answer}",
                   fill=(42, 120, 214), font=f_mono)
            y += prow
        d.line([pad, top, pad, y - 4], fill=(200, 199, 194), width=2)

    cx = W - pad - chip_w
    cy = pad * 0.7
    ch = round(fs * 1.25)
    d.rounded_rectangle([cx, cy, cx + chip_w, cy + ch], radius=ch // 2,
                        fill=(12, 163, 12) if hit else (208, 59, 59))
    tw = tmp.textlength(chip, font=f_sub)
    d.text((cx + (chip_w - tw) / 2, cy + (ch - fs * 0.72) / 2 - 1), chip,
           fill=(255, 255, 255), font=f_sub)

    out.paste(im, (0, bh))
    return out


def draw(base: Image.Image, gold, pred, scale=1.0):
    """Draw GT box + prediction cross + miss line onto a copy of `base`."""
    im = base.convert("RGB")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    W, H = im.size
    lw = max(3, round(max(W, H) / 500 * scale))
    fnt = font(max(15, round(max(W, H) / 62 * scale)))

    x0, y0, x1, y1 = [c * s for c, s in zip(gold, (W, H, W, H))]
    gx, gy = (x0 + x1) / 2, (y0 + y1) / 2
    px, py = pred[0] * W, pred[1] * H

    # miss line first, so the marks sit on top of it
    if 0 <= pred[0] <= 1 and 0 <= pred[1] <= 1:
        for col, wd in ((HALO + (170,), lw + 2), (PR_C + (200,), lw)):
            for i in range(0, int(((px - gx) ** 2 + (py - gy) ** 2) ** .5), 26):
                t0 = i / max(1e-6, ((px - gx) ** 2 + (py - gy) ** 2) ** .5)
                t1 = min(1.0, t0 + 13 / max(1e-6, ((px - gx) ** 2 + (py - gy) ** 2) ** .5))
                d.line([gx + (px - gx) * t0, gy + (py - gy) * t0,
                        gx + (px - gx) * t1, gy + (py - gy) * t1], fill=col, width=wd)

    # ground-truth box, haloed so it reads on light and dark UI alike
    d.rectangle([x0 - lw, y0 - lw, x1 + lw, y1 + lw], outline=HALO + (190,), width=lw + 2)
    d.rectangle([x0, y0, x1, y1], outline=GT_C + (255,), width=lw)

    # prediction crosshair
    r = lw * 8
    for col, wd in ((HALO + (190,), lw + 3), (PR_C + (255,), lw)):
        d.line([px - r, py, px + r, py], fill=col, width=wd)
        d.line([px, py - r, px, py + r], fill=col, width=wd)
        d.ellipse([px - r * .5, py - r * .5, px + r * .5, py + r * .5], outline=col, width=wd)

    # Labels sit outside the marks they name, never on top of them.
    tag(d, (x0, y0), "GROUND TRUTH", GT_C, fnt, (W, H),
        avoid=(x0 - lw, y0 - lw, x1 + lw, y1 + lw), prefer="above")
    tag(d, (px, py), "HAIKU CLICKED", PR_C, fnt, (W, H),
        avoid=(px - r, py - r, px + r, py + r), prefer="below")

    im.paste(Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB"))
    return im


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((R / "probe_uids.json").read_text())
    made = []
    for ds, uids in plan.items():
        exs = {e.uid: e for e in load(ds)}
        rows = {json.loads(l)["uid"]: json.loads(l)
                for l in open(R / f"{ds}__{TAG}.jsonl") if l.strip()}
        for uid in uids:
            ex, rec = exs[uid], rows.get(uid)
            if not rec or not rec.get("pred"):
                continue
            pred = tuple(rec["pred"])
            hit = bool(point_in_bbox(pred, ex.gold))
            base = Image.open(ex.images[0])
            W, H = base.size
            full = draw(base, ex.gold, pred)

            # zoom framing both marks
            x0, y0, x1, y1 = [c * s for c, s in zip(ex.gold, (W, H, W, H))]
            xs, ys = [x0, x1], [y0, y1]
            if 0 <= pred[0] <= 1 and 0 <= pred[1] <= 1:
                xs.append(pred[0] * W); ys.append(pred[1] * H)
            pad = max(x1 - x0, y1 - y0) * 2 + max(W, H) * 0.05
            box = (max(0, int(min(xs) - pad)), max(0, int(min(ys) - pad)),
                   min(W, int(max(xs) + pad)), min(H, int(max(ys) + pad)))
            crop = base.crop(box)
            g2 = [(x0 - box[0]) / crop.width, (y0 - box[1]) / crop.height,
                  (x1 - box[0]) / crop.width, (y1 - box[1]) / crop.height]
            p2 = ((pred[0] * W - box[0]) / crop.width, (pred[1] * H - box[1]) / crop.height)
            zoom = draw(crop, g2, p2, scale=1.9)

            gx = (ex.gold[0] + ex.gold[2]) / 2
            gy = (ex.gold[1] + ex.gold[3]) / 2
            off = ((pred[0] - gx) ** 2 + (pred[1] - gy) ** 2) ** .5
            side = (ex.meta.get("target_area_frac", 0) * W * H) ** .5
            sub = (f"{ds} \u00b7 {uid.split(':')[-1]} \u00b7 {W}\u00d7{H} \u00b7 "
                   f"target \u2248{side:.0f}px \u00b7 miss {off*100:.1f}% of screen")
            ptxt = prompt_text(ex)
            ans = rec.get("raw") or ""
            stem = uid.replace(":", "_")
            banner(full, ex.question, hit, sub, ptxt, ans).save(OUT / f"{stem}__1_full.png")
            banner(zoom, ex.question, hit, sub + " \u00b7 zoomed", ptxt, ans).save(
                OUT / f"{stem}__2_zoom.png")
            made.append((uid, ex.question[:52], hit, f"{W}x{H}"))
    print(f"{'uid':44s} {'hit':>4s}  {'size':>10s}  instruction")
    for uid, q, hit, sz in made:
        print(f"{uid:44s} {'YES' if hit else 'no':>4s}  {sz:>10s}  {q}")
    print(f"\n{len(made)*2} PNGs -> {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
