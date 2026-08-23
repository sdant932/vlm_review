"""Stage-1 contact sheet: four candidate example images per blind spot.

The report's centrepiece figure shows one real picture per blind spot. Which
picture is an editorial call, not a computable one, so this module lays out four
defensible candidates for each of the six problems and lets a human pick.

Every panel renders the image through `effective_size()`, so a candidate that
looks illegible here looked illegible to the model too.

    python -m blindspot.reporting.report_candidates      # -> outputs/report/candidates.html
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from PIL import Image

from blindspot.core.adapters import load
from blindspot.reporting.report_examples import as_model_saw, draw_target, fit
from blindspot.core.scoring import score

OUT = Path("outputs/report")
IMGS = OUT / "candidates"
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


def busiest_crop(im: Image.Image, w: int = 900, h: int = 600) -> Image.Image:
    """A 1:1 window over the most detailed part of the delivered image.

    Edge energy is a decent proxy for text density, and text is what the
    resolution argument is about.
    """
    from PIL import ImageFilter, ImageStat
    w, h = min(w, im.width), min(h, im.height)
    best, bx, by = -1.0, 0, 0
    edges = im.convert("L").filter(ImageFilter.FIND_EDGES)
    for y in range(0, max(1, im.height - h + 1), max(1, h // 2)):
        for x in range(0, max(1, im.width - w + 1), max(1, w // 2)):
            v = ImageStat.Stat(edges.crop((x, y, x + w, y + h))).mean[0]
            if v > best:
                best, bx, by = v, x, y
    return im.crop((bx, by, bx + w, by + h))


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
    from blindspot.analysis.svgloc_eval import load_run
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
    from blindspot.core.stats import is_na
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


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def main() -> int:
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

    (OUT / "candidates.html").write_text(
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
    print(f"wrote {OUT/'candidates.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
