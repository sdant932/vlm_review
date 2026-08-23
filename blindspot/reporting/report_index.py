"""Assemble the report's figures into one ordered set with editable captions.

The report is example-led and table-heavy: four figures survive, and each is a
real picture the model was actually scored on, apart from the pipeline block
diagram. Everything that used to be a chart is now a table in `tables.md`,
generated from the measured JSON.

Captions live here, not baked into the images, so they can be edited without
re-rendering anything.

    python -m blindspot.reporting.report_examples && python -m blindspot.reporting.report_index
"""

from __future__ import annotations

import html
import re
from pathlib import Path

OUT = Path("outputs/report")
FIGS = OUT / "figures"

# (file stem, kind, section, caption, confidence strip) in narrative order
ORDER = [
    ("e09_bad_gold", "example", "§2",
     "A scored failure where the model's answer and the gold answer differ only in "
     "surface form.",
     "One of 410 audited InfographicVQA failures."),
    ("f06_problems", "example", "§3",
     "The six candidate blind spots, each with one real item the model was scored on.",
     "Pictures only; the measured results are in Table 2."),
    ("e08_generation_pipeline", "diagram", "§5",
     "The deterministic scene-generation pipeline, and what the target filter rejected.",
     "Rejection counts are per scene and resolution, deduplicated across questions."),
    ("e05_instrument", "example", "§5",
     "Eight of the sixteen generated chart types, with the two delivered resolutions.",
     "Scenes are shown at the smaller of the two."),
]


# `[FIG:stem]` in the source, `[Figure 2]<!--FIG:stem-->` once resolved. The
# trailing comment is invisible in rendered markdown and keeps the reference
# re-resolvable, so figures can be reordered without hand-editing the prose.
REF_RE = re.compile(r"\[Figures? \d+\]<!--FIG:(\w+)-->|\[FIG:(\w+)\]")


def inject_refs(path: str = "outputs/report/blindspots.md") -> int:
    num = {stem: i for i, (stem, *_rest) in enumerate(ORDER, 1)}
    src = Path(path)
    text = src.read_text()
    missing = []

    def sub(m):
        stem = m.group(1) or m.group(2)
        if stem not in num:
            missing.append(stem)
            return m.group(0)
        return f"[Figure {num[stem]}]<!--FIG:{stem}-->"

    out, n = REF_RE.subn(sub, text)
    src.write_text(out)
    for stem in dict.fromkeys(missing):
        print(f"  !! reference to unknown figure: {stem}")
    unref = [s for s in num if f"FIG:{s}" not in out]
    for stem in unref:
        print(f"  !! figure never referenced in the text: {stem}")
    return n


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def main() -> int:
    n_ex = sum(1 for f in ORDER if f[1] == "example")
    md = ["# Figures\n",
          "Caption text for each figure, in the order they appear. The first line under "
          "each heading is the caption; the italic line is the confidence strip, which "
          "states the limit a reader should hold that figure to. Both are editable — "
          "neither is baked into the image.\n",
          f"\n{n_ex} of {len(ORDER)} figures are photographs of real scored items. "
          "The report's quantitative content is in `tables.md`, which is generated "
          "from the measured JSON rather than written by hand.\n"]
    cards, missing = [], []
    for i, (stem, kind, sec, cap, strip) in enumerate(ORDER, 1):
        png = FIGS / f"{stem}@2x.png"
        if not png.exists():
            missing.append(stem)
            continue
        md += [f"\n## Figure {i} — {sec} · {kind}\n", cap, f"\n\n*{strip}*\n",
               f"\n`figures/{stem}@2x.png`\n"]
        cards.append(
            f'<figure><figcaption><b>Figure {i}</b> · {sec} · {kind} · '
            f'<code>{stem}</code></figcaption>'
            f'<img src="{stem}@2x.png" alt="{esc(cap)}">'
            f'<p>{esc(cap)}</p><p class="strip"><i>{esc(strip)}</i></p></figure>')

    (OUT / "figures.md").write_text("".join(md))
    (FIGS / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Report figures</title>"
        "<style>body{background:#f2f2f0;font:15px/1.5 -apple-system,BlinkMacSystemFont,"
        "'Segoe UI',Roboto,sans-serif;margin:0;padding:28px}"
        "figure{background:#fff;border-radius:12px;padding:18px;margin:0 0 24px;"
        "box-shadow:0 1px 3px rgba(0,0,0,.12);max-width:1400px}"
        "img{width:100%;display:block;border:1px solid #e4e3df;border-radius:8px}"
        "figcaption{color:#8a8983;font-size:12px;margin-bottom:10px}"
        "p{margin:10px 0 0;color:#52514e}.strip{color:#8a8983;font-size:13px}</style>"
        + "".join(cards))
    print(f"{len(cards)} figures indexed ({n_ex} examples, "
          f"{len(ORDER) - n_ex} diagrams)")
    if missing:
        print("  !! missing:", ", ".join(missing))
    print(f"wrote {OUT / 'figures.md'} and {FIGS / 'index.html'}")
    print(f"resolved {inject_refs()} figure references in blindspots.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
