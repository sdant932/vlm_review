"""Build one self-contained page for pasting into Google Docs.

Docs does not follow local image paths, so every figure is embedded as a base64
data URI and travels with the clipboard. Styling is inline and deliberately
plain: Docs discards most CSS, so structure is carried by real heading,
paragraph, list and table elements rather than by classes.

Markdown tables become real `<table>` elements — Docs pastes those as editable
tables, which is the whole point of the report's tables being tables.

Figures are inserted after the paragraph that first references them, so the
narrative order in `blindspots.md` decides the layout.

    python -m blindspot.reporting.report_paste
"""

from __future__ import annotations

import base64
import html
import io
import re
from pathlib import Path

from PIL import Image

from blindspot.reporting.report_index import ORDER

OUT = Path("outputs/report")
FIGS = OUT / "figures"
PASTE_MAX_W = 1600          # ~245 DPI across a 6.5in Docs column

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Calibri, "
        "'Helvetica Neue', Arial, sans-serif")
BODY = f"font-family:{FONT};font-size:11pt;line-height:1.5;color:#111"
SMALL = f"font-family:{FONT};font-size:9pt;color:#666"

REF_RE = re.compile(r"\[Figures? (\d+)\]<!--FIG:(\w+)-->")
NUM = {stem: i for i, (stem, *_rest) in enumerate(ORDER, 1)}
META = {stem: (cap, strip) for stem, _k, _s, cap, strip in ORDER}


def esc(s) -> str:
    return html.escape(str(s), quote=False)


def data_uri(png: Path) -> str:
    im = Image.open(png).convert("RGB")
    if im.width > PASTE_MAX_W:
        im = im.resize((PASTE_MAX_W, round(im.height * PASTE_MAX_W / im.width)),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def inline(s: str) -> str:
    """Markdown inline spans -> HTML, with figure markers turned into plain text.

    A reference reads as a parenthetical, since the figure and its numbered
    caption are placed immediately after the paragraph that names it.
    """
    s = REF_RE.sub(lambda m: f"\x00(Figure {m.group(1)})\x01", s)
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"`(.+?)`", r'<span style="font-family:Consolas,monospace">\1</span>', s)
    return re.sub(r"\s+\x00", " \x00", s).replace("\x00", "").replace("\x01", "").strip()


def figure_html(stem: str) -> str:
    png = FIGS / f"{stem}@2x.png"
    if not png.exists():
        return ""
    cap, strip = META[stem]
    return (f'<p style="margin:22px 0 4px"><img src="{data_uri(png)}" '
            f'style="width:100%;max-width:660px" alt="{esc(cap)}"></p>'
            f'<p style="margin:0 0 4px;{SMALL};font-size:9.5pt;color:#444">'
            f'<strong>Figure {NUM[stem]}.</strong> {esc(cap)}</p>'
            f'<p style="margin:0 0 24px;{SMALL};color:#777"><em>{esc(strip)}</em></p>')


def is_row(s: str) -> bool:
    return s.startswith("|") and s.endswith("|")


def cells(s: str) -> list[str]:
    return [c.strip() for c in s.strip().strip("|").split("|")]


def table_html(rows: list[str]) -> str:
    """A markdown table as a real <table>; the second row is the alignment rule."""
    head, body = cells(rows[0]), [cells(r) for r in rows[2:]]
    th = ("padding:6px 9px;border:1px solid #c9c8c3;background:#f2f2f0;"
          "text-align:left;vertical-align:top;font-weight:700")
    td = "padding:6px 9px;border:1px solid #d9d8d3;vertical-align:top"
    out = [f'<table style="border-collapse:collapse;width:100%;{FONT and ""}'
           f'font-family:{FONT};font-size:9.5pt;color:#111;margin:0 0 6px">',
           "<tr>" + "".join(f'<th style="{th}">{inline(c)}</th>' for c in head) + "</tr>"]
    for r in body:
        out.append("<tr>" + "".join(f'<td style="{td}">{inline(c)}</td>'
                                    for c in r) + "</tr>")
    return "".join(out) + "</table>"


def build() -> str:
    lines = (OUT / "blindspots.md").read_text().splitlines()
    out: list[str] = []
    para: list[str] = []
    bullets: list[str] = []
    tbl: list[str] = []
    placed: set[str] = set()

    def emit_figs(text: str):
        for _n, stem in REF_RE.findall(text):
            if stem in META and stem not in placed:
                placed.add(stem)
                out.append(figure_html(stem))

    def flush_para():
        if para:
            text = " ".join(para)
            out.append(f'<p style="margin:0 0 11px;{BODY}">{inline(text)}</p>')
            emit_figs(text)
            para.clear()

    def flush_bullets():
        if bullets:
            items = "".join(f'<li style="margin:0 0 7px">{inline(b)}</li>'
                            for b in bullets)
            out.append(f'<ul style="margin:0 0 12px 20px;{BODY}">{items}</ul>')
            emit_figs(" ".join(bullets))
            bullets.clear()

    def flush_table():
        if tbl:
            out.append(table_html(tbl))
            tbl.clear()

    def flush_all():
        flush_para(); flush_bullets(); flush_table()

    for line in lines:
        s = line.rstrip()
        if s.startswith("<!--"):                 # generator markers, not content
            flush_all()
            continue
        if is_row(s):
            flush_para(); flush_bullets()
            tbl.append(s)
            continue
        flush_table()
        if s.startswith("- "):
            flush_para()
            bullets.append(s[2:])
            continue
        if s.startswith(("#", "---")) or not s.strip():
            flush_para(); flush_bullets()
        if not s.strip():
            continue
        if s.startswith("### "):
            out.append(f'<p style="margin:0 0 16px;font-family:{FONT};font-size:13pt;'
                       f'color:#555">{inline(s[4:])}</p>')
        elif s.startswith("## "):
            out.append(f'<h2 style="margin:26px 0 10px;font-family:{FONT};font-size:15pt;'
                       f'color:#111">{inline(s[3:])}</h2>')
        elif s.startswith("# "):
            out.append(f'<h1 style="margin:0 0 6px;font-family:{FONT};font-size:22pt;'
                       f'color:#111">{inline(s[2:])}</h1>')
        elif s.startswith("---"):
            out.append('<hr style="border:none;border-top:1px solid #ccc;margin:20px 0">')
        elif s.startswith("*") and s.endswith("*") and not s.startswith("**"):
            out.append(f'<p style="margin:0 0 14px;{SMALL}">'
                       f'<em>{inline(s.strip("*"))}</em></p>')
        else:
            para.append(s.strip())
    flush_all()

    for stem in NUM:                             # anything never referenced
        if stem not in placed:
            out.append(figure_html(stem))

    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Perception blind spots in Claude Haiku 4.5</title></head>"
            "<body style='margin:0 auto;padding:36px;max-width:720px;background:#fff'>"
            + "".join(out) + "</body></html>")


def main() -> int:
    doc = build()
    p = OUT / "paste_into_docs.html"
    p.write_text(doc)
    print(f"wrote {p}  ({len(doc)/1e6:.1f} MB, {doc.count('data:image/png;base64,')} "
          f"figures, {doc.count('<table')} tables)")
    print("Open it in a browser, select all, copy, and paste into Google Docs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
