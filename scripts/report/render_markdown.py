"""Render a markdown document to a self-contained HTML page.

The HTML is generated from the markdown, never edited by hand, so the two cannot
drift apart. Editing the markdown and re-running this is the whole workflow.

The parser is deliberately small -- it covers exactly what the document uses:
headings, paragraphs, ordered and unordered lists, tables, block quotes, fenced
code, horizontal rules, and inline bold / italic / code. No markdown library is
installed and one is not worth adding for this.

Usage
-----
    python -m scripts.report.render_markdown
    python -m scripts.report.render_markdown --src outputs/part3/part3.md --paste
"""

from __future__ import annotations

import argparse
import base64
import html
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CSS = """
:root{--bg:#fcfcfb;--fg:#1b1b19;--mut:#6a6a66;--line:#e4e4e0;--card:#fff;
      --code:#f4f4f1;--accent:#3a6ea5;--rule:#ebebe7}
@media (prefers-color-scheme:dark){
 :root{--bg:#151513;--fg:#ececea;--mut:#9d9d99;--line:#2d2d2a;--card:#1d1d1b;
       --code:#222220;--accent:#7aa8d6;--rule:#292926}}
*{box-sizing:border-box}
body{margin:0;padding:48px 24px 96px;background:var(--bg);color:var(--fg);
     font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
     -webkit-font-smoothing:antialiased}
.wrap{max-width:820px;margin:0 auto}
h1{font-size:30px;line-height:1.25;margin:0 0 28px;letter-spacing:-.015em}
h2{font-size:22px;margin:44px 0 14px;letter-spacing:-.01em;
   padding-bottom:7px;border-bottom:1px solid var(--rule)}
h3{font-size:17px;margin:30px 0 10px}
h4{font-size:15px;margin:22px 0 8px;color:var(--mut)}
p{margin:0 0 14px}
ul,ol{margin:0 0 16px;padding-left:24px}
li{margin:0 0 7px}
li>p{margin:0 0 7px}
strong{font-weight:650}
hr{border:0;border-top:1px solid var(--rule);margin:38px 0}
blockquote{margin:0 0 18px;padding:12px 18px;background:var(--card);
           border-left:3px solid var(--accent);border-radius:0 6px 6px 0;
           color:var(--mut)}
blockquote p{margin:0}
blockquote p+p{margin-top:8px}
code{background:var(--code);padding:1.5px 5px;border-radius:4px;
     font:13.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--code);border:1px solid var(--line);border-radius:8px;
    padding:14px 16px;overflow-x:auto;margin:0 0 18px}
pre code{background:none;padding:0;font-size:13px;line-height:1.55}
.fig{margin:0 0 20px;border:1px solid var(--line);border-radius:8px;
     padding:12px;background:var(--card)}
.fig img{display:block;max-width:100%;height:auto;border-radius:4px}
.fig figcaption{font-size:12.5px;color:var(--mut);margin-top:9px}
table{border-collapse:collapse;width:100%;margin:0 0 20px;font-size:14.5px;
      display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:8px 11px;text-align:left;
      vertical-align:top}
th{background:var(--card);font-weight:600}
tbody tr:nth-child(even){background:color-mix(in srgb,var(--card) 55%,transparent)}
"""

INLINE_CODE = re.compile(r"`([^`]+)`")
BOLD = re.compile(r"\*\*([^*]+)\*\*")
ITAL = re.compile(r"(?<![\*\w])\*([^*\n]+)\*(?!\*)")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def inline(text: str) -> str:
    """Escape, then re-introduce the small set of inline marks we allow.

    Backslash-escaped characters are pulled out before anything else. Without
    that, a literal asterisk written as `b\\*` inside a bold span terminates the
    span early and the `**` leaks into the page as text -- which is exactly what
    the loss table did.
    """
    slots: list[str] = []

    def stash(s: str) -> str:
        slots.append(s)
        return f"\x00{len(slots) - 1}\x00"

    # 1. escaped punctuation becomes a literal, invisible to every rule below
    text = re.sub(r"\\([\\*_|`\[\]])", lambda m: stash(html.escape(m.group(1))), text)
    # 2. code spans are literal too
    text = INLINE_CODE.sub(lambda m: stash(f"<code>{html.escape(m.group(1))}</code>"),
                           text)
    text = html.escape(text)
    text = LINK.sub(r'<a href="\2">\1</a>', text)
    text = BOLD.sub(r"<strong>\1</strong>", text)
    text = ITAL.sub(r"<em>\1</em>", text)
    for i, c in enumerate(slots):
        text = text.replace(f"\x00{i}\x00", c)
    return text


# --- Google Docs paste build -------------------------------------------------
#
# Docs keeps almost no CSS: a <style> block is dropped, class selectors are
# dropped, and custom properties certainly are. Anything that must survive the
# clipboard has to be an inline style attribute on the element itself, and the
# document structure has to be carried by real h1/h2/p/ul/table elements rather
# than by classes. Local image paths are not followed either, which is why every
# figure is already a base64 data URI.
FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Calibri, "
        "'Helvetica Neue', Arial, sans-serif")
MONO = "'SF Mono', Menlo, Consolas, monospace"

PASTE_STYLE = {
    "h1": "font-family:" + FONT + ";font-size:20pt;color:#111;margin:0 0 14pt",
    "h2": "font-family:" + FONT + ";font-size:15pt;color:#111;margin:20pt 0 8pt",
    "h3": "font-family:" + FONT + ";font-size:12.5pt;color:#111;margin:14pt 0 6pt",
    "h4": "font-family:" + FONT + ";font-size:11pt;color:#444;margin:12pt 0 5pt",
    "p": "font-family:" + FONT + ";font-size:11pt;line-height:1.5;color:#111;margin:0 0 9pt",
    "ul": "margin:0 0 9pt;padding-left:20pt",
    "ol": "margin:0 0 9pt;padding-left:20pt",
    "li": "font-family:" + FONT + ";font-size:11pt;line-height:1.5;color:#111;margin:0 0 5pt",
    "blockquote": "font-family:" + FONT + ";font-size:11pt;color:#444;margin:0 0 10pt;padding-left:10pt;border-left:3px solid #bbb",
    "pre": "font-family:" + MONO + ";font-size:9pt;line-height:1.45;background:#f4f4f2;border:1px solid #ddd;padding:8pt;margin:0 0 10pt;white-space:pre-wrap",
    "code": "font-family:" + MONO + ";font-size:9.5pt;background:#f4f4f2",
    "table": "border-collapse:collapse;width:100%;margin:0 0 12pt",
    "th": "font-family:" + FONT + ";font-size:9.5pt;color:#111;border:1px solid #ccc;padding:5pt 7pt;text-align:left;background:#f2f2f0;font-weight:bold",
    "td": "font-family:" + FONT + ";font-size:9.5pt;color:#111;border:1px solid #ccc;padding:5pt 7pt;text-align:left;vertical-align:top",
    "figure": "margin:0 0 14pt",
    "figcaption": "font-family:" + FONT + ";font-size:9pt;color:#666;margin-top:4pt",
    "img": "max-width:100%;height:auto",
}


def to_paste(body: str) -> str:
    """Push every style inline so it survives the clipboard."""
    body = body.replace('<figure class="fig">', "<figure>")
    for tag, style in PASTE_STYLE.items():
        body = re.sub(rf"<{tag}(?=[ >])", f'<{tag} style="{style}"', body)
        body = body.replace(f"<{tag}>", f'<{tag} style="{style}">')
    return body


def figure(alt: str, src: str, base: Path | None) -> str:
    """Inline a local image as base64 so the page stays one self-contained file.

    A remote or missing path is left as a plain link rather than silently
    producing a broken image.
    """
    path = (base / src) if base else Path(src)
    if not path.exists():
        return f'<figure><a href="{html.escape(src)}">{inline(alt)}</a></figure>'
    data = base64.b64encode(path.read_bytes()).decode()
    kind = "png" if path.suffix.lower() == ".png" else "jpeg"
    return (f'<figure class="fig">'
            f'<img src="data:image/{kind};base64,{data}" alt="{html.escape(alt)}">'
            f'<figcaption>{inline(alt)}</figcaption></figure>')


def split_row(line: str) -> list[str]:
    """Split a table row on unescaped pipes."""
    cells, cur, i = [], "", 0
    line = line.strip().strip("|")
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line):
            cur += line[i:i + 2]
            i += 2
        elif line[i] == "|":
            cells.append(cur)
            cur = ""
            i += 1
        else:
            cur += line[i]
            i += 1
    cells.append(cur)
    return [c.strip() for c in cells]


def render(md: str, base: Path | None = None) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # blank
        if not stripped:
            i += 1
            continue

        # fenced code
        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(buf))}</code></pre>")
            continue

        # horizontal rule
        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            out.append("<hr>")
            i += 1
            continue

        # heading
        m = re.match(r"(#{1,6})\s+(.*)$", stripped)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2).strip())}</h{lvl}>")
            i += 1
            continue

        # standalone image -> figure
        m = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if m:
            out.append(figure(m.group(1), m.group(2), base))
            i += 1
            continue

        # table: a pipe row followed by an alignment row
        if stripped.startswith("|") and i + 1 < n and re.fullmatch(
                r"\|?[\s:|-]+\|?", lines[i + 1].strip()) and "-" in lines[i + 1]:
            head = split_row(stripped)
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i].strip()))
                i += 1
            th = "".join(f"<th>{inline(c)}</th>" for c in head)
            body = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                for r in rows)
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>")
            continue

        # block quote
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            inner = render("\n".join(buf), base)
            out.append(f"<blockquote>{inner}</blockquote>")
            continue

        # Lists. A blank line ENDS the list: each marker opens its own list, and
        # an ordered one that resumes carries `start="N"` so the numbering is
        # continuous across whatever block interrupted it (in this document, a
        # table indented under the item). Items are not grouped, because the
        # source separates every item with a blank line.
        #
        # An item holding more than one block wraps each in <p> -- that is what
        # the `li>p` rule in CSS styles. A single-block item is left bare.
        m = re.match(r"(\s*)([-*+]|\d+\.)\s+(.*)$", line)
        if m:
            ordered = bool(re.fullmatch(r"\d+\.", m.group(2)))
            tag = "ol" if ordered else "ul"
            num = int(m.group(2)[:-1]) if ordered else 1
            blocks = [m.group(3)]
            i += 1
            cont = _continuation(lines, i, n)
            blocks[0] = _para_join([blocks[0]] + cont)
            i += len(cont)
            # further indented blocks belonging to this same item
            while i < n:
                j = i
                while j < n and not lines[j].strip():
                    j += 1
                if j >= n or _breaks_para(lines[j]) or not lines[j].startswith("   "):
                    break
                more = _continuation(lines, j, n)
                if not more:
                    break
                blocks.append(_para_join(more))
                i = j + len(more)
            inner = ("".join(f"<p>{b}</p>" for b in blocks)
                     if len(blocks) > 1 else blocks[0])
            li = f"<li>{inner}</li>"
            if not ordered:
                # bullets are grouped: consume the run of sibling markers
                while i < n:
                    j = i
                    while j < n and not lines[j].strip():
                        j += 1
                    mm = re.match(r"(\s*)([-*+])\s+(.*)$", lines[j]) if j < n else None
                    if not mm:
                        break
                    i = j + 1
                    c = _continuation(lines, i, n)
                    i += len(c)
                    li += f"<li>{_para_join([mm.group(3)] + c)}</li>"
                out.append(f"<ul>{li}</ul>")
                continue
            attr = f' start="{num}"' if num != 1 else ""
            out.append(f"<ol{attr}>{li}</ol>")
            continue

        # paragraph: this line plus any soft-wrapped continuation
        buf = [stripped]
        i += 1
        cont = _continuation(lines, i, n)
        buf += cont
        i += len(cont)
        out.append(f"<p>{_para_join(buf)}</p>")

    return "".join(out)


def _breaks_para(line: str) -> bool:
    """Does this line start a new block, rather than continue the current one?"""
    s = line.strip()
    if not s:
        return True
    return bool(
        s.startswith(("#", ">", "|", "```", "!["))
        or re.match(r"([-*+]|\d+\.)\s+", s)
        or re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", s))


def _continuation(lines: list[str], i: int, n: int) -> list[str]:
    """Gather the soft-wrapped lines that belong to the block starting before i."""
    buf: list[str] = []
    while i < n and not _breaks_para(lines[i]):
        buf.append(lines[i].strip())
        i += 1
    return buf


def _para_join(buf: list[str]) -> str:
    """Join wrapped source lines into one rendered paragraph string."""
    return inline(" ".join(x.strip() for x in buf if x.strip()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/part3/part3.md")
    ap.add_argument("--out", default=None)
    ap.add_argument("--paste", action="store_true",
                    help="also write a Google-Docs-pasteable copy")
    a = ap.parse_args()

    src = REPO / a.src
    out = Path(a.out) if a.out else src.with_suffix(".html")
    md = src.read_text(encoding="utf-8")

    title = "Document"
    for line in md.split("\n"):
        if line.strip().startswith("# "):
            title = line.strip()[2:].strip()
            break

    body = render(md, base=src.parent)
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style>"
            f"<body><div class='wrap'>{body}</div></body></html>")
    out.write_text(page, encoding="utf-8")
    print(f"{src.name} -> {out}  ({out.stat().st_size // 1024} KB)")

    if a.paste:
        pst = out.with_name(out.stem + "_paste.html")
        pst.write_text(
            "<!doctype html><html lang='en'><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title>"
            "<body style='margin:0;padding:24px;background:#fff'>"
            f"<div style='max-width:720px;margin:0 auto'>{to_paste(body)}</div>"
            "</body></html>", encoding="utf-8")
        print(f"{pst.name} -> {pst}  ({pst.stat().st_size // 1024} KB)  "
              "open, select all, copy, paste into Google Docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
