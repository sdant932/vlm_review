"""Small statistics and grid helpers shared across the analysis layer.

These lived in `reporting/cause_pages.py`, which is a 3,000-line HTML renderer.
Five analysis modules imported six functions from it, which made
`blindspot.analysis` depend on `blindspot.reporting` -- backwards, and enough
to pull a page builder into memory just to compute a confidence interval.

The definitions here are unchanged. `cause_pages` re-imports them, so its
behaviour is identical to before the move.
"""

from __future__ import annotations

import math


def is_na(v) -> bool:
    """True for the 'Not Applicable' gold CharXiv uses for inapplicable questions."""
    return "not applicable" in str(v).strip().lower()


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval; shown wherever an n is small enough to matter."""
    if n == 0:
        return (0.0, 1.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def quantiles(rows, keyfn, k=5):
    """Split `rows` into k equal-count bins by `keyfn`, dropping rows keyed None.

    Returns (low, high, rows) per bin, so a caller can label the bin by its
    realised range rather than a nominal cut point.
    """
    rs = sorted((r for r in rows if keyfn(r) is not None), key=keyfn)
    out = []
    for i in range(k):
        chunk = rs[i * len(rs) // k:(i + 1) * len(rs) // k]
        if chunk:
            out.append((keyfn(chunk[0]), keyfn(chunk[-1]), chunk))
    return out


# --- coarse g x g grid over the normalized [0,1] image plane.
# Used to ask "did the prediction land in the right region?" when it missed the
# box outright -- a near miss and a miss to the far corner are different failures.

def cell_of(x, y, g):
    """Grid cell containing normalized point (x, y). Clamps at the far edge."""
    return (min(int(x * g), g - 1), min(int(y * g), g - 1))


def centre_cell(box, g):
    """Grid cell containing the centre of normalized box [x0,y0,x1,y1]."""
    return cell_of((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, g)


def bbox_cells(box, g):
    """Every grid cell a normalized box touches, not just the one holding its centre."""
    x0, y0, x1, y1 = box
    return {(i, j)
            for i in range(min(int(x0 * g), g - 1), min(int(x1 * g), g - 1) + 1)
            for j in range(min(int(y0 * g), g - 1), min(int(y1 * g), g - 1) + 1)}
