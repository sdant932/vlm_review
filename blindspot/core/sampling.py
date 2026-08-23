"""Stratified sampling by primitive cell.

The pilot sampled CharXiv by *figure* with `random.sample(examples, limit)`.
Each figure contributes four of nineteen randomly-chosen question types, so a
200-figure sample produced per-question-type counts of 3 to 16 -- numbers with
no statistical content that nevertheless rendered as confident bars ("count
lines: 100%, n=3"). Sampling by the cell you intend to report is the fix.

Cells smaller than `per_cell` contribute their whole pool; the realised n is
returned so under-filled cells are reported rather than silently shipped.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Iterable

from blindspot.core.adapters import Example

# The slice each dataset is reported by. Keep in step with the report's axes:
# whatever you stratify on is what you can make claims about.
CELL_KEYS: dict[str, Callable[[Example], Any]] = {
    "charxiv": lambda e: e.meta.get("qid") or e.meta.get("split", "reasoning"),
    "infographicvqa": lambda e: tuple(e.meta.get("operation") or ["direct_lookup"])[0],
    "flowlearn_sim": lambda e: (e.meta.get("family"), e.meta.get("variant"),
                                e.meta.get("polarity")),
    "screenspot_pro": lambda e: (e.meta.get("ui_type"), _area_bucket(e)),
    "ai2d": lambda e: e.meta.get("qtype"),
    "slidevqa": lambda e: ("multi-page" if e.meta.get("multi_page") else "single-page",
                           "arithmetic" if e.meta.get("arithmetic") else "lookup"),
    "slidevqa_allpages": lambda e: ("multi-page" if e.meta.get("multi_page") else "single-page",
                                    "arithmetic" if e.meta.get("arithmetic") else "lookup"),
    "screenspot": lambda e: (e.meta.get("ui_type"), _area_bucket(e)),
}


def _area_bucket(e: Example) -> str:
    """Target size as it reaches the model, after the API's 1568px cap."""
    import math
    frac = e.meta.get("target_area_frac", 0) or 0
    side = math.sqrt(max(frac, 0) * 1568 * 882)
    for lim, name in ((12, "<12px"), (20, "12-20px"), (32, "20-32px"), (56, "32-56px")):
        if side < lim:
            return name
    return ">=56px"


def cell_key_for(dataset: str) -> Callable[[Example], Any]:
    return CELL_KEYS.get(dataset, lambda e: e.dataset)


def stratify(examples: Iterable[Example], key_fn: Callable[[Example], Any],
             per_cell: int = 300, seed: int = 0) -> tuple[list[Example], dict[Any, tuple[int, int]]]:
    """Return (sampled examples, {cell: (taken, pool_size)}).

    Deterministic for a given seed so a resumed run selects the same questions.
    """
    cells: dict[Any, list[Example]] = defaultdict(list)
    for e in examples:
        cells[key_fn(e)].append(e)

    rng = random.Random(seed)
    out: list[Example] = []
    realised: dict[Any, tuple[int, int]] = {}
    for cell in sorted(cells, key=lambda c: str(c)):
        pool = cells[cell]
        take = pool if len(pool) <= per_cell else rng.sample(pool, per_cell)
        out.extend(take)
        realised[cell] = (len(take), len(pool))
    return out, realised


def report_cells(dataset: str, realised: dict[Any, tuple[int, int]], min_n: int = 30) -> None:
    """Print realised n per cell and warn on any too small to interpret."""
    thin = [(c, t) for c, (t, _) in realised.items() if t < min_n]
    total = sum(t for t, _ in realised.values())
    print(f"  {dataset}: {total} questions across {len(realised)} cells")
    for cell, (took, pool) in sorted(realised.items(), key=lambda kv: kv[1][0]):
        mark = "  <-- thin" if took < min_n else ""
        print(f"    {str(cell):<44} n={took:>4} / pool {pool}{mark}")
    if thin:
        print(f"  !! {len(thin)} cell(s) below n={min_n}; report them as indicative only")
