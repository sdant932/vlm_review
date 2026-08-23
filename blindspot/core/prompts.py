"""Prompt construction and response schemas, keyed by `answer_type`.

Two rules drive everything here:

1. CharXiv descriptive prompts already carry their own answer-format rules
   (vendored verbatim). Adding our own instructions on top would change the
   task and make the numbers incomparable to CharXiv's published setup, so
   those questions are sent through untouched.

2. Answers come back via structured outputs rather than a regex over free
   text. With thinking enabled the model reasons in its thinking block and
   emits only the schema-conforming JSON, which removes the whole class of
   parser bugs HANDOFF.md warns about.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from PIL import Image

from blindspot.core.adapters import Example

# Haiku 4.5 downsizes anything larger to roughly this long edge (~1568 image
# tokens), measured directly against count_tokens. Sending native-resolution
# 4K screenshots therefore buys nothing but upload time -- but we still send
# native by default so the downscale ablation has something to compare against.
HAIKU_MAX_EDGE = 1568

# Hard API ingestion limits, hit for real during the pilot:
#   - a 2534x8369 InfographicVQA page -> 400 "image dimensions exceed max allowed
#     size: 8000 pixels"
#   - 5120x2880 Retina ScreenSpot-Pro screenshots -> 400 "image exceeds 10 MB"
# These reject the request outright, before the model sees anything. Since Haiku
# downscales to ~1568px regardless, shrinking to fit costs no model-visible
# fidelity -- but skipping it silently drops the largest, most interesting images
# from the eval, which would bias exactly the cases we care about.
API_MAX_DIM = 8000
API_MAX_B64_BYTES = 9_500_000  # margin under the 10 MB ceiling

MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
               ".gif": "image/gif", ".webp": "image/webp"}

SCHEMAS: dict[str, dict] = {
    "span": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    },
    "bbox": {
        "type": "object",
        "properties": {
            "x0": {"type": "integer"}, "y0": {"type": "integer"},
            "x1": {"type": "integer"}, "y1": {"type": "integer"},
        },
        "required": ["x0", "y0", "x1", "y1"],
        "additionalProperties": False,
    },
    "point": {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
        },
        "required": ["x", "y"],
        "additionalProperties": False,
    },
    # Constrained to the enum so a hedged "it appears there may be" can never
    # reach the scorer -- the model must commit.
    "boolean": {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
        "required": ["answer"],
        "additionalProperties": False,
    },
    # No `minimum`: structured outputs reject numerical constraints
    # ("For 'integer' type, property 'minimum' is not supported"). The count
    # scorer treats a negative as simply wrong, which is the right behaviour.
    # Multiple choice removes the answer-expression confound entirely: the model
    # picks rather than phrases, so a correct reading cannot be scored wrong for
    # wording -- which is the failure mode that costs InfographicVQA ~8 points.
    "choice": {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},
        "required": ["answer"],
        "additionalProperties": False,
    },
    "count": {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    },
}

SPAN_INSTRUCTION = (
    "Answer the question using only what is visible in the image. "
    "Respond with the shortest exact answer -- a value, name, or phrase copied "
    "from the image. Do not explain, and do not restate the question."
)

BOOLEAN_INSTRUCTION = (
    "Answer strictly yes or no, based only on what is drawn in the image.\n"
    "Commit to one answer even if you are uncertain.\n\n"
)

CHOICE_INSTRUCTION = (
    "Answer with the single letter of the correct option.\n"
    "Base your answer only on what is shown in the diagram.\n\n"
)

COUNT_INSTRUCTION = (
    "Count carefully and answer with a single whole number.\n"
    "Count only what is actually drawn in the image.\n\n"
)

# Coordinates are requested in a 0-1000 normalized space rather than pixels:
# the model never sees the native resolution (the API downscales first), so
# asking for pixel coordinates in an unknown coordinate space would inject an
# avoidable source of error into a localization measurement.
POINT_INSTRUCTION = (
    "Locate the described UI element in the screenshot and return the point at "
    "its center.\n"
    "Use a normalized coordinate system where x=0 is the left edge, x=1000 the "
    "right edge, y=0 the top edge, and y=1000 the bottom edge.\n"
    "Always return your single best guess, even if you are uncertain.\n\n"
    "Element: "
)


def encode_image(path: str, max_edge: int | None = None) -> tuple[str, str, tuple[int, int], bool]:
    """Return (base64, media_type, (width, height) as sent, was_downscaled).

    `max_edge` pre-downscales client-side. That is the lever for the resolution
    ablation: if scores are unchanged at max_edge=1568, a localization failure is
    perceptual rather than an artifact of the API resizing a 4K screenshot out
    from under the model.

    With `max_edge=None` the original is sent untouched *unless* it would breach
    an API ingestion limit, in which case it is shrunk just enough to pass.
    """
    p = Path(path)

    if max_edge is None:
        with Image.open(p) as im:
            size = im.size
        raw = p.read_bytes()
        b64 = base64.b64encode(raw).decode()
        if max(size) <= API_MAX_DIM and len(b64) <= API_MAX_B64_BYTES:
            return b64, MEDIA_TYPES[p.suffix.lower()], size, False
        # Too big to ingest: fall through and shrink to fit.

    with Image.open(p) as im:
        im = im.convert("RGB")
        original = im.size
        target = min(max_edge or API_MAX_DIM, API_MAX_DIM)
        if max(im.size) > target:
            scale = target / max(im.size)
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                           Image.LANCZOS)

        # Shrink until the encoded payload fits, rather than guessing a quality.
        for _ in range(8):
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode()
            if len(b64) <= API_MAX_B64_BYTES:
                return b64, "image/jpeg", im.size, im.size != original
            im = im.resize((max(1, int(im.width * 0.8)), max(1, int(im.height * 0.8))), Image.LANCZOS)

        raise ValueError(f"{path}: cannot fit under API image limits")


def prompt_text(ex: Example) -> str:
    """The exact text block sent alongside the image.

    Split out of build_request so tooling can show the real prompt without
    paying to re-encode the image -- and so the two can never drift apart.

    `meta["prompt_override"]` replaces the whole block. It exists so a prompt
    ablation can vary the wording while every other part of the request --
    image encoding, schema, model, thinking budget -- stays byte-identical.
    """
    ov = ex.meta.get("prompt_override")
    if ov:
        return ov
    if ex.answer_type == "point":
        return POINT_INSTRUCTION + ex.question
    if ex.answer_type == "boolean":
        return BOOLEAN_INSTRUCTION + ex.question
    if ex.answer_type == "count":
        return COUNT_INSTRUCTION + ex.question
    if ex.answer_type == "choice":
        opts = "\n".join(f"{k}. {v}" for k, v in zip("ABCD", ex.meta.get("options", [])))
        return f"{CHOICE_INSTRUCTION}{ex.question}\n\n{opts}"
    if ex.dataset == "svg_localization":
        return ex.question  # already self-contained and states its own answer format
    if ex.dataset == "charxiv" and ex.meta.get("split") == "descriptive":
        return ex.question  # vendored template already specifies the answer format
    return f"{ex.question}\n\n{SPAN_INSTRUCTION}"


def build_request(ex: Example, max_edge: int | None = None) -> tuple[list[dict], dict, list[tuple[int, int]], bool]:
    """Return (message content blocks, response schema, as-sent sizes, any_downscaled)."""
    content: list[dict] = []
    sizes: list[tuple[int, int]] = []
    downscaled = False
    for path in ex.images:
        b64, media_type, size, shrunk = encode_image(path, max_edge)
        sizes.append(size)
        downscaled |= shrunk
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        })

    content.append({"type": "text", "text": prompt_text(ex)})
    return content, SCHEMAS[ex.answer_type], sizes, downscaled


def parse_response(ex: Example, raw: str) -> Any:
    """Convert the schema-conforming JSON into the value the scorer expects."""
    import json

    obj = json.loads(raw)
    if ex.answer_type == "point":
        return (obj["x"] / 1000.0, obj["y"] / 1000.0)
    if ex.answer_type == "bbox":
        return (obj["x0"] / 1000.0, obj["y0"] / 1000.0,
                obj["x1"] / 1000.0, obj["y1"] / 1000.0)
    if ex.answer_type == "count":
        return int(obj["answer"])
    if ex.answer_type == "boolean":
        return str(obj["answer"]).strip().lower()
    if ex.answer_type == "choice":
        return str(obj["answer"]).strip().upper()
    return obj["answer"]
