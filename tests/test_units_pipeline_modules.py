"""Function-level tests for the five pipeline modules, in five sections.

    A. blindspot.generate           -- scenes, questions, audit, example pages
    B. blindspot.generate_finetune  -- the ladder, ExactInk, samples, audit
    C. blindspot.report_finetune    -- the sample gallery (byte-identical)
    D. blindspot.render_markdown    -- the tiny markdown parser
    E. blindspot.report             -- the non-API helpers and the three guards

WHY THIS FILE EXISTS
--------------------
Several of these modules were RECONSTRUCTED after the originals were lost. The
surviving artifacts under `data/` and `outputs/` are therefore the
specification, not a by-product: they are the only remaining statement of what
the code used to do. These tests pin the reconstruction against them.

Two of them are exact regression targets and are marked as such:

  * `test_gallery_reproduces_the_shipped_artifact_byte_for_byte` rebuilds
    `outputs/finetune/gallery.html` from `data/sft_bbox/sft_bbox_20.jsonl` and
    compares md5 and size. Every import, every resize rule and every PNG
    encoder setting in the gallery path is inside that hash, so this one
    assertion protects the whole rebuild.
  * `test_exactink_reproduces_the_shipped_sft_boxes` re-measures all twenty
    shipped `target.box_px`, `correction.layout_box_px` and `grew_px_lrtb`
    values. This was verified at 20/20 when the module was rebuilt and it must
    stay that way -- it is the check that says the rebuilt generator still
    draws the same pixels the shipped dataset was measured on.

Everything here is offline, deterministic and writes only into pytest's
`tmp_path`. The shipped artifacts are opened read-only. Generation is kept to
six scenes so the whole file runs in a few seconds; nothing here is slow enough
to need marking.
"""

from __future__ import annotations

import collections
import contextlib
import copy
import hashlib
import html as _html
import io
import json
import re
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from blindspot import generate as G
from blindspot import generate_finetune as GF
from blindspot import render_markdown as RM
from blindspot import report_finetune as RF

ROOT = Path(__file__).resolve().parents[1]

# Read-only reference artifacts. Nothing in this file writes into these paths.
SFT_RECORDS = ROOT / "data" / "sft_bbox" / "sft_bbox_20.jsonl"
LADDER = ROOT / "data" / "svgloc_mr"
SVGLOC = ROOT / "data" / "svg_localization"
SHIPPED_GALLERY = ROOT / "outputs" / "finetune" / "gallery.html"
PART3_MD = ROOT / "outputs" / "part3" / "part3.md"
PART3_HTML = ROOT / "outputs" / "part3" / "part3.html"

N_SCENES = 6            # small enough to run in seconds, wide enough to vary


# --------------------------------------------------------------- helpers

def _run(main_fn, argv):
    """Call a module `main(argv)` with stdout captured. Returns (rc, stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main_fn(argv)
    return rc, buf.getvalue()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _digests(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): _md5(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """Two independent generations of the same tiny set, plus derived questions.

    Both directories are built from identical arguments so the determinism test
    can compare them; every other Subject A test reads the first one.
    """
    base = tmp_path_factory.mktemp("svgloc")
    a, b = base / "a", base / "b"
    for out in (a, b):
        rc, _out = _run(G.main, ["scenes", "--count", str(N_SCENES),
                                 "--seed", "17", "--complexity", "4",
                                 "--out", str(out)])
        assert rc == 0
    rc, _out = _run(G.main, ["questions", "--data", str(a)])
    assert rc == 0
    return a, b


# =============================================================================
# A. blindspot.generate -- scenes | questions | audit | examples
# =============================================================================
"""The synthetic set is the study's own instrument, so its two load-bearing
properties are pinned here: the same seed produces the same bytes, and the
vector source and the raster cannot describe different pictures.
"""


def test_generating_twice_at_the_same_seed_is_byte_identical(generated):
    """The property the whole dataset rests on.

    Manifests, scenes.jsonl, SVGs and PNGs must all match: a PNG that differs
    by one pixel between runs means the gold boxes measured off it are not
    reproducible, and every published number is measured off those boxes.
    """
    a, b = generated
    # `a` also carries the derived word_mc/counting output; compare the scene
    # products only.
    da = {k: v for k, v in _digests(a).items()
          if not k.startswith(("word_mc/", "counting/", "verify/", "examples/"))}
    db = _digests(b)

    assert set(da) == set(db)
    assert {"manifest.jsonl", "scenes.jsonl"} <= set(da)
    assert any(k.endswith(".png") for k in da)
    assert any(k.endswith(".svg") for k in da)

    differing = sorted(k for k in da if da[k] != db[k])
    assert differing == []


def test_svg_and_png_describe_the_same_geometry(generated):
    """One list of drawing primitives feeds both outputs, so they cannot drift.

    Checked three ways: the SVG canvas is the canvas the manifest reports, the
    SVG's <text id="tN"> elements carry exactly the strings scenes.jsonl
    records at index N (and so exactly the manifest's target texts), and each
    SVG text anchor, scaled into raster space, lands on the ink box measured
    off the PNG.
    """
    data, _b = generated
    scenes = {s["graph_id"]: s for s in _jsonl(data / "scenes.jsonl")}
    rows = _jsonl(data / "manifest.jsonl")
    assert rows and scenes

    text_re = re.compile(r'<text id="t(\d+)"[^>]*x="([-\d.]+)" y="([-\d.]+)"'
                         r'[^>]*>(.*?)</text>')
    svg_texts = {}
    for gid, sc in scenes.items():
        svg = (data / sc["svg"]).read_text()
        head = re.search(r'width="(\d+)" height="(\d+)" viewBox="0 0 (\d+) (\d+)"', svg)
        assert head, f"g{gid:04d}.svg has no canvas header"
        w, h = int(head.group(1)), int(head.group(2))
        assert [w, h] == sc["canvas"]
        assert [int(head.group(3)), int(head.group(4))] == sc["canvas"]

        found = {int(i): (float(x), float(y), _html.unescape(t))
                 for i, x, y, t in text_re.findall(svg)}
        # index i in the SVG is index i in scenes.jsonl is target_idx in the
        # manifest -- that join is the only thing tying the three files together
        assert sorted(found) == [t["idx"] for t in sc["texts"]]
        for t in sc["texts"]:
            assert found[t["idx"]][2] == t["text"]
        svg_texts[gid] = found

    for r in rows:
        w, h = r["image_px"]
        with Image.open(data / r["image"]) as im:
            assert list(im.size) == [w, h]
        sc = scenes[r["graph_id"]]
        assert [round(sc["canvas"][0] * r["scale"]),
                round(sc["canvas"][1] * r["scale"])] == [w, h]

        x0, y0, x1, y1 = r["gold_bbox_px"]
        assert 0 <= x0 < x1 <= w, r["uid"]
        assert 0 <= y0 < y1 <= h, r["uid"]
        assert r["target_text"] == sc["texts"][r["target_idx"]]["text"]

        if r["qtype"] != "point":
            continue
        vx, vy, vtext = svg_texts[r["graph_id"]][r["target_idx"]]
        assert vtext == r["target_text"]
        ix0, iy0, ix1, iy1 = r["text_ink_bbox_px"]
        s = r["scale"]
        # The vector anchor point lands inside the raster ink box (within one
        # pixel of rounding). If the two paths drew different pictures this is
        # the first thing that would move.
        assert ix0 - 2 <= vx * s <= ix1 + 2, r["uid"]
        assert iy0 - 2 <= vy * s <= iy1 + 2, r["uid"]


def test_audit_reports_zero_consistency_errors_on_a_fresh_set(generated):
    data, _b = generated
    out = data / "verify" / "index.html"
    rc, text = _run(G.main, ["audit", "--data", str(data), "--out", str(out)])
    assert rc == 0
    assert "consistency errors: 0" in text
    assert out.exists()
    assert G.check_manifest(_jsonl(data / "manifest.jsonl")) == []


def test_audit_fails_on_a_corrupted_manifest(generated, tmp_path):
    """Otherwise the audit is a rubber stamp.

    One gold box is pushed past the right edge of its own image in a temp copy
    of the manifest; nothing else changes. The audit must notice, name the uid,
    and exit non-zero.
    """
    data, _b = generated
    rows = _jsonl(data / "manifest.jsonl")
    assert G.check_manifest(rows) == []     # so the failure below is attributable
    victim = next(r for r in rows if r["qtype"] == "point")

    bad = copy.deepcopy(rows)
    target = next(r for r in bad if r["uid"] == victim["uid"])
    target["gold_bbox_px"] = [target["gold_bbox_px"][0], target["gold_bbox_px"][1],
                              target["image_px"][0] + 40, target["gold_bbox_px"][3]]

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in bad))

    problems = G.check_manifest(bad)
    assert problems, "check_manifest accepted a box outside its image"
    assert any(p["uid"] == victim["uid"] for p in problems)

    rc, text = _run(G.main, ["audit", "--data", str(corrupt),
                             "--out", str(tmp_path / "verify.html")])
    assert rc == 1
    assert "consistency errors: 0" not in text
    assert victim["uid"] in (tmp_path / "verify.html").read_text()


def test_word_mc_distractors_appear_nowhere_in_the_scene(generated):
    """A distractor that occurs anywhere in the figure -- including inside a
    longer word, a title or a footnote -- makes the question unanswerable, so
    the rule is substring absence over the whole scene blob, not label
    inequality."""
    data, _b = generated
    rows = _jsonl(data / "word_mc" / "manifest.jsonl")
    assert rows, "the derived word_mc set is empty"
    scenes = {s["graph_id"]: s for s in _jsonl(data / "scenes.jsonl")}

    for r in rows:
        blob = G.scene_blob(scenes[r["graph_id"]])
        assert len(r["options"]) == 4
        assert len(set(r["options"])) == 4
        assert r["answer"] in "ABCD"
        assert r["options"]["ABCD".index(r["answer"])] == r["answer_text"]
        assert r["answer_text"].lower() in blob
        for d in r["distractors"]:
            assert d.lower() not in blob, f"{r['uid']}: distractor {d!r} is in the scene"
        assert r["answer_text"] not in r["distractors"]


def test_counting_gold_comes_from_the_record_and_survives_the_cross_check(generated):
    """Gold is read from `facts` (the semantic record captured at build time),
    never from counting marks in the raster, and each published count is then
    re-derived and cross-checked against the labels actually drawn."""
    data, _b = generated
    rows = _jsonl(data / "counting" / "manifest.jsonl")
    assert rows, "the derived counting set is empty"
    scenes = {s["graph_id"]: s for s in _jsonl(data / "scenes.jsonl")}

    for r in rows:
        scene = scenes[r["graph_id"]]
        specs = {q: (gold, check) for q, gold, check in G.counting_specs(scene)}
        assert r["question"] in specs, f"{r['uid']}: question not derivable from facts"
        gold, check = specs[r["question"]]
        assert r["answer"] == gold == r["true_count"]
        assert G.cross_check(scene, check, gold) is None, r["uid"]

    # Failures are reported rather than silently dropped.
    failures = json.loads((data / "counting" / "cross_check_failures.json").read_text())
    published = {(r["graph_id"], r["question"]) for r in rows}
    for w in failures:
        assert (w["graph_id"], w["question"]) not in published


def test_every_target_meets_the_wcag_threshold_for_its_font_size(generated):
    """AA is 4.5:1 for normal text and 3.0:1 at 24px and above, and the
    background is the one measured under the label rather than the theme's
    nominal colour."""
    data, _b = generated
    rows = _jsonl(data / "manifest.jsonl")
    assert rows
    for r in rows:
        need = G.required_contrast(r["font_px"])
        assert r["target_contrast"] >= need, f"{r['uid']}: {r['target_contrast']} < {need}"
        assert r["font_px"] >= G.MIN_LEGIBLE_PX


def test_required_contrast_switches_at_the_aa_large_text_boundary():
    assert G.required_contrast(G.LARGE_PX - 1) == G.WCAG_NORMAL == 4.5
    assert G.required_contrast(G.LARGE_PX) == G.WCAG_LARGE == 3.0


def test_the_example_pages_build_from_the_generated_set(generated):
    data, _b = generated
    rc, text = _run(G.main, ["examples", "--data", str(data),
                             "--out", str(data / "examples" / "index.html"),
                             "--per-type", "1"])
    assert rc == 0
    assert (data / "examples" / "index.html").read_text().startswith("<!doctype")

    rc, text = _run(G.main, ["examples-derived", "--data", str(data), "--per-type", "2"])
    assert rc == 0
    for which in ("word_mc", "counting"):
        assert (data / which / "examples.html").exists()


# =============================================================================
# B. blindspot.generate_finetune -- ladder | ExactInk | samples | audit
# =============================================================================
"""The ladder table and the ink boxes are both exact: the shipped
`data/svgloc_mr` and `data/sft_bbox` state what they must be, and these tests
read them rather than restating them.
"""


@pytest.fixture(scope="module")
def ladder_rows():
    if not (LADDER / "manifest.jsonl").exists():
        pytest.skip("data/svgloc_mr is not present")
    return _jsonl(LADDER / "manifest.jsonl")


@pytest.fixture(scope="module")
def sft_records():
    if not SFT_RECORDS.exists():
        pytest.skip("data/sft_bbox/sft_bbox_20.jsonl is not present")
    return _jsonl(SFT_RECORDS)


def test_the_ladder_is_six_aspects_by_four_rungs():
    assert len(GF.ASPECTS) == 6
    assert list(GF.RUNGS) == ["r55", "r70", "r85", "r100"]
    assert GF.RUNGS["r100"] == 1.00
    assert (GF.MAX_EDGE, GF.MAX_PIXELS) == (1568, 1_150_000)


def test_r100_is_the_largest_canvas_the_api_delivers_untouched():
    """Both caps bind, and neither may be exceeded by a pixel: a 1569px edge
    would make the API downscale the whole set, which is the one thing this
    dataset must not do."""
    for name, ratio in GF.ASPECTS:
        w, h = GF.canvas_for(ratio)
        assert max(w, h) <= GF.MAX_EDGE, name
        assert w * h <= GF.MAX_PIXELS, name
        assert list(G.effective_size(w, h)) == [w, h], name
        # one of the two caps is actually reached -- this is the *largest* such
        # canvas, not merely a legal one
        assert max(w, h) == GF.MAX_EDGE or w * h > GF.MAX_PIXELS * 0.99, name
    assert GF.canvas_for(7 / 3) == (1568, 671)          # the edge cap binds
    assert GF.canvas_for(G.BASE_W / G.BASE_H) == (1347, 853)


def test_all_twenty_four_ladder_canvases_match_the_shipped_manifest(ladder_rows):
    """The exact table, asserted against data/svgloc_mr rather than restated.

    r100 is `canvas_for(ratio)`; every lower rung is round(scale x r100).
    """
    delivered = collections.defaultdict(set)
    for r in ladder_rows:
        delivered[(r["aspect"], r["rung"])].add(tuple(r["image_px"]))

    checked = 0
    for name, ratio in GF.ASPECTS:
        base = GF.canvas_for(ratio)
        for rung, scale in GF.RUNGS.items():
            want = (round(base[0] * scale), round(base[1] * scale))
            got = delivered[(name, rung)]
            assert got == {want}, f"{name}/{rung}: manifest has {got}, expected {want}"
            checked += 1
    assert checked == 24


def test_no_ladder_row_is_downscaled_by_the_api(ladder_rows):
    assert not any(r["downscaled_by_api"] for r in ladder_rows)
    for r in ladder_rows:
        assert r["effective_px"] == r["image_px"], r["uid"]
        assert list(G.effective_size(*r["image_px"])) == r["image_px"], r["uid"]


# --------------------------------------------------------------- ExactInk

def test_exactink_reproduces_the_shipped_sft_boxes(sft_records):
    """REGRESSION TARGET -- verified at 20/20 and it must stay there.

    ExactInk recovers a label's painted extent by rendering the scene twice,
    with the label and without it, and taking the changed pixels. That is
    necessary because PIL's `textbbox` reports the font's LAYOUT box (advance
    widths, ascent, descent), which clips overhanging glyphs -- at these sizes a
    one-pixel-per-side error costs about 0.2 IoU.

    Re-measuring every shipped uid must reproduce `target.box_px`,
    `target.correction.layout_box_px` and `grew_px_lrtb` EXACTLY. If the
    generator drifts, the rebuilt scenes stop matching the shipped pixels and
    this is where it shows.
    """
    if not (SVGLOC / "manifest.jsonl").exists():
        pytest.skip("data/svg_localization is not present")
    manifest = {r["uid"]: r for r in _jsonl(SVGLOC / "manifest.jsonl")}

    ink = GF.ExactInk(dataset=SVGLOC)
    reproduced = 0
    for rec in sft_records:
        src = manifest[rec["uid"]]
        scene = ink.scene(src["graph_id"])
        assert scene is not None, rec["uid"]
        # The recorded layout box is passed in deliberately: the correction must
        # describe the box this dataset ships, not one recomputed against
        # whatever Pillow happens to be installed.
        got = ink.measure(scene, src["scale"], src["target_idx"],
                          src["text_ink_bbox_px"])
        assert got is not None, rec["uid"]
        want = rec["target"]
        assert got["box_px"] == want["box_px"], rec["uid"]
        assert got["layout_box_px"] == want["correction"]["layout_box_px"], rec["uid"]
        assert got["grew_px_lrtb"] == want["correction"]["grew_px_lrtb"], rec["uid"]
        assert got["method"] == want["correction"]["method"] == GF.ExactInk.METHOD
        reproduced += 1
    assert reproduced == len(sft_records) == 20


def test_the_layout_box_really_does_clip_the_painted_ink(sft_records):
    """The premise of ExactInk, stated as a measurement on the shipped set: the
    layout box is never the painted extent, and the ink escapes it outward."""
    for rec in sft_records:
        exact = rec["target"]["box_px"]
        layout = rec["target"]["correction"]["layout_box_px"]
        assert exact != layout, rec["uid"]
        assert GF.ExactInk.overlap(layout, exact) < 1.0, rec["uid"]
        assert max(rec["target"]["correction"]["grew_px_lrtb"]) > 0, rec["uid"]
        assert rec["target"]["source"] == GF.ExactInk.SOURCE


def test_exactink_growth_is_signed_outward_per_side():
    grew = GF.ExactInk.growth([10.0, 10.0, 20.0, 20.0], [9.0, 10.0, 22.0, 20.0])
    assert grew == [1.0, 0.0, 2.0, 0.0]     # left, top, right, bottom
    assert GF.ExactInk.growth([0, 0, 10, 10], [0, 0, 10, 10]) == [0.0, 0.0, 0.0, 0.0]


def test_exactink_overlap_is_iou():
    assert GF.ExactInk.overlap([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert GF.ExactInk.overlap([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    # a 30x9 box grown one pixel on every side is the case the docstring quotes
    assert GF.ExactInk.overlap([0, 0, 30, 9], [-1, -1, 31, 10]) == pytest.approx(0.77, abs=0.01)


# ----------------------------------------------------------------- samples

def test_samples_requires_an_explicit_out(capsys):
    """No default: a bare run must not be able to clobber the shipped
    reference artifact `data/sft_bbox/sft_bbox_20.jsonl`."""
    with pytest.raises(SystemExit) as e:
        GF.main(["samples", "--n", "2", "--seed", "0"])
    assert e.value.code == 2
    assert "--out" in capsys.readouterr().err
    # and the shipped file is still exactly where it was
    assert SFT_RECORDS.exists()


def test_samples_writes_only_where_told_and_bands_by_target_area(tmp_path, sft_records):
    """The curriculum structure is the reproducible part: five equal-count area
    bands, weighted 6/5/4/3/2 smallest-first, with the same band edges as the
    shipped set. (The individual picks are NOT asserted -- see the note in the
    report; `pick_spread`'s tie-break did not survive intact and reproduces 5 of
    the 20 shipped uids.)"""
    if not (SVGLOC / "manifest.jsonl").exists():
        pytest.skip("data/svg_localization is not present")
    dest = tmp_path / "nested" / "sft.jsonl"
    rc, text = _run(GF.main, ["samples", "--n", "20", "--seed", "0", "--out", str(dest)])
    assert rc == 0
    assert dest.exists()

    got = _jsonl(dest)
    assert len(got) == 20
    assert GF.allocate(20) == GF.WEIGHTS == [6, 5, 4, 3, 2]
    assert [r["curriculum"]["area_band"] for r in got] == \
           [r["curriculum"]["area_band"] for r in sft_records]
    assert {r["curriculum"]["area_band_label"] for r in got} == \
           {r["curriculum"]["area_band_label"] for r in sft_records}
    for r in got:
        assert r["task"] == "localize_bbox"
        assert r["image"]["delivered_untouched"] is True
        assert json.loads(r["completion"])["box"] == r["target"]["box_norm"]
        x0, y0, x1, y1 = r["target"]["box_norm"]
        assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0
        assert r["accept_region"]["answers_the_question"] == \
               (r["accept_region"]["hit_source"] != "shape")


def test_allocate_gives_every_band_a_record_even_below_one_per_weight():
    assert GF.allocate(20) == [6, 5, 4, 3, 2]
    assert sum(GF.allocate(37)) == 37
    assert GF.allocate(5) == [1, 1, 1, 1, 1]
    assert GF.allocate(3) == [1, 1, 1, 0, 0]
    assert sum(GF.allocate(100)) == 100
    assert all(a >= 1 for a in GF.allocate(100))


def test_stable_hash_does_not_depend_on_the_process():
    """`hash()` is salted per process, so a --seed built on it is not
    reproducible. This is the replacement and it must be stable."""
    assert GF.stable_hash(0, "svgloc:0001:small:00") == \
        GF.stable_hash(0, "svgloc:0001:small:00")
    assert GF.stable_hash(0, "a") != GF.stable_hash(1, "a")
    assert GF.stable_hash(0, "a") == int(
        hashlib.sha1(b"0:a").hexdigest(), 16)


# ------------------------------------------------------------------ audit

def _synthetic():
    """A 40x30 white frame with a 10x10 black block, and the row describing it.

    Every one of the five checks passes on this pair, so each broken variant
    below isolates exactly one check.
    """
    im = Image.new("RGB", (40, 30), (255, 255, 255))
    ImageDraw.Draw(im).rectangle([10, 10, 19, 19], fill=(0, 0, 0))
    row = {"uid": "synthetic", "image_px": [40, 30], "box_px": [10.0, 10.0, 20.0, 20.0],
           "bg_rgb": [255, 255, 255], "downscaled_by_api": False}
    return row, im


def test_the_five_audit_checks_pass_on_the_shipped_ladder(ladder_rows):
    names = ["in_bounds", "non_degenerate", "undownscaled", "has_ink", "tight"]
    fails = {k: [] for k in names}
    cache: dict[str, Image.Image] = {}
    for r in ladder_rows:
        if r["image"] not in cache:
            cache.clear()           # rows are grouped by image; one is enough
            cache[r["image"]] = Image.open(LADDER / r["image"]).convert("RGB")
        res = GF._checks(r, cache[r["image"]])
        assert sorted(res) == sorted(names)
        for k, ok in res.items():
            if not ok:
                fails[k].append(r["uid"])
    assert {k: v[:3] for k, v in fails.items() if v} == {}


def test_the_synthetic_control_passes_every_check():
    row, im = _synthetic()
    assert GF._checks(row, im) == {"in_bounds": True, "non_degenerate": True,
                                   "undownscaled": True, "has_ink": True, "tight": True}


@pytest.mark.parametrize("check,mutate", [
    # a box outside the frame cannot be normalised into [0,1]
    ("in_bounds", lambda r: r.update(box_px=[10.0, 10.0, 60.0, 20.0])),
    # one pixel wide is not a box; IoU against it is a coin flip
    ("non_degenerate", lambda r: r.update(box_px=[10.0, 10.0, 11.0, 20.0])),
    # the whole point of the ladder: what is rendered is what is delivered
    ("undownscaled", lambda r: r.update(downscaled_by_api=True)),
    # nothing is painted where the box says the text is
    ("has_ink", lambda r: r.update(box_px=[0.0, 0.0, 4.0, 4.0])),
    # ink present but the box does not reach all four edges -- it would teach
    # the model to overshoot
    ("tight", lambda r: r.update(box_px=[8.0, 8.0, 22.0, 22.0])),
])
def test_each_audit_check_fails_on_a_row_broken_only_for_it(check, mutate):
    row, im = _synthetic()
    mutate(row)
    res = GF._checks(row, im)
    assert res[check] is False, f"{check} did not notice its own breakage"


def test_the_audit_ink_threshold_is_zero():
    """A tolerance is tempting but wrong: at a 10px font the outermost glyph row
    sits one or two units off the background, so any tolerance starts reporting
    correct boxes as loose."""
    assert GF.INK_DELTA == 0


# =============================================================================
# C. blindspot.report_finetune -- BYTE-IDENTICAL REGRESSION TEST
# =============================================================================

GALLERY_MD5 = "62aa0e26cd9ff23df93234a930d84a0f"
GALLERY_BYTES = 1273816


def test_gallery_reproduces_the_shipped_artifact_byte_for_byte(tmp_path):
    """*** BYTE-IDENTICAL REGRESSION TEST -- protects the whole rebuild. ***

    Rendering `data/sft_bbox/sft_bbox_20.jsonl` must reproduce
    `outputs/finetune/gallery.html` to the byte. The page embeds twenty pairs of
    re-encoded PNGs as base64, so this single hash covers the crop arithmetic,
    the LANCZOS resize rule, the outline-drawn-outside rule, the PNG encoder
    settings and every import on the path. Any drift in any of them moves the
    hash.

    If this fails, DO NOT re-baseline the constant: the shipped file is the
    specification.
    """
    if not SFT_RECORDS.exists():
        pytest.skip("data/sft_bbox/sft_bbox_20.jsonl is not present")
    out = tmp_path / "gallery.html"
    rc, text = _run(RF.main, ["gallery", "--out", str(out)])
    assert rc == 0

    assert out.stat().st_size == GALLERY_BYTES
    assert _md5(out) == GALLERY_MD5

    if SHIPPED_GALLERY.exists():        # the artifact this constant came from
        assert _md5(SHIPPED_GALLERY) == GALLERY_MD5
        assert SHIPPED_GALLERY.stat().st_size == GALLERY_BYTES
        assert out.read_bytes() == SHIPPED_GALLERY.read_bytes()


def test_the_gallery_draws_only_the_supervision_target(tmp_path):
    """The wider `accept_region` is deliberately never stroked: for shape-held
    targets it is the whole enclosing node, and drawing it invites it to be read
    as the answer to "where is the text". It stays in the JSON, flagged."""
    if not SFT_RECORDS.exists():
        pytest.skip("data/sft_bbox/sft_bbox_20.jsonl is not present")
    out = tmp_path / "gallery.html"
    _run(RF.main, ["gallery", "--out", str(out)])
    page = out.read_text()
    assert page.count("<figure>") == 40          # two views per record
    assert "answers_the_question" in page
    assert page.count("supervision target") >= 1


# ------------------------------------------------- the figures' scene rebuild
"""`fig_box_extraction` re-renders the scene a manifest row was measured on and
indexes it with the row's `target_idx`. If the rebuilt scene is not the one the
ladder shipped, that index lands on a different label and the figure draws a box
around the wrong string while captioning the panel "as delivered to the model".
Nothing raises. It happened: `_scene` restated the ladder's recipe instead of
calling it, with `complexity = 1` against the ladder's 4, chart type by global
position instead of position within the aspect, and the canvas taken from a
hard-coded block of sixteen. 268 of the 1,513 shipped rows indexed out of range
and another 159 pointed at a different string.
"""


def test_the_rebuilt_scene_matches_the_manifest_on_every_sampled_row(ladder_rows):
    """The check that says the figure boxes the label the manifest names.

    Every 13th row, so the sample spans all six aspects and all four rungs. The
    label at `target_idx` in the rebuilt scene must be the manifest's
    `target_text` -- not merely in range, the same string.
    """
    scenes: dict[tuple, object] = {}
    checked = 0
    for row in ladder_rows[::13]:
        key = (row["graph_id"], row["chart_type"], tuple(row["canvas_px"]))
        if key not in scenes:
            scenes[key] = RF._scene(row)
        sc = scenes[key]
        assert sc is not None, row["uid"]
        assert (sc.w, sc.h) == tuple(row["canvas_px"]), row["uid"]
        texts = sc.texts
        assert row["target_idx"] < len(texts), (
            f"{row['uid']}: target_idx {row['target_idx']} of {len(texts)} labels")
        assert texts[row["target_idx"]]["s"] == row["target_text"], row["uid"]
        checked += 1
    assert checked >= 100


def test_the_rebuilt_scene_is_the_image_the_model_was_given(ladder_rows):
    """Panel 1 is captioned "as delivered to the model", so it has to be.

    Re-rendering the rebuilt scene at the row's rung must reproduce the shipped
    PNG byte for byte, which is the strongest available statement that the
    reconstruction is the delivered image and not a lookalike.
    """
    row = next(r for r in ladder_rows if r["uid"] == RF.FIG_UID)
    if not (LADDER / row["image"]).exists():
        pytest.skip("data/svgloc_mr images are not present")
    im = G.render(RF._scene(row), RF.RUNGS[row["rung"]])[0]
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    assert buf.getvalue() == (LADDER / row["image"]).read_bytes()


def test_the_ladder_constants_are_the_ladders_own(ladder_rows):
    """Not a second hand-copied table. The copy is what drifted."""
    assert RF.RUNGS is GF.RUNGS
    assert RF.ASPECTS == [(n, *GF.canvas_for(r)) for n, r in GF.ASPECTS]


def _small_ladder(tmp_path: Path, rows: list[dict], keep_pinned: bool = False) -> Path:
    """A manifest-only ladder in tmp_path, standing in for a dev-sized build.

    `fig_box_extraction` reads the manifest and re-renders from the scene, so it
    never opens an image and the images need not be copied. Rows are taken from
    the shipped manifest, so the reconstruction they exercise is the real one.
    """
    keep = [r for r in rows if r["graph_id"] < 4 or (keep_pinned and r["uid"] == RF.FIG_UID)]
    ds = tmp_path / "ladder"
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    return ds


def test_figures_builds_on_a_ladder_that_has_no_pinned_record(tmp_path, ladder_rows):
    """`--scenes-per-aspect 4` is the workflow docs/runme/FINETUNE.md recommends,
    and it produces 24 scenes -- no graph 89, so no `FIG_UID`. Requiring it fell
    through to `rows[0]`, whose `target_idx` then indexed off the end of a
    different scene: `IndexError` on the documented dev path."""
    ds = _small_ladder(tmp_path, ladder_rows)
    assert not any(r["uid"] == RF.FIG_UID for r in RF._manifest(ds))

    out = tmp_path / "assets"
    rc, _text = _run(RF.main, ["figures", "--out-dir", str(out), "--dataset", str(ds)])
    assert rc == 0
    assert (out / "fig_box_extraction.png").stat().st_size > 0
    assert (out / "fig_frames.png").stat().st_size > 0


def test_the_substitute_record_is_deterministic_and_fits_the_panel(tmp_path, ladder_rows):
    ds = _small_ladder(tmp_path, ladder_rows)
    rows = RF._manifest(ds)
    picked = RF._fig_row(rows)
    assert RF._fig_row(list(reversed(rows)))["uid"] == picked["uid"]
    x0, y0, x1, y1 = picked["box_px"]
    assert x1 - x0 <= RF.FIG_PANEL[0] - 2 * RF.FIG_FIT_PAD
    assert y1 - y0 <= RF.FIG_PANEL[1] - 2 * RF.FIG_FIT_PAD
    # and the substitute is only a substitute: the pin wins whenever it is there
    with_pin = RF._manifest(_small_ladder(tmp_path / "b", ladder_rows, keep_pinned=True))
    assert RF._fig_row(with_pin)["uid"] == RF.FIG_UID


def test_a_scene_that_does_not_match_its_row_is_refused_not_drawn(tmp_path, ladder_rows):
    """The failure this whole section exists for is silent, so it is asserted to
    be loud: a row whose `target_idx` does not name the manifest's own
    `target_text` must stop the figure rather than box the wrong label."""
    ds = _small_ladder(tmp_path, ladder_rows)
    rows = RF._manifest(ds)
    for r in rows:
        r["target_text"] = "definitely not the label"
    (ds / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit) as e:
        RF.fig_box_extraction(tmp_path / "x.png", str(ds))
    assert "does not match the manifest" in str(e.value)


# =============================================================================
# D. blindspot.render_markdown
# =============================================================================
"""The parser covers exactly what the document uses and no more. The surviving
`part3.html` is the specification for the whole of it; the unit tests below pin
the individual rules that produced it.
"""


def _mask_payloads(s: str) -> str:
    return re.sub(r"base64,[A-Za-z0-9+/=]+", "base64,PAYLOAD", s)


def _tag_counts(s: str) -> collections.Counter:
    return collections.Counter(re.findall(r"<(/?[a-zA-Z][a-zA-Z0-9]*)", s))


def test_part3_rerender_reproduces_every_html_tag_count(tmp_path, monkeypatch):
    """Tag counts, not bytes.

    The base64 image payloads are masked out (they are re-encoded on every run
    and are not what this test is about), and the comparison is structural: the
    same number of every element, in the same document, from the same markdown.
    """
    for p in (PART3_MD, PART3_HTML):
        if not p.exists():
            pytest.skip(f"{p} is not present")

    out = tmp_path / "part3.html"
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv",
                        ["render_markdown", "--src", "outputs/part3/part3.md",
                         "--out", str(out)])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert RM.main() == 0

    fresh = _mask_payloads(out.read_text())
    shipped = _mask_payloads(PART3_HTML.read_text())
    assert _tag_counts(fresh) == _tag_counts(shipped)
    # the CSS and the figure payloads are part of the same page, so check the
    # <style> block survived intact too
    assert re.search(r"<style>(.*?)</style>", fresh, re.S).group(1) == \
           re.search(r"<style>(.*?)</style>", shipped, re.S).group(1)
    assert fresh.count("base64,PAYLOAD") == shipped.count("base64,PAYLOAD")


def test_headings_render_at_their_level():
    assert RM.render("# One\n") == "<h1>One</h1>"
    assert RM.render("#### Four\n") == "<h4>Four</h4>"
    assert RM.render("Not # a heading\n") == "<p>Not # a heading</p>"


def test_an_ordered_list_that_resumes_carries_start():
    """A blank line ends the list, so an item after an interruption opens a new
    <ol>. Without `start="N"` the numbering restarts at 1 and the document
    silently renumbers itself."""
    md = ("1. first\n\n"
          "| a | b |\n| --- | --- |\n| 1 | 2 |\n\n"
          "2. second\n")
    out = RM.render(md)
    assert out.startswith("<ol><li>first</li></ol>")
    assert '<ol start="2"><li>second</li></ol>' in out
    assert out.index("<table") < out.index('<ol start="2"')
    # the first list is never given a redundant start="1"
    assert 'start="1"' not in out


def test_a_loose_list_item_wraps_each_block_in_p():
    """`li>p` is what the CSS styles; a single-block item is left bare so it does
    not pick up paragraph margins."""
    assert RM.render("1. head\n\n   second block\n") == \
        "<ol><li><p>head</p><p>second block</p></li></ol>"
    assert RM.render("1. head\n") == "<ol><li>head</li></ol>"


def test_bullets_are_grouped_into_one_list():
    assert RM.render("- one\n\n- two\n") == "<ul><li>one</li><li>two</li></ul>"


def test_tables_split_on_unescaped_pipes_only():
    assert RM.split_row("| a | b | c |") == ["a", "b", "c"]
    assert RM.split_row(r"| a \| b | c |") == [r"a \| b", "c"]
    out = RM.render("| a \\| b | c |\n| --- | --- |\n| 1 | 2 |\n")
    assert "<th>a | b</th>" in out and "<th>c</th>" in out
    assert out.count("<td>") == 2


def test_fenced_code_is_escaped_and_keeps_its_newlines():
    out = RM.render("```\nx = 1 < 2\ny = 2\n```\n")
    assert out == "<pre><code>x = 1 &lt; 2\ny = 2</code></pre>"


def test_block_quotes_render_their_contents_as_blocks():
    out = RM.render("> quoted **bold**\n> more\n")
    assert out == "<blockquote><p>quoted <strong>bold</strong> more</p></blockquote>"


def test_inline_bold_italic_and_code():
    out = RM.render("hello *world* and `x < y` and **bold**\n")
    assert "<em>world</em>" in out
    assert "<code>x &lt; y</code>" in out
    assert "<strong>bold</strong>" in out


def test_a_backslash_escaped_asterisk_survives_a_bold_span():
    """Without pulling escapes out first, a literal `*` written as `\\*` inside a
    bold span terminates the span early and the `**` leaks into the page as
    text -- which is exactly what the loss table did."""
    assert RM.render("**a b\\* c**\n") == "<p><strong>a b* c</strong></p>"
    assert "**" not in RM.render("**a b\\* c**\n")
    assert RM.inline(r"a \| b") == "a | b"
    assert RM.inline(r"literal \*not italic\*") == "literal *not italic*"


def test_a_missing_image_degrades_to_a_link_not_a_broken_img(tmp_path):
    out = RM.render("![alt text](nope.png)\n", base=tmp_path)
    assert out == '<figure><a href="nope.png">alt text</a></figure>'
    assert "<img" not in out


def test_a_present_image_is_inlined_as_a_data_uri(tmp_path):
    png = tmp_path / "shot.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(png, "PNG")
    out = RM.render("![a caption](shot.png)\n", base=tmp_path)
    assert '<figure class="fig">' in out
    assert "<img src=\"data:image/png;base64," in out
    assert "<figcaption>a caption</figcaption>" in out


def test_the_paste_build_pushes_every_style_inline():
    """Google Docs drops <style> blocks and class selectors, so anything that
    must survive the clipboard has to be an attribute on the element."""
    body = RM.render("# Title\n\nsome text\n")
    pasted = RM.to_paste(body)
    assert "<h1 style=" in pasted and "<p style=" in pasted
    assert "class=" not in RM.to_paste('<figure class="fig"><img src="x"></figure>')


# =============================================================================
# E. blindspot.report -- the non-API parts only
# =============================================================================
"""`tables`, `index` and `paste` each called `read_text()` unguarded on
`outputs/report/blindspots.md`, the hand-written prose spine, which is a
separate deliverable and is not in this repository. All three ended in a
FileNotFoundError -- `tables` and `index` AFTER writing their output, `paste`
before writing anything. These tests pin the guard, and the two formatting
helpers whose variants were the reason the duplication existed.
"""


@pytest.fixture
def report_module(monkeypatch, tmp_path):
    """`blindspot.report` with every output path redirected into tmp_path, and
    the prose spine guaranteed absent."""
    import blindspot.report as R
    monkeypatch.chdir(ROOT)             # its inputs are repo-relative
    out = tmp_path / "report"
    monkeypatch.setattr(R, "REPORT_OUT", out)
    monkeypatch.setattr(R, "FIGS_OUT", out / "figures")
    monkeypatch.setattr(R, "PROSE", out / "blindspots.md")
    assert not (out / "blindspots.md").exists()
    return R, out


class _Args:
    pass


def test_tables_skips_cleanly_without_the_prose_and_still_writes_tables_md(report_module):
    R, out = report_module
    if not Path("outputs/report/figures.json").exists():
        pytest.skip("outputs/report/figures.json is not present")
    rc, text = _run(lambda _a: R.cmd_tables(_Args()), None)
    assert rc == 0
    assert "blindspots.md not found" in text
    assert "Traceback" not in text
    assert (out / "tables.md").exists()
    assert (out / "tables.md").read_text().startswith("# Tables")


def test_index_skips_cleanly_without_the_prose_and_still_writes_its_output(report_module):
    R, out = report_module
    rc, text = _run(lambda _a: R.cmd_index(_Args()), None)
    assert rc == 0
    assert "blindspots.md not found" in text
    assert (out / "figures.md").exists()
    assert (out / "figures" / "index.html").exists()


def test_paste_skips_cleanly_and_writes_nothing_at_all(report_module):
    """Unlike the other two, paste has nothing to write without the prose: the
    page IS the prose, with the figures and tables interleaved."""
    R, out = report_module
    rc, text = _run(lambda _a: R.cmd_paste(_Args()), None)
    assert rc == 0
    assert "blindspots.md not found" in text
    assert R.build_paste() is None
    assert not (out / "paste_into_docs.html").exists()


def test_read_prose_returns_none_and_explains_itself(report_module):
    R, out = report_module
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert R.read_prose("wrote nothing") is None
    assert "blindspots.md not found; wrote nothing" in buf.getvalue()
    assert R.inject_tables() is None
    assert R.inject_refs() is None


def test_esc_escapes_quotes_by_default_but_not_on_the_paste_path():
    """`&quot;` pasted into a document editor shows as five literal characters,
    so the paste path -- and only the paste path -- passes quote=False."""
    import blindspot.report as R
    assert R.esc('say "hi" & <b>') == "say &quot;hi&quot; &amp; &lt;b&gt;"
    assert R.esc_nq('say "hi" & <b>') == 'say "hi" &amp; &lt;b&gt;'
    assert "&quot;" not in R.esc_nq('say "hi"')
    assert R.esc_nq("x") == R.esc("x", quote=False)
    # the paste page's inline formatter goes through esc_nq
    assert '"' in R.inline('a **bold** "quote"')
    assert "&quot;" not in R.inline('a **bold** "quote"')


def test_pct_uses_a_literal_em_dash_and_pct_html_uses_the_entity():
    """A literal `&mdash;` in a markdown table renders as five characters; a
    literal em-dash in HTML is fine but the pages were written with the entity.
    Getting this backwards is why there is one function and two callers."""
    import blindspot.report as R
    assert R.pct(None) == "—"
    assert "&mdash;" not in R.pct(None)
    assert R.pct_html(None) == "&mdash;"
    assert "—" not in R.pct_html(None)
    assert R.pct(0.1234) == "12.3%"
    assert R.pct(0.1234, 2) == "12.34%"
    assert R.pct_html(0.1234, 2) == "12.34%"
    assert R.pct(None, null="n/a") == "n/a"


def test_the_generated_markdown_tables_contain_no_html_entities(report_module):
    """The end-to-end statement of the rule above: tables.md is markdown, so an
    HTML entity anywhere in it is a rendering bug."""
    R, out = report_module
    if not Path("outputs/report/figures.json").exists():
        pytest.skip("outputs/report/figures.json is not present")
    with contextlib.redirect_stdout(io.StringIO()):
        R.cmd_tables(_Args())
    text = (out / "tables.md").read_text()
    assert re.search(r"&[a-zA-Z]+;", text) is None
    assert "&quot;" not in text and "&mdash;" not in text


def test_the_ablations_table_is_computed_from_figures_json_not_from_literals():
    """T7 reports eight arms that were run, scored and then left out of the
    report. Two of them are significant and both bear on conclusions stated
    elsewhere, so the table has to track `figures.json` rather than restate a
    remembered number: feeding it altered measurements must alter the table."""
    import blindspot.report as R
    if not Path("outputs/report/figures.json").exists():
        pytest.skip("outputs/report/figures.json is not present")

    f = R.load_json("outputs/report/figures.json")
    arms = f["synthetic"]["ablations"]["arms"]
    assert ("T7", "Prompt and answer-channel ablations", R.t8) in R.TABLES

    # every arm in the JSON is on the page, with the JSON's own numbers
    real = R.t8(f)
    for k, r in arms.items():
        assert f"`{k}`" in real
        assert R.pct(r["acc"], 2) in real
        assert f'{r["delta_pp"]:+.2f}pp' in real
        assert f'{r["chi2"]:.2f}' in real
    # the marker is the stored verdict, not a hand-kept list of arm names
    sig = {k for k, r in arms.items() if r["significant"]}
    assert sig == {"bbox", "cell_then_point", "quadrant_mc"}
    for k in sig:
        assert f'**{R.pct(arms[k]["acc"], 2)}**' in real
    for k in set(arms) - sig:
        assert f'**{R.pct(arms[k]["acc"], 2)}**' not in real

    # and nothing is baked in: perturbed input, perturbed table and note
    g = copy.deepcopy(f)
    a = g["synthetic"]["ablations"]["arms"]
    a["cell_then_point"].update(acc=0.4242, delta_pp=35.75, chi2=99.99)
    a["bbox"].update(acc=0.0333, delta_pp=-3.33, chi2=7.77)
    a["quadrant_mc"]["significant"] = False
    moved = R.t8(g)
    for gone in ("19.00%", "24.45", "+12.33pp", "1.33%", "10.23", "**80.33%**"):
        assert gone not in moved
    for shown in ("42.42%", "99.99", "+35.75pp", "3.33%", "7.77", "2.0×"):
        assert shown in moved
    # the marker follows the flag: unflagged, quadrant_mc keeps its row unbolded
    assert "80.33%" in moved and "**+14.00pp**" not in moved


def test_the_ablations_table_reaches_tables_md(report_module):
    R, out = report_module
    if not Path("outputs/report/figures.json").exists():
        pytest.skip("outputs/report/figures.json is not present")
    with contextlib.redirect_stdout(io.StringIO()):
        R.cmd_tables(_Args())
    text = (out / "tables.md").read_text()
    assert "## T7 — Prompt and answer-channel ablations" in text
    assert "cell_then_point" in text and "McNemar" in text


# ------------------------------------------- degenerate statistics and absent inputs
"""Two ways a report can publish something it did not measure.

`svgderived` formatted the McNemar chi-square unconditionally, and `eval` sets
it to None when no pair is discordant -- which is exactly what a perfect score
at both rungs produces, and word presence and counting both reach one in the
published Table 4. The page therefore died with a TypeError at precisely the
result it was built to report.

`gold_quality()` turned a missing ground-truth audit file into a contested rate
of 0.0 and carried it into `implied_floor`: an unmeasured dataset published as
"we found no contested gold", which is the most favourable value there is, and
indistinguishable from screenspot_pro's real 0 of 35.
"""

ZERO_DISCORDANT = {"a": "small", "b": "large", "n": 476, "acc_a": 1.0, "acc_b": 1.0,
                   "delta_pp": 0.0, "discordant_b": 0, "discordant_c": 0,
                   "mcnemar_chi2": None, "significant": False}


def test_mcnemar_with_no_discordant_pairs_says_so_instead_of_printing_zero():
    """0.00 would read as a computed non-result. The fact is that the test could
    not run, and why it could not run is the informative part."""
    import blindspot.report as R
    out = R.mcnemar_html(ZERO_DISCORDANT)
    assert "no discordant pairs" in out
    assert "&mdash;" in out
    assert "0.00" not in out and "0.0" not in out
    assert "not significant at p&lt;.05" not in out     # no verdict was reached
    # and the ordinary path is untouched
    assert R.mcnemar_html({"mcnemar_chi2": 4.567, "significant": True}) == \
        "&chi;&sup2;=4.57 &mdash; significant at p&lt;.05"
    assert R.mcnemar_html({"mcnemar_chi2": 1.2, "significant": False}) == \
        "&chi;&sup2;=1.20 &mdash; not significant at p&lt;.05"


def test_the_counting_prose_clause_drops_its_claim_when_there_is_no_test():
    """The sentence around the number asserts "nominally significant", which is
    not merely unprintable but false when no pair is discordant."""
    import blindspot.report as R
    out = R._counting_chi_clause(ZERO_DISCORDANT)
    assert "no discordant pair" in out
    assert "significant" not in out and "&chi;&sup2;=" not in out
    assert "nominally significant (&chi;&sup2;=4.57)" in \
        R._counting_chi_clause({"mcnemar_chi2": 4.567})


def test_a_hundred_percent_at_both_rungs_renders_instead_of_crashing():
    """End to end on the real summary, forced into the degenerate case.

    `eval derived` already handles a None chi-square; only the HTML step died,
    so the crash appeared for the first time at the point of publication.
    """
    import blindspot.report as R
    summary = ROOT / "outputs" / "svgderived" / "summary.json"
    if not summary.exists():
        pytest.skip("outputs/svgderived/summary.json is not present")
    d = json.loads(summary.read_text())
    cnt, mc = copy.deepcopy(d["counting"]), copy.deepcopy(d["word_mc"])
    for s in (cnt, mc):
        s["paired"].update(ZERO_DISCORDANT, n=s["paired"].get("n") or 1)

    page = R.svgderived_render(cnt, mc)          # used to raise TypeError
    assert page.count("no discordant pairs") == 2
    assert "Discordant 0/0, McNemar &chi;&sup2; &mdash; not computable" in page


def test_a_missing_gt_audit_is_absent_not_a_measured_zero(monkeypatch, tmp_path):
    """Absent and zero must not render the same, because they do not mean the
    same thing: one is a measurement, the other is the lack of one."""
    import blindspot.report as R
    monkeypatch.setattr(R, "RESULTS", tmp_path)

    absent = R.gold_quality()
    for ds, v in absent.items():
        assert v["audited"] == 0, ds
        assert v["contested_error_rate"] is None, ds
        assert v["implied_floor"] is None, ds

    # the same function, given a real audit in which nothing was contested
    rows = [{"uid": f"u{i}", "verdict": "gold_correct"} for i in range(35)]
    (tmp_path / "screenspot_pro__gtaudit.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    measured = R.gold_quality()["screenspot_pro"]
    assert measured["audited"] == 35
    assert measured["contested_error_rate"] == 0.0
    assert measured["implied_floor"] == 0.0


def test_the_gold_quality_line_reports_absence_as_absence():
    import blindspot.report as R
    absent = R.gold_quality_line(
        "charxiv", {"audited": 0, "contested": 0, "contested_error_rate": None,
                    "implied_floor": None})
    assert "not measured" in absent
    assert "0.0%" not in absent

    real_zero = R.gold_quality_line(
        "screenspot_pro", {"audited": 35, "contested": 0,
                           "contested_error_rate": 0.0, "implied_floor": 0.0})
    assert "0/35 = 0.0%" in real_zero
    assert "not measured" not in real_zero
