"""Authoring and scoring the hand-verified `<clip>.grid.csv`. See
Terrain_Perception_Build_Plan.md Phases I and L.

This is the one artifact a person has to make by hand, so most of what is tested here is about not
wasting or corrupting that work: not clobbering a filled-in file, not silently accepting a typo,
not letting the picture and the file disagree about which cell is which.
"""
from pathlib import Path

import numpy as np
import pytest

from brawl_vision.terrain import truth as T
from brawl_vision.terrain.evaluate import Mosaic, Window


class FakePlan:
    def __init__(self, size_tiles=(4, 4), ppt=8):
        self.origin_tile = (0, 0)
        self.size_tiles = size_tiles
        self.pixels_per_tile = ppt
        self.size_px = (size_tiles[0] * ppt, size_tiles[1] * ppt)
        self.valid = np.ones((self.size_px[1], self.size_px[0]), bool)


def _plate(scale=4, slots=5, window=None):
    plan = FakePlan()
    w = window or Window(row0=0, row1=8, col0=0, col1=8)
    return T.Plate(Mosaic(w, plan, grid_origin=(0, 0), scale=scale), slots=slots), plan


# ---------------------------------------------------------------------------
# the median plate -- the thing that makes tracing possible at all
# ---------------------------------------------------------------------------

def test_a_transient_is_outvoted_by_the_terrain_under_it():
    """The whole reason the plate is a median and not the Phase L mosaic's mean. A brawler standing
    on a tile for a minority of the frames must not tint the tile it stood on."""
    plate, plan = _plate(slots=5)
    h, w = plan.size_px[1], plan.size_px[0]
    ground = np.full((h, w, 3), 40, np.uint8)
    brawler = np.full((h, w, 3), 250, np.uint8)
    for i in range(9):
        plate.add(brawler if i < 3 else ground, (0.0, 0.0))
    img = plate.image()
    assert img[2, 2].max() < 100, "a minority transient survived the median"


def test_the_first_sample_fills_every_slot():
    """Otherwise the median of a pixel seen once is a median over zeros, and every thinly covered
    region of the plate comes out black -- which reads as 'not observed' when it was."""
    plate, plan = _plate(slots=5)
    plate.add(np.full((plan.size_px[1], plan.size_px[0], 3), 200, np.uint8), (0.0, 0.0))
    assert plate.image()[1, 1].tolist() == [200, 200, 200]


def test_samples_are_drawn_from_the_whole_history_not_just_the_end():
    """A regression test for reasoning that was wrong on the first pass.

    Keeping the last K writes round-robin *looks* like a spread and is not: for a pixel written 400
    times the survivors are writes 392-400, a seventh of a second apart at 60 fps. Anything that
    sits still for that long owns every slot and becomes the median. Reservoir sampling is what
    makes the slots a uniform draw over the pixel's whole history.

    Checked behaviourally: feed a long run of one value and then a short burst of another. Under
    round-robin the burst wins outright; under reservoir sampling it is outvoted.
    """
    plate, plan = _plate(slots=5)
    h, w = plan.size_px[1], plan.size_px[0]
    old = np.full((h, w, 3), 30, np.uint8)
    late = np.full((h, w, 3), 240, np.uint8)
    for _ in range(200):
        plate.add(old, (0.0, 0.0))
    for _ in range(5):                       # exactly enough to fill every slot round-robin
        plate.add(late, (0.0, 0.0))
    assert plate.image()[3, 3].max() < 120, "the last K writes took over the plate"


def test_coverage_is_per_tile_and_counts_only_written_pixels():
    plate, plan = _plate(scale=4)
    plate.add(np.full((plan.size_px[1], plan.size_px[0], 3), 90, np.uint8), (0.0, 0.0))
    cov = plate.coverage()
    assert cov.shape == (8, 8)
    assert cov[0, 0] == pytest.approx(1.0)      # inside the 4x4-tile patch
    assert cov[7, 7] == pytest.approx(0.0)      # far corner, never painted


# ---------------------------------------------------------------------------
# choosing the region
# ---------------------------------------------------------------------------

def test_the_region_is_the_largest_fully_covered_rectangle():
    covered = np.zeros((10, 10), bool)
    covered[2:8, 3:9] = True
    w = T.largest_covered_rect(covered, max_cols=99, max_rows=99)
    assert (w.row0, w.row1, w.col0, w.col1) == (2, 8, 3, 9)


def test_the_region_is_capped_about_its_centre():
    """A 40x30 template is not a job anyone finishes. The cap keeps it bounded, and centring the
    trim keeps the kept part the middle of the covered area rather than a corner of it."""
    covered = np.ones((30, 40), bool)
    w = T.largest_covered_rect(covered, max_cols=10, max_rows=6)
    assert w.cols == 10 and w.rows == 6
    assert w.col0 == 15 and w.row0 == 12

def test_no_covered_region_is_an_error_not_an_empty_template():
    with pytest.raises(ValueError, match="no fully covered"):
        T.largest_covered_rect(np.zeros((5, 5), bool), 10, 10)


# ---------------------------------------------------------------------------
# the reference image and the file must agree about which cell is which
# ---------------------------------------------------------------------------

def test_reference_image_is_one_tile_per_grid_square_plus_the_label_margin():
    """The correspondence the whole tool rests on. If the picture and the CSV disagree about cell
    size, someone can fill in a template perfectly and score zero, and nothing about the output
    would say why."""
    plate, plan = _plate(scale=6)
    plate.add(np.full((plan.size_px[1], plan.size_px[0], 3), 120, np.uint8), (0.0, 0.0))
    local = Window(row0=0, row1=4, col0=0, col1=4)
    img = T.reference_image(plate, local)
    assert img.shape[0] == 4 * 6 + T._MARGIN_PX
    assert img.shape[1] == 4 * 6 + T._MARGIN_PX


def test_template_records_where_it_sits_in_the_world():
    plate, plan = _plate(scale=4)
    plate.add(np.full((plan.size_px[1], plan.size_px[0], 3), 120, np.uint8), (0.0, 0.0))
    tmpl, local = T.build_template("clip", plate, grid_origin=(-64, -64), max_cols=99, max_rows=99)
    assert tmpl.origin_tile == (tmpl.window.col0 - 64, tmpl.window.row0 - 64)
    assert tmpl.size_tiles == (tmpl.window.cols, tmpl.window.rows)


# ---------------------------------------------------------------------------
# the file itself
# ---------------------------------------------------------------------------

def _template(rows=3, cols=4):
    return T.Template(clip="c", origin_tile=(0, 0), size_tiles=(cols, rows),
                      window=Window(row0=0, row1=rows, col0=0, col1=cols), coverage=1.0)


def test_a_generated_template_is_entirely_unlabelled(tmp_path):
    """Never seeded with the classifier's guess. A truth file started from the prediction it is
    grading measures how well someone spots errors they were shown, which is a different and much
    easier question than the one being asked."""
    out = tmp_path / "x.grid.csv"
    T.write_template(_template(), out)
    grid = T.load_grid_csv(out)
    assert grid.shape == (3, 4)
    assert (grid == T.UNLABELLED).all()


def test_regenerating_refuses_to_destroy_hand_written_work(tmp_path):
    """The one irreversible mistake this tool could make. Re-running the generator is the natural
    thing to do after tweaking a flag, and a template is hours of someone's attention."""
    out = tmp_path / "x.grid.csv"
    T.write_template(_template(), out)
    out.write_text(".,.,#,#\n.,.,.,.\n#,#,.,.\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="hand-written"):
        T.write_template(_template(), out)
    assert out.read_text(encoding="utf-8").startswith(".,.,#,#")
    T.write_template(_template(), out, overwrite=True)
    assert (T.load_grid_csv(out) == T.UNLABELLED).all()


def test_a_typo_is_an_error_not_a_wrong_answer(tmp_path):
    """A stray `W` or a capital `B` would otherwise be compared as a mismatch, and the reported
    accuracy would be a typo count wearing the classifier's name."""
    out = tmp_path / "x.grid.csv"
    out.write_text(".,.,W\n.,.,.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not one of"):
        T.load_grid_csv(out)


def test_ragged_rows_are_an_error(tmp_path):
    out = tmp_path / "x.grid.csv"
    out.write_text(".,.,.\n.,.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ragged"):
        T.load_grid_csv(out)


def test_whitespace_and_blank_lines_are_tolerated(tmp_path):
    """People edit these in a spreadsheet or by hand; a trailing newline or a padded cell is not a
    mistake worth refusing."""
    out = tmp_path / "x.grid.csv"
    out.write_text(". , . , #\n#, ., .\n\n", encoding="utf-8")
    assert T.load_grid_csv(out).tolist() == [[".", ".", "#"], ["#", ".", "."]]


def test_the_compact_one_character_per_tile_form_is_accepted(tmp_path):
    """The form a person can actually type against the reference picture.

    In the comma form column N sits at character 2N, so the file does not line up with the image
    and counting columns is where a long template goes wrong. One character per tile lines up
    exactly. Both spellings have to load to the same grid or the choice is a trap.
    """
    commas, compact = tmp_path / "a.grid.csv", tmp_path / "b.grid.csv"
    commas.write_text(".,.,#,~\n#,b,f,?\n", encoding="utf-8")
    compact.write_text("..#~\n#bf?\n", encoding="utf-8")
    assert T.load_grid_csv(commas).tolist() == T.load_grid_csv(compact).tolist()


def test_a_typo_in_the_compact_form_is_still_caught(tmp_path):
    out = tmp_path / "x.grid.csv"
    out.write_text("..#~\n#bZ?\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not one of"):
        T.load_grid_csv(out)


def test_progress_counts_what_is_filled(tmp_path):
    out = tmp_path / "x.grid.csv"
    out.write_text(".,?,#\n?,?,b\n", encoding="utf-8")
    pr = T.progress(T.load_grid_csv(out))
    assert pr["cells"] == 6 and pr["filled"] == 3
    assert pr["counts"] == {"#": 1, ".": 1, "b": 1}


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _chars(rows):
    return np.array([list(r) for r in rows], dtype="<U1")


def test_unknown_cells_are_skipped_on_both_sides():
    """"`?` costs nothing" is a promise the tool makes to whoever fills these in, so it is worth an
    assertion rather than trust in `compare_to_truth`'s docstring.

    The grid is deliberately larger than `compare_to_truth`'s `min_cells` floor of 12 -- a
    four-cell example returns "nothing comparable" rather than a perfect score, which is the floor
    working, not a bug.
    """
    pred = _chars(["." * 10, "#" * 10, "." * 10])
    truth = _chars(["?.?.?.?.?.", "?#?#?#?#?#", "?.?.?.?.?."])
    got = T.score(pred, truth, Window(row0=0, row1=3, col0=0, col1=10), max_shift=0)
    assert got["cells_compared"] == 15
    assert got["accuracy_at_best_offset"] == pytest.approx(1.0)


def test_a_resized_csv_is_refused_rather_than_misaligned(tmp_path):
    """If someone adds a row by hand, the file and its recorded window no longer describe the same
    region -- and silently comparing them would produce a number that looks fine and means
    nothing."""
    pred = _chars(["....", "####"])
    truth = _chars(["....", "####", "...."])
    with pytest.raises(ValueError, match="resized"):
        T.score(pred, truth, Window(row0=0, row1=2, col0=0, col1=4))


def test_wall_versus_fence_is_reported_on_its_own():
    """Per Phase H: both block movement, only WALL blocks shots, so a fence read as a wall tells
    the policy it has cover from a shot about to hit it. That error is invisible inside an overall
    accuracy dominated by floor -- here, 0.9 overall against 0.5 on the pair that matters."""
    pred = _chars(["..........", "..........", "##########"])
    truth = _chars(["..........", "..........", "#####fffff"])
    # max_shift=0 isolates classification. With the search on, a 3-row grid lets the offset search
    # slide the disagreeing row out of the overlap entirely and report a perfect 1.0 -- which is
    # worth knowing about (see `test_the_offset_search_can_fit_a_small_grid`) but is not what this
    # test is asking.
    got = T.score(pred, truth, Window(row0=0, row1=3, col0=0, col1=10), max_shift=0)
    assert got["accuracy_at_best_offset"] == pytest.approx(0.833, abs=0.01)
    assert got["wall_vs_fence"] == pytest.approx(0.5)


def test_the_offset_search_can_fit_a_small_grid_and_that_is_worth_knowing():
    """The offset search measures drift by trying alignments, so on a SMALL grid it can also slide
    the disagreeing part out of the overlap and report a perfect score at a large offset.

    `min_cells` bounds how far that can go, and the pair (accuracy, cells_compared) is what makes
    it visible -- which is why both are reported rather than accuracy alone. On a real 24x18
    template a 6-tile shift still compares 200+ cells, so the effect is small; on a toy grid it is
    total. Pinning it here stops someone reading a suspiciously perfect number as a result.
    """
    pred = _chars(["..........", "..........", "##########"])
    truth = _chars(["..........", "..........", "#####fffff"])
    searched = T.score(pred, truth, Window(row0=0, row1=3, col0=0, col1=10))
    exact = T.score(pred, truth, Window(row0=0, row1=3, col0=0, col1=10), max_shift=0)
    assert searched["accuracy_at_best_offset"] == pytest.approx(1.0)
    assert searched["cells_compared"] < exact["cells_compared"]


def test_wall_versus_fence_is_none_when_the_truth_has_neither():
    pred = _chars(["....", "...."])
    truth = _chars(["....", "...."])
    assert T.score(pred, truth, Window(row0=0, row1=2, col0=0, col1=4))["wall_vs_fence"] is None


def test_drift_and_misclassification_come_out_as_different_numbers():
    """The reason this file exists at all rather than one accuracy: a shifted-but-correct map and
    an aligned-but-wrong one are different bugs in different phases."""
    truth = _chars(["..###..", "..###..", "..###.."])
    shifted = _chars([".;.###.".replace(";", "."), "...###.", "...###."])
    got = T.score(shifted, truth, Window(row0=0, row1=3, col0=0, col1=7))
    assert got["position_error_tiles"] > 0
    assert got["accuracy_at_best_offset"] > got["accuracy_at_zero_offset"]


# ---------------------------------------------------------------------------
# against real footage
# ---------------------------------------------------------------------------

@pytest.mark.vision
def test_the_shipped_template_and_its_sidecar_still_agree():
    """The CSV is tracked and hand-edited; the sidecar says which region it covers. If they ever
    disagree, every number scored against that file is silently about the wrong part of the map."""
    import json
    base = Path(__file__).resolve().parent / "fixtures" / "vision"
    for csv_path in sorted(base.glob("*.grid.csv")):
        sidecar = csv_path.with_suffix("").with_suffix(".grid.json")
        assert sidecar.exists(), f"{csv_path.name} has no sidecar"
        meta = json.loads(sidecar.read_text())
        grid = T.load_grid_csv(csv_path)
        assert list(grid.shape[::-1]) == meta["size_tiles"], csv_path.name
        r0, r1, c0, c1 = meta["grid_window"]
        assert (r1 - r0, c1 - c0) == grid.shape, csv_path.name
