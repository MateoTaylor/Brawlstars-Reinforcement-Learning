"""brawl_vision.terrain.label_projection: the parts of map-label projection that need no video.

The label rule, the refusals, the scale fit and the per-frame track, on synthetic maps, frames and
odometry. The whole route over a recording (scripts/map_project_labels.py) is checked by
reproducing the committed Starfruit Supernova labels byte for byte
(scripts/probes/map_label_projection/README.md), which needs the gitignored footage.
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from brawl_vision.terrain import label_projection as lp
from brawl_vision.terrain.labeling import hud_plans, observed_cells

FIXTURE = (Path(__file__).parent / "fixtures" / "vision" / "labels"
           / "bluestacks-starfruit-supernova-west-upper_f640.json")


@pytest.fixture(scope="module")
def plan():
    return hud_plans()["emulator"]


def _grid(rows=8, cols=20, fill="."):
    return np.full((rows, cols), fill, "<U1")


# ---------------------------------------------------------------------------
# the map label as classes
# ---------------------------------------------------------------------------

def test_spawns_are_floor_and_unsettled_or_left_cells_never_label():
    chars = _grid(3, 4)
    chars[0, 0], chars[1, 1], chars[2, 2] = "S", "?", "#"
    g = lp.class_grid(chars, leave=[(3, 2)])
    assert g[0, 0] == lp.TERRAIN.index(".")
    assert g[1, 1] == lp.LEAVE and g[2, 3] == lp.LEAVE
    assert g[2, 2] == lp.TERRAIN.index("#")
    assert lp.robust_chars(g, (0.0, 0.0), (1.0, 1.0), (4, 3))[1, 1] == "?"


def test_a_map_label_with_a_box_or_a_cell_off_the_map_is_refused():
    chars = _grid(3, 4)
    chars[0, 0] = "X"
    with pytest.raises(ValueError, match="X"):
        lp.class_grid(chars)
    with pytest.raises(ValueError, match="off the 4 x 3 map"):
        lp.class_grid(_grid(3, 4), leave=[(4, 0)])


def test_leave_cells_parse_from_cells_and_inclusive_row_runs():
    assert lp.parse_leave(["3,26"], ["26:0-1,5", "33:7-8"]) == [
        (0, 26), (1, 26), (3, 26), (5, 26), (7, 33), (8, 33)]
    meeples = lp.parse_leave(row_runs=["26:0-3,8-14,18-21,25-31,35-38,43-44,50-52",
                                       "33:3-7,13-18,22-24,28-35,39-45,51-54,58-59"])
    assert len(meeples) == 66                      # Starfruit's, as MAP_ONBOARDING.md counts them
    for bad in (["3;26"], ["3"]):
        with pytest.raises(ValueError, match="C,R"):
            lp.parse_leave(cells=bad)
    for bad in (["26"], ["26:5-3"], ["x:1-2"], ["26:1-a"]):
        with pytest.raises(ValueError, match="row run"):
            lp.parse_leave(row_runs=bad)


def test_broken_cells_parse_from_c_r_at_f_and_a_typo_is_refused():
    assert lp.parse_broken(["41,33@2100", "30,6@1110", "29,6@1110"]) == [
        ((29, 6), 1110), ((30, 6), 1110), ((41, 33), 2100)]
    assert lp.parse_broken() == []
    for bad in (["41,33"], ["41,33@"], ["41;33@2100"], ["41@2100"], ["41,33@-5"], ["a,1@2"]):
        with pytest.raises(ValueError, match="C,R@F"):
            lp.parse_broken(bad)
    with pytest.raises(ValueError, match="twice"):
        lp.parse_broken(["1,2@10", "1,2@20"])
    with pytest.raises(ValueError, match="off the 4 x 3 map"):
        lp.grid_at(lp.class_grid(_grid(3, 4)), [((4, 0), 10)], 0)


def test_a_broken_wall_labels_from_the_map_until_its_frame_and_never_after():
    """A wall seen broken from frame 100: frame 99 still labels it `#`, frame 100 on leaves it,
    and the map's own grid stays as the match started for the next frame."""
    chars = _grid(3, 4)
    chars[1, 2] = "#"
    g = lp.class_grid(chars)
    broken = lp.parse_broken(["2,1@100"])
    before, at, after = (lp.grid_at(g, broken, f) for f in (99, 100, 3000))
    assert (before == g).all()
    assert at[1, 2] == after[1, 2] == lp.LEAVE and (at != g).sum() == 1
    assert g[1, 2] == lp.TERRAIN.index("#")
    view = (0.0, 0.0), (1.0, 1.0), (4, 3)
    assert lp.robust_chars(before, *view)[1, 2] == "#"
    assert lp.robust_chars(at, *view)[1, 2] == "?"


def test_the_canvas_puts_each_tile_where_the_lattice_says():
    """Tile (c, r)'s top-left corner lands at canvas px ((c + PAD) * ppt, (r + PAD) * ppt), on the
    edge between two canvas pixels. So the pixels either side of a tile boundary sit the same
    distance from it and sum to the two colours (bicubic weights sum to 1); half a canvas pixel
    off, which a tile-centre check cannot see, and they miss by half the colour step."""
    pitch, origin = (10.0, 12.0), (5.0, 3.0)
    colours = np.arange(9).reshape(3, 3) * 25 + 30
    image = np.zeros((3 + 3 * 12 + 4, 5 + 3 * 10 + 4, 3), np.uint8)
    for r in range(3):
        for c in range(3):
            image[3 + r * 12:3 + (r + 1) * 12, 5 + c * 10:5 + (c + 1) * 10] = colours[r, c]
    ppt = 20
    canvas = lp.map_canvas(image, pitch, origin, (3, 3), ppt)
    assert canvas.shape == ((3 + 2 * lp.PAD) * ppt, (3 + 2 * lp.PAD) * ppt, 3)
    for r in range(3):
        for c in range(3):
            y, x = (r + lp.PAD) * ppt + ppt // 2, (c + lp.PAD) * ppt + ppt // 2
            assert canvas[y, x, 0] == pytest.approx(colours[r, c], abs=1)
    mid = (1 + lp.PAD) * ppt + ppt // 2
    for k in (1, 2):
        e = (k + lp.PAD) * ppt                     # first canvas pixel of tile column / row k
        assert canvas[mid, e - 1, 0] + canvas[mid, e, 0] == pytest.approx(
            colours[1, k - 1] + colours[1, k], abs=1.5)
        assert canvas[e - 1, mid, 0] + canvas[e, mid, 0] == pytest.approx(
            colours[k - 1, 1] + colours[k, 1], abs=1.5)
    assert canvas[ppt // 2, ppt // 2, 0] == 0      # the padding is black


# ---------------------------------------------------------------------------
# the label rule
# ---------------------------------------------------------------------------

def _wall_column(col=10, rows=8, cols=20):
    chars = _grid(rows, cols)
    chars[:, col] = "#"
    return lp.class_grid(chars)


def test_a_cell_labels_only_when_its_class_holds_under_every_shift():
    """A wall column at map x 10-11, cells 1 tile wide. Cell 3 covers 70 % wall at A x 6.7, which
    falls to 56 % under the -0.12 shift: refused. At 6.75 it covers 75 %, 62.5 % at worst: wall."""
    g = _wall_column()
    size = (6, 2)
    at_70 = lp.robust_chars(g, (6.7, 2.0), (1.0, 1.0), size)
    at_75 = lp.robust_chars(g, (6.75, 2.0), (1.0, 1.0), size)
    assert at_70[0, 3] == "?" and at_75[0, 3] == "#"
    assert at_75[0, 2] == "." and at_75[0, 4] == "."
    half = lp.robust_chars(g, (6.5, 2.0), (1.0, 1.0), size)
    assert half[0, 3] == "?"                       # half wall, half floor


def test_cells_follow_the_fitted_scale_and_never_label_off_the_map():
    g = _wall_column(col=4)
    one = lp.robust_chars(g, (0.0, 2.0), (1.0, 1.0), (8, 1))
    wide = lp.robust_chars(g, (0.0, 2.0), (1.25, 1.0), (8, 1))
    assert one[0, 4] == "#" and one[0, 5] == "."
    # At 1.25 rectified tiles per map tile, cell 5 spans map x 4.0-4.8: the wall moved a cell over.
    assert wide[0, 4] == "." and wide[0, 5] == "#"
    off = lp.robust_chars(g, (-3.0, 2.0), (1.0, 1.0), (6, 1))
    assert list(off[0]) == ["?", "?", "?", ".", ".", "."]


# ---------------------------------------------------------------------------
# the refusals, on the emulator's real geometry
# ---------------------------------------------------------------------------

def _all_floor_labels(plan, gassed=None, boxes=(), transient=None):
    cols, rows = plan.size_tiles
    g = lp.class_grid(_grid(60, 60))
    gassed = np.zeros((rows, cols), bool) if gassed is None else gassed
    transient = np.zeros((rows, cols), bool) if transient is None else transient
    return lp.label_cells(g, (20.0, 20.0), (1.0, 1.0), plan, gassed, list(boxes), transient)


def _viewport_box(plan, col, row, half=12):
    """A viewport-px box of +/-half px around the centre of rectified cell (col, row)."""
    ppt = plan.pixels_per_tile
    p = np.array([[[col * ppt + ppt / 2, row * ppt + ppt / 2]]], np.float64)
    x, y = cv2.perspectiveTransform(p, plan.M_inv)[0, 0]
    return (x - half, y - half, x + half, y + half)


def test_with_nothing_in_view_every_observed_cell_off_the_hud_text_labels(plan):
    labels = _all_floor_labels(plan)
    expected = observed_cells(plan) & ~lp.text_cells(plan)
    assert np.array_equal(labels.chars == ".", expected)
    assert np.array_equal(labels.chars == "?", ~expected)
    assert labels.stats()["labelled"] == int(expected.sum())
    assert labels.projected == plan.size_tiles[0] * plan.size_tiles[1]


def test_the_text_and_kill_card_boxes_refuse_cells_the_hud_mask_lets_through(plan):
    text = lp.text_cells(plan)
    assert (text & observed_cells(plan)).any()
    rows, cols = np.nonzero(text)
    assert cols.max() < plan.size_tiles[0] // 2    # all in the top-left, where the readouts are


def test_gas_refuses_its_cells_and_one_cell_around_them(plan):
    cols, rows = plan.size_tiles
    gassed = np.zeros((rows, cols), bool)
    gassed[9, 15] = True
    labels = _all_floor_labels(plan, gassed=gassed)
    grown = labels.refused["gas"]
    assert grown[8:11, 14:17].all() and grown.sum() == 9
    assert (labels.chars[8:11, 14:17] == "?").all()
    assert labels.chars[9, 12] == "." and labels.chars[12, 15] == "."


def _viewport_rect(plan, x0, y0, x1, y1):
    """The viewport-px bounding box of the rectified-px rectangle (x0, y0)-(x1, y1)."""
    p = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], np.float64)
    v = cv2.perspectiveTransform(p, plan.M_inv)[0]
    return (*v.min(0), *v.max(0))


def test_detector_boxes_refuse_the_cells_under_them(plan):
    for kind in lp.RECT_PAD:
        labels = _all_floor_labels(plan, boxes=[(kind, _viewport_box(plan, 15, 9))])
        occ = labels.refused["occluded"]
        assert occ[9, 15] and labels.chars[9, 15] == "?"
        assert not occ[9, 12] and not occ[12, 15] and labels.chars[9, 12] == "."


def test_a_brawler_box_refuses_the_cell_it_stops_just_short_of(plan):
    """A box ending 8 rectified px before cell (16, 9) (7.3 once through the camera and back): a
    brawler's 14-px side pad covers 0.16 of that cell and refuses it; the 8-px pad of anything
    else covers 0.04, under the 0.10 that refuses."""
    ppt = plan.pixels_per_tile
    box = _viewport_rect(plan, 15 * ppt + 10, 9 * ppt + 10, 16 * ppt - 8, 10 * ppt - 10)
    assert lp.rect_box(plan, box)[:, 0].max() < 16 * ppt      # the box itself stays in cell 15
    brawler = lp.box_cells(plan, [("brawler", box)])
    other = lp.box_cells(plan, [("other", box)])
    assert brawler[9, 15] and other[9, 15]
    assert brawler[9, 16] and not other[9, 16]


def _texture(seed=0, size=(1112, 1640)):
    return np.random.default_rng(seed).integers(0, 256, size + (3,), dtype=np.uint8)


def _view(world, plan, dx_px=0, dy_px=0):
    w, h = plan.size_px
    return np.ascontiguousarray(world[100 + dy_px:100 + dy_px + h, 100 + dx_px:100 + dx_px + w])


def test_static_ground_is_not_transient_once_shifted_by_the_odometry_delta(plan):
    world = _texture()
    ppt = plan.pixels_per_tile
    here = _view(world, plan)
    ahead = _view(world, plan, 12, -24)            # the camera moved (0.25, -0.5) tiles
    behind = _view(world, plan, -24, 12)
    neighbours = [(ahead, (12 / ppt, -24 / ppt)), (behind, (-24 / ppt, 12 / ppt))]
    assert not lp.transient_cells(here, neighbours, plan).any()
    wrong = lp.transient_cells(here, [(ahead, (0.0, 0.0))], plan)
    assert wrong.sum() > 0.9 * observed_cells(plan).sum()


def test_a_splash_in_this_frame_alone_is_transient_and_only_its_cell(plan):
    world = _texture()
    ppt = plan.pixels_per_tile
    here = _view(world, plan).copy()
    here[9 * ppt:10 * ppt, 15 * ppt:16 * ppt] = 255
    neighbours = [(_view(world, plan, 12, -24), (12 / ppt, -24 / ppt))]
    moving = lp.transient_cells(here, neighbours, plan)
    assert moving[9, 15] and moving.sum() == 1
    labels = _all_floor_labels(plan, transient=moving)
    assert labels.chars[9, 15] == "?" and labels.stats()["transient"] == 1


# ---------------------------------------------------------------------------
# registration, on a frame rendered from a synthetic map
# ---------------------------------------------------------------------------

PITCH, ORIGIN, MAP_TILES = (8.0, 8.0), (3.0, 2.0), (36, 24)


def _synthetic_map(seed=3):
    """A map image of random-coloured 8-px tiles, tile (c, r) at image px (3 + 8c, 2 + 8r)."""
    cols, rows = MAP_TILES
    tiles = np.random.default_rng(seed).integers(20, 236, (rows, cols, 3)).astype(np.uint8)
    big = cv2.resize(tiles, (cols * 8, rows * 8), interpolation=cv2.INTER_NEAREST)
    image = np.zeros((2 + rows * 8 + 3, 3 + cols * 8 + 3, 3), np.uint8)
    image[2:2 + rows * 8, 3:3 + cols * 8] = big
    return image


def _render(image, plan, A, S):
    """The rectified frame whose px (0, 0) corner sits at map tile A, S rectified px per map px:
    written from the lattice directly, not through `map_canvas`, so the two can disagree."""
    ppt = plan.pixels_per_tile
    (px, py), (mx, my) = PITCH, ORIGIN
    M = np.array([[px / (ppt * S[0]), 0, mx + px * A[0] + 0.5 * px / (ppt * S[0]) - 0.5],
                  [0, py / (ppt * S[1]), my + py * A[1] + 0.5 * py / (ppt * S[1]) - 0.5]])
    return cv2.warpAffine(image, M, plan.size_px, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)


def test_registration_finds_a_rendered_frame_at_its_tile_and_shows_a_wrong_scale(plan):
    image = _synthetic_map()
    matcher = lp.MapMatcher(image, PITCH, ORIGIN, MAP_TILES, plan)
    A_true, S_true = np.array([4.3, 2.6]), (1.022, 1.004)
    rect = _render(image, plan, A_true, S_true)
    P, score, coarse, runner_up = matcher.coarse(rect)
    assert score > 0.8 and runner_up < 0.3         # one clear peak over the whole map
    # At scale 1 the match is right at the frame centre, so the corner is off by cen - cen / S.
    cen = np.array(plan.size_px) / plan.pixels_per_tile / 2
    corner = np.array(P) + plan.origin_tile + cen - cen / np.array(S_true)
    assert np.abs(corner - A_true).max() < 0.1
    A, score, blocks = matcher.at_scale(rect, S_true, P)
    assert score > 0.95
    assert np.array(A) == pytest.approx(A_true, abs=0.25 / plan.pixels_per_tile)   # 1/4 px
    left, right = np.array(blocks)[3], np.array(blocks)[5]         # middle row, outer blocks
    assert abs(left[0]) < 0.02 and abs(right[0]) < 0.02
    # Matched 2.2 % too large, the left of the frame lands right of where it should and the
    # right lands left: opposite residuals in the outer blocks are what a wrong S looks like.
    _, _, wrong = matcher.at_scale(rect, (1.0, S_true[1]), P)
    left, right = np.array(wrong)[3], np.array(wrong)[5]
    assert left[0] > 0.1 and right[0] < -0.1


def test_the_block_stretch_reads_zero_at_the_true_scale_and_points_back_to_it_from_a_wrong_one(plan):
    """What `map_project_labels.py --scale` is checked against when the odometry fit is bent (Truth,
    2026-11-05): the blocks of the real matcher on a rendered frame, not residuals made up here."""
    image = _synthetic_map()
    matcher = lp.MapMatcher(image, PITCH, ORIGIN, MAP_TILES, plan)
    A_true, S_true = np.array([4.3, 2.6]), (1.022, 1.004)
    rect = _render(image, plan, A_true, S_true)
    P = matcher.coarse(rect)[0]
    right = [matcher.at_scale(rect, S_true, P)[2]]
    assert np.abs(lp.block_stretch(right, S_true, plan.size_tiles)).max() < 0.004
    S = (1.0, S_true[1])                                            # x 2.2 % small, y right
    blocks = [matcher.at_scale(rect, S, P)[2]]
    kx, ky = lp.block_stretch(blocks, S, plan.size_tiles)
    assert kx < -0.01 and abs(ky) < 0.004
    assert S[0] < S[0] * (1 - kx) < S_true[0] + 0.005               # back toward the true scale
    hud = np.array(blocks, float)
    hud[0, 0] = np.nan                                              # a block under the HUD
    assert lp.block_stretch(hud, S, plan.size_tiles)[0] == pytest.approx(kx, abs=0.004)


# ---------------------------------------------------------------------------
# the scale fit and the track
# ---------------------------------------------------------------------------

S_TRUE, OFFSET = (1.022, 1.004), np.array([25.4, 19.0])


def _odometry(n=600, start=100, seed=1):
    """A lobby (segment 0) until `start`, then one long gameplay segment (1) wandering the map."""
    t = np.arange(n, dtype=float)
    pos = np.stack([12 * np.sin(t / 70), 9 * np.cos(t / 45) + 0.01 * t], axis=1)
    seg = np.where(t < start, 0, 1)
    idx = np.arange(0, n, 10)
    rng = np.random.default_rng(seed)
    P = pos[idx] / np.array(S_TRUE) + OFFSET + rng.normal(0, 0.05, (len(idx), 2))
    score = np.where(seg[idx] == 1, 0.6, 0.2)
    P[seg[idx] == 0] = rng.uniform(0, 60, (int((seg[idx] == 0).sum()), 2))   # NCC on the lobby
    return pos, seg, idx, P, score


def test_the_scale_fit_recovers_s_from_odometry_against_ncc():
    pos, seg, idx, P, score = _odometry()
    # Two gameplay samples whose NCC peak was wrong, scored as such: they never reach the fit.
    P[[20, 30]] += 7.0
    score[[20, 30]] = 0.3
    # Three lobby frames that happened to score well: the fit keeps to the segment with the most
    # good matches, so their random positions never reach it either.
    score[:3] = 0.5
    fit = lp.fit_scale(pos, seg, idx, P, score)
    assert fit.segment == 1 and fit.n == len(idx) - 10 - 2
    assert fit.S == S_TRUE
    assert np.allclose(fit.raw, S_TRUE, atol=1e-3)
    assert max(fit.resid_sd) < 0.1


def test_the_scale_fit_refuses_a_clip_that_never_matches_the_map():
    pos, seg, idx, P, score = _odometry()
    with pytest.raises(ValueError, match="no sampled frame"):
        lp.fit_scale(pos, seg, idx, P, np.full(len(idx), 0.2))
    few = np.full(len(idx), 0.2)
    few[20:25] = 0.6
    with pytest.raises(ValueError, match="only 5 samples"):
        lp.fit_scale(pos, seg, idx, P, few)


def test_only_segments_with_a_median_window_of_good_matches_are_registered():
    pos, seg, idx, P, score = _odometry()
    score[:3] = 0.5                                # three good lobby samples: not enough
    assert lp.segments_to_register(seg, idx, score) == [1]


def _track(seg_offsets=((1, OFFSET),), bad=None):
    pos, seg, idx, _, _ = _odometry()
    seg = seg.copy()
    seg[400:] = 2                                  # an odometry reset at frame 400
    keep = np.isin(seg[idx], [s for s, _ in seg_offsets])
    idx = idx[keep]
    A = pos[idx] / np.array(S_TRUE)
    for s, off in seg_offsets:
        A[seg[idx] == s] += off
    if bad is not None:
        A[bad] += 3.0
    return lp.Track(pos, seg, idx, A, S_TRUE), pos, idx


def test_one_bad_ncc_peak_cannot_move_a_label():
    track, pos, idx = _track(bad=15)
    frame = int(idx[15])
    assert np.allclose(track(frame), pos[frame] / np.array(S_TRUE) + OFFSET, atol=1e-9)


def test_each_odometry_segment_keeps_its_own_offset_and_others_are_refused():
    other = OFFSET + np.array([4.0, -3.0])
    track, pos, _ = _track(((1, OFFSET), (2, other)))
    for frame, off in ((395, OFFSET), (399, OFFSET), (400, other), (404, other)):
        assert np.allclose(track(frame), pos[frame] / np.array(S_TRUE) + off, atol=1e-9)
    assert track.span(1) == (100, 399) and track.span(2) == (400, 599)
    assert not track.covers(50)
    with pytest.raises(ValueError, match="outside every registered odometry segment"):
        track(50)


def test_the_offset_is_interpolated_between_samples():
    pos, seg, idx, _, _ = _odometry()
    idx = idx[seg[idx] == 1]
    drift = (idx - idx[0]) * 0.001                 # an offset drifting 0.01 tile per sample
    A = pos[idx] / np.array(S_TRUE) + OFFSET + drift[:, None]
    track = lp.Track(pos, seg, idx, A, S_TRUE)
    frame = int(idx[20] + 5)                       # halfway between two samples
    want = pos[frame] / np.array(S_TRUE) + OFFSET + (frame - idx[0]) * 0.001
    assert np.allclose(track(frame), want, atol=1e-9)


# ---------------------------------------------------------------------------
# what a label records
# ---------------------------------------------------------------------------

def test_the_note_is_the_one_the_committed_starfruit_labels_carry():
    notes = json.loads(FIXTURE.read_text())["notes"]
    date = re.match(r"Machine-made (\S+),", notes).group(1)
    A = tuple(float(v) for v in re.search(r"A=\(([-\d.]+), ([-\d.]+)\)", notes).groups())
    why = "meeple cells (map rows 26/33, wall-or-fence unsettled)"
    got = lp.note("brawl_deployment/data/maps/starfruit_supernova.csv", date, (1.022, 1.004), A, why)
    assert got == notes
    plain = lp.note("brawl_deployment/data/maps/x.csv", date, (1.0, 1.0), A)
    assert "shift; off-map" in plain and plain.isascii()


def test_the_note_names_the_cells_a_frame_leaves_broken():
    A, S = (10.0, 20.0), (1.026, 1.020)
    gone = lp.note("brawl_deployment/data/maps/x.csv", "2026-11-11", S, A, "why", [(29, 6), (41, 33)])
    assert "shift; why, map cells broken by this frame (29,6 41,33), off-map" in gone
    assert lp.note("m.csv", "d", S, A, "why", []) == lp.note("m.csv", "d", S, A, "why")


def test_labelled_tiles_are_the_map_tiles_under_labelled_cell_centres():
    chars = np.full((2, 3), "?", "<U1")
    chars[0, 0], chars[1, 2] = ".", "#"
    assert lp.labelled_tiles(chars, (10.7, 5.7), (1.0, 1.0)) == {(11, 6), (13, 7)}
    assert lp.labelled_tiles(chars, (10.7, 5.7), (2.0, 1.0)) == {(10, 6), (11, 7)}
