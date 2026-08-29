"""HUD masking. See Terrain_Perception_Build_Plan.md Phase B.

Split the same way as the other vision tests: the mask MACHINERY is exercised on synthetic
geometry so it runs on a fresh clone, and the shipped `hud_mask.json` is checked against the real
clips under `@pytest.mark.vision`.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from brawl_vision.camera import HudMask, HudRect, load_hud_mask
from brawl_vision.config import HUD_MASK_PATH

VIEWPORT = (1126, 2002)   # (h, w), the normalized viewport every source lands on


# ---------------------------------------------------------------------------
# HudRect / HudMask machinery
# ---------------------------------------------------------------------------

def test_normalized_rect_maps_to_expected_pixels():
    """Rects are stored normalized so the mask survives a resolution change -- a mask pinned in
    pixels would silently mis-cover the moment the capture size differed."""
    r = HudRect("x", x=0.25, y=0.5, w=0.25, h=0.25)
    assert r.pixels(1000, 800) == (250, 400, 499, 599)


def test_same_rect_scales_with_the_frame():
    r = HudRect("x", x=0.0, y=0.0, w=0.5, h=0.5)
    assert r.pixels(1000, 800) == (0, 0, 499, 399)
    assert r.pixels(2000, 1600) == (0, 0, 999, 799)


def test_rect_is_clipped_into_the_frame():
    """A rect running past the edge clips rather than raising: the joystick envelope genuinely
    reaches the bottom edge, and `y + h` rounding must not produce an out-of-bounds slice."""
    r = HudRect("x", x=0.9, y=0.9, w=0.5, h=0.5)
    x0, y0, x1, y1 = r.pixels(100, 100)
    assert (x1, y1) == (99, 99) and x0 <= x1 and y0 <= y1


def test_bool_mask_is_true_exactly_inside_the_rects():
    m = HudMask(rects=(HudRect("a", 0.0, 0.0, 0.5, 0.5),))
    b = m.bool_at((100, 100))
    assert b.dtype == bool and b.shape == (100, 100)
    assert b[:50, :50].all() and not b[50:, :].any() and not b[:, 50:].any()


def test_bool_mask_accepts_an_image_shape_with_a_channel_axis():
    m = HudMask(rects=(HudRect("a", 0.0, 0.0, 1.0, 1.0),))
    assert m.bool_at(np.zeros((20, 30, 3), np.uint8).shape).shape == (20, 30)


def test_overlapping_rects_union_rather_than_double_count():
    m = HudMask(rects=(HudRect("a", 0.0, 0.0, 0.6, 1.0), HudRect("b", 0.4, 0.0, 0.6, 1.0)))
    assert m.coverage((10, 100)) == pytest.approx(1.0)


def test_apply_fills_the_regions_and_leaves_a_copy():
    """`apply` must not mutate its input: Phase E's overlay draws a masked view alongside the
    raw frame, and an in-place fill would corrupt the very thing it is being compared against."""
    img = np.full((100, 100, 3), 200, np.uint8)
    m = HudMask(rects=(HudRect("a", 0.0, 0.0, 0.5, 0.5),))
    out = m.apply(img)
    assert (out[:50, :50] == 0).all()
    assert (out[50:, 50:] == 200).all()
    assert (img == 200).all(), "apply mutated its input"


def test_apply_accepts_a_colour_fill():
    img = np.zeros((10, 10, 3), np.uint8)
    out = HudMask(rects=(HudRect("a", 0.0, 0.0, 1.0, 1.0),)).apply(img, fill=(0, 0, 255))
    assert (out[..., 2] == 255).all() and (out[..., 0] == 0).all()


def test_empty_mask_file_is_rejected(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"rects": []}))
    with pytest.raises(ValueError, match="no rects"):
        load_hud_mask(p)


# ---------------------------------------------------------------------------
# the shipped mask
# ---------------------------------------------------------------------------

def test_shipped_mask_loads():
    m = load_hud_mask()
    assert {r.name for r in m.rects} == {
        "brawlers_left", "chat_bubble", "attack_button", "super_button", "joystick"}


def test_shipped_rects_are_normalized_not_pixels():
    """The single most likely authoring mistake: pasting measured pixel bounds straight in. Every
    value must be a fraction, and a stray 1388 would sail through JSON parsing unnoticed."""
    for r in load_hud_mask().rects:
        for name, v in (("x", r.x), ("y", r.y), ("w", r.w), ("h", r.h)):
            assert 0.0 <= v <= 1.0, f"{r.name}.{name} = {v} is not normalized"
        assert r.x + r.w <= 1.0001 and r.y + r.h <= 1.0001, f"{r.name} extends past the frame"


def test_shipped_mask_coverage_stays_within_budget():
    """The mask is subtracted from every downstream stage's evidence, so an over-broad one
    starves classification silently rather than failing. ~16% is the measured cost today and
    two thirds of it is the floating joystick's envelope; a jump past 25% means someone widened a
    rect without weighing that, and should re-read the Phase B notes."""
    assert load_hud_mask().coverage(VIEWPORT) < 0.25


def test_joystick_is_the_dominant_cost_and_that_is_expected():
    """Recorded so the shape of the mask is not mistaken for a bug. Brawl Stars' movement stick
    FLOATS -- it spawns wherever the thumb lands -- so it cannot be covered by a snug rect the way
    the fixed buttons can; this one is a motion envelope."""
    m = load_hud_mask()
    by_name = {r.name: r for r in m.rects}
    total = m.coverage(VIEWPORT)
    stick = HudMask(rects=(by_name["joystick"],)).coverage(VIEWPORT)
    assert stick > 0.5 * total


# ---------------------------------------------------------------------------
# against real footage
# ---------------------------------------------------------------------------

@pytest.mark.vision
def test_mask_geometry_matches_the_normalized_viewport():
    """The mask is authored against `normalize_viewport`'s output, not a raw capture. If a clip
    ever normalized to a different size, every rect would land somewhere else."""
    from brawl_vision.clips import ClipReader
    from tests.test_vision_clips import _path

    frame = next(iter(ClipReader(_path("zone_grows_from_east"))))
    assert frame.image.shape[:2] == VIEWPORT
    assert tuple(json.loads(HUD_MASK_PATH.read_text())["viewport"]) == (VIEWPORT[1], VIEWPORT[0])


@pytest.mark.vision
@pytest.mark.parametrize("index", [200, 700, 1150])
def test_hud_elements_are_actually_covered(index):
    """Phase B's acceptance criterion, spot-checked across several in-match states rather than
    one snapshot -- which is what caught the floating joystick: a box fitted to frames 200 and 700
    left the stick hanging outside it at 1150.

    Checks the three elements whose appearance is bright and distinctive against this map's dark
    purple: masking them must remove essentially all of their brightness from those regions.
    """
    from brawl_vision.clips import ClipReader
    from tests.test_vision_clips import _path

    frame = next(f for f in ClipReader(_path("zone_grows_from_east")) if f.index == index)
    m = load_hud_mask()
    masked = m.apply(frame.image)
    h, w = frame.image.shape[:2]
    for name in ("brawlers_left", "chat_bubble", "attack_button"):
        rect = next(r for r in m.rects if r.name == name)
        x0, y0, x1, y1 = rect.pixels(w, h)
        assert masked[y0:y1 + 1, x0:x1 + 1].max() == 0, f"{name} not fully cleared"


@pytest.mark.vision
@pytest.mark.parametrize("clip,y0,y1", [
    ("showdown_alternate_map2", 252, 355),      # bubble sits high: a spectator list is above it
    ("zone_grows_from_east", 372, 475),         # ...and low when there is not
])
def test_the_chat_bubble_rect_covers_both_heights_it_appears_at(clip, y0, y1):
    """A regression test for a real gap, found by Phase L's mosaic rather than by a test.

    The bubble is a 124x102 px element, and it renders at two different heights depending on
    whether a spectator/kill list is stacked above it. The original rect was fitted to the LOWER
    position only, so on two of six clips the bubble was unmasked -- and being screen-fixed, it
    then smeared across the world-anchored mosaic canvas at whatever tiles the camera happened to
    be looking at. A single-frame check of one clip passes straight through this.
    """
    from brawl_vision.clips import ClipReader
    from tests.test_vision_clips import _path

    m = load_hud_mask()
    covered = m.bool_at(VIEWPORT)
    assert covered[y0:y1, 1852:1978].all(), (
        f"the chat bubble's {clip} position (y {y0}-{y1}) is outside the mask"
    )
    frame = next(f for f in ClipReader(_path(clip)) if f.index == 200)
    assert m.apply(frame.image)[y0:y1, 1852:1978].max() == 0


@pytest.mark.vision
def test_mask_leaves_the_bulk_of_the_world_untouched():
    """The complement of the coverage ceiling: the mask must not be so broad that little world
    survives it. Checked against real footage rather than geometry alone."""
    from brawl_vision.clips import ClipReader
    from tests.test_vision_clips import _path

    frame = next(iter(ClipReader(_path("zone_grows_from_east"))))
    keep = ~load_hud_mask().bool_at(frame.image.shape)
    assert keep.mean() > 0.75


# ---------------------------------------------------------------------------
# Phase C -- the homography
# ---------------------------------------------------------------------------

import cv2  # noqa: E402
from brawl_vision.camera import (  # noqa: E402
    CameraModel, PIXELS_PER_TILE, load_camera_model, solve_camera_model,
)


def _synthetic_calibration(tmp_path, mu=1.02, noise=0.0, seed=0):
    """Build point files from a KNOWN ground homography, so the solver can be checked against an
    answer rather than against itself. The wall-top plane is generated by inverting the same
    homology the solver is supposed to undo."""
    rng = np.random.default_rng(seed)
    H_gt = np.array([[70.0, -6.0, 900.0],
                     [3.0, 52.0, 400.0],
                     [1.7e-3, -1.15e-2, 1.0]])
    V = np.array([1730.0, 13000.0, 1.0])
    horizon = np.linalg.inv(H_gt)[2, :].copy()
    horizon /= np.linalg.norm(horizon[:2])
    W = np.eye(3) + (mu - 1.0) * np.outer(V, horizon) / (horizon @ V)
    H_top = np.linalg.inv(W) @ H_gt          # so that W @ H_top == H_gt

    tiles = np.array([[x, y] for x in (-4, -1, 2, 5) for y in (-2, 1, 4)], np.float64)
    px = cv2.perspectiveTransform(tiles.reshape(-1, 1, 2), H_top).reshape(-1, 2)
    px = px + rng.normal(0, noise, px.shape)
    pts = tmp_path / "pts.csv"
    pts.write_text("px,py,tx,ty\n" + "".join(
        f"{p[0]:.2f},{p[1]:.2f},{t[0]:.0f},{t[1]:.0f}\n" for p, t in zip(px, tiles)))

    tops = cv2.perspectiveTransform(
        np.array([[[300., 250.]], [[1000., 300.]], [[1500., 700.]], [[600., 800.]]]),
        np.eye(3)).reshape(-1, 2)
    bases = cv2.perspectiveTransform(tops.reshape(-1, 1, 2), W).reshape(-1, 2)
    bases = bases + rng.normal(0, noise, bases.shape)
    hts = tmp_path / "h.csv"
    hts.write_text("pair,kind,px,py\n" + "".join(
        f"{i+1},top,{t[0]:.2f},{t[1]:.2f}\n{i+1},base,{b[0]:.2f},{b[1]:.2f}\n"
        for i, (t, b) in enumerate(zip(tops, bases))))
    return pts, hts, H_gt


def test_solver_recovers_a_known_ground_homography(tmp_path):
    """End-to-end on noiseless synthetic data: the solver must undo the wall-top offset exactly,
    not merely produce something self-consistent."""
    pts, hts, H_gt = _synthetic_calibration(tmp_path)
    model, report = solve_camera_model(pts, hts, viewport=(2002, 1126))
    probe = np.array([[0., 0.], [5., 3.], [-4., 7.], [2., -3.]])
    want = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), H_gt).reshape(-1, 2)
    got = model.tile_to_px(probe)
    assert np.abs(got - want).max() < 1.0, f"max px error {np.abs(got - want).max():.3f}"


def test_solver_reports_residuals_and_diagnostics(tmp_path):
    pts, hts, _ = _synthetic_calibration(tmp_path, noise=1.0, seed=3)
    _, report = solve_camera_model(pts, hts, viewport=(2002, 1126))
    for key in ("grid_residual_px", "base_residual_px", "vanishing_point", "mu",
                "H_perspective_row", "parallax_tiles", "height_pairs_used"):
        assert key in report


def test_solver_rejects_too_few_points(tmp_path):
    p = tmp_path / "few.csv"
    p.write_text("px,py,tx,ty\n1,1,0,0\n2,2,1,0\n3,3,2,0\n")
    h = tmp_path / "h.csv"
    h.write_text("pair,kind,px,py\n1,top,1,1\n1,base,1,2\n")
    with pytest.raises(ValueError, match=">=6"):
        solve_camera_model(p, h, viewport=(2002, 1126))


def test_exclude_pairs_drops_the_named_pair(tmp_path):
    pts, hts, _ = _synthetic_calibration(tmp_path)
    _, r_all = solve_camera_model(pts, hts, viewport=(2002, 1126))
    _, r_sub = solve_camera_model(pts, hts, viewport=(2002, 1126), exclude_pairs=("2",))
    assert "2" in r_all["height_pairs_used"]
    assert "2" not in r_sub["height_pairs_used"]
    assert r_sub["height_pairs_excluded"] == ["2"]


def test_too_few_usable_pairs_after_exclusion_raises(tmp_path):
    pts, hts, _ = _synthetic_calibration(tmp_path)
    with pytest.raises(ValueError, match=">=3 usable"):
        solve_camera_model(pts, hts, viewport=(2002, 1126), exclude_pairs=("1", "2"))


# --- the shipped model -----------------------------------------------------

def test_shipped_model_loads_and_round_trips():
    m = load_camera_model()
    probe = np.array([[0., 0.], [5., 3.], [-4., 7.]])
    assert np.abs(m.px_to_tile(m.tile_to_px(probe)) - probe).max() < 1e-6


def test_shipped_model_maps_straight_lines_to_straight_lines():
    """A homography's defining property, and the cheapest guard against a corrupted matrix."""
    m = load_camera_model()
    for fixed_axis in (0, 1):
        t = np.zeros((40, 2))
        t[:, fixed_axis] = 3.0
        t[:, 1 - fixed_axis] = np.linspace(-6, 10, 40)
        p = m.tile_to_px(t)
        u, v = (p[:, 0], p[:, 1]) if fixed_axis else (p[:, 1], p[:, 0])
        assert np.abs(np.polyval(np.polyfit(u, v, 1), u) - v).max() < 1e-6


def test_shipped_model_says_the_camera_is_perspective():
    """Measured |H[2,:2]| ~ 1.2e-02, three orders above the affine threshold. This is what makes
    the visible footprint a TRAPEZOID rather than a parallelogram, which Phase K needs to know
    when it picks the simulator's view rectangle."""
    m = load_camera_model()
    assert m.is_affine is False
    assert np.abs(m.H[2, :2]).max() > 1e-3


def test_shipped_model_is_pinned_to_the_normalized_viewport():
    m = load_camera_model()
    assert tuple(m.viewport) == (2002, 1126)
    with pytest.raises(ValueError, match="calibrated for"):
        m.check_viewport((1126, 2436, 3))          # a RAW capture frame, not normalized
    m.check_viewport((1126, 2002, 3))              # the normalized viewport is fine


def test_shipped_model_records_the_wall_height_parallax():
    """~0.87 tiles, and NEGATIVE: the ground point under a wall-top corner sits at a smaller ty.
    Nearly a whole cell, which is why the ground plane had to be recovered rather than assumed --
    a wall's visible top is one cell away from the footprint that actually blocks movement."""
    m = load_camera_model()
    assert -1.1 < m.parallax_tiles < -0.6


def test_shipped_calibration_residuals_are_within_tolerance():
    raw = json.loads((Path(__file__).resolve().parent.parent
                      / "brawl_vision/data/homography.json").read_text())
    rep = raw["report"]
    assert rep["n_grid_points"] >= 6
    assert rep["grid_residual_px"]["mean"] < 4.0
    assert rep["grid_residual_px"]["max"] < 10.0
    assert rep["base_residual_px"]["mean"] < 10.0
    assert rep["height_pairs_excluded"], "pair 3 was mis-clicked and must stay excluded"


def test_pixels_per_tile_is_a_choice_not_a_measurement():
    """Recorded so it is not mistaken for calibration output: it sets the rectified patch's
    resolution and can be changed freely without re-annotating anything."""
    assert load_camera_model().pixels_per_tile == PIXELS_PER_TILE


# ---------------------------------------------------------------------------
# Phase D -- rectification
# ---------------------------------------------------------------------------

from brawl_vision.camera import build_rectify_plan  # noqa: E402
from brawl_vision.clips import ClipReader  # noqa: E402

import csv  # noqa: E402

CLIPS = Path(__file__).resolve().parent / "fixtures" / "vision"
FRAMES = Path(__file__).resolve().parent.parent / "brawl_vision" / "data" / "frames"
_WIN = 192


def _synth_model(tmp_path):
    pts, hts, H_gt = _synthetic_calibration(tmp_path)
    model, _ = solve_camera_model(pts, hts, viewport=(2002, 1126))
    return model, H_gt


# Decoded frames are cached across tests. Two reasons, one of them not obvious: decoding is what
# makes these tests slow, and OpenCV 5.0.0 on Windows throws an intermittent access violation after
# roughly eight VideoCapture open/release cycles in a process. Reusing decoded frames keeps the
# cycle count low as well as the runtime.
_FRAME_CACHE: dict[tuple[str, int], np.ndarray] = {}


def _frames_at(clip, indices):
    want = set(indices)
    have = {i: _FRAME_CACHE[(clip, i)] for i in want if (clip, i) in _FRAME_CACHE}
    if len(have) == len(want):
        return have
    out = dict(have)
    for frame in ClipReader(CLIPS / f"{clip}.mp4"):
        if frame.index in want:
            out[frame.index] = frame.image
            _FRAME_CACHE[(clip, frame.index)] = frame.image
        if len(out) == len(want):
            break
    missing = want - set(out)
    assert not missing, f"{clip}.mp4 has no frame {sorted(missing)}"
    return out


def _axis_aligned_share(gray, mask):
    """Share of gradient energy lying within +-5 deg of a screen axis. Orientation is folded onto
    [0, 90), so 10 of 90 bins is 11.1% by chance."""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 90.0
    sel = mask & (mag > np.percentile(mag[mask], 92))
    hist, _ = np.histogram(ang[sel], bins=90, range=(0, 90), weights=mag[sel])
    hist = hist / hist.sum()
    return float(hist[:5].sum() + hist[85:].sum() + hist[40:50].sum())


def _core_of(plan):
    """`valid`, eroded so windows never straddle the trapezoid edge -- the boundary between world
    and border fill is a hard step that would dominate any gradient or correlation measurement."""
    return cv2.erode(plan.valid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)


def _window_shifts(plan, a, b, min_response=0.4):
    """Independent phase correlation over a grid of fully-in-core windows. Returns
    (cx, cy, dx, dy) rows."""
    core = _core_of(plan)
    han = cv2.createHanningWindow((_WIN, _WIN), cv2.CV_32F)
    rows = []
    for y in range(0, plan.size_px[1] - _WIN, _WIN // 2):
        for x in range(0, plan.size_px[0] - _WIN, _WIN // 2):
            if not core[y:y + _WIN, x:x + _WIN].all():
                continue
            (dx, dy), resp = cv2.phaseCorrelate(
                a[y:y + _WIN, x:x + _WIN].copy(), b[y:y + _WIN, x:x + _WIN].copy(), han)
            if resp > min_response:
                rows.append((x + _WIN / 2, y + _WIN / 2, dx, dy))
    return np.array(rows)


def _shift_field_trend(plan, shifts, reject=4.0):
    """Magnitude of the best-fit linear trend in the shift field, expressed as pixels of drift
    across the patch diagonal. Zero for a rigid translation."""
    d = shifts[:, 2:4]
    keep = np.linalg.norm(d - np.median(d, 0), axis=1) < reject
    A = np.c_[shifts[keep, 0], shifts[keep, 1], np.ones(keep.sum())]
    J = np.array([np.linalg.lstsq(A, d[keep, k], rcond=None)[0][:2] for k in (0, 1)])
    return float(np.linalg.norm(J, 2) * np.hypot(*plan.size_px))


def _rect_gray(plan, images, index):
    return cv2.cvtColor(plan.rectify(images[index]), cv2.COLOR_BGR2GRAY).astype(np.float32)


def test_plan_window_is_whole_tiles_and_covers_the_whole_viewport(tmp_path):
    """The window is snapped OUTWARD: an inward snap would silently crop real world off the edge
    of every rectified frame, which looks like nothing at all in an overlay."""
    model, _ = _synth_model(tmp_path)
    plan = build_rectify_plan(model)
    assert plan.size_px == (plan.size_tiles[0] * plan.pixels_per_tile,
                            plan.size_tiles[1] * plan.pixels_per_tile)
    w, h = model.viewport
    corners = model.px_to_tile(np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64))
    rect = plan.tile_to_rect(corners)
    assert rect.min() >= -1e-9
    assert rect[:, 0].max() <= plan.size_px[0] + 1e-9
    assert rect[:, 1].max() <= plan.size_px[1] + 1e-9


def test_integer_tiles_land_exactly_on_pixel_multiples(tmp_path):
    """The grid-phase invariant. If integer tile coordinates landed mid-pixel-block, every cell
    Phase H classified would straddle two real tiles -- plausible in an overlay, ruinous in the
    occupancy grid, and invisible without an assertion like this one."""
    model, _ = _synth_model(tmp_path)
    plan = build_rectify_plan(model)
    tiles = np.array([[0, 0], [1, 0], [-3, 7], [11, -5], [21, 10]], np.float64)
    assert np.allclose(plan.tile_to_rect(tiles) % plan.pixels_per_tile, 0.0)
    assert np.array_equal(plan.origin_tile, np.round(plan.origin_tile))


def test_rect_and_tile_coordinates_round_trip(tmp_path):
    model, _ = _synth_model(tmp_path)
    plan = build_rectify_plan(model)
    tiles = np.array([[0.0, 0.0], [3.25, -2.5], [17.75, 9.125]])
    assert np.allclose(plan.rect_to_tile(plan.tile_to_rect(tiles)), tiles)


def test_warp_matrix_agrees_with_the_coordinate_helpers(tmp_path):
    """`M` is what actually resamples pixels while `tile_to_rect` is what callers reason with.
    They come from the same S by different code paths, so a divergence would put the overlay and
    the pixels half a tile apart while both looked internally consistent."""
    model, _ = _synth_model(tmp_path)
    plan = build_rectify_plan(model)
    tiles = np.array([[0.0, 0.0], [4.0, -3.0], [12.5, 6.25]])
    via_matrix = cv2.perspectiveTransform(
        model.tile_to_px(tiles).reshape(-1, 1, 2), plan.M).reshape(-1, 2)
    assert np.allclose(plan.tile_to_rect(tiles), via_matrix, atol=1e-6)


def test_rectifying_a_synthetic_grid_returns_straight_square_tiles(tmp_path):
    """The end-to-end pixel check, against a KNOWN homography rather than against itself: draw a
    tile grid in ground coordinates, project it onto a fake screen, rectify, and require the lines
    to come back axis-aligned at exactly `pixels_per_tile` spacing."""
    model, H_gt = _synth_model(tmp_path)
    plan = build_rectify_plan(model)
    w, h = model.viewport

    frame = np.zeros((h, w), np.uint8)
    for k in range(-14, 28):
        for a, b in (((k, -16), (k, 16)), ((-16, k), (28, k))):
            p = cv2.perspectiveTransform(np.array([[a], [b]], np.float64), H_gt).reshape(-1, 2)
            cv2.line(frame, tuple(np.round(p[0]).astype(int)),
                     tuple(np.round(p[1]).astype(int)), 255, 2)

    rect = plan.rectify(frame, interpolation=cv2.INTER_NEAREST)
    core = _core_of(plan)
    lit = (rect > 128) & core
    ppt = plan.pixels_per_tile

    for axis, size in ((0, plan.size_px[0]), (1, plan.size_px[1])):
        mass = lit.sum(axis) / np.maximum(core.sum(axis), 1)
        peaks = np.where(mass > 0.5)[0]
        assert len(peaks) > 8, f"axis {axis}: rectification lost the grid lines"
        grid = np.arange(0, size + 1, ppt)
        off = np.min(np.abs(peaks[:, None] - grid[None, :]), axis=1)
        assert off.max() <= 2, f"axis {axis}: lines land up to {off.max()}px off the tile grid"


def test_valid_mask_excludes_outside_the_viewport_and_the_hud(tmp_path):
    model, _ = _synth_model(tmp_path)
    hud = HudMask(rects=(HudRect("corner", 0.0, 0.0, 0.25, 0.25),))
    bare = build_rectify_plan(model)
    with_hud = build_rectify_plan(model, hud=hud)
    assert with_hud.valid.sum() < bare.valid.sum()
    assert not with_hud.valid[~bare.valid].any()
    assert not (bare.valid[0, 0] and bare.valid[0, -1]), (
        "the window is the footprint's bounding box, so its corners cannot both be real world"
    )


def test_rectify_rejects_a_frame_that_is_not_the_normalized_viewport(tmp_path):
    model, _ = _synth_model(tmp_path)
    plan = build_rectify_plan(model)
    with pytest.raises(ValueError, match="normalize_viewport"):
        plan.rectify(np.zeros((720, 1280, 3), np.uint8))


def test_a_window_crossing_the_horizon_is_refused(tmp_path):
    """A homography is single-valued only on one side of its horizon; across it the world folds
    back mirrored. This camera's horizon sits thousands of pixels outside the frame, so the guard
    never fires in practice -- which is exactly why it needs a test that makes it fire."""
    model, _ = _synth_model(tmp_path)
    bad = model.H.copy()
    bad[2, :2] *= 40.0
    broken = CameraModel(H=bad, H_inv=np.linalg.inv(bad), H_top=model.H_top,
                         pixels_per_tile=model.pixels_per_tile, viewport=model.viewport,
                         is_affine=False, parallax_tiles=model.parallax_tiles)
    with pytest.raises(ValueError, match="horizon"):
        build_rectify_plan(broken)


def test_pad_tiles_only_adds_invalid_margin(tmp_path):
    model, _ = _synth_model(tmp_path)
    a = build_rectify_plan(model)
    b = build_rectify_plan(model, pad_tiles=2)
    assert b.size_tiles == (a.size_tiles[0] + 4, a.size_tiles[1] + 4)
    assert b.valid.sum() == pytest.approx(a.valid.sum(), rel=0.02)


# --- the shipped model, and real footage ------------------------------------

def test_shipped_plan_matches_the_recorded_window():
    """Pinned because everything downstream is sized by it: a silent change here quietly reshapes
    Phase I's accumulated grid."""
    plan = build_rectify_plan(load_camera_model(), load_hud_mask())
    assert plan.origin_tile == (-8, -8)
    assert plan.size_tiles == (30, 19)
    assert plan.size_px == (1440, 912)
    assert 0.65 < plan.valid.mean() < 0.75


def test_annotated_corners_land_on_the_tile_lattice():
    """Grid phase, on the real calibration frame rather than a synthetic one -- and on INDEPENDENT
    data: `calib_points_b.csv` was annotated from a freshly chosen origin and never entered the
    fit. Because a different origin differs by a whole number of tiles, distance to the nearest
    lattice line is origin-free, so the two sets are directly comparable.

    Set A is the fit's own residual (it should be small); Set B is the number that means
    something -- what a second annotation pass reproduces.

    Both are measured on the WALL-TOP plane, because that is where the clicked corners physically
    are. Checking them against the ground lattice would fail by the ~0.9-tile parallax and look
    like a grid-phase bug when it is the two planes doing exactly what they should.
    """
    model = load_camera_model()
    ppt = model.pixels_per_tile
    for name, mean_tol, max_tol in [("calib_points_a.csv", 2.5, 5.0),
                                    ("calib_points_b.csv", 9.0, 16.0)]:
        rows = [r for r in csv.DictReader((FRAMES / name).open())
                if not next(iter(r.values())).startswith("#")]
        px = np.array([[float(r["px"]), float(r["py"])] for r in rows], np.float64)
        rect = cv2.perspectiveTransform(
            px.reshape(-1, 1, 2), np.linalg.inv(model.H_top)).reshape(-1, 2) * ppt
        dist = np.linalg.norm(rect - np.round(rect / ppt) * ppt, axis=1)
        assert dist.mean() < mean_tol, f"{name}: mean {dist.mean():.2f}px off the tile lattice"
        assert dist.max() < max_tol, f"{name}: worst point {dist.max():.2f}px off the tile lattice"


@pytest.mark.vision
def test_rectified_content_translates_rigidly_across_the_patch():
    """The property Phase F depends on, and the strongest global check available: the game camera
    translates parallel to the ground, so rectifying with a fixed H must leave ground content
    shifted RIGIDLY. Windows all over the patch are correlated independently and must agree.

    This is what the calibration frame alone cannot show -- a fit can be right by construction at
    its own points and wrong everywhere else.
    """
    plan = build_rectify_plan(load_camera_model(), load_hud_mask())
    images = _frames_at("counted_walking", [286, 298])
    shifts = _window_shifts(plan, _rect_gray(plan, images, 286), _rect_gray(plan, images, 298))
    assert len(shifts) >= 20
    med = np.median(shifts[:, 2:4], 0)
    assert np.hypot(*med) > 20.0, "picked a pair with no motion; the test would pass vacuously"
    dev = np.linalg.norm(shifts[:, 2:4] - med, axis=1)
    kept = shifts[dev < 4.0]
    assert len(kept) >= 0.7 * len(shifts)
    assert kept[:, 2].std() < 1.0 and kept[:, 3].std() < 1.0


@pytest.mark.vision
def test_the_rigidity_check_actually_discriminates():
    """Guards the test above from being vacuous. The same measurement on a deliberately wrong
    rectification -- the perspective row dropped, i.e. treating the camera as orthographic -- must
    be markedly worse. Without this, a rigidity assertion that any homography would pass reads as
    evidence when it is not.

    Scored by the GRADIENT of the shift field rather than its scatter: a wrong homography does not
    add noise, it adds a smooth trend across the patch, and the scatter barely notices.
    """
    model = load_camera_model()
    flat = model.H.copy()
    flat[2, :2] = 0.0
    flat = flat / flat[2, 2]
    broken = CameraModel(H=flat, H_inv=np.linalg.inv(flat), H_top=model.H_top,
                         pixels_per_tile=model.pixels_per_tile, viewport=model.viewport,
                         is_affine=True, parallax_tiles=model.parallax_tiles)
    images = _frames_at("counted_walking", [286, 306])
    trends = []
    for plan in (build_rectify_plan(model, load_hud_mask()),
                 build_rectify_plan(broken, load_hud_mask())):
        s = _window_shifts(plan, _rect_gray(plan, images, 286), _rect_gray(plan, images, 306))
        trends.append(_shift_field_trend(plan, s))
    assert trends[1] > 2.5 * trends[0], (
        f"ground fit drifts {trends[0]:.1f}px across the patch, orthographic {trends[1]:.1f}px -- "
        "the check no longer distinguishes a correct rectification from a wrong one"
    )


# ---------------------------------------------------------------------------
# Phase K -- the simulator's view is derived from this homography, so it can go stale
# ---------------------------------------------------------------------------

# Hero's ground point, measured in Phase K over frames where the camera is actively TRACKING the
# player: +0.09 tiles horizontally (IQR +/-0.3) and +0.80 below it vertically (IQR [0.71, 0.85]).
# The condition matters -- the real camera does not always centre the hero, and pooling the
# stretches where it is not tracking moves this number by tiles. The vertical figure sets view_h.
_HERO_OFFSET_TILES = (0.0, 0.80)

# The window is pinned to the hero's CELL (`_view_origin` is `hero_ix - view_w // 2`), so the hero
# sits at some fraction f of the way across it and the window slides by up to a whole tile. Every
# placement has to fit, not just the nominal f=0.5 one: checking the centre alone would accept
# 23x13, which misses by half a tile whenever the hero drifts to one side of its cell.
_SUB_CELL = (0.0, 0.25, 0.5, 0.75, 1.0)


def _camera_geometry():
    """The ground quad the camera sees and the hero's place in it, both in tiles."""
    from brawl_vision.camera import load_camera_model
    model = load_camera_model()
    w, h = model.viewport
    quad = model.px_to_tile(np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64))
    hero = model.px_to_tile(np.array([[w / 2.0, h / 2.0]], np.float64))[0] + _HERO_OFFSET_TILES
    return quad, hero


def _depth(pts, quad):
    """Perpendicular distance from each point to the nearest quad edge, in tiles; negative means
    outside. The quad is a TRAPEZOID, so an axis-by-axis bounds check is not equivalent -- it
    would accept corners falling outside the slanted left/right edges, which is precisely the
    region the plan's inscribed-vs-bounding argument is about."""
    pts = np.atleast_2d(np.asarray(pts, float))
    worst = np.full(len(pts), np.inf)
    sign = None
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        e = b - a
        cross = (e[0] * (pts[:, 1] - a[1]) - e[1] * (pts[:, 0] - a[0])) / np.hypot(*e)
        if sign is None:
            sign = np.sign(cross[np.argmax(np.abs(cross))])
        worst = np.minimum(worst, cross * sign)
    return worst


def _view_depth(view_w, view_h, quad, hero, fractions):
    """Worst corner depth of the egocentric window over the given sub-cell placements."""
    kw, kh = view_w // 2, view_h // 2
    out = np.inf
    for fx in fractions:
        for fy in fractions:
            x_lo, x_hi = hero[0] - (kw + fx), hero[0] + (view_w - kw - fx)
            y_lo, y_hi = hero[1] - (kh + fy), hero[1] + (view_h - kh - fy)
            corners = np.array([[x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi]])
            out = min(out, _depth(corners, quad).min())
    return out


def test_the_sim_view_fits_inside_what_the_camera_can_actually_see():
    """`configs/default.yaml`'s `view` is DERIVED from this homography (plan Phase K), and nothing
    else connects the two files. A recalibration that narrowed the field of view would leave the
    sim quietly training on world the camera can no longer deliver -- the classic sim-to-real trap,
    and invisible because both files stay individually valid."""
    from brawl_sim.config import load_config

    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "default.yaml")
    quad, hero = _camera_geometry()
    d = _view_depth(cfg.view_w, cfg.view_h, quad, hero, _SUB_CELL)
    assert d > -0.05, (
        f"view {cfg.view_w}x{cfg.view_h} reaches {-d:.3f} tiles outside the camera footprint; the "
        f"sim would show world the vision stack cannot supply. The tolerance is the hero anchor's "
        f"own uncertainty (+/-0.05 tiles), not slack."
    )


def test_the_next_size_up_misses_by_a_wide_margin_not_a_rounding_error():
    """21x13 clears the footprint by 0.024 tiles, which on its own reads as a coin flip. It is not.

    The next odd size on either axis misses by 0.54 tiles (width) and 0.98 (height) -- twenty to
    forty times the margin 21x13 has, far outside anything the anchor's uncertainty could
    explain. So the tightness is a
    property of where the quantisation lands, not evidence that the answer is arbitrary. Without
    this the small positive margin above invites someone to 'round up' to 23.
    """
    from brawl_sim.config import load_config

    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "default.yaml")
    quad, hero = _camera_geometry()
    for w, h, name in ((cfg.view_w + 2, cfg.view_h, "view_w"), (cfg.view_w, cfg.view_h + 2, "view_h")):
        d = _view_depth(w, h, quad, hero, _SUB_CELL)
        assert d < -0.25, (
            f"{name} one size up misses by only {-d:.3f} tiles -- that is close enough that the "
            f"choice between them is being made by anchor noise, not by the measurement"
        )


def test_the_hero_sits_exactly_in_the_middle_of_the_sim_view():
    """Odd dimensions, per Phase K. `_view_origin` is `hero_ix - view_w // 2`, so an even width
    puts the hero one column off-centre -- harmless in itself, but it makes every left/right
    symmetry in the observation subtly false, and the old 40x20 had it on both axes."""
    from brawl_sim.config import load_config

    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "default.yaml")
    assert cfg.view_w % 2 == 1 and cfg.view_h % 2 == 1, (cfg.view_w, cfg.view_h)
