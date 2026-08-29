"""Terrain debug overlay. See Terrain_Perception_Build_Plan.md Phase E.

Almost all of this runs without footage: the overlay renders whatever it is handed, so synthetic
`StageFrame`s exercise every panel including the ones whose stages do not exist yet. Only the
end-to-end script test needs a real clip.
"""
import matplotlib
matplotlib.use("Agg")  # headless: never open a GUI window/event loop during tests

import numpy as np
import pytest
from matplotlib.colors import to_rgba

from brawl_sim.constants import Tile
from brawl_sim.render.viewer import TILE_COLORS
from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
from brawl_vision.terrain import overlay as overlay_mod
from brawl_vision.terrain.occupancy import UNKNOWN
from brawl_vision.terrain.overlay import (
    OCCUPANCY_CMAP, UNKNOWN_COLOR, StageFrame, TerrainOverlay,
)


@pytest.fixture(scope="module")
def plan():
    return build_rectify_plan(load_camera_model(), load_hud_mask())


@pytest.fixture
def ov(plan):
    o = TerrainOverlay(plan, load_hud_mask(), load_camera_model())
    yield o
    matplotlib.pyplot.close(o.fig)


def _blank(shape):
    return np.zeros(shape, np.uint8)


# ---------------------------------------------------------------------------
# the palette, which is shared with the simulator on purpose
# ---------------------------------------------------------------------------

def test_palette_comes_from_one_table_not_a_second_copy():
    """This used to require the SIMULATOR's colours, so a reconstruction and a replay were
    comparable by eye. That was given up deliberately -- the vision output is looked at next to
    game footage, and a palette tuned for that reads differently.

    What still has to hold is that there is exactly one table. A hand-copied palette would satisfy
    any colour assertion written here and drift the moment the shared one changed, so what is
    checked is that the values come from `terrain.palette`."""
    from brawl_vision.terrain import palette
    for tile in Tile:
        assert to_rgba(OCCUPANCY_CMAP(int(tile) - UNKNOWN)) == to_rgba(palette.TILE_COLORS[tile])


def test_unknown_renders_as_none_of_the_five_terrain_colours():
    """UNKNOWN exists so that "never observed" is visibly different from "observed, and it is
    floor". If it rendered as anything close to a real class the distinction would be invisible
    exactly when it matters -- a half-explored map.

    This is the constraint that sets how dark FLOOR may be: with UNKNOWN black, a near-black floor
    fails here, and it caught exactly that on the first pass at the current palette.
    """
    from brawl_vision.terrain import palette
    unknown = np.array(to_rgba(palette.UNKNOWN_COLOR))
    for tile in Tile:
        d = np.abs(unknown - np.array(to_rgba(palette.TILE_COLORS[tile]))).max()
        assert d > 0.3, f"{tile.name} is only {d:.3f} from UNKNOWN"


def test_unknown_maps_to_the_first_colormap_entry():
    assert to_rgba(OCCUPANCY_CMAP(UNKNOWN - UNKNOWN)) == to_rgba(UNKNOWN_COLOR)


def test_unknown_is_not_a_tile_member():
    """Guards the plan's Section 2 decision. Widening `Tile` would silently change `N_TILES`,
    which indexes the simulator's `_bool_table`, its colormap and the obs-schema tables."""
    assert UNKNOWN not in [int(t) for t in Tile]
    assert not hasattr(Tile, "UNKNOWN")


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

def test_builds_with_no_stage_output_at_all(ov):
    """The layout must be complete before the pipeline is. Panels for stages that do not exist
    say so rather than being absent, so wiring one up later is filling in a field, not redoing
    the figure."""
    assert len(ov.fig.axes) == 6
    assert {ax.get_title().split(" — ")[-1] for ax in ov._pending} == {"Phase G", "Phase I",
                                                                       "Phase F"}
    assert all(t.get_visible() for t in ov._pending.values())


def test_capture_and_rect_panels_take_their_images(ov, plan):
    vw, vh = plan.viewport
    rw, rh = plan.size_px
    frame = StageFrame(index=1, raw=_blank((vh, vw, 3)), rect=_blank((rh, rw, 3)))
    touched = ov.update(frame)
    assert ov.im_capture in touched and ov.im_rect in touched
    assert ov.im_capture.get_array().shape == (vh, vw, 3)


def test_images_are_converted_from_bgr(ov, plan):
    """Everything OpenCV produces is BGR and everything matplotlib draws is RGB. Getting this
    wrong is invisible on grey test data and swaps water for something orange on real footage."""
    vw, vh = plan.viewport
    raw = np.zeros((vh, vw, 3), np.uint8)
    raw[..., 0] = 255                      # pure blue in BGR
    ov.update(StageFrame(index=1, raw=raw))
    drawn = np.asarray(ov.im_capture.get_array())[0, 0]
    assert tuple(drawn) == (0, 0, 255), "channels were not swapped to RGB"


def test_a_stage_arriving_clears_its_placeholder(ov, plan):
    assert ov.ax_occ in ov._pending
    grid = np.full(plan.size_tiles[::-1], UNKNOWN, np.int16)
    ov.update(StageFrame(index=1, occupancy=grid))
    assert ov.ax_occ not in ov._pending
    assert ov.im_occ.get_visible()
    assert "not built yet" not in ov.ax_occ.get_title()


def test_occupancy_is_shifted_so_unknown_is_in_range(ov, plan):
    """`imshow` clips to vmin/vmax, so a raw -1 would render as UNKNOWN's neighbour rather than
    UNKNOWN. The shift is what keeps the whole domain inside the colormap."""
    grid = np.full(plan.size_tiles[::-1], UNKNOWN, np.int16)
    grid[0, 0] = int(Tile.WALL)
    ov.update(StageFrame(index=1, occupancy=grid))
    drawn = np.asarray(ov.im_occ.get_array())
    assert drawn.min() == 0 and drawn[0, 0] == int(Tile.WALL) - UNKNOWN
    assert ov.im_occ.get_clim() == (0, len(Tile))


def test_camera_track_accumulates_and_is_bounded(ov):
    for i in range(overlay_mod._TRACK_TAIL + 50):
        ov.update(StageFrame(index=i, camera_tile=(i * 0.1, -i * 0.05)))
    assert len(ov._track) == overlay_mod._TRACK_TAIL, "the track grows without bound"
    assert ov.dot_cam.get_data()[0][0] == pytest.approx((overlay_mod._TRACK_TAIL + 49) * 0.1)


def test_status_line_reports_frame_and_stage_timings(ov):
    ov.update(StageFrame(index=42, t=1.5, timings={"rectify": 0.9, "decode": 5.1}))
    text = ov._status.get_text()
    assert "42" in text and "1.50" in text
    assert "rectify" in text and "total" in text


def test_both_tile_grids_are_drawn_when_a_model_is_given(ov, plan):
    """The ground grid alone reads as a near-miss against the block-top seams, because walls are
    ~0.88 tiles tall in projected terms. Drawn beside the wall-top grid, that gap is legible as
    the parallax it is -- so losing the second grid makes a correct calibration look broken."""
    from matplotlib.collections import LineCollection
    grids = [c for c in ov.ax_capture.collections if isinstance(c, LineCollection)]
    assert len(grids) == 2
    colours = {tuple(np.round(c.get_edgecolor()[0][:3], 3)) for c in grids}
    assert {tuple(np.round(to_rgba("#ffd60a")[:3], 3)),
            tuple(np.round(to_rgba("#00e5ff")[:3], 3))} == colours


def test_each_grid_lies_on_its_own_plane(plan):
    """Not a rendering check: it verifies the two collections really are the two PLANES.

    Each grid's lines must land on integer tile coordinates *of the plane it claims to be* --
    which is exact and origin-independent, unlike comparing the gap between them against
    `parallax_tiles` (that is a mean over the viewport, and these lines run well outside it).
    Cross-checking each set against the other plane is what makes the test discriminating: a
    second collection accidentally drawn from the ground homography would pass the first half.
    """
    import cv2
    model = load_camera_model()
    ov = TerrainOverlay(plan, load_hud_mask(), model)
    try:
        ground, top = [np.array([s[0] for s in c.get_segments()])
                       for c in ov.ax_capture.collections[-2:]]
    finally:
        matplotlib.pyplot.close(ov.fig)

    def off_lattice(px, H):
        tiles = cv2.perspectiveTransform(px.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
        # Measured from the window's origin tile, which is (-8, -8) and so is NOT itself a
        # multiple of the grid step -- the lattice is anchored to the origin, not to zero.
        step = overlay_mod._GRID_EVERY_TILES
        rel = tiles - np.asarray(plan.origin_tile, np.float64)
        return np.abs(rel - np.round(rel / step) * step).max()

    assert off_lattice(ground, model.H) < 1e-6
    assert off_lattice(top, model.H_top) < 1e-6
    assert off_lattice(top, model.H) > 0.2, "the wall-top grid was drawn from the ground plane"


def test_legend_omits_the_marker_tiles(ov):
    """SPAWN and BOX are placement markers, not terrain, and can never be classifier outputs
    (plan Section 2). They also share FLOOR's colour, so listing them would be both wrong and
    visually meaningless."""
    labels = {t.get_text() for t in ov.ax_legend.get_legend().get_texts()}
    assert {"FLOOR", "WALL", "BUSH", "WATER", "FENCE", "UNKNOWN"} <= labels
    assert "SPAWN" not in labels and "BOX" not in labels


def test_model_is_optional(plan):
    ov = TerrainOverlay(plan, load_hud_mask())
    try:
        from matplotlib.collections import LineCollection
        assert len([c for c in ov.ax_capture.collections if isinstance(c, LineCollection)]) == 1
    finally:
        matplotlib.pyplot.close(ov.fig)


# ---------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------

def test_save_writes_a_gif(ov, plan, tmp_path):
    rw, rh = plan.size_px
    out = tmp_path / "clip.gif"
    ov.save(out, (StageFrame(index=i, rect=_blank((rh, rw, 3))) for i in range(3)),
            fps=4, dpi=40)
    assert out.stat().st_size > 0


def test_save_refuses_an_empty_source(ov, tmp_path):
    """A streamed source that yields nothing produced a zero-frame file rather than an error --
    silently, so the first sign was an unopenable video minutes later."""
    with pytest.raises(ValueError, match="empty"):
        ov.save(tmp_path / "empty.gif", iter(()), fps=4, dpi=40)


def test_save_rejects_an_unknown_extension(ov, tmp_path):
    with pytest.raises(ValueError, match="unsupported video extension"):
        ov.save(tmp_path / "clip.avi", iter((StageFrame(index=0),)), dpi=40)


# ---------------------------------------------------------------------------
# the script
# ---------------------------------------------------------------------------

def test_script_requires_exactly_one_source():
    from scripts import vision_watch
    for argv in ([], ["clip.mp4", "--live"]):
        with pytest.raises(SystemExit):
            vision_watch.main(argv)


@pytest.mark.vision
def test_script_renders_a_real_clip_end_to_end(tmp_path):
    """Phase E's acceptance, for the stages that exist: every implemented panel updates in sync
    over a real clip."""
    from scripts import vision_watch
    out = tmp_path / "watch.gif"
    rc = vision_watch.main([
        "tests/fixtures/vision/counted_walking.mp4",
        "--start", "300", "--stop", "304", "--fps", "4", "--save", str(out),
    ])
    assert rc == 0
    assert out.stat().st_size > 0


def test_play_streams_to_completion(ov, plan):
    """The interactive path, exercised headlessly. It is the one users actually run, and it is
    easy to leave broken because every other test drives `update()` directly."""
    rw, rh = plan.size_px
    ov.play((StageFrame(index=i, rect=_blank((rh, rw, 3))) for i in range(3)), fps=60)
    assert "frame     2" in ov._status.get_text()


def test_the_camera_track_follows_a_long_walk(plan):
    """The panel showed +-2 tiles through a 22-tile walk, for the whole run.

    `set_xlim` DISABLES autoscaling on that axis. A minimum-span floor applied on the first frame
    -- one point, zero span -- therefore froze the limits permanently. Autoscale is not used at
    all now; the limits come from the data.
    """
    ov = TerrainOverlay(plan)
    try:
        for i in range(300):
            ov.update(StageFrame(index=i, camera_tile=(-22.0 * i / 299, -15.0 * i / 299)))
        lo, hi = ov.ax_cam.get_xlim()
        assert hi - lo > 20.0, f"a 22-tile walk rendered in a {hi - lo:.1f}-tile window"
        top, bottom = ov.ax_cam.get_ylim()
        assert top > bottom, "tile y must render downward, as in the simulator"
    finally:
        matplotlib.pyplot.close(ov.fig)


def test_a_stationary_camera_still_gets_a_readable_track_window(plan):
    """The case the floor exists for: a dead-straight or motionless track must not autoscale into
    six-decimal ticks around a zero-width range."""
    ov = TerrainOverlay(plan)
    try:
        for i in range(20):
            ov.update(StageFrame(index=i, camera_tile=(0.0, 0.0)))
        lo, hi = ov.ax_cam.get_xlim()
        assert hi - lo >= overlay_mod._TRACK_MIN_SPAN_TILES
    finally:
        matplotlib.pyplot.close(ov.fig)
