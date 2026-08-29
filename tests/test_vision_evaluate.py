"""The evaluation viewer: gameplay mp4 in, reconstructed-map mp4 out. See
Terrain_Perception_Build_Plan.md Phase L.

Split the way the other vision tests are: the canvas geometry and the video sink are exercised on
synthetic data so they run on a fresh clone, and anything that needs real footage or a real encoder
is marked `vision` and skips without it.
"""
from pathlib import Path

import numpy as np
import pytest

from brawl_sim.constants import TILE_TO_CHAR
from brawl_sim.render.viewer import TILE_COLORS
from brawl_vision.terrain import evaluate as ev
from brawl_vision.terrain.labeling import CLASSES
from brawl_vision.terrain.occupancy import UNKNOWN, OccupancyMap
from brawl_vision.video import VideoSink, ffmpeg_available

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "vision"


class FakePlan:
    """The three things the canvas maths reads off a `RectifyPlan`, and nothing else.

    A real plan needs the shipped homography, which is gitignored footage's neighbour rather than a
    dependency this file's geometry has any business on.
    """
    def __init__(self, origin_tile=(-8, -8), size_tiles=(30, 19), pixels_per_tile=48):
        self.origin_tile = origin_tile
        self.size_tiles = size_tiles
        self.pixels_per_tile = pixels_per_tile
        cols, rows = size_tiles
        self.size_px = (cols * pixels_per_tile, rows * pixels_per_tile)
        self.valid = np.ones((self.size_px[1], self.size_px[0]), bool)


def _track(positions):
    t = ev.Track()
    for i, p in enumerate(positions):
        t.positions[i] = p
        t.segments[i] = 0
        t.statuses[i] = "ok"
        t.times[i] = i / 60.0
    t.n_frames = len(positions)
    return t


# ---------------------------------------------------------------------------
# the colour table
# ---------------------------------------------------------------------------

def test_lut_puts_unknown_at_index_zero_so_minus_one_does_not_wrap():
    """`best()` uses -1 for never-observed, and `lut[best]` would read that as the LAST row --
    silently painting unexplored ground in whatever the final class happens to be. The `+ 1` shift
    is the whole reason this table has an extra row, so it is worth a test that would notice if
    someone 'simplified' it away."""
    lut = ev.tile_lut()
    assert lut.shape == (len(CLASSES) + 1, 3)
    from matplotlib.colors import to_rgb
    expected = (np.array(to_rgb(ev.UNKNOWN_COLOR)) * 255).round().astype(np.uint8)
    assert np.array_equal(lut[UNKNOWN + 1], expected)


def test_lut_comes_from_the_vision_palette_and_not_the_simulator():
    """The vision map used to borrow `brawl_sim.render.viewer.TILE_COLORS` so a reconstruction and
    a simulator render could be compared by eye. That was given up deliberately: this output is
    looked at next to game FOOTAGE, and the palette that reads well there is a different one.

    What must not happen is the colours drifting back into three viewers separately, so this pins
    the single source rather than the specific hues.
    """
    from brawl_sim.render.viewer import TILE_COLORS as SIM
    from brawl_vision.terrain import palette
    from matplotlib.colors import to_rgb

    lut = ev.tile_lut()
    for i, tile in enumerate(CLASSES):
        want = (np.array(to_rgb(palette.TILE_COLORS[tile])) * 255).round().astype(np.uint8)
        assert np.array_equal(lut[i + 1], want), tile.name
    assert palette.TILE_COLORS[CLASSES[0]] != SIM[CLASSES[0]], (
        "the vision palette has silently become the simulator's again"
    )


def test_the_two_viewers_share_one_palette():
    """Phase E's overlay and Phase L's renderer draw the same map. Two tables that agree today and
    drift tomorrow is the failure this is here to prevent."""
    from brawl_vision.terrain import overlay, palette
    assert overlay.TILE_COLORS is palette.TILE_COLORS
    assert overlay.UNKNOWN_COLOR == palette.UNKNOWN_COLOR == ev.UNKNOWN_COLOR


def test_every_class_is_visually_separable_from_every_other():
    """The property a palette actually has to have. Two near-identical greys would pass any check
    that only asserts 'the colours are what the table says' while making the viewer useless -- and
    FLOOR against UNKNOWN is the pair most at risk, since one is a dark grey and the other black.

    The floor of 60 is set below the measured minimum (80.1, WATER against FENCE) with room, so
    this fails on a genuinely bad choice rather than on any change at all.
    """
    import itertools
    lut = ev.tile_lut().astype(float)
    names = ["UNKNOWN"] + [t.name for t in CLASSES]
    for i, j in itertools.combinations(range(len(lut)), 2):
        d = float(np.linalg.norm(lut[i] - lut[j]))
        assert d >= 60.0, f"{names[i]} and {names[j]} are only {d:.0f} apart in RGB"


# ---------------------------------------------------------------------------
# canvas sizing -- the whole point of pass one
# ---------------------------------------------------------------------------

def test_window_covers_every_cell_the_track_could_deposit_into():
    """A frame at camera position p deposits `plan.origin_tile + p` through `+ size_tiles`. If the
    window misses any of that, the output silently crops real map -- which is indistinguishable
    from terrain the camera never saw, and that distinction is the entire artifact."""
    plan = FakePlan()
    track = _track([(0.0, 0.0), (12.0, -5.0), (-7.0, 9.0)])
    w = ev.world_window(track, plan, grid_h=128, grid_w=128)
    ox, oy = plan.origin_tile
    cols, rows = plan.size_tiles
    gx, gy = -(128 // 2), -(128 // 2)
    for px, py in track.positions.values():
        c0 = int(round(px + ox)) - gx
        r0 = int(round(py + oy)) - gy
        assert w.col0 <= c0 and c0 + cols <= w.col1, (px, py)
        assert w.row0 <= r0 and r0 + rows <= w.row1, (px, py)


def test_window_is_clamped_to_the_grid():
    """A long walk in one direction runs the track off the fixed 128x128 occupancy grid. The window
    is an index range into that grid, so it has to stop at its edge rather than describe cells that
    do not exist."""
    plan = FakePlan()
    track = _track([(0.0, 0.0), (400.0, 400.0)])
    w = ev.world_window(track, plan, grid_h=128, grid_w=128)
    assert 0 <= w.row0 < w.row1 <= 128
    assert 0 <= w.col0 < w.col1 <= 128


def test_window_carries_a_margin_because_pass_one_only_samples():
    """Pass one reads every Nth frame, so its bounding box is an estimate of the track, not the
    track. The margin is what absorbs that and the drift a coarser odometry chain accumulates --
    without it the canvas is exactly as wrong as pass one's sampling."""
    plan = FakePlan()
    tight = ev.world_window(_track([(0.0, 0.0)]), plan, 128, 128, margin=0)
    padded = ev.world_window(_track([(0.0, 0.0)]), plan, 128, 128, margin=3)
    assert padded.rows == tight.rows + 6 and padded.cols == tight.cols + 6


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _occupancy_with(cells):
    occ = OccupancyMap(height=32, width=32)
    for (r, c), klass in cells.items():
        occ.votes[r, c, klass] = 9
    return occ


def test_render_map_paints_unknown_where_nothing_was_observed():
    occ = _occupancy_with({(4, 4): 0})
    w = ev.Window(row0=0, row1=8, col0=0, col1=8)
    img = ev.render_map(occ, w, ev.tile_lut(), scale=3)
    assert img.shape == (24, 24, 3)
    assert np.array_equal(img[0, 0], ev.tile_lut()[0])          # UNKNOWN
    assert np.array_equal(img[13, 13], ev.tile_lut()[1])        # the one observed cell


def test_render_map_never_invents_a_colour_between_two_tiles():
    """`INTER_NEAREST` is not a speed choice. Any interpolation blends the wall colour into the
    floor beside it, and a viewer reads that gradient as classifier uncertainty -- a claim the map
    is not making. Every output pixel must be a row of the table, exactly."""
    occ = _occupancy_with({(2, 2): 1, (2, 3): 0, (3, 2): 3})
    w = ev.Window(row0=0, row1=6, col0=0, col1=6)
    lut = ev.tile_lut()
    img = ev.render_map(occ, w, lut, scale=7)
    seen = np.unique(img.reshape(-1, 3), axis=0)
    for colour in seen:
        assert any(np.array_equal(colour, row) for row in lut), colour


def test_render_map_crops_to_the_window():
    occ = _occupancy_with({(1, 1): 0, (20, 20): 1})
    w = ev.Window(row0=0, row1=4, col0=0, col1=4)
    img = ev.render_map(occ, w, ev.tile_lut(), scale=2)
    assert img.shape == (8, 8, 3)


# ---------------------------------------------------------------------------
# the mosaic -- the odometry test the plan cares about
# ---------------------------------------------------------------------------

def _mosaic(scale=2):
    plan = FakePlan(origin_tile=(0, 0), size_tiles=(4, 4), pixels_per_tile=8)
    w = ev.Window(row0=0, row1=12, col0=0, col1=12)
    return ev.Mosaic(w, plan, grid_origin=(0, 0), scale=scale), plan


def test_mosaic_averages_rather_than_overwrites():
    """Overwriting shows the newest patch and hides disagreement. Averaging makes a wall seen at
    two odometry offsets appear twice at half weight, which is what drift LOOKS like -- and looking
    is the only thing this mode is for."""
    m, plan = _mosaic()
    white = np.full((plan.size_px[1], plan.size_px[0], 3), 255, np.uint8)
    black = np.zeros_like(white)
    m.add(white, (0.0, 0.0))
    m.add(black, (0.0, 0.0))
    assert m.image()[0, 0].tolist() == [127, 127, 127] or m.image()[0, 0].tolist() == [128, 128, 128]


def test_mosaic_places_a_patch_at_its_odometry_offset():
    m, plan = _mosaic()
    white = np.full((plan.size_px[1], plan.size_px[0], 3), 255, np.uint8)
    m.add(white, (5.0, 3.0))
    img = m.image()
    assert img[3 * 2 + 1, 5 * 2 + 1].max() > 0, "patch did not land at (5, 3) tiles"
    assert img[0, 0].max() == 0, "something was painted at the origin"


def test_a_patch_lands_at_its_sub_tile_position_not_the_rounded_one():
    """A regression test for a bug that no existing test could have caught.

    `place` used to round the camera position to whole tiles, discarding the fractional part Phase F
    measures. Measured on real footage the discarded remainder averaged 0.27 tiles and 80% of frames
    landed more than a quarter tile out -- so every composite built from this was averaging samples
    misregistered by ~12 px in random directions, and came out a smear.

    It survived because every mosaic test placed patches at INTEGER positions, where the rounded
    and exact answers agree. A half-tile offset is the smallest case that tells them apart.
    """
    m, plan = _mosaic(scale=8)
    white = np.full((plan.size_px[1], plan.size_px[0], 3), 255, np.uint8)
    m.add(white, (0.5, 0.0))
    img = m.image()
    col = img[:, :, 0].max(axis=0)              # brightest pixel in each column
    first_lit = int(np.argmax(col > 0))
    assert first_lit == 4, (
        f"content starts at x={first_lit}; a 0.5-tile offset at 8 px/tile must start at 4, and "
        f"starting at 0 means the fractional position was rounded away"
    )


def test_a_sub_tile_shift_does_not_smear_the_patch_border_into_the_canvas():
    """The mask has to move with the pixels and be re-thresholded. Warping it loosely leaves
    bilinear edge pixels that are part border fill, and those get painted in as if they were
    world -- a bright rim around every frame's footprint, accumulating into the map."""
    m, plan = _mosaic(scale=8)
    white = np.full((plan.size_px[1], plan.size_px[0], 3), 255, np.uint8)
    m.add(white, (0.5, 0.5))
    lit = m.image()[:, :, 0]
    partial = ((lit > 0) & (lit < 250)).sum()
    assert partial == 0, f"{partial} pixels are part border fill, part world"


def test_mosaic_clips_at_the_canvas_edge_instead_of_wrapping():
    """A frame at the edge of the track hangs off the canvas by a tile. Slicing with a negative
    index would wrap it to the OPPOSITE side, painting a copy of the map's east edge onto its west
    -- a corruption that reads as odometry drift and is nothing of the sort."""
    m, plan = _mosaic()
    white = np.full((plan.size_px[1], plan.size_px[0], 3), 255, np.uint8)
    m.add(white, (-3.0, -3.0))
    img = m.image()
    assert img[-1, -1].max() == 0, "content wrapped round to the far corner"
    assert img[0, 0].max() > 0, "the part that IS on the canvas was dropped"


def test_mosaic_ignores_pixels_the_plan_calls_invalid():
    """`plan.valid` is the HUD mask and the trapezoid footprint together. Pasting outside it puts
    screen-fixed furniture into a world-fixed canvas, where it smears across the map."""
    m, plan = _mosaic()
    m._valid[:] = False
    white = np.full((plan.size_px[1], plan.size_px[0], 3), 255, np.uint8)
    m.add(white, (0.0, 0.0))
    assert m.image().max() == 0


# ---------------------------------------------------------------------------
# layout and reporting
# ---------------------------------------------------------------------------

def test_side_by_side_matches_heights_and_keeps_both():
    src = np.zeros((100, 200, 3), np.uint8)
    canvas = np.full((60, 80, 3), 255, np.uint8)
    out = ev._side_by_side(src, canvas)
    assert out.shape[0] == 60
    assert out.shape[1] == 80 + 120


def test_the_view_crop_takes_the_tiles_the_frame_actually_covers():
    """The `view` layout only means anything if the map panel shows the SAME world as the frame
    panel. Put a known class at a known world cell, put the camera somewhere non-trivial, and check
    it lands where the frame's own geometry says it should."""
    plan = FakePlan(origin_tile=(-2, -1), size_tiles=(4, 3), pixels_per_tile=8)
    occ = OccupancyMap(height=16, width=16)
    gx, gy = occ.origin
    # A camera at (5, 4) with origin_tile (-2, -1) covers world x 3..6, y 3..5. So world tile
    # (x=3, y=3) is the crop's top-left cell -- and world (x=2, y=3), one column further left, is
    # NOT in view at all, which is the case that catches an off-by-one in the wrong direction.
    occ.votes[3 - gy, 3 - gx, 1] = 9                 # WALL at world (3, 3)
    img = ev.crop_to_view(occ, plan, (5.0, 4.0), ev.tile_lut(), scale=4)
    assert img.shape == (3 * 4, 4 * 4, 3)
    assert np.array_equal(img[2, 2], ev.tile_lut()[2]), "the WALL cell is not at the crop's origin"
    assert np.array_equal(img[2, 6], ev.tile_lut()[0]), "its neighbour should be UNKNOWN"

    occ2 = OccupancyMap(height=16, width=16)
    occ2.votes[3 - gy, 2 - gx, 1] = 9                # WALL one tile left of what the camera sees
    img2 = ev.crop_to_view(occ2, plan, (5.0, 4.0), ev.tile_lut(), scale=4)
    assert (img2 == ev.tile_lut()[0]).all(), "a cell outside the footprint leaked into the crop"


def test_the_view_crop_returns_unknown_past_the_edge_of_the_grid():
    """Near the boundary part of the footprint has no storage behind it. That is the same 'nothing
    is known here' the rest of the map already expresses, not an error to raise."""
    plan = FakePlan(origin_tile=(0, 0), size_tiles=(4, 3), pixels_per_tile=8)
    occ = OccupancyMap(height=16, width=16)
    img = ev.crop_to_view(occ, plan, (-100.0, -100.0), ev.tile_lut(), scale=2)
    assert img.shape == (3 * 2, 4 * 2, 3)
    assert (img == ev.tile_lut()[0]).all(), "off-grid should be entirely UNKNOWN"


def test_the_policy_window_outline_is_smaller_than_the_footprint():
    """The camera covers 30x19 tiles; the policy receives 21x13 of it (Phase K). Drawing the
    smaller box is what separates 'the map looks right' from 'the map looks right where the agent
    is actually reading it'."""
    plan = FakePlan(origin_tile=(0, 0), size_tiles=(30, 19), pixels_per_tile=48)
    canvas = np.zeros((19 * 4, 30 * 4, 3), np.uint8)
    ev.draw_policy_window(canvas, plan, 4, view_w=21, view_h=13)
    ys, xs = np.nonzero(canvas[:, :, 1] > 0)
    assert (xs.max() - xs.min()) < 30 * 4 - 1, "the window is not inside the footprint"
    assert abs((xs.max() - xs.min()) - 21 * 4) <= 3
    assert abs((ys.max() - ys.min()) - 13 * 4) <= 3


def test_the_view_layout_needs_a_classifier():
    """It renders classified tiles, so mosaic mode cannot produce it -- and failing at the argument
    is better than producing 269 frames of magenta."""
    plan = FakePlan()
    with pytest.raises(ValueError, match="classifier"):
        ev.render(iter([]), plan, _track([(0.0, 0.0)]), "x.mp4", classifier=None, layout="view")


def test_the_footprint_outline_marks_where_the_current_frame_is():
    """Side by side is one frame next to the whole explored map, so without this a viewer cannot
    tell which part of the map the frame corresponds to -- and every apparent disagreement becomes
    unfalsifiable. The outline has to land on the tiles the frame actually covers."""
    plan = FakePlan(origin_tile=(0, 0), size_tiles=(4, 4), pixels_per_tile=8)
    win = ev.Window(row0=0, row1=12, col0=0, col1=12)
    canvas = np.zeros((12 * 6, 12 * 6, 3), np.uint8)
    ev.draw_footprint(canvas, win, plan, (0, 0), (5.0, 3.0), scale=6)
    lit = canvas[:, :, 0] > 0
    ys, xs = np.nonzero(lit)
    # Tolerance is the stroke width: a 2 px line centred on the boundary lands a pixel either side.
    # Pinning the exact stroke pixels would make this a change detector for line thickness rather
    # than a check that the outline is in the right place.
    assert abs(xs.min() - 5 * 6) <= 2 and abs(ys.min() - 3 * 6) <= 2, "not at the camera position"
    assert abs(xs.max() - (5 + 4) * 6) <= 2 and abs(ys.max() - (3 + 4) * 6) <= 2, "wrong size"
    assert not lit[3 * 6 + 12, 5 * 6 + 12], "the outline is filled, not an outline"


def test_unknown_monotonic_is_the_odometry_lock_check():
    """Coverage can only grow: a rise in UNKNOWN means the world frame moved under the grid. This
    is the acceptance criterion a per-frame drift number cannot see, so the report has to be able
    to answer it rather than leaving it to someone watching the video."""
    r = ev.RenderReport()
    r.unknown_curve = [(0.0, 100), (1.0, 80), (2.0, 80), (3.0, 60)]
    assert r.unknown_is_monotonic()
    r.unknown_curve = [(0.0, 100), (1.0, 80), (2.0, 95)]
    assert not r.unknown_is_monotonic()


def test_realtime_factor_says_which_way_it_runs():
    r = ev.RenderReport(clip_seconds=80.0, render_seconds=40.0)
    assert r.realtime_factor == 2.0 and r.faster_than_realtime
    r = ev.RenderReport(clip_seconds=80.0, render_seconds=160.0)
    assert r.realtime_factor == 0.5 and not r.faster_than_realtime


def test_outside_window_counts_what_the_canvas_missed():
    """Pass one's window is a prediction. Cells deposited beyond it are invisible in the output and
    would otherwise be indistinguishable from cells the camera never reached."""
    occ = _occupancy_with({(1, 1): 0, (20, 20): 1})
    assert ev._outside(occ, ev.Window(row0=0, row1=8, col0=0, col1=8)) == 1
    assert ev._outside(occ, ev.Window(row0=0, row1=32, col0=0, col1=32)) == 0


# ---------------------------------------------------------------------------
# the video sink
# ---------------------------------------------------------------------------

def test_sink_forces_even_dimensions():
    """`yuv420p` subsamples chroma by two and rejects odd sizes with a message about pixel formats,
    which says nothing about the caller's tile scale. One padded pixel is invisible; failing at
    frame zero after a minute of pass one is not."""
    s = VideoSink("x.mp4", (101, 57))
    assert s.size == (102, 58)


def test_sink_rejects_an_extension_it_cannot_write():
    with pytest.raises(ValueError, match="mp4"):
        VideoSink("x.avi", (64, 64))


def test_sink_rejects_frames_that_are_not_rgb_uint8(tmp_path):
    s = VideoSink(tmp_path / "x.gif", (8, 8))
    with pytest.raises(ValueError, match="uint8"):
        s.write(np.zeros((8, 8, 3), np.float32))


def test_gif_refuses_to_buffer_a_whole_clip(tmp_path):
    """GIF cannot stream -- the format needs the entire animation before it writes anything. A
    4802-frame clip at this canvas size is gigabytes of RAM, and dying on an allocation ten minutes
    in is a worse answer than refusing at frame four."""
    s = VideoSink(tmp_path / "x.gif", (8, 8), max_gif_frames=3)
    with s:
        for _ in range(3):
            s.write(np.zeros((8, 8, 3), np.uint8))
        with pytest.raises(RuntimeError, match="max_gif_frames"):
            s.write(np.zeros((8, 8, 3), np.uint8))


def test_gif_round_trips(tmp_path):
    out = tmp_path / "x.gif"
    with VideoSink(out, (10, 6), fps=5) as s:
        for v in (0, 128, 255):
            s.write(np.full((6, 10, 3), v, np.uint8))
    assert out.exists() and out.stat().st_size > 0
    from PIL import Image
    with Image.open(out) as im:
        assert im.size == (10, 6)


@pytest.mark.skipif(not ffmpeg_available(), reason="no ffmpeg")
def test_mp4_is_written_and_decodes_to_the_frames_it_was_given(tmp_path):
    import cv2
    out = tmp_path / "x.mp4"
    with VideoSink(out, (32, 16), fps=10) as s:
        for _ in range(12):
            s.write(np.full((16, 32, 3), 200, np.uint8))
    assert out.exists() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == 12


# ---------------------------------------------------------------------------
# against real footage
# ---------------------------------------------------------------------------

def _clip(name):
    p = FIXTURES / f"{name}.mp4"
    if not p.exists():
        pytest.skip(f"{p.name} not present (gitignored game footage)")
    return p


@pytest.mark.vision
def test_stepping_keeps_frame_indices_file_relative():
    """Phase L lines up a cheap stepped scan with an expensive full render, and that only works if
    both agree about which frame is which. If `index` counted yielded frames instead, a step-8 read
    would call file frame 24 'frame 3' and the two passes would describe different clips."""
    from brawl_vision.clips import ClipReader
    full = [f.index for f, _ in zip(ClipReader(_clip("standstill")), range(64))]
    stepped = [f.index for f, _ in zip(ClipReader(_clip("standstill"), step=8), range(8))]
    assert full[:8] == list(range(8))
    assert stepped == [i for i in full if i % 8 == 0][:8]


@pytest.mark.vision
@pytest.mark.skipif(not ffmpeg_available(), reason="no ffmpeg")
def test_mosaic_renders_end_to_end_from_a_real_clip(tmp_path):
    """Mosaic mode needs only Phase F, so this runs without a trained classifier -- which is the
    property that made it the milestone to build first."""
    from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
    from brawl_vision.config import load_vision_config
    from brawl_vision.sources import open_source

    cfg = load_vision_config()
    plan = build_rectify_plan(load_camera_model(), load_hud_mask())
    clip = _clip("counted_walking")
    with open_source(clip, cfg, step=8) as src:
        track = ev.scan_track((f for f in src if f.index < 240), plan, cfg)
    assert track.n_frames > 10
    out = tmp_path / "mosaic.mp4"
    with open_source(clip, cfg, step=4) as src:
        report = ev.render((f for f in src if f.index < 240), plan, track, out,
                           cfg=cfg, out_fps=6.0, scale=6)
    assert out.exists() and report.out_frames > 1
    assert report.size[0] > 0 and report.size[1] > 0
    assert report.classified == 0, "mosaic mode must not need the classifier"


@pytest.mark.vision
def test_the_window_pass_one_sizes_actually_holds_the_map(tmp_path):
    """The one assumption two-pass rendering rests on: a stepped scan predicts the extent a full
    render will reach. `_outside` is the check, and it is checked here rather than left to a
    warning nobody reads."""
    from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
    from brawl_vision.config import load_vision_config
    from brawl_vision.sources import open_source

    cfg = load_vision_config()
    plan = build_rectify_plan(load_camera_model(), load_hud_mask())
    clip = _clip("counted_walking")
    with open_source(clip, cfg, step=8) as src:
        track = ev.scan_track((f for f in src if f.index < 400), plan, cfg)
    occ = OccupancyMap.from_config(cfg)
    window = ev.world_window(track, plan, occ.height, occ.width)

    from brawl_vision.terrain.odometry import Odometry
    odo = Odometry(plan, cfg)
    with open_source(clip, cfg) as src:
        for frame in src:
            if frame.index >= 400:
                break
            r = odo.update(plan.rectify(frame.image))
            ox, oy = plan.origin_tile
            gx, gy = occ.origin
            c0 = int(round(r.position_tiles[0] + ox)) - gx
            r0 = int(round(r.position_tiles[1] + oy)) - gy
            cols, rows = plan.size_tiles
            assert window.col0 <= c0 and c0 + cols <= window.col1, frame.index
            assert window.row0 <= r0 and r0 + rows <= window.row1, frame.index
