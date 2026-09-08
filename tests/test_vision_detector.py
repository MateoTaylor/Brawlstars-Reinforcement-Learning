"""The third-party object detector and its drawing. See brawl_vision/object_detection/.

**Split into three tiers, deliberately, because they fail for different reasons.**

* Postprocessing and drawing take synthetic arrays and run everywhere. They are the parts written
  here, so they are the parts tested unconditionally.
* Tests that construct an `ObjectDetector` skip without the weights, which are fetched rather
  than committed (`scripts/vision_fetch_detector.py`) -- a fresh clone has no `.onnx` and that is
  not a failure.
* Tests that run it over real footage carry `@pytest.mark.vision` on top of that, matching the
  rest of the vision suite: the clips are gitignored game capture.

Nothing here asserts an accuracy number. This is somebody else's model with somebody else's
training set, and a threshold on its output would be a regression test against a file this repo
does not control. What IS asserted is everything between the file and the boxes: that the class
names come from the model rather than a constant, that the coordinate transform inverts the
letterbox, that BGR goes in, and that drawing never mutates its input.
"""
import numpy as np
import pytest

from brawl_vision.config import VisionConfig, load_vision_config, validate
from brawl_vision.object_detection import Detection, draw_detections
from brawl_vision.object_detection import weights as weights_mod
from brawl_vision.object_detection.detector import _nms
from brawl_vision.object_detection.project import draw_markers, to_tiles, to_tile_quads
from brawl_vision.object_detection.draw import (
    CLASS_COLORS, _sizes, color_for, draw_summary,
)

CLIP = "tests/fixtures/vision/showdown_alternate_map.mp4"


def _has_weights() -> bool:
    return weights_mod.weights_path(weights_mod.DEFAULT_MODEL).exists()


needs_weights = pytest.mark.skipif(
    not _has_weights(),
    reason="detector weights are fetched, not committed -- scripts/vision_fetch_detector.py",
)


@pytest.fixture(scope="module")
def detector():
    pytest.importorskip("onnxruntime")
    from brawl_vision.object_detection import ObjectDetector
    return ObjectDetector.from_config(load_vision_config(), ignore=())


# ---------------------------------------------------------------------------
# Detection: the geometry a caller reads off a box
# ---------------------------------------------------------------------------

def test_ground_point_is_the_bottom_of_the_box_not_its_centre():
    """The tile a brawler stands on is under its FEET. A sprite is drawn with height, so the box
    centre projects to a tile behind the one being occupied -- which is the whole reason this is
    a named property and not left to each caller."""
    det = Detection("enemy", 0.9, (100.0, 200.0, 200.0, 400.0))
    assert det.centre == (150.0, 300.0)
    assert det.ground_point == (150.0, 400.0)


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def test_nms_keeps_the_best_of_a_stack_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 12, 12]], np.float32)
    scores = np.array([0.7, 0.9, 0.6], np.float32)
    assert _nms(boxes, scores, 0.5).tolist() == [1]


def test_nms_keeps_boxes_that_do_not_overlap():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], np.float32)
    keep = _nms(boxes, np.array([0.6, 0.9], np.float32), 0.5)
    assert sorted(keep.tolist()) == [0, 1]


def test_nms_returns_highest_score_first():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [200, 200, 210, 210]], np.float32)
    keep = _nms(boxes, np.array([0.3, 0.95, 0.6], np.float32), 0.5)
    assert keep.tolist() == [1, 2, 0]


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def test_drawing_does_not_mutate_the_frame_it_was_given():
    """The load-bearing one. In `terrain.evaluate.render` the array handed here is the SAME
    array odometry and the terrain classifier read from that iteration, so drawing in place
    would feed box outlines into the perception pipeline -- surfacing much later as a locked
    cell that disagrees with the footage."""
    frame = np.full((200, 400, 3), 30, np.uint8)
    before = frame.copy()
    out = draw_detections(frame, [Detection("enemy", 0.8, (10.0, 10.0, 90.0, 90.0))])
    assert np.array_equal(frame, before)
    assert not np.array_equal(out, frame)


def test_a_box_running_off_frame_is_clamped_for_drawing_only():
    """`Detection.xyxy` keeps the model's own coordinates -- a brawler at the screen edge really
    does have a box that runs past it -- so the clamp must live in the drawing, not in the box."""
    det = Detection("player", 0.9, (-50.0, -50.0, 500.0, 500.0))
    out = draw_detections(np.zeros((100, 200, 3), np.uint8), [det])
    assert out.shape == (100, 200, 3)
    assert det.xyxy == (-50.0, -50.0, 500.0, 500.0)
    assert out.any(), "a box covering the whole frame should have drawn something"


def test_annotation_scales_with_frame_width():
    """A fixed 2 px line is a sub-pixel smear after `_side_by_side` shrinks a 2002 px frame into
    a 476 px-tall panel, which is exactly the output the boxes exist to be looked at in."""
    thin, small = _sizes(500)
    thick, large = _sizes(2002)
    assert thick > thin and large > small
    # Sized so a full-resolution frame survives evaluate's ~2.4x downscale with a line still
    # more than one pixel wide.
    assert thick / 2.4 > 1.0


def test_an_explicit_thickness_overrides_the_derived_one():
    """The auto-sizing is a default, not a policy: a caller compositing at some other scale has
    to be able to say what it wants."""
    assert _sizes(2002, thickness=1, font_scale=0.5) == (1, 0.5)


def test_every_class_this_project_cares_about_has_a_distinct_colour():
    colors = [CLASS_COLORS[c] for c in ("enemy", "player", "teammate")]
    assert len(set(colors)) == 3


def test_an_unknown_class_still_draws_in_the_fallback_colour():
    """These weights carry three classes; the wall detectors upstream carry three or fifteen
    others. A model with a class nobody wrote a colour for must still be viewable."""
    assert color_for("close_bush") == color_for("no_such_class_at_all")
    out = draw_detections(np.zeros((100, 200, 3), np.uint8),
                          [Detection("close_bush", 0.7, (10.0, 10.0, 50.0, 50.0))])
    assert out.any()


def test_the_tally_counts_what_was_drawn():
    frame = np.zeros((200, 400, 3), np.uint8)
    dets = [Detection("enemy", 0.9, (10.0, 10.0, 50.0, 50.0)),
            Detection("enemy", 0.8, (60.0, 60.0, 90.0, 90.0))]
    assert draw_summary(frame.copy(), dets).any()
    assert not draw_summary(frame.copy(), []).any(), "no detections should write no tally"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_the_shipped_config_validates():
    validate(load_vision_config())


def test_detector_ignore_rejects_a_bare_string():
    """`ignore: teammate` instead of `ignore: [teammate]` is the obvious YAML slip, and it would
    otherwise be read character by character -- ignoring the classes 't', 'e', 'a', 'm'... which
    match nothing, so it would silently do nothing at all."""
    with pytest.raises(ValueError, match="must be a LIST"):
        validate(VisionConfig(detector_ignore="teammate"))


@pytest.mark.parametrize("field, value", [("detector_conf", 0.0), ("detector_conf", 1.5),
                                          ("detector_iou", -0.1)])
def test_detector_thresholds_must_be_fractions(field, value):
    with pytest.raises(ValueError, match="fraction in"):
        validate(VisionConfig(**{field: value}))


def test_detector_device_is_a_closed_set():
    with pytest.raises(ValueError, match="'auto', 'cpu' or 'cuda'"):
        validate(VisionConfig(detector_device="gpu"))


def test_an_unknown_model_name_names_the_ones_that_exist():
    with pytest.raises(KeyError, match="entity_v2"):
        weights_mod.weights_path("PylaWallDetectorV9")


def test_a_missing_weights_file_says_how_to_get_it(tmp_path, monkeypatch):
    monkeypatch.setattr(weights_mod, "WEIGHTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="vision_fetch_detector"):
        weights_mod.require()


# ---------------------------------------------------------------------------
# the model itself
# ---------------------------------------------------------------------------

@needs_weights
def test_class_names_are_read_from_the_file_not_hardcoded(detector):
    """V1 and V2 carry the SAME three classes in DIFFERENT orders (V1 enemy/player/teammate, V2
    enemy/teammate/player). A hardcoded list silently swaps player and teammate on the other
    file -- the one pair this project cannot afford to confuse."""
    assert detector.names == {0: "enemy", 1: "teammate", 2: "player"}


@needs_weights
def test_the_fetched_file_is_the_one_that_was_pinned():
    """Upstream tracks `main` with no releases and no tags, so the file at that URL is not a
    fixed thing. A replaced model would otherwise show up as mysteriously different boxes."""
    name, _, digest = weights_mod.MODELS[weights_mod.DEFAULT_MODEL]
    assert weights_mod.sha256(weights_mod.weights_path()) == digest


@needs_weights
def test_predict_rejects_anything_that_is_not_a_uint8_bgr_frame(detector):
    with pytest.raises(ValueError, match="uint8 BGR"):
        detector.predict(np.zeros((100, 100, 3), np.float32))
    with pytest.raises(ValueError, match="uint8 BGR"):
        detector.predict(np.zeros((100, 100), np.uint8))


@needs_weights
def test_a_flat_grey_frame_detects_nothing(detector):
    """Not an accuracy claim -- a floor. A detector that fires on uniform noise-free grey is
    broken in a way no amount of real footage would make obvious."""
    assert detector.predict(np.full((1126, 2002, 3), 128, np.uint8)) == []


@needs_weights
def test_boxes_come_back_in_raw_frame_coordinates_not_letterboxed_ones(detector):
    """The 640x640 input is a 3.1x downscale of a 2002x1126 frame into the top-left of a padded
    canvas. If `predict` forgot to undo that scale, every box would land in the top-left third
    of the frame and still look plausible."""
    frame = np.full((1126, 2002, 3), 128, np.uint8)
    rng = np.random.default_rng(0)
    frame[:] = rng.integers(0, 255, frame.shape, dtype=np.uint8)
    for det in detector.predict(frame, conf=0.05):
        x0, y0, x1, y1 = det.xyxy
        assert x1 > x0 and y1 > y0
        assert -50 <= x0 and x1 <= 2002 + 50
        assert -50 <= y0 and y1 <= 1126 + 50


@pytest.mark.vision
@needs_weights
def test_it_finds_exactly_one_player_per_frame_of_real_footage(detector):
    """Solo Showdown has exactly one of you. Weak enough to be about the WIRING rather than the
    model's accuracy -- but it is the assertion that catches a swapped colour space, because fed
    BGR this same clip yields duplicate players and a flood of teammates."""
    from brawl_vision.sources import open_source
    clip = pytest.importorskip("pathlib").Path(CLIP)
    if not clip.exists():
        pytest.skip(f"{CLIP} is gitignored game footage")
    seen = 0
    with open_source(clip, load_vision_config()) as source:
        for frame in source:
            if frame.index % 60 or frame.index > 600:
                if frame.index > 600:
                    break
                continue
            dets = detector.predict(frame.image)
            labels = [d.label for d in dets]
            assert labels.count("player") == 1, f"frame {frame.index}: {labels}"
            assert "teammate" not in labels, (
                f"frame {frame.index} has a teammate in SOLO showdown: {labels}. The usual cause "
                f"is feeding the model BGR -- it is trained on RGB.")
            seen += 1
    assert seen >= 5


@pytest.mark.vision
@needs_weights
def test_detections_are_sorted_most_confident_first(detector):
    from brawl_vision.sources import open_source
    clip = pytest.importorskip("pathlib").Path(CLIP)
    if not clip.exists():
        pytest.skip(f"{CLIP} is gitignored game footage")
    with open_source(clip, load_vision_config()) as source:
        for frame in source:
            dets = detector.predict(frame.image, conf=0.1)
            if len(dets) >= 2:
                assert dets == sorted(dets, key=lambda d: d.confidence, reverse=True)
                return
    pytest.skip("no frame in the clip produced two detections")


@pytest.mark.vision
@needs_weights
def test_ignore_removes_a_class_and_leaves_the_others_alone(detector):
    """After NMS, not before: a suppressed class still has to compete for boxes, or an enemy the
    model half-called a teammate survives as a duplicate enemy just because teammates are off."""
    from brawl_vision.object_detection import ObjectDetector
    from brawl_vision.sources import open_source
    clip = pytest.importorskip("pathlib").Path(CLIP)
    if not clip.exists():
        pytest.skip(f"{CLIP} is gitignored game footage")
    muted = ObjectDetector.from_config(load_vision_config(), ignore=("player",))
    with open_source(clip, load_vision_config()) as source:
        frame = next(iter(source))
    full = detector.predict(frame.image)
    assert any(d.label == "player" for d in full)
    kept = muted.predict(frame.image)
    assert not any(d.label == "player" for d in kept)
    assert [d for d in full if d.label != "player"] == kept


# ---------------------------------------------------------------------------
# projection: screen box -> tile map
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def plan():
    from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
    return build_rectify_plan(load_camera_model(), load_hud_mask())


def test_projecting_nothing_is_not_an_error(plan):
    assert to_tiles([], plan) == []


def test_the_anchor_is_a_fraction_of_the_box_not_its_bottom_edge(plan):
    """Deliberately NOT invariant to box height. The box encloses a nameplate above the head and
    an aura below the feet, both of which scale with it, so a taller box for the same brawler
    means the feet sit further up from the bottom edge -- see `Detection.anchor` for the
    wall-and-water measurement that put the fraction at 0.30."""
    short = Detection("enemy", 0.9, (900.0, 500.0, 1000.0, 600.0))
    tall = Detection("enemy", 0.9, (900.0, 200.0, 1000.0, 600.0))
    (_, a), (_, b) = to_tiles([short, tall], plan)
    assert b[1] < a[1], "the taller box should anchor further up, i.e. nearer the camera"


def test_a_zero_anchor_is_the_raw_box_bottom(plan):
    """The old behaviour is still reachable, because it is what every other implementation of
    this does and comparing against it is how the 0.30 was justified."""
    det = Detection("enemy", 0.9, (900.0, 200.0, 1000.0, 600.0))
    (_, at_zero), = to_tiles([det], plan, anchor_frac=0.0)
    from brawl_vision.object_detection.project import to_tiles as _tt
    assert det.anchor(0.0) == det.ground_point
    assert at_zero[1] > _tt([det], plan, anchor_frac=0.30)[0][1][1]


def test_boxes_of_equal_geometry_land_on_the_same_tile(plan):
    a = Detection("enemy", 0.9, (900.0, 200.0, 1000.0, 600.0))
    b = Detection("player", 0.7, (900.0, 200.0, 1000.0, 600.0))
    (_, ta), (_, tb) = to_tiles([a, b], plan)
    assert ta == pytest.approx(tb)


def test_ground_offset_moves_the_point_toward_the_camera_only_in_y(plan):
    """It compensates for the measured overshoot along the view axis. Moving x as well would be
    inventing a correction in the one direction the measurements say has no stable bias."""
    det = Detection("player", 0.9, (900.0, 400.0, 1000.0, 600.0))
    (_, plain), = to_tiles([det], plan)
    (_, shifted), = to_tiles([det], plan, ground_offset_tiles=1.2)
    assert shifted[0] == pytest.approx(plain[0])
    assert shifted[1] == pytest.approx(plain[1] - 1.2)


def test_a_tile_maps_to_the_canvas_pixel_the_caller_declared(plan):
    """`draw_markers` is told (origin_tile, scale) and nothing else -- that one formula is what
    lets the same function serve the view panels and the cropped world canvas."""
    canvas = np.zeros((200, 200, 3), np.uint8)
    draw_markers(canvas, [(Detection("player", 0.9, (0, 0, 0, 0)), (3.0, 2.0))],
                 (1.0, 1.0), 50.0)
    # Centroid of the COLOURED ink, not the pixel at dead centre -- that one is the black dot
    # marking the sub-tile position, so probing it reads (0, 0, 0) and proves nothing.
    ys, xs = np.nonzero(canvas.max(axis=2) > 100)
    # tile (3,2) with origin (1,1) at 50 px/tile -> pixel (100, 50)
    assert xs.mean() == pytest.approx(100, abs=1)
    assert ys.mean() == pytest.approx(50, abs=1)


def test_a_marker_projected_off_the_panel_is_dropped_not_clamped(plan):
    """Clamping would park it on the edge, which reads as a real detection at the frame border."""
    canvas = np.zeros((100, 100, 3), np.uint8)
    draw_markers(canvas, [(Detection("enemy", 0.9, (0, 0, 0, 0)), (99.0, 99.0))], (0.0, 0.0), 20.0)
    assert not canvas.any()


def test_markers_are_drawn_rgb_because_render_canvases_are():
    """`draw_detections` annotates a BGR frame and `draw_markers` annotates an RGB canvas. Getting
    this backwards renders every enemy BLUE -- a bug indistinguishable from a design choice."""
    def ink(rgb):
        canvas = np.zeros((200, 200, 3), np.uint8)
        draw_markers(canvas, [(Detection("enemy", 0.9, (0, 0, 0, 0)), (1.0, 1.0))],
                     (0.0, 0.0), 80.0, rgb=rgb)
        # Averaged over every coloured pixel rather than probed at a fixed spot: the marker's
        # SHAPE is not what this test is about, and a coordinate probe breaks the day it changes.
        lit = canvas[canvas.max(axis=2) > 100]
        assert len(lit), "the marker drew nothing"
        return lit.mean(axis=0)
    assert ink(True)[0] > ink(True)[2], "enemy should be RED on an RGB canvas"
    assert ink(False)[2] > ink(False)[0]


def test_the_marker_is_the_sim_s_collision_footprint():
    """Sized from the simulator's own `entities.unit_radius`, so "a brawler is here" means the
    same span on this map as it does in `brawl_sim`. A marker sized to anything else would make
    the comparison the view layout exists for read wrong by construction."""
    from brawl_vision.object_detection.project import FOOTPRINT_TILES
    assert FOOTPRINT_TILES == pytest.approx(0.8)   # 2 * entities.unit_radius, configs/default.yaml
    scale = 40.0
    canvas = np.zeros((300, 300, 3), np.uint8)
    draw_markers(canvas, [(Detection("player", 0.9, (0, 0, 0, 0)), (3.0, 3.0))],
                 (0.0, 0.0), scale)
    ys, xs = np.nonzero(canvas.max(axis=2) > 0)
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    # The box plus its black rim, so a little wider than the bare footprint but not by a tile.
    assert FOOTPRINT_TILES * scale <= span <= (FOOTPRINT_TILES + 0.4) * scale


# ---------------------------------------------------------------------------
# what render() refuses
# ---------------------------------------------------------------------------

def _render_kwargs(plan, **over):
    from brawl_vision.terrain.evaluate import Track
    kwargs = dict(source=iter(()), plan=plan, track=Track(), path="unused.mp4",
                  classifier=object(), detector=object())
    kwargs.update(over)
    return kwargs


@pytest.mark.parametrize("over, message", [
    ({"layout": "map"}, "somewhere to draw"),
    ({"layout": "view"}, "detect_on_map"),
    ({"layout": "view", "detect_on_map": True, "classifier": None}, "needs a classifier"),
])
def test_render_refuses_a_detector_with_nowhere_to_put_it(plan, over, message):
    """Silently dropping the boxes would look like a broken detector rather than a request the
    layout cannot satisfy."""
    from brawl_vision.terrain.evaluate import render
    with pytest.raises(ValueError, match=message):
        render(**_render_kwargs(plan, **over))


@pytest.mark.parametrize("over, message", [
    # HP is read INSIDE a detection box, so without a detector there is nothing to read.
    ({"layout": "side-by-side", "health": object(), "detector": None}, "needs a detector"),
    # ...and it is written next to that box on the RAW frame, which only one layout shows.
    ({"layout": "view", "detect_on_map": True, "health": object()}, "annotates the RAW frame"),
])
def test_render_refuses_health_reading_with_nowhere_to_read_or_draw(plan, over, message):
    from brawl_vision.terrain.evaluate import render
    with pytest.raises(ValueError, match=message):
        render(**_render_kwargs(plan, **over))


@pytest.mark.parametrize("over, message", [
    ({"layout": "side-by-side", "map_extent": "fov"}, "unknown map_extent"),
    # 'view' is ALREADY the footprint crop and 'map' is the whole artifact, so cropping either is
    # a silently ignored argument -- the one outcome worse than an error.
    ({"layout": "view", "detect_on_map": True, "map_extent": "view"}, "only applies to"),
    ({"layout": "side-by-side", "map_extent": "policy", "classifier": None},
     "needs a classifier"),
])
def test_render_refuses_a_map_extent_it_cannot_honour(plan, over, message):
    from brawl_vision.terrain.evaluate import render
    with pytest.raises(ValueError, match=message):
        render(**_render_kwargs(plan, **over))


@pytest.mark.parametrize("over, message", [
    ({"cv_fps": 0}, "must be > 0"),
    # Results computed between two written frames are simply discarded.
    ({"out_fps": 8.0, "cv_fps": 24.0}, "never"),
])
def test_render_refuses_a_cv_rate_that_wastes_work(plan, over, message):
    from brawl_vision.terrain.evaluate import render
    with pytest.raises(ValueError, match=message):
        render(**_render_kwargs(plan, layout="side-by-side", **over))


def test_a_held_detection_is_corrected_for_the_camera_moving_under_it():
    """With `cv_fps` below `out_fps` a detection is drawn on frames the detector never saw, and
    its tile coordinates are relative to the camera that DID see it. Left uncorrected every
    marker slides backwards as the camera pans and then snaps when the detector next runs, which
    reads as a projection bug rather than a stale frame of reference."""
    from brawl_vision.terrain.evaluate import _held_origin

    class _Plan:
        origin_tile = (-7, -5)

    plan = _Plan()
    # Nothing held yet, or no motion since: the plain origin, unchanged.
    assert _held_origin(plan, (10.0, 4.0), None) == (-7, -5)
    assert _held_origin(plan, (10.0, 4.0), (10.0, 4.0)) == (-7, -5)
    # Camera moved +2.5 x and -1.0 y since the detection: the origin moves with it, so the
    # marker keeps the WORLD tile it was measured at.
    assert _held_origin(plan, (12.5, 3.0), (10.0, 4.0)) == (-4.5, -6.0)


def test_the_policy_extent_crops_the_footprint_and_moves_the_marker_origin_with_it():
    """The crop shifts the panel's top-left tile, so the marker origin has to shift by the same
    amount. Getting it wrong offsets every marker by ~4 tiles, which reads as a projection bug
    rather than the arithmetic slip it is -- so the arithmetic is pinned here rather than left to
    be eyeballed in a video."""
    cols, rows, view_w, view_h = 30, 19, 21, 13
    dx, dy = round((cols - view_w) / 2), round((rows - view_h) / 2)
    assert (dx, dy) == (4, 3)
    origin = (-7, -5)                                   # a plausible plan.origin_tile
    assert (origin[0] + dx, origin[1] + dy) == (-3, -2)


def test_a_warped_box_encloses_the_same_sprite_it_did_on_screen(plan):
    """The rectified panel is the raw frame through `plan.M`, so pushing the box corners through
    the SAME matrix warps them exactly as the pixels under them were warped. Checked by mapping
    each corner independently and confirming it agrees with the quad."""
    import cv2
    det = Detection("enemy", 0.9, (900.0, 300.0, 1100.0, 560.0))
    (_, quad), = to_tile_quads([det], plan)
    assert quad.shape == (4, 2)
    x0, y0, x1, y1 = det.xyxy
    for i, (px, py) in enumerate([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]):
        r = cv2.perspectiveTransform(np.array([[[px, py]]], np.float64), plan.M).reshape(-1, 2)
        assert quad[i] == pytest.approx(plan.rect_to_tile(r)[0])


def test_the_full_box_is_taller_than_the_footprint_it_resolves_to(plan):
    """Not a rounding detail -- the box top is a NAMEPLATE floating above the world, so it
    projects several tiles away. That is why the map panel gets a footprint instead: a box
    enclosing a nameplate means something on a picture of a screen and nothing on a tile grid."""
    from brawl_vision.object_detection.project import FOOTPRINT_TILES
    det = Detection("player", 0.9, (900.0, 300.0, 1100.0, 560.0))
    (_, quad), = to_tile_quads([det], plan)
    assert quad[:, 1].max() - quad[:, 1].min() > FOOTPRINT_TILES * 2


def test_boxes_and_markers_share_one_canvas_mapping(plan):
    """`draw_boxes` and `draw_markers` take the same `(origin_tiles, scale)`, so the footprint on
    the map panel lands where the base of the box lands on the rectified panel. Two mappings would
    be two things to keep in sync, and the layout's whole value is that they agree."""
    from brawl_vision.object_detection.project import draw_boxes
    det = Detection("player", 0.9, (900.0, 300.0, 1100.0, 560.0))
    (_, quad), = to_tile_quads([det], plan)
    placed = to_tiles([det], plan)
    origin, scale = plan.origin_tile, 30.0

    boxed = draw_boxes(np.zeros((600, 900, 3), np.uint8), [(det, quad)], origin, scale,
                       label=False)
    marked = draw_markers(np.zeros((600, 900, 3), np.uint8), placed, origin, scale)
    bx = np.nonzero(boxed.max(axis=2))[1]
    mx = np.nonzero(marked.max(axis=2))[1]
    # The footprint sits horizontally inside the box's span -- same mapping, same world.
    assert bx.min() - 2 <= mx.mean() <= bx.max() + 2


def test_a_box_projecting_past_the_horizon_is_skipped_not_drawn_enormous(plan):
    """A corner near the horizon blows up through the homography. Drawing it would paint a
    thousand-tile shape across the panel; skipping it loses one annotation."""
    from brawl_vision.object_detection.project import draw_boxes
    det = Detection("enemy", 0.9, (0.0, 0.0, 10.0, 10.0))
    canvas = np.zeros((200, 200, 3), np.uint8)
    huge = np.array([[0, 0], [1e9, 0], [1e9, 1e9], [0, 1e9]], np.float64)
    draw_boxes(canvas, [(det, huge)], plan.origin_tile, 30.0)
    assert not canvas.any()


# ---------------------------------------------------------------------------
# the GPU path, which fails at predict rather than at construction
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@needs_weights
def test_both_detectors_run_a_real_inference_on_cuda_with_torch_loaded():
    """The live loop holds ONNX detectors AND a torch policy in one process, and those two fight
    over the single `cudnn64_9.dll` a process gets (BRAWL_DEPLOYMENT_DESIGN.md 9.12, fix 4).

    **The provider name is not the assertion.** The broken configuration constructed cleanly,
    reported `CUDAExecutionProvider`, and then failed inside `predict` -- the entity model with
    `Failed to initialize CUDNN Frontend`, the projectile model with
    `CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE`. So this runs both models on a real frame
    with torch imported, which is the arrangement `loop.py` will have.

    Skips rather than fails without CUDA: this is a property of one machine's DLL set, and the
    siloed CPU suite is not the place to demand a GPU.
    """
    import onnxruntime as ort

    from brawl_vision.object_detection.detector import preload_cuda_dlls

    preload_cuda_dlls()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("this onnxruntime build has no CUDA provider")
    pytest.importorskip("torch")

    from brawl_vision.object_detection import ObjectDetector
    from brawl_vision.object_detection.projectile_detection.detect import ProjectileDetector

    cfg = load_vision_config("configs/vision.yaml")
    frame = np.random.default_rng(0).integers(0, 255, (1126, 2002, 3), dtype=np.uint8)
    for name, build in (("entity", lambda: ObjectDetector.from_config(cfg, device="cuda")),
                        ("projectile", lambda: ProjectileDetector.from_config(cfg, device="cuda"))):
        try:
            det = build()
        except FileNotFoundError:                       # projectile model is trained, not fetched
            pytest.skip(f"{name} weights are not present")
        assert det.provider == "CUDAExecutionProvider", name
        det.predict(frame)                              # the assertion is that this does not raise
