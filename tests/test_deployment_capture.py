"""`brawl_deployment.capture` -- the resize to the calibrated viewport, and the guards on it.

The guards are the point of this file. `DeployCapture` itself is thirty lines around one
`cv2.resize`; what is worth testing is that it REFUSES the way a capture can be geometrically
wrong while still looking fine, because that corrupts every projected tile coordinate downstream
and does not announce itself.

Fixtures only, no `mss`, no screen -- these belong to the fast siloed set.
"""
import numpy as np
import pytest

from brawl_deployment.capture import MIN_HEIGHT_COVERAGE, DeployCapture
from brawl_vision.capture import Frame, detect_content_box, normalize_viewport

MONITOR = (2560, 1440)
BLACK = 12          # below capture_black_level (25), i.e. reads as bar rather than as content


class _FakeScreenCapture:
    """Stands in for `ScreenCapture`: the same handful of attributes `DeployCapture` reads."""

    def __init__(self, box, raw_size=MONITOR, image=None, box_supplied=False):
        self.content_box = box
        self.raw_size = raw_size
        self.box_supplied = box_supplied
        self.grab_seconds = 0.0
        if image is None:
            h = box[3] - box[2] + 1
            image = np.zeros((h, int(round(h * 16 / 9)), 3), np.uint8)
        self._image = image
        self.index = 0

    def grab(self):
        f = Frame(image=self._image, t=0.0, index=self.index)
        self.index += 1
        return f


def _capture(box, raw_size=MONITOR, image=None) -> DeployCapture:
    cap = DeployCapture()
    cap._cap = _FakeScreenCapture(box, raw_size, image)
    return cap


def _monitor_frame() -> np.ndarray:
    """A synthetic fullscreen grab: lit everywhere, so the content box is the whole screen."""
    w, h = MONITOR
    return np.full((h, w, 3), 200, np.uint8)


# ---------------------------------------------------------------------------------------------
# What occlusion actually does to the content box. This is the EVIDENCE for the height guard --
# the same experiment that was run against a real desktop grab, reduced to a fixture.
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,rect,moves", [
    # (x0, y0, x1, y1) of the dark rectangle, exclusive upper bounds.
    ("centred window",   (700, 350, 1700, 1050),  False),
    ("full-height band", (1000, 0, 1400, 1440),   False),   # touches top AND bottom, not a side
    ("left edge",        (0, 300, 600, 1100),     True),
    ("right edge",       (1960, 300, 2560, 1100), True),
    ("top strip",        (0, 0, 2560, 120),       True),
    ("bottom strip",     (0, 1320, 2560, 1440),   True),
])
def test_only_occlusion_touching_a_frame_edge_can_move_the_content_box(name, rect, moves):
    """A bounding box cannot be dented. Pass 2 of `detect_content_box` only ever reads the FIRST
    and LAST surviving column and row, so a dark region with lit content on both sides of it --
    in either axis -- changes nothing at all. Only a region reaching an edge removes an extreme.

    The "full-height band" case is the one worth stating outright: it blacks out every row it
    covers, top edge to bottom edge, and still moves nothing, because there are lit columns to
    its left and to its right. An overlapping window in the middle of the screen is this case.
    """
    img = _monitor_frame()
    x0, y0, x1, y1 = rect
    img[y0:y1, x0:x1] = BLACK

    baseline = detect_content_box([_monitor_frame()])
    box = detect_content_box([img])
    assert (box != baseline) == moves, f"{name}: box {box} against baseline {baseline}"


@pytest.mark.parametrize("side", ["left", "right"])
def test_horizontal_occlusion_raises_in_normalize_viewport(side):
    """The loud half. `normalize_viewport` compares width against `height * aspect` and refuses to
    upscale, so a bar down either side is caught before it can reach the pipeline."""
    img = _monitor_frame()
    if side == "left":
        img[:, :600] = BLACK
    else:
        img[:, -600:] = BLACK
    with pytest.raises(ValueError, match="Refusing to upscale"):
        normalize_viewport(img, detect_content_box([img]))


@pytest.mark.parametrize("side", ["top", "bottom"])
def test_vertical_occlusion_normalizes_silently_at_the_wrong_scale(side):
    """The quiet half, and the reason `MIN_HEIGHT_COVERAGE` exists.

    Losing rows shrinks the target width in proportion, so the aspect test passes and what comes
    back is a well-formed 16:9 frame -- just a smaller one, of less world. Downstream it is
    resized to 2002x1126 and is then indistinguishable from a correct frame, while every tile
    coordinate is off by the height ratio. Nothing `normalize_viewport` can see would catch this,
    which is why the check lives in `DeployCapture`, against the grab.
    """
    img = _monitor_frame()
    if side == "top":
        img[:120] = BLACK
    else:
        img[-120:] = BLACK

    out = normalize_viewport(img, detect_content_box([img]))      # no raise
    h, w = out.shape[:2]
    assert h == 1320                                              # 120 rows of world gone
    assert abs(w / h - 16 / 9) < 0.01                             # and still 16:9


# ---------------------------------------------------------------------------------------------
# The guard itself.
# ---------------------------------------------------------------------------------------------

def test_a_fullscreen_grab_passes_the_height_guard():
    """1439 of 1440 rows -- the box a real fullscreen 1440p BlueStacks grab produces. The single
    lost row is boundary rounding, and the guard must not be tight enough to trip on it."""
    cap = _capture((0, 2559, 0, 1438))
    cap.grab()
    assert cap.source_size is not None


@pytest.mark.parametrize("box", [
    (0, 2559, 120, 1438),      # a strip along the top
    (0, 2559, 0, 1319),        # a strip along the bottom
    (0, 2559, 864, 1438),      # dark over most of every column
])
def test_a_box_that_lost_rows_fails_closed(box):
    cap = _capture(box)
    with pytest.raises(ValueError, match="touching the top or bottom edge"):
        cap.grab()


def test_the_guard_runs_before_the_upscale_check():
    """Order matters for the message, not the outcome. A badly cropped box can also be small
    enough to trip `MAX_UPSCALE`, and "something is covering your screen" is the actionable
    diagnosis where "raise your resolution" sends the operator somewhere useless."""
    cap = _capture((0, 533, 1140, 1439), image=np.zeros((300, 534, 3), np.uint8))
    with pytest.raises(ValueError, match="touching the top or bottom edge"):
        cap.grab()


def test_a_narrow_box_is_left_to_normalize_viewport():
    """Width is deliberately NOT checked here. A side-edge occlusion already raises upstream, and
    a legitimately pillarboxed source -- which is what `detect_content_box` was written for -- has
    a box narrower than its grab by design."""
    cap = _capture((600, 2559, 0, 1438))
    cap.grab()             # 1439/1440 rows: this class has no opinion about the missing columns
    assert cap.source_size is not None


def test_the_guard_is_skipped_when_there_is_nothing_to_check_against():
    """`normalize=False` leaves the box `None`. Silence is right: with no crop there is nothing to
    be wrong about."""
    cap = _capture(None, raw_size=None, image=np.zeros((1126, 2002, 3), np.uint8))
    cap.grab()
    assert cap.source_size == (2002, 1126)


def test_the_guard_runs_once_not_per_frame():
    """It reads a value that is computed once and cached, so a per-frame check would be pure cost
    against the 50 ms budget -- and, worse, would imply the box can change mid-session. It cannot,
    which is what confines this whole failure mode to the first frames of a session."""
    cap = _capture((0, 2559, 0, 1438))
    cap.grab()
    cap._cap.content_box = (0, 2559, 900, 1438)   # would fail the guard if it were re-read
    cap.grab()
    cap.grab()


# ---------------------------------------------------------------------------------------------
# The supplied-box path, which is what the deployed loop uses.
# ---------------------------------------------------------------------------------------------

def test_a_supplied_box_skips_detection_entirely():
    """`from_window` hands the crop rectangle straight down. Nothing samples frames, nothing
    infers, and the guard above has nothing to guard -- the geometry came from the window manager
    rather than from whatever the app happened to be drawing."""
    cap = DeployCapture(box=(0, 2559, 0, 1439))
    assert cap._cap.box_supplied
    assert cap._cap.content_box == (0, 2559, 0, 1439)


def test_the_height_guard_does_not_second_guess_a_supplied_box():
    """A short supplied box is the caller's decision, and it has already been checked somewhere
    with better information: a client area that does not fill its monitor is `WindowGuard`'s
    business and raises there with a message that names the actual problem."""
    box = (0, 2559, 300, 1438)                      # would fail the guard if it were inferred
    cap = _capture(box, image=np.zeros((1139, 2025, 3), np.uint8))
    cap._cap.box_supplied = True
    cap.grab()
    assert cap.source_size == (2025, 1139)


def test_the_lobby_box_is_refused_when_inferred_and_accepted_when_supplied():
    """The regression for the live finding, with the real numbers.

    A black band across the top of the frame is indistinguishable, in pixels, from an occluding
    window -- and at the Nulls Brawl lobby it is neither: the app is simply not drawing there.
    Inference cannot tell those apart, which is precisely why the deployed path stops inferring.
    """
    img = _monitor_frame()
    img[:21] = BLACK                                # the measured lobby band

    box = detect_content_box([img])
    assert box == (0, 2559, 21, 1439)
    assert (box[3] - box[2] + 1) / 1440 == pytest.approx(0.9854, abs=1e-4)

    inferred = _capture(box)
    with pytest.raises(ValueError, match="Nulls Brawl lobby"):
        inferred.grab()

    supplied = _capture((0, 2559, 0, 1439))         # what the window manager reports
    supplied._cap.box_supplied = True
    supplied.grab()
    assert supplied.source_size is not None


def _far_edge_shift(src_a: int, src_b: int, viewport: int) -> float:
    """How far apart two crop widths put the same raw pixel, at the far edge of the frame."""
    x = min(src_a, src_b) - 1
    return abs(x * viewport / src_a - x * viewport / src_b)


def test_the_window_box_moves_a_gameplay_frame_by_about_a_pixel_and_a_half():
    """How much switching to the window box perturbs the verified calibration.

    MEASURED live during gameplay, inference returned 2558x1439 of a 2560x1440 grab -- it trims
    two columns and a row of genuinely dark game edge. The window box keeps them, so the same raw
    pixel lands ~1.6 px further left and ~1.0 px further up at the viewport's far edge. That is
    NOT nothing, and this test states the real number rather than rounding it to "sub-pixel":
    it is 3% of a 48 px tile, well inside `MatchState.refine`'s +/-12 px rescan, and in the
    direction of the authoritative answer -- the window says the render rectangle is the full
    client area, and the tile-grid check passed with a box 1.6 px off from that.

    Re-verify against a gameplay frame when one is available; it is on the friendly-battle list.
    """
    dx = _far_edge_shift(2558, 2560, 2002)
    dy = _far_edge_shift(1439, 1440, 1126)
    assert 1.0 < dx < 2.0
    assert 0.5 < dy < 1.5
    assert dx < 12.0 and dy < 12.0, "outside MatchState.refine's rescan window"

    # The lobby box, by contrast, is an order of magnitude worse -- which is the whole point.
    assert _far_edge_shift(1418, 1440, 1126) > 15.0


def test_the_threshold_leaves_room_for_rounding_but_not_for_occlusion():
    """Pins the constant between the two numbers that bracket it, so a later loosening has to
    argue with them: observed rounding costs one row out of 1440, and the smallest occlusion worth
    catching is a 1% scale error."""
    assert MIN_HEIGHT_COVERAGE < 1439 / 1440
    assert MIN_HEIGHT_COVERAGE >= 0.99
