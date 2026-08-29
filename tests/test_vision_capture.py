"""Frame geometry primitives and live screen capture. See Terrain_Perception_Build_Plan.md
Phase A and Section 3.

Everything about `detect_content_box` / `normalize_viewport` is tested on SYNTHETIC frames, so
this file runs on a fresh clone with no game footage. The same primitives are exercised against
the real clips in tests/test_vision_clips.py, which is marked `vision` and skips without them.
"""
import ctypes
import sys

import numpy as np
import pytest

from brawl_vision.capture import (
    SHOWDOWN_ASPECT,
    Frame,
    ScreenCapture,
    detect_content_box,
    normalize_viewport,
)
from brawl_vision.config import VisionConfig

# The real fixture geometry, from tests/fixtures/vision/README.md. Reused here so the synthetic
# tests exercise the same numbers the real clips do.
FRAME_W, FRAME_H = 2436, 1126
CAVE_BOX = (0, 2435, 0, 1125)        # training cave: game fills the frame
SHOWDOWN_BOX = (217, 2218, 0, 1125)  # Solo Showdown: 16:9, centered in the 2436 px frame
NORM_W = 2002                        # round(1126 * 16/9)


def _synthetic(box, value=200, w=FRAME_W, h=FRAME_H):
    """A black frame with `box` filled in -- stands in for a pillarboxed game capture."""
    x0, x1, y0, y1 = box
    img = np.zeros((h, w, 3), np.uint8)
    img[y0:y1 + 1, x0:x1 + 1] = value
    return img


# ---------------------------------------------------------------------------
# detect_content_box
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("box", [CAVE_BOX, SHOWDOWN_BOX, (100, 500, 50, 300)])
def test_detects_the_box_it_was_given(box):
    assert detect_content_box([_synthetic(box)]) == box


def test_letterbox_and_pillarbox_both_recovered():
    assert detect_content_box([_synthetic((0, FRAME_W - 1, 200, 900))]) == (0, FRAME_W - 1, 200, 900)


def test_box_is_the_union_over_frames_not_the_last_one():
    """A dark frame -- a night map, a death fade, the match-start transition -- can have real
    content dimmer than black_level at its edges. Trusting one frame would report a box tighter
    than the truth, and cropping to it silently discards world pixels that every later stage then
    never sees. The union is the guard."""
    bright = _synthetic(SHOWDOWN_BOX, value=200)
    dim = _synthetic((600, 1600, 300, 800), value=200)  # a frame whose content happens to be small
    assert detect_content_box([bright, dim]) == SHOWDOWN_BOX
    assert detect_content_box([dim, bright]) == SHOWDOWN_BOX, "union must not depend on order"


def test_all_black_frames_raise_rather_than_returning_an_empty_box():
    with pytest.raises(ValueError, match="black_level"):
        detect_content_box([np.zeros((10, 10, 3), np.uint8)])


def test_no_images_raises():
    with pytest.raises(ValueError, match="no images"):
        detect_content_box([])


def test_black_level_is_above_zero_so_compression_noise_is_not_content():
    """Video compression does not preserve pure black exactly, so a bar can decode as 1-8 rather
    than 0. A threshold of 0 would find 'content' in the bars and defeat the whole detection."""
    img = _synthetic(SHOWDOWN_BOX)
    img[:, :50] = 6  # faint compression noise in what is really a black bar
    assert detect_content_box([img], black_level=12) == SHOWDOWN_BOX
    assert detect_content_box([img], black_level=0)[0] == 0, "sanity: threshold 0 is fooled"


# ---------------------------------------------------------------------------
# normalize_viewport -- the Section 3 claim
# ---------------------------------------------------------------------------

def test_cave_and_showdown_normalize_to_identical_geometry():
    """The core Section 3 claim: Solo Showdown constricts the visible WIDTH at the same zoom, so
    a symmetric trim makes training-cave footage geometrically identical to Showdown footage.
    This is what lets one homography serve every mode."""
    cave = normalize_viewport(_synthetic(CAVE_BOX), CAVE_BOX)
    show = normalize_viewport(_synthetic(SHOWDOWN_BOX), SHOWDOWN_BOX)
    assert cave.shape == show.shape == (FRAME_H, NORM_W, 3)


def test_cave_trim_is_symmetric_and_lands_on_the_showdown_box():
    """(2436 - 2002) / 2 = 217 per side, which IS the Showdown content box's x0."""
    marked = _synthetic(CAVE_BOX)
    marked[:, 217] = [0, 0, 255]                    # mark the expected left edge
    marked[:, 217 + NORM_W - 1] = [0, 255, 0]       # ...and the expected right edge
    out = normalize_viewport(marked, CAVE_BOX)
    assert out.shape[1] == NORM_W
    assert (out[:, 0] == [0, 0, 255]).all(), "left edge is not at x=217"
    assert (out[:, -1] == [0, 255, 0]).all(), "right edge is not at x=2218"


def test_showdown_width_is_already_normalized_and_is_left_alone():
    marked = _synthetic(SHOWDOWN_BOX)
    marked[:, 217] = [0, 0, 255]
    out = normalize_viewport(marked, SHOWDOWN_BOX)
    assert out.shape[1] == NORM_W
    assert (out[:, 0] == [0, 0, 255]).all()


def test_narrower_than_target_raises_rather_than_upscaling():
    """Stretching would invent world the game never rendered and hand Phase C a homography fit
    against fabricated pixels. There is no correct recovery, only a louder or quieter failure."""
    narrow = (0, 999, 0, 1125)
    with pytest.raises(ValueError, match="Refusing to upscale"):
        normalize_viewport(_synthetic(narrow), narrow)


def test_a_few_pixels_short_is_tolerated_as_boundary_rounding():
    """The detected viewport can land a pixel or two inside the exact aspect because the content
    boundary is anti-aliased, not a step. Refusing to crop over 1 px would make every real clip
    unreadable, so a small shortfall is absorbed -- while a genuine framing mismatch (hundreds of
    px) still raises."""
    tight = (218, 2217, 0, 1125)  # 2000 px: 2 short of the 2002 target
    assert normalize_viewport(_synthetic(tight), tight).shape[1] == NORM_W


def test_glow_outside_the_viewport_is_not_counted_as_content():
    """A regression test for a real bug. The boundary is not a step from black to content: there
    is a ~10 px band of dim HUD glow outside the world viewport, where only 2-23% of each column
    is lit. A max-based detector latched onto it and reported the viewport ~11 px too wide and
    apparently off-center -- which is what made a genuinely 16:9 viewport measure as 1.7815."""
    img = _synthetic(SHOWDOWN_BOX)
    # a sparse glow band just outside the viewport: bright pixels, but only a few per column
    img[::12, 205:217] = 200
    assert detect_content_box([img])[0] == SHOWDOWN_BOX[0]


def test_aspect_is_resolution_independent():
    """Expressed as an aspect, not a pixel width, so a different capture resolution still
    normalizes correctly -- a calibration is tied to a resolution, 'how much world Showdown
    shows you' is not."""
    half = (0, 1216, 0, 562)  # roughly half-resolution cave frame
    out = normalize_viewport(_synthetic(half, w=1218, h=563), half)
    assert out.shape[0] == 563
    assert out.shape[1] == round(563 * SHOWDOWN_ASPECT)


def test_normalized_aspect_default_matches_the_measurement():
    assert VisionConfig().capture_normalized_aspect == pytest.approx(16 / 9)
    assert round(FRAME_H * VisionConfig().capture_normalized_aspect) == NORM_W


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

def test_frame_size_is_w_h_not_numpy_shape_order():
    """OpenCV takes (w, h); numpy reports (h, w). Every size bug in a CV pipeline is this one."""
    f = Frame(image=np.zeros((1126, NORM_W, 3), np.uint8), t=0.0, index=0)
    assert f.size == (NORM_W, 1126)


# ---------------------------------------------------------------------------
# ScreenCapture
# ---------------------------------------------------------------------------

def test_grab_outside_a_context_manager_raises():
    """mss holds an OS handle; a capture loop that leaks one per restart is exactly the silent
    failure Phase A's acceptance criterion checks for, so the lifecycle is enforced."""
    with pytest.raises(RuntimeError, match="context manager"):
        ScreenCapture().grab()


def _rss_bytes():
    """Process-scoped resident set size, or None. Deliberately NOT a GPU or system-wide reading:
    the question is whether THIS process leaks."""
    if sys.platform != "win32":
        return None
    import ctypes.wintypes as wt

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    # argtypes/restype are NOT optional here. GetCurrentProcess returns the pseudo-handle -1;
    # left to ctypes' default int marshalling that is truncated on the way into a 64-bit HANDLE
    # parameter and the call fails silently, returning 0 with no last-error set -- which would
    # skip this test rather than fail it, quietly retiring Phase A's acceptance criterion.
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32.GetCurrentProcess.restype = wt.HANDLE
    k32.GetCurrentProcess.argtypes = []
    psapi.GetProcessMemoryInfo.restype = wt.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]

    counters = PMC()
    counters.cb = ctypes.sizeof(PMC)
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return counters.WorkingSetSize


@pytest.fixture
def screen():
    mss = pytest.importorskip("mss", reason="needs the [vision] extra")
    try:
        with ScreenCapture(normalize=False) as cap:
            yield cap
    except mss.exception.ScreenShotError as exc:
        pytest.skip(f"no capturable display: {exc}")


def test_grab_returns_a_bgr_uint8_frame(screen):
    f = screen.grab()
    assert f.image.dtype == np.uint8
    assert f.image.ndim == 3 and f.image.shape[2] == 3, "alpha must be dropped; BGR not BGRA"
    assert f.index == 0 and f.t >= 0.0


def test_timestamps_are_monotonic_and_indices_sequential(screen):
    frames = [screen.grab() for _ in range(8)]
    assert [f.index for f in frames] == list(range(8))
    ts = [f.t for f in frames]
    assert ts == sorted(ts), "timestamps went backwards"


def test_consecutive_grabs_do_not_alias_one_anothers_buffers(screen):
    """mss reuses its internal buffer between grabs. If the returned array were a view onto it,
    holding two frames would silently give you the same pixels twice -- and phase correlation
    between a frame and itself reports zero motion, which reads as 'the camera is still' rather
    than as a bug."""
    a, b = screen.grab(), screen.grab()
    assert a.image.base is None or not np.shares_memory(a.image, b.image)


def test_capture_loop_does_not_grow_memory(screen):
    """Phase A's acceptance criterion. A short run, not the multi-minute soak -- this catches a
    per-frame leak, which is the realistic failure; a slow drift needs the manual long run."""
    baseline = _rss_bytes()
    if baseline is None:
        pytest.skip("no process RSS reading available on this platform")
    for _ in range(30):
        screen.grab()
    warm = _rss_bytes()
    for _ in range(150):
        screen.grab()
    grown = _rss_bytes() - warm
    assert grown < 40 * 1024 * 1024, f"RSS grew {grown/1e6:.1f} MB over 150 grabs"
