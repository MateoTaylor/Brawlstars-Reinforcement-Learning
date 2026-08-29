"""Recorded-clip playback against the real fixtures. See Terrain_Perception_Build_Plan.md Phase A.

Every test here is marked `vision` and skips when the footage is absent: the clips are gitignored
game capture (Supercell IP), so a fresh clone has the annotations but not the pixels. Run with
`pytest -m vision` once you have them.

The numbers asserted below were measured with `cv2` specifically. `imageio_ffmpeg`'s streaming
reader over-yields on these files (503 / 565 / 1364 against cv2's 478 / 547 / 1351), so a
frame-indexed annotation is only meaningful next to the reader that produced it -- which is most
of why this module standardized on one.
"""
import json

import cv2
import numpy as np
import pytest

from brawl_vision.capture import Frame
from brawl_vision.clips import ClipReader, detect_usable_range, load_bounds
from brawl_vision.config import VisionConfig

pytestmark = pytest.mark.vision

FIXTURES = pytest.importorskip("pathlib").Path(__file__).resolve().parent / "fixtures" / "vision"

# name -> (n_frames, last usable index, pillarboxed?). Measured with cv2; see the fixtures README.
#
# Detected content boxes, for reference: the cave clips give x=[0, 2433] and the Showdown clip
# x=[217, 2217] -- 2002x1126, exactly 16:9, exactly centered in the 2436 px frame.
#
# Those exact bounds are deliberately NOT asserted here. They are stable now, but they are the
# output of a detector with two tunables (`black_level` and the mostly-lit fraction) plus a
# sampling stride, and pinning them would turn this file into a change detector for those knobs
# rather than a test of the behaviour that has to hold: that all three clips, framed differently
# by the game, normalize to identical geometry. `test_content_box_is_wide_enough_and_full_height`
# and `test_showdown_is_pillarboxed_and_the_cave_is_not` assert the properties that matter.
CLIPS = {
    # standstill's last usable index moved 408 -> 405 when the tail threshold started ignoring the
    # top 2% of body diffs (see `test_a_late_spike_cannot_raise_the_threshold_that_should_catch_it`).
    # The three frames it now drops were checked frame by frame, not assumed: iOS dims and blurs the
    # whole screen as Control Center opens, and 406-408 are already visibly darkened gameplay. The
    # new cut is the better one -- those are exactly the frames that would poison a colour model.
    "standstill":           (478,  405, False),
    "counted_walking":      (547,  495, False),
    "zone_grows_from_east": (1351, 1219, True),
}
FRAME_W = 2436
NORM_W, NORM_H = 2002, 1126   # round(1126 * 16/9)


def _path(name):
    p = FIXTURES / f"{name}.mp4"
    if not p.exists():
        pytest.skip(f"{p.name} not present (gitignored game footage)")
    return p


@pytest.fixture(params=sorted(CLIPS))
def clip_name(request):
    return request.param


# ---------------------------------------------------------------------------
# the Section 3 claim, against real footage
# ---------------------------------------------------------------------------

def test_every_clip_normalizes_to_the_same_viewport():
    """The whole point of Section 3. Training-cave clips render full-screen 2436 px wide and the
    Showdown clip 2002 px (exactly 16:9), at the same zoom; after normalization all three must be
    pixel-identical in geometry, or a single homography cannot serve every mode."""
    shapes = {}
    for name in CLIPS:
        reader = ClipReader(_path(name))
        shapes[name] = next(iter(reader)).image.shape
    assert set(shapes.values()) == {(NORM_H, NORM_W, 3)}, shapes


def test_content_box_is_wide_enough_and_full_height(clip_name):
    x0, x1, y0, y1 = ClipReader(_path(clip_name)).content_box
    assert (y0, y1) == (0, NORM_H - 1), "vertical extent is not constricted in any mode"
    # >= target minus the boundary-rounding slack normalize_viewport allows: the content edge is
    # anti-aliased, so the detected box can land a pixel or two inside the exact 16:9 width.
    assert x1 - x0 + 1 >= NORM_W - 4, "content is meaningfully narrower than the viewport"


def test_showdown_is_pillarboxed_and_the_cave_is_not(clip_name):
    """The asymmetry that motivates normalizing at all. Worth knowing if a future re-recording
    removes it -- that would mean the mode no longer constricts the view, and Phase K's whole
    field-of-view argument would need revisiting."""
    x0, x1, _, _ = ClipReader(_path(clip_name)).content_box
    if CLIPS[clip_name][2]:
        # A real pillarbox is ~217 px of bar per side, not a rounding artifact.
        assert x0 > 100 and x1 < FRAME_W - 100, "expected a pillarbox, found full-width content"
    else:
        assert x0 == 0 and x1 >= FRAME_W - 4, "expected full-width content, found a pillarbox"


# ---------------------------------------------------------------------------
# frame counts and tail trimming
# ---------------------------------------------------------------------------

def test_untrimmed_frame_count(clip_name):
    n, _, _ = CLIPS[clip_name]
    assert ClipReader(_path(clip_name), trim_tail=False).n_frames == n


def test_trimming_drops_the_ios_overlay_tail(clip_name):
    """Every fixture ends with the iOS Control Center swiping in -- the recording was stopped from
    the phone. Those frames are phone UI, not Brawl Stars, and integrating them in Phase F would
    add a large bogus shift right at the end of every regression fixture."""
    n, last, _ = CLIPS[clip_name]
    reader = ClipReader(_path(clip_name), trim_tail=True)
    assert reader.n_frames == last + 1
    assert reader.n_frames < n, "nothing was trimmed"


def test_iteration_yields_exactly_n_frames(clip_name):
    reader = ClipReader(_path(clip_name))
    frames = list(reader)
    assert len(frames) == reader.n_frames == len(reader)
    assert [f.index for f in frames] == list(range(len(frames)))


def test_trimming_only_removes_from_the_end():
    """Ground truth is annotated by frame index, so trimming must never renumber a frame. Frame
    k from a trimmed reader must be the same image as frame k from an untrimmed one."""
    name = "counted_walking"
    trimmed = iter(ClipReader(_path(name), trim_tail=True))
    raw = iter(ClipReader(_path(name), trim_tail=False))
    for _ in range(20):
        a, b = next(trimmed), next(raw)
        assert a.index == b.index
        assert np.array_equal(a.image, b.image)


# ---------------------------------------------------------------------------
# timestamps
# ---------------------------------------------------------------------------

def test_timestamps_are_monotonic(clip_name):
    ts = [f.t for f in ClipReader(_path(clip_name))]
    assert all(b >= a for a, b in zip(ts, ts[1:])), "presentation times went backwards"


def test_timestamps_are_real_not_synthesized_from_a_constant_fps():
    """These are phone screen recordings that genuinely drop frames. `standstill.mp4` carries a
    166.67 ms gap -- 10 frames at 60 Hz -- in the middle of an otherwise 16.7 ms stream. Phase F
    converts pixel shift to tile velocity, so a uniform `index / fps` would be wrong exactly where
    the camera moved furthest between samples."""
    ts = np.array([f.t for f in ClipReader(_path("standstill"))])
    dt = np.diff(ts)
    assert dt.std() > 0.001, "dt is constant -- timestamps look synthesized, not read"
    assert dt.max() > 0.1, f"expected a >100 ms dropped-frame gap, largest was {dt.max()*1000:.1f} ms"


# ---------------------------------------------------------------------------
# determinism, color order, sidecar
# ---------------------------------------------------------------------------

def test_playback_is_deterministic():
    """Phase A's acceptance criterion, and the precondition for every later phase's regression
    fixture: a replay that varies run to run cannot anchor an assertion."""
    name = "counted_walking"
    a = [f.image for f in list(ClipReader(_path(name)))[:40]]
    b = [f.image for f in list(ClipReader(_path(name)))[:40]]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


# Whole-frame blue-vs-red dominance, measured. The two sources sit on OPPOSITE sides of it --
# the Showdown map is dark purple (B > R), the training cave is pink/salmon (R > B) -- so a
# channel swap inverts both, and no single global flip could satisfy them together.
BLUE_DOMINANT = {"zone_grows_from_east": True, "standstill": False, "counted_walking": False}


def test_frames_are_bgr_not_rgb(clip_name):
    """Deliberately NOT built on the gas, despite gas being the color Phase G cares about.
    Swapping R and B reflects hue about the green axis, so a green-dominant color barely moves:
    measured, the gas mask is 6.10% of the viewport read correctly and 6.11% read backwards. It
    simply cannot detect the bug. Terrain, where green is not the channel maximum, inverts
    cleanly -- purple goes from hue ~119 to ~4."""
    frame = next(f for f in ClipReader(_path(clip_name)) if f.index == 200)
    b, g, r = frame.image.reshape(-1, 3).mean(axis=0)
    if BLUE_DOMINANT[clip_name]:
        assert b > r + 15, f"expected a blue-dominant purple map; got BGR=({b:.0f},{g:.0f},{r:.0f})"
    else:
        assert r > b + 15, f"expected a red-dominant pink map; got BGR=({b:.0f},{g:.0f},{r:.0f})"


def test_reader_does_not_convert_color_at_all(clip_name):
    """The above pins the ORDER; this pins that we do not touch the pixels. Our frame must be the
    raw cv2 decode, merely cropped -- so an accidental cvtColor added later fails here rather than
    quietly shifting every hue threshold downstream."""
    reader = ClipReader(_path(clip_name))
    ours = next(iter(reader))

    cap = cv2.VideoCapture(str(_path(clip_name)))
    try:
        ok, raw = cap.read()
    finally:
        cap.release()
    assert ok

    x0, x1, y0, y1 = reader.content_box
    # normalize_viewport centers on the detected VIEWPORT, not on the frame -- mirrored here so
    # this test pins "no color conversion" rather than accidentally re-testing the centering rule.
    nx0 = int(round((x0 + x1 + 1) / 2.0 - NORM_W / 2.0))
    nx0 = max(0, min(nx0, raw.shape[1] - NORM_W))
    assert np.array_equal(ours.image, raw[y0:y1 + 1, nx0:nx0 + NORM_W])


def test_bounds_sidecar_is_written_and_reused(tmp_path):
    """Detection is a full decode pass (seconds per clip) whose answer never changes, so it is
    cached beside the clip as small TRACKED text -- a fresh clone keeps the annotation even
    though the footage itself is gitignored."""
    src = _path("standstill")
    copy = tmp_path / "standstill.mp4"
    copy.write_bytes(src.read_bytes())
    sidecar = copy.with_suffix(".mp4.bounds.json")

    assert not sidecar.exists()
    first = load_bounds(copy)
    assert sidecar.exists()
    assert json.loads(sidecar.read_text()) == first

    # A second call must read the file rather than recompute: corrupt the cache and check it wins.
    sidecar.write_text(json.dumps({**first, "n_frames": 12345}))
    assert load_bounds(copy)["n_frames"] == 12345
    assert load_bounds(copy, refresh=True)["n_frames"] == first["n_frames"]


def test_detect_usable_range_agrees_with_the_recorded_measurements(clip_name):
    n, last, pillarboxed = CLIPS[clip_name]
    got = detect_usable_range(_path(clip_name))
    assert got["n_frames"] == n
    # The tail onset is a threshold crossing on a noisy signal; a few frames either way is fine,
    # a different answer is not.
    assert abs(got["usable"][1] - last) <= 3, got["usable"]


def test_content_box_excludes_the_contaminated_tail():
    """A regression test for a real bug. The iOS overlay that ends every clip spans the FULL
    capture width, so unioning boxes over the whole file reported `zone_grows_from_east` as
    x=[0, 2435] rather than a pillarbox -- which would have skipped normalization entirely and
    handed Phase C a homography fit against black bars."""
    got = detect_usable_range(_path("zone_grows_from_east"))
    x0, x1, _, _ = got["content_box"]
    assert x0 > 0 and x1 < FRAME_W - 1, (
        f"content box {got['content_box']} spans the full frame -- the tail is leaking in again"
    )


def test_contamination_just_inside_the_search_boundary_is_still_found():
    """A regression test for a real bug, found by running the full pipeline rather than by a test.

    `training_gadget.mp4` cuts to a full-screen brawler splash at frame 508 of 599 -- 0.848 of the
    clip, one frame outside the 0.85 search window. That single excluded spike was still inside the
    sample used to set the threshold, where it raised the 6-sigma bar from ~6 to ~29 and hid every
    later frame of the very contamination being hunted. The detector reported the clip usable to
    557 and the classifier cheerfully deposited 50 frames of character art into the occupancy map.

    The two errors compound in the same direction, which is why the failure was total rather than
    marginal, and why "a few frames either way is fine" would not have caught it.
    """
    got = detect_usable_range(_path("training_gadget"))
    assert got["n_frames"] == 599
    assert abs(got["usable"][1] - 508) <= 3, (
        f"usable range {got['usable']} -- gameplay stops at 508, the rest is a splash screen"
    )


def test_a_late_spike_cannot_raise_the_threshold_that_should_catch_it():
    """The mechanism above, isolated from any particular clip.

    A synthetic signal: quiet gameplay, then contamination beginning just before the search
    boundary and continuing to the end. The onset must be found. Computing mean and sd over the
    untrimmed body lets the onset itself set a bar that the rest of the contamination cannot clear.
    """
    n = 600
    d = np.full(n, 1.0)
    d[300:310] = 3.0                      # ordinary gameplay violence, nowhere near the tail
    d[508] = 100.0                        # the cut, one frame inside the 0.85 boundary at 510
    d[509:] = 12.0                        # the contaminated tail: high, but far below that spike

    head, split = 10, int(n * 0.85)
    body = d[head:split]
    untrimmed = body.mean() + 6.0 * body.std()
    trimmed_body = body[body <= np.quantile(body, 0.98)]
    trimmed = trimmed_body.mean() + 6.0 * trimmed_body.std()

    assert untrimmed > 12.0, "the spike must be what inflates the threshold, or this proves nothing"
    assert trimmed < 12.0, "trimming must bring the threshold back under the contaminated tail"
    assert (d[split:] > trimmed).any() and not (d[split:] > untrimmed).any()


def test_portrait_decode_is_rejected(tmp_path):
    """These files are stored portrait with a rotation flag. Whether a reader honors container
    rotation has changed across OpenCV versions, so a source coming back taller than wide raises
    rather than feeding every later stage a sideways world."""
    portrait = tmp_path / "portrait.mp4"
    writer = cv2.VideoWriter(str(portrait), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (240, 480))
    if not writer.isOpened():
        pytest.skip("no mp4v encoder available")
    for _ in range(5):
        writer.write(np.full((480, 240, 3), 128, np.uint8))
    writer.release()

    with pytest.raises(ValueError, match="portrait"):
        detect_usable_range(portrait)


def test_reader_reports_its_own_timing():
    """The plan asks every stage to report its own timing from day one, so the 250 ms decision
    budget is answerable by measurement rather than argument once the pipeline exists."""
    reader = ClipReader(_path("standstill"))
    assert reader.decode_seconds == 0.0
    for _ in zip(range(25), reader):
        pass
    assert reader.decode_seconds > 0.0
