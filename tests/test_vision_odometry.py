"""Camera translation tracking. See Terrain_Perception_Build_Plan.md Phase F.

Split as elsewhere: the estimator is exercised on synthetic frames translated by KNOWN amounts, so
it is checked against an answer rather than against itself, and the real clips run under
`@pytest.mark.vision`.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
from brawl_vision.config import VisionConfig
from brawl_vision.sources import open_source
from brawl_vision.terrain.odometry import Odometry

CLIPS = Path(__file__).resolve().parent / "fixtures" / "vision"


@pytest.fixture(scope="module")
def plan():
    return build_rectify_plan(load_camera_model(), load_hud_mask())


def _textured(plan, seed=0):
    """A frame with dense, non-periodic, high-contrast detail everywhere. Deliberately NOT a tile
    lattice: a synthetic periodic pattern would make an aliasing failure look like success."""
    rng = np.random.default_rng(seed)
    w, h = plan.size_px
    img = rng.integers(0, 255, (h // 4, w // 4), dtype=np.uint8)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def _shift(img, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------------------
# the estimator, against known translations
# ---------------------------------------------------------------------------

def test_first_frame_initialises_without_moving(plan):
    odo = Odometry(plan)
    r = odo.update(_textured(plan))
    assert r.status == "init"
    assert r.delta_tiles == (0.0, 0.0) and r.position_tiles == (0.0, 0.0)


@pytest.mark.parametrize("dx,dy", [(0, 0), (5, 0), (0, -7), (-13, 9), (48, 48), (72, -60)])
def test_recovers_a_known_translation(plan, dx, dy):
    """Sign convention included: content sliding one way means the camera moved the other, and an
    inverted sign here would send Phase I's accumulated map off in exactly the wrong direction."""
    base = _textured(plan)
    odo = Odometry(plan)
    odo.update(base)
    r = odo.update(_shift(base, dx, dy))
    ppt = plan.pixels_per_tile
    assert r.status == "ok"
    assert r.delta_tiles[0] == pytest.approx(-dx / ppt, abs=0.01)
    assert r.delta_tiles[1] == pytest.approx(-dy / ppt, abs=0.01)


def test_a_full_tile_shift_is_not_confused_with_zero(plan):
    """Wall blocks repeat every `pixels_per_tile`, so a one-tile move is the case where a
    correlator that latched onto the lattice would confidently return zero."""
    base = _textured(plan)
    odo = Odometry(plan)
    odo.update(base)
    r = odo.update(_shift(base, plan.pixels_per_tile, 0))
    assert r.delta_tiles[0] == pytest.approx(-1.0, abs=0.02)


def test_position_integrates_across_frames(plan):
    base = _textured(plan)
    odo = Odometry(plan)
    odo.update(base)
    for _ in range(4):
        odo.update(_shift(base, 0, 0))       # reference is replaced each frame; motion is stepwise
    odo._prev = base
    for step in range(1, 5):
        odo.update(_shift(base, -12 * step, 0))
        base = _shift(base, -12 * step, 0)
    assert odo.position[0] == pytest.approx(12 * (1 + 2 + 3 + 4) / plan.pixels_per_tile, abs=0.05)


def test_a_moving_sprite_over_part_of_the_frame_does_not_capture_the_estimate(plan):
    """The failure that motivated the whole design. A single correlation over a large window
    reported 2.8 tiles of camera travel on real footage where the background provably did not
    move, because the strongest peak belonged to the foreground -- at a response of 0.92-0.98,
    indistinguishable from a good estimate. The median over a grid of windows outvotes it."""
    base = _textured(plan)
    moved = base.copy()
    w, h = plan.size_px
    blob = _shift(base, 60, 0)[h // 3:2 * h // 3, w // 3:2 * w // 3]
    moved[h // 3:2 * h // 3, w // 3:2 * w // 3] = blob      # a third of the frame slides
    odo = Odometry(plan)
    odo.update(base)
    r = odo.update(moved)
    assert abs(r.delta_tiles[0]) < 0.05, "the moving region captured the estimate"
    # Right answer, honestly low confidence: a third of the patch moving independently is real
    # cause to distrust the frame as terrain evidence, even though the median survived it.
    assert r.tracking and not r.ok
    assert r.segment == 0


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def test_uncorrelated_frames_are_called_a_cut(plan):
    odo = Odometry(plan)
    odo.update(_textured(plan, seed=1))
    r = odo.update(_textured(plan, seed=2))
    assert r.status == "lost"
    assert r.delta_tiles == (0.0, 0.0)
    assert r.position_tiles == (0.0, 0.0), "a cut must not move the accumulated position"
    assert r.segment == 1


def test_a_cut_reseeds_so_tracking_resumes_on_the_next_frame(plan):
    """Keeping the pre-cut frame as the reference would leave every later frame comparing against
    a stale world, i.e. permanently lost after one discontinuity."""
    odo = Odometry(plan)
    odo.update(_textured(plan, seed=1))
    after = _textured(plan, seed=2)
    assert odo.update(after).status == "lost"
    r = odo.update(_shift(after, 10, 0))
    assert r.status == "ok"
    assert r.delta_tiles[0] == pytest.approx(-10 / plan.pixels_per_tile, abs=0.01)


def test_an_implausibly_large_shift_is_refused(plan):
    cfg = VisionConfig(odometry_max_shift_tiles=0.1)
    base = _textured(plan)
    odo = Odometry(plan, cfg)
    odo.update(base)
    r = odo.update(_shift(base, 96, 0))          # 2 tiles, well past the bound
    assert r.status == "lost" and r.segment == 1


def test_window_disagreement_is_what_flags_a_doubtful_frame(plan):
    """Response cannot do this job. Half the frame moving one way and half the other correlates
    strongly in EVERY individual window and is still not a rigid translation.

    The outcome is `uncertain`, not `lost`: there is no discontinuity here, just no trustworthy
    answer, and a segment break would throw away a world frame that is still valid."""
    base = _textured(plan)
    w, h = plan.size_px
    split = base.copy()
    split[:, :w // 2] = _shift(base, 40, 0)[:, :w // 2]
    split[:, w // 2:] = _shift(base, -40, 0)[:, w // 2:]
    odo = Odometry(plan)
    odo.update(base)
    r = odo.update(split)
    assert r.response > 0.3, "the premise fails: individual windows did not correlate well"
    assert r.inlier_ratio < 0.75, "a bare majority was treated as agreement"
    assert r.status == "uncertain"
    assert not r.ok and r.tracking
    assert r.segment == 0, "a doubtful frame must not break the segment"


def test_reset_clears_position_but_advances_the_segment(plan):
    odo = Odometry(plan)
    base = _textured(plan)
    odo.update(base)
    odo.update(_shift(base, 24, 0))
    assert odo.position[0] != 0.0
    odo.reset()
    assert odo.position.tolist() == [0.0, 0.0] and odo.segment == 1
    assert odo.update(base).status == "init"


def test_a_non_rectified_frame_is_refused(plan):
    with pytest.raises(ValueError, match="rectified"):
        Odometry(plan).update(np.zeros((1126, 2002, 3), np.uint8))


def test_too_few_windows_is_a_construction_error(plan):
    with pytest.raises(ValueError, match="min_windows"):
        Odometry(plan, VisionConfig(odometry_window_px=896, odometry_stride_px=896,
                                    odometry_min_windows=8))


def test_bgr_and_grayscale_frames_agree(plan):
    base = _textured(plan)
    bgr = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    a, b = Odometry(plan), Odometry(plan)
    a.update(base)
    b.update(bgr)
    ra = a.update(_shift(base, 20, 0))
    rb = b.update(cv2.cvtColor(_shift(base, 20, 0).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    assert ra.delta_tiles[0] == pytest.approx(rb.delta_tiles[0], abs=1e-6)


# ---------------------------------------------------------------------------
# real footage
# ---------------------------------------------------------------------------

def _track(plan, clip, lo=0, hi=None, cfg=None):
    odo = Odometry(plan, cfg)
    results = []
    with open_source(CLIPS / f"{clip}.mp4") as src:
        for frame in src:
            if frame.index < lo:
                continue
            if hi is not None and frame.index > hi:
                break
            results.append(odo.update(plan.rectify(frame.image)))
    return odo, results


@pytest.mark.vision
def test_a_stationary_camera_does_not_drift(plan):
    """The plan's second acceptance criterion, and the one that catches a systematic scale error
    a walking test can mask. 400 frames of a genuinely still camera."""
    odo, results = _track(plan, "standstill", hi=400)
    assert np.linalg.norm(odo.position) < 0.05, f"drifted to {odo.position} over 400 frames"
    assert all(r.status != "lost" for r in results[1:])


@pytest.mark.vision
def test_the_stretch_with_no_camera_motion_reports_none(plan):
    """`counted_walking` f50-f145 is the regression case for the single-window failure: the
    background is provably identical across it (the camera never moved, only sprites did), and a
    single large correlation window reports 2.8 tiles of travel there."""
    odo, _ = _track(plan, "counted_walking", lo=50, hi=145)
    assert np.linalg.norm(odo.position) < 0.15, f"invented {odo.position} tiles of motion"


@pytest.mark.vision
def test_the_real_camera_leg_is_measured_consistently(plan):
    """The clip's one genuine camera movement. Pinned as a regression fixture per the plan; the
    tolerance is wide because what is asserted is the measurement's stability, not its agreement
    with a walk count -- see the plan on why the two are not the same number."""
    odo, results = _track(plan, "counted_walking", lo=250, hi=360)
    assert odo.position[1] == pytest.approx(3.55, abs=0.25)
    assert abs(odo.position[0]) < 0.25, "a straight leg picked up lateral drift"
    assert all(r.status != "lost" for r in results[1:])


@pytest.mark.vision
@pytest.mark.parametrize("clip", ["zone_grows_from_east", "showdown_alternate_map2"])
def test_tracking_holds_across_a_long_real_clip(plan, clip):
    # 400 frames, not the whole clip: decode dominates the runtime of this file, and continuous
    # tracking either holds for 400 frames or it does not. The full-length sweep lives in the
    # plan's Phase F table, where it is a recorded measurement rather than a per-run cost.
    _, results = _track(plan, clip, hi=400)
    ok = [r for r in results if r.status == "ok"]
    assert len(ok) > 0.97 * (len(results) - 1)
    assert np.median([r.agreement_tiles for r in ok]) < 0.02
    assert min(r.inlier_ratio for r in ok) >= 0.75
