"""brawl_deployment's projectile tracker -- velocity and time-to-closest from anonymous boxes.

Synthetic detections through the same stub plan `test_deployment_tracker.py` uses: no emulator, no
model weights, no fixture. What is under test is bookkeeping over tile coordinates, and feeding it
real YOLO output would test the detector instead.

The stub anchors at the box CENTRE rather than the entity tracker's 0.30, because that is what
`PROJ_ANCHOR_FRAC` is. If those two ever diverge this file is measuring the wrong thing, so the
first test pins it.
"""
import math

import numpy as np
import pytest

from brawl_deployment.perception.projectiles import (
    DUPLICATE_TILES, GATE_NOISE_TILES, MAX_COAST_S, MAX_PROJ_TILES_S, MIN_SAMPLES,
    PROJ_ANCHOR_FRAC, ProjectileTracker, _dedupe,
)
from brawl_vision.object_detection.detector import Detection

PPT = 48
TICK = 0.05          # 20 Hz, the rate this tracker is designed for


class StubPlan:
    """The two things `to_tiles` touches: a pixel->rect matrix and rect->tile."""
    M = np.eye(3, dtype=np.float64)

    @staticmethod
    def rect_to_tile(rect):
        return np.asarray(rect, np.float64).reshape(-1, 2) / PPT


class StubOdo:
    def __init__(self, position=(0.0, 0.0), status="ok", segment=0):
        self.position_tiles = position
        self.status = status
        self.segment = segment


def det_at(tx, ty, conf=0.9, size=10.0):
    """A box whose CENTRE lands exactly on tile `(tx, ty)`."""
    px, py = tx * PPT, ty * PPT
    return Detection(label="Projectile", confidence=conf,
                     xyxy=(px - size / 2, py - size / 2, px + size / 2, py + size / 2))


def run(trk, frames, t0=0.0, odo=None):
    """Feed a list of per-tick detection lists at 20 Hz; return the last result."""
    res = None
    for i, dets in enumerate(frames):
        res = trk.update(dets, StubPlan(), odo or StubOdo(), t=t0 + i * TICK)
    return res


def test_the_stub_places_detections_where_asked():
    from brawl_vision.object_detection.project import to_tiles
    (_, (tx, ty)), = to_tiles([det_at(3.0, -2.5)], StubPlan(), anchor_frac=PROJ_ANCHOR_FRAC)
    assert (tx, ty) == pytest.approx((3.0, -2.5), abs=1e-9)


# ---------------------------------------------------------------- velocity

def test_two_samples_are_enough_for_a_velocity():
    """The whole reason detection runs at 20 Hz rather than the 4 Hz decision rate."""
    trk = ProjectileTracker()
    res = run(trk, [[det_at(0.0, 0.0)], [det_at(0.5, 0.0)]])
    assert len(res.live) == 1
    assert res.live[0].vel == pytest.approx((10.0, 0.0), abs=1e-6)


def test_one_sample_is_not_reported_at_all():
    """A position with no direction. Reporting it with vel=(0,0) is not a lesser version of this
    -- `closest_approach` gives a stationary point t=0, `obs_select` sorts ascending, so it would
    outrank every real incoming projectile and evict them from the twelve slots."""
    trk = ProjectileTracker()
    res = run(trk, [[det_at(2.0, 2.0)]])
    assert res.n_detections == 1
    assert res.live == []
    assert trk.tracks, "the track must exist for the NEXT tick to associate against"


def test_min_samples_is_what_gates_it_not_an_accident_of_timing():
    trk = ProjectileTracker(min_samples=3)
    res = run(trk, [[det_at(0.0, 0.0)], [det_at(0.4, 0.0)]])
    assert res.live == []
    res = trk.update([det_at(0.8, 0.0)], StubPlan(), StubOdo(), t=2 * TICK)
    assert len(res.live) == 1


def test_velocity_is_fitted_over_the_whole_span_not_the_last_two_samples():
    """The measurable consequence of `_advance`'s span estimator, and the reason it was chosen.

    One noisy sample in the middle of a straight flight. A last-two-samples difference carries that
    noise into the answer at full weight forever after; the span fit divides it by the whole
    elapsed time, so by the sixth sample it is nearly gone.
    """
    true_v = 8.0
    frames = []
    for i in range(6):
        x = true_v * i * TICK + (0.15 if i == 2 else 0.0)     # one 0.15-tile anchor error
        frames.append([det_at(x, 0.0)])
    span = run(ProjectileTracker(), frames).live[0].vel[0]

    last_two = (frames[-1][0].centre[0] - frames[-2][0].centre[0]) / PPT / TICK
    assert abs(span - true_v) < abs(last_two - true_v)
    assert span == pytest.approx(true_v, abs=0.2)


def test_a_stationary_camera_is_not_required():
    """Association is in world tiles for `tracker.py`'s reason -- in pixels a projectile fired
    across a panning camera is indistinguishable from one fired the other way."""
    trk = ProjectileTracker()
    for i in range(4):
        cam = (0.5 * i, 0.0)                       # camera chases the projectile
        world_x = 0.4 * i
        res = trk.update([det_at(world_x - cam[0], 0.0)], StubPlan(),
                         StubOdo(position=cam), t=i * TICK)
    assert res.live[0].vel[0] == pytest.approx(8.0, abs=1e-6)


# ---------------------------------------------------------------- the gate

def test_the_gate_covers_the_fastest_measured_projectile():
    """18.4 tiles/s was the fastest unambiguous step in the footage. A gate that drops it would
    restart the track every tick and never produce a velocity."""
    step = 18.4 * TICK
    trk = ProjectileTracker()
    res = run(trk, [[det_at(0.0, 0.0)], [det_at(step, 0.0)]])
    assert len(res.live) == 1, "the fastest measured projectile must stay one track"


def test_a_jump_beyond_the_gate_becomes_a_new_projectile():
    trk = ProjectileTracker()
    far = (GATE_NOISE_TILES + MAX_PROJ_TILES_S * TICK) * 1.5
    run(trk, [[det_at(0.0, 0.0)], [det_at(far, 0.0)]])
    assert len(trk.tracks) == 2


def test_the_gate_widens_when_a_tick_is_slow():
    """A dropped frame doubles the true step. A gate sized for the nominal 20 Hz would reject
    every correct match exactly when the loop is already struggling."""
    trk = ProjectileTracker()
    trk.update([det_at(0.0, 0.0)], StubPlan(), StubOdo(), t=0.0)
    assert trk.gate_for(2 * TICK) == pytest.approx(trk.gate_for(TICK) + MAX_PROJ_TILES_S * TICK)
    res = trk.update([det_at(15.0 * 2 * TICK, 0.0)], StubPlan(), StubOdo(), t=2 * TICK)
    assert len(res.live) == 1


# ---------------------------------------------------------------- duplicate boxes

def test_a_duplicate_box_does_not_become_a_second_projectile():
    """Measured: 17 of 269 simultaneous pairs across six clips are one projectile detected twice,
    and every one of them projected within 0.061 tiles. Two tracks on one projectile split the
    association and each ends up with a velocity from a half-length baseline."""
    trk = ProjectileTracker()
    twin = [det_at(1.0, 1.0, conf=0.9), det_at(1.06, 1.0, conf=0.4)]
    res = trk.update(twin, StubPlan(), StubOdo(), t=0.0)
    assert res.n_detections == 2
    assert res.n_merged == 1
    assert len(trk.tracks) == 1


def test_the_surviving_duplicate_is_the_more_confident_one():
    kept = _dedupe([((1.0, 1.0), 0.4), ((1.0, 1.0), 0.95)], DUPLICATE_TILES)
    assert len(kept) == 1 and kept[0][1] == pytest.approx(0.95)


def test_two_distinct_projectiles_close_together_are_both_kept():
    """The failure `_decode_end2end` refuses NMS to avoid: a volley along one line. The closest
    genuinely distinct pair in the footage projected 0.910 tiles apart, three times the threshold
    -- and the direction of the error matters, because a wrong merge hides an incoming shot."""
    volley = [det_at(1.0, 1.0), det_at(1.91, 1.0)]
    res = ProjectileTracker().update(volley, StubPlan(), StubOdo(), t=0.0)
    assert res.n_merged == 0
    assert len(res.live) == 0 and len(ProjectileTracker().tracks) == 0


def test_the_threshold_sits_inside_the_measured_gap():
    """Duplicates topped out at 0.061 tiles and the closest distinct pair was 0.910. A threshold
    outside that window is either merging real projectiles or letting duplicates through, and the
    module docstring's table is the only thing that says where the window is."""
    assert 0.061 < DUPLICATE_TILES < 0.910


def test_dedupe_does_not_depend_on_the_order_detections_arrive_in():
    """Confidence decides which of a pair survives, so the same frame must dedupe the same way
    however the detector happened to sort it."""
    pts = [((0.0, 0.0), 0.7), ((0.02, 0.0), 0.9), ((5.0, 0.0), 0.8)]
    for order in (pts, list(reversed(pts)), [pts[1], pts[2], pts[0]]):
        assert _dedupe(order, DUPLICATE_TILES) == [((0.02, 0.0), 0.9), ((5.0, 0.0), 0.8)]


def test_dedupe_measures_against_survivors_not_against_dropped_points():
    """Standard NMS semantics, and NOT transitive clustering: a point beyond the threshold from
    the survivor is kept even though it is within the threshold of something the survivor already
    absorbed. That is the right behaviour here rather than a wart -- real duplicates sit inside
    0.061 tiles, so a 0.2-spaced chain is three projectiles, not one smeared detection, and
    collapsing it would hide two of them."""
    chain = [((0.0, 0.0), 0.9), ((0.2, 0.0), 0.8), ((0.4, 0.0), 0.7)]
    assert _dedupe(chain, DUPLICATE_TILES) == [((0.0, 0.0), 0.9), ((0.4, 0.0), 0.7)]


# ---------------------------------------------------------------- coasting and expiry

def test_a_missed_tick_keeps_the_track_but_reports_nothing():
    """Coasting buys association continuity across one dropped detection. It is not a claim that
    the projectile still exists -- the usual reason one stops being detected is that it hit
    something, and reporting it tells the policy to dodge a bullet that is gone."""
    trk = ProjectileTracker()
    run(trk, [[det_at(0.0, 0.0)], [det_at(0.4, 0.0)]])
    res = trk.update([], StubPlan(), StubOdo(), t=2 * TICK)
    assert res.live == []
    assert len(trk.tracks) == 1, "still there for association"


def test_a_reacquired_projectile_keeps_its_original_velocity_baseline():
    """The payoff for coasting: without it the reacquisition starts a fresh track and the policy
    waits another two ticks for a velocity it already had."""
    trk = ProjectileTracker()
    run(trk, [[det_at(0.0, 0.0)], [det_at(0.4, 0.0)]])
    trk.update([], StubPlan(), StubOdo(), t=2 * TICK)
    res = trk.update([det_at(1.2, 0.0)], StubPlan(), StubOdo(), t=3 * TICK)
    assert len(res.live) == 1
    assert res.live[0].hits == 3
    assert res.live[0].vel[0] == pytest.approx(8.0, abs=1e-6)


def test_the_coast_window_tolerates_exactly_one_missed_tick_at_the_deployed_rate():
    """The constant is in seconds but the property is in TICKS: one miss survives, two do not.
    That held at 20 Hz by 17 ms and had to be rechecked when the rate moved to 12 (§6.13), which
    is the whole reason this test exists alongside the absolute-time one below."""
    from brawl_deployment.config import DeploymentConfig, resolve_rates

    class _Sim:
        dt, action_repeat = 0.05, 5

    period = resolve_rates(_Sim(), DeploymentConfig()).tick_seconds
    assert period < MAX_COAST_S < 2 * period


def test_a_track_is_dropped_once_it_outlives_the_coast_window():
    trk = ProjectileTracker()
    run(trk, [[det_at(0.0, 0.0)], [det_at(0.4, 0.0)]])
    trk.update([], StubPlan(), StubOdo(), t=MAX_COAST_S + 3 * TICK)
    assert trk.tracks == []


# ---------------------------------------------------------------- odometry gating

def test_an_unusable_odometry_tick_reports_nothing_and_costs_no_track():
    """`uncertain` has a world frame that is not trustworthy evidence. Admitting it would move
    every track by the odometry error and read as every projectile changing course at once."""
    trk = ProjectileTracker()
    run(trk, [[det_at(0.0, 0.0)], [det_at(0.4, 0.0)]])
    res = trk.update([det_at(50.0, 50.0)], StubPlan(), StubOdo(status="uncertain"), t=2 * TICK)
    assert res.status == "uncertain"
    assert res.live == []
    assert len(trk.tracks) == 1 and trk.tracks[0].vel[0] == pytest.approx(8.0, abs=1e-6)


def test_a_segment_change_clears_everything():
    """A new segment is a new world frame with no defined offset to the old one, so every stored
    position is meaningless. A projectile's whole lifetime is shorter than the discontinuity."""
    trk = ProjectileTracker()
    run(trk, [[det_at(0.0, 0.0)], [det_at(0.4, 0.0)]])
    res = trk.update([det_at(0.8, 0.0)], StubPlan(), StubOdo(segment=1), t=2 * TICK)
    assert res.status == "reset"
    assert trk.tracks == []
    assert res.segment == 1


def test_the_first_tick_adopts_odometrys_segment_rather_than_resetting_on_it():
    trk = ProjectileTracker()
    res = trk.update([det_at(0.0, 0.0)], StubPlan(), StubOdo(segment=7), t=0.0)
    assert res.status == "ok" and len(trk.tracks) == 1


# ---------------------------------------------------------------- the observation

def test_snapshot_reports_positions_relative_to_the_hero():
    trk = ProjectileTracker()
    run(trk, [[det_at(5.0, 3.0)], [det_at(5.4, 3.0)]])
    (rel, vel, _), = trk.snapshot((1.0, 1.0))
    assert rel == pytest.approx((4.4, 2.0), abs=1e-5)
    assert vel == pytest.approx((8.0, 0.0), abs=1e-5)


def test_time_to_closest_is_the_simulators_own_function():
    """Not "the same formula" -- the same call. The value has to mean in deployment exactly what
    it meant in training, including the clamp at zero."""
    import torch

    from brawl_sim.core.geometry import closest_approach
    trk = ProjectileTracker()
    run(trk, [[det_at(0.0, 0.0)], [det_at(0.5, 0.0)]])
    hero = (4.0, 1.0)
    (_, _, ttc), = trk.snapshot(hero)
    tr = trk.live()[0]
    want, _ = closest_approach(torch.tensor([tr.pos]), torch.tensor([tr.vel]),
                               torch.tensor([hero]))
    assert ttc == pytest.approx(float(want[0]), abs=1e-6)


def test_a_projectile_already_past_the_hero_reports_zero_not_a_negative_time():
    """The clamp the sim applies. A negative time in this column would be a value the policy never
    saw in training, in the field `obs_select` sorts the whole group by."""
    trk = ProjectileTracker()
    run(trk, [[det_at(5.0, 0.0)], [det_at(5.5, 0.0)]])     # moving +x, away from a hero at x=0
    (_, _, ttc), = trk.snapshot((0.0, 0.0))
    assert ttc == 0.0


def test_snapshot_is_empty_when_nothing_is_confirmed():
    trk = ProjectileTracker()
    run(trk, [[det_at(1.0, 1.0)]])
    assert trk.snapshot((0.0, 0.0)) == []


def test_snapshot_does_not_order_the_slots():
    """`obs_select` sorts by `time_to_closest` and takes the nearest twelve. Sorting here too would
    be harmless today and wrong the moment the two disagree about ties or about `alive`."""
    trk = ProjectileTracker()
    near, far = [det_at(1.0, 0.0), det_at(9.0, 0.0)], [det_at(1.4, 0.0), det_at(9.4, 0.0)]
    run(trk, [near, far])
    ttcs = [s[2] for s in trk.snapshot((20.0, 0.0))]
    assert len(ttcs) == 2
    assert ttcs != sorted(ttcs), "insertion order, not threat order"


# ---------------------------------------------------------------- constants

def test_the_gate_is_wider_than_the_fastest_projectile_the_simulator_can_fire():
    """`configs/brawlers.yaml` tops out at super_proj_speed 12.0 tiles/s and the footage measured
    18.4, so the constant is sized by the game rather than by the sim. If someone later reads it
    as a sim-derived number this is where that goes wrong."""
    import yaml
    speeds = []
    for kit in yaml.safe_load(open("configs/brawlers.yaml")).values():
        if isinstance(kit, dict):
            speeds += [v for k, v in kit.items()
                       if k in ("proj_speed", "super_proj_speed") and isinstance(v, (int, float))]
    assert speeds, "no projectile speeds found -- the roster file moved"
    assert MAX_PROJ_TILES_S > max(speeds)


def test_min_samples_is_two_because_a_velocity_needs_two_points():
    assert MIN_SAMPLES == 2


def test_the_projectile_gate_is_tighter_at_rest_than_the_entity_gate():
    """The entity tracker's 1.0 tiles was sized around a brawler box that encloses a nameplate, a
    health bar and an aura. A projectile box encloses a projectile."""
    from brawl_deployment.perception.tracker import GATE_NOISE_TILES as ENTITY_NOISE
    assert GATE_NOISE_TILES < ENTITY_NOISE


def test_projectiles_are_anchored_at_the_box_centre_not_the_brawler_anchor():
    from brawl_vision.object_detection.project import DEFAULT_ANCHOR_FRAC
    assert PROJ_ANCHOR_FRAC == 0.5
    assert PROJ_ANCHOR_FRAC != DEFAULT_ANCHOR_FRAC


def test_a_tracker_holds_no_state_across_matches_once_reset():
    trk = ProjectileTracker()
    run(trk, [[det_at(0.0, 0.0)], [det_at(0.4, 0.0)]])
    trk.reset(0)
    assert trk.tracks == [] and trk.live() == [] and trk.snapshot((0.0, 0.0)) == []


def test_speed_is_the_norm_of_the_velocity():
    trk = ProjectileTracker()
    run(trk, [[det_at(0.0, 0.0)], [det_at(0.3, 0.4)]])
    assert trk.live()[0].speed == pytest.approx(math.hypot(6.0, 8.0), abs=1e-6)
