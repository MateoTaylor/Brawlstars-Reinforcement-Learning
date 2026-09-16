"""`brawl_deployment.perception.lattice` -- where the game's tiles are, from the crates.

The plan is an identity homography at 48 px a tile, so a box's anchor maps to tiles by hand
arithmetic. The boxes are real `Detection`s, at a crate's real on-screen size: the height test reads
`xyxy`, and a stub box of an invented size would fail or pass it by accident.
"""
import numpy as np
import pytest

from brawl_deployment.perception.lattice import (CRATE_HEIGHT_PX, HEIGHT_RATIO, LOCK_SIGHTINGS,
                                                 REBASE_TILES, LatticePhase)
from brawl_deployment.perception.loot import EDGE_MARGIN_PX
from brawl_vision.object_detection.detector import Detection
from brawl_vision.object_detection.project import CRATE_ANCHOR_FRAC
from brawl_vision.object_detection.projectile_detection.classes import CUBE_BOX, PROJECTILE
from brawl_vision.terrain.odometry import OdometryResult

PPT = 48
ORIGIN = (-10.0, -6.0)


class _Plan:
    """Identity homography: viewport px / 48 + origin = camera-relative tile."""
    viewport = (2002, 1126)
    pixels_per_tile = PPT
    origin_tile = ORIGIN
    M = np.eye(3)

    def rect_to_tile(self, rect):
        return np.asarray(rect, np.float64).reshape(-1, 2) / PPT + np.asarray(ORIGIN)


PLAN = _Plan()


def _odo(position=(0.0, 0.0), status="ok", segment=0):
    return OdometryResult(delta_tiles=(0.0, 0.0), position_tiles=position, response=1.0,
                          agreement_tiles=0.0, inlier_ratio=1.0, n_windows=11, status=status,
                          segment=segment)


def _crate(tile, height_frac=1.0, label=CUBE_BOX, conf=0.9, width=120.0):
    """A crate box whose anchor projects to camera-relative `tile`, `height_frac` of the height a
    crate that low on screen is expected to be."""
    ax = (tile[0] - ORIGIN[0]) * PPT
    ay = (tile[1] - ORIGIN[1]) * PPT
    a, b = CRATE_HEIGHT_PX
    # anchor = y1 - frac * h, with h = height_frac * (a + b * y1); solve for y1.
    y1 = (ay + CRATE_ANCHOR_FRAC * height_frac * a) / (1.0 - CRATE_ANCHOR_FRAC * height_frac * b)
    h = height_frac * (a + b * y1)
    return Detection(label, conf, (ax - width / 2, y1 - h, ax + width / 2, y1))


def _world_anchor(lat, det, odo):
    """Where the lattice's world frame puts this crate's anchor."""
    x0, y0, x1, y1 = det.xyxy
    ax, ay = det.anchor(CRATE_ANCHOR_FRAC)
    world = lat.world(odo)
    return (ax / PPT + ORIGIN[0] + world.position_tiles[0],
            ay / PPT + ORIGIN[1] + world.position_tiles[1])


def test_the_helper_box_really_projects_where_it_says():
    """The rest of the file leans on `_crate`; check it once against the real formula."""
    d = _crate((2.8, 1.25))
    ax, ay = d.anchor(CRATE_ANCHOR_FRAC)
    assert (ax / PPT + ORIGIN[0], ay / PPT + ORIGIN[1]) == pytest.approx((2.8, 1.25))
    x0, y0, x1, y1 = d.xyxy
    assert y1 - y0 == pytest.approx(CRATE_HEIGHT_PX[0] + CRATE_HEIGHT_PX[1] * y1)


def test_a_crate_off_its_tile_centre_moves_the_world_onto_the_game_lattice():
    """The crate reads (+0.3, -0.25) off a tile centre. Once there are enough sightings the frame
    moves by exactly that, and the crate's anchor sits on a tile centre in the new frame."""
    lat = LatticePhase()
    crate = _crate((2.8, 1.25))
    odo = _odo()
    statuses = [lat.update([crate], PLAN, odo).status for _ in range(LOCK_SIGHTINGS)]
    assert statuses == ["unlocked"] * (LOCK_SIGHTINGS - 1) + ["rebased"]
    assert lat.phase == pytest.approx((0.3, -0.25))
    assert _world_anchor(lat, crate, odo) == pytest.approx((2.5, 1.5))
    # And it holds there: more of the same evidence is not a reason to move again.
    assert lat.update([crate], PLAN, odo).status == "locked"
    assert lat.epoch == 1


def test_the_phase_is_read_in_the_odometry_frame_not_the_camera_frame():
    """The camera moves; the crate does not. Sightings from two camera positions agree only once
    the odometry position is added, and the lock must come out the same as from one position."""
    lat = LatticePhase()
    for pos in [(0.0, 0.0), (0.7, -0.4), (1.9, 0.35)]:
        # The same world crate at (2.8, 1.25), seen from wherever the camera now is.
        lat.update([_crate((2.8 - pos[0], 1.25 - pos[1]))], PLAN, _odo(pos))
    assert lat.phase == pytest.approx((0.3, -0.25))


def test_a_phase_inside_the_tolerance_leaves_the_frame_alone():
    """Every re-lock costs a reset of every tracker. Under `REBASE_TILES` it is not worth one."""
    lat = LatticePhase()
    off = REBASE_TILES * 0.6
    for _ in range(5):
        res = lat.update([_crate((2.5 + off, 1.5 - off))], PLAN, _odo())
    assert res.status == "locked"
    assert res.estimate == pytest.approx((off, -off))
    assert lat.phase == (0.0, 0.0) and lat.epoch == 0


def test_a_partly_hidden_crate_is_not_read():
    """A short box has lost its bottom behind something, and its anchor climbs with it: the
    replay's short boxes read a median 0.17 - 0.25 tile off in y. Just under the ratio is out;
    just over it is in."""
    lat = LatticePhase()
    for _ in range(5):
        res = lat.update([_crate((2.8, 1.25), height_frac=HEIGHT_RATIO - 0.02)], PLAN, _odo())
    assert res.n_sightings == 0 and res.status == "unlocked"
    res = lat.update([_crate((2.8, 1.25), height_frac=HEIGHT_RATIO + 0.02)], PLAN, _odo())
    assert res.n_used == 1


def test_only_crates_are_read_and_not_near_the_frame_edge():
    lat = LatticePhase()
    shot = _crate((2.8, 1.25), label=PROJECTILE)
    x0, y0, x1, y1 = _crate((2.8, 1.25)).xyxy
    # The same crate slid left until its box is just inside the edge margin.
    near_edge = Detection(CUBE_BOX, 0.9, (EDGE_MARGIN_PX - 1.0, y0, EDGE_MARGIN_PX - 1.0 + x1 - x0, y1))
    res = lat.update([shot, near_edge], PLAN, _odo())
    assert res.n_used == 0


def test_a_crate_boxed_twice_is_one_sighting():
    """The detector runs without NMS and boxes one crate twice (`loot.py`). Counted twice, one
    crate would reach the lock alone in half the ticks and weigh double in the mean."""
    lat = LatticePhase()
    a = _crate((2.8, 1.25), conf=0.9)
    b = _crate((2.85, 1.25), conf=0.8)
    res = lat.update([a, b], PLAN, _odo())
    assert res.n_used == 1


def test_readings_either_side_of_half_a_tile_are_neighbours():
    """+0.45 and -0.45 are 0.1 apart on the lattice. A plain mean of the two says 0, which is
    the one answer they both rule out."""
    lat = LatticePhase()
    for _ in range(3):
        lat.update([_crate((2.95, 1.5)), _crate((6.05, 1.5))], PLAN, _odo())
    est = lat.estimate()
    assert abs(abs(est[0]) - 0.5) < 1e-6


def test_a_re_lock_moves_the_frame_by_the_short_way_round():
    """The frame jumps to the estimate by at most half a tile. Evidence moving from +0.4 to -0.4
    is a move of +0.2, and the applied phase goes to +0.6, not back through zero to -0.4."""
    lat = LatticePhase()
    for _ in range(3):
        lat.update([_crate((2.9, 1.5))], PLAN, _odo())
    assert lat.phase[0] == pytest.approx(0.4)
    for _ in range(40):
        res = lat.update([_crate((2.1, 1.5))], PLAN, _odo())
    assert res.estimate[0] == pytest.approx(-0.4, abs=0.05)
    assert 0.5 < lat.phase[0] < 0.9                     # up through +0.5, the short way
    gap = (lat.phase[0] - res.estimate[0] + 0.5) % 1.0 - 0.5
    assert abs(gap) <= REBASE_TILES                     # and on the estimate, modulo a tile
    assert lat.epoch == 2


def test_a_new_odometry_segment_starts_over_at_phase_zero():
    """A new segment's world frame has its own arbitrary phase. Keeping the old one would shift
    the new frame by a number that describes a frame that no longer exists."""
    lat = LatticePhase()
    for _ in range(3):
        lat.update([_crate((2.8, 1.25))], PLAN, _odo(segment=0))
    epoch = lat.epoch
    res = lat.update([], PLAN, _odo(status="init", segment=1))
    assert lat.phase == (0.0, 0.0) and res.n_sightings == 0
    assert lat.epoch == epoch + 1
    assert lat.world(_odo(segment=1)).segment == lat.epoch


def test_the_first_tick_adopts_odometry_segment_without_a_new_epoch():
    """Epoch 0 on the first tick, whatever segment odometry is on, because `OccupancyMap` starts
    at 0 and has no `None` sentinel: a bump here would drop its first deposit for nothing."""
    lat = LatticePhase()
    lat.update([], PLAN, _odo(segment=4))
    assert lat.epoch == 0
    lat.update([], PLAN, _odo(segment=4))
    assert lat.epoch == 0


def test_an_untrusted_tick_reads_nothing():
    """"uncertain" still tracks, but its position is not evidence: a crate placed with it would
    carry the odometry error into the phase."""
    lat = LatticePhase()
    for _ in range(5):
        res = lat.update([_crate((2.8, 1.25))], PLAN, _odo(status="uncertain"))
    assert res.status == "uncertain" and res.n_sightings == 0


def test_the_world_odometry_is_the_same_result_shifted_and_re_labelled():
    lat = LatticePhase()
    for _ in range(3):
        lat.update([_crate((2.8, 1.25))], PLAN, _odo())
    raw = _odo(position=(4.0, -2.0))
    world = lat.world(raw)
    assert world.position_tiles == pytest.approx((3.7, -1.75))
    assert world.segment == lat.epoch != raw.segment
    assert (world.status, world.delta_tiles, world.response) == (raw.status, raw.delta_tiles,
                                                                 raw.response)


def test_a_reset_is_a_new_world_frame():
    """The loop's match start. The last match's crates say nothing about this map."""
    lat = LatticePhase()
    for _ in range(3):
        lat.update([_crate((2.8, 1.25))], PLAN, _odo())
    epoch = lat.epoch
    lat.reset(0)
    assert lat.phase == (0.0, 0.0) and lat.estimate() is None and lat.epoch == epoch + 1
    lat.update([], PLAN, _odo(segment=0))
    assert lat.epoch == epoch + 1               # same odometry segment: no second bump
