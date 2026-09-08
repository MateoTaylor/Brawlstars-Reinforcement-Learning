"""brawl_deployment's entity tracker -- identity and velocity from anonymous boxes.

Runs entirely on synthetic detections through a stub rectification plan: no emulator, no model
weights, no fixture. The tracker's job is bookkeeping over tile coordinates, and feeding it real
YOLO output would test the detector instead of the thing under test.
"""
import numpy as np
import pytest

from brawl_deployment.perception import EntityTracker
from brawl_deployment.perception.tracker import GATE_NOISE_TILES, MAX_WALK_TILES_S, _greedy_match
from brawl_vision.object_detection.detector import Detection

PPT = 48


class StubPlan:
    """The two things `to_tiles` touches: a pixel->rect matrix and rect->tile. Identity warp and a
    plain scale, so a detection's tile position is exactly what the test asked for."""
    M = np.eye(3, dtype=np.float64)

    @staticmethod
    def rect_to_tile(rect):
        return np.asarray(rect, np.float64).reshape(-1, 2) / PPT


class StubOdo:
    def __init__(self, position=(0.0, 0.0), status="ok", segment=0):
        self.position_tiles = position
        self.status = status
        self.segment = segment


def det_at(tx, ty, label="enemy", conf=0.9):
    """A box whose 0.30 anchor lands exactly on tile `(tx, ty)`."""
    px, py = tx * PPT, ty * PPT
    h = 10.0
    return Detection(label=label, confidence=conf,
                     xyxy=(px - 5.0, py + 0.3 * h - h, px + 5.0, py + 0.3 * h))


def test_the_stub_places_detections_where_asked():
    """If this drifts every other test in the file is measuring the wrong thing."""
    from brawl_vision.object_detection.project import to_tiles
    (d, (tx, ty)), = to_tiles([det_at(3.0, -2.5)], StubPlan())
    assert (tx, ty) == pytest.approx((3.0, -2.5), abs=1e-9)


# ---------------------------------------------------------------- identity and slots

def test_a_stationary_entity_keeps_one_slot_and_reads_as_stationary():
    trk = EntityTracker()
    for i in range(6):
        res = trk.update([det_at(4.0, 4.0)], StubPlan(), StubOdo(), t=i * 0.25)
    live = [e for e in res.enemies if e]
    assert len(live) == 1
    assert live[0].slot == 0
    assert np.hypot(*live[0].vel) == pytest.approx(0.0, abs=1e-6)


def test_velocity_is_recovered_in_tiles_per_second():
    """A constant 2 tiles/s must read as 2 tiles/s. The EMA lags, so check it converges rather
    than demanding the first estimate be exact."""
    trk = EntityTracker()
    for i in range(10):
        res = trk.update([det_at(1.0 + 0.5 * i, 3.0)], StubPlan(), StubOdo(), t=i * 0.25)
    e = [x for x in res.enemies if x][0]
    assert e.vel[0] == pytest.approx(2.0, abs=0.05)
    assert e.vel[1] == pytest.approx(0.0, abs=1e-6)


def test_camera_motion_is_cancelled_by_odometry():
    """THE point of tracking in tile space. A stationary enemy while the camera pans must not
    look like a moving one -- in pixels it would be crossing the screen."""
    trk = EntityTracker()
    for i in range(8):
        cam = (0.6 * i, 0.3 * i)                       # camera runs away
        rel = (5.0 - cam[0], 5.0 - cam[1])             # so the enemy slides the other way on screen
        res = trk.update([det_at(*rel)], StubPlan(), StubOdo(position=cam), t=i * 0.25)
    e = [x for x in res.enemies if x][0]
    assert e.pos == pytest.approx((5.0, 5.0), abs=1e-6)
    assert np.hypot(*e.vel) == pytest.approx(0.0, abs=1e-6), "camera motion leaked into velocity"


def test_slots_are_index_stable_while_a_track_lives():
    """`agent_obs_lowinfo.yaml` declares enemy slots index-stable within an episode. If slot 1
    silently becomes a different enemy, the policy's memory of it is worse than useless."""
    trk = EntityTracker()
    a, b = 2.0, 12.0
    for i in range(4):
        trk.update([det_at(a, 2.0), det_at(b, 2.0)], StubPlan(), StubOdo(), t=i * 0.25)
    slots = {e.slot: e.id for e in trk.enemies.values()}
    for i in range(4, 10):
        a += 0.4                                        # one walks, the other does not
        trk.update([det_at(a, 2.0), det_at(b, 2.0)], StubPlan(), StubOdo(), t=i * 0.25)
    assert {e.slot: e.id for e in trk.enemies.values()} == slots


def test_a_freed_slot_is_reused_not_leaked():
    trk = EntityTracker(max_misses=1)
    for i in range(4):
        trk.update([det_at(2.0, 2.0), det_at(12.0, 2.0)], StubPlan(), StubOdo(), t=i * 0.25)
    assert sorted(trk.enemies) == [0, 1]
    for i in range(4, 9):                               # drop one entirely
        trk.update([det_at(12.0, 2.0)], StubPlan(), StubOdo(), t=i * 0.25)
    assert len(trk.enemies) == 1
    for i in range(9, 13):                              # a new enemy appears elsewhere
        trk.update([det_at(12.0, 2.0), det_at(20.0, 9.0)], StubPlan(), StubOdo(), t=i * 0.25)
    assert sorted(trk.enemies) == [0, 1], "the vacated slot should have been reused"


# ---------------------------------------------------------------- association

def test_global_greedy_beats_per_track_greedy():
    """The specific failure a per-track loop has: the first track claims a point that fits the
    second far better. Ordered so a naive implementation gets it wrong."""
    tracks = [(0.0, 0.0), (1.0, 0.0)]
    points = [(1.1, 0.0), (5.0, 0.0)]
    matched, unmatched = _greedy_match(tracks, points, gate=2.0)
    # Per-track greedy hands track 0 the point at 1.1 (distance 1.1) and leaves track 1 nothing
    # inside the gate. Globally, the 0.1 pair must win first.
    assert matched == {1: 0}
    assert unmatched == [1]


def test_association_respects_the_gate():
    tracks = [(0.0, 0.0)]
    assert _greedy_match(tracks, [(0.5, 0.0)], gate=1.0)[0] == {0: 0}
    assert _greedy_match(tracks, [(1.5, 0.0)], gate=1.0)[0] == {}


def test_gate_widens_with_a_late_tick():
    """A dropped frame widens the true step. A gate fixed at the nominal rate would reject every
    correct match exactly when the loop is already behind."""
    trk = EntityTracker()
    trk.update([det_at(0.0, 0.0)], StubPlan(), StubOdo(), t=0.0)
    nominal = trk.gate_for(0.25, list(trk._pending))
    late = trk.gate_for(1.0, list(trk._pending))
    assert late > nominal
    assert nominal == pytest.approx(GATE_NOISE_TILES + MAX_WALK_TILES_S * 0.25)


def test_a_walking_entity_is_not_relabelled_between_ticks():
    """0.68 tiles is one decision of the roster's fastest walk. The same enemy must survive it."""
    trk = EntityTracker()
    x = 5.0
    for i in range(3):
        trk.update([det_at(x, 5.0)], StubPlan(), StubOdo(), t=i * 0.25)
        x += MAX_WALK_TILES_S * 0.25
    first = [e for e in trk.enemies.values()][0].id
    for i in range(3, 8):
        trk.update([det_at(x, 5.0)], StubPlan(), StubOdo(), t=i * 0.25)
        x += MAX_WALK_TILES_S * 0.25
    assert [e for e in trk.enemies.values()][0].id == first


# ---------------------------------------------------------------- lifecycle

def test_one_frame_noise_never_takes_a_slot():
    """Slots are index-stable and there are only nine, so a spurious box that claims one keeps it
    wrong until it ages out."""
    trk = EntityTracker()
    trk.update([det_at(7.0, 7.0)], StubPlan(), StubOdo(), t=0.0)
    assert trk.enemies == {}
    res = trk.update([], StubPlan(), StubOdo(), t=0.25)
    assert all(e is None for e in res.enemies)
    assert trk._pending == []


def test_a_track_coasts_then_retires():
    trk = EntityTracker(max_misses=2)
    for i in range(4):
        trk.update([det_at(6.0, 6.0)], StubPlan(), StubOdo(), t=i * 0.25)
    assert len(trk.enemies) == 1
    for i in range(4, 6):
        res = trk.update([], StubPlan(), StubOdo(), t=i * 0.25)
        assert len([e for e in res.enemies if e]) == 1, "must coast through a brief miss"
        assert not [e for e in res.enemies if e][0].seen_now
    res = trk.update([], StubPlan(), StubOdo(), t=1.5)
    assert all(e is None for e in res.enemies)


def test_seen_now_distinguishes_a_detection_from_a_coast():
    """`entities.revealed_to_hero` is exactly this bit: a coasting track is a remembered enemy,
    not a visible one, and reporting it as visible would invent information."""
    trk = EntityTracker()
    for i in range(3):
        res = trk.update([det_at(6.0, 6.0)], StubPlan(), StubOdo(), t=i * 0.25)
    assert [e for e in res.enemies if e][0].seen_now
    res = trk.update([], StubPlan(), StubOdo(), t=0.75)
    assert not [e for e in res.enemies if e][0].seen_now


# ---------------------------------------------------------------- odometry interlock

def test_the_first_tick_adopts_a_segment_rather_than_resetting():
    """A tracker starting in segment 0 has not just seen a segment CHANGE. Getting this wrong
    throws away the first decision of every match -- cheap, but silent."""
    trk = EntityTracker()
    res = trk.update([det_at(1.0, 1.0)], StubPlan(), StubOdo(segment=0), t=0.0)
    assert res.status == "ok"
    assert len(trk._pending) == 1


def test_a_new_segment_drops_every_track():
    """A segment change is a new world frame with no defined offset to the old one. Carrying a
    position across teleports it -- `occupancy.update` resets for the same reason."""
    trk = EntityTracker()
    for i in range(4):
        trk.update([det_at(3.0, 3.0)], StubPlan(), StubOdo(), t=i * 0.25)
    assert trk.enemies
    res = trk.update([det_at(3.0, 3.0)], StubPlan(), StubOdo(segment=1), t=1.0)
    assert res.status == "reset"
    assert trk.enemies == {} and trk.hero is None


def test_untrusted_odometry_does_not_move_tracks():
    """`uncertain` has a world frame that is not trustworthy evidence. Admitting a tick on one
    would displace every track by the odometry error at once -- every entity accelerating
    together, which is not a thing that happens."""
    trk = EntityTracker()
    for i in range(4):
        trk.update([det_at(3.0, 3.0)], StubPlan(), StubOdo(), t=i * 0.25)
    before = dict((s, (e.pos, e.vel)) for s, e in trk.enemies.items())
    res = trk.update([det_at(9.0, 9.0)], StubPlan(), StubOdo(status="uncertain"), t=1.0)
    assert res.status == "uncertain"
    assert dict((s, (e.pos, e.vel)) for s, e in trk.enemies.items()) == before


# ---------------------------------------------------------------- the hero

def test_the_hero_is_tracked_separately_from_the_slots():
    """The hero is not an `entities` slot -- it is the origin every `rel_pos` is measured from."""
    trk = EntityTracker()
    for i in range(4):
        res = trk.update([det_at(10.0, 10.0, label="player"), det_at(4.0, 4.0)],
                         StubPlan(), StubOdo(), t=i * 0.25)
    assert res.hero is not None and res.hero.label == "player"
    assert res.hero.pos == pytest.approx((10.0, 10.0), abs=1e-6)
    assert [e for e in res.enemies if e][0].label == "enemy"


def test_a_second_player_box_does_not_create_a_second_hero():
    trk = EntityTracker()
    trk.update([det_at(10.0, 10.0, label="player")], StubPlan(), StubOdo(), t=0.0)
    res = trk.update([det_at(10.2, 10.0, label="player"), det_at(18.0, 2.0, label="player")],
                     StubPlan(), StubOdo(), t=0.25)
    assert res.hero.pos == pytest.approx((10.2, 10.0), abs=1e-6), "should take the nearer box"


def test_teammate_boxes_are_ignored_entirely():
    """`detector.ignore` already drops them, but a teammate reaching the tracker must not silently
    become an enemy slot -- Solo Showdown has no teammates and a box labelled one is noise."""
    trk = EntityTracker()
    for i in range(4):
        res = trk.update([det_at(4.0, 4.0, label="teammate")], StubPlan(), StubOdo(), t=i * 0.25)
    assert all(e is None for e in res.enemies)
    assert res.hero is None
