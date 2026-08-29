"""Occupancy accumulation, voting and locking. See Terrain_Perception_Build_Plan.md Phase I.

Every test here drives the map with synthetic classified grids and synthetic odometry, because
what needs checking is the accumulation logic -- the gates, the rounding, the locking -- and
feeding it a real classifier would make a failure ambiguous between this stage and that one.
"""
import numpy as np
import pytest

from brawl_sim.constants import TILE_TO_CHAR, Tile
from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
from brawl_vision.config import VisionConfig
from brawl_vision.terrain.labeling import CLASS_INDEX, CLASSES
from brawl_vision.terrain.occupancy import (
    UNKNOWN, OccupancyMap, compare_to_truth,
)
from brawl_vision.terrain.odometry import OdometryResult


@pytest.fixture(scope="module")
def plan():
    return build_rectify_plan(load_camera_model(), load_hud_mask())


def _odo(pos=(0.0, 0.0), status="ok", segment=0):
    return OdometryResult(delta_tiles=(0.0, 0.0), position_tiles=pos, response=1.0,
                          agreement_tiles=0.0, inlier_ratio=1.0, n_windows=11,
                          status=status, segment=segment)


def _cells(plan, klass=Tile.FLOOR):
    cols, rows = plan.size_tiles
    return np.full((rows, cols), CLASS_INDEX[klass], np.int8)


def _map(cfg=None):
    return OccupancyMap.from_config(cfg)


# ---------------------------------------------------------------------------
# accumulation and locking
# ---------------------------------------------------------------------------

def test_a_fresh_map_is_entirely_unknown(plan):
    m = _map()
    assert (m.best() == UNKNOWN).all()
    assert not m.observed.any() and not m.locked.any()
    assert m.observed_bounds() is None


def test_votes_accumulate_and_then_lock(plan):
    cfg = VisionConfig()
    m = _map(cfg)
    cells = _cells(plan, Tile.WALL)
    for i in range(cfg.occupancy_min_votes):
        r = m.update(cells, _odo(), plan, cfg=cfg)
    assert r.locked_now > 0, "nothing locked after min_votes agreeing observations"
    assert m.locked.sum() == r.locked_now, "locked cells and the reported count disagree"
    assert (m.votes.sum(axis=2)[m.locked] >= cfg.occupancy_min_votes).all()
    best = m.best()
    assert (best[m.observed] == CLASS_INDEX[Tile.WALL]).all()


def test_a_locked_cell_stops_taking_votes(plan):
    """Locking is the efficiency win as much as the robustness one: a locked cell must cost
    nothing, and must not be reconsidered for the rest of the match."""
    cfg = VisionConfig()
    m = _map(cfg)
    for _ in range(cfg.occupancy_min_votes + 1):
        m.update(_cells(plan, Tile.WALL), _odo(), plan, cfg=cfg)
    before = m.votes.copy()
    r = m.update(_cells(plan, Tile.BUSH), _odo(), plan, cfg=cfg)
    assert r.voted == 0 and r.skipped_locked > 0
    assert (m.votes == before).all(), "a locked cell was updated"
    assert (m.best()[m.observed] == CLASS_INDEX[Tile.WALL]).all()


def test_a_single_bad_frame_is_outvoted(plan):
    """The robustness half: a player standing on a tile, or one blurred frame, must not become the
    map. This is why votes exist rather than last-write-wins."""
    cfg = VisionConfig(occupancy_min_votes=5, occupancy_lock_ratio=0.8)
    m = _map(cfg)
    m.update(_cells(plan, Tile.BUSH), _odo(), plan, cfg=cfg)      # the bad frame, first
    for _ in range(3):
        m.update(_cells(plan, Tile.WALL), _odo(), plan, cfg=cfg)
    assert (m.best()[m.observed] == CLASS_INDEX[Tile.WALL]).all()


def test_a_disputed_cell_does_not_lock(plan):
    """min_votes alone is not enough -- a cell can be observed many times and still be a coin
    flip. lock_ratio is what stops that becoming a permanent answer."""
    cfg = VisionConfig(occupancy_min_votes=4, occupancy_lock_ratio=0.8)
    m = _map(cfg)
    for i in range(8):
        m.update(_cells(plan, Tile.WALL if i % 2 else Tile.BUSH), _odo(), plan, cfg=cfg)
    assert m.observed.any()
    assert not m.locked.any(), "a 50/50 cell locked despite the ratio gate"


# ---------------------------------------------------------------------------
# the four gates
# ---------------------------------------------------------------------------

def test_a_lost_frame_deposits_nothing(plan):
    m = _map()
    r = m.update(_cells(plan), _odo(status="lost"), plan)
    assert r.voted == 0 and not m.observed.any()


def test_an_uncertain_frame_deposits_nothing_but_keeps_the_map(plan):
    """Phase F separates these two precisely so this stage can keep tracking while declining to
    treat the frame as evidence. Conflating them would either poison the map or throw it away."""
    cfg = VisionConfig()
    m = _map(cfg)
    m.update(_cells(plan, Tile.WALL), _odo(), plan, cfg=cfg)
    seen = m.observed.sum()
    r = m.update(_cells(plan, Tile.BUSH), _odo(status="uncertain"), plan, cfg=cfg)
    assert r.voted == 0 and r.status == "uncertain"
    assert m.observed.sum() == seen, "an uncertain frame changed the map"
    assert m.segment == 0, "an uncertain frame was treated as a cut"


def test_a_new_segment_resets_the_map(plan):
    """There is no defined offset between segments until loop closure exists. Carrying the old
    grid across would silently overlay two unrelated maps, which is worse than losing one."""
    cfg = VisionConfig()
    m = _map(cfg)
    for _ in range(3):
        m.update(_cells(plan, Tile.WALL), _odo(), plan, cfg=cfg)
    assert m.observed.any()
    r = m.update(_cells(plan), _odo(segment=1), plan, cfg=cfg)
    assert r.status == "reset" and m.segment == 1 and m.segments_seen == 2
    assert not m.observed.any() and not m.locked.any()


def test_gassed_cells_are_never_voted_on(plan):
    """Gas tints the terrain beneath it, so a gassed cell's colours were shifted by a full-screen
    effect. Voting on it locks a misclassification permanently -- this is why Phase G is a
    prerequisite for this phase rather than an independent side quest."""
    cfg = VisionConfig()
    cols, rows = plan.size_tiles
    zone = np.zeros((rows, cols), bool)
    zone[:, : cols // 2] = True
    m = _map(cfg)
    m.update(_cells(plan, Tile.WALL), _odo(), plan, zone=zone, cfg=cfg)

    # Where the patch landed: rect cell column c sits at grid column col0 + c.
    col0 = int(round(plan.origin_tile[0])) - m.origin[0]
    assert m.observed[:, col0:col0 + cols // 2].sum() == 0, "voted on gassed cells"
    assert m.observed[:, col0 + cols // 2:col0 + cols].sum() > 0, "the ungassed half was skipped"

    clean = _map(cfg)
    clean.update(_cells(plan, Tile.WALL), _odo(), plan, cfg=cfg)
    assert m.observed.sum() < clean.observed.sum()


def test_occluded_cells_are_never_voted_on(plan):
    """The loot-box gate. No detector exists yet, so this checks the wiring is real rather than a
    parameter that is accepted and ignored."""
    cols, rows = plan.size_tiles
    occluded = np.ones((rows, cols), bool)
    m = _map()
    r = m.update(_cells(plan), _odo(), plan, occluded=occluded)
    assert r.voted == 0 and not m.observed.any()


def test_cells_outside_the_true_footprint_are_skipped(plan):
    """The patch is a rectangle; the world in it is a trapezoid, because the camera is perspective
    (Phase C). Cells straddling that edge are interpolated from border fill and classify as
    garbage, so the bounding rectangle is the wrong footprint to deposit."""
    cols, rows = plan.size_tiles
    m = _map()
    r = m.update(_cells(plan), _odo(), plan)
    assert r.voted < rows * cols, "the whole bounding rectangle was deposited"
    assert r.voted > 0.4 * rows * cols, "the footprint gate rejected almost everything"


# ---------------------------------------------------------------------------
# position, and the rounding that must not compound
# ---------------------------------------------------------------------------

def test_sub_tile_motion_accumulates_instead_of_being_rounded_away(plan):
    """The specific mechanism the plan calls out. Ten steps of 0.3 tiles is three tiles of travel;
    a stage that rounded each increment to zero would never move, and the map would pile every
    frame on top of the first one while odometry insisted the camera had moved."""
    cfg = VisionConfig()
    m = _map(cfg)
    for i in range(11):
        m.update(_cells(plan, Tile.WALL), _odo(pos=(0.0, 0.3 * i)), plan, cfg=cfg)
    r0, r1, _, _ = m.observed_bounds()

    baseline = _map(cfg)
    baseline.update(_cells(plan, Tile.WALL), _odo(pos=(0.0, 0.0)), plan, cfg=cfg)
    b0, b1, _, _ = baseline.observed_bounds()

    # `observed_bounds` is the union over time, so the top edge stays where the first frame put it
    # and the bottom edge advances by exactly the distance travelled.
    assert r0 == b0, "the first frame's footprint moved retroactively"
    assert r1 - b1 == 3, (
        f"10 steps of 0.3 tiles is 3 tiles of travel, but the map grew by {r1 - b1}. "
        "0 would mean each increment was rounded away instead of accumulating."
    )


def test_position_rounds_to_the_nearest_tile(plan):
    cfg = VisionConfig()
    near = _map(cfg)
    near.update(_cells(plan), _odo(pos=(0.4, 0.0)), plan, cfg=cfg)
    zero = _map(cfg)
    zero.update(_cells(plan), _odo(pos=(0.0, 0.0)), plan, cfg=cfg)
    assert near.observed_bounds() == zero.observed_bounds()
    far = _map(cfg)
    far.update(_cells(plan), _odo(pos=(0.6, 0.0)), plan, cfg=cfg)
    assert far.observed_bounds()[2] == zero.observed_bounds()[2] + 1


def test_walking_off_the_grid_is_counted_not_crashed(plan):
    m = OccupancyMap(height=24, width=24)
    r = m.update(_cells(plan), _odo(pos=(500.0, 500.0)), plan)
    assert r.out_of_bounds > 0 and r.voted == 0


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def test_unknown_is_not_a_tile_member_and_never_defaults_to_floor(plan):
    """A consumer that treats unexplored ground as walkable will walk into walls. UNKNOWN has to
    survive all the way out of this module as its own state."""
    m = _map()
    m.update(_cells(plan, Tile.WALL), _odo(), plan)
    best = m.best()
    assert (best[~m.observed] == UNKNOWN).all()
    assert UNKNOWN not in [int(t) for t in CLASSES]
    assert (m.to_chars()[~m.observed] == "?").all()


def test_chars_use_the_map_csv_legend(plan):
    m = _map()
    m.update(_cells(plan, Tile.WATER), _odo(), plan)
    assert (m.to_chars()[m.observed] == TILE_TO_CHAR[Tile.WATER]).all()


def test_confidence_reflects_agreement(plan):
    cfg = VisionConfig(occupancy_min_votes=99, occupancy_lock_ratio=0.99)
    m = _map(cfg)
    for i in range(4):
        m.update(_cells(plan, Tile.WALL if i else Tile.BUSH), _odo(), plan, cfg=cfg)
    conf = m.confidence()[m.observed]
    assert np.allclose(conf, 0.75)


# ---------------------------------------------------------------------------
# comparing against ground truth
# ---------------------------------------------------------------------------

def _truth(seed=0, h=10, w=12):
    rng = np.random.default_rng(seed)
    return np.array([list("".join(rng.choice(list(".#b~f"), w))) for _ in range(h)])


def test_an_identical_grid_reports_no_offset_and_perfect_accuracy():
    """A naive version using np.roll let a large shift wrap around to the identity and score 1.0,
    so identical grids were reported as displaced by six tiles. Found by testing exactly this."""
    t = _truth()
    r = compare_to_truth(t, t)
    assert r["best_offset"] == (0, 0)
    assert r["accuracy_at_best_offset"] == 1.0 and r["position_error_tiles"] == 0.0


def test_a_displaced_grid_reads_as_position_error_not_classifier_error():
    """The distinction the acceptance criterion demands: position and classification errors have
    different causes -- Phase F versus Phase H -- so a drift bug must not read as a classifier
    bug."""
    t = _truth(1)
    shifted = np.full_like(t, "?")
    shifted[:, 2:] = t[:, :-2]
    r = compare_to_truth(shifted, t)
    assert r["best_offset"] == (0, 2)
    assert r["accuracy_at_best_offset"] == 1.0
    assert r["accuracy_at_zero_offset"] < 0.5


def test_misclassified_cells_read_as_classifier_error_not_drift():
    t = _truth(2)
    wrong = t.copy()
    wrong[0, :4] = "b"
    wrong[5, :4] = "b"
    r = compare_to_truth(wrong, t)
    assert r["best_offset"] == (0, 0)
    assert 0.8 < r["accuracy_at_best_offset"] < 1.0
    assert r["accuracy_at_zero_offset"] == r["accuracy_at_best_offset"]


def test_unknown_cells_are_ignored_on_both_sides():
    t = _truth(3)
    partial = t.copy()
    partial[:5] = "?"
    r = compare_to_truth(partial, t)
    assert r["accuracy_at_best_offset"] == 1.0
    assert r["cells_compared"] == int((partial != "?").sum())


def test_a_sliver_of_overlap_cannot_win_the_alignment():
    """Without a floor on the compared count, a big offset that leaves a handful of lucky cells
    overlapping scores 1.0 and hides real drift behind a nonsense number."""
    t = _truth(4, h=10, w=12)
    r = compare_to_truth(t, t, max_shift=8, min_cells=12)
    assert r["best_offset"] == (0, 0)


# ---------------------------------------------------------------------------
# the whole chain
# ---------------------------------------------------------------------------

def test_a_walk_reconstructs_a_known_map(plan):
    """End to end with everything synthetic but the geometry: a camera walks across a known world,
    each frame is classified perfectly, and the accumulated grid must reproduce the world at the
    right place. Any sign error in the deposit offset shows up here as a mirrored map."""
    cfg = VisionConfig()
    cols, rows = plan.size_tiles
    world = _truth(7, h=60, w=60)
    lookup = {TILE_TO_CHAR[t]: CLASS_INDEX[t] for t in CLASSES}
    m = _map(cfg)
    ox, oy = m.origin
    for step in range(12):
        px, py = 0.5 * step, 0.0
        cells = np.zeros((rows, cols), np.int8)
        for r in range(rows):
            for c in range(cols):
                wx = int(round(plan.origin_tile[0] + px)) + c + 20
                wy = int(round(plan.origin_tile[1] + py)) + r + 20
                cells[r, c] = lookup[world[wy % 60, wx % 60]]
        m.update(cells, _odo(pos=(px, py)), plan, cfg=cfg)
    assert m.observed.any()
    got = m.to_chars()
    r0, r1, c0, c1 = m.observed_bounds()
    # Rebuild what the world says those grid cells should hold, and compare.
    want = np.full_like(got, "?")
    for r in range(r0, r1):
        for c in range(c0, c1):
            want[r, c] = world[(r + oy + 20) % 60, (c + ox + 20) % 60]
    res = compare_to_truth(got[r0:r1, c0:c1], want[r0:r1, c0:c1])
    assert res["best_offset"] == (0, 0), f"deposit is displaced: {res}"
    assert res["accuracy_at_best_offset"] > 0.99, res
