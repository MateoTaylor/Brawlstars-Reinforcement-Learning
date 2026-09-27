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


def test_a_locked_cell_keeps_voting_and_later_views_can_overturn_it(plan):
    """Locks used to be forever, and that froze the map's worst errors: a cell's first votes come
    from nearly one viewpoint at one sub-tile phase, so when they are wrong they are wrong
    together. Five wrong votes here are that early mistake; the later views must win."""
    cfg = VisionConfig(occupancy_min_votes=5, occupancy_lock_ratio=0.8)
    m = _map(cfg)
    for _ in range(5):
        m.update(_cells(plan, Tile.BUSH), _odo(), plan, cfg=cfg)
    seen = m.observed
    assert m.locked[seen].all()

    for _ in range(4):
        r = m.update(_cells(plan, Tile.WALL), _odo(), plan, cfg=cfg)
        assert r.voted == seen.sum(), "a locked cell refused a vote"
    # 5 BUSH against 4 WALL: under lock_ratio, but a lock holds while its class is the majority.
    assert m.locked[seen].all() and (m.best()[seen] == CLASS_INDEX[Tile.BUSH]).all()

    released = 0
    for _ in range(2):
        released += m.update(_cells(plan, Tile.WALL), _odo(), plan, cfg=cfg).released_now
    assert (m.best()[seen] == CLASS_INDEX[Tile.WALL]).all(), "later views never overturned the lock"
    assert released == seen.sum() and not m.locked.any()


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
# the one-wide-gap rule
# ---------------------------------------------------------------------------

def _interior(plan, m):
    """(row, col) of a cell the footprint gate admits together with its four neighbours, in the
    frame's own indices: where a pinch can be laid without touching a gate."""
    allow = m._footprint(plan)
    rows, cols = allow.shape
    for r in range(2, rows - 2):
        for c in range(2, cols - 2):
            if allow[r - 1:r + 2, c - 1:c + 2].all():
                return r, c
    raise AssertionError("no cell of the plan's footprint has an admitted 3x3 neighbourhood")


def _pinched(plan, m, klass=Tile.WALL):
    """All FLOOR, with `klass` either side of one interior cell: a frame claiming a passage one
    tile wide. Returns the grid and the (row, col) of the pinched cell."""
    r, c = _interior(plan, m)
    cells = _cells(plan)
    cells[r, c - 1] = cells[r, c + 1] = CLASS_INDEX[klass]
    return cells, (r, c)


def _map_cell(m, plan, rc):
    """Map indices of frame cell `rc` deposited at the origin."""
    ox, oy = m.origin
    return rc[0] + int(round(plan.origin_tile[1])) - oy, rc[1] + int(round(plan.origin_tile[0])) - ox


def test_a_frame_claiming_a_one_wide_gap_votes_at_the_rule_weight(plan):
    """The rule's whole mechanism: the pinched cell and both flanks get a fraction of a vote,
    every other cell a whole one, and the frame reports how many cells it touched."""
    cfg = VisionConfig(occupancy_gap_rule_weight=0.25)
    m = _map(cfg)
    cells, (r, c) = _pinched(plan, m)
    res = m.update(cells, _odo(), plan, cfg=cfg)
    assert res.gap_weighted == 3
    mr, mc = _map_cell(m, plan, (r, c))
    total = m.votes.sum(axis=2)
    assert total[mr, mc] == pytest.approx(0.25)
    assert total[mr, mc - 1] == pytest.approx(0.25) and total[mr, mc + 1] == pytest.approx(0.25)
    assert total[mr - 1, mc] == pytest.approx(1.0) and total[mr, mc + 2] == pytest.approx(1.0)
    # Was 1.0 until 2026-09-26, asserted as "a fractional vote is still 100% agreed". It is the
    # agreement of one view and that much is true, but reporting it as full confidence threw the
    # rule's doubt away: the weight divided straight back out of the share. `confidence` now
    # divides by the number of views instead, so the fraction survives into the answer.
    assert m.confidence()[mr, mc] == pytest.approx(0.25), "the rule's doubt must reach confidence"


def test_views_making_no_impossible_claim_decide_a_disputed_cell(plan):
    """A two-wide corridor whose far wall the classifier over-extends by a row from most views:
    the majority of frames say WALL at the corridor's second cell -- and claim a one-wide gap by
    doing so. Under the rule, the minority of views that see the corridor whole win the cell."""
    m = _map()
    cells_pinched, (r, c) = _pinched(plan, m)          # WALL at c-1 and c+1, FLOOR at c
    cells_open = cells_pinched.copy()
    cells_open[r, c + 1] = CLASS_INDEX[Tile.FLOOR]     # c and c+1 both floor: a two-wide gap
    mr, mc = _map_cell(m, plan, (r, c))

    off = VisionConfig(occupancy_gap_rule_weight=1.0)
    on = VisionConfig(occupancy_gap_rule_weight=0.1)
    for cfg in (off, on):
        m = _map(cfg)
        for _ in range(5):
            m.update(cells_pinched, _odo(), plan, cfg=cfg)
        for _ in range(2):
            m.update(cells_open, _odo(), plan, cfg=cfg)
        want = Tile.WALL if cfg is off else Tile.FLOOR
        assert m.best()[mr, mc + 1] == CLASS_INDEX[want], (
            f"with the rule {'off' if cfg is off else 'on'} the disputed flank should be {want.name}")
        assert m.best()[mr, mc] == CLASS_INDEX[Tile.FLOOR], "the pinched cell was floor in every view"


def test_the_rule_never_flips_a_cell_every_view_agrees_on(plan):
    """Symmetry: all three cells of a pinch are scaled alike, so unanimous views keep their
    argmax at any weight -- and the cells are still observed, never left UNKNOWN (which the
    deploy grid would read as FLOOR)."""
    cells, (r, c) = _pinched(plan, _map())
    maps = {}
    for w in (1.0, 0.1, 0.01):
        cfg = VisionConfig(occupancy_gap_rule_weight=w)
        maps[w] = _map(cfg)
        for _ in range(4):
            maps[w].update(cells, _odo(), plan, cfg=cfg)
    for w in (0.1, 0.01):
        assert (maps[w].best() == maps[1.0].best()).all()
        assert (maps[w].observed == maps[1.0].observed).all()
    mr, mc = _map_cell(maps[1.0], plan, (r, c))
    assert maps[0.01].best()[mr, mc - 1] == CLASS_INDEX[Tile.WALL]
    assert maps[0.01].best()[mr, mc] == CLASS_INDEX[Tile.FLOOR]


def test_a_flank_the_frame_may_not_vote_on_does_not_pinch(plan):
    """The rule's notion of known is the deposit's: an occluded blocker (a crate, a brawler) is
    not evidence of a passage, so the cells beside it vote in full."""
    cfg = VisionConfig(occupancy_gap_rule_weight=0.1)
    m = _map(cfg)
    cells, (r, c) = _pinched(plan, m)
    occluded = np.zeros(cells.shape, bool)
    occluded[r, c + 1] = True
    res = m.update(cells, _odo(), plan, occluded=occluded, cfg=cfg)
    assert res.gap_weighted == 0
    mr, mc = _map_cell(m, plan, (r, c))
    assert m.votes.sum(axis=2)[mr, mc] == pytest.approx(1.0)
    assert m.votes.sum(axis=2)[mr, mc + 1] == 0, "the occluded flank itself never voted"


# ---------------------------------------------------------------------------
# the rule's doubt reaching confidence, and the floor that acts on it
# ---------------------------------------------------------------------------

def test_confidence_is_a_share_of_views_not_of_vote_weight(plan):
    """The defect this fixed, stated as the numbers that used to be equal. Thirty views all
    claiming the same one-wide passage leave a tenth of the evidence a clean cell has, and the
    old share-of-mass reported both as 1.000 because the weight cancelled out of the ratio."""
    cfg = VisionConfig(occupancy_gap_rule_weight=0.1, occupancy_min_votes=99)
    m = _map(cfg)
    cells, (r, c) = _pinched(plan, m)
    for _ in range(30):
        m.update(cells, _odo(), plan, cfg=cfg)
    mr, mc = _map_cell(m, plan, (r, c))
    assert m.views[mr, mc] == 30, "every one of those frames looked at the cell"
    assert m.votes.sum(axis=2)[mr, mc] == pytest.approx(3.0), "and left a tenth of the evidence"
    assert m.confidence()[mr, mc] == pytest.approx(0.1)
    # The flanks are weighted alike, and an untouched cell two columns over is not.
    assert m.confidence()[mr, mc + 1] == pytest.approx(0.1)
    assert m.confidence()[mr, mc + 2] == pytest.approx(1.0)


def test_confidence_is_unchanged_wherever_the_gap_rule_never_fired(plan):
    """The property that makes the change safe to ship: off the rule's cells the summed weight IS
    the view count, so share-of-views and the old share-of-mass are the same expression. Asserted
    against the old formula directly rather than against remembered numbers."""
    cfg = VisionConfig(occupancy_min_votes=99, occupancy_lock_ratio=0.99)
    m = _map(cfg)
    for i in range(7):
        m.update(_cells(plan, Tile.WALL if i % 3 else Tile.BUSH), _odo(), plan, cfg=cfg)
    assert m.votes.sum(axis=2).max() > 0, "the deposit has to have happened"
    old = np.where(m.views > 0, m.votes.max(axis=2) / np.maximum(m.votes.sum(axis=2), 1e-9), 0.0)
    assert np.allclose(m.confidence(), old)


def test_a_cell_every_view_calls_an_impossible_passage_never_locks(plan):
    """Locking moved onto the same share. A pinched cell used to reach share 1.0 and lock the
    moment its weighted mass cleared `min_votes`; now the share itself never clears the ratio, so
    the map declines to call impossible geometry settled however long it looks at it."""
    cfg = VisionConfig(occupancy_gap_rule_weight=0.1, occupancy_min_votes=5,
                       occupancy_lock_ratio=0.8)
    m = _map(cfg)
    cells, (r, c) = _pinched(plan, m)
    for _ in range(200):
        m.update(cells, _odo(), plan, cfg=cfg)
    mr, mc = _map_cell(m, plan, (r, c))
    assert m.votes.sum(axis=2)[mr, mc] > cfg.occupancy_min_votes, "mass alone would have locked it"
    assert not m.locked[mr, mc]
    assert m.locked[mr, mc + 2], "an ordinary cell beside it still locks"


def test_the_confidence_floor_blanks_a_cell_to_unknown(plan):
    """`best` is where the floor lives, so the deploy grid, the loop and the render all get it."""
    cfg = VisionConfig(occupancy_gap_rule_weight=0.1, occupancy_min_votes=99)
    m = OccupancyMap.from_config(cfg)
    cells, (r, c) = _pinched(plan, m)
    for _ in range(30):
        m.update(cells, _odo(), plan, cfg=cfg)
    mr, mc = _map_cell(m, plan, (r, c))
    assert m.best()[mr, mc] == CLASS_INDEX[Tile.FLOOR], "floor 0.0 leaves the argmax alone"

    lifted = OccupancyMap.from_config(VisionConfig(occupancy_gap_rule_weight=0.1,
                                                   occupancy_min_votes=99,
                                                   occupancy_confidence_floor=0.5))
    for _ in range(30):
        lifted.update(cells, _odo(), plan, cfg=cfg)
    assert lifted.best()[mr, mc] == UNKNOWN
    assert lifted.best()[mr, mc + 1] == UNKNOWN, "the flanking walls blank too -- see the config"
    assert lifted.best()[mr, mc + 2] == CLASS_INDEX[Tile.FLOOR]


def test_the_floor_can_only_take_a_class_away_never_invent_one(plan):
    """Unobserved cells read confidence 0, which is under every legal floor. They must stay
    UNKNOWN by the same answer rather than be swept into a class by the comparison."""
    cfg = VisionConfig(occupancy_confidence_floor=0.7)
    m = OccupancyMap.from_config(cfg)
    assert (m.best() == UNKNOWN).all(), "a fresh map is unknown, floor or no floor"
    m.update(_cells(plan), _odo(), plan, cfg=cfg)
    blanked = m.best() == UNKNOWN
    assert blanked[~m.observed].all(), "never-observed cells are still unknown"


def test_the_floor_blanks_what_best_claims_and_not_what_the_map_observed(plan):
    """`observed` must stay clear of the floor, because a coverage curve built on `best` is no
    longer an odometry signal once a floor exists: a cell can return to UNKNOWN with the world
    frame perfectly still. `evaluate.RenderReport.unknown_curve` counts `~observed` for this
    reason, and its monotonicity check is only about odometry while that holds."""
    cfg = VisionConfig(occupancy_gap_rule_weight=0.1, occupancy_min_votes=99)
    floored = OccupancyMap.from_config(VisionConfig(occupancy_gap_rule_weight=0.1,
                                                    occupancy_min_votes=99,
                                                    occupancy_confidence_floor=0.5))
    plain = OccupancyMap.from_config(cfg)
    cells, _ = _pinched(plan, plain)
    for _ in range(30):
        plain.update(cells, _odo(), plan, cfg=cfg)
        floored.update(cells, _odo(), plan, cfg=cfg)
    assert (floored.observed == plain.observed).all(), "the floor is not allowed to unobserve"
    assert (floored.best() == UNKNOWN).sum() > (plain.best() == UNKNOWN).sum(), \
        "and it has to actually blank something, or this proves nothing"


def test_a_new_segment_resets_the_views_with_the_votes(plan):
    """A stale view count would make the new segment's first cells look long-observed and
    under-confident at once -- the votes are gone but the denominator is not."""
    cfg = VisionConfig()
    m = _map(cfg)
    m.update(_cells(plan), _odo(), plan, cfg=cfg)
    assert m.views.max() > 0
    m.update(_cells(plan), _odo(segment=1), plan, cfg=cfg)
    assert m.views.max() == 0, "reset drops the denominator too"


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


@pytest.mark.parametrize("position", [(0.3, -0.2), (7.45, 3.55), (-2.5, 0.5)])
def test_a_registered_deposit_covers_exactly_the_map_cell_it_votes_for(plan, position):
    """Rounding keeps the map from shearing, but through the fixed plan each frame's cell still
    covers a world square up to half a tile off the map cell it votes for. Through
    `plan.registered(position)` the two are the same square: the cell's corner, carried through
    the plan's own coordinates and the camera position, is the map cell's corner exactly."""
    cfg = VisionConfig()
    cols, rows = plan.size_tiles
    r, c = rows // 2, cols // 2

    def corner_error(p):
        cells = _cells(p)
        cells[r, c] = CLASS_INDEX[Tile.WALL]
        m = _map(cfg)
        m.update(cells, _odo(pos=position), p, cfg=cfg)
        (i,), (j,) = np.nonzero(m.best() == CLASS_INDEX[Tile.WALL])
        seen = p.rect_to_tile(np.array([[c, r]], np.float64) * p.pixels_per_tile)[0] + position
        return np.abs(seen - (j + m.origin[0], i + m.origin[1])).max()

    at = np.asarray(plan.origin_tile, np.float64) + position
    assert corner_error(plan) == pytest.approx(np.abs(at - np.round(at)).max())
    assert corner_error(plan.registered(position)) < 1e-9


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
