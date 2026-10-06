"""`brawl_deployment.perception.localize`: finding and keeping a fix on a known map.
KNOWN_MAP_LOCALIZATION_PLAN.md step K2.

Each view is cut from a map label at the camera's true place, then damaged the way a classified
frame is:

  * cells outside the footprint or under the HUD are garbage, and the localizer must mask them
    itself from the real emulator plan;
  * a gassed band and the hero's box are garbage, and come with their masks;
  * the classifier's errors at its held-out BlueStacks F1 (.851): 15% of the non-floor cells
    read as floor, and as many floor cells beside them read as a neighbour's class, a third of
    them under 0.5 confidence;
  * a few walls read as floor, as if destroyed.

What these tests cannot check is the camera model. Views are cut where the same homography,
`HERO_ANCHOR_TILES` and clamp geometry put the camera, so an error in those constants would move
the views and the localizer together. The one camera constant still unmeasured is where the camera
stops at the map edges (P1 has lower bounds only), so the game's camera here stops at different
onsets from the ones the localizer is told. K4's recorded matches test the rest.
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pytest

import brawl_deployment.perception.localize as localize
from brawl_deployment.perception.known_map import KnownMap, read_chars
from brawl_deployment.perception.lattice import CRATE_HEIGHT_PX
from brawl_deployment.perception.localize import (
    FLOOR, SCORE_CLASS, MapLocalizer, _nominal, score_placements,
)
from brawl_sim.config import EnvConfig
from brawl_sim.constants import CHAR_TO_TILE, Tile
from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
from brawl_vision.config import EMULATOR_HUD_MASK_PATH
from brawl_vision.object_detection.detector import Detection
from brawl_vision.object_detection.project import CRATE_ANCHOR_FRAC
from brawl_vision.object_detection.projectile_detection.classes import CUBE_BOX
from brawl_vision.terrain.labeling import CLASS_INDEX
from brawl_vision.terrain.occupancy import _MIN_FOOTPRINT
from brawl_vision.terrain.odometry import OdometryResult

SIM_MAPS = Path(__file__).resolve().parent.parent / "brawl_sim" / "maps" / "csv"
SPAWNS = [(6, 5), (26, 5), (43, 6), (53, 15), (6, 24), (57, 38), (6, 47), (47, 53), (34, 54),
          (15, 55)]
# (west, east, north, south). The localizer is told the sim's; the game's camera in these tests
# stops 1 - 1.5 tiles away from that on every side.
TOLD = EnvConfig().camera_clamp_onset
GAME = (10.6, 14.0, 7.9, 6.5)
# K5: the game's camera as the live matches and the tuning recordings found it, which is up to 2.9
# tiles past the sim's limits (1.9 on the west). Here, 3 past on every side.
LIVE = tuple(v - 3.0 for v in TOLD)
# Map world = odometry + ORIGIN: odometry's frame starts wherever it starts.
ORIGIN = (-17.0, 11.0)
HZ = 12.0
OPEN = CLASS_INDEX[Tile.FLOOR]
WALL = CLASS_INDEX[Tile.WALL]


@lru_cache(maxsize=None)
def _plan():
    return build_rectify_plan(load_camera_model(), load_hud_mask(EMULATOR_HUD_MASK_PATH))


@lru_cache(maxsize=None)
def _dark_passage():
    return KnownMap.load("dark_passage")


def _sim_map(name):
    """A sim map as classes, spawns and crate spots read as floor, and its first spawn."""
    chars = read_chars(SIM_MAPS / f"{name}.csv")
    out = np.full(chars.shape, OPEN, np.int8)
    for char, tile in CHAR_TO_TILE.items():
        if tile in CLASS_INDEX:
            out[chars == char] = CLASS_INDEX[tile]
    row, col = np.argwhere(chars == "S")[0]
    return out, (int(col), int(row))


def _odo(position, status="ok", segment=0):
    return OdometryResult(delta_tiles=(0.0, 0.0), position_tiles=tuple(map(float, position)),
                          response=1.0, agreement_tiles=0.0, inlier_ratio=1.0, n_windows=11,
                          status=status, segment=segment)


class _Game:
    """One match: a hero walking on `truth` (classes, map tiles), a camera that follows it and
    stops at `onsets`, and odometry reporting where the camera is in its own frame, which is the
    map frame less `origin`."""

    def __init__(self, truth, spawn, heading=None, origin=ORIGIN, seed=0, damage=True,
                 off_map=WALL, misread=0.15, onsets=GAME):
        self.truth = truth
        self.onsets = onsets
        self.half = np.array(truth.shape[::-1]) // 2
        self.hero = np.array(spawn, np.float64) + 0.5
        self.origin = np.array(origin, np.float64)
        h = np.asarray(heading if heading is not None else self.half - self.hero, np.float64)
        self.heading = h / np.linalg.norm(h)
        self.segment = 0
        self.t = 0.0
        self.rng = np.random.default_rng(seed)
        self.damage = damage
        self.off_map = off_map
        self.misread = misread        # the share of non-floor cells that read as floor
        self.events = []

    def camera(self) -> np.ndarray:
        """The map-tile point the camera centres on: the hero, clamped."""
        west, east, north, south = self.onsets
        rows, cols = self.truth.shape
        return np.clip(self.hero, (west, north), (cols - east, rows - south))

    def true_world(self) -> np.ndarray:
        """Where a correct fix puts odometry's position: the camera's, in the map frame."""
        return self.camera() - self.half - _nominal(_plan())

    def odometry(self, status="ok"):
        return _odo(self.true_world() - self.origin, status, self.segment)

    def tick(self, loc, walk=3.0 / HZ, status="ok", detections=()):
        """One perception tick through `loc`, in the loop's order (`DeployLoop._perceive`)."""
        self.hero = np.clip(self.hero + walk * self.heading, 3.0, min(self.truth.shape) - 3.0)
        self.t += 1.0 / HZ
        odo = self.odometry(status)
        up = loc.update(list(detections), _plan(), odo)
        world = loc.world(odo)
        plan = _plan().registered(world.position_tiles)
        cells, confidence, zone, occluded = self.view(plan, world, odo)
        ob = loc.observe(self.t, cells, confidence, plan, world, zone, occluded)
        self.events += [e for e in (up.event, ob.event) if e]
        return ob

    def view(self, plan, world, odo):
        """(cells, confidence, zone, occluded) on `plan`, cut from the truth and damaged."""
        cols, rows = plan.size_tiles
        x0 = int(round(plan.origin_tile[0] + world.position_tiles[0]))
        y0 = int(round(plan.origin_tile[1] + world.position_tiles[1]))
        # How far the handed-out frame is off the map frame, to the nearest tile.
        off = np.asarray(odo.position_tiles) + self.origin - np.asarray(world.position_tiles)
        shift = np.rint(off).astype(int)
        mr, mc = np.meshgrid(y0 + self.half[1] + shift[1] + np.arange(rows),
                             x0 + self.half[0] + shift[0] + np.arange(cols), indexing="ij")

        def truth(dr=0, dc=0):
            r, c = mr + dr, mc + dc
            on = (r >= 0) & (r < self.truth.shape[0]) & (c >= 0) & (c < self.truth.shape[1])
            out = np.full((rows, cols), self.off_map, np.int8)
            out[on] = self.truth[r[on], c[on]]
            return out

        cells = truth()
        # The rest of `off` is a fraction of a tile, so each cell straddles two map tiles and reads
        # as the other one with that fraction's chance.
        for axis, f in enumerate(off - shift):
            if abs(f) > 1e-9:
                other = truth(dc=int(np.sign(f))) if axis == 0 else truth(dr=int(np.sign(f)))
                cells = np.where(self.rng.random((rows, cols)) < abs(f), other, cells)
        confidence = np.full((rows, cols), 0.95, np.float32)
        zone = np.zeros((rows, cols), bool)
        occluded = np.zeros((rows, cols), bool)
        if not self.damage:
            return cells, confidence, zone, occluded
        rng = self.rng
        ppt = plan.pixels_per_tile
        foot = plan.valid.reshape(rows, ppt, cols, ppt).mean(axis=(1, 3)) >= _MIN_FOOTPRINT

        # The classifier at its held-out BlueStacks F1 (.851): 15% of the non-floor cells read as
        # floor, and as many floor cells beside them read as a neighbour's class. A third of the
        # errors are under 0.5 confidence.
        padded = np.pad(cells, 1, mode="edge")
        nbrs = np.stack([padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2],
                         padded[1:-1, 2:]])
        solid = np.flatnonzero(foot & (cells != OPEN))
        beside = np.flatnonzero(foot & (cells == OPEN) & (nbrs != OPEN).any(axis=0))
        k = int(round(self.misread * len(solid)))
        misread = rng.choice(solid, size=k, replace=False)
        added = rng.choice(beside, size=min(k, len(beside)), replace=False)
        for f in added:
            r, c = divmod(int(f), cols)
            cells[r, c] = rng.choice([int(v) for v in nbrs[:, r, c] if v != OPEN])
        cells.flat[misread] = OPEN
        confidence.flat[np.concatenate([misread, added])] = rng.uniform(0.3, 0.9, k + len(added))
        # A few walls down, as if destroyed; the classifier is sure of them.
        walls = np.flatnonzero(foot & (cells == WALL))
        cells.flat[rng.choice(walls, size=min(3, len(walls)), replace=False)] = OPEN
        # Garbage wherever the localizer must not look: off the footprint, the gas, the hero.
        zone[:2] = True
        hero = np.floor(self.hero - self.half - shift).astype(int) - (x0, y0)
        occluded[max(hero[1] - 1, 0):max(hero[1] + 1, 0), max(hero[0], 0):max(hero[0] + 1, 0)] \
            = True
        garbage = ~foot | zone | occluded
        cells[garbage] = rng.integers(0, len(CLASS_INDEX), int(garbage.sum()))
        return cells, confidence, zone, occluded


def _fix(game, loc, ticks=6):
    """From the gate tick until the localizer commits; the ticks it took."""
    loc.reset(game.segment)
    for n in range(1, ticks + 1):
        if game.tick(loc).state == "fixed":
            return n
    raise AssertionError(f"no fix in {ticks} ticks: {game.events}")


def _walk(game, loc, seconds):
    for _ in range(int(round(seconds * HZ))):
        game.tick(loc)


def _crate(plan, tile):
    """A full-height crate box whose anchor projects to camera-relative `tile` through `plan`."""
    rect = plan.tile_to_rect(np.asarray(tile, np.float64)).reshape(1, 1, 2)
    ax, ay = cv2.perspectiveTransform(rect, plan.M_inv).reshape(2)
    a, b = CRATE_HEIGHT_PX
    # anchor = y1 - frac * h with h = a + b * y1; solve for y1.
    y1 = (ay + CRATE_ANCHOR_FRAC * a) / (1.0 - CRATE_ANCHOR_FRAC * b)
    h = a + b * y1
    return Detection(CUBE_BOX, 0.9, (ax - 60.0, y1 - h, ax + 60.0, y1))


def _moved(label, step):
    """`label` moved so that each map tile shows what the label holds `step` (x, y) tiles on, and
    floor where that is off the map."""
    out = np.full_like(label, OPEN)
    rows, cols = label.shape
    dx, dy = step
    out[max(-dy, 0):rows + min(-dy, 0), max(-dx, 0):cols + min(-dx, 0)] = \
        label[max(dy, 0):rows + min(dy, 0), max(dx, 0):cols + min(dx, 0)]
    return out


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------

def test_the_score_is_a_direct_count_with_floor_on_floor_left_out():
    rng = np.random.default_rng(3)
    label = rng.integers(0, 4, (12, 10))
    view = rng.integers(0, 4, (4, 5))
    scored = rng.random((4, 5)) < 0.8
    score, n = score_placements(view, scored, label)
    assert score.shape == n.shape == (12 + 4 - 1, 10 + 5 - 1)
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            agree = count = 0
            for r in range(4):
                for c in range(5):
                    lr, lc = i - 4 + 1 + r, j - 5 + 1 + c
                    if not (scored[r, c] and 0 <= lr < 12 and 0 <= lc < 10):
                        continue
                    if view[r, c] == FLOOR and label[lr, lc] == FLOOR:
                        continue
                    count += 1
                    agree += int(view[r, c] == label[lr, lc])
            assert (score[i, j], n[i, j]) == (2 * agree - count, count), (i, j)


def test_a_fence_scores_as_a_wall():
    assert SCORE_CLASS[CLASS_INDEX[Tile.FENCE]] == SCORE_CLASS[CLASS_INDEX[Tile.WALL]]


# ---------------------------------------------------------------------------
# the six K2 cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spawn", SPAWNS)
def test_every_spawn_is_fixed_right_within_three_ticks(spawn):
    game = _Game(_dark_passage().classes, spawn, seed=100 * spawn[0] + spawn[1])
    loc = MapLocalizer(_dark_passage(), TOLD)
    _fix(game, loc, ticks=3)
    assert loc.world(game.odometry()).position_tiles == pytest.approx(tuple(game.true_world()))


@pytest.mark.parametrize("spawn", SPAWNS)
def test_a_view_of_open_ground_never_commits(spawn):
    game = _Game(np.full((60, 60), OPEN, np.int8), spawn, damage=False, off_map=OPEN)
    loc = MapLocalizer(_dark_passage(), TOLD)
    loc.reset(0)
    states = {game.tick(loc).state for _ in range(int(7 * HZ))}
    assert "fixed" not in states, game.events


def test_a_whole_tile_drift_is_corrected_once_and_the_fix_kept():
    """Odometry loses a tile without a cut, so the fix is a tile off. The neighbour check moves it
    back once, and the guard never drops it."""
    game = _Game(_dark_passage().classes, (6, 24), heading=(1, 0))
    loc = MapLocalizer(_dark_passage(), TOLD)
    _fix(game, loc)
    _walk(game, loc, 2.0)
    epoch = loc.epoch
    game.origin += (1.0, 0.0)
    game.events.clear()
    _walk(game, loc, 5.0)
    assert game.events == ["fix slipped a tile: (+1, +0)"]
    assert loc.state == "fixed" and loc.epoch == epoch + 1
    assert loc.world(game.odometry()).position_tiles == pytest.approx(tuple(game.true_world()))


def test_a_cut_with_no_motion_keeps_the_fix_and_the_epoch():
    game = _Game(_dark_passage().classes, (6, 24), heading=(1, 0))
    loc = MapLocalizer(_dark_passage(), TOLD)
    _fix(game, loc)
    _walk(game, loc, 2.0)
    epoch = loc.epoch
    game.events.clear()
    game.segment += 1
    game.tick(loc, walk=0.0, status="lost")
    assert loc.state == "confirm" and loc.epoch == epoch
    assert loc.world(game.odometry()).position_tiles == pytest.approx(tuple(game.true_world()))
    _walk(game, loc, 2.0)
    assert game.events == ["odometry cut: confirming the carried fix", "cut confirmed, fix kept"]
    assert loc.state == "fixed" and loc.epoch == epoch


def test_a_three_tile_jump_inside_a_cut_is_found_locally_in_a_new_epoch():
    """The camera moves 3 tiles inside the cut, which odometry never sees: a cut leaves its
    position where it was."""
    game = _Game(_dark_passage().classes, (6, 24), heading=(1, 0))
    loc = MapLocalizer(_dark_passage(), TOLD)
    _fix(game, loc)
    _walk(game, loc, 2.0)
    epoch = loc.epoch
    game.events.clear()
    game.segment += 1
    before = game.odometry().position_tiles
    game.hero += (3.0, 0.0)
    game.origin += (3.0, 0.0)
    assert game.odometry().position_tiles == pytest.approx(before), "the camera was clamped"
    game.tick(loc, walk=0.0, status="lost")
    _walk(game, loc, 2.0)
    assert game.events == ["odometry cut: confirming the carried fix",
                           "cut moved the fix by (+3, +0)"]
    assert loc.state == "fixed" and loc.epoch == epoch + 1
    assert loc.world(game.odometry()).position_tiles == pytest.approx(tuple(game.true_world()))


def test_a_fix_lost_without_a_cut_is_dropped_and_found_again():
    """Odometry jumps 5 tiles and reports it as motion, so nothing within a tile of the fix
    matches: the guard drops it, and the whole-map search finds the right one. That search runs in
    the lost fix's frame (K4), and the leader it reports is the offset the commit then takes, not
    the step from that frame (K5)."""
    game = _Game(_dark_passage().classes, (6, 24), heading=(1, 0))
    loc = MapLocalizer(_dark_passage(), TOLD)
    _fix(game, loc)
    _walk(game, loc, 2.0)
    game.origin += (5.0, 0.0)
    game.events.clear()
    leaders = []
    for _ in range(int(8 * HZ)):
        out = game.tick(loc)
        if out.state == "search" and out.leader is not None:
            leaders.append(out.leader)
    assert game.events == ["fix dropped: agreement under 0.6 for 2 s",
                           "fixed on dark_passage at offset (-12.00, +11.00)"]
    assert loc.world(game.odometry()).position_tiles == pytest.approx(tuple(game.true_world()))
    assert leaders and leaders[-1] == pytest.approx(tuple(game.origin))


def test_views_from_another_map_end_in_the_wrong_map_log():
    truth, spawn = _sim_map("ghost_point")
    game = _Game(truth, spawn)
    loc = MapLocalizer(_dark_passage(), TOLD)
    loc.reset(0)
    states = [game.tick(loc).state for _ in range(int(6 * HZ))]
    assert "fixed" not in states, game.events
    assert game.events == ["map mismatch: expected dark_passage"]
    assert states[-1] == "mismatch"


# ---------------------------------------------------------------------------
# the agreement a commit needs
# ---------------------------------------------------------------------------

# Nearly twice K2's share of non-floor cells misread as floor. The right placement's running
# agreement at landing then sits at 0.62 - 0.64 over four seeds: over the guard's 0.6, under
# min_agreement's 0.65. A held-out match lost its fix in a stretch like that and never found it
# again (K4), and K5's first live landing read 0.62 - 0.65 for the whole of its 5 s.
POOR = 0.29


def test_a_poor_view_lands_inside_the_spawn_windows():
    """In the spawn windows a commit needs only the guard's agreement (K5), so a view this poor
    lands on its spawn. The premise: with the windows held to `min_agreement`, as they were before
    K5, the same view does not land in them."""
    game = _Game(_dark_passage().classes, (6, 24), heading=(1, 0), misread=POOR)
    loc = MapLocalizer(_dark_passage(), TOLD)
    _fix(game, loc, ticks=int(loc.search_seconds * HZ))
    assert game.events == ["fixed on dark_passage at offset (-17.00, +11.00)"]
    assert loc.world(game.odometry()).position_tiles == pytest.approx(tuple(game.true_world()))

    before = _Game(_dark_passage().classes, (6, 24), heading=(1, 0), misread=POOR)
    held = MapLocalizer(_dark_passage(), TOLD, guard_agreement=loc.min_agreement)
    held.reset(0)
    _walk(before, held, loc.search_seconds)
    assert before.events == [] and held.state == "search"


def test_a_poor_view_does_not_land_on_the_whole_map_however_far_it_leads():
    """Before the match's first fix, the whole-map search still needs `min_agreement`: every
    placement on the map is eligible there, and on K2's views of other maps a lower floor let more
    of them commit (K5). The hero starts off every spawn, so the windows hold no right placement,
    as when the loop joins mid-match. The search's leader is right, and the result says where it
    is."""
    game = _Game(_dark_passage().classes, (20, 40), heading=(1, 0), misread=POOR)
    loc = MapLocalizer(_dark_passage(), TOLD)
    loc.reset(0)
    _walk(game, loc, 8.0)
    out = game.tick(loc)
    assert game.events == [] and out.state == "search"
    # What held it off was the agreement, not the margin or the wrong-map verdict.
    assert loc.guard_agreement <= out.agreement < loc.min_agreement
    assert out.margin >= loc.whole_map_margin
    assert out.offset is None and out.leader == pytest.approx(tuple(game.origin))


def test_after_a_first_fix_a_poor_view_finds_it_again():
    """The fix is lost where views read poorly, and the search after it needs only the agreement
    that keeps a fix. On views this damaged check 3 may then step the fix by a fifth of a tile, so
    the end is checked to the tile."""
    game = _Game(_dark_passage().classes, (6, 24), heading=(1, 0))
    loc = MapLocalizer(_dark_passage(), TOLD)
    _fix(game, loc)
    _walk(game, loc, 2.0)
    game.misread = POOR
    game.origin += (5.0, 0.0)
    game.events.clear()
    leader = None
    for _ in range(int(8 * HZ)):
        out = game.tick(loc)
        if out.state == "search":
            leader = out.agreement
    assert game.events == ["fix dropped: agreement under 0.6 for 2 s",
                           "fixed on dark_passage at offset (-12.00, +11.00)"]
    assert leader < loc.min_agreement, "the view read well enough for min_agreement"
    err = np.subtract(loc.world(game.odometry()).position_tiles, game.true_world())
    assert loc.state == "fixed" and np.abs(err).max() < 0.5, err


def test_after_a_first_fix_a_poor_view_confirms_a_cut():
    """The same floor holds while a fix carried across an odometry cut is confirmed. Under
    `min_agreement` this cut stays unconfirmed (6 of 8 seeds, this one included), and the fix
    would drop 5 s after it. While it confirms, the leader reported is the carried fix, the offset
    the confirm keeps (K5)."""
    game = _Game(_dark_passage().classes, (6, 24), heading=(1, 0))
    loc = MapLocalizer(_dark_passage(), TOLD)
    _fix(game, loc)
    _walk(game, loc, 2.0)
    epoch = loc.epoch
    game.misread = POOR
    game.events.clear()
    game.segment += 1
    game.tick(loc, walk=0.0, status="lost")
    rates, leaders = [], []
    for _ in range(int(3 * HZ)):
        out = game.tick(loc)
        if out.state == "fixed" and out.agreement is not None:
            rates.append(out.agreement)
        if out.state == "confirm":
            leaders.append(out.leader)
    assert game.events == ["odometry cut: confirming the carried fix", "cut confirmed, fix kept"]
    assert loc.state == "fixed" and loc.epoch == epoch
    assert np.median(rates) < loc.min_agreement, "the view read well enough for min_agreement"
    assert leaders and all(v == pytest.approx(tuple(game.origin)) for v in leaders)


# ---------------------------------------------------------------------------
# check 3
# ---------------------------------------------------------------------------

def test_a_sub_tile_slip_is_read_off_the_terrain_and_corrected():
    """Odometry slips a fraction of a tile, so every cell straddles two map tiles and the fix's
    neighbours on the slip's side agree better than those on the other. K4 found crates cannot
    correct this on Dark Passage (they read 0.41 tile off the terrain), so check 3 reads the
    terrain. It stops once its estimate is inside `SUBTILE_TILES`, and like odometry's own drift
    the step is neither an epoch nor a log line."""
    game = _Game(_dark_passage().classes, (6, 24), heading=(1, 0))
    loc = MapLocalizer(_dark_passage(), TOLD)
    _fix(game, loc)
    _walk(game, loc, 2.0)
    epoch = loc.epoch
    game.origin += (0.4, -0.3)
    game.events.clear()
    _walk(game, loc, 4.0)
    assert game.events == [] and loc.state == "fixed" and loc.epoch == epoch
    err = np.subtract(loc.world(game.odometry()).position_tiles, game.true_world())
    assert np.abs(err).max() < 0.25, err


def test_a_fix_on_the_terrain_is_left_alone():
    """No slip: every estimate stays inside `SUBTILE_TILES`, so the fix does not move."""
    game = _Game(_dark_passage().classes, (6, 24), heading=(1, 0))
    loc = MapLocalizer(_dark_passage(), TOLD)
    _fix(game, loc)
    _walk(game, loc, 8.0)
    assert loc.world(game.odometry()).position_tiles == pytest.approx(tuple(game.true_world()))


def test_a_crate_sighting_is_telemetry_only():
    """A crate a fraction of a tile off its tile centre reports the residual but moves nothing."""
    game = _Game(_dark_passage().classes, (6, 24), heading=(1, 0))
    loc = MapLocalizer(_dark_passage(), TOLD)
    _fix(game, loc)
    epoch = loc.epoch
    game.events.clear()
    for _ in range(4):
        # A crate 0.3 east and 0.2 north of the centre of the tile 2 right and 1 up of the camera's.
        centre = np.floor(game.camera()) + 0.5 + (2.3, -1.2)
        at = centre - game.half - game.origin - np.asarray(game.odometry().position_tiles)
        out = game.tick(loc, walk=0.0, detections=[_crate(_plan(), at)])
        assert out.crate_resid == (pytest.approx((-0.3, 0.2), abs=0.02),)
    assert game.events == [] and loc.epoch == epoch


# ---------------------------------------------------------------------------
# the camera box (K5)
# ---------------------------------------------------------------------------

def test_a_placement_whose_camera_the_game_does_not_allow_never_commits(monkeypatch):
    """K5's second control match in miniature. There a right fix dropped, and the search then
    committed on a corner placement whose view hung mostly off the map, on the sliver left on it.
    Here the views after the drop show what the label holds 14 tiles east and 14 north of the
    camera, so their best placement puts the camera 12.8 tiles past the clamp limits. It never
    commits, and once the views are the map's again the right fix is found. The premise: without
    the camera box the same views commit there."""
    label = _dark_passage().classes

    def run():
        game = _Game(label, (53, 15), heading=(1, 0))
        loc = MapLocalizer(_dark_passage(), TOLD)
        _fix(game, loc)
        _walk(game, loc, 1.0)
        game.truth = _moved(label, (14, -14))
        game.events.clear()
        _walk(game, loc, 4.0)
        return game, loc

    game, loc = run()
    assert game.events == ["fix dropped: agreement under 0.6 for 2 s"]
    assert tuple(game.camera()) == (46.0, 15.5), "the camera this test puts 12.8 past"
    game.truth = label
    _walk(game, loc, 10.0)
    assert game.events[1:] == ["fixed on dark_passage at offset (-17.00, +11.00)"]
    assert loc.world(game.odometry()).position_tiles == pytest.approx(tuple(game.true_world()))

    monkeypatch.setattr(localize, "CAMERA_TILES", 100.0)
    before, _ = run()
    assert before.events == ["fix dropped: agreement under 0.6 for 2 s",
                             "fixed on dark_passage at offset (-3.00, -3.00)"]


def test_a_fix_that_puts_the_camera_off_the_map_drops_on_that_tick(monkeypatch):
    """Odometry jumps 40 tiles and reports it as motion, so the fix's view is off the map, with no
    cells for the guard to judge. Before K5 that fix was kept for the rest of the match (the
    premise, at the end). The camera it implies is far past the clamp limits, so it drops on the
    tick of the jump, and the whole-map search finds the right fix."""
    def run():
        game = _Game(_dark_passage().classes, (6, 24), heading=(1, 0))
        loc = MapLocalizer(_dark_passage(), TOLD)
        _fix(game, loc)
        _walk(game, loc, 2.0)
        game.origin += (40.0, 0.0)
        game.events.clear()
        game.tick(loc)
        return game, loc

    game, loc = run()
    past = TOLD[0] - (game.camera()[0] - 40.0)          # west of the west limit
    assert game.events == [f"fix dropped: it puts the camera {past:.1f} tiles past the clamp "
                           "limits"]
    _walk(game, loc, 6.0)
    assert game.events[1:] == ["fixed on dark_passage at offset (+23.00, +11.00)"]
    assert loc.world(game.odometry()).position_tiles == pytest.approx(tuple(game.true_world()))

    monkeypatch.setattr(localize, "CAMERA_TILES", 100.0)
    before, held = run()
    _walk(before, held, 6.0)
    assert before.events == [] and held.state == "fixed"


def test_a_camera_three_tiles_past_the_sims_limits_keeps_its_fix_along_the_edges(monkeypatch):
    """The game's camera goes further than the sim's (K5). Here it goes 3 tiles past the sim's
    limits on every side, 2.5 on the south, where the hero stops: landing in the north-west corner
    and walking the north, east and south edges, the fix is found and never moves or drops.

    Off the map reads as floor here. Read as wall, a camera this far past pulls the fix a tile
    inward, where the map's own border walls agree with it. The live matches and the tuning
    recordings do not show that: hundreds of their fixed ticks had the camera over a tile past,
    with no whole-tile move. The premise: at the 2 tiles first proposed, this landing never
    commits."""
    def start():
        game = _Game(_dark_passage().classes, (6, 5), heading=(1, 0), off_map=OPEN,
                     onsets=LIVE)
        loc = MapLocalizer(_dark_passage(), TOLD)
        return game, loc

    game, loc = start()
    _fix(game, loc, ticks=int(loc.search_seconds * HZ))
    seen = [game.camera()]
    for heading in ((1, 0), (0, 1), (-1, 0)):
        game.heading = np.array(heading, np.float64)
        for _ in range(int(18.0 * HZ)):
            game.tick(loc)
            seen.append(game.camera())
    assert game.events == ["fixed on dark_passage at offset (-17.00, +11.00)"]
    err = np.subtract(loc.world(game.odometry()).position_tiles, game.true_world())
    assert loc.state == "fixed" and np.abs(err).max() < 0.5, err
    west, east, north, south = TOLD
    seen = np.array(seen)
    past = (west - seen[:, 0].min(), seen[:, 0].max() - (60 - east),
            north - seen[:, 1].min(), seen[:, 1].max() - (60 - south))
    assert past == pytest.approx((3.0, 3.0, 3.0, 2.5)), "the camera went that far"

    monkeypatch.setattr(localize, "CAMERA_TILES", 2.0)
    game, held = start()
    held.reset(0)
    _walk(game, held, held.search_seconds)
    assert game.events == [] and held.state == "search"
