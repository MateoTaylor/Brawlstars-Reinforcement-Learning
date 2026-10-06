"""Where the camera is on a known map. KNOWN_MAP_LOCALIZATION_PLAN.md sections 6 and 7, step K2.

`MapLocalizer` stands where `LatticePhase` stands in the deploy loop and hands out the world frame,
behind the same four members (`reset`, `update`, `world`, `epoch`). With a fix, that frame is the
MAP frame of `known_map.py` (world = map tile - 30), so every consumer's positions are map
positions. Without one, it hands out the lattice frame exactly as the loop does today. A fifth
member, `observe`, takes each tick's classified cells after the occupancy map has had them; that
is where the fix is found and checked. Nothing here knows about the loop.

#### One score, and why floor on floor is not in it

A classified view placed on the label at a whole-tile offset scores +1 for each cell whose class
agrees with the label cell under it and -1 for each that differs (`score_placements`). Wall and
fence count as one class; a spawn is floor. A cell is not scored when it is outside the footprint
or under the HUD (`plan.valid`), gassed, under a crate or brawler box (the occupancy map's own
masks), under 0.5 classifier confidence, or off the map.

**Nor when view and label are both floor** (a change from the plan's first draft). Floor is two
thirds of Dark Passage, so floor landing on floor says almost nothing about where the view is.
Counted, it let a view of open ground commit. An all-floor view scored inside each spawn's landing
window (the 323 cells of the emulator plan the occupancy map would take) put its best placement
0 - 32 cells a tick ahead of the rest of the window, at 0.72 - 0.86 agreement. From a still camera
that reaches the provisional margin within 5 ticks at 6 of the 10 spawns (measured 2026-09-28).
Without floor on floor, the same view agrees on no cell anywhere, so it reaches neither agreement
floor. A view's floor on a label wall still counts -1, so floor still rules placements out.

#### The camera stops at the map edges, and it does so at spawn

The game camera stops following the hero near an edge (`camera.clamp_onset`,
`brawl_sim/core/camera.py`), and `scripts/probes/edge_probe.py` found that nearly every clamp run in
its footage started at spawn (BRAWL_SIM_DESIGN.md §9). Nine of Dark Passage's ten spawns sit inside
the default band, seven of them by more than `PRIOR_TILES` and one by 10.3 tiles. So the landing
prior does not put the viewport's nominal point on the spawn. It puts it on the spawn clamped as the
sim clamps its camera. The onsets are still geometry defaults, and the probe could only bound them
from below, so each spawn's window runs from that clamped point back to the spawn itself,
`PRIOR_TILES` past either end.

#### Two statistics that need support, and the fixed state's timing

Agreement over a handful of cells means nothing. Views of other maps scored against Dark Passage
put their whole-map leader on as few as 4 cells over 5 s, at up to 1.0 agreement. So the wrong-map
verdict counts the leader only when it averaged `MIN_SCORED` cells a tick, and the smoothed rates
follow the same floor. Over three seeds of every sim map that gave the verdict in 112 of 114 runs.
The other 2 (vine_springs, leader at 0.56) stayed in search, and none committed. Staying in search
is the cheap way to be wrong, since the verdict holds for the rest of the match.

The guard reads the best agreement within a tile of the fix, not at it. A fix one tile off is the
neighbour check's to move, and a guard read at the fix dropped the K2 drift case before the
neighbour check could move it. `SMOOTH_SECONDS` and `slip_margin` come from 240 seeded one-tile
odometry slips on the K2 test views, with the guard as it is now (2026-09-28):

| smoothing, margin | slips moved once | dropped | never moved | time to move, median / worst |
|---|---|---|---|---|
| 2 s, 30 cells | 160 | 50 | 30 | 4.3 s / 7.8 s |
| 1 s, 15 cells | 240 | 0 | 0 | 2.2 s / 3.5 s |

With the fix right, no neighbour's smoothed lead rose above -8 cells in 80 walks of 12 s. The
weakest real slip led by 22.

#### Sub-tile registration comes from the terrain, not the crates (K4)

The recorded Dark Passage matches (2026-09-28) put every problem the first replays showed on one
cause: a fix right to the whole tile but a fraction of a tile off the terrain's own lattice. Half
a tile cost the true fix 0.15 - 0.2 of its agreement (0.81 down to 0.59 - 0.66), so the guard
dropped right fixes, and a whole-map search run in a frame that far off locked wrong. Three
things put it there:

- The crates. Dark Passage's read (-0.10, +0.41) tile off the tile centres the terrain agrees
  best on, the same to 0.02 on both clips that had crates (not the candles: every sighting landed
  on floor), and the first draft's crate check moved every fix by that much. Crate residuals are
  telemetry only now.
- No crates: a fix landed on odometry's own sub-tile phase, which is arbitrary at each start. That
  phase was all that carried over between matches replayed on one stack.
- Odometry drift: the terrain's best sub-tile placement moved by up to half a tile in 15 s.

So check 3 tracks the terrain for the whole match. Every `SUBTILE_SECONDS` it fits a tent per
axis to the agreement rates a tile either side of the fix (`_subtile`), and an estimate past
`SUBTILE_TILES` moves the fix by it. On the recordings' frames re-registered at 64 sub-tile
shifts, the fit read 0.3 - 0.85 of a quarter-tile error, the right way on average, so a step
undershoots rather than overshoots. Run closed-loop over those frames it raised the fix's smoothed
agreement from a median of 0.66 to 0.76 and cut the time the guard's reading spent under 0.6 from
19% to 6.5%, at about 10 steps a minute. So a step is not an epoch: an epoch drops every track,
the loot, the gas map and the history, and a step this small is what odometry's drift already
does between epochs. After a drop, the search keeps the lost fix's frame rather than the
lattice's, so the next fix starts on the terrain's lattice as well.

#### States

    search    no fix: the lattice frame, or the last fix's once one was lost. Every placement's
              score accumulates; a clear leader commits (a new epoch). At landing only the spawn
              windows may commit, and after `search_seconds` without one the whole map may, at
              `whole_map_margin`. That whole-map search is the one place a commit needs
              `min_agreement`, and only before the match's first fix. A commit in the spawn
              windows, or after a first fix (here and in confirm), needs `guard_agreement`.
              Only a placement whose camera the game allows may lead or commit (K5).
    fixed     odometry + offset. A fix whose camera the game does not allow drops at once (K5).
              Checks 1-3 run (agreement, a neighbour's lead, the terrain's sub-tile), then the
              guard. A whole-tile move is a new epoch; a sub-tile step is not.
    confirm   after an odometry cut: the fix carried across, same epoch, while a search within
              `search_tiles` confirms it (epoch kept) or moves it (new epoch).
    mismatch  the landing search found nothing like the label: the lattice frame until `reset`.

K4 (2026-09-28) kept the plan's margins and agreements. On the three tuning recordings, fresh and
back to back on one stack, every landing committed within 0.1 s of the gate on the right spawn
and no fix was dropped. The fix's smoothed agreement had a 5th percentile of 0.64 - 0.65 against
the guard's 0.6, and the best neighbour's smoothed lead peaked at -7 cells against `slip_margin`.

One held-out match then dropped a right fix where the view read under 0.6 at every placement.
The search after it led by `whole_map_margin` within 0.8 s, but a running total begun in that
stretch averaged 0.648 at best in the 15 s left, so `min_agreement` held it off: a fix could be
kept at 0.6 but not found again under 0.65. So after the match's first fix a commit needs only
the guard's agreement (the user, 2026-09-28). The margins, which are what rule out a wrong
placement, are unchanged.

K5's first live landing was held off the same way. Its leader in the spawn windows cleared
`commit_margin` on the first tick but agreed at 0.62 - 0.65 for the 5 s the match lasted. On K2's
test views of every sim map told Dark Passage (378 runs, 7 s each), no wrong-map view committed in
the spawn windows at 0.65 or at 0.6. Dark Passage's own views, misread nearly twice as often as
K2's, landed right in the windows at 9 of the 10 spawns at 0.6, against 4 at 0.65. So a commit in
the spawn windows needs only the guard's agreement as well (the user, 2026-09-28). The whole-map
search before a first fix keeps `min_agreement`: every placement on the map is eligible there,
and at 0.6 it let 7 of those 378 runs commit, against 4. As built, they commit 4 times, all on the
whole map, as before.

K5's second control match (2026-09-28) dropped a right fix and then committed 45 tiles away, on a
placement in the map's north-east corner that put the camera 12 - 21 tiles past where the sim's
camera stops. Most of that placement's view hung off the map, where nothing is scored, and the
sliver left on it matched the edge's bush band at 0.604. Once committed it could not be dropped:
on fewer than `MIN_SCORED` cells a tick the guard does not judge. So a placement may lead and
commit, and a fix may stay, only while the camera's nominal point it implies is within
`CAMERA_TILES` of the sim's clamp limits (the user, 2026-09-29). The game's camera goes further
than the sim's: right fixes on the live matches and the tuning recordings put it up to 2.9 tiles
past on the north and south edges, 2.5 on the east and 1.9 on the west. `CAMERA_TILES` covers
that and a fix a tile off that the neighbour check has yet to move, with a tile to spare. A fix
past it drops on that tick, not after `GUARD_SECONDS`: no misread puts the camera there.
"""
import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from brawl_sim.constants import Tile
from brawl_vision.camera import HERO_ANCHOR_TILES
from brawl_vision.terrain.labeling import CLASSES
from brawl_vision.terrain.occupancy import _MIN_FOOTPRINT

from .known_map import KnownMap
from .lattice import LatticePhase, LatticeResult, _wrap

__all__ = ["MapLocalizer", "MapResult", "score_placements", "SCORE_CLASS", "MIN_CONFIDENCE",
           "PRIOR_TILES"]

# The score's classes. Fence counts as wall, since both stop a body (section 6.1).
FLOOR, WALL, BUSH, WATER = range(4)
_SCORED_AS = {Tile.FLOOR: FLOOR, Tile.WALL: WALL, Tile.FENCE: WALL, Tile.BUSH: BUSH,
              Tile.WATER: WATER}
SCORE_CLASS = np.array([_SCORED_AS[t] for t in CLASSES], np.int8)    # CLASSES index -> class

MIN_CONFIDENCE = 0.5      # a classified cell below this confidence is not scored
PRIOR_TILES = 2           # the landing window's reach past each spawn's own range
SMOOTH_SECONDS = 1.0      # time constant of the fixed state's smoothed scores
SLIP_SECONDS = 1.0        # how long a neighbour must lead before the fix moves onto it
GUARD_SECONDS = 2.0       # how long smoothed agreement may sit under the guard
SUBTILE_SECONDS = 1.0     # K4: each block of ticks the sub-tile check reads one estimate from
SUBTILE_TILES = 0.2       # K4: an estimate past this on either axis moves the fix by it
MIN_SCORED = 20.0         # smoothed scored cells a tick below which agreement is not judged
CAMERA_TILES = 5.0        # K5: how far past the sim's clamp limits a fix may put the camera

SEARCH, CONFIRM, FIXED, MISMATCH = "search", "confirm", "fixed", "mismatch"


def score_placements(view: np.ndarray, scored: np.ndarray, label: np.ndarray):
    """Score every whole-tile placement of a classified view on a label (section 6.1).

    `view` and `label` hold `SCORE_CLASS` classes, and `scored` marks the view cells that may
    count. Returns `(score, n)`, two `(label_rows + rows - 1, label_cols + cols - 1)` arrays in
    which index `(i, j)` puts the view's top-left cell on label cell `(i - rows + 1,
    j - cols + 1)`, so every placement that overlaps the map by a cell is there. `score` is agreeing
    minus differing cells and `n` the cells that counted, so the agreement rate is
    `(score + n) / (2 n)`. Off-map cells and floor on floor count in neither.

    One `cv2.matchTemplate` per plane: 0.2 ms for all five at 19 x 30 on 60 x 60, and exact once
    rounded (checked against a direct count, 2026-09-28).
    """
    rows, cols = view.shape
    pad = ((rows - 1, rows - 1), (cols - 1, cols - 1))
    lab = np.pad(label.astype(np.int16), pad, constant_values=-1)
    weight = scored.astype(np.float32)

    def count(plane, template):
        return np.rint(cv2.matchTemplate(plane.astype(np.float32), template, cv2.TM_CCORR))

    agree = sum(count(lab == k, weight * (view == k)) for k in (WALL, BUSH, WATER))
    n = count(lab >= 0, weight) - count(lab == FLOOR, weight * (view == FLOOR))
    return 2.0 * agree - n, n


@dataclass(frozen=True)
class MapResult:
    """One call's outcome, for the log and the `map_*` telemetry columns (section 7)."""
    state: str                              # "search" | "confirm" | "fixed" | "mismatch"
    epoch: int                              # what the handed-out world carries as its `segment`
    offset: tuple[float, float] | None      # map world = odometry + offset; None without a fix
    agreement: float | None                 # fixed: smoothed rate at the fix; else the leader's
    margin: float | None                    # fixed: best neighbour's smoothed lead; else the leader's
    crate_resid: tuple                      # this tick's crate residuals, (dx, dy) each
    event: str                              # one line for the log, or ""
    lattice: LatticeResult | None           # the lattice's own result, for its telemetry columns
    # K5: the search's leader as the offset a commit on it would take; None while fixed
    leader: tuple[float, float] | None = None


class MapLocalizer:
    """Feed it every perception tick as the loop feeds `LatticePhase` (`update`, then `world`),
    and `observe` once that tick's cells are classified. One per loop; `reset` on each gate tick.

    `clamp_onset` is the sim's `camera_clamp_onset`, (west, east, north, south) tiles from each
    map edge: the landing prior and the camera box (K5) have to know where the camera stops.
    """

    def __init__(self, known: KnownMap, clamp_onset, *, commit_margin: float = 60.0,
                 whole_map_margin: float = 120.0, slip_margin: float = 15.0,
                 min_agreement: float = 0.65, guard_agreement: float = 0.6,
                 wrong_map_agreement: float = 0.55, search_tiles: int = 4,
                 search_seconds: float = 5.0):
        self.name = known.name
        self.commit_margin = commit_margin
        self.whole_map_margin = whole_map_margin
        self.slip_margin = slip_margin
        self.min_agreement = min_agreement
        self.guard_agreement = guard_agreement
        self.wrong_map_agreement = wrong_map_agreement
        self.search_tiles = search_tiles
        self.search_seconds = search_seconds

        self._label = SCORE_CLASS[known.classes]
        rows, cols = known.tiles.shape
        self._half = np.array([cols // 2, rows // 2])                     # map tile = world + half
        self._spawns = known.spawns.astype(np.float64) + self._half      # map tiles
        west, east, north, south = clamp_onset
        self._cam_lo = np.array([west, north], np.float64)
        self._cam_hi = np.array([cols - east, rows - south], np.float64)

        self.lattice = LatticePhase()
        self.epoch = 0
        self.state = SEARCH
        self.in_map_frame = False     # whether the last `world` call handed out the map frame
        self._segment: int | None = None
        self._lattice_epoch = self.lattice.epoch
        self._offset: np.ndarray | None = None
        # K4: once a fix is lost, the search frame keeps it (map world = odometry + this), so the
        # search reads cells on the lattice the fix had found. None: the lattice's frame.
        self._frame: np.ndarray | None = None
        self._last_world: np.ndarray | None = None
        self._armed = False           # the next "ok" tick anchors the spawn prior
        self._prior = None            # (tracked, clamped) odometry-frame offsets, one row a spawn
        self._landing = True          # the wrong-map verdict is still to come
        self._t0: float | None = None
        self._deadline: float | None = None
        self._tick_resid: tuple = ()
        self._lat: LatticeResult | None = None
        self._clear_search()
        self._clear_checks()

    # -- the four members the loop already calls on LatticePhase ---------------

    def reset(self, segment: int) -> None:
        """A new match, on its gate tick: the lattice starts over, the fix is dropped, and the spawn
        prior anchors on the next "ok" tick."""
        self.lattice.reset(segment)
        self._segment = segment
        self._landing = True
        self.in_map_frame = False
        self._to_search(armed=True)
        self._frame = None

    def update(self, detections, plan, odometry) -> MapResult:
        """`LatticePhase.update`'s contract: every tick, before `world`, with the projectile
        model's boxes and the base plan. On top of the lattice it carries the fix across an
        odometry cut and anchors the landing prior, since each can change the frame this tick's
        consumers should use, and reads the crate residuals for telemetry."""
        self._lat = self.lattice.update(detections, plan, odometry)
        self._tick_resid = ()
        event = ""
        if self._segment is not None and odometry.segment != self._segment:
            event = self._cut(odometry)
        self._segment = odometry.segment
        if self._offset is None and self._frame is None and \
                self.lattice.epoch != self._lattice_epoch:
            # The lattice re-locked or restarted, so the frame handed out has moved.
            self.epoch += 1
            self._clear_search()
        self._lattice_epoch = self.lattice.epoch
        if odometry.status == "ok":
            if self._armed:
                self._anchor(plan, odometry)
            if self.state == FIXED and self._lat.sightings:
                self._crates(self._lat.sightings)
        if self._offset is not None and odometry.tracking:
            self._last_world = np.asarray(odometry.position_tiles, np.float64) + self._offset
        return self._result(event)

    def world(self, odometry):
        """`odometry` in the frame handed out: the map frame while there is a fix (or one being
        confirmed across a cut), otherwise the frame the search runs in: the lattice's until a
        fix is lost, then the lost fix's. `segment` is the epoch.

        `in_map_frame` records which, for `KnownTerrain`. It is set here and not read off the
        state, because `observe` runs later in the same tick: a fix it commits, moves or drops
        applies from the next `world` call, and until then every consumer, the terrain included,
        must stay in the frame this call handed out."""
        self.in_map_frame = self._offset is not None
        if self._offset is None and self._frame is None:
            return replace(self.lattice.world(odometry), segment=self.epoch)
        base = self._offset if self._offset is not None else self._frame
        px, py = odometry.position_tiles
        return replace(odometry, position_tiles=(px + base[0], py + base[1]), segment=self.epoch)

    # -- the new member ------------------------------------------------------

    def observe(self, t: float, cells, confidence, plan, world, zone=None,
                occluded=None) -> MapResult:
        """Score this tick's classified cells against the label and act on the result.

        `cells` and `confidence` are the classifier's output on `plan`, the plan registered at
        `world`, which is what `world()` handed out this tick. `zone` and `occluded` are the masks
        the occupancy map got. A change made here applies from the next tick.
        """
        if world.status != "ok" or self.state == MISMATCH:
            return self._result("")
        rows, cols = cells.shape
        ppt = plan.pixels_per_tile
        scored = plan.valid.reshape(rows, ppt, cols, ppt).mean(axis=(1, 3)) >= _MIN_FOOTPRINT
        scored &= np.asarray(confidence) >= MIN_CONFIDENCE
        if zone is not None:
            scored &= ~zone
        if occluded is not None:
            scored &= ~occluded
        score, n = score_placements(SCORE_CLASS[cells], scored, self._label)
        # The placement at offset (0, 0) of the handed-out frame: the view's top-left world tile,
        # onto the map, plus the padding `score_placements` indexes with.
        x0 = int(round(plan.origin_tile[0] + world.position_tiles[0]))
        y0 = int(round(plan.origin_tile[1] + world.position_tiles[1]))
        ref = (y0 + int(self._half[1]) + rows - 1, x0 + int(self._half[0]) + cols - 1)
        # K5: the camera's nominal point at that placement, in map tiles. A placement `s` whole
        # tiles on puts it at `cam + s`.
        cam = np.asarray(world.position_tiles, np.float64) + _nominal(plan) + self._half
        if self.state == FIXED:
            return self._result(self._check(t, score, n, ref, cam))
        return self._result(self._search(t, score, n, ref, world, cam))

    # -- searching -----------------------------------------------------------

    def _search(self, t, score, n, ref, world, cam) -> str:
        """Sections 6.3, 6.5 and 6.6. Totals are kept per placement and moved with the camera, so
        each one stays the same hypothesis about the offset from tick to tick."""
        if self._totals is None:
            self._totals = np.zeros(score.shape)
            self._counts = np.zeros(score.shape)
            self._mask = self._eligible(ref, score.shape)
        else:
            dy, dx = ref[0] - self._ref[0], ref[1] - self._ref[1]
            self._totals = _shift(self._totals, dy, dx, 0.0)
            self._counts = _shift(self._counts, dy, dx, 0.0)
            if self._mask is not None:
                self._mask = _shift(self._mask, dy, dx, False)
        self._ref = ref
        self._totals += score
        self._counts += n
        self._ticks += 1
        if self._t0 is None:
            self._t0 = t
        if self.state == CONFIRM and self._deadline is None:
            self._deadline = t + self.search_seconds

        # K5: only where the game's camera can be, this tick. A placement whose view hangs off the
        # map is scored on the sliver left on it, and on a live match a corner's sliver led.
        eligible = _box(score.shape, ref, *self._camera_steps(cam))
        if self._mask is not None:
            eligible &= self._mask
        best, lead, agree = _leader(self._totals, self._counts, eligible)
        self._agree, self._margin = agree, lead
        step = None if best is None else np.array([best[1] - ref[1], best[0] - ref[0]], np.float64)
        # K5: the offset a commit on the leader would take (as `_commit` takes it), for the log, so
        # a search that never commits can still be checked against the spawn its match began on.
        self._lead = None if step is None else \
            (self._offset if self.state == CONFIRM else self._base()) + step
        margin = self.whole_map_margin if self._mask is None else self.commit_margin
        # Once this match has had a fix (carried across a cut, or lost), the map is known and only
        # the offset is in doubt, which the margin settles. So a commit needs only the agreement
        # that keeps a fix. So does one in the spawn windows (K5): no view of another map committed
        # there at that rate. In the whole-map search before a first fix, where every placement on
        # the map may lead, `min_agreement` still stands between a confident fit and the wrong map.
        had_fix = self._offset is not None or self._frame is not None
        in_windows = self.state == SEARCH and self._mask is not None
        floor = self.guard_agreement if had_fix or in_windows else self.min_agreement
        if best is not None and lead >= margin and agree >= floor:
            return self._commit(step, world)

        if self.state == CONFIRM:
            if t >= self._deadline:
                self._to_search()
                return f"fix dropped: no commit within {self.search_seconds:g} s of the cut"
            return ""
        if t - self._t0 >= self.search_seconds:
            if self._landing:
                self._landing = False
                best, _, anywhere = _leader(self._totals, self._counts, None)
                # A placement that barely overlaps the map, or sees little but floor, can agree
                # on all of its few cells.
                if best is None or self._counts[best] < MIN_SCORED * self._ticks:
                    anywhere = 0.0
                if anywhere < self.wrong_map_agreement:
                    self.state = MISMATCH
                    return f"map mismatch: expected {self.name}"
            if self._mask is not None:
                # Nothing in the spawn windows in time: the whole map, at its larger margin.
                self._prior = None
                self._mask = None
        return ""

    def _eligible(self, ref, shape):
        """The placements that may commit: a window around the carried fix while confirming, the
        spawn windows at landing, and None (every placement) otherwise."""
        if self.state == CONFIRM:
            r = self.search_tiles
            return _box(shape, ref, (-r, -r), (r, r))
        if self._prior is None:
            return None
        # The frame is odometry + `_base()`, so an odometry-frame offset o is a shift of
        # o - `_base()` from it.
        tracked, clamped = (np.rint(o - self._base()).astype(int) for o in self._prior)
        lo = np.minimum(tracked, clamped) - PRIOR_TILES
        hi = np.maximum(tracked, clamped) + PRIOR_TILES
        mask = np.zeros(shape, bool)
        for a, b in zip(lo, hi):
            mask |= _box(shape, ref, a, b)
        return mask

    def _camera_steps(self, cam):
        """K5: the whole-tile steps from the frame handed out, (x, y) as `_box` takes them, that
        keep the camera's nominal point (`cam` at no step) within `CAMERA_TILES` of the clamp
        limits."""
        lo = np.ceil(self._cam_lo - CAMERA_TILES - cam).astype(int)
        hi = np.floor(self._cam_hi + CAMERA_TILES - cam).astype(int)
        return lo, hi

    def _anchor(self, plan, odometry) -> None:
        """The spawn prior (section 6.3), on the match's first "ok" tick, when the hero is still
        on its spawn: per spawn, the odometry-frame offset (map world = odometry + offset) that
        puts the camera's nominal point on the spawn, and the one that puts it where the sim's
        camera stops."""
        self._armed = False
        nominal = np.asarray(odometry.position_tiles, np.float64) + _nominal(plan)
        clamped = np.clip(self._spawns, self._cam_lo, self._cam_hi)
        self._prior = (self._spawns - self._half - nominal, clamped - self._half - nominal)
        self._clear_search()

    def _commit(self, step, world) -> str:
        """Take the leading placement, `step` whole tiles from the frame handed out this tick."""
        if self.state == CONFIRM:
            self._offset = self._offset + step
            if step.any():
                self.epoch += 1
                event = f"cut moved the fix by ({step[0]:+.0f}, {step[1]:+.0f})"
            else:
                event = "cut confirmed, fix kept"
        else:
            # That frame was odometry + `_base()`: the map frame is it plus `step`.
            self._offset = step + self._base()
            self.epoch += 1
            event = (f"fixed on {self.name} at offset ({self._offset[0]:+.2f}, "
                     f"{self._offset[1]:+.2f})")
        self._last_world = np.asarray(world.position_tiles, np.float64) + step
        self.state = FIXED
        self._landing = False
        self._clear_search()
        self._clear_checks()
        return event

    def _base(self) -> np.ndarray:
        """The search frame, as map world = odometry + this: the lost fix's, or the lattice's
        (odometry - phase)."""
        if self._frame is not None:
            return self._frame
        return -np.asarray(self.lattice.phase, np.float64)

    def _cut(self, odometry) -> str:
        """Odometry opened a new segment (section 6.5). A fix is carried across on the assumption
        that the camera did not move, and must be confirmed before it counts as fixed again. A
        search loses its spawn prior, which was tied to the old segment. A search in a lost fix's
        frame keeps that frame on the same assumption, in a new epoch, as the lattice's own
        restart would give it."""
        if self._offset is None:
            self._prior = None
            if self._frame is not None:
                self.epoch += 1
                self._clear_search()
            return ""
        if self._last_world is not None:
            self._offset = self._last_world - np.asarray(odometry.position_tiles, np.float64)
        was = self.state
        self.state = CONFIRM
        self._clear_search()
        self._clear_checks()
        return "odometry cut: confirming the carried fix" if was == FIXED else ""

    def _to_search(self, armed: bool = False) -> None:
        """No fix, in a new epoch. A fix being dropped becomes the search frame (K4): its
        sub-tile phase is the terrain's, where the lattice's is the crates' or none."""
        if self._offset is not None:
            self._frame = self._offset
        self.state = SEARCH
        self._offset = None
        self._last_world = None
        self._armed = armed
        self._prior = None
        self._t0 = None
        self._deadline = None
        self.epoch += 1
        self._lattice_epoch = self.lattice.epoch
        self._clear_search()
        self._clear_checks()

    # -- keeping the fix -----------------------------------------------------

    def _check(self, t, score, n, ref, cam) -> str:
        """Checks 1-3 and the guard (section 6.4): the score and agreement rate at the fix and at
        its 8 neighbours, each smoothed over `SMOOTH_SECONDS` for checks 1 and 2 and the guard, and
        summed over `SUBTILE_SECONDS` for check 3. A stretch too bare to rate, such as open floor,
        neither drops the fix nor counts toward dropping it. Before all of them, the camera box
        (K5): a fix that far off can see too little of the map for the guard to judge."""
        past = float(np.max(np.maximum(self._cam_lo - cam, cam - self._cam_hi)))
        if past > CAMERA_TILES:
            self._to_search()
            return f"fix dropped: it puts the camera {past:.1f} tiles past the clamp limits"

        s, c = _window(score, *ref), _window(n, *ref)
        now = (s, (s + c) / 2.0, c)
        if self._ema is None:
            self._ema = now
        else:
            a = 1.0 - math.exp(-max(t - self._t_last, 0.0) / SMOOTH_SECONDS)
            self._ema = tuple(e + a * (v - e) for e, v in zip(self._ema, now))
        self._t_last = t
        scores, agreeing, counted = self._ema
        lead = scores - scores[1, 1]
        lead[1, 1] = -np.inf
        k = np.unravel_index(np.argmax(lead), lead.shape)
        self._margin = float(lead[k])
        self._agree = _rate(agreeing[1, 1], counted[1, 1])

        if lead[k] > self.slip_margin:
            if self._slip is None or self._slip[0] != k:
                self._slip = (k, t)
            elif t - self._slip[1] >= SLIP_SECONDS:
                step = np.array([k[1] - 1, k[0] - 1], np.float64)
                self._move(step)
                return f"fix slipped a tile: ({step[0]:+.0f}, {step[1]:+.0f})"
        else:
            self._slip = None

        # Check 3 (K4): the sub-tile error, from the same counts summed over a block of ticks. No
        # epoch and no log line: a step this small is what odometry's own drift does to every
        # consumer between epochs, and it comes several times a minute.
        if self._block is None:
            self._block = [t, 0, np.zeros((3, 3)), np.zeros((3, 3))]
        self._block[1] += 1
        self._block[2] += now[1]
        self._block[3] += now[2]
        if t - self._block[0] >= SUBTILE_SECONDS:
            step = _subtile(*self._block[1:])
            self._block = None
            if np.abs(step).max() > SUBTILE_TILES:
                self._offset = self._offset + step
                if self._last_world is not None:
                    self._last_world = self._last_world + step

        # The guard reads the best agreement within a tile of the fix. A fix a tile off is the
        # neighbour check's to move; dropping it would reset every consumer for nothing.
        near = max((r for r in map(_rate, agreeing.ravel(), counted.ravel()) if r is not None),
                   default=None)
        if near is not None and near < self.guard_agreement:
            if self._low_since is None:
                self._low_since = t
            elif t - self._low_since >= GUARD_SECONDS:
                self._to_search()
                return (f"fix dropped: agreement under {self.guard_agreement:g} for "
                        f"{GUARD_SECONDS:g} s")
        else:
            self._low_since = None
        return ""

    def _crates(self, sightings) -> None:
        """Each full-height crate sighting's distance from a tile centre in the map frame, for
        the `map_crate_resid` column. A crate sits on a tile centre, but K4 found Dark Passage's
        read 0.41 tile off the terrain's, so they no longer move the fix (see check 3)."""
        at = np.asarray(sightings, np.float64) + self._offset
        self._tick_resid = tuple((float(x), float(y)) for x, y in -_wrap(at - 0.5))

    def _move(self, step) -> None:
        """Shift the fix a whole tile. Always a new epoch, because every consumer holds world
        positions."""
        self._offset = self._offset + step
        if self._last_world is not None:
            self._last_world = self._last_world + step
        self.epoch += 1
        self._clear_checks()

    # -- bookkeeping ---------------------------------------------------------

    def _clear_search(self) -> None:
        self._totals = self._counts = self._mask = self._ref = None
        self._ticks = 0

    def _clear_checks(self) -> None:
        self._ema = self._t_last = self._slip = self._low_since = self._block = None
        self._agree = self._margin = self._lead = None

    def _result(self, event: str) -> MapResult:
        offset, lead = (None if v is None else (float(v[0]), float(v[1]))
                        for v in (self._offset, self._lead))
        return MapResult(self.state, self.epoch, offset, self._agree, self._margin,
                         self._tick_resid, event, self._lat, lead)


def _nominal(plan) -> np.ndarray:
    """Where the hero ring sits while the camera tracks the hero, in the plan's camera-relative
    tiles: the viewport centre through the plan plus `HERO_ANCHOR_TILES`. The sim's camera is
    centred on this point (`brawl_sim/core/camera.py`); `EntityTracker._nominal_tile` is it plus
    the player box's own offset."""
    w, h = plan.viewport
    rect = cv2.perspectiveTransform(np.array([[[w / 2.0, h / 2.0]]], np.float64), plan.M)
    return plan.rect_to_tile(rect.reshape(1, 2))[0] + np.asarray(HERO_ANCHOR_TILES, np.float64)


def _rate(agreeing, counted):
    """A smoothed agreement rate, or None when fewer than `MIN_SCORED` cells a tick counted."""
    return float(agreeing / counted) if counted >= MIN_SCORED else None


def _subtile(ticks, agreeing, counted) -> np.ndarray:
    """Check 3's estimate (K4): how far the terrain's best placement sits from the fix, (x, y)
    tiles, from a block of ticks' agreeing and scored cells at the fix and its 8 neighbours.

    Per axis, the agreement rates a tile back, at the fix and a tile on (m, c, p) are read as a
    tent peaked at the optimum: d = (p - m) / (2 (c - min(m, p))). The fit needs c above both,
    which keeps |d| under a half. Otherwise, or on too few cells, that axis reads 0. It all
    reads 0 while any neighbour agrees better than the fix: that is a whole tile, the neighbour
    check's, and the fix's own row and column say nothing about the sub-tile then."""
    out = np.zeros(2)
    rate = np.where(counted >= MIN_SCORED * ticks, agreeing / np.maximum(counted, 1e-9), np.nan)
    if np.isnan(rate[1, 1]) or np.nanmax(rate) > rate[1, 1]:
        return out
    for axis, (m, c, p) in enumerate(((rate[1, 0], rate[1, 1], rate[1, 2]),
                                      (rate[0, 1], rate[1, 1], rate[2, 1]))):
        if np.isnan((m, c, p)).any() or c <= max(m, p):
            continue
        out[axis] = (p - m) / (2.0 * (c - min(m, p)))
    return out


def _leader(totals, counts, mask):
    """The best eligible placement, its lead over every eligible placement outside its own 3 x 3,
    and its agreement rate. `(None, 0.0, 0.0)` when nothing is eligible."""
    t = totals if mask is None else np.where(mask, totals, -np.inf)
    i, j = np.unravel_index(np.argmax(t), t.shape)
    if not np.isfinite(t[i, j]):
        return None, 0.0, 0.0
    rest = t.copy()
    rest[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2] = -np.inf
    n = counts[i, j]
    agree = float((t[i, j] + n) / (2.0 * n)) if n > 0 else 0.0
    return (int(i), int(j)), float(t[i, j] - rest.max()), agree


def _shift(a, dy, dx, fill):
    """`a` moved by (dy, dx) cells, with `fill` where nothing moved in."""
    out = np.full_like(a, fill)
    h, w = a.shape
    if abs(dy) >= h or abs(dx) >= w:
        return out
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        a[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def _box(shape, ref, lo, hi):
    """Placements whose shift from `ref` is within [lo, hi] on both axes, (x, y), inclusive."""
    mask = np.zeros(shape, bool)
    i0, i1 = max(ref[0] + lo[1], 0), max(ref[0] + hi[1] + 1, 0)
    j0, j1 = max(ref[1] + lo[0], 0), max(ref[1] + hi[0] + 1, 0)
    mask[i0:i1, j0:j1] = True
    return mask


def _window(a, i, j):
    """The 3 x 3 of `a` centred on (i, j), zero past its edges."""
    h, w = a.shape
    if not (0 <= i < h and 0 <= j < w):
        return np.zeros((3, 3))
    return np.pad(a, 1)[i:i + 3, j:j + 3].astype(np.float64)
