"""Which world tiles are the game's tiles, read off the crates. The terrain plan's snapping, step B.

Odometry starts each segment's world frame wherever the camera happened to be, so the world's whole
tiles sit at an arbitrary sub-tile offset from the game's. Registration (`RectifyPlan.registered`,
step A) makes every deposit land on the MAP's lattice; it cannot say where the GAME's lattice is.
When the two are half a tile apart, every wall straddles two map cells. The sim's walls fill whole
tiles and its crop starts at `floor(hero_pos)` (`grid.py`), so this offset is also the one thing
standing between the deployed grid and the sim's.

A crate sits on a tile centre, forever (`loot.py`), so each crate sighting measures that offset:
`wrap(world anchor - 0.5)`. This module averages those readings for the segment and hands the
loop a world frame shifted by the result. The shift is the same for every consumer, so hero,
enemies, projectiles, loot, gas and terrain stay in one frame.

#### Measured: what one crate says about the phase

The replay behind every number here: the 28 labelled clips at 12 Hz, shipped projectile model,
crate boxes that pass loot's edge rule. That is 49 odometry segments and 1,237 s of "ok" footage
(`scratchpad/exp_phase*.py`, 2026-09-15).

  * **The anchor is good; a PARTLY HIDDEN crate is not.** On the calibration clip one crate reads
    0.03 tile from its true centre (`loot.py`). Across all clips, two crates in the same frame
    share the odometry error exactly, yet their readings still differ by a median 0.10 / 0.07 tile
    (x / y), and by more than 0.25 on some axis in 25% of pairs. The bad ones are boxed short: a
    crate behind another crate or a wall loses its bottom, and the anchor climbs with it.

        box height / expected    n      |y error| median    over 0.25 tile
        under 0.8               329    0.173               47%
        0.8 - 0.9               239    0.248               54%
        0.9 - 1.0              1446    0.044               11%
        1.0 - 1.1              2056    0.063                7%

    "Expected" is `CRATE_HEIGHT_PX`, a line fitted on 10,219 full boxes against the box's bottom
    row: `152 + 0.0315 * y1` viewport px. Full boxes sit at 0.94 - 1.04 of it (p5 - p95), so
    `HEIGHT_RATIO = 0.9` drops the short ones and keeps nearly every full one. One recording
    (ScreenRecording_09-04-2026 15-04-03_1) boxes every crate at about 0.66 of it. There every
    box fails the test, the phase never locks, and the frame stays where step A left it.
  * **Distance from the screen centre was tried and dropped.** Readings do lean with screen
    position, by up to +0.16 tile x at the right edge (a camera-model term). But cutting sightings
    beyond 600 px of the centre did not improve the estimate. It also discards the calibration
    clip's own crate.

#### Measured: one running mean per segment, not a window

Each estimator was scored leave-one-crate-out: the estimate at a sighting's time is built only
from OTHER crates' earlier sightings. The score is the held-out full-height sighting's
disagreement with it, max over axes, so it includes that crate's own error.

    step A (phase 0)                    median 0.26
    first 3 sightings, then frozen      median 0.17     p90 0.41
    running mean of the whole segment   median 0.11     p90 0.25
    mean of the last 5 / 10 / 20 s      median 0.11     p90 0.26 - 0.30

A window buys nothing: at these time scales the error between crates outweighs any wander of
the phase over time. The phase does wander: in four long segments, early crates and late crates
disagree by 0.3 - 0.45 tile over 45 - 55 s. That could be odometry drift or the screen-position
lean; this replay cannot tell them apart. So `update` keeps one phasor sum per axis (a circular
mean, since 0.49 and -0.49 are neighbours) plus a count. That is three numbers, never a history.
It keeps following the estimate after the first lock, which three sightings make poorly.

#### The rule: re-lock in jumps, and each jump is a new epoch

The world frame must not slide under stored state. Every consumer stores world positions, and a
frame that moved 0.01 tile a tick would smear the terrain votes and put a velocity into every
track. So the applied phase is PIECEWISE CONSTANT. It starts at 0, which is step A. Once there are
`LOCK_SIGHTINGS` usable sightings, it jumps to the running estimate whenever the two differ by more
than `REBASE_TILES` on either axis. A jump opens a new EPOCH: the world odometry's `segment`
changes, and every consumer resets through the path it already has for a new odometry segment.
The terrain map's votes cannot be shifted by a fraction of a cell, and keeping them would overlay
two lattices half a tile apart. That is the same reason a new segment drops them.

Simulated on the replay against a local truth proxy (the mean of full-height sightings within
10 s either side), time-weighted, both axes:

                              |err| mean   within 0.15   within 0.25   resets
    step A (phase 0)          0.273        23%           46%             -
    lock once, then frozen    0.157        59%           81%           28 (1.4 / min)
    re-lock past 0.25         0.152        57%           84%           18 (0.9 / min)
    re-lock past 0.15         0.129        72%           89%           30 (1.5 / min)   <- shipped
    re-lock past 0.10         0.118        76%           89%           42 (2.0 / min)

Most resets are the first lock itself: 21 of the 30. The first lock comes a median 0.2 s into the
segment (p75 3.3 s), when the map holds about one view. The other 9 are later refinements.
Segments that never lock are mostly short ones; they add up to 16% of the ok time. What a reset
costs is the loop's usual price for a cut. That is one skipped
decision (the tracker's reset tick), crates missing from the grid for the quarter second loot needs
to re-confirm them, and whatever the map and the gas remembered outside the current view. That
last is mostly the HUD hole (`grid.py`), which closes again as the hero walks.

Checked against the one known answer: the calibration clip's lattice IS the game's from f060, and
its crate reads (+0.03, +0.02). That is inside `REBASE_TILES`, so the frame correctly stays at 0.

#### Validated against the wall tops, which share no input with it

A wall's top face is a whole tile on the wall-top plane (`CameraModel.H_top`). Rectify through
`S @ inv(H_top)` instead of the ground homography and every wall-top edge lands on a tile boundary.
So fold each frame's |Sobel| energy onto its world phase, per axis, and the peak is a lattice
estimate with no detector and no anchor in it (`scratchpad/exp_B_edges*.py`, 2026-09-15, every
2nd "ok" frame of the same replay). It has to be the top plane. On the ground rectification the
crispest edges are those same wall tops, sitting 0.88 tile of parallax off a boundary; there the
calibration clip peaks at (-0.10, +0.17). On the top plane it peaks at (+0.003, +0.027).

Over the 25 other segments this module locks on, |edge peak - phase|:

                                         x median   within 0.1     y median   within 0.1
    this estimate                        0.044      76%            0.090      56%
    step A (phase 0)                     0.154      36%            0.173      32%
      where the estimate is > 0.15 from 0 (16 segments on x, 11 on y):
    this estimate                        0.063      75%            0.118      36%
    step A (phase 0)                     0.286      19%            0.380       9%

Overlays of both lattices on raw frames side with the crates wherever those move off 0. The two
misses found, both confirmed on overlays:

  * **showdown_alternate_map.** Its crates read (+0.05, -0.04), so the frame stays at 0, but the
    wall tops sit at about (-0.18, +0.18). That map's crates are a squat reskin, and
    `CRATE_ANCHOR_FRAC` was fitted on the standard box.
  * **day12_recording2, second segment.** About 0.2 off on y and 0.1 on x.

The folded edges are weak evidence per frame: the peak stands 1.1 - 1.5x above the mean on a
typical segment, against 2 - 2.5x on the calibration clip, whose walls are unusually dense.

Through the deploy loop itself, on the three BlueStacks replays (`BRAWL_DEPLOYMENT_DESIGN.md` 4),
each clip locked once and stayed within 0.07 - 0.10 tile of the wall-top edges on the same ticks,
where phase 0 was 0.10 - 0.37 off.

#### What is deliberately NOT here

  * **Wall tops (H_top) as the classifier's input.** Measured and not shipped: against the
    ground plane alone it won one seed and lost the other and the BlueStacks hold-out
    (`brawl_vision/terrain/classifier.py`). The lattice here is the ground plane's; the wall-top
    plane only checks it.
  * **Other anchors.** Walls, bushes and water edges also sit on the lattice. Nothing folds them in
    yet, so a segment without a full-height crate stays at phase 0. The wall-top edges above are
    the obvious second source: they would rescue the squat-crate map and the 16% of "ok" time that
    never locks. But they converge slowly (`scratchpad/exp_B2_converge.py`, 2026-09-15): on 32
    segments, the share within 0.1 tile (max over axes) of each source's own whole-segment answer:

        seconds into the segment           1      3      10     20
        wall-top edges, running fold       25%    50%    58%    65%
        crates, where an estimate exists   64%    72%    91%    85%
        a crate estimate exists at all     44%    56%    85%    87%

    So edges could only be a fallback for the stretches with no crate, not a replacement. Even
    at 3 s they sit a median 0.13 tile from the segment's final crate answer, where phase 0 sits
    0.27 on the same segments. Whether that gain pays for the extra resets a noisier source would
    trigger is unmeasured.
  * **Rescuing a lock that comes late.** A segment whose first crate arrives 40 s in has lived 40 s
    on step A's arbitrary phase. Its lock resets whatever was built by then. bluestacks-example-zone
    is one: no full-height crate until 48 s, and 0.37 tile off on x until then.
"""
import math
from dataclasses import dataclass, replace

import numpy as np

from brawl_vision.object_detection.project import CRATE_ANCHOR_FRAC, to_tiles
from brawl_vision.object_detection.projectile_detection.classes import CUBE_BOX

from .loot import DUPLICATE_TILES, EDGE_MARGIN_PX, _clear_of_edges, _merge

__all__ = ["LatticePhase", "LatticeResult", "LOCK_SIGHTINGS", "REBASE_TILES", "HEIGHT_RATIO",
           "CRATE_HEIGHT_PX"]

# Usable sightings before the estimate is trusted: one confirmed crate's worth (`loot.MIN_HITS`).
# 6 or 12 changed the lock's error by under 0.01 tile and only delayed it.
LOCK_SIGHTINGS = 3

# How far the running estimate may sit from the applied phase before the frame re-locks. See the
# module docstring's table for the trade against resets.
REBASE_TILES = 0.15

# A crate box shorter than this share of `CRATE_HEIGHT_PX` is taken as partly hidden and not read.
HEIGHT_RATIO = 0.9

# Expected crate box height in viewport px, `a + b * y1` with y1 the box's bottom row. Fitted on
# 10,219 full boxes (2026-09-15); perspective makes a crate lower on screen a little taller.
CRATE_HEIGHT_PX = (152.0, 0.0315)


def _wrap(x) -> np.ndarray:
    """Into [-0.5, 0.5): the signed distance past the nearest whole tile."""
    return (np.asarray(x, np.float64) + 0.5) % 1.0 - 0.5


@dataclass(frozen=True)
class LatticeResult:
    """One perception tick's outcome, for telemetry."""
    status: str                              # "unlocked" | "locked" | "rebased", or odometry's own
    epoch: int                               # what the world odometry carries as its `segment`
    phase: tuple[float, float]               # applied: world = odometry position - phase
    estimate: tuple[float, float] | None     # running estimate, None before LOCK_SIGHTINGS
    n_sightings: int                         # usable sightings this segment
    n_used: int                              # usable sightings this tick
    # This tick's usable sightings in ODOMETRY's frame (the phase not taken off), for
    # `localize.MapLocalizer`'s crate check, which reads crates against a known map instead.
    sightings: tuple = ()


class LatticePhase:
    """Feed it every perception tick's projectile-model boxes; read `world(odometry)`. One per loop.

    `update` must see the tick before `world` is asked for it, because a re-lock on this tick
    changes the frame this tick's consumers should deposit into.
    """

    def __init__(self, *, lock_sightings: int = LOCK_SIGHTINGS,
                 rebase_tiles: float = REBASE_TILES, height_ratio: float = HEIGHT_RATIO,
                 crate_height_px: tuple[float, float] = CRATE_HEIGHT_PX,
                 edge_margin_px: float = EDGE_MARGIN_PX,
                 duplicate_tiles: float = DUPLICATE_TILES):
        self.lock_sightings = lock_sightings
        self.rebase_tiles = rebase_tiles
        self.height_ratio = height_ratio
        self.crate_height_px = crate_height_px
        self.edge_margin_px = edge_margin_px
        self.duplicate_tiles = duplicate_tiles
        # Starts at 0 because `OccupancyMap.segment` does, and odometry's first segment is 0: the
        # first tick must not read as a change to a consumer that has no `None` sentinel.
        self.epoch = 0
        # None, not -1: the first tick adopts odometry's segment. Same as every tracker here.
        self._segment: int | None = None
        self._clear()

    def _clear(self) -> None:
        self._phase = np.zeros(2)
        self._sum = np.zeros(2, np.complex128)
        self._n = 0

    # -- reading -------------------------------------------------------------

    @property
    def phase(self) -> tuple[float, float]:
        return (float(self._phase[0]), float(self._phase[1]))

    def estimate(self) -> tuple[float, float] | None:
        if self._n < self.lock_sightings:
            return None
        est = np.angle(self._sum) / (2.0 * math.pi)
        return (float(est[0]), float(est[1]))

    def world(self, odometry):
        """`odometry` moved onto the game's lattice: the position less the applied phase, and the
        segment replaced by the epoch, which also changes on every re-lock."""
        px, py = odometry.position_tiles
        return replace(odometry, position_tiles=(px - self._phase[0], py - self._phase[1]),
                       segment=self.epoch)

    # -- the tick ------------------------------------------------------------

    def update(self, detections, plan, odometry) -> LatticeResult:
        """`detections` in viewport pixels, all classes; only full-height crates are read.

        Gated like the trackers: a new odometry segment starts over at phase 0 in a new epoch, and
        a tick that is not `"ok"` reads nothing, because its position is not trusted.
        """
        if self._segment is None:
            self._segment = odometry.segment
        elif odometry.segment != self._segment:
            self.reset(odometry.segment)
        if odometry.status != "ok":
            return self._result(odometry.status)

        crates = [d for d in detections if d.label == CUBE_BOX
                  and _clear_of_edges(d.xyxy, plan.viewport, self.edge_margin_px)]
        px, py = odometry.position_tiles
        placed = [((tx + px, ty + py), d) for d, (tx, ty) in
                  to_tiles(crates, plan, anchor_frac=CRATE_ANCHOR_FRAC)]
        used = []
        for (x, y), d in _merge(placed, self.duplicate_tiles):
            if not self._full_height(d):
                continue
            self._sum += np.exp(2j * math.pi * _wrap(np.array([x, y]) - 0.5))
            self._n += 1
            used.append((x, y))

        est = self.estimate()
        if est is None:
            return self._result("unlocked", used)
        step = _wrap(np.asarray(est) - self._phase)
        if np.abs(step).max() <= self.rebase_tiles:
            return self._result("locked", used)
        # The smallest jump onto the estimate, so the frame moves at most half a tile.
        self._phase = self._phase + step
        self.epoch += 1
        return self._result("rebased", used)

    def _full_height(self, det) -> bool:
        x0, y0, x1, y1 = det.xyxy
        a, b = self.crate_height_px
        return (y1 - y0) >= self.height_ratio * (a + b * y1)

    def _result(self, status: str, used=()) -> LatticeResult:
        return LatticeResult(status, self.epoch, self.phase, self.estimate(), self._n, len(used),
                             tuple(used))

    def reset(self, segment: int) -> None:
        """A new world frame: odometry's new segment, or a new match. Phase back to 0, evidence
        dropped, and a new epoch so every consumer drops its state too."""
        self._clear()
        self._segment = segment
        self.epoch += 1
