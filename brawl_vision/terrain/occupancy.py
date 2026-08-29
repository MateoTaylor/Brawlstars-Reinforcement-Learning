"""Occupancy accumulation: voting, locking, and the UNKNOWN state. See
Terrain_Perception_Build_Plan.md Phase I.

This is where the per-frame stages become a map. Each frame's classified cells are deposited into a
persistent world-frame grid at the position Phase F reports, every cell accumulates votes, and a
cell locks once it has been seen enough times with enough agreement -- after which it is never
reconsidered.

**Locking does two jobs at once.** It is the efficiency win (a locked cell costs nothing to keep)
and the robustness mechanism: one frame's misclassification -- a player standing on a tile, a
moment of motion blur, a damage number over a wall -- is outvoted instead of becoming the map.

**Four gates before any vote is cast**, each tracing back to an earlier phase, and each of them a
way the map gets silently poisoned if it is missing:

1. **Odometry lost** (`status == "lost"`, post-cut) -- deposit nothing. The world frame is gone.
2. **Odometry uncertain** -- keep tracking, cast no votes. Phase F distinguishes these precisely so
   this stage can keep its position while declining to trust the frame as evidence.
3. **Under gas** -- skip. Gas tints the terrain beneath it, so a gassed cell's colours have been
   shifted by a full-screen effect; voting on it locks a misclassification permanently.
4. **Outside the true footprint** -- skip. The rectified patch is a rectangle but the world in it is
   a trapezoid (Phase C: the camera is perspective). Cells near that boundary are interpolated from
   the border fill and classify as garbage. `plan.valid` is the real footprint; the bounding
   rectangle is not.

A fifth gate, **loot-box occlusion**, is specified by the plan and cannot be implemented here: no
box detector exists until the entity chunk does. `update()` takes an `occluded` mask so the wiring
is ready, and a box sitting on floor is not evidence about floor until something fills it in.

**Sub-tile position is never discarded.** Phase F integrates in floating point and this stage
rounds only at deposit time, so a run of 0.3-tile steps accumulates to 3 tiles after ten frames.
Accumulating *rounded* increments instead would round each 0.3 to 0 and the map would never move --
that is the specific mechanism by which a plausible odometry error becomes a visibly sheared map,
and `tests/test_vision_occupancy.py` pins it.

**Never-observed cells stay UNKNOWN, and that is correct fog-of-war, not a gap to fill.** Whatever
consumes this map must treat UNKNOWN as its own state: a policy that believes unexplored ground is
walkable will walk into walls.

**Known gap, deliberately deferred: destructible terrain.** "Classify once, cache forever" goes
stale the moment a locked wall is destroyed. The fix is either low-frequency re-validation of
locked cells or triggering reclassification near a detected destruction event, and both want the
entity chunk. Nothing here attempts it; a destroyed wall stays a wall in the map for the rest of
the match.
"""
from dataclasses import dataclass, field

import numpy as np

from brawl_sim.constants import TILE_TO_CHAR

from ..camera import RectifyPlan
from ..config import VisionConfig
from .labeling import CLASSES

# A never-observed cell. Deliberately NOT an eighth `brawl_sim.constants.Tile` member: `N_TILES`
# indexes the simulator's `_bool_table`, its `ListedColormap`, and the obs-schema tables, and
# widening that enum to carry a PERCEPTION state would push a vision concern into the simulator's
# physical vocabulary. -1 also keeps it outside any `range(N_TILES)` loop by construction.
UNKNOWN = -1

# Minimum share of a cell that must be real world before that cell may be voted on at all. Matches
# the reasoning in zone.py: a cell mostly outside the viewport is unknown, not evidence.
_MIN_FOOTPRINT = 0.9


@dataclass(frozen=True)
class DepositResult:
    """What one frame actually contributed. Returned rather than logged so a caller can watch the
    map fill in -- and so a run that deposits nothing for a hundred frames is visible instead of
    looking like a slow classifier."""
    voted: int
    locked_now: int
    skipped_gas: int
    skipped_locked: int
    out_of_bounds: int
    status: str                 # "ok" | "uncertain" | "lost" | "reset"
    segment: int


@dataclass
class OccupancyMap:
    """A persistent world-frame grid of terrain votes for one match."""
    height: int
    width: int
    n_classes: int = len(CLASSES)
    votes: np.ndarray = field(default=None)
    locked: np.ndarray = field(default=None)
    segment: int = 0
    segments_seen: int = 1

    def __post_init__(self):
        if self.votes is None:
            self.votes = np.zeros((self.height, self.width, self.n_classes), np.int32)
        if self.locked is None:
            self.locked = np.zeros((self.height, self.width), bool)

    @classmethod
    def from_config(cls, cfg: VisionConfig | None = None) -> "OccupancyMap":
        cfg = cfg or VisionConfig()
        return cls(height=cfg.occupancy_grid_h, width=cfg.occupancy_grid_w)

    # -- reading -------------------------------------------------------------

    @property
    def origin(self) -> tuple[int, int]:
        """World tile (x, y) sitting at grid index [0, 0].

        The grid is centred on wherever tracking started, because there is no pre-built map to
        align to and the camera can walk off in any direction from that point.
        """
        return (-(self.width // 2), -(self.height // 2))

    @property
    def observed(self) -> np.ndarray:
        return self.votes.sum(axis=2) > 0

    def best(self) -> np.ndarray:
        """(h, w) int8 of indices into `CLASSES`, `UNKNOWN` where never observed."""
        out = np.full((self.height, self.width), UNKNOWN, np.int8)
        seen = self.observed
        out[seen] = self.votes[seen].argmax(axis=1).astype(np.int8)
        return out

    def confidence(self) -> np.ndarray:
        """Winning class's share of each cell's votes; 0 where unobserved."""
        total = self.votes.sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total > 0, self.votes.max(axis=2) / np.maximum(total, 1), 0.0)

    def to_chars(self) -> np.ndarray:
        """(h, w) of map-CSV legend characters, `?` where UNKNOWN -- the form a hand-verified
        `.grid.csv` is written in, so the two can be diffed directly."""
        best = self.best()
        out = np.full(best.shape, "?", dtype="<U1")
        for i, tile in enumerate(CLASSES):
            out[best == i] = TILE_TO_CHAR[tile]
        return out

    def observed_bounds(self) -> tuple[int, int, int, int] | None:
        """(r0, r1, c0, c1) of the observed region, or None. For cropping a 128x128 grid down to
        the part a match actually touched before showing or comparing it."""
        rows, cols = np.where(self.observed)
        if not len(rows):
            return None
        return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1

    # -- writing -------------------------------------------------------------

    def reset(self, segment: int) -> None:
        """Drop the map and start a new world frame.

        Called when odometry opens a new segment: there is no defined offset between the old frame
        and the new one, so the accumulated grid cannot be carried across. Recovering it would be
        loop closure, which does not exist yet -- and keeping it anyway would silently overlay two
        unrelated maps, which is far worse than losing one.
        """
        self.votes[:] = 0
        self.locked[:] = False
        self.segment = segment
        self.segments_seen += 1

    def update(self, cells: np.ndarray, odometry, plan: RectifyPlan,
               zone: np.ndarray | None = None, occluded: np.ndarray | None = None,
               cfg: VisionConfig | None = None) -> DepositResult:
        """Deposit one classified frame.

        `cells` is the classifier's (rows, cols) grid of class indices in the rectified patch's
        camera-relative frame; `odometry` is Phase F's `OdometryResult` for the same frame.
        """
        cfg = cfg or VisionConfig()
        if odometry.segment != self.segment:
            self.reset(odometry.segment)
            return self._empty("reset")
        if odometry.status != "ok":
            # "lost" has no world frame; "uncertain" has one but the frame is not trustworthy
            # evidence. Both deposit nothing; only "lost" reaches the branch above, next frame.
            return self._empty(odometry.status)

        cols, rows = plan.size_tiles
        if cells.shape != (rows, cols):
            raise ValueError(
                f"classified grid is {cells.shape} but the rectification grid is {(rows, cols)}"
            )

        allow = self._footprint(plan)
        if zone is not None:
            allow &= ~zone
        if occluded is not None:
            allow &= ~occluded
        skipped_gas = int((zone & self._footprint(plan)).sum()) if zone is not None else 0

        # Round ONLY here: `odometry.position_tiles` stays continuous, so sub-tile motion
        # accumulates instead of being discarded a fraction at a time.
        px, py = odometry.position_tiles
        ox, oy = self.origin
        col0 = int(round(plan.origin_tile[0] + px)) - ox
        row0 = int(round(plan.origin_tile[1] + py)) - oy

        r_lo, r_hi = max(row0, 0), min(row0 + rows, self.height)
        c_lo, c_hi = max(col0, 0), min(col0 + cols, self.width)
        oob = rows * cols - max(r_hi - r_lo, 0) * max(c_hi - c_lo, 0)
        if r_hi <= r_lo or c_hi <= c_lo:
            return DepositResult(0, 0, skipped_gas, 0, oob, "ok", self.segment)

        sub = (slice(r_lo - row0, r_hi - row0), slice(c_lo - col0, c_hi - col0))
        dst = (slice(r_lo, r_hi), slice(c_lo, c_hi))
        allow_w = allow[sub]
        locked_w = self.locked[dst]
        skipped_locked = int((allow_w & locked_w).sum())
        target = allow_w & ~locked_w
        if not target.any():
            return DepositResult(0, 0, skipped_gas, skipped_locked, oob, "ok", self.segment)

        rr, cc = np.where(target)
        classes = cells[sub][target].astype(np.int64)
        np.add.at(self.votes[dst], (rr, cc, classes), 1)

        # Lock only cells this frame touched: a cell cannot cross the threshold without a new vote.
        v = self.votes[dst][rr, cc]
        total = v.sum(axis=1)
        share = v.max(axis=1) / np.maximum(total, 1)
        newly = (total >= cfg.occupancy_min_votes) & (share >= cfg.occupancy_lock_ratio)
        if newly.any():
            self.locked[dst][rr[newly], cc[newly]] = True
        return DepositResult(int(target.sum()), int(newly.sum()), skipped_gas, skipped_locked,
                             oob, "ok", self.segment)

    # -- helpers -------------------------------------------------------------

    def _footprint(self, plan: RectifyPlan) -> np.ndarray:
        """Cells substantially inside the patch's TRUE footprint -- the trapezoid, not the bounding
        rectangle. Cells straddling that edge are interpolated from border fill."""
        cols, rows = plan.size_tiles
        ppt = plan.pixels_per_tile
        return plan.valid.reshape(rows, ppt, cols, ppt).mean(axis=(1, 3)) >= _MIN_FOOTPRINT

    def _empty(self, status: str) -> DepositResult:
        return DepositResult(0, 0, 0, 0, 0, status, self.segment)


def compare_to_truth(predicted: np.ndarray, truth: np.ndarray, max_shift: int = 6,
                     min_cells: int = 12) -> dict:
    """Score an accumulated grid against a hand-verified one, separating the two error types.

    **Position error and classification error have completely different causes** -- Phase F versus
    Phase H -- and collapsing them into one accuracy makes a drift bug read as a classifier bug.
    So this searches integer offsets, reports the offset that fits best, and reports accuracy both
    there and at zero offset. A large `best_offset` with high accuracy is odometry drift; zero
    offset with low accuracy is the classifier.

    Both grids are character grids in the map-CSV legend; `?` is ignored on either side.

    Two details that a naive version gets wrong, both found by testing it against identical grids:
    the shift must NOT wrap (`np.roll` lets a large offset wrap around to the identity and score a
    perfect match), and ties must resolve to the SMALLEST offset (otherwise the search order
    decides, and identical grids get reported as displaced). `min_cells` stops a sliver of overlap
    at a big offset from beating the real alignment on a handful of lucky cells.
    """
    def overlap(dr, dc):
        h, w = predicted.shape
        pr0, tr0 = max(dr, 0), max(-dr, 0)
        pc0, tc0 = max(dc, 0), max(-dc, 0)
        rows, cols = h - abs(dr), w - abs(dc)
        if rows <= 0 or cols <= 0:
            return None, None
        return (predicted[pr0:pr0 + rows, pc0:pc0 + cols],
                truth[tr0:tr0 + rows, tc0:tc0 + cols])

    def score(dr, dc):
        a, b = overlap(dr, dc)
        if a is None:
            return None
        sel = (a != "?") & (b != "?")
        n = int(sel.sum())
        if n < min_cells:
            return None
        return float((a[sel] == b[sel]).mean()), n

    offsets = sorted(((dr, dc) for dr in range(-max_shift, max_shift + 1)
                      for dc in range(-max_shift, max_shift + 1)),
                     key=lambda o: (abs(o[0]) + abs(o[1]), abs(o[0]), abs(o[1])))
    best = None
    at_zero = score(0, 0)
    for dr, dc in offsets:
        got = score(dr, dc)
        if got is None:
            continue
        # Strictly greater, over offsets sorted by magnitude: ties keep the smaller shift.
        if best is None or got[0] > best[1]:
            best = ((dr, dc), got[0], got[1])
    if best is None:
        return {"best_offset": None, "accuracy_at_best_offset": float("nan"),
                "cells_compared": 0, "accuracy_at_zero_offset": float("nan"),
                "position_error_tiles": float("nan")}
    return {
        "best_offset": best[0],
        "accuracy_at_best_offset": best[1],
        "cells_compared": best[2],
        "accuracy_at_zero_offset": at_zero[0] if at_zero else float("nan"),
        "position_error_tiles": float(np.hypot(*best[0])),
    }
