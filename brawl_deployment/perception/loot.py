"""Crate and power-cube boxes -> the grid's `box` and `pickup` channels. BRAWL_DEPLOYMENT_DESIGN.md 6.2.

The projectile model boxes three classes (`projectile_detection/classes.py`). `projectiles.py` keeps
the `Projectile` ones. This module keeps the other two, and the loop hands both the same list.

#### What the sim puts in those channels, and what it leaves out

`observation._build_grid` scatters `box_pos` where `box_alive` into `box`, and `pku_pos` where
`pku_alive` into `pickup`. Both are counts per cell: one intact crate is 1 in its cell, one pickup
object is 1 in its cell. That is all. Box HP is tracked by the sim, because a crate breaks when its
HP reaches 0, but no observation spec reads it, and nothing here reads the HP number drawn over a
damaged crate either. Neither channel is gated by `fair`: they cover the whole crop, including
ground the hero has seen before but cannot see now. So both lists here are STICKY. An object stays
until the camera is looking at its spot and it is not there.

#### Where a box is on the ground: the two anchors

`to_tiles` projects one point of each box through the ground homography, which is only right for a
point that is on the ground. Which point that is differs per class, and both were measured:

  * **Crate: `CRATE_ANCHOR_FRAC = 0.30`**, the brawler default, and it holds for a different
    reason. On the calibration clip (the one piece of footage whose world lattice is the true tile
    lattice) a crate at tile (-1, 3) projects to (-0.48, 3.53) against the true centre
    (-0.50, 3.50), sd 0.03 tiles over 272 frames. The HP number over a damaged crate is inside the
    box and moves the anchor about 0.07 tiles.
  * **Cube: `CUBE_ANCHOR_FRAC = -0.15`, BELOW the box.** A dropped cube floats and spins, and the
    box holds only the cube. Its ground point is the blob shadow under it, a median 10 px below the
    box bottom (5-17 px per cube, read at 4x on 7 cubes over clean floor) against a median cube box
    of 65 px. A negative fraction is just that: `Detection.anchor` puts it under the bottom edge.
    It moves the cube about 0.15 tiles, against a pickup radius of 0.6.

#### One fixed cell per object

The sim's crate sits on a tile centre, forever. Here each object keeps its observed world
positions (the last `MAX_SAMPLES`) and, once it has `MIN_HITS` of them, fixes its cell at the
median. That cell is what the grid gets from then on. The median keeps associating (the gate is
measured from it, not from the cell centre), but the cell does not move. Two reasons to fix it:

  * The world lattice is not the game's. Each odometry segment starts its lattice at whatever
    sub-tile phase the camera had, so a crate's true tile can straddle two world cells, and a live
    median near a boundary would flip between them tick to tick. Correcting the phase is out of
    scope; not amplifying it into flicker is not.
  * It agreed with the terrain map, which locked its cells for good. That map keeps re-voting every
    cell in view since 2026-09-14, so the first reason now stands on its own.

Odometry drift inside a segment can carry a crate's observations away from its fixed cell. If they
drift past the gate, the next sighting opens a new object and the old one is retired by the rule
below, because it is in view and not being seen. So drift is repaired, not accumulated.

#### Duplicates are merged: measured

The detector runs without NMS (`projectiles._dedupe` explains why) and does box one crate twice.
Simultaneous same-class pairs on the 27-clip replay (12 Hz, the shipped model), in tiles after
projection:

    crates   IoU > 0.5   n 594   separation max 0.151      IoU <= 0.3   min 0.663 (adjacent crates)
    cubes    IoU > 0.5   n 349   separation max 0.205      IoU <= 0.3   min 0.390 (a death pile)

`DUPLICATE_TILES = 0.3` is between the two on both classes, and it is the projectile tracker's
number too. An unmerged duplicate would open a second object 0.01 tiles from the first.

#### Boxes near the frame edge are dropped: measured

A box cut by the frame edge has its centre and its height wrong, so its anchor is wrong. 17% of
crate boxes touch an edge. Along the bottom, boxes ending in rows 1100-1123 (of 1126) read a
median -0.23 tiles off (p10 -0.67) and are visibly shorter; above row 1100 they stay within
+-0.1. `EDGE_MARGIN_PX = 27` is the smallest margin that drops every box ending in that band
(1126 - 27 = 1099), and it is applied to ANY edge: the other three were not measured, and the
failure is the same geometry. Nothing is lost for it: the map is sticky, and the crate was seen
further in on the way to the edge.

#### When an object is gone

A crate is broken, or a cube collected, when the camera is looking at its spot and the detector
does not find it. "Looking at" is the same test as the edge rule, applied to where the object's
last box would be now (`_in_view`), plus the ground point not being under the HUD. Out of view,
nothing counts against it. In view, unseen time accumulates and a detection clears it. `GONE_S`
is how much is allowed, per class, sized off the replay's gaps between detections of one object
with both ends well inside the frame:

    crates   11,659 gaps   > 0.5 s: 31   > 1.0 s: 5    max 1.92 s
    cubes     2,420 gaps   > 0.5 s: 28   > 1.0 s: 13   max 2.87 s    object life p50 0.6 s

Crates get 1.0 s: five false retirements in 26 minutes, each costing a crate that blinks out until
it is re-confirmed a quarter of a second later. Cubes get 0.5 s, because a cube's whole life is
usually shorter than a second, and a collected cube left on the map for a full second would be a
pickup the policy walks to and finds nothing.

#### Replayed

This module, unchanged, over the same replay (18,926 ticks, 75 odometry segments):

    crates   145 confirmed   98 retired    7 back within a cell inside 10 s (5 of them within 1 s)
    cubes    202 confirmed  179 retired   21 back within a cell inside 10 s

Confirmation takes a median 0.17 s, the third sighting. `update` plus `crate_occlusion` cost
1.3 ms a tick (p99 1.8). The crates' five quick returns are the five gaps over 1 s in the table
above, as predicted. The cubes' 21 are the price of 0.5 s: at 0.75 s it is 15, at 1.0 s 13, while
the average number of cubes on the map rises from 0.39 to 0.50 because every collected one lingers
longer. Some returns are not blinks at all: a cube that moves (popping out of its crate, flying to
a brawler) leaves the gate, becomes a new object, and the old one is retired half a second later.

#### The terrain map's fifth gate

`OccupancyMap.update` takes an `occluded` mask that nothing filled until now (the terrain plan:
"a cell visually occluded by a box is unobserved"). A crate stands for tens of seconds and every
tick of that is a vote, so an unmasked crate wins its cell as a misclassification. `crate_occlusion`
warps each crate box through the same homography the frame goes through, so the quad covers exactly
the rectified pixels the crate sprite landed on, and masks every cell more than
`1 - occupancy._MIN_FOOTPRINT` covered. That is the footprint gate's own tolerance for pixels that
are not ground. Every crate box counts, edge ones included, and before merging: a clipped box is a
bad position but its pixels are still crate.

How much that masks, on the replay: one crate box covers 5-11 cells (p10-p90, never more than
12) of the patch's 570, and a tick with any crate on screen masks a median 11 cells, 26 at p90.
The sprite is 2.7 tiles tall on screen, so most of that is ground BEHIND the crate. Those cells
stay UNKNOWN while the crate stands (the camera only translates, so nothing reveals them) and the
grid reads UNKNOWN as FLOOR. For the crate's own cell that is right: a box spot is floor in the
sim. For the cells behind it, it is the grid's usual UNKNOWN trade (`grid.py`).

#### What is deliberately NOT here

  * **Box HP.** Out of scope for this stage, and no spec asks for it.
  * **The lattice phase**, see above.
  * **Boxes cut by the HUD.** The edge rule is for the frame edge, which was measured. A crate
    half under the joystick is clipped the same way and is not caught.
  * **Collection by proximity.** The sim removes a pickup the tick a brawler comes within
    `pickup_radius`. Here a collected cube stays until `GONE_S` of not seeing it, including one the
    hero just walked over.
  * **Death piles.** The sim drops ONE pickup per corpse. On the replay, 168 times a tick showed
    2-4 cubes within two tiles of each other, against 2,728 lone cubes, which looks like a corpse
    dropping several. Each is a separate object here, as it is on screen.
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from brawl_vision.object_detection.project import (CRATE_ANCHOR_FRAC, CUBE_ANCHOR_FRAC,
                                                   LOOT_ANCHOR_FRAC, to_tiles)
from brawl_vision.object_detection.projectile_detection.classes import CUBE_BOX, CUBE_DROPPED
from brawl_vision.terrain.occupancy import _MIN_FOOTPRINT

from .tracker import _greedy_match

__all__ = ["Loot", "LootResult", "LootMap", "box_occlusion", "crate_occlusion",
           "require_loot_classes",
           "CRATE_ANCHOR_FRAC", "CUBE_ANCHOR_FRAC", "EDGE_MARGIN_PX", "DUPLICATE_TILES",
           "GATE_TILES", "MIN_HITS", "GONE_S", "MAX_SAMPLES"]

# The two anchors, measured as the module docstring says. Defined in brawl_vision's project.py
# (crate 0.30, cube -0.15, below its box) so the render projects crates and cubes the same way.
ANCHOR_FRAC = LOOT_ANCHOR_FRAC

# Boxes closer than this to any frame edge are dropped. Measured on the bottom edge: the biased
# band starts at row 1100 of 1126.
EDGE_MARGIN_PX = 27

# Merge radius in tiles, between the widest duplicate (0.205) and the closest distinct pair (0.39).
DUPLICATE_TILES = 0.3

# Association radius in tiles from an object's median. Crates are the tight case: adjacent crates
# project 0.66+ apart, and one crate's samples agree to 0.06 tiles median.
GATE_TILES = 0.5

# Sightings before an object reaches the grid, a quarter of a second at the 12 Hz perception rate.
# One sighting could be a false positive, and fixing a cell on it would fix it for good.
MIN_HITS = 3

# In-view unseen seconds before an object is retired. Per class; see the module docstring.
GONE_S = {CUBE_BOX: 1.0, CUBE_DROPPED: 0.5}

# Samples kept per object for its median. Three seconds at 12 Hz is plenty for a stationary thing.
MAX_SAMPLES = 32


def require_loot_classes(names, channels) -> None:
    """Raise if the grid wants `box`/`pickup` and the projectile model has no class for it.

    The same failure `projectiles.require_projectile_class` stops: a model without the class loads
    cleanly, the label filter drops nothing because there is nothing to drop, and the policy reads
    an empty channel where it was trained to read crates.
    """
    labels = set(names.values()) if isinstance(names, dict) else set(names)
    wanted = {"box": CUBE_BOX, "pickup": CUBE_DROPPED}
    missing = [(ch, wanted[ch]) for ch in channels if ch in wanted and wanted[ch] not in labels]
    if missing:
        raise ValueError(
            f"the grid reads {[ch for ch, _ in missing]}, but the projectile model has no "
            f"{[lab for _, lab in missing]} class (it has {sorted(labels)}). Check "
            f"`projectile.model` in configs/vision.yaml against the checkpoint's agent_obs spec.")


def _merge(placed, threshold: float):
    """`projectiles._dedupe`'s rule, keeping the box: of any boxes closer than `threshold` tiles
    after projection, only the most confident survives. `placed` is `[((x, y), Detection)]`."""
    keep = []
    for p, d in sorted(placed, key=lambda pd: -pd[1].confidence):
        if any(math.hypot(p[0] - q[0], p[1] - q[1]) < threshold for q, _ in keep):
            continue
        keep.append((p, d))
    return keep


def _clear_of_edges(xyxy, viewport, margin: float) -> bool:
    x0, y0, x1, y1 = xyxy
    w, h = viewport
    return x0 >= margin and y0 >= margin and x1 <= w - margin and y1 <= h - margin


def crate_occlusion(detections, plan, max_fraction: float = 1.0 - _MIN_FOOTPRINT) -> np.ndarray:
    """`(rows, cols)` bool in the rectified patch's cell grid: cells the crates on screen cover.

    For `OccupancyMap.update(occluded=...)`. See "The terrain map's fifth gate" in the module
    docstring. All zeros when there is no crate, which is also what `occluded=None` means.
    """
    return box_occlusion([d for d in detections if d.label == CUBE_BOX], plan, max_fraction)


def box_occlusion(detections, plan, max_fraction: float = 1.0 - _MIN_FOOTPRINT) -> np.ndarray:
    """`crate_occlusion` for EVERY box it is given, whatever the label.

    The loop feeds it the entity model's boxes, all three of whose classes are brawlers. A brawler
    sprite is no more evidence about the ground under it than a crate is. On the label frames
    (2026-09-14), about half the map's false walls had no true wall within a cell, and sprites and
    uncovered HUD were what sat there. The hero is the bad case: a hero that stops holds its
    sprite over the same world cells, tick after tick, for as long as it stands there.
    """
    cols, rows = plan.size_tiles
    ppt = plan.pixels_per_tile
    if not detections:
        return np.zeros((rows, cols), bool)
    mask = np.zeros((rows * ppt, cols * ppt), np.float32)
    for d in detections:
        x0, y0, x1, y1 = d.xyxy
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64).reshape(-1, 1, 2)
        quad = cv2.perspectiveTransform(corners, plan.M).reshape(-1, 2)
        if not np.isfinite(quad).all():
            continue
        cv2.fillConvexPoly(mask, np.clip(quad, -1e4, 1e4).round().astype(np.int32), 1.0)
    # INTER_AREA at a whole-number scale is the exact block mean: each cell's covered fraction.
    return cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_AREA) > max_fraction


@dataclass
class Loot:
    """One crate or cube, in WORLD tiles of the current odometry segment."""
    id: int
    label: str
    samples: list = field(default_factory=list)   # recent world positions, oldest first
    hits: int = 0
    t: float = 0.0                                 # last sighting
    unseen_s: float = 0.0                          # in-view seconds since the last sighting
    cell: tuple[int, int] | None = None            # fixed at MIN_HITS
    box_px: tuple[float, float] = (0.0, 0.0)       # last box (w, h), for the in-view test

    @property
    def pos(self) -> tuple[float, float]:
        """Median of the recent samples: robust to the odd bad anchor, and what association uses."""
        a = np.asarray(self.samples, np.float64)
        return (float(np.median(a[:, 0])), float(np.median(a[:, 1])))

    @property
    def confirmed(self) -> bool:
        return self.cell is not None

    @property
    def centre(self) -> tuple[float, float]:
        """The fixed cell's centre, which is where the sim keeps a crate: `box_pos` is a tile
        centre, so `floor` in `_scatter_count` lands on one cell with no rounding question."""
        return (self.cell[0] + 0.5, self.cell[1] + 0.5)


@dataclass(frozen=True)
class LootResult:
    """One perception tick's outcome, for logging."""
    status: str                # "ok" | "reset" | odometry's own status when unusable
    segment: int
    n_boxes: int               # crate and cube boxes this tick, before the edge rule
    n_edge: int                # dropped by the edge rule
    n_merged: int              # dropped as duplicates
    n_retired: int             # objects retired this tick


class LootMap:
    """Feed it every perception tick's detections; read `crates()` and `cubes()`. One per match."""

    def __init__(self, *, edge_margin_px: float = EDGE_MARGIN_PX,
                 duplicate_tiles: float = DUPLICATE_TILES, gate_tiles: float = GATE_TILES,
                 min_hits: int = MIN_HITS, gone_s: dict | None = None,
                 max_samples: int = MAX_SAMPLES):
        self.edge_margin_px = edge_margin_px
        self.duplicate_tiles = duplicate_tiles
        self.gate_tiles = gate_tiles
        self.min_hits = min_hits
        self.gone_s = dict(GONE_S if gone_s is None else gone_s)
        self.max_samples = max_samples
        self.objects: list[Loot] = []
        self._next_id = 0
        # Time of the last tick that did its accounting; None after an unusable one, so a stretch
        # of bad odometry is not counted as seconds the camera spent looking.
        self._t_last: float | None = None
        # None, not -1: the first tick adopts odometry's segment. Same as every tracker here.
        self._segment: int | None = None

    # -- reading -------------------------------------------------------------

    def crates(self) -> list[tuple[float, float]]:
        return [o.centre for o in self.objects if o.label == CUBE_BOX and o.confirmed]

    def cubes(self) -> list[tuple[float, float]]:
        return [o.centre for o in self.objects if o.label == CUBE_DROPPED and o.confirmed]

    # -- the tick ------------------------------------------------------------

    def update(self, detections, plan, odometry, t: float) -> LootResult:
        """`detections` in viewport pixels, all classes; anything but a crate or a cube is ignored.

        Gated on odometry like the trackers: a new segment drops every object (its world frame has
        no defined offset to the old one), and a tick that is not `"ok"` changes nothing.
        """
        boxes = [d for d in detections if d.label in ANCHOR_FRAC]
        if self._segment is None:
            self._segment = odometry.segment
        elif odometry.segment != self._segment:
            self.reset(odometry.segment)
            return LootResult("reset", odometry.segment, len(boxes), 0, 0, 0)
        if odometry.status != "ok":
            self._t_last = None
            return LootResult(odometry.status, self._segment, len(boxes), 0, 0, 0)

        dt = 0.0 if self._t_last is None else max(0.0, t - self._t_last)
        self._t_last = t
        kept = [d for d in boxes if _clear_of_edges(d.xyxy, plan.viewport, self.edge_margin_px)]
        px, py = odometry.position_tiles
        n_merged = n_retired = 0
        for label, frac in ANCHOR_FRAC.items():
            mine = [d for d in kept if d.label == label]
            placed = [((tx + px, ty + py), d) for d, (tx, ty) in
                      to_tiles(mine, plan, anchor_frac=frac)]
            points = _merge(placed, self.duplicate_tiles)
            n_merged += len(placed) - len(points)
            n_retired += self._associate(label, points, plan, odometry, t, dt)
        return LootResult("ok", self._segment, len(boxes), len(boxes) - len(kept), n_merged,
                          n_retired)

    def _associate(self, label, points, plan, odometry, t: float, dt: float) -> int:
        objs = [o for o in self.objects if o.label == label]
        matched, fresh = _greedy_match([o.pos for o in objs], [p for p, _ in points],
                                       self.gate_tiles)
        for oi, pi in matched.items():
            self._sighting(objs[oi], *points[pi], t)
        for pi in fresh:
            obj = Loot(id=self._take_id(), label=label)
            self._sighting(obj, *points[pi], t)
            self.objects.append(obj)

        retired = set()
        for oi, obj in enumerate(objs):
            if oi in matched:
                continue
            if not obj.confirmed:
                # Never made it to the grid, so there is no in-view question to ask: drop it once
                # it has gone as long unseen as a confirmed one would be allowed to.
                if t - obj.t > self.gone_s[label]:
                    retired.add(obj.id)
                continue
            if self._in_view(obj, plan, odometry):
                obj.unseen_s += dt
                if obj.unseen_s > self.gone_s[label]:
                    retired.add(obj.id)
        self.objects = [o for o in self.objects if o.id not in retired]
        return len(retired)

    def _sighting(self, obj: Loot, pos, det, t: float) -> None:
        obj.samples.append(pos)
        del obj.samples[:-self.max_samples]
        obj.hits += 1
        obj.t = t
        obj.unseen_s = 0.0
        x0, y0, x1, y1 = det.xyxy
        obj.box_px = (x1 - x0, y1 - y0)
        if obj.cell is None and obj.hits >= self.min_hits:
            mx, my = obj.pos
            obj.cell = (math.floor(mx), math.floor(my))

    def _in_view(self, obj: Loot, plan, odometry) -> bool:
        """Would a box for this object, here and now, pass the edge rule and sit on real ground?

        The object's last box, re-centred on where its anchor projects this tick. The same test
        the detections go through, so an object is never charged for being missed in a place where
        a sighting would have been thrown away. Plus the anchor's rectified pixel must be `valid`,
        which excludes the HUD.
        """
        px, py = odometry.position_tiles
        cam = np.array([[obj.pos[0] - px, obj.pos[1] - py]], np.float64)
        rx, ry = plan.tile_to_rect(cam)[0]
        h_px, w_px = plan.valid.shape
        if not (0 <= rx < w_px and 0 <= ry < h_px) or not plan.valid[int(ry), int(rx)]:
            return False
        ax, ay = cv2.perspectiveTransform(np.array([[[rx, ry]]], np.float64), plan.M_inv)[0, 0]
        w, h = obj.box_px
        frac = ANCHOR_FRAC[obj.label]
        y1 = ay + frac * h            # `Detection.anchor` inverted: anchor = y1 - frac * h
        box = (ax - w / 2, y1 - h, ax + w / 2, y1)
        return _clear_of_edges(box, plan.viewport, self.edge_margin_px)

    def _take_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def reset(self, segment: int) -> None:
        self.objects.clear()
        self._t_last = None
        self._segment = segment
