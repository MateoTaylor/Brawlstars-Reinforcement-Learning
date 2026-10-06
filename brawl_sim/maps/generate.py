"""Procedural Solo Showdown maps. Library half; `scripts/gen_maps.py` is the CLI and
`tests/test_map_generate.py` pins every primitive.

Not a hot path: plain Python + numpy on a 60x60 char grid, run once per map at authoring time.
Deterministic in `(seed, family, symmetry)` -- every random draw comes from one
`random.Random(seed)`, sets are never iterated for sampling, and numpy is used only for counting.

Pipeline (`generate`):

1. border ring of `#`; for the water-border family a 2-tile `~` moat `Family.moat_inset` tiles in,
   with a floor rim outside it and four 3-tile gaps (two drawn, two mirrored, off-centre);
2. centre feature on point-symmetric maps (2x2 wall + 4 bush, a small pond, or an open cross kept
   clear of everything else); the five screenshot families run their `LAYOUTS` pass here
   instead, which lays the reference map's skeleton (maze rings, a lake ring, vines);
3. stamps drawn in one half of the map and written together with their mirror
   (`mirror_point` for 180-degree symmetry, `mirror_lr` for left-right): wall stamps, ponds and
   channels until the family's wall and water shares are met (water first on the screenshot
   families), then bush patches until the bush share is met -- solid stamps keep a 2-cell margin
   from each other and the border so no 1-wide corridor can exist, bush is allowed to touch
   anything and merge into fields;
4. repair: punch a 2-tile gap in any straight interior wall run longer than 8, fill 1-wide dead
   ends with whatever blocks them, carve the shortest path between the two largest passable
   components until there is one;
5. spawns: 16 `S`, 8 drawn on a square ring ~7 tiles inside the border at even angles and mirrored,
   each nudged to the nearest floor cell; crates: `X` singles and pairs next to cover, mirrored;
6. `check`: `loader.validate_map`, the family's bands, the pocket check, bush spread and the bush
   waypoint cap. A seed that fails any of them is rejected with the reason and the next seed is
   tried, so the seed `generate` reports regenerates the same map on its own.

Grid coordinates are `(row, col)` = `(y, x)`, matching the CSV and `origin="upper"` rendering.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from ..constants import MAP_CHAR_TO_TILE
from . import loader

SIZE = 60
FLOOR, WALL, BUSH, WATER, SPAWN, BOX = ".", "#", "b", "~", "S", "X"
PASSABLE = frozenset((FLOOR, BUSH, SPAWN, BOX))
SOLID = frozenset((WALL, WATER))
POINT, MIRROR = "point", "mirror"

N_SPAWNS = 16
SPAWN_INSET = 7          # the square ring the spawns sit on, tiles in from the border (+-1 jitter)
MAX_WALL_RUN = 8         # README: "no long solid barriers"
MOAT_GAP = 3
POCKET_RADIUS = 6        # a BFS of this radius from every floor cell ...
# ... must reach >= 24 of the 85 cells an open field offers (0.28), scaled by what is in bounds
# so a corner cell is judged against its 28. 0.28 passes a 2-wide lane between two features
# (reach 26; the reference maps are full of them) and rejects what the README means by a
# pocket: a 2x3 notch reaches ~20 (0.24), a dead-end lane ~15 (0.18).
POCKET_MIN_REACH = 24
POCKET_OPEN_FIELD = 85
BUSH_SPREAD_MIN = 0.80   # a bush cell in >= 80% of the hunt cells
HUNT_CELL_TILES = 10     # mirrors configs/default.yaml bots.hunt_cell_tiles
STAMP_MARGIN = 2         # cells of floor kept around every solid stamp (and from the border)


# ---- families -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    """Style bands: interior shares (rows/cols 1..58, so 3364 cells; the moat counts as water)
    and crate counts. `count` is how many shipped maps come from this family (the ten generated
    maps, then three per screenshot family); `symmetry` is the default and
    `generate(..., symmetry=)` overrides it (the standard family ships five point-symmetric maps
    and one mirror map)."""
    name: str
    count: int
    wall: tuple[float, float]
    bush: tuple[float, float]
    water: tuple[float, float]
    boxes: tuple[int, int]
    symmetry: str = POINT
    moat_inset: int = 0  # 0 = no moat; otherwise the moat's outer edge is this many tiles in
    # For the screenshot families: `layout` names the LAYOUTS pass that lays the reference map's
    # skeleton in place of the centre feature, and `wall_mix` weights the WALL_STAMPS the wall
    # pass fills the band with. A one-entry mix draws nothing extra, which keeps the four
    # original families' shipped seeds byte-identical.
    layout: str = ""
    wall_mix: tuple[tuple[str, float], ...] = (("cluster", 1.0),)


FAMILIES: dict[str, Family] = {
    "standard": Family("standard", 6, (0.07, 0.12), (0.20, 0.30), (0.03, 0.08), (20, 32)),
    "open": Family("open", 1, (0.04, 0.06), (0.12, 0.18), (0.00, 0.03), (16, 24)),
    "dense": Family("dense", 1, (0.08, 0.12), (0.32, 0.38), (0.02, 0.05), (24, 36), symmetry=MIRROR),
    # The moat is drawn 4 tiles in: a 3-wide floor rim outside it, water at rows/cols 4-5, the
    # island from 6. Flush against the border the four gaps would lead nowhere; a rim
    # narrower than 3 fails the pocket check by construction. The moat alone is ~11% water.
    "water_border": Family("water_border", 2, (0.06, 0.10), (0.20, 0.28), (0.12, 0.20), (20, 32),
                           moat_inset=4),
    # One family per map transcribed from a screenshot (README "The five screenshot maps");
    # bands sit around the transcription's own shares. All five are mirror maps: these maps must
    # be symmetric across at least one axis, and a 180-degree turn has none. Most of
    # each layout is drawn in one quarter and reflected across both axes, and the rest is stamped
    # with the family's symmetry, so `--symmetry point` works too.
    "maze": Family("maze", 3, (0.10, 0.15), (0.08, 0.14), (0.01, 0.03), (22, 32), symmetry=MIRROR,
                   layout="rings", wall_mix=(("fence", 0.5), ("stub", 0.5))),
    "lake_ring": Family("lake_ring", 3, (0.07, 0.11), (0.13, 0.19), (0.06, 0.10), (16, 26),
                        symmetry=MIRROR, layout="lake_ring", wall_mix=(("stub", 0.5), ("cluster", 0.5))),
    "branches": Family("branches", 3, (0.10, 0.14), (0.07, 0.12), (0.01, 0.03), (20, 28),
                       symmetry=MIRROR, layout="branches", wall_mix=(("branch", 0.8), ("stub", 0.2))),
    "crescent": Family("crescent", 3, (0.07, 0.11), (0.10, 0.16), (0.07, 0.11), (18, 26),
                       symmetry=MIRROR, layout="crescent", wall_mix=(("stub", 0.6), ("cluster", 0.4))),
    "vines": Family("vines", 3, (0.06, 0.10), (0.17, 0.24), (0.02, 0.05), (20, 28), symmetry=MIRROR,
                    layout="vines", wall_mix=(("stub", 0.5), ("cluster", 0.5))),
}


# ---- grid primitives --------------------------------------------------------------------------


def new_grid(size: int = SIZE) -> np.ndarray:
    """A `size x size` char grid: `#` border, `.` interior."""
    grid = np.full((size, size), FLOOR, dtype="<U1")
    border(grid)
    return grid


def border(grid: np.ndarray) -> None:
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL


def mirror_point(cell: tuple[int, int], size: int = SIZE) -> tuple[int, int]:
    r, c = cell
    return size - 1 - r, size - 1 - c


def mirror_lr(cell: tuple[int, int], size: int = SIZE) -> tuple[int, int]:
    r, c = cell
    return r, size - 1 - c


def mirror_fn(symmetry: str):
    if symmetry == POINT:
        return mirror_point
    if symmetry == MIRROR:
        return mirror_lr
    raise ValueError(f"symmetry must be {POINT!r} or {MIRROR!r}, got {symmetry!r}")


def mirror_grid(grid: np.ndarray, symmetry: str) -> np.ndarray:
    """The grid seen through the symmetry -- equal to `grid` iff the map is symmetric."""
    return grid[::-1, ::-1] if symmetry == POINT else grid[:, ::-1]


def stamp(grid: np.ndarray, cells: dict[tuple[int, int], str], symmetry: str | None) -> None:
    """Writes `cells` ({(r, c): char}) and, with a symmetry, their mirror images. Clips to the
    interior so a stamp can never overwrite the border."""
    mirror = mirror_fn(symmetry) if symmetry else None
    size = grid.shape[0]
    for cell, ch in cells.items():
        for r, c in ((cell,) if mirror is None else (cell, mirror(cell))):
            if 0 < r < size - 1 and 0 < c < size - 1:
                grid[r, c] = ch


def add_moat(grid: np.ndarray, rng: random.Random, inset: int, symmetry: str = POINT) -> None:
    """A 2-tile `~` ring whose outer edge is `inset` tiles in from the border, with a 3-tile floor
    gap on each side: the top and left gaps are drawn off-centre and the bottom and right ones are
    their mirrors, so the crossings are never a symmetric funnel."""
    size = grid.shape[0]
    lo, hi = inset, size - 1 - inset
    grid[lo:lo + 2, lo:hi + 1] = WATER
    grid[hi - 1:hi + 1, lo:hi + 1] = WATER
    grid[lo:hi + 1, lo:lo + 2] = WATER
    grid[lo:hi + 1, hi - 1:hi + 1] = WATER
    # Gap starts are drawn from the first third and the last third of the side so the two drawn
    # gaps sit on different sides of the centre line.
    third = (hi - lo) // 3
    top = rng.randint(lo + 2, lo + third)
    left = rng.randint(hi - third, hi - 2 - MOAT_GAP)
    gaps = {}
    if symmetry == MIRROR:
        # Left-right symmetry maps the top side onto itself, so a drawn top gap would come back
        # as a second top gap and the bottom would get none: give the top and bottom a centred
        # 4-wide gap each (their own mirror image) and keep the drawn left gap for left/right.
        mid = size // 2
        for c in range(mid - 2, mid + 2):
            for r in (lo, lo + 1, hi - 1, hi):
                gaps[(r, c)] = FLOOR
    else:
        for c in range(top, top + MOAT_GAP):
            gaps[(lo, c)] = FLOOR
            gaps[(lo + 1, c)] = FLOOR
    for r in range(left, left + MOAT_GAP):
        gaps[(r, lo)] = FLOOR
        gaps[(r, lo + 1)] = FLOOR
    stamp(grid, gaps, symmetry)


# ---- stamps -----------------------------------------------------------------------------------
# Each returns {(r, c): char} relative to an anchor the caller chose; nothing is written here.


def wall_cluster(rng: random.Random, anchor: tuple[int, int]) -> dict[tuple[int, int], str]:
    """A 2x2..4x6 rectangle (either orientation), optionally an L (a 2-wide, 2-3 long arm off one
    end of a side at least 5 long, so the inside corner is >= 3 wide and never a notch -- not a T,
    whose stem leaves 2-wide notches that are pockets by the check below), and a 1-tile bush skirt
    on one or two random sides -- the "wall with grass attached" motif of every reference
    screenshot."""
    r0, c0 = anchor
    h, w = rng.randint(2, 4), rng.randint(2, 6)
    if rng.random() < 0.5:
        h, w = w, h
    cells = {(r, c): WALL for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
    if max(h, w) >= 5 and rng.random() < 0.4:
        arm = rng.randint(2, 3)
        if w >= h:  # arm hangs below one end of the bar
            start = c0 if rng.random() < 0.5 else c0 + w - 2
            for r in range(r0 + h, r0 + h + arm):
                for c in range(start, start + 2):
                    cells[(r, c)] = WALL
        else:       # arm sticks out to the right of one end
            start = r0 if rng.random() < 0.5 else r0 + h - 2
            for r in range(start, start + 2):
                for c in range(c0 + w, c0 + w + arm):
                    cells[(r, c)] = WALL
    walls = list(cells)  # the skirt hugs the wall only, never an earlier side's skirt
    for side in rng.sample(("N", "S", "E", "W"), rng.randint(1, 2)):
        for (r, c) in walls:
            nr, nc = {"N": (r - 1, c), "S": (r + 1, c), "E": (r, c + 1), "W": (r, c - 1)}[side]
            if (nr, nc) not in cells:
                cells[(nr, nc)] = BUSH
    return cells


def pond(rng: random.Random, centre: tuple[int, int], radius: int | None = None,
         rim: bool = True, full_rim: bool = False) -> dict[tuple[int, int], str]:
    """A filled ellipse of water with semi-axes 2..4 (`radius` pins both), and a 1-tile bush rim
    along a random half of its perimeter (all of it with `full_rim`)."""
    cr, cc = centre
    a = radius if radius is not None else rng.randint(2, 4)
    b = radius if radius is not None else rng.randint(2, 4)
    cells = {}
    for r in range(cr - a, cr + a + 1):
        for c in range(cc - b, cc + b + 1):
            if ((r - cr) / (a + 0.5)) ** 2 + ((c - cc) / (b + 0.5)) ** 2 <= 1.0:
                cells[(r, c)] = WATER
    if rim:
        arc0 = rng.uniform(0.0, 2 * np.pi)
        for (r, c) in list(cells):
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in cells:
                    continue
                angle = (np.arctan2(nr - cr, nc - cc) - arc0) % (2 * np.pi)
                if full_rim or angle <= np.pi:
                    cells[(nr, nc)] = BUSH
    return cells


def channel(rng: random.Random, start: tuple[int, int]) -> dict[tuple[int, int], str]:
    """A 1-2 wide water strip 6-12 long, straight or with one right-angle bend. (Not 14: a 14-long
    strip a lane away from the moat or another feature is right at the pocket check's edge, 12 is
    not.)"""
    r, c = start
    width = rng.randint(1, 2)
    length = rng.randint(6, 12)
    cells = {}
    horizontal = rng.random() < 0.5
    bend_at = rng.randint(3, length - 3) if rng.random() < 0.5 else None
    dr, dc = (0, 1) if horizontal else (1, 0)
    for i in range(length):
        if bend_at is not None and i == bend_at:
            dr, dc = (dc, dr)  # turn 90 degrees, keeping the same handedness
            horizontal = not horizontal
        for k in range(width):
            cells[(r + (k if horizontal else 0), c + (0 if horizontal else k))] = WATER
        r, c = r + dr, c + dc
    return cells


def bush_patch(rng: random.Random, centre: tuple[int, int], free, size: int | None = None) -> dict[tuple[int, int], str]:
    """A blob of 6-30 bush cells grown from `centre` over cells `free(cell)` accepts; returns what
    it managed to grow (fewer than asked if boxed in)."""
    target = size if size is not None else rng.randint(6, 30)
    if not free(centre):
        return {}
    blob = [centre]
    have = {centre}
    tries = 0
    while len(blob) < target and tries < target * 20:
        tries += 1
        r, c = rng.choice(blob)
        dr, dc = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        cell = (r + dr, c + dc)
        if cell not in have and free(cell):
            have.add(cell)
            blob.append(cell)
    return {cell: BUSH for cell in blob}


def crate_spot(rng: random.Random, cell: tuple[int, int], free) -> dict[tuple[int, int], str]:
    """One `X`, or a pair on adjacent cells when the neighbour is free."""
    cells = {cell: BOX}
    if rng.random() < 0.4:
        r, c = cell
        options = [n for n in ((r, c + 1), (r + 1, c), (r, c - 1), (r - 1, c)) if free(n)]
        if options:
            cells[rng.choice(options)] = BOX
    return cells


def centre_feature(rng: random.Random, size: int = SIZE) -> tuple[dict[tuple[int, int], str], set[tuple[int, int]]]:
    """For point-symmetric maps: cells to stamp at the centre and cells to keep clear of every
    other stamp. One of a 2x2 wall with four bush corners, a small pond, or an open cross."""
    m = size // 2  # 30: the centre is between cells m-1 and m
    kind = rng.choice(("wall", "pond", "cross"))
    if kind == "wall":
        cells = {(r, c): WALL for r in (m - 1, m) for c in (m - 1, m)}
        for r, c in ((m - 2, m - 2), (m - 2, m + 1), (m + 1, m - 2), (m + 1, m + 1)):
            cells[(r, c)] = BUSH
        return cells, set()
    if kind == "pond":
        cells = {}
        for r in range(m - 3, m + 3):
            for c in range(m - 3, m + 3):
                d = ((r - m + 0.5) / 2.5) ** 2 + ((c - m + 0.5) / 2.5) ** 2
                if d <= 1.0:
                    cells[(r, c)] = WATER
        return cells, set()
    reserved = set()  # an open plus, arms 4 wide and 12 long, kept clear of every stamp
    for r in range(m - 6, m + 6):
        for c in range(m - 6, m + 6):
            if abs(r - m + 0.5) < 2.5 or abs(c - m + 0.5) < 2.5:
                reserved.add((r, c))
    return {}, reserved


def _orient(rng: random.Random, shape: dict[tuple[int, int], str],
            anchor: tuple[int, int]) -> dict[tuple[int, int], str]:
    """`shape` ({(dr, dc): char} drawn round (0, 0)) under a random one of its 8 rotations and
    reflections, moved to `anchor`."""
    turns, flip = rng.randrange(4), rng.random() < 0.5
    cells = {}
    for (dr, dc), ch in shape.items():
        if flip:
            dc = -dc
        for _ in range(turns):
            dr, dc = dc, -dr
        cells[(anchor[0] + dr, anchor[1] + dc)] = ch
    return cells


def wall_stub(rng: random.Random, anchor: tuple[int, int]) -> dict[tuple[int, int], str]:
    """A small wall with no skirt, the cacti, posts and fence pieces the screenshot maps scatter
    over open floor: one tile, a 2-3 long bar either way, a 2x2 block, or a 1-wide L."""
    r0, c0 = anchor
    kind = rng.randrange(4)
    if kind == 0:
        return {anchor: WALL}
    if kind == 1:
        n = rng.randint(2, 3)
        if rng.random() < 0.5:
            return {(r0, c0 + i): WALL for i in range(n)}
        return {(r0 + i, c0): WALL for i in range(n)}
    if kind == 2:
        return {(r0 + dr, c0 + dc): WALL for dr in (0, 1) for dc in (0, 1)}
    return _orient(rng, {(0, 0): WALL, (0, 1): WALL, (1, 0): WALL}, anchor)


def branch_cluster(rng: random.Random, anchor: tuple[int, int]) -> dict[tuple[int, int], str]:
    """Shadow Spirits' antlers: a 1-wide wall trunk 4-7 long with a 1-wide branch of 2-3 off one
    end at a right angle and, half the time, a second off the other end turning the other way
    (a Z; turning the same way would make a C, whose bay is a pocket), with a 2x2 bush clump in
    the crook of each branch."""
    length = rng.randint(4, 7)
    shape = {(i, 0): WALL for i in range(length)}
    ends = [(length - 1, 1)] + ([(0, -1)] if rng.random() < 0.5 else [])
    for row, side in ends:
        for k in range(1, rng.randint(2, 3) + 1):
            shape[(row, side * k)] = WALL
        inward = -1 if row > 0 else 1  # the crook is on the trunk's side of the branch
        for dr in (1, 2):
            for dc in (1, 2):
                shape.setdefault((row + inward * dr, side * dc), BUSH)
    return _orient(rng, shape, anchor)


def fence_line(rng: random.Random, anchor: tuple[int, int]) -> dict[tuple[int, int], str]:
    """A 1-wide straight wall 4-8 long either way, the fence pieces that make Hot Maze's
    corridors."""
    r0, c0 = anchor
    n = rng.randint(4, MAX_WALL_RUN)
    if rng.random() < 0.5:
        return {(r0, c0 + i): WALL for i in range(n)}
    return {(r0 + i, c0): WALL for i in range(n)}


WALL_STAMPS = {"cluster": wall_cluster, "stub": wall_stub, "branch": branch_cluster, "fence": fence_line}


# ---- layouts (the screenshot families) ---------------------------------------------------------
# Each lays one reference map's skeleton on a fresh builder before the wall / water / bush passes
# run. Shapes are drawn in the top-left quarter and reflected across both axes by `quad`, except
# where a note says otherwise, and every layout is stamped with the family's own symmetry on top.
# Solid pieces keep the same >= 2 passable cells from each other and the border that
# STAMP_MARGIN gives the passes; bush may touch anything.


def quad(cells: dict[tuple[int, int], str], size: int = SIZE) -> dict[tuple[int, int], str]:
    """`cells` plus their reflections across both axes, a shape every symmetry here keeps."""
    out = {}
    for (r, c), ch in cells.items():
        for cell in ((r, c), (r, size - 1 - c), (size - 1 - r, c), (size - 1 - r, size - 1 - c)):
            out[cell] = ch
    return out


def ring_path(inset: int, size: int = SIZE) -> list[tuple[int, int]]:
    """The top-left quarter of the square ring through row/col `inset`, as a path from the
    vertical axis round the corner to the horizontal one. It stops 2 cells short of each axis,
    so the reflected ring opens 4 wide in the middle of every side."""
    mid = size // 2 - 1
    return [(inset, c) for c in range(mid - 2, inset - 1, -1)] + [(r, inset) for r in range(inset + 1, mid - 1)]


def border_path(size: int = SIZE) -> list[tuple[int, int]]:
    """The top-left quarter of the ring just inside the border (row/col 1), axis to axis."""
    mid = size // 2 - 1
    return [(1, c) for c in range(mid, 0, -1)] + [(r, 1) for r in range(2, mid + 1)]


def broken_line(rng: random.Random, path, ch: str, run=(4, MAX_WALL_RUN), gap=(3, 4)) -> dict[tuple[int, int], str]:
    """`path` cut into runs of `ch` separated by gaps, starting 0-2 cells in."""
    cells, i = {}, rng.randint(0, 2)
    while i < len(path):
        n = rng.randint(*run)
        for cell in path[i:i + n]:
            cells[cell] = ch
        i += n + rng.randint(*gap)
    return cells


def annulus(centre: tuple[float, float], inner: float, outer: float, ch: str, keep=None,
            size: int = SIZE) -> dict[tuple[int, int], str]:
    """Cells more than `inner` and at most `outer` from a float `centre` ((29.5, 29.5) is the
    map's), filtered by `keep(dr, dc)` when given. A negative `inner` gives a filled disc."""
    cr, cc = centre
    cells = {}
    for r in range(size):
        for c in range(size):
            dr, dc = r - cr, c - cc
            if inner < np.hypot(dr, dc) <= outer and (keep is None or keep(dr, dc)):
                cells[(r, c)] = ch
    return cells


def _layout_rings(b: "_Builder") -> None:
    """Hot Maze: two square rings of 1-wide wall, an outer one 8-10 tiles in and an inner one
    20-22 in round the centre court, cut into runs of 5-8 with 2-3 wide gaps, a bush plus in
    the court (arms 4 wide, 8-12 long) and a broken bush band just inside the border -- the
    fences leave bush patches too little floor to reach every hunt cell on their own."""
    cells = broken_line(b.rng, border_path(b.size), BUSH, run=(5, 10), gap=(3, 5))
    for inset in (b.rng.randint(8, 10), b.rng.randint(20, 22)):
        cells.update(broken_line(b.rng, ring_path(inset, b.size), WALL, run=(5, MAX_WALL_RUN), gap=(2, 3)))
    m = (b.size - 1) / 2
    arm = b.rng.randint(3, 5) + 0.5
    for r in range(b.size):
        for c in range(b.size):
            dr, dc = abs(r - m), abs(c - m)
            if (dr <= arm and dc <= 1.5) or (dc <= arm and dr <= 1.5):
                cells[(r, c)] = BUSH
    stamp(b.grid, quad(cells, b.size), b.symmetry)


def _layout_lake_ring(b: "_Builder") -> None:
    """Ghost Point: a broken bush band just inside the border, and eight 2-wide lakes on the
    square ring 10-12 tiles in, an L in each corner (arms 5-7) and a bar across the middle of
    each side (8-12 long), all rimmed with bush on the side facing the centre, round a small
    bush-ringed pond in the middle."""
    rng, size = b.rng, b.size
    mid = size // 2 - 1
    inset, arm, bar = rng.randint(10, 12), rng.randint(5, 7), rng.randint(4, 6)
    water = {}
    for k in (0, 1):
        for i in range(arm):
            water[(inset + k, inset + i)] = WATER
            water[(inset + i, inset + k)] = WATER
        for i in range(bar):
            water[(inset + k, mid - i)] = WATER
            water[(mid - i, inset + k)] = WATER
    cells = dict(water)
    for r, c in water:
        for n in ((r + 1, c), (r, c + 1)):
            if n not in water and n[0] <= mid and n[1] <= mid:  # a rim cell past an axis would
                cells[n] = BUSH                                  # reflect onto the lake itself
    cells.update(broken_line(rng, border_path(size), BUSH, run=(5, 10), gap=(3, 5)))
    m = (size - 1) / 2
    cells.update(annulus((m, m), 2.0, 3.2, BUSH, size=size))
    cells.update(annulus((m, m), -1.0, 2.0, WATER, size=size))
    stamp(b.grid, quad(cells, size), b.symmetry)


def _layout_branches(b: "_Builder") -> None:
    """Shadow Spirits: a broken bush band just inside the border and a 2-wide lake bar 10-14 long
    across the vertical axis, 4-8 rows below the centre (not reflected top to bottom; a point
    map gets its turned copy above the centre from `stamp`)."""
    rng, size = b.rng, b.size
    mid = size // 2 - 1
    cells = quad(broken_line(rng, border_path(size), BUSH, run=(5, 10), gap=(3, 5)), size)
    row, half = mid + rng.randint(4, 8), rng.randint(5, 7)
    for r in (row, row + 1):
        for c in range(mid + 1 - half, mid + 1 + half):
            cells[(r, c)] = WATER
    stamp(b.grid, cells, b.symmetry)


def _layout_crescent(b: "_Builder") -> None:
    """The fifth screenshot map: two 2-wide water crescents facing each other round a centre
    arena (a ring of radius 6-7 split by 4 or 6 wide openings, on the vertical axis or the
    horizontal one) with bush along their backs, and a pond in a full bush ring 11-15 tiles in
    from each corner."""
    rng, size = b.rng, b.size
    m = (size - 1) / 2
    radius, gap = rng.randint(6, 7), rng.choice((2.0, 3.0))
    if rng.random() < 0.5:
        keep = lambda dr, dc: abs(dc) > gap  # noqa: E731 -- openings top and bottom
    else:
        keep = lambda dr, dc: abs(dr) > gap  # noqa: E731 -- openings left and right
    cells = annulus((m, m), radius, radius + 1, BUSH, keep, size)
    cells.update(annulus((m, m), radius - 2, radius, WATER, keep, size))
    corner = rng.randint(11, 15)
    cells.update(pond(rng, (corner, corner), radius=rng.randint(2, 3), full_rim=True))
    stamp(b.grid, quad(cells, size), b.symmetry)


def _layout_vines(b: "_Builder") -> None:
    """Twisting Vines: three bush rivers from the top border to the bottom one. The middle one is
    2-6 wide, symmetric about both axes, with a 2-wide water stripe 3-5 long on the axis. The
    side one's centre wanders a column every 2-4 rows between cols 6 and 14, 3 or 5 wide, with
    2x2 and 3x2 water pockets sunk into its edges every 7-11 rows; it is drawn full height and
    only `stamp` copies it, so a mirror map gets it on the right and a point map gets it turned."""
    rng, size = b.rng, b.size
    mid = size // 2 - 1
    middle, half = {}, rng.randint(1, 3)
    for r in range(1, mid + 1):
        if rng.random() < 0.3:
            half = min(3, max(1, half + rng.choice((-1, 1))))
        for c in range(mid + 1 - half, mid + 1):
            middle[(r, c)] = BUSH
    top = rng.randint(3, mid - 6)  # one stripe per half: two could stop a row apart, a 1-wide gap
    for r in range(top, top + rng.randint(3, 5)):
        middle[(r, mid)] = WATER
    side, cols, halves = {}, {}, {}
    col, half, step = rng.randint(8, 12), rng.randint(1, 2), 0
    for r in range(1, size - 1):
        if step == 0:
            col = min(14, max(6, col + rng.choice((-1, 1))))
            if rng.random() < 0.3:
                half = 3 - half
            step = rng.randint(2, 4)
        step -= 1
        cols[r], halves[r] = col, half
        for c in range(col - half, col + half + 1):
            side[(r, c)] = BUSH
    r = rng.randint(3, 8)
    while r < size - 5:
        edge = cols[r] - halves[r] if rng.random() < 0.5 else cols[r] + halves[r] - 1
        for rr in range(r, r + rng.randint(2, 3)):
            side[(rr, edge)] = side[(rr, edge + 1)] = WATER
        r += rng.randint(7, 11)
    stamp(b.grid, quad(middle, size), b.symmetry)
    stamp(b.grid, side, b.symmetry)


LAYOUTS = {"rings": _layout_rings, "lake_ring": _layout_lake_ring, "branches": _layout_branches,
           "crescent": _layout_crescent, "vines": _layout_vines}


# ---- repair -----------------------------------------------------------------------------------


def _wall_runs(grid: np.ndarray):
    """Maximal straight runs of interior `#`, as (cells) lists, horizontal then vertical."""
    size = grid.shape[0]
    runs = []
    for r in range(1, size - 1):
        run = []
        for c in range(1, size):
            if c < size - 1 and grid[r, c] == WALL:
                run.append((r, c))
            else:
                if run:
                    runs.append(run)
                run = []
    for c in range(1, size - 1):
        run = []
        for r in range(1, size):
            if r < size - 1 and grid[r, c] == WALL:
                run.append((r, c))
            else:
                if run:
                    runs.append(run)
                run = []
    return runs


def cap_wall_runs(grid: np.ndarray, symmetry: str | None, max_run: int = MAX_WALL_RUN) -> int:
    """Punches a 2-tile floor gap through the middle of every straight interior wall run longer
    than `max_run` (and through its mirror). Returns the number of gaps punched. Loops because a
    punched run can leave a remainder that is still too long."""
    punched = 0
    for _ in range(64):
        long_runs = [run for run in _wall_runs(grid) if len(run) > max_run]
        if not long_runs:
            return punched
        for run in long_runs:
            mid = len(run) // 2
            stamp(grid, {run[mid - 1]: FLOOR, run[mid]: FLOOR}, symmetry)
            punched += 1
    return punched


def fill_dead_ends(grid: np.ndarray) -> int:
    """Turns every 1-wide dead-end cell (a passable cell with at most one passable 4-neighbour)
    into whatever blocks it, until none is left. Symmetric grids stay symmetric because the rule
    is local and run to a fixpoint. Returns the number of cells filled."""
    size = grid.shape[0]
    filled = 0
    changed = True
    while changed:
        changed = False
        for r in range(1, size - 1):
            for c in range(1, size - 1):
                if grid[r, c] not in PASSABLE:
                    continue
                neighbours = [grid[nr, nc] for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))]
                if sum(n in PASSABLE for n in neighbours) <= 1:
                    solids = [n for n in neighbours if n in SOLID]
                    grid[r, c] = WATER if solids.count(WATER) > solids.count(WALL) else WALL
                    filled += 1
                    changed = True
    return filled


def _components(passable: np.ndarray) -> list[list[tuple[int, int]]]:
    size = passable.shape[0]
    seen = np.zeros_like(passable)
    comps = []
    for r in range(size):
        for c in range(size):
            if not passable[r, c] or seen[r, c]:
                continue
            comp = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < size and 0 <= nc < size and passable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def connect_components(grid: np.ndarray, symmetry: str | None) -> int:
    """While the passable region has more than one component, carves the cheapest floor path
    (0-1 BFS: free through passable cells, one per blocking cell) from the largest to the
    second-largest, mirrored. Returns the number of cells carved."""
    size = grid.shape[0]
    carved = 0
    for _ in range(32):
        passable = np.isin(grid, list(PASSABLE))
        comps = _components(passable)
        if len(comps) <= 1:
            return carved
        target = set(comps[1])
        dist = {cell: 0 for cell in comps[0]}
        parent = {}
        dq = deque(comps[0])
        found = None
        while dq:
            cell = dq.popleft()
            if cell in target:
                found = cell
                break
            r, c = cell
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 < nr < size - 1 and 0 < nc < size - 1):
                    continue
                cost = 0 if passable[nr, nc] else 1
                nd = dist[cell] + cost
                if nd < dist.get((nr, nc), 10 ** 9):
                    dist[(nr, nc)] = nd
                    parent[(nr, nc)] = cell
                    if cost == 0:
                        dq.appendleft((nr, nc))
                    else:
                        dq.append((nr, nc))
        if found is None:
            return carved
        cells = {}
        cell = found
        while cell in parent:
            if not passable[cell]:
                cells[cell] = FLOOR
            cell = parent[cell]
        stamp(grid, cells, symmetry)
        carved += len(cells)
    return carved


def repair(grid: np.ndarray, symmetry: str | None) -> dict:
    """The full pass, in the order that converges: cap runs, fill dead ends, connect, then cap and
    fill once more for anything the carving/filling introduced."""
    stats = {"gaps_punched": cap_wall_runs(grid, symmetry), "dead_ends_filled": fill_dead_ends(grid)}
    stats["cells_carved"] = connect_components(grid, symmetry)
    stats["gaps_punched"] += cap_wall_runs(grid, symmetry)
    stats["dead_ends_filled"] += fill_dead_ends(grid)
    return stats


# ---- checks -----------------------------------------------------------------------------------


def interior_shares(grid: np.ndarray) -> dict[str, float]:
    inner = grid[1:-1, 1:-1]
    n = inner.size
    return {
        "wall": float(np.sum(inner == WALL)) / n,
        "bush": float(np.sum(inner == BUSH)) / n,
        "water": float(np.sum(inner == WATER)) / n,
    }


def reach_counts(passable: np.ndarray, radius: int = POCKET_RADIUS) -> np.ndarray:
    """(H, W) int: for every passable cell, how many passable cells a BFS of `radius` steps
    reaches (itself included); 0 elsewhere. One multi-source dilation over a
    (sources, H, W) boolean stack instead of a Python BFS per cell."""
    ys, xs = np.nonzero(passable)
    n = len(ys)
    reach = np.zeros((n,) + passable.shape, dtype=bool)
    reach[np.arange(n), ys, xs] = True
    for _ in range(radius):
        grown = reach.copy()
        grown[:, 1:, :] |= reach[:, :-1, :]
        grown[:, :-1, :] |= reach[:, 1:, :]
        grown[:, :, 1:] |= reach[:, :, :-1]
        grown[:, :, :-1] |= reach[:, :, 1:]
        grown &= passable[None]
        reach = grown
    counts = np.zeros(passable.shape, dtype=np.int64)
    counts[ys, xs] = reach.reshape(n, -1).sum(axis=1)
    return counts


@lru_cache(maxsize=4)
def _open_field_capacity(size: int, radius: int) -> np.ndarray:
    """reach_counts of an all-floor interior: what each cell could reach with no terrain."""
    interior = np.zeros((size, size), dtype=bool)
    interior[1:-1, 1:-1] = True
    return reach_counts(interior, radius)


def pocket_fractions(grid: np.ndarray, radius: int = POCKET_RADIUS) -> np.ndarray:
    """(H, W) float: reach / open-field capacity for every passable cell, 1.0 elsewhere -- so
    `.min()` is the worst pocket on the map. `check` wants it >= POCKET_MIN_REACH /
    POCKET_OPEN_FIELD (0.28) of what is geometrically in bounds, which is the plain reach test
    away from the border and a fair one at it."""
    passable = np.isin(grid, list(PASSABLE))
    counts = reach_counts(passable, radius)
    capacity = _open_field_capacity(grid.shape[0], radius)
    frac = np.ones(grid.shape, dtype=np.float64)
    frac[passable] = counts[passable] / capacity[passable]
    return frac


def bush_spread(grid: np.ndarray, cell_tiles: int = HUNT_CELL_TILES) -> float:
    """Fraction of the `cell_tiles` x `cell_tiles` hunt cells that hold at least one bush."""
    size = grid.shape[0]
    cells = 0
    with_bush = 0
    for r0 in range(0, size, cell_tiles):
        for c0 in range(0, size, cell_tiles):
            cells += 1
            if np.any(grid[r0:r0 + cell_tiles, c0:c0 + cell_tiles] == BUSH):
                with_bush += 1
    return with_bush / cells


def to_tiles(grid: np.ndarray) -> np.ndarray:
    """The loader's (H, W) int64 tile-id view of the char grid."""
    tiles = np.zeros(grid.shape, dtype=np.int64)
    for ch, tile in MAP_CHAR_TO_TILE.items():
        tiles[grid == ch] = int(tile)
    return tiles


def check(grid: np.ndarray, family: Family, symmetry: str) -> list[str]:
    """Every reason this grid is not an acceptable `family` map; empty means it is."""
    reasons = []
    size = grid.shape[0]
    try:
        loader.validate_map(to_tiles(grid), SimpleNamespace(map_h=size, map_w=size))
    except ValueError as exc:
        reasons.append(f"loader: {exc}")
    if not np.array_equal(grid, mirror_grid(grid, symmetry)):
        reasons.append(f"not {symmetry}-symmetric")
    shares = interior_shares(grid)
    for name in ("wall", "bush", "water"):
        lo, hi = getattr(family, name)
        if not lo <= shares[name] <= hi:
            reasons.append(f"{name} share {shares[name]:.3f} outside [{lo:.2f}, {hi:.2f}]")
    n_boxes = int(np.sum(grid == BOX))
    if not family.boxes[0] <= n_boxes <= family.boxes[1]:
        reasons.append(f"{n_boxes} boxes outside {family.boxes}")
    if int(np.sum(grid == SPAWN)) != N_SPAWNS:
        reasons.append(f"{int(np.sum(grid == SPAWN))} spawns, want {N_SPAWNS}")
    longest_run = max((len(run) for run in _wall_runs(grid)), default=0)
    if longest_run > MAX_WALL_RUN:
        reasons.append(f"a straight wall run of {longest_run} > {MAX_WALL_RUN}")
    worst = float(pocket_fractions(grid).min())
    if worst < POCKET_MIN_REACH / POCKET_OPEN_FIELD:
        reasons.append(f"pocket: worst cell reaches {worst:.2f} of its open-field capacity")
    spread = bush_spread(grid)
    if spread < BUSH_SPREAD_MIN:
        reasons.append(f"bush spread {spread:.2f} < {BUSH_SPREAD_MIN}")
    n_waypoints = len(loader.bush_waypoints(to_tiles(grid), HUNT_CELL_TILES))
    if n_waypoints > loader.MAX_BUSH_WAYPOINTS:
        reasons.append(f"{n_waypoints} bush waypoints > {loader.MAX_BUSH_WAYPOINTS}")
    return reasons


def stats_of(grid: np.ndarray) -> dict:
    shares = interior_shares(grid)
    return {
        **shares,
        "boxes": int(np.sum(grid == BOX)),
        "spawns": int(np.sum(grid == SPAWN)),
        "bush_waypoints": len(loader.bush_waypoints(to_tiles(grid), HUNT_CELL_TILES)),
        "bush_spread": bush_spread(grid),
        "pocket_min": float(pocket_fractions(grid).min()),
        "longest_wall_run": max((len(run) for run in _wall_runs(grid)), default=0),
    }


# ---- the generator ----------------------------------------------------------------------------


class _Builder:
    """One attempt at one seed. Holds the grid, the RNG, the symmetry and the placement region,
    and knows how to test a footprint and stamp it with its mirror."""

    def __init__(self, seed: int, family: Family, symmetry: str, size: int = SIZE):
        self.rng = random.Random(seed)
        self.family = family
        self.symmetry = symmetry
        self.mirror = mirror_fn(symmetry)
        self.size = size
        self.grid = new_grid(size)
        self.reserved: set[tuple[int, int]] = set()
        # What a stamp's margin ring may hold. A layout lays bush before any stamp is drawn, and
        # bush is walkable, so it still leaves >= 2 passable cells between two solids; the four
        # original families keep floor-only, which is what their shipped seeds were drawn under.
        self.margin_ok = frozenset((FLOOR, BUSH)) if family.layout else frozenset((FLOOR,))
        inner_lo, inner_hi = 1, size - 2
        if family.moat_inset:
            add_moat(self.grid, self.rng, family.moat_inset, symmetry)
            inner_lo = family.moat_inset + 2 + 1
            inner_hi = size - 1 - family.moat_inset - 2 - 1
        self.inner = (inner_lo, inner_hi)          # where stamps may go (inclusive)
        half = size // 2 - 1                        # 29: the half the stamps are drawn in
        if symmetry == POINT:
            self.rows, self.cols = (inner_lo, min(half, inner_hi)), (inner_lo, inner_hi)
        else:
            self.rows, self.cols = (inner_lo, inner_hi), (inner_lo, min(half, inner_hi))

    # -- geometry helpers --

    def _in_inner(self, cell) -> bool:
        lo, hi = self.inner
        return lo <= cell[0] <= hi and lo <= cell[1] <= hi

    def _is_floor(self, cell) -> bool:
        return self._in_inner(cell) and self.grid[cell] == FLOOR and cell not in self.reserved

    def _anchor(self) -> tuple[int, int]:
        return self.rng.randint(*self.rows), self.rng.randint(*self.cols)

    def _footprint_ok(self, cells, margin: int) -> bool:
        """Every cell is free floor inside the placement region; every cell within `margin`
        (Chebyshev) of the footprint is floor or the border/moat is not there; and the mirror
        image satisfies the same and does not come within `margin` of the original."""
        own = set(cells)
        mirrored = {self.mirror(c) for c in own}
        for group in (own, mirrored):
            for cell in group:
                if not self._is_floor(cell):
                    return False
        for group, other in ((own, mirrored), (mirrored, own)):
            for r, c in group:
                for dr in range(-margin, margin + 1):
                    for dc in range(-margin, margin + 1):
                        n = (r + dr, c + dc)
                        if n in group:
                            continue
                        # The margin ring must be plain floor: not the other half's image, not
                        # the border, moat or any earlier stamp, not a reserved centre cell.
                        if n in other or not (0 <= n[0] < self.size and 0 <= n[1] < self.size):
                            return False
                        if self.grid[n] not in self.margin_ok or n in self.reserved:
                            return False
        return True

    def _place(self, make, margin: int, tries: int = 80) -> int:
        """Draws anchors until `make(anchor)`'s footprint fits; stamps it and returns the number
        of cells written (both halves), or 0 if nothing fit."""
        for _ in range(tries):
            cells = make(self._anchor())
            if cells and self._footprint_ok(cells, margin):
                stamp(self.grid, cells, self.symmetry)
                return 2 * len(cells)
        return 0

    # -- passes --

    def _cells_for(self, share: float) -> int:
        return int(round(share * (self.size - 2) ** 2))

    def _count(self, ch: str) -> int:
        return int(np.sum(self.grid[1:-1, 1:-1] == ch))

    def _target(self, band) -> float:
        lo, hi = band
        return self.rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo))

    def centre(self) -> None:
        if self.symmetry != POINT:
            return
        cells, reserved = centre_feature(self.rng, self.size)
        self.reserved |= reserved
        for cell in cells:
            self.reserved.add(cell)
        stamp(self.grid, cells, None)
        # keep the margin around the centre feature too
        for r, c in list(cells):
            for dr in range(-STAMP_MARGIN, STAMP_MARGIN + 1):
                for dc in range(-STAMP_MARGIN, STAMP_MARGIN + 1):
                    self.reserved.add((r + dr, c + dc))

    def _wall_stamp(self, anchor) -> dict[tuple[int, int], str]:
        """One stamp from the family's `wall_mix`; a one-entry mix spends no draw on the pick."""
        mix = self.family.wall_mix
        kind = mix[0][0]
        if len(mix) > 1:
            x = self.rng.random()
            for kind, weight in mix:
                if x < weight:
                    break
                x -= weight
        return WALL_STAMPS[kind](self.rng, anchor)

    def walls(self) -> None:
        want = self._cells_for(self._target(self.family.wall))
        hi = self._cells_for(self.family.wall[1])
        stalls = 0
        while self._count(WALL) < want and stalls < 12:
            snapshot = self.grid.copy()
            written = self._place(self._wall_stamp, STAMP_MARGIN)
            if self._count(WALL) > hi:
                # too big a bite: the band is the contract, so give the last one back
                self.grid = snapshot
                stalls += 1
                continue
            stalls = stalls + 1 if written == 0 else 0

    def water(self) -> None:
        want = self._cells_for(self._target(self.family.water))
        hi = self._cells_for(self.family.water[1])
        stalls = 0
        while self._count(WATER) < want and stalls < 12:
            room = hi - self._count(WATER)
            if room < 12:
                return
            if room < 40 or self.rng.random() < 0.4:
                make = lambda a: channel(self.rng, a)  # noqa: E731
            else:
                make = lambda a: pond(self.rng, a)  # noqa: E731
            snapshot = self.grid.copy()
            written = self._place(make, STAMP_MARGIN)
            if self._count(WATER) > hi:
                self.grid = snapshot
                stalls += 1
                continue
            stalls = stalls + 1 if written == 0 else 0

    def bushes(self) -> None:
        want = self._cells_for(self._target(self.family.bush))
        hi = self._cells_for(self.family.bush[1])
        stalls = 0
        cell_tiles = HUNT_CELL_TILES
        while self._count(BUSH) < want and stalls < 30:
            room = hi - self._count(BUSH)
            if room < 8:
                return
            # Half the time aim at a hunt cell that has no bush yet, so HUNTER's sweep has
            # somewhere to go everywhere (the bush-spread check) by construction.
            anchor = None
            if self.rng.random() < 0.5:
                empty = []
                for r0 in range(0, self.size, cell_tiles):
                    for c0 in range(0, self.size, cell_tiles):
                        if not np.any(self.grid[r0:r0 + cell_tiles, c0:c0 + cell_tiles] == BUSH):
                            empty.append((r0, c0))
                if empty:
                    r0, c0 = self.rng.choice(empty)
                    anchor = (self.rng.randint(r0, r0 + cell_tiles - 1), self.rng.randint(c0, c0 + cell_tiles - 1))
            if anchor is None or not self._is_floor(anchor):
                anchor = self._anchor()
            size = self.rng.randint(6, min(30, max(6, room // 2)))
            cells = bush_patch(self.rng, anchor, self._is_floor, size)
            mirrored = {self.mirror(c) for c in cells}
            if len(cells) < 4 or any(not self._is_floor(c) for c in mirrored) or mirrored & set(cells):
                stalls += 1
                continue
            stamp(self.grid, cells, self.symmetry)
            stalls = 0

    def spawns(self) -> None:
        """8 spawns on the square ring SPAWN_INSET (+-1) tiles in from the border, at even angles
        over half the circle, mirrored to 16; each nudged to the nearest floor cell inside the
        placement region."""
        size = self.size
        centre = (size - 1) / 2.0
        half_n = N_SPAWNS // 2
        cells = {}
        for i in range(half_n):
            if self.symmetry == POINT:
                angle = np.pi + (i + 0.5) * (np.pi / half_n)          # the top half (rows < 30)
            else:
                angle = -np.pi / 2 + (i + 0.5) * (np.pi / half_n)     # the right half; mirrored left
            angle += self.rng.uniform(-0.05, 0.05)
            inset = SPAWN_INSET + self.rng.randint(-1, 1)
            half_extent = centre - inset
            dx, dy = np.cos(angle), np.sin(angle)
            t = min(half_extent / abs(dx) if abs(dx) > 1e-9 else np.inf,
                    half_extent / abs(dy) if abs(dy) > 1e-9 else np.inf)
            target = (int(round(centre + t * dy)), int(round(centre + t * dx)))
            cell = self._nearest_floor(target, taken=set(cells))
            if cell is not None:
                cells[cell] = SPAWN
        stamp(self.grid, cells, self.symmetry)

    def _nearest_floor(self, target, taken):
        """BFS over the whole grid from `target` for the nearest free floor cell in the placement
        region that is not within 2 of a taken cell."""
        seen = {target}
        dq = deque([target])
        while dq:
            cell = dq.popleft()
            if self._is_floor(cell) and all(max(abs(cell[0] - t[0]), abs(cell[1] - t[1])) > 2 for t in taken):
                return cell
            r, c = cell
            for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= n[0] < self.size and 0 <= n[1] < self.size and n not in seen:
                    seen.add(n)
                    dq.append(n)
        return None

    def crates(self) -> None:
        """`X` singles and pairs on cells 1 tile from cover (a wall, water or bush 8-neighbour),
        >= 3 apart from each other and >= 2 from any spawn, drawn in the half and mirrored so the
        count is always even and inside the family's band."""
        lo, hi = self.family.boxes
        want_half = self.rng.randint((lo + 1) // 2, hi // 2)
        spawns = list(zip(*np.nonzero(self.grid == SPAWN)))
        spots: list[tuple[int, int]] = []
        # The border ring counts as cover, so every edge cell looks sheltered and crates would
        # line the edge, which no screenshot shows. The screenshot families keep them 3 tiles in;
        # the four original families' seeds were drawn without the rule (0 changes nothing).
        edge = 3 if self.family.layout else 0

        def free(cell):
            return (self._is_floor(cell) or (self._in_inner(cell) and self.grid[cell] == BUSH)) \
                and min(cell[0], cell[1], self.size - 1 - cell[0], self.size - 1 - cell[1]) >= edge \
                and all(max(abs(cell[0] - s[0]), abs(cell[1] - s[1])) >= 2 for s in spawns) \
                and all(max(abs(cell[0] - x[0]), abs(cell[1] - x[1])) >= 3 for x in spots)

        def sheltered(cell):
            r, c = cell
            return any(self.grid[r + dr, c + dc] in (WALL, WATER, BUSH)
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))

        placed = 0
        for _ in range(400):
            if placed >= want_half:
                break
            cell = self._anchor()
            if not free(cell) or not sheltered(cell):
                continue
            mirrored = self.mirror(cell)
            if max(abs(cell[0] - mirrored[0]), abs(cell[1] - mirrored[1])) < 3:
                continue
            cells = crate_spot(self.rng, cell, lambda n: free(n) and self.mirror(n) != cell and n != mirrored)
            if placed + len(cells) > want_half:
                cells = {cell: BOX}
            stamp(self.grid, cells, self.symmetry)
            spots.extend(cells)
            spots.extend(self.mirror(c) for c in cells)
            placed += len(cells)

    def build(self) -> np.ndarray:
        if self.family.layout:
            LAYOUTS[self.family.layout](self)
            # Water first: a pond needs a 9x9 hole and the stub-heavy wall mixes leave none.
            self.water()
            self.walls()
        else:
            self.centre()
            self.walls()
            self.water()
        self.bushes()
        repair(self.grid, self.symmetry)
        self.spawns()
        self.crates()
        return self.grid


def generate(seed: int, family: str | Family, symmetry: str | None = None,
             max_attempts: int = 200) -> tuple[np.ndarray, dict]:
    """The map for the first seed in `seed, seed+1, ...` that passes `check`, with its stats.
    `stats["seed"]` is that seed -- `generate(stats["seed"], ...)` reproduces the grid at once --
    and `stats["rejected"]` lists `(seed, reasons)` for the ones that failed (`scripts/gen_maps.py`
    prints them; a rejected seed is simply never shipped)."""
    fam = FAMILIES[family] if isinstance(family, str) else family
    sym = symmetry or fam.symmetry
    rejected = []
    for attempt in range(max_attempts):
        s = seed + attempt
        grid = _Builder(s, fam, sym).build()
        reasons = check(grid, fam, sym)
        if not reasons:
            stats = stats_of(grid)
            stats.update(seed=s, family=fam.name, symmetry=sym, attempts=attempt + 1, rejected=rejected)
            return grid, stats
        rejected.append((s, reasons))
    raise RuntimeError(f"{fam.name}: no seed in [{seed}, {seed + max_attempts}) passed; last: {rejected[-1][1]}")


# ---- I/O --------------------------------------------------------------------------------------


def to_csv(grid: np.ndarray) -> str:
    return "\n".join(",".join(row) for row in grid.tolist()) + "\n"


def render(grid: np.ndarray) -> str:
    return "\n".join("".join(row) for row in grid.tolist())
