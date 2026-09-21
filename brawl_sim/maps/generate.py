"""Procedural Solo Showdown maps (SIM_OVERHAUL_PLAN.md Step M2). Library half; `scripts/gen_maps.py`
is the CLI and `tests/test_map_generate.py` pins every primitive.

Not a hot path: plain Python + numpy on a 60x60 char grid, run once per map at authoring time.
Deterministic in `(seed, family, symmetry)` -- every random draw comes from one
`random.Random(seed)`, sets are never iterated for sampling, and numpy is used only for counting.

Pipeline (`generate`):

1. border ring of `#`; for the water-border family a 2-tile `~` moat `Family.moat_inset` tiles in,
   with a floor rim outside it and four 3-tile gaps (two drawn, two mirrored, off-centre);
2. centre feature on point-symmetric maps (2x2 wall + 4 bush, a small pond, or an open cross kept
   clear of everything else);
3. stamps drawn in one half of the map and written together with their mirror
   (`mirror_point` for 180-degree symmetry, `mirror_lr` for left-right): wall clusters, ponds and
   channels until the family's wall and water shares are met, then bush patches until the bush
   share is met -- solid stamps keep a 2-cell margin from each other and the border so no 1-wide
   corridor can exist, bush is allowed to touch anything and merge into fields;
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
POCKET_RADIUS = 6        # plan: BFS radius 6 from every floor cell ...
# ... must reach >= 24 of the 85 cells an open field offers (0.28), scaled by what is in bounds
# so a corner cell is judged against its 28. The plan said 40 (0.47); measured on generated maps
# that rejected every 2-wide lane between two features (a channel two rows below the moat, two
# clusters a margin apart) -- reach 26 -- and the reference maps are full of exactly those.
# 0.28 still rejects what the README means by a pocket: a 2x3 notch reaches ~20 (0.24), a
# dead-end lane ~15 (0.18).
POCKET_MIN_REACH = 24
POCKET_OPEN_FIELD = 85
BUSH_SPREAD_MIN = 0.80   # a bush cell in >= 80% of the hunt cells
HUNT_CELL_TILES = 10     # mirrors configs/default.yaml bots.hunt_cell_tiles
STAMP_MARGIN = 2         # cells of floor kept around every solid stamp (and from the border)


# ---- families -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    """Style bands from SIM_OVERHAUL_PLAN.md Step M2: interior shares (rows/cols 1..58, so 3364
    cells; the moat counts as water) and crate counts. `count` is how many of the ten maps M3
    draws from this family; `symmetry` is the default and `generate(..., symmetry=)` overrides it
    (the standard family ships five point-symmetric maps and one mirror map)."""
    name: str
    count: int
    wall: tuple[float, float]
    bush: tuple[float, float]
    water: tuple[float, float]
    boxes: tuple[int, int]
    symmetry: str = POINT
    moat_inset: int = 0  # 0 = no moat; otherwise the moat's outer edge is this many tiles in


FAMILIES: dict[str, Family] = {
    "standard": Family("standard", 6, (0.07, 0.12), (0.20, 0.30), (0.03, 0.08), (20, 32)),
    "open": Family("open", 1, (0.04, 0.06), (0.12, 0.18), (0.00, 0.03), (16, 24)),
    "dense": Family("dense", 1, (0.08, 0.12), (0.32, 0.38), (0.02, 0.05), (24, 36), symmetry=MIRROR),
    # The moat is drawn 4 tiles in: a 3-wide floor rim outside it, water at rows/cols 4-5, the
    # island from 6. Flush against the border the plan's four gaps would lead nowhere; a rim
    # narrower than 3 fails the pocket check by construction. The moat alone is ~11% water.
    "water_border": Family("water_border", 2, (0.06, 0.10), (0.20, 0.28), (0.12, 0.20), (20, 32),
                           moat_inset=4),
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
    end of a side at least 5 long, so the inside corner is >= 3 wide and never a notch -- T stems
    were tried and their 2-wide notches are pockets by the check below), and a 1-tile bush skirt
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
         rim: bool = True) -> dict[tuple[int, int], str]:
    """A filled ellipse of water with semi-axes 2..4 (`radius` pins both), and a 1-tile bush rim
    along a random half of its perimeter."""
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
                if angle <= np.pi:
                    cells[(nr, nc)] = BUSH
    return cells


def channel(rng: random.Random, start: tuple[int, int]) -> dict[tuple[int, int], str]:
    """A 1-2 wide water strip 6-12 long, straight or with one right-angle bend. (The plan allowed
    14; a 14-long strip a lane away from the moat or another feature is right at the pocket
    check's edge, 12 is not.)"""
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
    `.min()` is the worst pocket on the map. The plan's ">= 40 of 85" becomes >= 0.47 of what is
    geometrically in bounds, which is the same test away from the border and a fair one at it."""
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
                        if self.grid[n] != FLOOR or n in self.reserved:
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

    def walls(self) -> None:
        want = self._cells_for(self._target(self.family.wall))
        hi = self._cells_for(self.family.wall[1])
        stalls = 0
        while self._count(WALL) < want and stalls < 12:
            snapshot = self.grid.copy()
            written = self._place(lambda a: wall_cluster(self.rng, a), STAMP_MARGIN)
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

        def free(cell):
            return (self._is_floor(cell) or (self._in_inner(cell) and self.grid[cell] == BUSH)) \
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
    and `stats["rejected"]` lists `(seed, reasons)` for the ones that failed, for the README."""
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


def from_csv(text: str) -> np.ndarray:
    rows = [line.split(",") for line in text.strip("\n").split("\n") if line]
    return np.array(rows, dtype="<U1")


def render(grid: np.ndarray) -> str:
    return "\n".join("".join(row) for row in grid.tolist())
