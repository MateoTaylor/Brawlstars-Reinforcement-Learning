"""A hand-labelled game map, for the deploy loop to localize against. KNOWN_MAP_LOCALIZATION_PLAN.md
sections 2, 4 and 7.

The label is `brawl_deployment/data/maps/<name>.csv`, drawn with `scripts/map_label.py` over a
top-down image of the map (`<name>.png`; the lattice it was drawn against is in `<name>.json`,
which only the tool reads).

#### The file

The sim's map CSV format (`brawl_sim/maps/csv/`): 60 comma-separated rows of 60 cells, `.` floor,
`#` wall, `b` bush, `~` water, `S` spawn. It differs from a sim map in four ways, all because it is
the TRUE game layout rather than a training map:

  * No forced WALL ring. Bush and floor reach the edge of a real map.
  * `f` is allowed: a fence stops a body and passes a shot. The deploy grid already renders one
    (`grid._tile_lut`). The sim refuses the character (`MAP_CHAR_TO_TILE`).
  * No `X`. Power cube boxes spawn semi-randomly each match, so a map image shows only the fixed
    terrain and a label holds no crates. (The blue-and-green sprites on Dark Passage are candles,
    which are walls that break easily.)
  * No spawn minimum. The checks are the ones that stay true of any real map (`check`).

`?` is a cell nobody has labelled yet. The labeling tool saves it, so an unfinished label is never
lost; `KnownMap.load` refuses it.

#### The frame

world (x, y) = (col - 30, row - 30), so tile (c, r)'s centre is at (c - 29.5, r - 29.5). That makes
`MapFrame.pos_norm` equal col / 60, puts world (0, 0) at the map centre where the gas map already
assumes it is, and lands the whole map on indices 34-93 of the 128 x 128 grids (plan section 2).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brawl_sim.constants import CHAR_TO_TILE, Tile
from brawl_sim.maps.loader import _count_unit_passable_components
from brawl_vision.terrain.labeling import CLASS_INDEX

MAPS_DIR = Path(__file__).resolve().parent.parent / "data" / "maps"

# (cols, rows). Every Solo Showdown map is 60 x 60, and so is the sim's map_w x map_h, which is
# where `MapFrame` takes its origin and extent from. The frame below is only right on this size.
MAP_TILES = (60, 60)

UNLABELLED = "?"

# The sim's alphabet with `f` (MAP_CHAR_TO_TILE drops it; the real game has fences) and without
# `X` (boxes spawn semi-randomly each match, so no map image shows one).
LEGEND = {char: tile for char, tile in CHAR_TO_TILE.items() if tile != Tile.BOX}


def read_chars(path) -> np.ndarray:
    """`(rows, cols)` array of single characters from a map CSV. Refuses a ragged file, since no
    array can hold one, and nothing else: whether the characters make a legal map is `check`'s
    job, so the labeling tool can reopen a label that is still unfinished."""
    text = Path(path).read_text().strip("\n")
    rows = [line.split(",") for line in text.split("\n")]
    for r, row in enumerate(rows):
        if len(row) != len(rows[0]):
            raise ValueError(f"{path}: row {r} has {len(row)} cells, row 0 has {len(rows[0])}")
        for c, cell in enumerate(row):
            # "<U1" below would silently keep the first character of a longer cell.
            if len(cell) != 1:
                raise ValueError(f"{path}: cell (col {c}, row {r}) is {cell!r}, not one character")
    return np.array(rows, dtype="<U1")


def write_chars(path, chars: np.ndarray) -> None:
    Path(path).write_text("\n".join(",".join(row) for row in chars) + "\n")


def check(chars: np.ndarray) -> None:
    """The rules every real map label obeys (plan section 4). Raises `ValueError` naming the first
    one broken.

    `validate_map`'s other rules bind sim maps only: a real map has no wall ring, its spawn count is
    whatever the game put there, and its crates are never in the label.
    """
    cols, rows = MAP_TILES
    if chars.shape != (rows, cols):
        raise ValueError(f"the label is {chars.shape[1]} x {chars.shape[0]} cells (cols x rows); "
                         f"a map is {cols} x {rows}")
    todo = np.argwhere(chars == UNLABELLED)
    if len(todo):
        r, c = todo[0]
        raise ValueError(f"{len(todo)} cells are still unlabelled '?', the first at "
                         f"(col {c}, row {r})")
    stray = np.argwhere(~np.isin(chars, list(LEGEND)))
    if len(stray):
        r, c = stray[0]
        # str(): numpy 2 reprs an element as "np.str_('Z')".
        raise ValueError(f"{str(chars[r, c])!r} at (col {c}, row {r}) is not in the legend "
                         f"{''.join(LEGEND)}")
    tiles = _to_tiles(chars)
    if not (tiles == Tile.SPAWN).any():
        raise ValueError("the label has no spawn 'S'")
    pieces = _count_unit_passable_components(tiles)
    if pieces != 1:
        raise ValueError(f"the walkable ground is {pieces} separate regions, not one: a pocket is "
                         f"sealed off by walls, water or fences")


def _to_tiles(chars: np.ndarray) -> np.ndarray:
    tiles = np.zeros(chars.shape, np.int64)
    for char, tile in LEGEND.items():
        tiles[chars == char] = int(tile)
    return tiles


@dataclass(frozen=True)
class KnownMap:
    name: str
    tiles: np.ndarray       # (rows, cols) int64 Tile ids, SPAWN included

    @classmethod
    def load(cls, name_or_path) -> "KnownMap":
        """A map by name (`dark_passage`, from `MAPS_DIR`) or by CSV path, checked."""
        path = Path(name_or_path)
        if path.suffix != ".csv":
            path = MAPS_DIR / f"{path.name}.csv"
        chars = read_chars(path)
        try:
            check(chars)
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None
        return cls(name=path.stem, tiles=_to_tiles(chars))

    def centres(self, tile: Tile) -> np.ndarray:
        """`(K, 2)` float32 `(x, y)` centres of every `tile` cell, in WORLD tiles, raster order."""
        rows, cols = np.nonzero(self.tiles == tile)
        half_w, half_h = self.tiles.shape[1] / 2, self.tiles.shape[0] / 2
        return np.stack([cols + 0.5 - half_w, rows + 0.5 - half_h], axis=1).astype(np.float32)

    @property
    def spawns(self) -> np.ndarray:
        return self.centres(Tile.SPAWN)

    @property
    def classes(self) -> np.ndarray:
        """`(rows, cols)` int8 indices into `labeling.CLASSES`, the terrain classifier's alphabet. A
        spawn is FLOOR, which is what it looks like on screen."""
        out = np.full(self.tiles.shape, CLASS_INDEX[Tile.FLOOR], np.int8)
        for tile, index in CLASS_INDEX.items():
            out[self.tiles == tile] = index
        return out

    def world_grid(self, height: int, width: int) -> np.ndarray:
        """`classes` painted into a `(height, width)` world-frame grid, the occupancy map's frame
        (index `[0, 0]` is world tile `(-(width // 2), -(height // 2))`), WALL off the map. The sim
        pads its tile bank with WALL too (`maps/loader.py`), so a crop past the edge reads the same
        in both."""
        rows, cols = self.tiles.shape
        r0, c0 = height // 2 - rows // 2, width // 2 - cols // 2
        if r0 < 0 or c0 < 0 or r0 + rows > height or c0 + cols > width:
            raise ValueError(f"a {cols} x {rows} map does not fit a {width} x {height} grid")
        out = np.full((height, width), CLASS_INDEX[Tile.WALL], np.int8)
        out[r0:r0 + rows, c0:c0 + cols] = self.classes
        return out


class KnownTerrain:
    """The terrain the deploy grid reads when a map is named (plan section 7): the label while the
    localizer's `world` hands out the map frame, the occupancy map otherwise. It has the occupancy
    map's `height`, `width`, `origin` and `best()`, with their meanings, so `GridBuilder`, `GasMap`
    and the loop's cell lookups take it unchanged.

    It follows the frame handed out THIS tick (`MapLocalizer.in_map_frame`), not the localizer's
    newest state. On the tick `observe` commits a fix, the hero and every track are still in the
    lattice frame, and the label under them would be the map read at the wrong place.

    The map frame is handed out while fixed and while a fix carried across an odometry cut is
    being confirmed. The label is used for both: the loot and the tracks keep their state through
    a confirmation, and switching the static planes to the occupancy map for the tick or two it
    takes would flicker the policy's terrain.
    """

    def __init__(self, known: KnownMap, occupancy, localizer):
        self.occupancy = occupancy
        self.localizer = localizer
        self.height, self.width = occupancy.height, occupancy.width
        self.label = known.world_grid(self.height, self.width)
        self.label.setflags(write=False)     # handed out by `best()`; nobody may paint on it

    @property
    def origin(self) -> tuple[int, int]:
        return self.occupancy.origin

    def best(self) -> np.ndarray:
        return self.label if self.localizer.in_map_frame else self.occupancy.best()
