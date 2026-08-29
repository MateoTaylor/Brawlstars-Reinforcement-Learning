"""Map CSV loading and the device-resident MapBank. See BRAWL_SIM_BUILD_PLAN.md Step 6.

Runs once at env construction, not in the sim's hot path -- numpy and plain Python are fine
here (CONVENTIONS.md's no-host-sync rule scopes to BrawlVecEnv.step() and anything it calls).

Critical performance rule (unchanged downstream): never materialize a per-env (N, H, W) map
slice. Point queries gather-index MapBank's (M, H, W) tensors by map_id; the only per-env grids
that ever exist are the observation grids, built by scatter into a zeroed buffer (Step 25).
"""
from collections import deque
from pathlib import Path

import numpy as np
import torch

from ..constants import (
    CHAR_TO_TILE,
    TILE_BLOCKS_PROJ,
    TILE_BLOCKS_UNIT,
    TILE_IS_BUSH,
    TILE_IS_WATER,
    Tile,
)

CSV_DIR = Path(__file__).resolve().parent / "csv"

MAX_SPAWNS = 32
MAX_BOX_SPOTS = 64
MIN_SPAWNS = 8  # blank.csv's stated exception to the general >=12 minimum (Step 5)
MIN_BOX_SPOTS = 8  # blank.csv's stated exception to the general >=16 minimum (Step 5)
# Bush waypoints for the HUNTER personality (Step 41). Capped at 63 because a bot's visited-set is
# a single int64 BITMASK (state.ent_hunt_seen) -- one bit per waypoint, which keeps "everywhere
# I have already looked" to one scalar per entity instead of a ring buffer of positions.
MAX_BUSH_WAYPOINTS = 63


def load_map_csv(path, cfg) -> np.ndarray:
    """Parses a map CSV into an (H, W) int64 tile-id grid. Does not validate content beyond
    a shape check against cfg -- see validate_map for border/connectivity/marker checks."""
    text = Path(path).read_text().strip("\n")
    rows = [line.split(",") for line in text.split("\n") if line]
    h = len(rows)
    w = len(rows[0]) if h else 0
    if h != cfg.map_h or w != cfg.map_w:
        raise ValueError(f"{path}: map is {h}x{w}, cfg expects {cfg.map_h}x{cfg.map_w}")

    tiles = np.zeros((h, w), dtype=np.int64)
    for y, row in enumerate(rows):
        if len(row) != w:
            raise ValueError(f"{path}: row {y} has {len(row)} columns, expected {w}")
        for x, ch in enumerate(row):
            if ch not in CHAR_TO_TILE:
                raise ValueError(f"{path}: unknown tile character {ch!r} at row {y}, col {x}")
            tiles[y, x] = int(CHAR_TO_TILE[ch])
    return tiles


def _count_unit_passable_components(tiles: np.ndarray) -> int:
    blocks_unit = TILE_BLOCKS_UNIT.numpy()[tiles]
    h, w = blocks_unit.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for y in range(h):
        for x in range(w):
            if blocks_unit[y, x] or seen[y, x]:
                continue
            count += 1
            dq = deque([(y, x)])
            seen[y, x] = True
            while dq:
                cy, cx = dq.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and not blocks_unit[ny, nx]:
                        seen[ny, nx] = True
                        dq.append((ny, nx))
    return count


def validate_map(tiles: np.ndarray, cfg) -> None:
    h, w = tiles.shape
    if h != cfg.map_h or w != cfg.map_w:
        raise ValueError(f"map is {h}x{w}, cfg expects {cfg.map_h}x{cfg.map_w}")
    if not np.all(tiles[0, :] == Tile.WALL) or not np.all(tiles[-1, :] == Tile.WALL):
        raise ValueError("map's top/bottom border row is not entirely WALL")
    if not np.all(tiles[:, 0] == Tile.WALL) or not np.all(tiles[:, -1] == Tile.WALL):
        raise ValueError("map's left/right border column is not entirely WALL")

    n_spawn = int(np.sum(tiles == Tile.SPAWN))
    n_box = int(np.sum(tiles == Tile.BOX))
    if n_spawn < MIN_SPAWNS:
        raise ValueError(f"map has {n_spawn} SPAWN markers, need >= {MIN_SPAWNS}")
    if n_box < MIN_BOX_SPOTS:
        raise ValueError(f"map has {n_box} BOX markers, need >= {MIN_BOX_SPOTS}")
    if n_spawn > MAX_SPAWNS:
        raise ValueError(f"map has {n_spawn} SPAWN markers, exceeds MAX_SPAWNS={MAX_SPAWNS}")
    if n_box > MAX_BOX_SPOTS:
        raise ValueError(f"map has {n_box} BOX markers, exceeds MAX_BOX_SPOTS={MAX_BOX_SPOTS}")

    if _count_unit_passable_components(tiles) != 1:
        raise ValueError("map's unit-passable region is not a single connected component")


def _tile_points(tiles: np.ndarray, tile_id: int) -> np.ndarray:
    ys, xs = np.nonzero(tiles == tile_id)
    return np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float32)  # (x, y) tile centers


def spawn_points(tiles: np.ndarray) -> np.ndarray:
    """(K, 2) f32 tile centers, sorted by angle from the map's geometric center -- the
    spawner (Step 24) depends on this order to hand out evenly-spaced starting positions."""
    pts = _tile_points(tiles, Tile.SPAWN)
    h, w = tiles.shape
    cx, cy = w / 2.0, h / 2.0
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    return pts[np.argsort(angles)]


def box_points(tiles: np.ndarray) -> np.ndarray:
    """(M, 2) f32 tile centers, in raster scan order (no ordering requirement downstream)."""
    return _tile_points(tiles, Tile.BOX)


def bush_waypoints(tiles: np.ndarray, cell_tiles: int) -> np.ndarray:
    """(K, 2) f32 bush tile centers, at most one per `cell_tiles` x `cell_tiles` map cell: the
    bush tile closest to each cell's own center, skipping cells with no bush in them. Sorted by
    angle from the map center, like spawn_points, purely so the order is stable and readable.

    **Why waypoints exist at all.** The HUNTER personality is supposed to sweep the map looking
    for players it cannot see, and it was first built on `perception.bush_scan` -- "walk to the
    nearest bush tile you have not searched yet, within the 4-tile scan radius". Measured on
    `bushy` (23.7% bush tiles) that produced a hunter which walked 75 tiles in 60 seconds for a
    NET displacement of 15.7 (ratio 0.29): in a dense bush field there is always another
    unsearched tile one step away, so hunters shuffled around their spawn instead of going
    anywhere. Bush-camping at the far side of the map was as safe as it had ever been.

    A coarse per-cell subsample fixes that by construction: consecutive waypoints are `cell_tiles`
    apart, so "go to the nearest unvisited one" is always a real journey, and the set covers the
    whole playable map, so a hunter that visits them all has genuinely looked everywhere. It is
    also far cheaper than a bigger tile scan -- the scan's cost grows with the SQUARE of its
    radius (a 4-tile radius is 49 probes; a map-sized one would be thousands), while this is a
    fixed <=63-element gather no matter how large the map is.
    """
    if cell_tiles < 1:
        raise ValueError(f"cell_tiles must be >= 1, got {cell_tiles}")
    h, w = tiles.shape
    picks = []
    for cy0 in range(0, h, cell_tiles):
        for cx0 in range(0, w, cell_tiles):
            cell = tiles[cy0:cy0 + cell_tiles, cx0:cx0 + cell_tiles]
            ys, xs = np.nonzero(cell == int(Tile.BUSH))
            if len(ys) == 0:
                continue
            mid_y = cy0 + cell.shape[0] / 2.0
            mid_x = cx0 + cell.shape[1] / 2.0
            gy, gx = ys + cy0 + 0.5, xs + cx0 + 0.5
            best = np.argmin((gx - mid_x) ** 2 + (gy - mid_y) ** 2)
            picks.append((gx[best], gy[best]))

    if not picks:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(picks, dtype=np.float32)
    if len(pts) > MAX_BUSH_WAYPOINTS:
        # Keep an evenly-spaced subset rather than the first N, so a truncated set still spans the
        # whole map instead of only its top-left corner.
        keep = np.linspace(0, len(pts) - 1, MAX_BUSH_WAYPOINTS).round().astype(np.int64)
        pts = pts[np.unique(keep)]
    angles = np.arctan2(pts[:, 1] - h / 2.0, pts[:, 0] - w / 2.0)
    return pts[np.argsort(angles)]


class MapBank:
    __slots__ = (
        "tiles", "blocks_unit", "blocks_proj", "is_bush", "is_water",
        "pad_tiles", "pad_blocks_unit", "pad_blocks_proj", "pad_is_bush", "pad_is_water",
        "spawns", "n_spawns", "box_spots", "n_box_spots",
        "bush_wp", "n_bush_wp",
    )


def _pad_slots(points_list: list[np.ndarray], max_slots: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array([len(p) for p in points_list], dtype=np.int64)
    padded = np.zeros((len(points_list), max_slots, 2), dtype=np.float32)
    for i, pts in enumerate(points_list):
        padded[i, : len(pts)] = pts
    return padded, counts


def build_map_bank(cfg, device) -> MapBank:
    tiles_list, spawn_list, box_list, bush_wp_list = [], [], [], []
    for name in cfg.map_names:
        tiles = load_map_csv(CSV_DIR / f"{name}.csv", cfg)
        validate_map(tiles, cfg)
        tiles_list.append(tiles)
        spawn_list.append(spawn_points(tiles))
        box_list.append(box_points(tiles))
        bush_wp_list.append(bush_waypoints(tiles, cfg.bots_hunt_cell_tiles))

    bank = MapBank()

    tiles_np = np.stack(tiles_list, axis=0)  # (M, H, W)
    bank.tiles = torch.as_tensor(tiles_np, dtype=torch.int64, device=device)
    unit_lut = TILE_BLOCKS_UNIT.to(device)
    proj_lut = TILE_BLOCKS_PROJ.to(device)
    bush_lut = TILE_IS_BUSH.to(device)
    water_lut = TILE_IS_WATER.to(device)
    bank.blocks_unit = unit_lut[bank.tiles]
    bank.blocks_proj = proj_lut[bank.tiles]
    bank.is_bush = bush_lut[bank.tiles]
    bank.is_water = water_lut[bank.tiles]

    pad_h, pad_w = cfg.view_h // 2, cfg.view_w // 2
    bank.pad_tiles = torch.nn.functional.pad(
        bank.tiles, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=int(Tile.WALL),
    )
    bank.pad_blocks_unit = unit_lut[bank.pad_tiles]
    bank.pad_blocks_proj = proj_lut[bank.pad_tiles]
    bank.pad_is_bush = bush_lut[bank.pad_tiles]
    bank.pad_is_water = water_lut[bank.pad_tiles]

    spawns_np, n_spawns_np = _pad_slots(spawn_list, MAX_SPAWNS)
    box_np, n_box_np = _pad_slots(box_list, MAX_BOX_SPOTS)
    bush_wp_np, n_bush_wp_np = _pad_slots(bush_wp_list, MAX_BUSH_WAYPOINTS)
    bank.spawns = torch.as_tensor(spawns_np, dtype=torch.float32, device=device)
    bank.n_spawns = torch.as_tensor(n_spawns_np, dtype=torch.int64, device=device)
    bank.box_spots = torch.as_tensor(box_np, dtype=torch.float32, device=device)
    bank.n_box_spots = torch.as_tensor(n_box_np, dtype=torch.int64, device=device)
    # A map with no bush tiles at all (blank.csv, walled.csv) legitimately gets 0 waypoints; the
    # HUNTER personality reads that as "nothing to sweep" and falls back to exploring.
    bank.bush_wp = torch.as_tensor(bush_wp_np, dtype=torch.float32, device=device)
    bank.n_bush_wp = torch.as_tensor(n_bush_wp_np, dtype=torch.int64, device=device)

    return bank
