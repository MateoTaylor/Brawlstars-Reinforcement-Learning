"""Checks the committed brawl_sim/maps/csv/*.csv files directly, reading the raw CSV text rather
than going through the loader; tests/test_maps.py adds a second round of checks on top of this
via the real MapBank machinery.
"""
from collections import deque
from pathlib import Path

import pytest

from brawl_sim.constants import MAP_CHAR_TO_TILE, Tile

MAPS_DIR = Path(__file__).resolve().parent.parent / "brawl_sim" / "maps" / "csv"

DIMENSIONS = {
    "blank": (20, 20), "open": (60, 60), "bushy": (60, 60), "walled": (60, 60),
    "skull_creek": (60, 60), "feast_or_famine": (60, 60),
    "scorched_stone": (60, 60), "island_invasion": (60, 60),
    # The ten generated maps (brawl_sim/maps/README.md has the
    # seed table). brawl_sim/maps/generate.py only makes 60x60 grids.
    "broken_wall": (60, 60), "stone_fort": (60, 60), "twin_ponds": (60, 60),
    "cross_creek": (60, 60), "split_river": (60, 60), "narrow_pass": (60, 60),
    "dry_gulch": (60, 60), "thorn_field": (60, 60), "reed_marsh": (60, 60),
    "hollow_ring": (60, 60),
    # Five maps transcribed from Solo Showdown screenshots and fifteen from the generator
    # families modeled on them (2026-09-27; both tables in brawl_sim/maps/README.md).
    "hot_maze": (60, 60), "ghost_point": (60, 60), "shadow_spirits": (60, 60),
    "crescent_lakes": (60, 60), "twisting_vines": (60, 60),
    "pond_maze": (60, 60), "canal_maze": (60, 60), "picket_maze": (60, 60),
    "lagoon_ring": (60, 60), "square_lakes": (60, 60), "bush_halo": (60, 60),
    "bramble_ponds": (60, 60), "bramble_bend": (60, 60), "bramble_stars": (60, 60),
    "moon_gate": (60, 60), "half_moon": (60, 60), "moon_pools": (60, 60),
    "vine_springs": (60, 60), "vine_canal": (60, 60), "vine_hollow": (60, 60),
}
ALL_MAPS = list(DIMENSIONS)
GENERATED_MAPS = [
    "broken_wall", "stone_fort", "twin_ponds", "cross_creek", "split_river", "narrow_pass",
    "dry_gulch", "thorn_field", "reed_marsh", "hollow_ring",
    "pond_maze", "canal_maze", "picket_maze", "lagoon_ring", "square_lakes", "bush_halo",
    "bramble_ponds", "bramble_bend", "bramble_stars", "moon_gate", "half_moon", "moon_pools",
    "vine_springs", "vine_canal", "vine_hollow",
]
# The real maps have one spawn per player, ten, and the spawner needs one per entity (the hero
# plus default.yaml's nine enemies), so the transcriptions keep exactly ten.
SCREENSHOT_MAPS = ["hot_maze", "ghost_point", "shadow_spirits", "crescent_lakes", "twisting_vines"]
MIN_SPAWN = {name: 12 for name in DIMENSIONS} | {"blank": 8} | {name: 10 for name in SCREENSHOT_MAPS}
MIN_BOX = {name: 16 for name in DIMENSIONS} | {"blank": 8}

UNIT_BLOCKING = {Tile.WALL, Tile.WATER}


def _read_grid(name: str) -> list[list[str]]:
    text = (MAPS_DIR / f"{name}.csv").read_text().strip("\n")
    return [line.split(",") for line in text.split("\n")]


def _connected_components(grid: list[list[str]]) -> list[list[tuple[int, int]]]:
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    comps = []
    for y in range(h):
        for x in range(w):
            tile = MAP_CHAR_TO_TILE[grid[y][x]]
            if tile in UNIT_BLOCKING or seen[y][x]:
                continue
            comp = []
            dq = deque([(y, x)])
            seen[y][x] = True
            while dq:
                cy, cx = dq.popleft()
                comp.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx]:
                        if MAP_CHAR_TO_TILE[grid[ny][nx]] not in UNIT_BLOCKING:
                            seen[ny][nx] = True
                            dq.append((ny, nx))
            comps.append(comp)
    return comps


@pytest.mark.parametrize("name", ALL_MAPS)
def test_map_dimensions(name):
    grid = _read_grid(name)
    h, w = DIMENSIONS[name]
    assert len(grid) == h
    assert all(len(row) == w for row in grid)


@pytest.mark.parametrize("name", ALL_MAPS)
def test_every_character_is_known(name):
    grid = _read_grid(name)
    for row in grid:
        for ch in row:
            assert ch in MAP_CHAR_TO_TILE, f"{name}.csv has unknown character {ch!r}"


@pytest.mark.parametrize("name", ALL_MAPS)
def test_border_is_wall(name):
    grid = _read_grid(name)
    h, w = len(grid), len(grid[0])
    assert all(grid[0][x] == "#" for x in range(w))
    assert all(grid[h - 1][x] == "#" for x in range(w))
    assert all(grid[y][0] == "#" for y in range(h))
    assert all(grid[y][w - 1] == "#" for y in range(h))


@pytest.mark.parametrize("name", ALL_MAPS)
def test_marker_counts_meet_minimums(name):
    grid = _read_grid(name)
    n_spawn = sum(row.count("S") for row in grid)
    n_box = sum(row.count("X") for row in grid)
    assert n_spawn >= MIN_SPAWN[name], f"{name}.csv has {n_spawn} S markers, need >= {MIN_SPAWN[name]}"
    assert n_box >= MIN_BOX[name], f"{name}.csv has {n_box} X markers, need >= {MIN_BOX[name]}"


@pytest.mark.parametrize("name", ALL_MAPS)
def test_unit_passable_region_is_fully_connected(name):
    grid = _read_grid(name)
    comps = _connected_components(grid)
    assert len(comps) == 1, f"{name}.csv has {len(comps)} disconnected unit-passable regions"


def test_bushy_density_is_roughly_a_quarter():
    grid = _read_grid("bushy")
    h, w = len(grid), len(grid[0])
    interior = (h - 2) * (w - 2)
    n_bush = sum(row.count("b") for row in grid)
    fraction = n_bush / interior
    assert 0.15 <= fraction <= 0.35, f"bushy.csv bush fraction {fraction:.2f} is not ~25%"


def test_walled_has_water():
    grid = _read_grid("walled")
    assert sum(row.count("~") for row in grid) > 0


@pytest.mark.parametrize("name", GENERATED_MAPS)
def test_generated_map_has_exactly_sixteen_spawns(name):
    """The generator places `N_SPAWNS = 16` markers -- a hand edit that
    drops or duplicates one would still pass the >= 12 minimum above, so the exact count is
    pinned here. The literal 16 is deliberate: reading `generate.N_SPAWNS` would let the test
    follow a changed constant."""
    grid = _read_grid(name)
    assert sum(row.count("S") for row in grid) == 16


@pytest.mark.parametrize("name", ALL_MAPS)
def test_no_map_contains_fences(name):
    """Fences are not a sim tile (2026-09). `test_every_character_is_known` already refuses `f`
    through the vocabulary; this one says so by name, so a reintroduced fence fails readably."""
    grid = _read_grid(name)
    assert sum(row.count("f") for row in grid) == 0, f"{name}.csv carries a fence cell"
