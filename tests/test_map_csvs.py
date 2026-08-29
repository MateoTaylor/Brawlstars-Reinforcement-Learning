"""Checks the committed brawl_sim/maps/csv/*.csv files directly against Step 5's acceptance
criteria. brawl_sim/maps/loader.py (Step 6) doesn't exist yet, so this reads the raw CSV text
rather than going through a loader -- Step 6 adds a second round of checks on top of this via
the real MapBank machinery.
"""
from collections import deque
from pathlib import Path

import pytest

from brawl_sim.constants import CHAR_TO_TILE, Tile

MAPS_DIR = Path(__file__).resolve().parent.parent / "brawl_sim" / "maps" / "csv"

DIMENSIONS = {
    "blank": (20, 20), "open": (60, 60), "bushy": (60, 60), "walled": (60, 60),
    "skull_creek": (60, 60), "feast_or_famine": (60, 60),
    "scorched_stone": (60, 60), "island_invasion": (60, 60),
}
ALL_MAPS = list(DIMENSIONS)
MIN_SPAWN = {name: 12 for name in DIMENSIONS} | {"blank": 8}
MIN_BOX = {name: 16 for name in DIMENSIONS} | {"blank": 8}

UNIT_BLOCKING = {Tile.WALL, Tile.WATER, Tile.FENCE}


def _read_grid(name: str) -> list[list[str]]:
    text = (MAPS_DIR / f"{name}.csv").read_text().strip("\n")
    return [line.split(",") for line in text.split("\n")]


def _connected_components(grid: list[list[str]]) -> list[list[tuple[int, int]]]:
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    comps = []
    for y in range(h):
        for x in range(w):
            tile = CHAR_TO_TILE[grid[y][x]]
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
                        if CHAR_TO_TILE[grid[ny][nx]] not in UNIT_BLOCKING:
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
            assert ch in CHAR_TO_TILE, f"{name}.csv has unknown character {ch!r}"


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


def test_walled_has_water_and_fences():
    grid = _read_grid("walled")
    assert sum(row.count("~") for row in grid) > 0
    assert sum(row.count("f") for row in grid) > 0
