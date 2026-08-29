from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from brawl_sim.config import EnvConfig, load_config
from brawl_sim.constants import Tile
from brawl_sim.maps.loader import (
    MAX_BOX_SPOTS,
    MAX_SPAWNS,
    box_points,
    build_map_bank,
    load_map_csv,
    spawn_points,
    validate_map,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
CSV_DIR = Path(__file__).resolve().parent.parent / "brawl_sim" / "maps" / "csv"


def _small_grid_csv(tmp_path, rows) -> Path:
    path = tmp_path / "tiny.csv"
    path.write_text("\n".join(",".join(row) for row in rows) + "\n")
    return path


# ---- load_map_csv / validate_map --------------------------------------------

def test_load_map_csv_reads_real_open_map():
    cfg = EnvConfig(map_h=60, map_w=60)
    tiles = load_map_csv(CSV_DIR / "open.csv", cfg)
    assert tiles.shape == (60, 60)
    assert tiles.dtype == np.int64
    assert np.all(tiles[0, :] == Tile.WALL)


def test_load_map_csv_dimension_mismatch_raises(tmp_path):
    path = _small_grid_csv(tmp_path, [["#", "#", "#"], ["#", ".", "#"], ["#", "#", "#"]])
    cfg = EnvConfig(map_h=5, map_w=5)
    with pytest.raises(ValueError):
        load_map_csv(path, cfg)


def test_load_map_csv_unknown_character_raises(tmp_path):
    path = _small_grid_csv(tmp_path, [["#", "#", "#"], ["#", "?", "#"], ["#", "#", "#"]])
    cfg = EnvConfig(map_h=3, map_w=3)
    with pytest.raises(ValueError):
        load_map_csv(path, cfg)


def test_validate_map_passes_on_real_maps():
    cfg60 = EnvConfig(map_h=60, map_w=60)
    for name in ("open", "bushy", "walled"):
        tiles = load_map_csv(CSV_DIR / f"{name}.csv", cfg60)
        validate_map(tiles, cfg60)  # must not raise

    cfg20 = EnvConfig(map_h=20, map_w=20)
    tiles = load_map_csv(CSV_DIR / "blank.csv", cfg20)
    validate_map(tiles, cfg20)


def test_validate_map_rejects_broken_border():
    h = w = 10
    tiles = np.full((h, w), Tile.FLOOR, dtype=np.int64)
    tiles[0, :] = Tile.WALL
    tiles[-1, :] = Tile.WALL
    tiles[:, 0] = Tile.WALL
    tiles[:, -1] = Tile.WALL
    tiles[0, 5] = Tile.FLOOR  # break the border
    tiles[1:3, 1] = Tile.SPAWN
    cfg = EnvConfig(map_h=h, map_w=w)
    with pytest.raises(ValueError):
        validate_map(tiles, cfg)


def test_validate_map_rejects_too_few_markers():
    h = w = 10
    tiles = np.full((h, w), Tile.FLOOR, dtype=np.int64)
    tiles[0, :] = Tile.WALL
    tiles[-1, :] = Tile.WALL
    tiles[:, 0] = Tile.WALL
    tiles[:, -1] = Tile.WALL
    tiles[1, 1] = Tile.SPAWN  # only 1, need >= 8
    tiles[1, 2] = Tile.BOX
    cfg = EnvConfig(map_h=h, map_w=w)
    with pytest.raises(ValueError):
        validate_map(tiles, cfg)


def test_validate_map_rejects_disconnected_region():
    h = w = 10
    tiles = np.full((h, w), Tile.FLOOR, dtype=np.int64)
    tiles[0, :] = Tile.WALL
    tiles[-1, :] = Tile.WALL
    tiles[:, 0] = Tile.WALL
    tiles[:, -1] = Tile.WALL
    tiles[:, 5] = Tile.WALL  # split the interior into two sealed halves
    for i in range(8):
        tiles[1 + (i % 4), 1 if i < 4 else 6] = Tile.SPAWN
    for i in range(8):
        tiles[6 - (i % 3), 2 if i < 4 else 7] = Tile.BOX
    cfg = EnvConfig(map_h=h, map_w=w)
    with pytest.raises(ValueError):
        validate_map(tiles, cfg)


# ---- spawn_points / box_points ------------------------------------------

def test_spawn_points_are_angle_sorted():
    # four spawns placed at the cardinal directions around a small grid's center
    h = w = 11  # center at (5.5, 5.5) in tile-center coords... use odd size for exact centering
    tiles = np.full((h, w), Tile.FLOOR, dtype=np.int64)
    cy = cx = h // 2
    tiles[cy, cx + 4] = Tile.SPAWN  # east,  angle ~0
    tiles[cy + 4, cx] = Tile.SPAWN  # south, angle ~+pi/2 (y increases downward)
    tiles[cy, cx - 4] = Tile.SPAWN  # west,  angle ~+-pi
    tiles[cy - 4, cx] = Tile.SPAWN  # north, angle ~-pi/2
    pts = spawn_points(tiles)
    angles = np.arctan2(pts[:, 1] - h / 2.0, pts[:, 0] - w / 2.0)
    assert np.all(np.diff(angles) >= 0)  # ascending


def test_spawn_points_on_blank_csv_are_angle_sorted():
    cfg = EnvConfig(map_h=20, map_w=20)
    tiles = load_map_csv(CSV_DIR / "blank.csv", cfg)
    pts = spawn_points(tiles)
    angles = np.arctan2(pts[:, 1] - 10.0, pts[:, 0] - 10.0)
    assert np.all(np.diff(angles) >= 0)


def test_box_points_returns_tile_centers():
    h = w = 6
    tiles = np.full((h, w), Tile.FLOOR, dtype=np.int64)
    tiles[2, 3] = Tile.BOX
    pts = box_points(tiles)
    assert pts.shape == (1, 2)
    assert pts[0, 0] == pytest.approx(3.5)  # x = col + 0.5
    assert pts[0, 1] == pytest.approx(2.5)  # y = row + 0.5


# ---- MapBank / build_map_bank --------------------------------------------

def _default_cfg():
    return load_config(CONFIGS / "default.yaml")


def _debug_tiny_cfg():
    overrides = yaml.safe_load((CONFIGS / "presets" / "debug_tiny.yaml").read_text())
    return load_config(CONFIGS / "default.yaml", overrides=overrides)


def test_build_map_bank_leading_dim_matches_map_names():
    cfg = _default_cfg()
    bank = build_map_bank(cfg, device="cpu")
    m = len(cfg.map_names)
    assert bank.tiles.shape == (m, cfg.map_h, cfg.map_w)
    assert bank.blocks_unit.shape == (m, cfg.map_h, cfg.map_w)
    assert bank.blocks_proj.shape == (m, cfg.map_h, cfg.map_w)
    assert bank.is_bush.shape == (m, cfg.map_h, cfg.map_w)
    assert bank.is_water.shape == (m, cfg.map_h, cfg.map_w)
    assert bank.spawns.shape == (m, MAX_SPAWNS, 2)
    assert bank.box_spots.shape == (m, MAX_BOX_SPOTS, 2)
    assert bank.n_spawns.shape == (m,)
    assert bank.n_box_spots.shape == (m,)


def test_build_map_bank_dtypes():
    cfg = _default_cfg()
    bank = build_map_bank(cfg, device="cpu")
    assert bank.tiles.dtype == torch.int64
    assert bank.blocks_unit.dtype == torch.bool
    assert bank.blocks_proj.dtype == torch.bool
    assert bank.is_bush.dtype == torch.bool
    assert bank.is_water.dtype == torch.bool
    assert bank.spawns.dtype == torch.float32
    assert bank.n_spawns.dtype == torch.int64


def test_build_map_bank_padded_shape_and_blocking():
    cfg = _default_cfg()
    bank = build_map_bank(cfg, device="cpu")
    m = len(cfg.map_names)
    # `map + 2 * (view // 2)`, NOT `map + view`. The loader pads by `view // 2` on each side, so
    # the two agree only for EVEN view dimensions -- which every config had until the view was
    # measured against the real camera (Terrain_Perception_Build_Plan.md Phase K) and became odd.
    # The padding itself was always right: with view_w = 2k+1 the widest index the observation
    # reads is `(map_w - 1 - k) + (2k) + k = map_w - 1 + 2k`, which is exactly the last column of
    # this shape.
    pad_h_full, pad_w_full = 2 * (cfg.view_h // 2), 2 * (cfg.view_w // 2)
    expected_shape = (m, cfg.map_h + pad_h_full, cfg.map_w + pad_w_full)
    assert bank.pad_tiles.shape == expected_shape
    assert bank.pad_blocks_unit.shape == expected_shape
    assert bank.pad_blocks_proj.shape == expected_shape

    pad_h, pad_w = cfg.view_h // 2, cfg.view_w // 2
    # every padding cell reads as WALL: blocks unit AND projectiles, is neither bush nor water
    assert torch.all(bank.pad_tiles[:, :pad_h, :] == Tile.WALL)
    assert torch.all(bank.pad_tiles[:, -pad_h:, :] == Tile.WALL)
    assert torch.all(bank.pad_tiles[:, :, :pad_w] == Tile.WALL)
    assert torch.all(bank.pad_tiles[:, :, -pad_w:] == Tile.WALL)
    assert torch.all(bank.pad_blocks_unit[:, :pad_h, :])
    assert torch.all(bank.pad_blocks_proj[:, :pad_h, :])
    assert not torch.any(bank.pad_is_bush[:, :pad_h, :])
    assert not torch.any(bank.pad_is_water[:, :pad_h, :])

    # the real map content is untouched, just shifted by the padding offset
    assert torch.equal(bank.pad_tiles[:, pad_h:-pad_h, pad_w:-pad_w], bank.tiles)


def test_build_map_bank_n_spawns_and_n_box_spots_match_counts():
    cfg = _default_cfg()
    bank = build_map_bank(cfg, device="cpu")
    for i, name in enumerate(cfg.map_names):
        tiles = load_map_csv(CSV_DIR / f"{name}.csv", cfg)
        assert int(bank.n_spawns[i]) == int(np.sum(tiles == Tile.SPAWN))
        assert int(bank.n_box_spots[i]) == int(np.sum(tiles == Tile.BOX))
        # unused slots stay zeroed
        n = int(bank.n_spawns[i])
        assert torch.all(bank.spawns[i, n:] == 0)


def test_build_map_bank_debug_tiny_single_blank_map():
    cfg = _debug_tiny_cfg()
    bank = build_map_bank(cfg, device="cpu")
    assert bank.tiles.shape == (1, 20, 20)
    assert int(bank.n_spawns[0]) == 8
    assert int(bank.n_box_spots[0]) == 8


def test_gather_indexing_never_materializes_per_env_slice():
    # the whole point of MapBank: point queries gather-index the (M,H,W) bank directly
    cfg = _default_cfg()
    bank = build_map_bank(cfg, device="cpu")
    n = 5
    map_id = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64)
    iy = torch.tensor([10, 20, 30, 5, 6], dtype=torch.int64)
    ix = torch.tensor([10, 20, 30, 5, 6], dtype=torch.int64)
    blocked = bank.blocks_unit[map_id, iy, ix]
    assert blocked.shape == (n,)
