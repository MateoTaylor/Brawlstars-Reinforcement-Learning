from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from brawl_sim.config import EnvConfig, load_config
from brawl_sim.constants import Tile
from brawl_sim.env import BrawlVecEnv
from brawl_sim.maps.generate import generate, to_csv
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


def test_load_map_csv_refuses_fence_character(tmp_path):
    """Fences left the map vocabulary in 2026-09. `f` is still a character in the full tile
    alphabet (brawl_vision's label files use it), so this pins that the LOADER, specifically,
    refuses it rather than mapping it to a wall or a floor."""
    path = _small_grid_csv(tmp_path, [["#", "#", "#"], ["#", "f", "#"], ["#", "#", "#"]])
    cfg = EnvConfig(map_h=3, map_w=3)
    with pytest.raises(ValueError, match="'f'"):
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


# ---- the ten generated maps (SIM_OVERHAUL_PLAN.md Step M3) ---------------------------------

#: name -> (seed, family, symmetry, boxes). Mirrors the table in brawl_sim/maps/README.md; the
#: box counts are literals on purpose (reading them from the CSV would make the test
#: self-consistent). Spawns are 16 on every generated map.
GENERATED_MAPS = {
    "broken_wall": (2, "standard", "point", 26),
    "stone_fort": (3, "standard", "point", 30),
    "twin_ponds": (4, "standard", "point", 28),
    "cross_creek": (100, "standard", "point", 32),
    "split_river": (303, "standard", "point", 26),
    "narrow_pass": (502, "standard", "mirror", 22),
    "dry_gulch": (702, "open", "point", 18),
    "thorn_field": (800, "dense", "mirror", 24),
    "reed_marsh": (901, "water_border", "point", 26),
    "hollow_ring": (904, "water_border", "point", 20),
}

#: `configs/default.yaml` `world.maps` once M3.3 lands: the six that trained every checkpoint so
#: far, then the ten in README order.
SIXTEEN_MAP_ROTATION = (
    "open", "bushy", "skull_creek", "feast_or_famine", "scorched_stone", "island_invasion",
    "broken_wall", "stone_fort", "twin_ponds", "cross_creek", "split_river", "narrow_pass",
    "dry_gulch", "thorn_field", "reed_marsh", "hollow_ring",
)


@pytest.mark.parametrize("name", list(GENERATED_MAPS))
def test_generated_map_loads_and_validates_with_its_marker_counts(name):
    seed, family, symmetry, boxes = GENERATED_MAPS[name]
    cfg = EnvConfig(map_h=60, map_w=60)
    tiles = load_map_csv(CSV_DIR / f"{name}.csv", cfg)
    validate_map(tiles, cfg)  # border, 8-32 spawns, 8-64 boxes, one passable component
    assert int(np.sum(tiles == Tile.SPAWN)) == 16
    assert int(np.sum(tiles == Tile.BOX)) == boxes
    assert spawn_points(tiles).shape == (16, 2)


@pytest.mark.parametrize("name", list(GENERATED_MAPS))
def test_generated_map_is_reproduced_by_its_seed_at_the_first_attempt(name):
    """The README promises `generate(seed, family, symmetry)` returns the shipped CSV at attempt
    1, which is what makes a map swappable by seed. A hand edit to a CSV, or a generator change
    that moves any random draw, breaks that promise and fails here -- either is a deliberate
    act that must then update the README table (or accept the map is now hand-authored)."""
    seed, family, symmetry, _ = GENERATED_MAPS[name]
    grid, stats = generate(seed, family, symmetry=symmetry)
    assert stats["attempts"] == 1
    assert stats["seed"] == seed
    assert to_csv(grid) == (CSV_DIR / f"{name}.csv").read_text()


def test_default_rotation_is_the_six_plus_the_ten():
    cfg = _default_cfg()
    assert len(cfg.map_names) == 16
    assert cfg.map_names == SIXTEEN_MAP_ROTATION
    bank = build_map_bank(cfg, device="cpu")
    assert bank.tiles.shape == (16, 60, 60)


@pytest.mark.parametrize("name", list(GENERATED_MAPS))
def test_smoke_episode_on_generated_map_holds_the_invariants(name):
    """One 1000-tick episode per generated map with `engine.debug_checks` on, so
    `check_invariants` runs after the reset and after every decision: a map whose spawn ring,
    crate spots or water layout breaks the sim (a NaN, a unit outside the map, a dead entity
    with hp) fails here by name.

    "1000-step" is read in the config's own unit, SIM TICKS (`sim.max_episode_steps`): 200
    decisions at the shipped `action_repeat: 5`. Measured 2026-09-18 on CPU the sim costs
    ~14 ms per tick whatever `n_envs` or `n_enemies` is (kernel-launch bound), so this is
    ~15 s per map and ~2.5 min for the ten; 1000 DECISIONS per map would be ~12 min. Random
    move/attack/super actions rather than idling so the hero walks into walls, water and
    crates and the super's projectiles cross them; the attack column is drawn over the action
    spec's full width so a super (value 2) fires on every map, not only ordinary attacks."""
    cfg = load_config(CONFIGS / "default.yaml", overrides={
        "world": {"maps": [name], "map_selection": "fixed", "fixed_map": name},
        "sim": {"max_episode_steps": 1000},
        "engine": {"debug_checks": True},
    })
    assert cfg.debug_checks is True  # the override key spelled wrong would make this test vacuous
    assert cfg.max_episode_steps == 1000 and cfg.action_repeat == 5
    env = BrawlVecEnv(cfg, n_envs=2, device="cpu", seed=0, verbose=False)
    env.reset()
    assert env.state.map_id.tolist() == [0, 0]
    n_move, n_attack = cfg.action_nvec
    assert n_move == 17
    assert n_attack >= 3  # 0 nothing / 1 attack / 2 super: the super must be drawable below
    gen = torch.Generator().manual_seed(0)
    n_done = 0
    for _ in range(200):
        action = torch.stack([torch.randint(0, n_move, (2,), generator=gen),
                              torch.randint(0, n_attack, (2,), generator=gen)], dim=1)
        obs, reward, terminated, truncated, info = env.step(action)
        n_done += int((terminated | truncated).sum())
        assert torch.isfinite(reward).all()
    # 1000 ticks with max_episode_steps 1000: every env ended at least once (by death or by
    # the clock), so the autoreset path ran on this map too.
    assert n_done >= 2
