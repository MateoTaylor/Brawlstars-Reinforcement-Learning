import math
from pathlib import Path

import torch
import yaml

from brawl_sim.config import EnvConfig, load_config
from brawl_sim.constants import TILE_BLOCKS_PROJ, TILE_BLOCKS_UNIT, Tile
from brawl_sim.core import geometry as geo
from brawl_sim.core import terrain
from brawl_sim.maps.loader import build_map_bank

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _cfg(**kw):
    base = dict(map_h=10, map_w=10, view_h=4, view_w=4, los_step_tiles=0.5, max_ray_tiles=20.0)
    base.update(kw)
    return EnvConfig(**base)


def _tiles(h, w, fill=Tile.FLOOR):
    t = torch.full((h, w), int(fill), dtype=torch.int64)
    t[0, :] = Tile.WALL
    t[-1, :] = Tile.WALL
    t[:, 0] = Tile.WALL
    t[:, -1] = Tile.WALL
    return t


def _unit_mask(tiles):
    return TILE_BLOCKS_UNIT[tiles].unsqueeze(0)  # (1, H, W)


def _proj_mask(tiles):
    return TILE_BLOCKS_PROJ[tiles].unsqueeze(0)  # (1, H, W)


class _FakeBank:
    def __init__(self, blocks_proj):
        self.blocks_proj = blocks_proj


# ---- to_tile / oob ------------------------------------------------------

def test_to_tile_floors_toward_negative_infinity():
    pos = torch.tensor([[0.0, 0.0], [1.9, 2.1], [-0.1, -0.1]])
    ix, iy = terrain.to_tile(pos)
    assert torch.equal(ix, torch.tensor([0, 1, -1]))
    assert torch.equal(iy, torch.tensor([0, 2, -1]))


def test_oob_detects_out_of_range():
    cfg = _cfg(map_h=10, map_w=10)
    ix = torch.tensor([-1, 0, 9, 10])
    iy = torch.tensor([0, 0, 9, 0])
    assert torch.equal(terrain.oob(ix, iy, cfg), torch.tensor([True, False, False, True]))


# ---- sample ---------------------------------------------------------------

def test_sample_oob_returns_blocked():
    cfg = _cfg(map_h=10, map_w=10)
    tiles = _tiles(10, 10)
    mask = _unit_mask(tiles)
    pos = torch.tensor([[-5.0, 5.0], [50.0, 5.0], [5.0, -5.0], [5.0, 50.0]])
    map_id = torch.zeros(4, dtype=torch.int64)
    blocked = terrain.sample(mask, map_id, pos, cfg)
    assert torch.all(blocked)


def test_sample_matches_grid_content():
    cfg = _cfg(map_h=10, map_w=10)
    tiles = _tiles(10, 10)
    tiles[5, 5] = Tile.WALL
    mask = _unit_mask(tiles)
    pos = torch.tensor([[5.5, 5.5], [2.5, 2.5]])  # inside the wall tile vs. open floor
    map_id = torch.zeros(2, dtype=torch.int64)
    blocked = terrain.sample(mask, map_id, pos, cfg)
    assert torch.equal(blocked, torch.tensor([True, False]))


def test_sample_broadcasts_map_id_over_entity_dim():
    cfg = _cfg(map_h=10, map_w=10)
    tiles_a = _tiles(10, 10)
    tiles_b = _tiles(10, 10)
    tiles_b[5, 5] = Tile.WALL
    mask = torch.stack([TILE_BLOCKS_UNIT[tiles_a], TILE_BLOCKS_UNIT[tiles_b]], dim=0)  # (2,H,W)
    # N=2 envs, E=3 entities, all at the same spot; env 0 uses map 0 (open), env 1 uses map 1 (wall)
    pos = torch.full((2, 3, 2), 5.5)
    map_id = torch.tensor([0, 1])  # (N,) broadcasts to (N, E)
    blocked = terrain.sample(mask, map_id, pos, cfg)
    assert blocked.shape == (2, 3)
    assert torch.all(~blocked[0])
    assert torch.all(blocked[1])


# ---- circle_blocked ---------------------------------------------------------

def test_circle_blocked_true_when_radius_reaches_wall():
    cfg = _cfg(map_h=10, map_w=10)
    tiles = _tiles(10, 10)
    tiles[5, 6] = Tile.WALL
    mask = _unit_mask(tiles)
    pos = torch.tensor([[5.5, 5.5]])  # center of tile (5,5), wall is the tile to its east
    map_id = torch.zeros(1, dtype=torch.int64)
    far = terrain.circle_blocked(mask, map_id, pos, torch.tensor([0.1]), cfg)
    near = terrain.circle_blocked(mask, map_id, pos, torch.tensor([0.6]), cfg)
    assert not bool(far)
    assert bool(near)


# ---- resolve_move -----------------------------------------------------------

def test_resolve_move_slides_along_a_vertical_wall():
    cfg = _cfg(map_h=12, map_w=12)
    tiles = _tiles(12, 12)
    tiles[:, 6] = Tile.WALL  # a wall column blocking x-movement past col 6
    mask = _unit_mask(tiles)
    map_id = torch.zeros(1, dtype=torch.int64)
    pos = torch.tensor([[5.5, 5.5]])
    delta = torch.tensor([[1.0, 1.0]])  # would cross the wall in x; y alone is clear
    radius = torch.tensor([0.1])
    new_pos = terrain.resolve_move(mask, map_id, pos, delta, radius, cfg)
    assert new_pos[0, 0].item() == 5.5  # x move rejected
    assert new_pos[0, 1].item() > 5.5  # y move still applied -- this is the "slide"


def test_resolve_move_walking_straight_into_wall_stops():
    cfg = _cfg(map_h=12, map_w=12)
    tiles = _tiles(12, 12)
    tiles[:, 6] = Tile.WALL
    mask = _unit_mask(tiles)
    map_id = torch.zeros(1, dtype=torch.int64)
    pos = torch.tensor([[5.5, 5.5]])
    delta = torch.tensor([[1.0, 0.0]])
    radius = torch.tensor([0.1])
    new_pos = terrain.resolve_move(mask, map_id, pos, delta, radius, cfg)
    assert torch.allclose(new_pos, pos)


def test_resolve_move_open_floor_moves_freely():
    cfg = _cfg(map_h=12, map_w=12)
    tiles = _tiles(12, 12)
    mask = _unit_mask(tiles)
    map_id = torch.zeros(1, dtype=torch.int64)
    pos = torch.tensor([[5.5, 5.5]])
    delta = torch.tensor([[0.3, -0.2]])
    radius = torch.tensor([0.1])
    new_pos = terrain.resolve_move(mask, map_id, pos, delta, radius, cfg)
    assert torch.allclose(new_pos, pos + delta)


# ---- march -----------------------------------------------------------------

def test_march_hits_wall_at_expected_distance():
    cfg = _cfg(map_h=20, map_w=20, los_step_tiles=0.1, max_ray_tiles=30.0)
    tiles = _tiles(20, 20)
    mask = _unit_mask(tiles)  # border wall only; interior open
    map_id = torch.zeros(1, dtype=torch.int64)
    p0 = torch.tensor([[10.0, 10.0]])
    dir_ = torch.tensor([[1.0, 0.0]])
    max_dist = torch.tensor([30.0])
    hit, hit_pos, hit_t = terrain.march(mask, map_id, p0, dir_, max_dist, cfg)
    assert bool(hit)
    # wall tile starts at x=19 (border column), 9 tiles east of x=10
    assert abs(hit_t.item() - 9.0) <= cfg.los_step_tiles + 1e-6


def test_march_no_hit_when_max_dist_too_short():
    cfg = _cfg(map_h=20, map_w=20, los_step_tiles=0.5, max_ray_tiles=30.0)
    tiles = _tiles(20, 20)
    mask = _unit_mask(tiles)
    map_id = torch.zeros(1, dtype=torch.int64)
    p0 = torch.tensor([[10.0, 10.0]])
    dir_ = torch.tensor([[1.0, 0.0]])
    max_dist = torch.tensor([3.0])  # wall is 9 tiles away, well beyond this
    hit, _, _ = terrain.march(mask, map_id, p0, dir_, max_dist, cfg)
    assert not bool(hit)


def test_march_blank_csv_terminates_at_border_every_direction():
    overrides = yaml.safe_load((CONFIGS / "presets" / "debug_tiny.yaml").read_text())
    cfg = load_config(CONFIGS / "default.yaml", overrides=overrides)
    bank = build_map_bank(cfg, device="cpu")

    n_dirs = 16
    idx = torch.arange(n_dirs, dtype=torch.int64)
    dirs = geo.dir_from_bin(idx, n_dirs)
    p0 = torch.full((n_dirs, 2), 10.0)  # blank.csv is 20x20, center at (10, 10)
    map_id = torch.zeros(n_dirs, dtype=torch.int64)
    max_dist = torch.full((n_dirs,), float(cfg.max_ray_tiles))

    hit, hit_pos, hit_t = terrain.march(bank.blocks_unit, map_id, p0, dirs, max_dist, cfg)
    assert torch.all(hit)
    assert torch.all(hit_t > 0.0)
    assert torch.all(hit_t < cfg.max_ray_tiles)
    # every hit lands within the map's bounds
    assert torch.all(hit_pos >= 0.0) and torch.all(hit_pos <= cfg.map_w)


def test_march_no_hxw_allocation_scales_with_batch_not_map():
    # a large map with a small batch should be just as cheap as a small map -- if this
    # accidentally materialized an (H, W) grid per sample, this would be the first place
    # a naive implementation blows up.
    cfg = _cfg(map_h=200, map_w=200, los_step_tiles=0.5, max_ray_tiles=50.0)
    tiles = _tiles(200, 200)
    mask = _unit_mask(tiles)
    map_id = torch.zeros(3, dtype=torch.int64)
    p0 = torch.full((3, 2), 100.0)
    dir_ = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    max_dist = torch.full((3,), 50.0)
    hit, hit_pos, hit_t = terrain.march(mask, map_id, p0, dir_, max_dist, cfg)
    assert hit.shape == (3,)
    assert hit_pos.shape == (3, 2)


# ---- line_of_sight (Notice 4: only WALL blocks) ------------------------------

def test_line_of_sight_blocked_by_wall_only():
    cfg = _cfg(map_h=10, map_w=10, los_step_tiles=0.1, max_ray_tiles=20.0)
    for tile_type, expect_blocked in (
        (Tile.WALL, True),
        (Tile.WATER, False),
        (Tile.BUSH, False),
        (Tile.FENCE, False),
    ):
        tiles = _tiles(10, 10)
        tiles[5, 5] = tile_type
        bank = _FakeBank(blocks_proj=_proj_mask(tiles))
        map_id = torch.zeros(1, dtype=torch.int64)
        p0 = torch.tensor([[2.5, 5.5]])
        p1 = torch.tensor([[8.5, 5.5]])
        los = terrain.line_of_sight(bank, map_id, p0, p1, cfg)
        assert bool(los) == (not expect_blocked), f"tile {tile_type!r}"


def test_line_of_sight_open_floor_is_visible():
    cfg = _cfg(map_h=10, map_w=10)
    tiles = _tiles(10, 10)
    bank = _FakeBank(blocks_proj=_proj_mask(tiles))
    map_id = torch.zeros(1, dtype=torch.int64)
    p0 = torch.tensor([[2.0, 2.0]])
    p1 = torch.tensor([[7.0, 7.0]])
    assert bool(terrain.line_of_sight(bank, map_id, p0, p1, cfg))


# ---- batched leading-dim smoke test -------------------------------------------

def test_batched_leading_dims_64x7():
    cfg = _cfg(map_h=30, map_w=30, los_step_tiles=0.5, max_ray_tiles=20.0)
    tiles = _tiles(30, 30)
    tiles[15, 15] = Tile.WALL
    unit_mask = _unit_mask(tiles)
    proj_mask = _proj_mask(tiles)
    bank = _FakeBank(blocks_proj=proj_mask)

    batch = (64, 7)
    pos = torch.rand(*batch, 2) * 28 + 1
    delta = (torch.rand(*batch, 2) - 0.5) * 2
    radius = torch.rand(*batch) * 0.3 + 0.1
    dir_ = geo.normalize(torch.rand(*batch, 2) - 0.5)
    max_dist = torch.rand(*batch) * 10 + 1
    map_id = torch.zeros(64, dtype=torch.int64)  # (N,) broadcasts to (N, E)

    blocked = terrain.sample(unit_mask, map_id, pos, cfg)
    cblocked = terrain.circle_blocked(unit_mask, map_id, pos, radius, cfg)
    moved = terrain.resolve_move(unit_mask, map_id, pos, delta, radius, cfg)
    hit, hit_pos, hit_t = terrain.march(unit_mask, map_id, pos, dir_, max_dist, cfg)
    los = terrain.line_of_sight(bank, map_id, pos, pos + delta, cfg)

    for out in (blocked, cblocked, hit, hit_t, los):
        assert out.shape == batch
    assert moved.shape == (*batch, 2)
    assert hit_pos.shape == (*batch, 2)
    for out in (moved, hit_pos, hit_t):
        assert not torch.any(torch.isnan(out))
