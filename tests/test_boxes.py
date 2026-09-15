import pytest
import torch

from brawl_sim.config import load_config, build_params, validate
from brawl_sim.constants import Tile, TILE_BLOCKS_UNIT
from brawl_sim.core import boxes, terrain
from brawl_sim.core.state import allocate
from brawl_sim.maps.loader import MAX_BOX_SPOTS

CONFIGS_DEFAULT = "configs/default.yaml"


def _cfg_and_params(n_envs=1, max_boxes=16, n_boxes=8, cubes=None):
    cfg = load_config(CONFIGS_DEFAULT, overrides={
        "world": {"map_h": 20, "map_w": 20},
        "entities": {"n_enemies": 1},
        "limits": {"max_boxes": max_boxes},
        "boxes": {"n_boxes": n_boxes},
        "zone": {"enabled": False},
        "cubes": cubes or {},
    })
    import yaml
    spec = {
        **yaml.safe_load(open(CONFIGS_DEFAULT).read()),
        **yaml.safe_load(open("configs/brawlers.yaml").read()),
    }
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    params = build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=spec)
    return cfg, params, gen


def _walled_floor(size=20):
    tiles = torch.full((size, size), int(Tile.FLOOR), dtype=torch.int64)
    tiles[0, :] = Tile.WALL
    tiles[-1, :] = Tile.WALL
    tiles[:, 0] = Tile.WALL
    tiles[:, -1] = Tile.WALL
    return tiles


def _fake_bank(spot_list, n_envs_maps=1, tiles=None):
    """spot_list: list of (x, y) floor-tile centers -- mirrors loader.box_points()' contract
    that box_spots only ever contains passable-tile coordinates. Padded to MAX_BOX_SPOTS like
    the real MapBank (loader._pad_slots). `blocks_unit` is built from `tiles` through the same
    lookup table the loader uses; the default is open floor inside a wall border."""
    n = len(spot_list)
    padded = spot_list + [(0.0, 0.0)] * (MAX_BOX_SPOTS - n)
    box_spots = torch.tensor([padded] * n_envs_maps, dtype=torch.float32)
    n_box_spots = torch.tensor([n] * n_envs_maps, dtype=torch.int64)
    tiles = _walled_floor() if tiles is None else tiles

    class _FakeBank:
        pass

    bank = _FakeBank()
    bank.box_spots = box_spots
    bank.n_box_spots = n_box_spots
    bank.blocks_unit = TILE_BLOCKS_UNIT[tiles].unsqueeze(0).expand(n_envs_maps, -1, -1)
    return bank


def _fresh_state(cfg, n_envs=1):
    state = allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)
    state.map_id.fill_(0)
    return state


_KNOWN_FLOOR_SPOTS = [(float(x) + 0.5, 5.5) for x in range(2, 18)]  # 16 distinct floor spots


# ---- spawn_boxes (acceptance) ---------------------------------------------------------

def test_boxes_never_spawn_on_a_blocked_tile():
    cfg, params, gen = _cfg_and_params(n_boxes=8)
    state = _fresh_state(cfg)
    bank = _fake_bank(_KNOWN_FLOOR_SPOTS)
    reset_mask = torch.ones(1, dtype=torch.bool)
    tiles = _walled_floor()

    boxes.spawn_boxes(state, reset_mask, bank, params, cfg, gen)

    ix = state.box_pos[..., 0].floor().to(torch.int64)
    iy = state.box_pos[..., 1].floor().to(torch.int64)
    blocked = TILE_BLOCKS_UNIT[tiles[iy, ix]]
    assert not torch.any(blocked & state.box_alive)


def test_no_two_boxes_share_a_spot():
    cfg, params, gen = _cfg_and_params(n_boxes=8)
    state = _fresh_state(cfg)
    bank = _fake_bank(_KNOWN_FLOOR_SPOTS)
    reset_mask = torch.ones(1, dtype=torch.bool)

    boxes.spawn_boxes(state, reset_mask, bank, params, cfg, gen)

    alive_pos = state.box_pos[0][state.box_alive[0]]
    unique_pos = torch.unique(alive_pos, dim=0)
    assert unique_pos.shape[0] == alive_pos.shape[0]
    assert alive_pos.shape[0] == 8


def test_n_boxes_greater_than_n_box_spots_clamps():
    cfg, params, gen = _cfg_and_params(n_boxes=50)  # way more than the map can offer
    state = _fresh_state(cfg)
    spots = _KNOWN_FLOOR_SPOTS[:5]  # only 5 real spots on this "map"
    bank = _fake_bank(spots)
    reset_mask = torch.ones(1, dtype=torch.bool)

    boxes.spawn_boxes(state, reset_mask, bank, params, cfg, gen)
    assert int(state.box_alive[0].sum()) == 5


def test_n_boxes_greater_than_max_boxes_clamps():
    cfg, params, gen = _cfg_and_params(max_boxes=4, n_boxes=50)
    state = _fresh_state(cfg)
    bank = _fake_bank(_KNOWN_FLOOR_SPOTS)  # 16 spots available, but only 4 box slots exist
    reset_mask = torch.ones(1, dtype=torch.bool)

    boxes.spawn_boxes(state, reset_mask, bank, params, cfg, gen)
    assert int(state.box_alive[0].sum()) == 4
    assert state.box_pos.shape[1] == 4


def test_spawn_boxes_only_touches_masked_rows():
    cfg, params, gen = _cfg_and_params(n_envs=4, n_boxes=8)
    state = allocate(cfg, n_envs=4, device="cpu", verbose=False)
    state.map_id.fill_(0)
    bank = _fake_bank(_KNOWN_FLOOR_SPOTS, n_envs_maps=1)

    full_mask = torch.ones(4, dtype=torch.bool)
    boxes.spawn_boxes(state, full_mask, bank, params, cfg, gen)
    snapshot_pos = state.box_pos.clone()
    snapshot_alive = state.box_alive.clone()
    snapshot_hp = state.box_hp.clone()

    partial_mask = torch.tensor([False, True, False, True])
    boxes.spawn_boxes(state, partial_mask, bank, params, cfg, gen)

    for row in (0, 2):
        assert torch.equal(state.box_pos[row], snapshot_pos[row])
        assert torch.equal(state.box_alive[row], snapshot_alive[row])
        assert torch.equal(state.box_hp[row], snapshot_hp[row])


def test_spawned_box_hp_matches_params():
    cfg, params, gen = _cfg_and_params(n_boxes=3)
    state = _fresh_state(cfg)
    bank = _fake_bank(_KNOWN_FLOOR_SPOTS)
    reset_mask = torch.ones(1, dtype=torch.bool)

    boxes.spawn_boxes(state, reset_mask, bank, params, cfg, gen)
    alive = state.box_alive[0]
    assert torch.allclose(state.box_hp[0][alive], params.box_hp[0].expand(int(alive.sum())))
    assert torch.allclose(state.box_max_hp[0][alive], params.box_hp[0].expand(int(alive.sum())))
    assert torch.all(state.box_hp[0][~alive] == 0)


# ---- damage_boxes / resolve_broken_boxes ------------------------------------------------

def test_damage_boxes_clamps_at_zero():
    cfg, params, gen = _cfg_and_params(n_boxes=1)
    state = _fresh_state(cfg)
    bank = _fake_bank(_KNOWN_FLOOR_SPOTS)
    reset_mask = torch.ones(1, dtype=torch.bool)
    boxes.spawn_boxes(state, reset_mask, bank, params, cfg, gen)

    dmg = torch.full_like(state.box_hp, 999999.0)
    boxes.damage_boxes(state, dmg)
    assert torch.all(state.box_hp >= 0)
    assert torch.all(state.box_hp == 0)


def test_breaking_a_box_yields_exactly_cubes_per_box():
    cfg, params, gen = _cfg_and_params(n_boxes=1)
    state = _fresh_state(cfg)
    bank = _fake_bank(_KNOWN_FLOOR_SPOTS)
    reset_mask = torch.ones(1, dtype=torch.bool)
    boxes.spawn_boxes(state, reset_mask, bank, params, cfg, gen)

    dmg = torch.zeros_like(state.box_hp)
    dmg[0, 0] = 999999.0
    boxes.damage_boxes(state, dmg)

    newly_broken = boxes.resolve_broken_boxes(state, bank, params, cfg, gen)
    assert bool(newly_broken[0, 0])
    assert not bool(state.box_alive[0, 0])
    assert int(state.boxes_broken[0]) == 1

    claimed = state.pku_alive[0]
    assert int(claimed.sum()) == 1
    pickup_idx = int(torch.nonzero(claimed)[0, 0])
    assert int(state.pku_cubes[0, pickup_idx]) == int(params.cubes_per_box[0].item())


def test_resolve_broken_boxes_ignores_undamaged_boxes():
    cfg, params, gen = _cfg_and_params(n_boxes=2)
    state = _fresh_state(cfg)
    bank = _fake_bank(_KNOWN_FLOOR_SPOTS)
    reset_mask = torch.ones(1, dtype=torch.bool)
    boxes.spawn_boxes(state, reset_mask, bank, params, cfg, gen)

    newly_broken = boxes.resolve_broken_boxes(state, bank, params, cfg, gen)
    assert not torch.any(newly_broken)
    assert int(state.boxes_broken[0]) == 0
    assert torch.all(state.box_alive[0] == (state.box_hp[0] > 0))


# ---- batched / no-NaN smoke ------------------------------------------------------------

def test_batched_smoke():
    cfg, params, gen = _cfg_and_params(n_envs=8, n_boxes=8)
    state = allocate(cfg, n_envs=8, device="cpu", verbose=False)
    state.map_id.fill_(0)
    bank = _fake_bank(_KNOWN_FLOOR_SPOTS, n_envs_maps=1)
    reset_mask = torch.ones(8, dtype=torch.bool)

    boxes.spawn_boxes(state, reset_mask, bank, params, cfg, gen)
    assert not torch.any(torch.isnan(state.box_pos))
    assert torch.all(state.box_alive.sum(dim=1) == 8)

    dmg = torch.rand(8, state.box_pos.shape[1]) * 5000
    boxes.damage_boxes(state, dmg)
    newly_broken = boxes.resolve_broken_boxes(state, bank, params, cfg, gen)
    assert newly_broken.shape == (8, state.box_pos.shape[1])
    assert not torch.any(torch.isnan(state.pku_pos))


# ---- where a broken crate's cube lands ----------------------------------------------------

def _spawned(n_envs=64, spots=None, tiles=None, cubes=None):
    spots = spots or _KNOWN_FLOOR_SPOTS
    cfg, params, gen = _cfg_and_params(n_envs=n_envs, n_boxes=len(spots), cubes=cubes)
    state = allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)
    state.map_id.fill_(0)
    bank = _fake_bank(spots, tiles=tiles)
    boxes.spawn_boxes(state, torch.ones(n_envs, dtype=torch.bool), bank, params, cfg, gen)
    return cfg, params, gen, bank, state


def test_the_default_config_scatters_cubes_by_the_band_measured_on_footage():
    cfg, *_ = _cfg_and_params()
    assert (cfg.box_scatter_min_tiles, cfg.box_scatter_max_tiles) == (0.3, 1.8)


def test_a_cube_lands_inside_the_band_in_every_direction():
    spot = [(10.5, 10.5)]  # 8.5+ tiles from the border wall: every draw is legal
    cfg, params, gen, bank, state = _spawned(n_envs=512, spots=spot)
    land = boxes.landing_spots(state, bank, params, cfg, gen)[:, 0]
    off = land - state.box_pos[:, 0]
    d = off.norm(dim=-1)
    assert d.min() >= cfg.box_scatter_min_tiles - 1e-5
    assert d.max() <= cfg.box_scatter_max_tiles + 1e-5
    # Not collapsed onto one spot or one side: all four quadrants are used.
    quadrant = (off[:, 0] > 0).to(torch.int64) * 2 + (off[:, 1] > 0).to(torch.int64)
    assert set(quadrant.tolist()) == {0, 1, 2, 3}


def test_a_cube_never_lands_where_a_unit_cannot_stand():
    tiles = _walled_floor()
    tiles[:, 11] = Tile.WALL           # a wall one tile east of the crate
    tiles[12, :] = Tile.WATER          # water two rows south
    spot = [(10.5, 10.5)]
    cfg, params, gen, bank, state = _spawned(n_envs=512, spots=spot, tiles=tiles)
    land = boxes.landing_spots(state, bank, params, cfg, gen)[:, 0]
    radius = params.unit_radius
    assert not torch.any(terrain.circle_blocked(bank.blocks_unit, state.map_id, land, radius, cfg))
    assert torch.any((land - state.box_pos[:, 0]).norm(dim=-1) > 0)  # the scatter still happened


def test_a_crate_with_no_legal_spot_in_the_band_drops_on_itself():
    tiles = torch.full((20, 20), int(Tile.WALL), dtype=torch.int64)
    tiles[10, 10] = Tile.FLOOR         # a one-tile pocket: nothing 0.3+ tiles out fits a unit
    spot = [(10.5, 10.5)]
    cfg, params, gen, bank, state = _spawned(n_envs=64, spots=spot, tiles=tiles)
    land = boxes.landing_spots(state, bank, params, cfg, gen)[:, 0]
    assert torch.equal(land, state.box_pos[:, 0])


def test_with_the_scatter_off_the_cube_lands_on_the_crate_and_draws_nothing():
    cubes = {"box_scatter_min_tiles": 0.0, "box_scatter_max_tiles": 0.0}
    cfg, params, gen, bank, state = _spawned(n_envs=4, cubes=cubes)
    before = gen.get_state()
    assert torch.equal(boxes.landing_spots(state, bank, params, cfg, gen), state.box_pos)
    assert torch.equal(gen.get_state(), before)


def test_a_broken_crate_puts_its_pickup_at_its_landing_spot():
    spot = [(10.5, 10.5)]
    cfg, params, gen, bank, state = _spawned(n_envs=64, spots=spot)
    boxes.damage_boxes(state, torch.full_like(state.box_hp, 999999.0))
    boxes.resolve_broken_boxes(state, bank, params, cfg, gen)
    assert torch.all(state.pku_alive.sum(dim=1) == 1)
    pos = state.pku_pos[state.pku_alive]          # (64, 2), one per env
    d = (pos - state.box_pos[:, 0]).norm(dim=-1)
    assert torch.all((d >= cfg.box_scatter_min_tiles - 1e-5) & (d <= cfg.box_scatter_max_tiles + 1e-5))


def test_validate_rejects_a_scatter_band_upside_down():
    import yaml
    band = {"cubes": {"box_scatter_min_tiles": 2.0, "box_scatter_max_tiles": 1.0}}
    cfg = load_config(CONFIGS_DEFAULT, overrides=band)   # the real map size, so nothing else trips
    spec = {**yaml.safe_load(open(CONFIGS_DEFAULT).read()),
            **yaml.safe_load(open("configs/brawlers.yaml").read())}
    params = build_params(cfg, n_envs=1, device="cpu", gen=torch.Generator().manual_seed(0), spec=spec)
    with pytest.raises(ValueError, match="box_scatter"):
        validate(cfg, params)
