import math

import torch
import yaml

from brawl_sim.bots import perception
from brawl_sim.config import build_params, load_config
from brawl_sim.constants import N_KINDS, N_PROJ_KINDS, Kind
from brawl_sim.core import hero as hero_mod
from brawl_sim.core import observation as obs_mod
from brawl_sim.core import spawn
from brawl_sim.core.state import allocate
from brawl_sim.maps.loader import build_map_bank

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())


def _cfg_params_bank(n_envs=8, seed=0, overrides=None, tiny=True):
    merged = dict(CONFIGS_TINY) if tiny else {}
    if overrides:
        merged = {**merged, **overrides}
    cfg = load_config(CONFIGS_DEFAULT, overrides=merged or None)
    spec = {
        **yaml.safe_load(open(CONFIGS_DEFAULT).read()),
        **yaml.safe_load(open("configs/brawlers.yaml").read()),
    }
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    params = build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=spec)
    bank = build_map_bank(cfg, device="cpu")
    return cfg, params, bank, gen, spec


def _reset(cfg, params, bank, gen, spec, n_envs=8):
    state = allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)
    mask = torch.ones(n_envs, dtype=torch.bool)
    spawn.reset_envs(state, mask, bank, params, cfg, gen, spec)
    return state


def _vis_los(state, bank, params, cfg):
    vis = perception.visibility(state, bank, params, cfg)
    los = perception.raw_los(state, bank, cfg)
    return vis, los


def _assert_all_finite(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            _assert_all_finite(v, f"{path}.{k}")
    else:
        if node.is_floating_point():
            assert torch.isfinite(node).all(), f"non-finite values in {path}"


# ---- shapes ---------------------------------------------------------------------------

def test_shapes_match_expected():
    n_envs = 8
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=n_envs, overrides={"observation": {"include_world_grid": True}})
    state = _reset(cfg, params, bank, gen, spec, n_envs)
    vis, los = _vis_los(state, bank, params, cfg)

    obs = obs_mod.build_obs(state, bank, vis, los, params, cfg)
    E, P, B, U = cfg.n_entities, cfg.max_projectiles, cfg.max_boxes, cfg.max_pickups

    assert obs["hero"]["pos"].shape == (n_envs, 2)
    assert obs["hero"]["facing_vec"].shape == (n_envs, 2)
    assert obs["hero"]["rank"].shape == (n_envs,)

    assert obs["entities"]["pos"].shape == (n_envs, E, 2)
    # One-hot widths come from the enums, not from literals: they were written as 5 and 4, and
    # Step B1 added Proj.SUPER_BOLT, so the projectile literal became wrong while reading as a
    # deliberate assertion about observation shape. Deriving them means adding an enum member
    # updates this test for free, and a width that DISAGREES with its enum still fails.
    assert obs["entities"]["kind_onehot"].shape == (n_envs, E, N_KINDS)
    assert obs["entities"]["dist_rank"].shape == (n_envs, E)
    assert obs["entities"]["privileged"]["move_intent"].shape == (n_envs, E, 2)

    assert obs["projectiles"]["pos"].shape == (n_envs, P, 2)
    assert obs["projectiles"]["kind_onehot"].shape == (n_envs, P, N_PROJ_KINDS)

    assert obs["boxes"]["pos"].shape == (n_envs, B, 2)
    assert obs["pickups"]["pos"].shape == (n_envs, U, 2)

    assert obs["zone"]["hero_margin"].shape == (n_envs, 4)
    assert obs["visibility"]["vis"].shape == (n_envs, E, E)
    assert obs["visibility"]["dist_matrix"].shape == (n_envs, E, E)

    assert obs["view"].shape == (n_envs, 12, cfg.view_h, cfg.view_w)
    assert obs["view"].dtype == torch.uint8
    assert obs["world"].shape == (n_envs, 12, cfg.map_h, cfg.map_w)

    assert obs["action_mask"]["move"].shape == (n_envs, cfg.n_move_bins + 1)
    # Width comes from cfg.action_nvec, not a literal: Step D2 widened the attack dim to 3
    # (0 = nothing, 1 = attack, 2 = super).
    assert obs["action_mask"]["attack"].shape == (n_envs, cfg.action_nvec[1])

    assert obs["meta"]["map_h"].shape == (n_envs,)
    assert torch.all(obs["meta"]["map_h"] == cfg.map_h)


def test_all_float_fields_finite():
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=8)
    state = _reset(cfg, params, bank, gen, spec, 8)
    vis, los = _vis_los(state, bank, params, cfg)
    obs = obs_mod.build_obs(state, bank, vis, los, params, cfg)
    _assert_all_finite(obs)


def test_action_mask_matches_hero_action_mask():
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=8)
    state = _reset(cfg, params, bank, gen, spec, 8)
    vis, los = _vis_los(state, bank, params, cfg)
    obs = obs_mod.build_obs(state, bank, vis, los, params, cfg)
    expected = hero_mod.action_mask(state, params, cfg)
    assert torch.equal(obs["action_mask"]["move"], expected["move"])
    assert torch.equal(obs["action_mask"]["attack"], expected["attack"])


def test_privileged_and_world_grid_toggles():
    cfg, params, bank, gen, spec = _cfg_params_bank(
        n_envs=4, overrides={"observation": {"include_privileged": False, "include_world_grid": False}},
    )
    state = _reset(cfg, params, bank, gen, spec, 4)
    vis, los = _vis_los(state, bank, params, cfg)
    obs = obs_mod.build_obs(state, bank, vis, los, params, cfg)
    assert "privileged" not in obs["entities"]
    assert "world" not in obs

    cfg2, params2, bank2, gen2, spec2 = _cfg_params_bank(
        n_envs=4, overrides={"observation": {"include_privileged": True, "include_world_grid": True}},
    )
    state2 = _reset(cfg2, params2, bank2, gen2, spec2, 4)
    vis2, los2 = _vis_los(state2, bank2, params2, cfg2)
    obs2 = obs_mod.build_obs(state2, bank2, vis2, los2, params2, cfg2)
    assert "privileged" in obs2["entities"]
    assert set(obs2["entities"]["privileged"].keys()) == {
        "target_id", "react_t", "reveal_t", "decision_phase", "move_intent",
    }
    assert "world" in obs2


# ---- dist_rank is a valid permutation ---------------------------------------------------

def test_dist_rank_is_valid_permutation():
    n_envs = 6
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=n_envs)
    state = _reset(cfg, params, bank, gen, spec, n_envs)
    # force distinct, well-separated positions so there are no distance ties
    E = cfg.n_entities
    for e in range(E):
        state.ent_pos[:, e, 0] = 2.0 + e * 1.7
        state.ent_pos[:, e, 1] = 10.0
    vis, los = _vis_los(state, bank, params, cfg)
    obs = obs_mod.build_obs(state, bank, vis, los, params, cfg)

    dist_rank = obs["entities"]["dist_rank"]
    expected = torch.arange(E)
    for row in range(n_envs):
        assert torch.equal(torch.sort(dist_rank[row]).values, expected)
    # entity 0 (hero) is always its own nearest (dist 0) -> rank 0
    assert torch.all(dist_rank[:, 0] == 0)


# ---- bush hiding: full information + correct annotation --------------------------------

def test_bush_hidden_enemy_is_still_fully_present():
    n_envs = 1
    # debug_tiny's blank.csv has no bushes at all -- use the real bushy map instead.
    cfg, params, bank, gen, spec = _cfg_params_bank(
        n_envs=n_envs, tiny=False,
        overrides={
            "entities": {"n_enemies": 1},
            "world": {"map_selection": "fixed", "fixed_map": "bushy"},
        },
    )
    state = _reset(cfg, params, bank, gen, spec, n_envs)

    # place the hero away from any bush, the lone enemy inside a known bush tile, far enough
    # that bush_reveal_radius (default 2.0) doesn't kick in, and never having attacked.
    map_idx = int(state.map_id[0])
    bush_tiles = (bank.tiles[map_idx] == 2).nonzero(as_tuple=False)  # Tile.BUSH == 2
    assert bush_tiles.shape[0] > 0
    by, bx = bush_tiles[0].tolist()
    bush_pos = torch.tensor([bx + 0.5, by + 0.5])

    hero_pos = bush_pos + torch.tensor([6.0, 0.0])
    hero_pos[0] = torch.clamp(hero_pos[0], 1.0, cfg.map_w - 2.0)

    state.ent_pos[0, 0] = hero_pos
    state.ent_pos[0, 1] = bush_pos
    state.ent_alive[0, :] = True
    state.ent_reveal_t[0, 1] = 0.0

    dist = (hero_pos - bush_pos).norm().item()
    assert dist > params.bush_reveal_radius[0].item()

    vis, los = _vis_los(state, bank, params, cfg)
    obs = obs_mod.build_obs(state, bank, vis, los, params, cfg)

    assert bool(obs["entities"]["alive"][0, 1])
    assert torch.allclose(obs["entities"]["pos"][0, 1], bush_pos)
    assert not bool(obs["entities"]["revealed_to_hero"][0, 1])
    assert bool(obs["entities"]["hidden_by_bush"][0, 1])

    world = obs["world"]
    iy, ix = int(bush_pos[1]), int(bush_pos[0])
    assert int(world[0, 5, iy, ix]) >= 1  # enemy_any
    assert int(world[0, 6, iy, ix]) == 0  # enemy_revealed
    assert int(world[0, 7, iy, ix]) >= 1  # enemy_hidden


# ---- corner/edge view is roughly half wall ----------------------------------------------

def test_edge_view_is_roughly_half_wall():
    n_envs = 1
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=n_envs)  # blank map, all floor
    state = _reset(cfg, params, bank, gen, spec, n_envs)

    # hero at the map's left edge, vertically centered -> the view window is truncated by
    # off-map WALL padding on (roughly) the left half only.
    state.ent_pos[0, 0] = torch.tensor([0.5, cfg.map_h / 2.0])
    vis, los = _vis_los(state, bank, params, cfg)
    obs = obs_mod.build_obs(state, bank, vis, los, params, cfg)

    wall_frac = obs["view"][0, 0].float().mean().item()
    assert 0.3 < wall_frac < 0.7


# ---- zone fields ------------------------------------------------------------------------

def test_zone_fields():
    n_envs = 1
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=n_envs, overrides={"zone": {"enabled": True}})
    state = _reset(cfg, params, bank, gen, spec, n_envs)

    state.zone_lo[0] = torch.tensor([3.0, 3.0])
    state.zone_hi[0] = torch.tensor([13.0, 13.0])
    state.ent_pos[0, 0] = torch.tensor([15.0, 8.0])  # outside the rect, to the right

    vis, los = _vis_los(state, bank, params, cfg)
    obs = obs_mod.build_obs(state, bank, vis, los, params, cfg)

    z = obs["zone"]
    assert bool(z["active"][0])
    expected_frac = (10.0 * 10.0) / (cfg.map_w * cfg.map_h)
    assert abs(z["safe_area_frac"][0].item() - expected_frac) < 1e-5
    margin = z["hero_margin"][0]
    assert abs(margin[0].item() - (15.0 - 3.0)) < 1e-4  # x - lo.x
    assert abs(margin[1].item() - (13.0 - 15.0)) < 1e-4  # hi.x - x (negative: outside)
    assert abs(margin[2].item() - (8.0 - 3.0)) < 1e-4
    assert abs(margin[3].item() - (13.0 - 8.0)) < 1e-4


# ---- projectile threat features ----------------------------------------------------------

def test_projectile_threatens_hero():
    n_envs = 1
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=n_envs)
    state = _reset(cfg, params, bank, gen, spec, n_envs)

    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.prj_alive[0, 0] = True
    state.prj_owner[0, 0] = 1  # not the hero
    state.prj_pos[0, 0] = torch.tensor([5.0, 10.0])
    state.prj_vel[0, 0] = torch.tensor([3.0, 0.0])  # heading straight at the hero
    state.prj_radius[0, 0] = 0.3

    vis, los = _vis_los(state, bank, params, cfg)
    obs = obs_mod.build_obs(state, bank, vis, los, params, cfg)
    assert bool(obs["projectiles"]["threatens_hero"][0, 0])
    assert obs["projectiles"]["closest_dist"][0, 0].item() < 0.1


# ---- batched / no-NaN smoke on the real default config -----------------------------------

def test_batched_smoke_default_config():
    n_envs = 16
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=n_envs, tiny=False)
    state = _reset(cfg, params, bank, gen, spec, n_envs)
    vis, los = _vis_los(state, bank, params, cfg)
    obs = obs_mod.build_obs(state, bank, vis, los, params, cfg)
    _assert_all_finite(obs)
    assert obs["world"].shape == (n_envs, 12, cfg.map_h, cfg.map_w)
    assert not torch.any(torch.isnan(obs["view"].float()))
