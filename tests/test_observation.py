import math

import pytest
import torch
import yaml

from brawl_sim.bots import perception
from brawl_sim.config import build_params, load_config
from brawl_sim.constants import N_KINDS, N_PROJ_KINDS, Kind
from brawl_sim.core import camera
from brawl_sim.core import hero as hero_mod
from brawl_sim.core import observation as obs_mod
from brawl_sim.core import obs_select
from brawl_sim.core import spawn
from brawl_sim.core import zone as zone_mod
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

    # 12 base channels, then one enemy_hist plane per history slot (Step H2, history_frames 3).
    assert obs["view"].shape == (n_envs, 15, cfg.view_h, cfg.view_w)
    assert obs["view"].dtype == torch.uint8
    assert obs["world"].shape == (n_envs, 15, cfg.map_h, cfg.map_w)

    assert obs["action_mask"]["move"].shape == (n_envs, cfg.n_move_bins + 1)
    # Step D2 widened the attack dim to 3 (0 = nothing, 1 = attack, 2 = super) and Step G3 to 4
    # (3 = gadget). Checked against cfg AND a literal: the cfg comparison alone cannot fail.
    assert obs["action_mask"]["attack"].shape == (n_envs, cfg.action_nvec[1])
    assert obs["action_mask"]["attack"].shape == (n_envs, 4)

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


# ---- camera-limited reveal (OBS_PARITY_TASKS.md C3) --------------------------------------

def _camera_scene(enemy_offsets, hero=(30.5, 30.5), objects=(), overrides=None):
    """The 60x60 `walled` map (no bush tile anywhere, so concealment never enters), the hero
    at `hero`, one enemy per (dx, dy) offset from it and, per `objects` offset, a live
    projectile, box and pickup in that slot. Returns cfg, state, obs and `vis`."""
    merged = {
        "entities": {"n_enemies": len(enemy_offsets)},
        "world": {"maps": ["walled"], "map_selection": "fixed", "fixed_map": "walled"},
        **(overrides or {}),
    }
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=1, tiny=False, overrides=merged)
    state = _reset(cfg, params, bank, gen, spec, 1)
    assert not bank.is_bush[int(state.map_id[0])].any()
    state.ent_pos[0, 0] = torch.tensor(hero)
    for j, (dx, dy) in enumerate(enemy_offsets, start=1):
        state.ent_pos[0, j] = torch.tensor([hero[0] + dx, hero[1] + dy])
    state.ent_alive[0, :] = True
    for i, (dx, dy) in enumerate(objects):
        for pos, alive in ((state.prj_pos, state.prj_alive), (state.box_pos, state.box_alive),
                           (state.pku_pos, state.pku_alive)):
            pos[0, i] = torch.tensor([hero[0] + dx, hero[1] + dy])
            alive[0, i] = True
    vis, los = _vis_los(state, bank, params, cfg)
    return cfg, state, obs_mod.build_obs(state, bank, vis, los, params, cfg), vis


def test_the_reveal_is_the_camera_quad_not_the_crop_and_not_the_map():
    """Hero on tile (30, 30) mid-map, camera tracking. The quad reaches about 13 tiles east and
    10.8 north at the hero's row/column but only 7.2 south; the 21 x 13 crop reaches 10 and 6.
    A (8 east) is in both; B (12 east) and D (9 north) are on screen but outside the crop, so
    the entity flag and the world grid reveal them while the view grid cannot hold them; C (16
    east) and E (9 south) are off screen: alive, not revealed, and not `hidden_by_bush`."""
    offsets = [(8, 0), (12, 0), (16, 0), (0, -9), (0, 9)]
    cfg, state, obs, vis = _camera_scene(offsets)
    ent = obs["entities"]
    assert vis[0, 0, 1:].all(), "no bush anywhere: concealment alone reveals all five"
    assert ent["revealed_to_hero"][0, 1:].tolist() == [True, True, False, True, False]
    assert ent["alive"][0, 1:].all()
    assert not ent["hidden_by_bush"][0].any(), "the flag is about bushes, not the screen"
    assert ent["hero_revealed_to"][0, 1:].all(), "the bots' side stays whole-map"

    ch = obs_select.channel_index(cfg)
    world = obs["world"][0]
    for (dx, dy), revealed in zip(offsets, (1, 1, 0, 1, 0)):
        iy, ix = 30 + dy, 30 + dx
        assert int(world[ch["enemy_any"], iy, ix]) == 1, (dx, dy)
        assert int(world[ch["enemy_revealed"], iy, ix]) == revealed, (dx, dy)
        assert int(world[ch["enemy_hidden"], iy, ix]) == 1 - revealed, (dx, dy)
    # The crop's origin is (20, 24): only A falls inside it, at row 6, column 18.
    view = obs["view"][0]
    assert view[ch["enemy_any"]].nonzero().tolist() == [[6, 18]]
    assert view[ch["enemy_revealed"]].nonzero().tolist() == [[6, 18]]
    assert not view[ch["enemy_hidden"]].any()


def test_a_camera_pinned_at_the_west_edge_reveals_the_far_east_of_its_window():
    """Hero 2.5 tiles from the west edge: the camera stops at x 12.1, so an enemy 21 tiles east
    of the hero is 11.4 from the camera centre and on screen, although a hero-centred window
    would have lost it at 10. The world grid agrees; the hero-centred crop ends 10 east."""
    cfg, state, obs, _ = _camera_scene([(21, 0)], hero=(2.5, 30.5))
    assert bool(obs["entities"]["revealed_to_hero"][0, 1])
    ch = obs_select.channel_index(cfg)
    assert int(obs["world"][0, ch["enemy_revealed"], 30, 23]) == 1
    assert not obs["view"][0, ch["enemy_any"]].any()


def test_in_view_is_the_camera_window_for_every_kind_of_object():
    """12 tiles east of a mid-map hero is on screen (the quad reaches about 13 there) and 16
    is not, for entities, projectiles, boxes and pickups alike; the 21-wide crop would have
    refused both (OBS_PARITY_TASKS.md C4)."""
    _, _, obs, _ = _camera_scene([(12, 0), (16, 0)], objects=[(12, 0), (16, 0)])
    assert obs["entities"]["in_view"][0, 1:].tolist() == [True, False]
    for group in ("projectiles", "boxes", "pickups"):
        assert obs[group]["in_view"][0, :2].tolist() == [True, False], group


@pytest.mark.parametrize("hero, expected", [
    ((30.0, 30.0), False),   # mid-map: the camera is tracking
    ((3.0, 30.0), True),     # 9.1 past the west stop at x 12.1
    ((30.0, 57.0), True),    # 2.5 past the south stop at y 54.5
    ((11.5, 30.0), False),   # 0.6 past
    ((9.5, 30.0), True),     # 2.6 past
    ((10.5, 30.0), False),   # 1.6 past: under the 2-tile threshold
])
def test_hero_near_edge_says_the_camera_has_stopped_following(hero, expected):
    _, _, obs, _ = _camera_scene([(2, 0)], hero=hero)
    assert bool(obs["hero"]["near_edge"][0]) is expected, hero


def test_hero_near_edge_threshold_is_camera_edge_flag_tiles():
    _, _, obs, _ = _camera_scene([(2, 0)], hero=(10.5, 30.0),
                                 overrides={"camera": {"edge_flag_tiles": 1.0}})
    assert bool(obs["hero"]["near_edge"][0])


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
    assert not obs_mod.build_obs(state, bank, vis, los, params, cfg)["zone"]["active"][0], \
        "active is a latch (C5): nothing has marked gas as seen yet"
    state.zone_step[0] = 1  # the hand-set rect is a shrunk one, so there is gas to see
    zone_mod.mark_seen(state, camera.camera_centre(state.ent_pos[:, 0], cfg), cfg)
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


def test_zone_hero_margin_local_saturates_at_the_sensing_horizon():
    """`hero_margin_local` is the field configs/agent_obs_deploy2.yaml trains on, and the reason it
    exists is that the deployed loop has no safe rect to subtract from -- it scans a sticky map of
    gas it has SEEN (brawl_deployment/perception/grid.py) and stops at a horizon. So the sim
    saturates at the same number, and this test pins the three cases that matter: a margin inside
    the horizon passes through, one beyond it saturates, and a NEGATIVE margin (hero standing in
    the gas) saturates the same way on the other side -- the blindness is symmetric, because
    nearby clear ground is as unsensed from deep inside the gas as distant gas is from safety.

    If this drifts from brawl_deployment's own clamp, training and deployment disagree about what
    a large margin means, silently. See BRAWL_DEPLOYMENT_DESIGN.md 9.14."""
    n_envs = 1
    cfg, params, bank, gen, spec = _cfg_params_bank(
        n_envs=n_envs,
        overrides={"zone": {"enabled": True, "margin_horizon_tiles": 4.0}},
    )
    state = _reset(cfg, params, bank, gen, spec, n_envs)
    assert cfg.zone_margin_horizon_tiles == 4.0

    state.zone_lo[0] = torch.tensor([3.0, 3.0])
    state.zone_hi[0] = torch.tensor([13.0, 13.0])
    state.ent_pos[0, 0] = torch.tensor([15.0, 8.0])  # 2 tiles outside the rect on the +x side

    vis, los = _vis_los(state, bank, params, cfg)
    z = obs_mod.build_obs(state, bank, vis, los, params, cfg)["zone"]

    raw, local = z["hero_margin"][0], z["hero_margin_local"][0]
    assert local.shape == raw.shape
    assert abs(local[0].item() - 4.0) < 1e-4    # raw 12.0, beyond the horizon -> saturated
    assert abs(local[1].item() - (-2.0)) < 1e-4  # raw -2.0, inside it -> untouched, sign kept
    assert abs(local[2].item() - 4.0) < 1e-4    # raw 5.0 -> saturated
    assert abs(local[3].item() - 4.0) < 1e-4    # raw 5.0 -> saturated

    # And the other side of the symmetry: deep in the gas, far from any clear ground.
    state.ent_pos[0, 0] = torch.tensor([19.0, 8.0])
    vis, los = _vis_los(state, bank, params, cfg)
    local = obs_mod.build_obs(state, bank, vis, los, params, cfg)["zone"]["hero_margin_local"][0]
    assert abs(local[1].item() - (-4.0)) < 1e-4  # raw -6.0 -> saturated negative


def test_zone_hero_margin_local_is_the_raw_margin_when_the_horizon_is_wide():
    """The clamp must be a clamp and nothing else -- no rescaling, no offset, no dropped sign.
    Widen the horizon past anything a 20x20 map can produce and the two fields must coincide
    exactly, for every env, not just the hero's own."""
    n_envs = 8
    cfg, params, bank, gen, spec = _cfg_params_bank(
        n_envs=n_envs,
        overrides={"zone": {"enabled": True, "margin_horizon_tiles": 1000.0}},
    )
    state = _reset(cfg, params, bank, gen, spec, n_envs)
    vis, los = _vis_los(state, bank, params, cfg)
    z = obs_mod.build_obs(state, bank, vis, los, params, cfg)["zone"]
    assert torch.equal(z["hero_margin_local"], z["hero_margin"])


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
    assert obs["world"].shape == (n_envs, 15, cfg.map_h, cfg.map_w)
    assert not torch.any(torch.isnan(obs["view"].float()))


# ---- the gadget fields (SIM_OVERHAUL Step G4) ----------------------------------------------
# 18.0 below is Mortis's `gadget_cooldown` in configs/brawlers.yaml, written as a literal: a value
# read back out of params would let a wrong cooldown pass.

def _build(state, bank, params, cfg):
    vis, los = _vis_los(state, bank, params, cfg)
    return obs_mod.build_obs(state, bank, vis, los, params, cfg)


def test_the_gadget_starts_charged():
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=4)
    state = _reset(cfg, params, bank, gen, spec, 4)
    hero = _build(state, bank, params, cfg)["hero"]
    assert hero["gadget_ready"].dtype == torch.bool
    assert hero["gadget_charge_frac"].dtype == torch.float32
    assert hero["gadget_ready"].tolist() == [True, True, True, True]
    assert hero["gadget_charge_frac"].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_the_charge_is_one_minus_the_timer_over_18_seconds():
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=5)
    state = _reset(cfg, params, bank, gen, spec, 5)
    state.ent_gadget_cd[:, 0] = torch.tensor([18.0, 13.5, 9.0, 0.05, 0.0])
    hero = _build(state, bank, params, cfg)["hero"]
    assert hero["gadget_charge_frac"].tolist() == pytest.approx([0.0, 0.25, 0.5, 0.9972222, 1.0], abs=1e-6)
    assert hero["gadget_ready"].tolist() == [False, False, False, False, True]


def test_gadget_ready_is_the_masks_gadget_column_and_only_it_reads_alive():
    """G3 made `hero.gadget_ready(state, params)` the one predicate behind the mask and the field
    reuses it, so the field carries the `alive` term that G4's checklist formula leaves out. The
    fraction has no `alive` term, like `super_charge_frac`."""
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=4)
    state = _reset(cfg, params, bank, gen, spec, 4)
    assert state.ent_kind[:, 0].tolist() == [0, 0, 0, 0]  # Mortis: column 0 is the hero's cooldown
    state.ent_gadget_cd[1, 0] = 9.0                         # env 1: halfway through the cooldown
    state.ent_alive[2, 0] = False                           # env 2: dead, with the timer run out
    state.ent_hp[2, 0] = 0.0
    params.gadget_cooldown[3, 0] = 0.0                      # env 3: a hero kind with no gadget
    obs = _build(state, bank, params, cfg)
    assert obs["hero"]["gadget_ready"].tolist() == [True, False, False, False]
    assert obs["action_mask"]["attack"][:, 3].tolist() == [True, False, False, False]
    assert obs["hero"]["gadget_charge_frac"].tolist() == [1.0, 0.5, 1.0, 0.0]


def _env(action_repeat):
    from brawl_sim.env import BrawlVecEnv

    cfg = load_config(CONFIGS_DEFAULT, overrides={
        **CONFIGS_TINY,
        "sim": {**CONFIGS_TINY["sim"], "action_repeat": action_repeat},
        "engine": {"debug_checks": True, "compile": False},
    })
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0, verbose=False)
    env.reset()
    return env


_IDLE = torch.tensor([[0, 0]])
_THROW = torch.tensor([[0, 3]])  # attack column 3 = the gadget


def test_the_fields_follow_a_throw_through_env_step():
    env = _env(action_repeat=1)
    obs, *_ = env.step(_THROW)
    assert obs["hero"]["gadget_ready"].tolist() == [False]
    assert obs["hero"]["gadget_charge_frac"].tolist() == [0.0]           # the timer reads 18.0
    obs, *_ = env.step(_IDLE)
    assert obs["hero"]["gadget_charge_frac"].tolist() == pytest.approx([0.0027778], abs=1e-6)  # 0.05 / 18
    env.state.ent_gadget_cd[0, 0] = 0.05                                  # the cooldown's last tick
    obs, *_ = env.step(_IDLE)
    assert obs["hero"]["gadget_ready"].tolist() == [True]
    assert obs["hero"]["gadget_charge_frac"].tolist() == [1.0]
    assert obs["action_mask"]["attack"][0].tolist() == [True, True, False, True]


def test_at_the_shipped_decision_rate_the_first_observation_after_a_throw_shows_four_ticks():
    """action_repeat 5: the throw lands on the decision's first sub-tick and the other four count
    the timer down before the observation is built, so the policy never sees 0.0 here. The same
    offset is why the gadget is legal again on the 73rd observation, 18.25 s rather than 18.0
    (SIM_OVERHAUL_STEPS.md G3, behaviour 1)."""
    env = _env(action_repeat=5)
    obs, *_ = env.step(_THROW)
    assert env.state.ent_gadget_cd[0, 0].item() == pytest.approx(17.8, abs=1e-5)
    assert obs["hero"]["gadget_charge_frac"].tolist() == pytest.approx([0.0111111], abs=1e-6)  # 0.2 / 18
    assert obs["hero"]["gadget_ready"].tolist() == [False]


# ---- the history fields and planes (SIM_OVERHAUL Step H2) -------------------------------------
# K = 3 (configs/default.yaml's history_frames), 17 move bins, 4 attack columns and debug_tiny's
# 10 x 14 view are literals here, like the 18.0 above. The view's origin is the hero's tile minus
# (7, 5), so a tile (x, y) lands at view row y - hero_y + 5, column x - hero_x + 7.

_HIST_FIELDS = ("valid", "move_onehot", "attack_onehot", "hp", "ammo_frac", "displacement")


def _hist_state(overrides=None):
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=1, overrides=overrides)
    return cfg, params, bank, _reset(cfg, params, bank, gen, spec, 1)


def test_a_fresh_episode_has_empty_history_in_the_declared_shapes():
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=2)
    state = _reset(cfg, params, bank, gen, spec, 2)
    obs = _build(state, bank, params, cfg)
    hist = obs["hist"]
    assert tuple(hist) == _HIST_FIELDS
    assert {k: (tuple(v.shape), v.dtype) for k, v in hist.items()} == {
        "valid": ((2, 3), torch.bool),
        "move_onehot": ((2, 3, 17), torch.uint8),
        "attack_onehot": ((2, 3, 4), torch.uint8),
        "hp": ((2, 3), torch.float32),
        "ammo_frac": ((2, 3), torch.float32),
        "displacement": ((2, 3, 2), torch.float32),
    }
    for name in _HIST_FIELDS:
        assert not hist[name].any(), name
    assert not obs["view"][:, 12:].any()


def test_every_field_is_zero_in_a_slot_that_is_not_valid():
    """Slot 1 holds values and is marked empty. Its action (7, 2) would one-hot like any other,
    and an empty slot's usual (0, 0) is a real "idle, no attack", so the mask is what makes a slot
    read as nothing."""
    cfg, params, bank, state = _hist_state()
    assert state.ent_kind[0, 0].item() == 0  # Mortis, max_ammo 3
    state.ent_pos[0, 0] = torch.tensor([10.5, 10.5])
    state.hist_valid[0] = torch.tensor([True, False, True])
    state.hist_action[0] = torch.tensor([[5, 1], [7, 2], [16, 3]])
    state.hist_hp[0] = torch.tensor([3000.0, 2000.0, 1000.0])
    state.hist_ammo[0] = torch.tensor([3.0, 1.5, 0.75])
    state.hist_pos[0] = torch.tensor([[11.5, 10.5], [9.0, 9.0], [10.5, 12.0]])
    hist = _build(state, bank, params, cfg)["hist"]
    assert hist["valid"][0].tolist() == [True, False, True]
    assert hist["move_onehot"][0].nonzero().tolist() == [[0, 5], [2, 16]]
    assert hist["attack_onehot"][0].nonzero().tolist() == [[0, 1], [2, 3]]
    assert int(hist["move_onehot"].sum()) == 2 and int(hist["attack_onehot"].sum()) == 2
    assert hist["hp"][0].tolist() == [3000.0, 0.0, 1000.0]
    assert hist["ammo_frac"][0].tolist() == [1.0, 0.0, 0.25]
    assert hist["displacement"][0].tolist() == [[1.0, 0.0], [0.0, 0.0], [0.0, 1.5]]


def test_through_env_step_slot_0_is_the_previous_observation_and_the_action_it_got():
    """Each observation against the one before it: slot 0 is that observation's hero plus the
    action that answered it, and the displacement is where the hero was then minus where it is
    now. The expectation comes from the previous observation's own fields, not from the rings."""
    env = _env(action_repeat=5)
    prev = obs_mod.clone_obs(env.reset())  # a clone: build_obs hands out views into state
    moved = []
    for move, attack in ((5, 1), (9, 0), (13, 0)):
        obs, *_ = env.step(torch.tensor([[move, attack]]))
        hist = obs["hist"]
        assert hist["move_onehot"][0, 0].nonzero().flatten().tolist() == [move]
        assert hist["attack_onehot"][0, 0].nonzero().flatten().tolist() == [attack]
        assert torch.equal(hist["hp"][:, 0], prev["hero"]["hp"])
        assert torch.equal(hist["ammo_frac"][:, 0], prev["hero"]["ammo_frac"])
        assert torch.equal(hist["displacement"][:, 0], prev["hero"]["pos"] - obs["hero"]["pos"])
        moved.append(bool(hist["displacement"][0, 0].abs().sum() > 0))
        prev = obs_mod.clone_obs(obs)
    assert hist["valid"].tolist() == [[True, True, True]]
    assert all(moved), "every decision moved the hero, so no displacement here is a trivial 0 - 0"


def test_an_enemy_seen_two_decisions_ago_lands_in_enemy_hist2_at_its_world_cell():
    """World positions into the CURRENT window, no re-centering: the hero is on tile (12, 11) now,
    so the view's origin is (5, 6) and the enemy's old tile (10, 10) is view row 4, column 5. On
    the full-map grid it is row 10, column 10."""
    cfg, params, bank, state = _hist_state(overrides={"observation": {"include_world_grid": True}})
    state.ent_pos[0, 0] = torch.tensor([12.5, 11.5])
    state.hist_valid[0] = torch.tensor([True, True, False])
    state.hist_enemy_seen[0, 1, 1] = True
    state.hist_enemy_pos[0, 1, 1] = torch.tensor([10.2, 10.7])
    obs = _build(state, bank, params, cfg)
    assert obs["view"][0, 12:].nonzero().tolist() == [[1, 4, 5]]  # 12 + 1 = enemy_hist2
    assert obs["view"][0, 13, 4, 5].item() == 1
    assert obs["world"][0, 12:].nonzero().tolist() == [[1, 10, 10]]


@pytest.mark.parametrize("seen, valid", [(False, True), (True, False)])
def test_a_sighting_that_was_hidden_or_sits_in_an_empty_slot_draws_nothing(seen, valid):
    cfg, params, bank, state = _hist_state()
    state.ent_pos[0, 0] = torch.tensor([12.5, 11.5])
    state.hist_valid[0, 1] = valid
    state.hist_enemy_seen[0, 1, 1] = seen
    state.hist_enemy_pos[0, 1, 1] = torch.tensor([10.2, 10.7])
    assert not _build(state, bank, params, cfg)["view"][0, 12:].any()


def test_the_radius_is_a_9_by_9_block_of_tiles_around_the_hero():
    """Chebyshev distance on TILE indices at history_radius_tiles 4: the hero is on tile (10, 10),
    so tiles 6..14 on each axis are in. A and C sit on the block's edge while standing more than
    4.0 from the hero's position, so a distance on raw positions would drop them. B and D are one
    tile past it and still inside the view, so only the radius can drop them."""
    cfg, params, bank, state = _hist_state(overrides={"entities": {"n_enemies": 4}})
    state.ent_pos[0, 0] = torch.tensor([10.1, 10.5])
    state.hist_valid[0, 0] = True
    state.hist_enemy_seen[0, 0, 1:] = True
    state.hist_enemy_pos[0, 0, 1:] = torch.tensor([
        [14.9, 10.5],  # A: tile (14, 10), 4 away -> view row 5, column 11
        [15.0, 10.5],  # B: tile (15, 10), 5 away
        [6.0, 14.9],   # C: tile (6, 14), 4 away on both axes -> view row 9, column 3
        [5.9, 10.5],   # D: tile (5, 10), 5 away
    ])
    plane = _build(state, bank, params, cfg)["view"][0, 12]
    assert plane.nonzero().tolist() == [[5, 11], [9, 3]]
    assert int(plane.sum()) == 2


def test_the_radius_is_measured_from_where_the_hero_is_now():
    """The hero stood on tile (3, 10) back then and is on (10, 10) now. P was 1 tile from it then
    and is 6 from it now; Q was 10 away then and is 3 away now."""
    cfg, params, bank, state = _hist_state()
    state.ent_pos[0, 0] = torch.tensor([10.5, 10.5])
    state.hist_valid[0, 0] = True
    state.hist_pos[0, 0] = torch.tensor([3.5, 10.5])
    state.hist_enemy_pos[0, 0, 0] = torch.tensor([3.5, 10.5])  # the hero's own column, as push writes it
    state.hist_enemy_seen[0, 0, 1:] = True
    state.hist_enemy_pos[0, 0, 1:] = torch.tensor([[4.5, 10.5], [13.5, 10.5]])  # P, Q
    plane = _build(state, bank, params, cfg)["view"][0, 12]
    assert plane.nonzero().tolist() == [[5, 10]]  # Q; P would have been row 5, column 1


def test_after_an_idle_step_enemy_hist1_is_the_last_revealed_plane_cut_to_the_block():
    """End to end through env.step. The hero has not moved, so neither has the window, and one
    idle decision later `enemy_hist1` is exactly the previous observation's `enemy_revealed`
    plane (channel 6) cut to the 9 x 9 block."""
    env = _env(action_repeat=1)
    env.state.ent_pos[0, 0] = torch.tensor([10.5, 10.5])  # tile (10, 10): view origin (3, 5)
    env.state.ent_pos[0, 1] = torch.tensor([12.5, 11.5])  # tile (12, 11): inside the block
    env.state.ent_pos[0, 2] = torch.tensor([16.5, 10.5])  # tile (16, 10): 6 away, still in view
    prev = obs_mod.clone_obs(env._build_observation())    # also re-stashes the visibility push reads
    assert prev["view"][0, 6].nonzero().tolist() == [[5, 13], [6, 9]]
    obs, *_ = env.step(_IDLE)
    assert obs["hero"]["tile"][0].tolist() == [10, 10]
    block = torch.zeros(10, 14, dtype=torch.uint8)
    block[1:10, 3:12] = 1
    assert torch.equal(obs["view"][0, 12], prev["view"][0, 6] * block)
    assert obs["view"][0, 12].nonzero().tolist() == [[6, 9]]
    assert not obs["view"][0, 13:].any()


@pytest.mark.parametrize("frames, channels", [(1, 13), (4, 16)])
def test_the_history_width_follows_history_frames(frames, channels):
    cfg, params, bank, gen, spec = _cfg_params_bank(
        n_envs=2, overrides={"observation": {"history_frames": frames, "include_world_grid": True}})
    state = _reset(cfg, params, bank, gen, spec, 2)
    obs = _build(state, bank, params, cfg)
    assert obs["view"].shape == (2, channels, 10, 14)
    assert obs["world"].shape == (2, channels, 20, 20)
    assert obs["hist"]["move_onehot"].shape == (2, frames, 17)
    assert obs["hist"]["displacement"].shape == (2, frames, 2)
