"""Mortis's Super: charge, action space, piercing bolt, lifesteal.

Its own file because the mechanic spans four modules (hero/projectiles/combat/env) and is the one
place the action space is three-valued. Deliberately exercises everything through the KIND-generic
path -- `hero.super_ready`, `params.super_*`, `spawn_supers(fire_mask=(N,E))` -- rather than
special-casing entity 0, because bots are expected to get supers and the machinery is already
shaped for that.
"""
import math

import pytest
import torch
import yaml

from brawl_sim.config import build_params, load_config
from brawl_sim.constants import TILE_BLOCKS_PROJ, Kind, Proj, ProjClass, Tile
from brawl_sim.core import combat, hero
from brawl_sim.core import projectiles as proj
from brawl_sim.core.state import allocate
from brawl_sim.env import BrawlVecEnv

CONFIGS = "configs"
_HERO = int(Kind.HERO_MORTIS)
_TINY = yaml.safe_load(open(f"{CONFIGS}/presets/debug_tiny.yaml"))


class _FakeBank:
    def __init__(self, tiles):
        self.blocks_proj = TILE_BLOCKS_PROJ[tiles].unsqueeze(0)


def _cfg_and_params(n_enemies=3, map_h=30, map_w=30):
    cfg = load_config(f"{CONFIGS}/default.yaml", overrides={
        "world": {"map_h": map_h, "map_w": map_w},
        "entities": {"n_enemies": n_enemies},
        "limits": {"max_projectiles": 32},
        "zone": {"enabled": False},
    })
    spec = {**yaml.safe_load(open(f"{CONFIGS}/default.yaml")),
            **yaml.safe_load(open(f"{CONFIGS}/brawlers.yaml"))}
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    return cfg, build_params(cfg, n_envs=1, device="cpu", gen=gen, spec=spec)


def _state(cfg, params):
    state = allocate(cfg, n_envs=1, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    state.map_id.fill_(0)
    state.ent_kind[0, 0] = _HERO
    for e in range(1, cfg.n_entities):
        state.ent_kind[0, e] = int(Kind.BOT_SNIPER)
    state.ent_max_hp.fill_(1e9)
    state.ent_hp.fill_(1e9)
    return state


def _open_map(h=30, w=30, wall_col=None):
    tiles = torch.zeros(h, w, dtype=torch.int64)
    tiles[0, :] = tiles[-1, :] = tiles[:, 0] = tiles[:, -1] = int(Tile.WALL)
    if wall_col is not None:
        tiles[:, wall_col] = int(Tile.WALL)
    return tiles


# ---- charge -----------------------------------------------------------------------------

def test_charge_accrues_to_the_configured_hit_count_and_caps_there():
    cfg, params = _cfg_and_params()
    state = _state(cfg, params)
    needed = int(params.super_charge_hits[0, _HERO])

    assert not bool(hero.super_ready(state, params)[0, 0])
    for _ in range(needed):
        hero.add_super_charge(state, torch.tensor([[1] + [0] * (cfg.n_entities - 1)], dtype=torch.int32), params)
    assert int(state.ent_super_charge[0, 0]) == needed
    assert bool(hero.super_ready(state, params)[0, 0])

    hero.add_super_charge(state, torch.tensor([[99] + [0] * (cfg.n_entities - 1)], dtype=torch.int32), params)
    assert int(state.ent_super_charge[0, 0]) == needed, "charge banked past full"


def test_charge_fraction_ramps_then_saturates():
    cfg, params = _cfg_and_params()
    state = _state(cfg, params)
    needed = int(params.super_charge_hits[0, _HERO])
    for expected, hits in ((0.0, 0), (1.0 / needed, 1), (1.0, needed)):
        state.ent_super_charge.fill_(0)
        hero.add_super_charge(state, torch.full((1, cfg.n_entities), hits, dtype=torch.int32), params)
        assert abs(float(hero.super_charge_frac(state, params)[0, 0]) - expected) < 1e-5


def test_a_kind_without_a_super_never_charges_or_becomes_ready():
    """Every bot resolves super_charge_hits to 0. The gate is on that field, not on entity index,
    which is what will let a future bot super be a config block rather than a rewrite."""
    cfg, params = _cfg_and_params()
    state = _state(cfg, params)
    hero.add_super_charge(state, torch.full((1, cfg.n_entities), 50, dtype=torch.int32), params)
    for e in range(1, cfg.n_entities):
        assert int(state.ent_super_charge[0, e]) == 0
        assert not bool(hero.super_ready(state, params)[0, e])
        assert float(hero.super_charge_frac(state, params)[0, e]) == 0.0


# ---- action space -----------------------------------------------------------------------

def test_action_mask_exposes_super_only_when_charged():
    cfg, params = _cfg_and_params()
    state = _state(cfg, params)
    state.ent_ammo[:, 0] = 3.0
    state.ent_attack_cd[:, 0] = 0.0
    state.ent_dash_t[:, 0] = 0.0

    mask = hero.action_mask(state, params, cfg)["attack"]
    assert mask.shape == (1, 4)   # [no-fire, attack, super, gadget]
    assert bool(mask[0, 0]) and bool(mask[0, 1]) and not bool(mask[0, 2])

    state.ent_super_charge[0, 0] = int(params.super_charge_hits[0, _HERO])
    assert bool(hero.action_mask(state, params, cfg)["attack"][0, 2])


def test_super_needs_no_ammo_but_still_respects_cooldown_and_dashing():
    """It costs CHARGE, so an empty clip must not block it -- but it must not become a way to
    sidestep the cooldown or fire mid-dash either."""
    cfg, params = _cfg_and_params()
    state = _state(cfg, params)
    state.ent_super_charge[0, 0] = int(params.super_charge_hits[0, _HERO])
    state.ent_ammo[:, 0] = 0.0
    state.ent_attack_cd[:, 0] = 0.0
    state.ent_dash_t[:, 0] = 0.0

    mask = hero.action_mask(state, params, cfg)["attack"]
    assert not bool(mask[0, 1]), "no ammo, so the ordinary attack must be masked"
    assert bool(mask[0, 2]), "super must not require ammo"

    state.ent_attack_cd[:, 0] = 0.2
    assert not bool(hero.action_mask(state, params, cfg)["attack"][0, 2])
    state.ent_attack_cd[:, 0] = 0.0
    state.ent_dash_t[:, 0] = 0.1
    assert not bool(hero.action_mask(state, params, cfg)["attack"][0, 2])


def test_decode_action_maps_two_to_super_and_one_to_attack():
    cfg, params = _cfg_and_params()
    state = _state(cfg, params)
    state.ent_ammo[:, 0] = 3.0
    state.ent_super_charge[0, 0] = int(params.super_charge_hits[0, _HERO])

    _mv, fire, sup, _gadget, _auto = hero.decode_action(torch.tensor([[0, 1]]), state, params, cfg)
    assert bool(fire[0]) and not bool(sup[0])
    _mv, fire, sup, _gadget, _auto = hero.decode_action(torch.tensor([[0, 2]]), state, params, cfg)
    assert not bool(fire[0]) and bool(sup[0])
    _mv, fire, sup, _gadget, _auto = hero.decode_action(torch.tensor([[0, 0]]), state, params, cfg)
    assert not bool(fire[0]) and not bool(sup[0])


def test_an_uncharged_super_request_is_a_silent_no_op():
    cfg, params = _cfg_and_params()
    state = _state(cfg, params)
    state.ent_super_charge.fill_(0)
    _mv, _fire, sup, _gadget, _auto = hero.decode_action(torch.tensor([[0, 2]]), state, params, cfg)
    assert not bool(sup[0])


# ---- the bolt ---------------------------------------------------------------------------

def _fire_super(cfg, params, state, direction=(1.0, 0.0)):
    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 0] = True
    aim = torch.zeros(1, cfg.n_entities, 2)
    aim[0, 0] = torch.tensor(list(direction))
    proj.spawn_supers(state, fire, state.ent_pos.clone(), aim, params, cfg)
    return int(torch.nonzero(state.prj_alive[0])[0])


def test_bolt_uses_the_super_stat_block_not_the_weapon_one():
    cfg, params = _cfg_and_params()
    state = _state(cfg, params)
    state.ent_pos[0, 0] = torch.tensor([5.0, 15.0])
    j = _fire_super(cfg, params, state)

    assert int(state.prj_kind[0, j]) == int(Proj.SUPER_BOLT)
    assert int(state.prj_class[0, j]) == int(ProjClass.PROJECTILE)
    assert bool(state.prj_pierce[0, j])
    assert abs(float(state.prj_damage[0, j]) - float(params.super_damage[0, _HERO])) < 1e-3
    assert abs(float(state.prj_dist_left[0, j]) - float(params.super_range[0, _HERO])) < 1e-3
    assert abs(float(state.prj_radius[0, j]) - float(params.super_radius[0, _HERO])) < 1e-3
    assert float(state.prj_aoe[0, j]) == 0.0


def test_bolt_pierces_units_and_walls_and_heals_per_player_hit():
    """The whole ability in one pass: it must pass THROUGH the first victim, THROUGH a wall, hit a
    second victim behind both, damage each exactly once, and heal its owner per player hit."""
    cfg, params = _cfg_and_params()
    state = _state(cfg, params)
    bank = _FakeBank(_open_map(wall_col=12))

    state.ent_pos[0, 0] = torch.tensor([5.0, 15.0])
    state.ent_pos[0, 1] = torch.tensor([9.0, 15.0])    # in front of the wall
    state.ent_pos[0, 2] = torch.tensor([14.0, 15.0])   # BEHIND the wall
    state.ent_pos[0, 3] = torch.tensor([25.0, 25.0])   # nowhere near
    _fire_super(cfg, params, state)

    total = torch.zeros(1, cfg.n_entities)
    healed = 0.0
    for _ in range(80):
        dmg_ent, _dmg_by, _dmg_box, heal_ent, _charge = proj.step_projectiles(state, bank, params, cfg)
        total += dmg_ent
        healed += float(heal_ent[0, 0])
        if not bool(state.prj_alive.any()):
            break

    dmg = float(params.super_damage[0, _HERO])
    assert abs(float(total[0, 1]) - dmg) < 1e-3, "did not hit the first victim exactly once"
    assert abs(float(total[0, 2]) - dmg) < 1e-3, "did not pierce the wall to reach the second victim"
    assert float(total[0, 3]) == 0.0
    assert float(total[0, 0]) == 0.0, "the bolt damaged its own owner"
    assert abs(healed - 2 * float(params.super_heal[0, _HERO])) < 1e-3


def test_bolt_never_damages_the_same_victim_twice():
    """A piercing bolt overlaps a victim for several ticks. Without `prj_hits` it would re-damage
    them on every one -- at 12 tiles/s and a 0.70 radius that is ~2 ticks of double damage, and a
    slower bolt would be far worse. Placed so the bolt sits inside the victim for many ticks."""
    cfg, params = _cfg_and_params()
    state = _state(cfg, params)
    bank = _FakeBank(_open_map())
    params.super_proj_speed[0, _HERO] = 1.0   # crawls, so it overlaps for ~20 ticks

    state.ent_pos[0, 0] = torch.tensor([5.0, 15.0])
    state.ent_pos[0, 1] = torch.tensor([7.0, 15.0])
    for e in range(2, cfg.n_entities):
        state.ent_pos[0, e] = torch.tensor([25.0, 25.0])
    _fire_super(cfg, params, state)

    total = 0.0
    for _ in range(200):
        dmg_ent, _b, _bx, _h, _c = proj.step_projectiles(state, bank, params, cfg)
        total += float(dmg_ent[0, 1])
        if not bool(state.prj_alive.any()):
            break
    assert abs(total - float(params.super_damage[0, _HERO])) < 1e-3, (
        f"victim took {total}, expected exactly one hit of {float(params.super_damage[0, _HERO])}"
    )


def test_hit_memory_is_cleared_when_a_slot_is_reused():
    """`prj_hits` is cleared on DEATH, so a slot handed to a new projectile starts blank. If it
    were not, a second super could pass straight through a victim the FIRST one had hit."""
    cfg, params = _cfg_and_params()
    state = _state(cfg, params)
    bank = _FakeBank(_open_map())
    state.ent_pos[0, 0] = torch.tensor([5.0, 15.0])
    state.ent_pos[0, 1] = torch.tensor([9.0, 15.0])
    for e in range(2, cfg.n_entities):
        state.ent_pos[0, e] = torch.tensor([25.0, 25.0])

    per_shot = []
    for _ in range(2):
        _fire_super(cfg, params, state)
        got = 0.0
        for _ in range(80):
            dmg_ent, _b, _bx, _h, _c = proj.step_projectiles(state, bank, params, cfg)
            got += float(dmg_ent[0, 1])
            if not bool(state.prj_alive.any()):
                break
        per_shot.append(got)

    dmg = float(params.super_damage[0, _HERO])
    assert all(abs(g - dmg) < 1e-3 for g in per_shot), (
        f"per-shot damage was {per_shot}; the second shot should hit just as hard as the first"
    )


def test_lifesteal_is_clamped_to_max_hp_and_never_revives_the_dead():
    cfg, params = _cfg_and_params()
    state = _state(cfg, params)
    state.ent_max_hp[0, 0] = 8000.0
    state.ent_hp[0, 0] = 7000.0
    healed = combat.apply_heal(state, torch.tensor([[5000.0] + [0.0] * (cfg.n_entities - 1)]))
    assert float(state.ent_hp[0, 0]) == 8000.0
    # What it REPORTS is the 1000 that fit, not the 5000 asked for -- info["hp_healed_tick"] and
    # the reward's hp_healed term are downstream of this number.
    assert float(healed[0, 0]) == 1000.0

    state.ent_alive[0, 1] = False
    state.ent_hp[0, 1] = 0.0
    healed = combat.apply_heal(state, torch.tensor([[0.0, 5000.0] + [0.0] * (cfg.n_entities - 2)]))
    assert float(state.ent_hp[0, 1]) == 0.0, "healed a corpse -- check_invariants forbids hp>0 while dead"
    assert float(healed[0, 1]) == 0.0


# ---- aiming the idle super (the game's tap-to-fire; user decision, 2026-09-30) -----------

def test_the_idle_super_aims_at_the_nearest_alive_enemy_in_reach_and_skips_crates():
    """Nearest alive enemy by centre distance, with no crate in the running: the bolt passes
    through crates without damaging them, so a nearer crate must not steal the aim, as it does
    for `hero.auto_aim_target`'s dash. With nothing in reach the facing comes back."""
    cfg, params = _cfg_and_params()
    state = _state(cfg, params)
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_facing[0, 0] = math.pi / 2                      # facing +y
    state.ent_pos[0, 1:] = torch.tensor([50.0, 50.0])          # everyone else far away
    state.box_alive.fill_(False)
    direction, has = hero.super_aim_target(state, params, cfg)
    assert direction.shape == (1, 2) and has.shape == (1,)
    assert not bool(has[0])
    assert torch.allclose(direction[0], torch.tensor([0.0, 1.0]), atol=1e-5)

    # An enemy 6 tiles east: past the dash's 3.77-tile reach, inside the bolt's. +x.
    state.ent_pos[0, 1] = torch.tensor([16.0, 10.0])
    direction, has = hero.super_aim_target(state, params, cfg)
    assert bool(has[0]) and torch.allclose(direction[0], torch.tensor([1.0, 0.0]), atol=1e-5)

    # A crate 1.5 tiles north is nearer, and changes nothing.
    state.box_alive[0, 0] = True
    state.box_pos[0, 0] = torch.tensor([10.0, 8.5])
    direction, has = hero.super_aim_target(state, params, cfg)
    assert bool(has[0]) and torch.allclose(direction[0], torch.tensor([1.0, 0.0]), atol=1e-5)

    # A dead enemy one tile west does not count; a live one 5 tiles north is the new nearest.
    state.ent_alive[0, 2] = False
    state.ent_pos[0, 2] = torch.tensor([9.0, 10.0])
    direction, has = hero.super_aim_target(state, params, cfg)
    assert bool(has[0]) and torch.allclose(direction[0], torch.tensor([1.0, 0.0]), atol=1e-5)
    state.ent_pos[0, 3] = torch.tensor([10.0, 5.0])
    direction, has = hero.super_aim_target(state, params, cfg)
    assert bool(has[0]) and torch.allclose(direction[0], torch.tensor([0.0, -1.0]), atol=1e-5)

    # An enemy ON the hero has no bearing: no target, and the facing again.
    state.ent_pos[0, 1] = torch.tensor([50.0, 50.0])
    state.ent_pos[0, 3] = torch.tensor([10.0, 10.0])
    direction, has = hero.super_aim_target(state, params, cfg)
    assert not bool(has[0])
    assert torch.allclose(direction[0], torch.tensor([0.0, 1.0]), atol=1e-5)


def test_the_idle_super_reach_is_the_bolts_range_plus_both_bodies():
    """`super_range + super_radius + unit_radius`: the farthest centre a straight flight of
    `super_range` can touch. Just outside is no target, just inside is, and a charged long dash
    does not stretch it: the multiplier is the dash's."""
    cfg, params = _cfg_and_params()
    state = _state(cfg, params)
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_pos[0, 1:] = torch.tensor([50.0, 50.0])
    reach = (float(params.super_range[0, _HERO]) + float(params.super_radius[0, _HERO])
             + float(params.unit_radius[0]))
    assert reach == pytest.approx(11.1, abs=0.01)

    state.ent_pos[0, 1] = torch.tensor([10.0 + reach + 0.05, 10.0])
    assert not bool(hero.super_aim_target(state, params, cfg)[1][0])
    state.ent_attack_idle_t[0, 0] = float(params.long_dash_seconds[0, _HERO]) + 1.0
    assert bool(hero.long_dash_ready(state, params)[0, 0])
    assert not bool(hero.super_aim_target(state, params, cfg)[1][0])
    state.ent_pos[0, 1] = torch.tensor([10.0 + reach - 0.05, 10.0])
    assert bool(hero.super_aim_target(state, params, cfg)[1][0])


# ---- integration through the real env ---------------------------------------------------

def test_firing_the_super_through_env_spends_charge_and_breaks_concealment():
    """A super must count as ATTACKING for everything that word implies: it reveals the hero out of
    a bush, resets out-of-combat regen, and spends the long-dash charge. Missing any of those would
    make it a free way to shoot from cover or keep a banked long dash."""
    cfg = load_config(f"{CONFIGS}/default.yaml", overrides={
        "entities": {"n_enemies": 2}, "zone": {"enabled": False},
        "observation": {"include_world_grid": False}})
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0, verbose=False)
    env.reset()
    st = env.state
    st.ent_super_charge[0, 0] = int(env.params.super_charge_hits[0, _HERO])
    st.ent_attack_cd[0, 0] = 0.0
    st.ent_dash_t[0, 0] = 0.0
    st.ent_attack_idle_t[0, 0] = 999.0
    st.ent_out_of_combat_t[0, 0] = 999.0
    st.ent_reveal_t[0, 0] = 0.0

    obs, _r, _t, _tr, _i = env.step(torch.tensor([[0, 2]], dtype=torch.int64))

    assert int(st.ent_super_charge[0, 0]) == 0, "firing did not spend the charge"
    assert float(st.ent_attack_idle_t[0, 0]) < 1.0, "super did not spend the long-dash charge"
    assert float(st.ent_out_of_combat_t[0, 0]) < 1.0, "super did not break out-of-combat regen"
    assert float(st.ent_reveal_t[0, 0]) > 0.0, "super did not break concealment"
    assert not bool(obs["hero"]["super_ready"][0])


def test_an_idle_super_through_env_flies_and_expires_at_its_range():
    """The idle-super mine, through `env.step`: action (0, 2) on a charged super launched the
    bolt along move bin 0's zero `move_dir`, so it sat at its spawn point until reset, hitting
    every player who touched it. It must fly at `super_proj_speed` and die on the tick its range
    runs out, in each case one step covers: idle with an enemy in reach (at the nearest), idle
    with none (along facing), and moving (along the move bin, whatever is in reach).

    A `tick_hook` records every sub-tick, so the lifetime is pinned to the tick, at the shipped
    `action_repeat`: the super goes on sub-tick 1 and `_held` zeroes the column after it."""
    cfg = load_config(f"{CONFIGS}/default.yaml", overrides=_TINY)
    env = BrawlVecEnv(cfg, n_envs=3, device="cpu", seed=0, autoreset=False, verbose=False)
    env.reset()
    st, params = env.state, env.params
    st.box_alive.fill_(False)
    # env 0: idle; bot 1 is 5 tiles off along (0.6, 0.8), bot 2 10 tiles off, both in reach.
    # env 1: idle; both bots ~19 tiles off, out of reach; facing +y.
    # env 2: move bin 1 (+x) with env 0's bots: the bin wins over the enemy in reach.
    hero_at = torch.tensor([[4.0, 10.0], [3.0, 3.0], [4.0, 10.0]])
    st.ent_pos[:, 0] = hero_at
    st.ent_pos[:, 1] = torch.tensor([[7.0, 14.0], [17.0, 16.0], [7.0, 14.0]])
    st.ent_pos[:, 2] = torch.tensor([[10.0, 2.0], [16.0, 17.0], [10.0, 2.0]])
    st.ent_facing[:, 0] = torch.tensor([math.pi, math.pi / 2, math.pi])
    st.ent_super_charge[:, 0] = int(params.super_charge_hits[0, _HERO])
    st.ent_attack_cd[:, 0] = 0.0
    st.ent_dash_t[:, 0] = 0.0
    expected = torch.tensor([[0.6, 0.8], [0.0, 1.0], [1.0, 0.0]])

    def hero_bolts(s):
        return s.prj_alive & (s.prj_kind == int(Proj.SUPER_BOLT)) & (s.prj_owner == 0)  # (N,P)

    trace = []   # per sub-tick: ((N,) live bolt count, (N,2) the live bolt's position)

    def record(e):
        live = hero_bolts(e.state)
        trace.append((live.sum(dim=1), (e.state.prj_pos * live.unsqueeze(-1)).sum(dim=1)))

    env.tick_hook = record
    env.step(torch.tensor([[0, 2], [0, 2], [1, 2]]))

    live = hero_bolts(st)
    assert live.sum(dim=1).tolist() == [1, 1, 1], "each env should have fired exactly one bolt"
    vel = (st.prj_vel * live.unsqueeze(-1)).sum(dim=1)                        # (N,2)
    speed = float(params.super_proj_speed[0, _HERO])
    assert vel.norm(dim=-1).tolist() == pytest.approx([speed] * 3, abs=1e-4), "a bolt is not moving"
    assert torch.allclose(vel / speed, expected, atol=1e-5)

    for _ in range(4):
        env.step(torch.zeros(3, 2, dtype=torch.int64))

    step = speed * cfg.dt
    life = math.ceil(float(params.super_range[0, _HERO]) / step)   # 17 ticks: 16 x 0.6 < 10
    counts = torch.stack([c for c, _ in trace])                     # (ticks, N)
    assert counts.shape[0] == 5 * cfg.action_repeat > life
    assert (counts[:life - 1] == 1).all(), "a bolt died before its range ran out"
    assert (counts[life - 1:] == 0).all(), "a bolt outlived its range"
    for tick in range(life - 1):
        travelled = trace[tick][1] - hero_at
        assert torch.allclose(travelled, expected * step * (tick + 1), atol=1e-4), f"tick {tick + 1}"


def test_super_charge_comes_from_player_hits_not_box_hits():
    """Charge is derived from the attacker x victim matrix, which box damage never enters -- so a
    hero cannot farm his super off crates. Asserted directly because the alternative implementation
    ('did I deal damage this tick') would look identical in every other test."""
    cfg = load_config(f"{CONFIGS}/default.yaml", overrides={
        "entities": {"n_enemies": 2}, "zone": {"enabled": False},
        "observation": {"include_world_grid": False}})
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0, verbose=False)
    env.reset()
    st = env.state
    st.ent_super_charge.fill_(0)

    # Park a box right on top of the hero and dash into it repeatedly; no enemy is in reach.
    st.ent_pos[0, 1] = torch.tensor([1.5, 1.5])
    st.ent_pos[0, 2] = torch.tensor([1.5, 1.5])
    st.ent_alive[0, 1] = False
    st.ent_alive[0, 2] = False
    st.box_alive[0, :] = False
    st.box_alive[0, 0] = True
    st.box_hp[0, 0] = 1e9
    st.box_max_hp[0, 0] = 1e9
    st.box_pos[0, 0] = st.ent_pos[0, 0] + torch.tensor([1.0, 0.0])

    for _ in range(20):
        st.ent_ammo[0, 0] = 3.0
        st.ent_attack_cd[0, 0] = 0.0
        env.step(torch.tensor([[1, 1]], dtype=torch.int64))

    assert int(st.ent_super_charge[0, 0]) == 0, (
        "box damage charged the super -- charge must come from hits on living players only"
    )
