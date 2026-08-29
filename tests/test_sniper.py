"""Sniper archetype tests. After Step 41's combat/movement split this file covers FIRE and AIM
only; the movement behaviors Step 17 originally put in sniper.policy (hold 0.85 of attack_range,
drift toward bushes when idle, flee below 40% HP) are now personality behaviors and are tested in
tests/test_personality.py -- the range-holding one as KITE, the bush one as CAMPER/TRAPPER.
"""
import torch

from tests.bot_fixtures import FakeBank, build_targeting, cfg_and_params, fresh_state, grid
from brawl_sim.bots import combat_rules, policy
from brawl_sim.constants import DeathCause, Kind, Person, Tile
from brawl_sim.core import combat, geometry as geo, hero, movement, projectiles, stats


def _run_tick(state, bank, params, cfg, gen, hero_moves=False, hero_fires=False):
    """One manual tick driven by the REAL dispatcher (policy.all_bot_intents), which already
    zeroes entity 0's intent itself -- before Step 41 this helper had to mimic that rule by hand
    because the dispatcher didn't exist when Step 17 was written."""
    intent = policy.all_bot_intents(state, _vis(state, bank, params, cfg), bank, params, cfg, gen)

    move_dir = intent.move_dir.clone()
    if not hero_moves:
        move_dir[:, 0] = 0.0
    movement.apply_movement(state, move_dir, bank, params, cfg)

    fire_mask = intent.fire.clone()
    if not hero_fires:
        fire_mask[:, 0] = False
    damage = stats.effective_damage(state.ent_kind, state.ent_cubes, params)
    projectiles.spawn_volley(state, fire_mask, state.ent_pos, intent.aim_dir, intent.aim_point,
                             state.ent_kind, damage, params, cfg)

    hero.tick_timers(state, params, cfg)
    dmg_ent, _dmg_by, _dmg_box, _heal = projectiles.step_projectiles(state, bank, params, cfg)
    attacker = torch.ones_like(state.ent_last_hit_by)  # only entity 1 ever fires here
    combat.apply_damage(state, dmg_ent, int(DeathCause.COMBAT), attacker, params, cfg)
    # Section 4 phase 15. Not optional: all_bot_intents' decision-period gate is
    # `(step_count + entity_index) % decision_period == 0`, so a step_count frozen at 0 lets
    # entity 1 fire only if its own period happens to divide 1 -- with bot_sniper's period of 4
    # it would never fire at all, and every acceptance test below would fail for a reason that
    # has nothing to do with the archetype.
    state.step_count += 1
    return intent, dmg_ent


def _vis(state, bank, params, cfg):
    from brawl_sim.bots import perception
    return perception.visibility(state, bank, params, cfg)


# ---- fire gating ---------------------------------------------------------------------

def test_fire_requires_los_even_though_target_is_visible():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params)
    tiles = grid(20, 20)
    tiles[10, 7] = Tile.WALL  # between sniper and hero (x in [7, 8)), neither in a bush
    bank = FakeBank(tiles)

    state.ent_pos[0, 0] = torch.tensor([5.5, 10.5])
    state.ent_pos[0, 1] = torch.tensor([9.5, 10.5])

    _vis_m, tgt = build_targeting(state, bank, params, cfg)
    assert state.ent_target[0, 1].item() == 0  # targeting ignores the wall (Notice 4)

    fire, _aim_dir, _aim_point = combat_rules.combat(state, tgt, bank, params, cfg, gen)
    assert not bool(fire[0, 1])  # but firing needs a real physical shot


def test_fire_requires_target_within_attack_range():
    cfg, params, gen = cfg_and_params(n_enemies=1, map_h=40, map_w=40)
    state = fresh_state(cfg, params)
    bank = FakeBank(grid(40, 40))
    attack_range = params.attack_range[0, int(Kind.BOT_SNIPER)].item()

    state.ent_pos[0, 0] = torch.tensor([5.0, 20.0])
    state.ent_pos[0, 1] = torch.tensor([5.0 + attack_range + 3.0, 20.0])
    _vis_m, tgt = build_targeting(state, bank, params, cfg)
    fire, _, _ = combat_rules.combat(state, tgt, bank, params, cfg, gen)
    assert not bool(fire[0, 1])

    state.ent_pos[0, 1] = torch.tensor([5.0 + attack_range - 1.0, 20.0])
    _vis_m, tgt = build_targeting(state, bank, params, cfg)
    fire, _, _ = combat_rules.combat(state, tgt, bank, params, cfg, gen)
    assert bool(fire[0, 1])


def test_no_nan_in_combat_output_ever():
    cfg, params, gen = cfg_and_params(n_envs=8, n_enemies=3)
    state = fresh_state(cfg, params, n_envs=8)
    tiles = grid(20, 20)
    tiles[5, 5] = Tile.BUSH
    bank = FakeBank(tiles)

    for _ in range(300):
        state.ent_pos.uniform_(1, 19)
        state.ent_hp.uniform_(0, 1)
        state.ent_hp.mul_(state.ent_max_hp)
        state.ent_alive.copy_(torch.rand(state.ent_alive.shape) > 0.3)
        state.ent_ammo.uniform_(0, 5)
        state.ent_target.random_(-1, cfg.n_entities)

        _vis_m, tgt = build_targeting(state, bank, params, cfg)
        _fire, aim_dir, aim_point = combat_rules.combat(state, tgt, bank, params, cfg, gen)

        for t in (aim_dir, aim_point):
            assert torch.all(torch.isfinite(t))


# ---- full-loop acceptance ----------------------------------------------------------

def test_kiting_sniper_reaches_and_holds_range_and_lands_a_hit():
    """The Step 17 acceptance test, re-expressed: range-holding is now the KITE personality's
    job, and the sniper's own preferred fraction of its range comes from the per-kind
    `desired_range_fraction` param (Step E1; it was policy.RANGE_FRACTION_BY_KIND)."""
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, person=Person.KITE)
    bank = FakeBank(grid(20, 20))

    state.ent_pos[0, 0] = torch.tensor([3.0, 10.0])    # hero, stationary
    state.ent_pos[0, 1] = torch.tensor([17.0, 10.0])   # 14 tiles away, well inside the map

    attack_range = params.attack_range[0, int(Kind.BOT_SNIPER)].item()
    fraction = params.desired_range_fraction[0, int(Kind.BOT_SNIPER)].item()
    desired = fraction * attack_range
    # Precondition, not a restatement of the config: the bot starts 14 tiles out, so the test
    # only means anything if its ideal range is somewhere strictly inside its own reach.
    assert 0 < fraction <= 1.0 and desired < 14.0
    deadband = 1.5

    hit_tick = None
    for step in range(200):
        _intent, dmg_ent = _run_tick(state, bank, params, cfg, gen)
        if hit_tick is None and dmg_ent[0, 0].item() > 0:
            hit_tick = step
        if step == 149:
            dist = geo.dist(state.ent_pos[0, 0], state.ent_pos[0, 1]).item()
            # Wider tolerance than Step 17's +-0.5: the KITE blend also carries a strafe term, so
            # the bot orbits the ideal radius instead of parking exactly on it.
            assert (desired - deadband - 2.0) <= dist <= (desired + deadband + 2.0)

    assert hit_tick is not None and hit_tick < 200


def test_never_misses_stationary_target_at_half_range_with_zero_noise():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params)
    bank = FakeBank(grid(20, 20))
    params.aim_noise_std_rad[:, int(Kind.BOT_SNIPER)] = 0.0

    attack_range = params.attack_range[0, int(Kind.BOT_SNIPER)].item()
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([10.0, 10.0 + 0.5 * attack_range])

    hit = False
    for _ in range(60):  # generous upper bound on travel time across half range
        _intent, dmg_ent = _run_tick(state, bank, params, cfg, gen)
        if dmg_ent[0, 0].item() > 0:
            hit = True
            break
        # keep both stationary so this stays a pure "does a fired shot connect" check
        state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
        state.ent_pos[0, 1] = torch.tensor([10.0, 10.0 + 0.5 * attack_range])

    assert hit
