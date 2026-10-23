"""Rifle archetype tests: FIRE and AIM only. Kite-and-strafe movement and the low-HP flee are
personality behaviors, tested in tests/test_personality.py (as KITE and as the RETREAT mode
respectively).

Two tests are deliberately GONE (2026-10-20). `test_a_box_target_never_trips_the_lateral_hold`
could not fail: its fallback target stood still and its box sat inside the hold-free range, and
tests/test_combat_rules.py::test_a_stationary_loot_box_never_triggers_the_lateral_hold now owns
that claim. The no-NaN loop is one all-kinds test in test_combat_rules.py.
"""
import torch

from tests.bot_fixtures import FakeBank, build_targeting, cfg_and_params, fresh_state, grid
from brawl_sim.bots import combat_rules, perception, policy
from brawl_sim.constants import DeathCause, Kind, Person, Tile
from brawl_sim.core import combat, geometry as geo, hero, movement, projectiles, stats


def _run_tick(state, bank, params, cfg, gen):
    """One manual tick through the real dispatcher. See tests/test_sniper.py::_run_tick for why
    step_count must advance."""
    vis = perception.visibility(state, bank, params, cfg)
    intent = policy.all_bot_intents(state, vis, bank, params, cfg, gen)

    move_dir = intent.move_dir.clone()
    move_dir[:, 0] = 0.0
    movement.apply_movement(state, move_dir, bank, params, cfg)

    fire_mask = intent.fire.clone()
    fire_mask[:, 0] = False
    damage = stats.effective_damage(state.ent_kind, state.ent_cubes, params)
    projectiles.spawn_volley(state, fire_mask, state.ent_pos, intent.aim_dir, intent.aim_point,
                             state.ent_kind, damage, params, cfg)

    hero.tick_timers(state, params, cfg)
    dmg_ent, _dmg_by, _dmg_box, _heal, _charge = projectiles.step_projectiles(state, bank, params, cfg)
    attacker = torch.ones_like(state.ent_last_hit_by)
    combat.apply_damage(state, dmg_ent, int(DeathCause.COMBAT), attacker, params, cfg)
    state.step_count += 1
    return intent, dmg_ent


# ---- fire gating ---------------------------------------------------------------------

def test_fire_requires_los():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, enemy_kind=Kind.BOT_RIFLE)
    tiles = grid(20, 20)
    tiles[10, 7] = Tile.WALL  # between rifle and hero (x in [7, 8)), neither in a bush
    bank = FakeBank(tiles)

    state.ent_pos[0, 0] = torch.tensor([5.5, 10.5])
    state.ent_pos[0, 1] = torch.tensor([9.5, 10.5])

    _vis, tgt = build_targeting(state, bank, params, cfg)
    assert state.ent_target[0, 1].item() == 0  # targeting ignores the wall

    fire, _, _ = combat_rules.combat(state, tgt, bank, params, cfg, gen)
    assert not bool(fire[0, 1])


def test_fire_requires_0p9_range_not_just_base_range():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, enemy_kind=Kind.BOT_RIFLE)
    bank = FakeBank(grid(20, 20))

    attack_range = params.attack_range[0, int(Kind.BOT_RIFLE)].item()
    state.ent_pos[0, 0] = torch.tensor([3.0, 10.0])
    # inside base attack_range but outside 0.9*attack_range
    state.ent_pos[0, 1] = torch.tensor([3.0 + 0.95 * attack_range, 10.0])
    _vis, tgt = build_targeting(state, bank, params, cfg)
    fire, _, _ = combat_rules.combat(state, tgt, bank, params, cfg, gen)
    assert not bool(fire[0, 1])  # the fan would diverge too much this far out

    state.ent_pos[0, 1] = torch.tensor([3.0 + 0.5 * attack_range, 10.0])  # well inside 0.9x
    _vis, tgt = build_targeting(state, bank, params, cfg)
    fire, _, _ = combat_rules.combat(state, tgt, bank, params, cfg, gen)
    assert bool(fire[0, 1])


def test_holds_fire_on_fast_lateral_mover_beyond_0p6_range():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, enemy_kind=Kind.BOT_RIFLE)
    bank = FakeBank(grid(20, 20))

    attack_range = params.attack_range[0, int(Kind.BOT_RIFLE)].item()
    state.ent_pos[0, 0] = torch.tensor([3.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([3.0 + 0.7 * attack_range, 10.0])  # beyond 0.6x, within 0.9x
    state.ent_vel[0, 0] = torch.tensor([0.0, 5.0])  # hero strafing fast, perpendicular to the shot

    _vis, tgt = build_targeting(state, bank, params, cfg)
    fire, _, _ = combat_rules.combat(state, tgt, bank, params, cfg, gen)
    assert not bool(fire[0, 1])  # holds fire on the fast lateral mover

    state.ent_pos[0, 1] = torch.tensor([3.0 + 0.5 * attack_range, 10.0])  # within 0.6x now
    _vis, tgt = build_targeting(state, bank, params, cfg)
    fire, _, _ = combat_rules.combat(state, tgt, bank, params, cfg, gen)
    assert bool(fire[0, 1])  # close enough that the hold-fire rule no longer applies


# ---- full-loop acceptance ----------------------------------------------------------

def test_kiting_rifle_reaches_and_holds_range_and_lands_a_hit():
    """The rifle as a KITE. personality.RANGE_DEADBAND of 1.5 keeps the whole settle band inside
    firing distance (only 0.9*range is fire-eligible), so there is no sliver of it from which the
    bot cannot shoot."""
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, enemy_kind=Kind.BOT_RIFLE, person=Person.KITE)
    bank = FakeBank(grid(20, 20))

    state.ent_pos[0, 0] = torch.tensor([3.0, 10.0])    # hero, stationary
    state.ent_pos[0, 1] = torch.tensor([17.0, 10.0])   # 14 tiles away

    attack_range = params.attack_range[0, int(Kind.BOT_RIFLE)].item()
    fraction = params.desired_range_fraction[0, int(Kind.BOT_RIFLE)].item()
    desired = fraction * attack_range
    # See tests/test_sniper.py's copy of this line: a precondition on the setup, not a copy of
    # the configured number.
    assert 0 < fraction <= 1.0 and desired < 14.0

    hit_tick = None
    for step in range(300):
        _intent, dmg_ent = _run_tick(state, bank, params, cfg, gen)
        if hit_tick is None and dmg_ent[0, 0].item() > 0:
            hit_tick = step
        if step == 249:
            dist = geo.dist(state.ent_pos[0, 0], state.ent_pos[0, 1]).item()
            assert (desired - 4.0) <= dist <= (desired + 4.0)

    assert hit_tick is not None, "the kiting rifle never landed a hit"


def test_never_misses_stationary_target_at_half_range_with_zero_noise():
    """ONE blast, and every pellet of it connects: with the fan collapsed to a single line the
    five pellets fly together, so they land on the same tick for the full volley's damage.

    `_run_tick` books no ammo or cooldown (env._attack_phase does that), so the single shot is
    enforced here by emptying the clip once the bot fires. Until 2026-10-20 it was not: the bot
    fired every decision tick, and this passed if any of ~15 volleys connected."""
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, enemy_kind=Kind.BOT_RIFLE)
    bank = FakeBank(grid(20, 20))
    k = int(Kind.BOT_RIFLE)
    params.aim_noise_std_rad[:, k] = 0.0
    params.proj_spread_rad[:, k] = 0.0  # collapse the fan to one line

    attack_range = params.attack_range[0, k].item()
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([10.0, 10.0 + 0.5 * attack_range])

    fired, damage = False, 0.0
    for _ in range(60):
        intent, dmg_ent = _run_tick(state, bank, params, cfg, gen)
        if not fired and bool(intent.fire[0, 1]):
            fired = True
            state.ent_ammo[0, 1] = 0.0          # that was the shot: no ammo, and no reload
            params.reload_seconds[:, k] = 1e9
        damage = dmg_ent[0, 0].item()
        if damage > 0:
            break
        state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
        state.ent_pos[0, 1] = torch.tensor([10.0, 10.0 + 0.5 * attack_range])

    assert fired, "the rifle never fired"
    pellet = stats.effective_damage(state.ent_kind, state.ent_cubes, params)[0, 1].item()
    volley = int(params.proj_count[0, k]) * pellet
    assert abs(damage - volley) < 1e-3, f"the one blast dealt {damage}, not its full {volley}"
