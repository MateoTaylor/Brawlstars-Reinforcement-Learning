"""Artillery archetype tests. FIRE and AIM only after Step 41's combat/movement split.

`test_cover_direction_pulls_toward_a_wall_between_it_and_the_target` is deliberately GONE, not
ported: `artillery._cover_direction` no longer exists. It was movement, it cost 3.8 ms/tick at
n_envs=1024, and steering artillery into cover made it a third archetype that broke contact
instead of applying pressure -- see bots/personality.py's module docstring for the full
reasoning.
"""
import torch

from tests.bot_fixtures import FakeBank, build_targeting, cfg_and_params, fresh_state, grid
from brawl_sim.bots import combat_rules, perception, policy
from brawl_sim.constants import DeathCause, Kind, Person, Tile
from brawl_sim.core import combat, geometry as geo, hero, movement, projectiles, stats


def _run_tick(state, bank, params, cfg, gen):
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
    dmg_ent, _dmg_by, _dmg_box, _heal = projectiles.step_projectiles(state, bank, params, cfg)
    attacker = torch.ones_like(state.ent_last_hit_by)
    combat.apply_damage(state, dmg_ent, int(DeathCause.COMBAT), attacker, params, cfg)
    state.step_count += 1  # see tests/test_sniper.py::_run_tick
    return intent, dmg_ent


# ---- fire gating ---------------------------------------------------------------------

def test_fire_does_not_require_los():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, enemy_kind=Kind.BOT_ARTILLERY)
    tiles = grid(20, 20)
    tiles[10, 7] = Tile.WALL  # squarely between the two, neither in a bush
    bank = FakeBank(tiles)

    state.ent_pos[0, 0] = torch.tensor([5.5, 10.5])
    state.ent_pos[0, 1] = torch.tensor([9.5, 10.5])

    _vis, tgt = build_targeting(state, bank, params, cfg)
    assert state.ent_target[0, 1].item() == 0
    assert not bool(tgt.los[0, 1])  # confirm the wall really is in the way physically

    fire, _, _ = combat_rules.combat(state, tgt, bank, params, cfg, gen)
    assert bool(fire[0, 1])  # fires anyway -- Step 18's documented design choice


def test_aim_point_is_exact_leaded_position_with_zero_noise():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, enemy_kind=Kind.BOT_ARTILLERY)
    bank = FakeBank(grid(20, 20))
    params.aim_noise_tiles[:, int(Kind.BOT_ARTILLERY)] = 0.0

    state.ent_pos[0, 0] = torch.tensor([5.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([12.0, 10.0])

    _vis, tgt = build_targeting(state, bank, params, cfg)
    _fire, _aim_dir, aim_point = combat_rules.combat(state, tgt, bank, params, cfg, gen)
    assert torch.allclose(aim_point[0, 1], state.ent_pos[0, 0], atol=1e-4)


def test_lead_uses_the_shell_s_fixed_flight_time_not_its_speed():
    """A timed shell's intercept is closed form -- the target is led by exactly
    lead_target_fraction * proj_flight_seconds of its own velocity, with no dependence on how
    far away it is (which is what geo.lead_target's fixed-point iteration was solving for)."""
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, enemy_kind=Kind.BOT_ARTILLERY)
    bank = FakeBank(grid(20, 20))
    params.aim_noise_tiles[:, int(Kind.BOT_ARTILLERY)] = 0.0

    kind = int(Kind.BOT_ARTILLERY)
    flight = params.proj_flight_seconds[0, kind].item()
    fraction = params.lead_target_fraction[0, kind].item()
    assert flight > 0  # otherwise this test silently checks the constant-speed path instead

    state.ent_pos[0, 1] = torch.tensor([12.0, 10.0])
    for hero_x in (5.0, 9.0):  # near and far: the lead must not change with distance
        state.ent_pos[0, 0] = torch.tensor([hero_x, 10.0])
        state.ent_vel[0, 0] = torch.tensor([0.0, 2.5])  # walking south

        _vis, tgt = build_targeting(state, bank, params, cfg)
        _fire, _aim_dir, aim_point = combat_rules.combat(state, tgt, bank, params, cfg, gen)

        expected = state.ent_pos[0, 0] + state.ent_vel[0, 0] * (fraction * flight)
        assert torch.allclose(aim_point[0, 1], expected, atol=1e-4)


def test_no_nan_in_combat_output_ever():
    cfg, params, gen = cfg_and_params(n_envs=8, n_enemies=3)
    state = fresh_state(cfg, params, n_envs=8, enemy_kind=Kind.BOT_ARTILLERY)
    tiles = grid(20, 20)
    tiles[9, 9] = Tile.WALL
    bank = FakeBank(tiles)

    for _ in range(300):
        state.ent_pos.uniform_(1, 19)
        state.ent_hp.uniform_(0, 1)
        state.ent_hp.mul_(state.ent_max_hp)
        state.ent_alive.copy_(torch.rand(state.ent_alive.shape) > 0.3)
        state.ent_ammo.uniform_(0, 5)
        state.ent_target.random_(-1, cfg.n_entities)

        _vis, tgt = build_targeting(state, bank, params, cfg)
        _fire, aim_dir, aim_point = combat_rules.combat(state, tgt, bank, params, cfg, gen)
        for t in (aim_dir, aim_point):
            assert torch.all(torch.isfinite(t))


# ---- full-loop acceptance ----------------------------------------------------------

def test_kiting_artillery_reaches_and_holds_range_and_lands_a_hit():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, enemy_kind=Kind.BOT_ARTILLERY, person=Person.KITE)
    bank = FakeBank(grid(20, 20))

    state.ent_pos[0, 0] = torch.tensor([3.0, 10.0])    # hero, stationary
    state.ent_pos[0, 1] = torch.tensor([17.0, 10.0])   # 14 tiles away

    attack_range = params.attack_range[0, int(Kind.BOT_ARTILLERY)].item()
    fraction = params.desired_range_fraction[0, int(Kind.BOT_ARTILLERY)].item()
    desired = fraction * attack_range
    # See tests/test_sniper.py's copy of this line: a precondition on the setup, not a copy of
    # the configured number.
    assert 0 < fraction <= 1.0 and desired < 14.0

    hit_tick = None
    for step in range(200):
        _intent, dmg_ent = _run_tick(state, bank, params, cfg, gen)
        if hit_tick is None and dmg_ent[0, 0].item() > 0:
            hit_tick = step
        if step == 149:
            dist = geo.dist(state.ent_pos[0, 0], state.ent_pos[0, 1]).item()
            assert (desired - 4.0) <= dist <= (desired + 4.0)

    assert hit_tick is not None and hit_tick < 200


def test_never_misses_stationary_target_at_half_range_with_zero_noise():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, enemy_kind=Kind.BOT_ARTILLERY)
    bank = FakeBank(grid(20, 20))
    params.aim_noise_tiles[:, int(Kind.BOT_ARTILLERY)] = 0.0

    attack_range = params.attack_range[0, int(Kind.BOT_ARTILLERY)].item()
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([10.0, 10.0 + 0.5 * attack_range])

    hit = False
    for _ in range(80):
        _intent, dmg_ent = _run_tick(state, bank, params, cfg, gen)
        if dmg_ent[0, 0].item() > 0:
            hit = True
            break
        state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
        state.ent_pos[0, 1] = torch.tensor([10.0, 10.0 + 0.5 * attack_range])

    assert hit
