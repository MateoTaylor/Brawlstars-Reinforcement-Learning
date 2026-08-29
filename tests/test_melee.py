"""Melee archetype tests. FIRE and AIM only after Step 41's combat/movement split.

The two patrol tests (`test_patrols_to_zone_center_when_no_target_visible` /
`test_patrols_to_map_center_when_zone_rect_degenerate`) are deliberately GONE, not ported:
`melee._patrol_direction` no longer exists, and the behavior it tested is precisely the bug Step
41 set out to remove -- every idle melee bot steering at one shared point. Idle movement is now
per-entity randomized exploration, asserted in tests/test_personality.py
(`test_idle_bots_do_not_converge_on_a_shared_point`). "Closes and never retreats" survives as the
RUSH personality and is tested there too.
"""
import torch

from tests.bot_fixtures import FakeBank, build_targeting, cfg_and_params, fresh_state, grid
from brawl_sim.bots import combat_rules, perception, policy
from brawl_sim.constants import DeathCause, Kind, Person
from brawl_sim.core import combat, geometry as geo, hero, movement


def _run_tick(state, bank, params, cfg, gen):
    vis = perception.visibility(state, bank, params, cfg)
    intent = policy.all_bot_intents(state, vis, bank, params, cfg, gen)

    move_dir = intent.move_dir.clone()
    move_dir[:, 0] = 0.0
    fire_mask = intent.fire.clone()
    fire_mask[:, 0] = False

    # Melee damage is hitscan in phase 6, resolved BEFORE movement (Section 4), so this helper
    # keeps that order rather than the projectile archetypes' spawn-then-integrate shape.
    dmg_ent, dmg_by, _dmg_box = combat.melee_hitscan(state, fire_mask, bank, params, cfg)
    combat.apply_damage(state, dmg_ent, int(DeathCause.COMBAT), combat.dominant_attacker(dmg_by),
                        params, cfg)
    movement.apply_movement(state, move_dir, bank, params, cfg)
    hero.tick_timers(state, params, cfg)
    state.step_count += 1  # see tests/test_sniper.py::_run_tick
    return intent, dmg_ent


def test_strafe_sign_alternates_by_entity_slot():
    # policy.strafe_sign is a pure per-slot pattern -- (-1)**entity_index -- test it directly
    # rather than through a policy: with two entities approaching from OPPOSITE sides of the same
    # target, perp() itself already flips sign between them, so an equal-and-opposite alternating
    # strafe sign on top of that does NOT produce a globally-opposite world-space lateral offset --
    # that combination is a genuine geometric wash for diametrically-opposite approachers, not a
    # bug (verified against steering.strafe directly, which test_steering.py also covers).
    sign = policy.strafe_sign(4, torch.device("cpu"))
    assert torch.equal(sign, torch.tensor([1.0, -1.0, 1.0, -1.0]))


# ---- fire gating ---------------------------------------------------------------------

def test_fire_requires_target_inside_attack_arc_of_current_facing():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, enemy_kind=Kind.BOT_MELEE)
    bank = FakeBank(grid(20, 20))

    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([11.0, 10.0])  # hero is 1 tile west of the bot
    state.ent_facing[0, 1] = 0.0  # facing east -- away from the hero, outside the arc

    # Asserted, not asserted-in-a-comment: this line read "within attack_range (2.0)" while the
    # shipped value was 3.0, so it had been lying for some time. This test is about the ARC, so
    # the hero being inside the bot's reach is a precondition -- if a restat ever breaks it, the
    # test must fail loudly rather than start passing for the wrong reason.
    melee_range = params.attack_range[0, int(Kind.BOT_MELEE)].item()
    assert 1.0 < melee_range, f"test precondition: BOT_MELEE attack_range {melee_range} <= 1.0"

    _vis, tgt = build_targeting(state, bank, params, cfg)
    fire, _, _ = combat_rules.combat(state, tgt, bank, params, cfg, gen)
    assert not bool(fire[0, 1])

    state.ent_facing[0, 1] = 3.14159  # now facing west, straight at the hero
    fire, _, _ = combat_rules.combat(state, tgt, bank, params, cfg, gen)
    assert bool(fire[0, 1])


def test_no_nan_in_combat_output_ever():
    cfg, params, gen = cfg_and_params(n_envs=8, n_enemies=3)
    state = fresh_state(cfg, params, n_envs=8, enemy_kind=Kind.BOT_MELEE)
    bank = FakeBank(grid(20, 20))

    for _ in range(300):
        state.ent_pos.uniform_(1, 19)
        state.ent_hp.uniform_(0, 1)
        state.ent_hp.mul_(state.ent_max_hp)
        state.ent_alive.copy_(torch.rand(state.ent_alive.shape) > 0.3)
        state.ent_ammo.uniform_(0, 5)
        state.ent_facing.uniform_(-3.14, 3.14)
        state.ent_target.random_(-1, cfg.n_entities)
        state.zone_lo.uniform_(0, 5)
        state.zone_hi.uniform_(5, 20)

        _vis, tgt = build_targeting(state, bank, params, cfg)
        _fire, aim_dir, aim_point = combat_rules.combat(state, tgt, bank, params, cfg, gen)
        for t in (aim_dir, aim_point):
            assert torch.all(torch.isfinite(t))


# ---- full-loop acceptance ----------------------------------------------------------

def test_rushing_melee_closes_to_range_and_lands_a_hit():
    cfg, params, gen = cfg_and_params(n_enemies=1)
    state = fresh_state(cfg, params, enemy_kind=Kind.BOT_MELEE, person=Person.RUSH)
    bank = FakeBank(grid(20, 20))

    state.ent_pos[0, 0] = torch.tensor([3.0, 10.0])    # hero, stationary
    state.ent_pos[0, 1] = torch.tensor([17.0, 10.0])   # 14 tiles away

    attack_range = params.attack_range[0, int(Kind.BOT_MELEE)].item()

    hit_tick = None
    for step in range(200):
        _intent, dmg_ent = _run_tick(state, bank, params, cfg, gen)
        if hit_tick is None and dmg_ent[0, 0].item() > 0:
            hit_tick = step
        if step == 149:
            dist = geo.dist(state.ent_pos[0, 0], state.ent_pos[0, 1]).item()
            # closed to (and holds) melee range: RUSH never backs off. The margin covers the
            # strafe term's orbit plus soft separation's push-out at contact.
            assert dist <= attack_range + 1.5

    assert hit_tick is not None and hit_tick < 200
