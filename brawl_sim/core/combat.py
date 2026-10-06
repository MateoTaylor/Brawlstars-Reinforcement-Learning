"""Damage application, melee hit detection, death resolution, cube drops/pickups, and regen.

Death cause has no field of its own: apply_damage's `attacker` doubles as the cause signal -- an
entity index for combat sources, the sentinel -1 for zone damage -- and resolve_deaths derives
ent_death_cause from whether ent_last_hit_by is the sentinel at the moment of death.

drop_cubes_on_death reuses projectiles.alloc_slots/_set_scalar/_set_vec2 (pku_alive standing in
for prj_alive): they are algorithm-generic despite the module they live in.
"""
import torch

from ..constants import DeathCause
from . import geometry as geo
from . import projectiles as proj
from . import stats
from . import terrain

_NO_ATTACKER = -1


def apply_damage(state, dmg: torch.Tensor, cause: int, attacker: torch.Tensor, params, cfg) -> None:
    """dmg/attacker: (N,E). cause: a DeathCause value (or plain int), the SAME for the whole
    call -- callers invoke this once per damage source (dash, melee, projectiles, zone), never
    mixing causes within one call. MUTATES: ent_hp, ent_damage_taken, ent_last_hit_by,
    ent_out_of_combat_t. Where invuln_t > 0, COMBAT damage is zeroed; ZONE damage is not, unless
    cfg.iframes_block_zone (default false). Nothing seeds ent_invuln_t (the dash has no
    i-frames), so this gate is dormant."""
    blocked = state.ent_invuln_t > 0
    if int(cause) == int(DeathCause.ZONE) and not cfg.iframes_block_zone:
        effective_dmg = dmg
    else:
        effective_dmg = torch.where(blocked, torch.zeros_like(dmg), dmg)

    # An entity already at 0 HP is dead in everything but the bookkeeping (`resolve_deaths` runs
    # once, after every damage source of the tick), so a later source must not overwrite the
    # finisher's `ent_last_hit_by`: the gas (phase 10, after the dash and projectile phases)
    # cannot steal kill credit (user decision, 2026-09-25).
    took_damage = (effective_dmg > 0) & state.ent_alive & (state.ent_hp > 0)

    state.ent_hp.copy_(torch.clamp(state.ent_hp - effective_dmg, min=0))
    state.ent_damage_taken.copy_(state.ent_damage_taken + torch.where(took_damage, effective_dmg, torch.zeros_like(effective_dmg)))
    state.ent_last_hit_by.copy_(torch.where(took_damage, attacker, state.ent_last_hit_by))
    state.ent_out_of_combat_t.copy_(torch.where(took_damage, torch.zeros_like(state.ent_out_of_combat_t), state.ent_out_of_combat_t))


def apply_heal(state, heal: torch.Tensor) -> torch.Tensor:
    """heal: (N,E) HP to restore. MUTATES: ent_hp, clamped to ent_max_hp. Returns the (N,E) HP
    ACTUALLY restored after the alive gate and the max-HP clamp -- the number reward shaping
    prices (`training/reward.py`'s `hp_healed`, via `info["hp_healed_tick"]`), so a topped-up
    hero is not paid for lifesteal that healed nothing.

    Combat healing (super and melee lifesteal), separate from `apply_regen`'s out-of-combat
    trickle, which is gated on a stopwatch and a config toggle.

    The `ent_alive` gate is not optional: a healed corpse would have `ent_hp > 0` while dead, a
    state `state.check_invariants` rejects and `resolve_deaths` never cleans up.
    """
    alive_heal = torch.where(state.ent_alive, heal, torch.zeros_like(heal))
    new_hp = torch.clamp(state.ent_hp + alive_heal, max=state.ent_max_hp)
    healed = new_hp - state.ent_hp
    state.ent_hp.copy_(new_hp)
    return healed


def melee_lifesteal(state, dmg_by: torch.Tensor, params) -> torch.Tensor:
    """(N,E) HP each attacker should be healed for, off the melee damage it landed on PLAYERS this
    tick. `dmg_by` is `melee_hitscan`'s (N,E,E) attacker x victim matrix.

    Computes; does NOT apply: the caller commits it through `apply_heal`, which owns the alive
    gate and the max-HP clamp. Boxes are excluded structurally (`melee_hitscan` returns box damage
    separately): healing off a crate would be a free full heal on every map.

    I-frames are re-applied because `apply_damage` masks only its own summed copy, never `dmg_by`;
    without this, lifesteal would pay for damage an invulnerable victim never took. Nothing seeds
    `ent_invuln_t` (the dash has no i-frames), so the mask is dormant. Reading it after
    `apply_damage` (as `env._attack_phase` does) is safe: that function writes neither
    `ent_invuln_t` nor `ent_alive`.

    OVERKILL COUNTS: a 1080 hit on a 100-HP victim heals off the full 1080, matching what
    `ent_damage_taken`/`ent_damage_dealt` record; clamping would weaken lifesteal the closer Edgar
    is to a kill.
    """
    fraction = stats.gather_kind(params.melee_lifesteal_fraction, state.ent_kind)  # (N,E) attacker
    blocked = (state.ent_invuln_t > 0).unsqueeze(1)  # (N,1,E) over the VICTIM axis
    landed = torch.where(blocked, torch.zeros_like(dmg_by), dmg_by)
    return landed.sum(dim=2) * fraction


def dominant_attacker(dmg_by: torch.Tensor) -> torch.Tensor:
    """dmg_by: (N,E,E) attacker x victim. Returns (N,E) i64: the attacker dealing the MOST
    damage to each victim this tick, or `_NO_ATTACKER` (-1) where nobody dealt any.
    `apply_damage` takes one attacker per victim, but a wide cone, converging projectiles or two
    dashers can all hit one victim in the same tick; "whoever hit hardest" is the least arbitrary
    rule for `ent_last_hit_by`/kill credit. Used for melee, dash and projectile damage alike."""
    max_dmg, attacker_idx = dmg_by.max(dim=1)
    return torch.where(max_dmg > 0, attacker_idx, torch.full_like(attacker_idx, _NO_ATTACKER))


def melee_hitscan(state, fire_mask: torch.Tensor, bank, params, cfg, cone_dir: torch.Tensor | None = None):
    """Returns (dmg_ent (N,E), dmg_by (N,E,E), dmg_box (N,B)). All live entities/boxes within
    attack_range, inside attack_arc_rad of the cone centre, with a clear physical path
    (terrain.line_of_sight via blocks_proj -- walls only). No single-target restriction: unlike
    a projectile, a wide arc can hit several entities (and boxes) in the same swing.

    `cone_dir` (N,E) is the angle the cone is centred on; `None` means `state.ent_facing`, a
    single-swing melee. Swept melee passes each sub-swing's angle (core/melee_sweep.py).
    """
    E = state.ent_pos.shape[1]
    B = state.box_pos.shape[1]
    device = state.ent_pos.device

    attack_range = stats.gather_kind(params.attack_range, state.ent_kind)
    attack_arc = stats.gather_kind(params.attack_arc_rad, state.ent_kind)
    damage = stats.effective_damage(state.ent_kind, state.ent_cubes, params)

    if cone_dir is None:
        cone_dir = state.ent_facing

    origin = state.ent_pos.unsqueeze(2)     # (N,E,1,2)
    victim = state.ent_pos.unsqueeze(1)     # (N,1,E,2)
    facing = cone_dir.unsqueeze(2)          # (N,E,1)
    half_angle = (attack_arc / 2.0).unsqueeze(2)
    radius = attack_range.unsqueeze(2)

    in_cone = geo.point_in_cone(victim, origin, facing, radius, half_angle)  # (N,E,E)

    # `params.cone_ray_tiles` caps this dense (N,E,E) LOS march at the longest a CONE can reach
    # instead of the full cfg.ray_steps budget. That makes `los` WRONG for pairs further apart
    # than the budget (a far wall is never sampled), which is sound only because `los` is used
    # once, ANDed with `in_cone`, and `point_in_cone` already requires the victim within
    # `attack_range` <= cone_ray_tiles, so `can_hit` is bit-identical.
    # tests/test_combat.py::test_melee_los_budget_matches_full_budget_ray pins that equality.
    origin_b = state.ent_pos.unsqueeze(2).expand(-1, -1, E, -1)
    victim_b = state.ent_pos.unsqueeze(1).expand(-1, E, -1, -1)
    los = terrain.line_of_sight(bank, state.map_id, origin_b, victim_b, cfg,
                                max_tiles=params.cone_ray_tiles)

    not_self = ~torch.eye(E, dtype=torch.bool, device=device).unsqueeze(0)
    victim_alive = state.ent_alive.unsqueeze(1)
    attacker_ok = fire_mask.unsqueeze(2) & state.ent_alive.unsqueeze(2)  # (N,E,1)

    can_hit = in_cone & los & not_self & victim_alive & attacker_ok
    dmg_by = torch.where(can_hit, damage.unsqueeze(2), torch.zeros_like(in_cone, dtype=damage.dtype))
    dmg_ent = dmg_by.sum(dim=1)

    # --- boxes: same cone + physical-LOS test, against box_pos/box_alive instead ---
    box_victim = state.box_pos.unsqueeze(1)  # (N,1,B,2)
    in_cone_box = geo.point_in_cone(box_victim, origin, facing, radius, half_angle)  # (N,E,B)

    origin_box_b = state.ent_pos.unsqueeze(2).expand(-1, -1, B, -1)
    box_pos_b = state.box_pos.unsqueeze(1).expand(-1, E, -1, -1)
    los_box = terrain.line_of_sight(bank, state.map_id, origin_box_b, box_pos_b, cfg,
                                    max_tiles=params.cone_ray_tiles)  # same argument as above

    can_hit_box = in_cone_box & los_box & state.box_alive.unsqueeze(1) & attacker_ok
    dmg_box_by = torch.where(can_hit_box, damage.unsqueeze(2), torch.zeros_like(in_cone_box, dtype=damage.dtype))
    dmg_box = dmg_box_by.sum(dim=1)

    return dmg_ent, dmg_by, dmg_box


def resolve_deaths(state, cfg):
    """MUTATES: ent_alive, ent_death_step, ent_death_cause, ent_kills (via ent_last_hit_by),
    n_alive. Returns newly_dead (N,E) bool."""
    newly_dead = state.ent_alive & (state.ent_hp <= 0)

    state.ent_alive.copy_(state.ent_alive & ~newly_dead)

    step_count_b = state.step_count.unsqueeze(-1).to(state.ent_death_step.dtype)
    state.ent_death_step.copy_(torch.where(newly_dead, step_count_b, state.ent_death_step))

    is_zone_death = state.ent_last_hit_by < 0
    cause_val = torch.where(
        is_zone_death,
        torch.full_like(state.ent_death_cause, int(DeathCause.ZONE)),
        torch.full_like(state.ent_death_cause, int(DeathCause.COMBAT)),
    )
    state.ent_death_cause.copy_(torch.where(newly_dead, cause_val, state.ent_death_cause))

    credit = newly_dead & (state.ent_last_hit_by >= 0)
    killer_idx = torch.clamp(state.ent_last_hit_by, min=0)
    state.ent_kills.scatter_add_(1, killer_idx, credit.to(state.ent_kills.dtype))

    state.n_alive.copy_(state.ent_alive.sum(dim=1).to(state.n_alive.dtype))

    return newly_dead


def drop_cubes_on_death(state, newly_dead: torch.Tensor, params, cfg) -> None:
    """MUTATES: pku_*. One pickup per corpse with cubes_on_kill_base + (victim cubes if
    drop_victim_cubes). Zone deaths drop cubes too -- newly_dead doesn't distinguish cause,
    so this applies uniformly regardless of what killed them."""
    demand = newly_dead.to(torch.int64)  # (N,E), 0 or 1 per entity
    idx3, ok3 = proj.alloc_slots(state.pku_alive, demand, max_per_entity=1)
    idx, ok = idx3.squeeze(-1), ok3.squeeze(-1)  # (N,E)

    bonus = state.ent_cubes if cfg.drop_victim_cubes else torch.zeros_like(state.ent_cubes)
    cubes_new = params.cubes_on_kill_base.unsqueeze(-1) + bonus
    age_new = torch.zeros_like(state.ent_pos[..., 0])
    alive_new = torch.ones_like(ok)

    proj._set_vec2(state.pku_pos, idx, ok, state.ent_pos)
    proj._set_scalar(state.pku_cubes, idx, ok, cubes_new)
    proj._set_scalar(state.pku_age, idx, ok, age_new)
    proj._set_scalar(state.pku_alive, idx, ok, alive_new)


def collect_pickups(state, params, cfg):
    """MUTATES: ent_cubes, ent_hp, ent_max_hp, pku_alive. Returns gained (N,E) i64. Builds the
    (N,E,U) eligibility matrix, argmax over E (lowest index wins ties -- argmax on a bool-as-
    int tensor returns the first True), masks so only the winner claims. Capped entities
    (checked against their cube count BEFORE this tick's gains) do not claim and leave the
    pickup."""
    N, E = state.ent_pos.shape[:2]
    U = state.pku_pos.shape[1]

    dist = geo.dist(state.ent_pos.unsqueeze(2), state.pku_pos.unsqueeze(1))  # (N,E,U)
    within_radius = dist <= params.pickup_radius.view(-1, 1, 1)
    not_capped = state.ent_cubes < params.max_cubes.view(-1, 1)  # (N,E)

    eligible = (
        within_radius & state.ent_alive.unsqueeze(2)
        & state.pku_alive.unsqueeze(1) & not_capped.unsqueeze(2)
    )  # (N,E,U)

    eligible_pu = eligible.permute(0, 2, 1)  # (N,U,E)
    winner_e = torch.argmax(eligible_pu.to(torch.int64), dim=-1)  # (N,U)
    has_any = eligible_pu.any(dim=-1)  # (N,U)

    winner_onehot = torch.zeros(N, U, E, dtype=torch.bool, device=state.ent_pos.device)
    winner_onehot.scatter_(-1, winner_e.unsqueeze(-1), has_any.unsqueeze(-1))
    claim = winner_onehot.permute(0, 2, 1) & eligible  # (N,E,U)

    pku_cubes_b = state.pku_cubes.unsqueeze(1).expand(-1, E, -1)
    gained_matrix = torch.where(claim, pku_cubes_b, torch.zeros_like(pku_cubes_b))
    gained = gained_matrix.sum(dim=2)  # (N,E)

    old_cubes = state.ent_cubes
    new_cubes = old_cubes + gained
    new_hp = stats.apply_cube_gain(state.ent_hp, state.ent_kind, old_cubes, new_cubes, params)
    new_max_hp = stats.effective_max_hp(state.ent_kind, new_cubes, params)

    state.ent_hp.copy_(new_hp)
    state.ent_max_hp.copy_(new_max_hp)
    state.ent_cubes.copy_(new_cubes)

    pickup_claimed = claim.any(dim=1)  # (N,U)
    state.pku_alive.copy_(state.pku_alive & ~pickup_claimed)

    return gained


def apply_regen(state, params, cfg) -> torch.Tensor:
    """MUTATES: ent_hp. No-op when disabled. Returns the (N,E) HP actually restored this tick
    (zeros when disabled), on `apply_heal`'s "what the clamp let through" contract -- both feed
    `info["hp_healed_tick"]`.

    Heals `regen_fraction_per_second` of the entity's OWN max HP per second once it has been out
    of combat (no damage taken AND no offensive action -- see ent_out_of_combat_t) for
    `regen_delay` seconds. A fraction, as in the game: a flat rate would top small brawlers up
    faster and make every power cube (which raises max_hp) lengthen the top-up.

    Runs in tick phase 2, BEFORE this tick's attack phase, so an entity that fires this tick banks
    one tick of regen before its attack zeroes the stopwatch. Deliberate: a fatal hit later in the
    tick can then never be undone by a regen tick that has not seen it (env._tick_timers).
    """
    if not cfg.regen_enabled:
        return torch.zeros_like(state.ent_hp)
    can_regen = state.ent_alive & (state.ent_out_of_combat_t >= params.regen_delay.unsqueeze(-1))
    per_tick = params.regen_fraction_per_second.unsqueeze(-1) * state.ent_max_hp * cfg.dt
    regen_amount = torch.where(can_regen, per_tick, torch.zeros_like(state.ent_hp))
    new_hp = torch.clamp(state.ent_hp + regen_amount, max=state.ent_max_hp)
    healed = new_hp - state.ent_hp
    state.ent_hp.copy_(new_hp)
    return healed
