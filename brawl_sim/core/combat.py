"""Damage application, melee hit detection, death resolution, cube drops/pickups, and regen.
See BRAWL_SIM_BUILD_PLAN.md Step 14.

Death-cause bookkeeping note: resolve_deaths needs to know whether a kill was COMBAT or ZONE
(SimState has no separate "pending cause" field for this). ent_last_hit_by already exists for
kill-credit, so apply_damage's `attacker` argument doubles as the cause signal: callers pass
a valid entity index for combat sources and the sentinel -1 for zone damage (which has no
attacker). resolve_deaths then derives ent_death_cause purely from whether ent_last_hit_by is
the sentinel at the moment of death -- no extra field needed.

Slot allocation note: drop_cubes_on_death needs the exact same collision-free "find a free
slot" scheme Step 13 built for projectiles (pku_alive stands in for prj_alive). Rather than
duplicate alloc_slots/_set_scalar/_set_vec2, this reuses them directly from .projectiles --
they're algorithm-generic, not projectile-specific, despite the module they live in.

Retroactive fix (found while building Step 22, core/boxes.py): Step 22's own text states boxes
"take damage from projectiles, AoE, melee cones, and the dash" as an already-true property.
Dash (Step 11's advance_dash) and projectiles (Step 13's step_projectiles) already computed
box damage; melee_hitscan did not -- there was nothing to hit yet when Step 14 was written, so
the gap was invisible until boxes actually started spawning. melee_hitscan now also returns
dmg_box (N,B), computed with the exact same in-cone + physical-LOS pattern already used for
units, checked against box_pos/box_alive instead. This changes melee_hitscan's return
signature from (dmg_ent, dmg_by) to (dmg_ent, dmg_by, dmg_box) -- every call site (tests, and
the melee archetype's docstring reference, now bots/combat_rules.py) has been updated to match.
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
    cfg.iframes_block_zone (default false)."""
    blocked = state.ent_invuln_t > 0
    if int(cause) == int(DeathCause.ZONE) and not cfg.iframes_block_zone:
        effective_dmg = dmg
    else:
        effective_dmg = torch.where(blocked, torch.zeros_like(dmg), dmg)

    took_damage = (effective_dmg > 0) & state.ent_alive

    state.ent_hp.copy_(torch.clamp(state.ent_hp - effective_dmg, min=0))
    state.ent_damage_taken.copy_(state.ent_damage_taken + torch.where(took_damage, effective_dmg, torch.zeros_like(effective_dmg)))
    state.ent_last_hit_by.copy_(torch.where(took_damage, attacker, state.ent_last_hit_by))
    state.ent_out_of_combat_t.copy_(torch.where(took_damage, torch.zeros_like(state.ent_out_of_combat_t), state.ent_out_of_combat_t))


def apply_heal(state, heal: torch.Tensor) -> torch.Tensor:
    """heal: (N,E) HP to restore. MUTATES: ent_hp, clamped to ent_max_hp. Returns the (N,E) HP
    ACTUALLY restored -- requested heal minus whatever the alive gate and the max-HP clamp threw
    away, so a full-HP entity "healing" for 5000 returns 0. That is the number reward shaping
    prices (`training/reward.py`'s `hp_healed`, via `info["hp_healed_tick"]`); crediting the
    requested amount instead would pay a topped-up hero for lifesteal that healed nothing.

    Combat healing (Step D2's super lifesteal) as opposed to `apply_regen`'s out-of-combat trickle,
    and deliberately a separate function: regen is gated on a stopwatch and a config toggle, while
    this is an immediate consequence of landing a hit.

    **Gated on `ent_alive`, which is not optional.** Healing a corpse would leave it with
    `ent_hp > 0` while `ent_alive` is False -- a state `core.state.check_invariants` explicitly
    rejects ("a dead entity has nonzero hp"), and one that `resolve_deaths` would never clean up,
    since it only ever kills entities that are still alive.
    """
    alive_heal = torch.where(state.ent_alive, heal, torch.zeros_like(heal))
    new_hp = torch.clamp(state.ent_hp + alive_heal, max=state.ent_max_hp)
    healed = new_hp - state.ent_hp
    state.ent_hp.copy_(new_hp)
    return healed


def melee_lifesteal(state, dmg_by: torch.Tensor, params) -> torch.Tensor:
    """(N,E) HP each attacker should be healed for, off the melee damage it landed on PLAYERS this
    tick. `dmg_by` is `melee_hitscan`'s (N,E,E) attacker x victim matrix.

    Computes; does NOT apply. Same contract as `melee_hitscan` and `projectiles.step_projectiles`
    -- the caller commits it through `apply_heal`, which owns the alive gate and the max-HP clamp
    and reports what actually landed.

    **Boxes are excluded structurally, not by a flag**: this reads only the entity matrix, and
    `melee_hitscan` returns box damage as a separate `dmg_box`. Edgar healing off a loot box would
    hand him a free full heal on every map, since boxes do not fight back.

    **I-frames are re-applied here rather than inherited.** `dmg_by` is raw cone output; it is
    `apply_damage` that zeroes damage against an entity with `ent_invuln_t > 0`, and it does that
    to its own summed (N,E) copy, leaving `dmg_by` untouched. Lifesteal read straight off `dmg_by`
    would therefore pay Edgar for swinging into a dashing Mortis -- damage the victim never took.
    Mirrored explicitly (rather than by having `apply_damage` hand back a masked matrix) because
    the mask is one line and the alternative is widening a signature four other call sites use to
    carry a value only this one wants.

    Reading `ent_invuln_t` AFTER `apply_damage` has run is safe and is what `env._attack_phase`
    does: nothing in `apply_damage` writes `ent_invuln_t` or `ent_alive`, so the value is the same
    one it gated on.

    OVERKILL COUNTS. A 1080-damage hit on a victim with 100 HP left heals off the full 1080, which
    is what `ent_damage_taken` and `ent_damage_dealt` already record for the same blow -- the
    accounting stays consistent, and the alternative (clamping to the victim's remaining HP) would
    make lifesteal quietly worse the closer Edgar is to finishing a kill.
    """
    fraction = stats.gather_kind(params.melee_lifesteal_fraction, state.ent_kind)  # (N,E) attacker
    blocked = (state.ent_invuln_t > 0).unsqueeze(1)  # (N,1,E) over the VICTIM axis
    landed = torch.where(blocked, torch.zeros_like(dmg_by), dmg_by)
    return landed.sum(dim=2) * fraction


def dominant_attacker(dmg_by: torch.Tensor) -> torch.Tensor:
    """dmg_by: (N,E,E) attacker x victim. Returns (N,E) i64: the attacker dealing the MOST
    damage to each victim this tick, or `_NO_ATTACKER` (-1) where nobody dealt any. Filled in
    for Step 29 (`env.py`): `apply_damage`'s own `attacker` parameter is one index per victim,
    but melee's wide cone, converging projectiles from different owners, or two simultaneous
    dashers can all land damage on the same victim in the same tick -- some single-attacker
    resolution rule is unavoidable for `ent_last_hit_by`/kill credit, and "whoever hit hardest"
    is the least arbitrary of the reasonable options. Used identically for all three combat
    damage sources (melee, dash, projectiles) by `env.py`'s attack/dash/projectile phases."""
    max_dmg, attacker_idx = dmg_by.max(dim=1)
    return torch.where(max_dmg > 0, attacker_idx, torch.full_like(attacker_idx, _NO_ATTACKER))


def melee_hitscan(state, fire_mask: torch.Tensor, bank, params, cfg, cone_dir: torch.Tensor | None = None):
    """Returns (dmg_ent (N,E), dmg_by (N,E,E), dmg_box (N,B)). All live entities/boxes within
    attack_range, inside attack_arc_rad of the cone centre, with a clear physical path
    (terrain.line_of_sight via blocks_proj -- walls only, Notice 4). No single-target
    restriction: a wide arc can hit several entities (and boxes) in the same swing (unlike
    projectiles, this step's spec doesn't ask for an earliest-wins rule here).

    `cone_dir` (N,E) is the angle the cone is centred on. `None` means `state.ent_facing`, which is
    what every caller wanted before Step C2 and is still what a single-swing melee does. The
    argument exists for SWEPT melee (Buzz): his one attack is five cones fired at five different
    angles across `attack_cooldown`, so the centre is no longer simply "where he is looking" --
    see core/melee_sweep.py, which computes it.
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

    # `params.cone_ray_tiles` caps the LOS march at the longest a CONE can possibly reach (3.0
    # tiles today) instead of the full cfg.ray_steps budget (24 tiles / 48 samples), which is what
    # this dense (N,E,E) march used to pay every single tick. Measured 4.15 -> 1.95 ms/tick at
    # n_envs=1024 (bot_overhaul.md Step A1).
    #
    # This makes `los` WRONG for pairs further apart than the budget -- a wall 10 tiles away is
    # simply never sampled, so those pairs come back "clear". That is sound here and only here:
    # `los` is used in exactly one place, ANDed with `in_cone`, and `point_in_cone` already
    # requires the victim within `attack_range` <= cone_ray_tiles. Every pair whose LOS answer
    # changed is a pair `in_cone` was already False for, so `can_hit` is bit-identical.
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
    (zeros when disabled), on the same "what the clamp let through" contract as `apply_heal` --
    both feed `info["hp_healed_tick"]`, and an entity already at max HP must not be paid for a
    regen tick that added nothing.

    Heals `regen_fraction_per_second` of the entity's OWN max HP per second, once it has been out
    of combat (no damage taken AND no attack made -- see ent_out_of_combat_t) for `regen_delay`
    seconds. A fraction rather than the flat per-second amount this used to take: the real mechanic
    is proportional, so a flat rate would make a 2000-HP brawler heal in a third of the time a
    6000-HP one does, and would make every power cube collected (which raises max_hp) *lengthen*
    the time to top up rather than leave it unchanged.

    Runs in tick phase 2, BEFORE this tick's attack phase resolves. An entity that fires this tick
    therefore banks one tick of regen (0.05s worth, ~1% of max HP at the defaults) before its own
    attack zeroes the stopwatch. Deliberate: phase 2 is also where a fatal hit must not be
    un-fatal-ed by a regen tick that hasn't seen it yet (see env.py's `_tick_timers`), and that
    ordering matters far more than one tick of leakage.
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
