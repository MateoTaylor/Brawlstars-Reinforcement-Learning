"""Hero action decoding and the kind-agnostic dash mechanic. See BRAWL_SIM_BUILD_PLAN.md
Step 11.

Note on dash_t ownership: Section 4's tick order lists dash_t among the timers "advanced" in
phase 2, but this step's own advance_dash docstring also decrements it in phase 8 ("dash_t -=
dt"). Doing both would double-decrement every continuing dash (it'd travel half the intended
clipped_distance before dash_t hits 0), which contradicts start_dash's
dash_speed = clipped_distance / dash_duration contract. tick_timers does NOT touch dash_t;
advance_dash is its sole owner, since that's where the paired "decrement, then clean up on
completion" logic lives.

Note on ent_out_of_combat_t direction: despite sitting alongside the other countdown timers in
both Section 4's phase-2 listing and this step's own MUTATES line, "time since combat" is
a stopwatch, not a countdown -- it pairs with SimParams.regen_delay (Step 14's apply_regen:
"has it been at least regen_delay seconds since I was last in a fight"), which only makes sense
against an incrementing value reset to 0 by combat. It counts UP here, not down.

**Two things reset it, and both are elsewhere:** taking damage (combat.apply_damage) and
ATTACKING (env.py's `_attack_phase`). It was named `ent_no_damage_t` while only the first
existed; regen that a player could out-heal while still shooting is not "out of combat" regen,
so the attack reset came with the field's new name.
"""
import torch

from . import geometry as geo
from . import stats
from . import terrain

_EPS = 1e-6
# Extra clearance a clipped dash keeps from a wall face, beyond the geometric back-off. See
# start_dash: landing with the hitbox edge exactly ON the face still reads as blocked.
_WALL_CLEARANCE = 1e-3


def action_mask(state, params, cfg) -> dict:
    """Hero-only (entity index 0). {"move": (N,17) bool, "attack": (N,3) bool}.

    The attack column is `[no-fire, attack, SUPER]` (Step D2 / bot_overhaul.md D1). Widening the
    existing dimension rather than adding a third keeps `action.shape == (N,2)`, so every wrapper,
    `act_buf`, and `env._held`'s fire-clearing keep working untouched -- the super is a distinct
    VALUE in the attack column, not a new column, which is why one decision still means at most one
    attack attempt for supers too.
    """
    hero_alive = state.ent_alive[:, 0]
    hero_ammo = state.ent_ammo[:, 0]
    hero_cd = state.ent_attack_cd[:, 0]
    hero_dash_t = state.ent_dash_t[:, 0]

    n = hero_alive.shape[0]
    move = torch.ones(n, cfg.n_move_bins + 1, dtype=torch.bool, device=hero_alive.device)

    ready = hero_alive & (hero_cd <= 0) & (hero_dash_t <= 0)
    fire_ok = ready & (hero_ammo >= 1.0)
    # A super costs CHARGE, not ammo -- an empty clip must not block it. It shares the cooldown and
    # the no-dashing rule, so it cannot be used to sidestep either.
    super_ok = ready & super_ready(state, params)[:, 0]
    no_fire_ok = torch.ones_like(fire_ok)
    attack = torch.stack([no_fire_ok, fire_ok, super_ok], dim=1)

    return {"move": move, "attack": attack}


def decode_action(action: torch.Tensor, state, params, cfg):
    """Hero-only. action: (N,2) i64. Returns (move_dir (N,2) f32, fire (N,) bool,
    super_fire (N,) bool).

    `action[:, 1]` is now three-valued: 0 = nothing, 1 = attack, 2 = super. Both outputs are
    ANDed with the mask, so an illegal request of either kind is a silent no-op rather than an
    error -- unchanged in spirit from before, just over three values instead of two."""
    move_bin = action[:, 0]
    is_idle = move_bin == 0
    dirs = geo.dir_from_bin(torch.clamp(move_bin - 1, min=0), cfg.n_move_bins)
    move_dir = torch.where(is_idle.unsqueeze(-1), torch.zeros_like(dirs), dirs)

    mask = action_mask(state, params, cfg)["attack"]
    fire = (action[:, 1] == 1) & mask[:, 1]
    super_fire = (action[:, 1] == 2) & mask[:, 2]

    return move_dir, fire, super_fire


def tick_timers(state, params, cfg) -> None:
    """MUTATES: ent_ammo, ent_attack_cd, ent_invuln_t, ent_reveal_t, ent_react_t,
    ent_out_of_combat_t (NOT ent_dash_t -- see module docstring). ammo += dt/reload_seconds
    clamped to max_ammo, EXCEPT while attack_cd is running; countdowns clamp at 0;
    ent_out_of_combat_t counts up instead (see module docstring).

    **Firing pauses the reload for `attack_cooldown` seconds (Step C1 / bot_overhaul.md D7).**
    `attack_cooldown` now does double duty: for its duration you can neither attack (already
    enforced by `_attack_phase`'s `can_attack`, `action_mask`'s `fire_ok`, and
    `policy.fire_gate`) nor accrue ammo. That makes sustained fire `attack_cooldown +
    reload_seconds` per shot rather than `reload_seconds`, which is the whole point: a 5-shot
    Buzz sweep costs a full second of not reloading, and a brawler that empties its clip is out of
    the fight for the sum of both.

    This replaced a much larger design (an `attack_anim_seconds` param plus an `ent_attack_anim_t`
    state field modelling "I am mid-attack"). `ent_attack_cd` already WAS that timer -- it exists,
    is already set by every attack source (`_attack_phase` for ranged/melee, `start_dash` for
    dashes), and is already decremented right here -- so the entire mechanic is the one gate below.

    Two ordering notes:
      - The gate is read BEFORE the decrement, so a cooldown of `k * dt` blocks exactly `k` ticks.
      - This runs in phase 2 and attacks resolve in phase 6, so on the tick an entity fires it has
        already banked one tick (0.05 s) of reload before its own shot stops it. Same shape as the
        one-tick leakage `combat.apply_regen` documents and accepts, and far cheaper than
        reordering the tick for it.
    """
    reload_seconds = stats.gather_kind(params.reload_seconds, state.ent_kind)
    max_ammo = stats.gather_kind(params.max_ammo, state.ent_kind)

    reloading = state.ent_attack_cd <= 0
    gain = torch.where(reloading, cfg.dt / reload_seconds, torch.zeros_like(reload_seconds))
    state.ent_ammo.copy_(torch.clamp(state.ent_ammo + gain, max=max_ammo))
    state.ent_attack_cd.copy_(torch.clamp(state.ent_attack_cd - cfg.dt, min=0))
    state.ent_invuln_t.copy_(torch.clamp(state.ent_invuln_t - cfg.dt, min=0))
    state.ent_reveal_t.copy_(torch.clamp(state.ent_reveal_t - cfg.dt, min=0))
    state.ent_react_t.copy_(torch.clamp(state.ent_react_t - cfg.dt, min=0))
    state.ent_out_of_combat_t.copy_(state.ent_out_of_combat_t + cfg.dt)  # stopwatch, not a countdown
    # Also a stopwatch. Reset ONLY by attacking (env._attack_phase), never by taking damage --
    # see the field's own note in core/state.py for why that distinction is load-bearing.
    state.ent_attack_idle_t.copy_(state.ent_attack_idle_t + cfg.dt)


def long_dash_ready(state, params) -> torch.Tensor:
    """(N,E) bool -- has this entity gone `long_dash_seconds` without attacking, so its next dash
    reaches `long_dash_multiplier` times as far? (Step D1 / CHARACTER_DETAILS: Mortis.)

    False for any kind that does not configure the ability (`long_dash_seconds` resolves to 0),
    which is every kind but the hero.
    """
    threshold = stats.gather_kind(params.long_dash_seconds, state.ent_kind)
    return (threshold > 0) & (state.ent_attack_idle_t >= threshold)


def long_dash_charge_frac(state, params) -> torch.Tensor:
    """(N,E) f32 in [0,1] -- progress toward the long dash. 1.0 means ready.

    The observation carries this CONTINUOUS value alongside the boolean, because a policy can plan
    against "0.8 charged, hold off attacking for one more second" in a way it cannot against a flag
    that stays 0 until it snaps to 1. Zero for kinds without the ability.
    """
    threshold = stats.gather_kind(params.long_dash_seconds, state.ent_kind)
    frac = state.ent_attack_idle_t / torch.clamp(threshold, min=_EPS)
    return torch.where(threshold > 0, torch.clamp(frac, max=1.0), torch.zeros_like(frac))


def long_dash_scale(state, params) -> torch.Tensor:
    """(N,E) f32 -- the multiplier to apply to `dash_distance` right now: `long_dash_multiplier`
    where the long dash is charged, else 1.0."""
    multiplier = stats.gather_kind(params.long_dash_multiplier, state.ent_kind)
    ready = long_dash_ready(state, params)
    return torch.where(ready, torch.clamp(multiplier, min=1.0), torch.ones_like(multiplier))


def has_super(state, params) -> torch.Tensor:
    """(N,E) bool -- does this entity's KIND have a super at all? (`super_charge_hits > 0`.)

    Every super predicate is gated on this rather than on "is this entity slot 0", so a bot super
    is a `configs/brawlers.yaml` block rather than a plumbing change (Step D2)."""
    return stats.gather_kind(params.super_charge_hits, state.ent_kind) > 0


def super_ready(state, params) -> torch.Tensor:
    """(N,E) bool -- charged enough to fire. False for any kind without a super."""
    needed = stats.gather_kind(params.super_charge_hits, state.ent_kind)
    return has_super(state, params) & (state.ent_super_charge >= needed.to(state.ent_super_charge.dtype))


def super_charge_frac(state, params) -> torch.Tensor:
    """(N,E) f32 in [0,1] -- progress toward the super. The observation carries this alongside the
    boolean for the same reason `long_dash_frac` does: a flag alone gives a policy no gradient to
    climb, while a fraction alone hides the moment the ability actually becomes available."""
    needed = stats.gather_kind(params.super_charge_hits, state.ent_kind).to(torch.float32)
    frac = state.ent_super_charge.to(torch.float32) / torch.clamp(needed, min=_EPS)
    return torch.where(needed > 0, torch.clamp(frac, max=1.0), torch.zeros_like(frac))


def add_super_charge(state, hits: torch.Tensor, params) -> None:
    """MUTATES: ent_super_charge. `hits` is (N,E) -- how many qualifying hits each entity landed
    this tick. Clamped at `super_charge_hits`, so charge never banks past full.

    Only hits on living PLAYERS count. Boxes are excluded by the caller (env._bookkeeping), which
    is the whole reason charge is accumulated from the attacker x victim damage matrix rather than
    from a simpler "did I deal damage" flag -- box damage flows through a different path and would
    otherwise let a hero farm his super off crates."""
    needed = stats.gather_kind(params.super_charge_hits, state.ent_kind).to(state.ent_super_charge.dtype)
    charged = state.ent_super_charge + hits.to(state.ent_super_charge.dtype)
    state.ent_super_charge.copy_(torch.where(has_super(state, params), torch.minimum(charged, needed), state.ent_super_charge))


def start_dash(state, fire: torch.Tensor, move_dir: torch.Tensor, bank, params, cfg) -> None:
    """fire: (N,E) bool, move_dir: (N,E,2) f32 -- kind-agnostic, gated on dash_distance[kind] >
    0, so melee lunges and Mortis's Super can reuse this later (Step 11).

    MUTATES: ent_ammo, ent_attack_cd, ent_dash_t, ent_dash_dir, ent_dash_speed, ent_dash_hits,
    ent_invuln_t, ent_facing, ent_shots_fired.
    """
    E = state.ent_kind.shape[1]

    dash_distance = stats.gather_kind(params.dash_distance, state.ent_kind)
    dash_duration = stats.gather_kind(params.dash_duration, state.ent_kind)
    attack_cooldown = stats.gather_kind(params.attack_cooldown, state.ent_kind)

    # LONG DASH (Step D1). After `long_dash_seconds` without attacking, the next dash covers
    # `long_dash_multiplier` times its normal distance. `dash_duration` is deliberately NOT
    # scaled, so a long dash is also genuinely FASTER (5.34 tiles in 0.30 s instead of 2.67) --
    # that is what the real ability feels like, and everything downstream is distance-agnostic:
    # the terrain clip below, its `safe_hit_t` back-off, and `dash_speed = clipped / duration`.
    #
    # Gated on `long_dash_seconds > 0` so a kind without the ability can never trigger it, rather
    # than relying on a huge threshold never being reached.
    dash_distance = dash_distance * long_dash_scale(state, params)

    move_is_zero = (move_dir[..., 0] == 0) & (move_dir[..., 1] == 0)
    can_dash = fire & state.ent_alive & (dash_distance > 0)
    if cfg.dash_on_idle == "block":
        can_dash = can_dash & ~move_is_zero

    facing_vec = geo.from_angle(state.ent_facing)
    dash_dir = torch.where(move_is_zero.unsqueeze(-1), facing_vec, move_dir)

    # Retroactive fix (found while building Step 29's end-to-end integration tests): march()
    # only samples every los_step_tiles, so hit_t is the distance to the first BLOCKED sample
    # -- the true wall boundary lies somewhere in the los_step_tiles gap before it, not AT it.
    # Clipping to hit_t directly could land (or, with float rounding, land fractionally past)
    # the entity right on the wall tile itself, which check_invariants correctly flags as
    # out-of-bounds near a map edge. Backing off by one full los_step_tiles guarantees the
    # clipped landing point is on the known-safe side of that gap.
    #
    # march() samples the bare CENTER point against blocks_unit -- unlike resolve_move's
    # circle_blocked, it has no notion of the dasher's own body. Backing off only by
    # los_step_tiles leaves the landing point's clearance from the true wall face anywhere in
    # [0, los_step_tiles), so a center that clears the tile boundary by less than unit_radius
    # still lands with its hitbox embedded in the wall. That's a one-way trap: apply_movement's
    # resolve_move (post-dash, once dash_t <= 0) rejects a candidate step outright unless the
    # ENTIRE circle at the destination is clear, so from inside the wall every escape attempt
    # keeps failing the same way every tick -- permanently stuck. Backing off by radius as well
    # guarantees at least unit_radius of clearance from the wall face, so the dash can never
    # land the body inside it in the first place.
    radius = params.unit_radius.unsqueeze(-1)  # (N,1), broadcasts against (N,E)
    # **March PAST the landing point, by the same margin the back-off below removes.**
    #
    # Marching only to `dash_distance` finds walls the CENTRE's path crosses, but the thing that
    # traps an entity is its BODY overlapping one. `terrain.circle_blocked` probes all 8 compass
    # points at a candidate destination, so a circle overlapping a wall is blocked in EVERY
    # direction -- including away from it -- and `movement.resolve_move` then rejects every escape
    # step identically, forever. A dash landing within `unit_radius` of a wall face is therefore
    # just as stuck as one landing on the wall itself.
    #
    # Because march samples every `los_step_tiles`, a wall in that band could sit between the last
    # sample and the endpoint and never be seen at all -- in which case `hit` was False, no
    # back-off was applied, and the entity landed grazing the wall. Extending the probe by
    # `los_step_tiles + radius` makes exactly that band visible, so the existing back-off can do
    # its job. Found by test_a_charged_dash_into_a_wall_still_lands_the_body_clear_of_it: Step D1
    # doubled the dash distance, which made the failure easy to reproduce, but the trap predates it
    # and applies to an ordinary 2.67-tile dash too.
    probe_distance = dash_distance + cfg.los_step_tiles + radius
    hit, _, hit_t = terrain.march(bank.blocks_unit, state.map_id, state.ent_pos, dash_dir, probe_distance, cfg)
    # `_WALL_CLEARANCE` on top of the two geometric terms: backing off by exactly
    # `los_step_tiles + radius` puts the hitbox edge EXACTLY on the wall face, and a tile lookup
    # of `floor(30.0)` is tile 30 -- the wall. Touching counts as blocked, so exact is not enough.
    # One millitile is far below anything observable and clears the boundary (and float32's ~2e-6
    # ulp at these magnitudes) outright.
    safe_hit_t = torch.clamp(hit_t - cfg.los_step_tiles - radius - _WALL_CLEARANCE, min=0.0)
    # `minimum` because the probe now reaches beyond the dash: a wall found past the landing point
    # must not EXTEND the dash, only ever shorten it.
    clipped_distance = torch.where(hit, torch.minimum(safe_hit_t, dash_distance), dash_distance)
    dash_speed = clipped_distance / torch.clamp(dash_duration, min=_EPS)

    state.ent_ammo.copy_(torch.where(can_dash, state.ent_ammo - 1.0, state.ent_ammo))
    state.ent_attack_cd.copy_(torch.where(can_dash, attack_cooldown, state.ent_attack_cd))
    state.ent_dash_t.copy_(torch.where(can_dash, dash_duration, state.ent_dash_t))
    state.ent_dash_dir.copy_(torch.where(can_dash.unsqueeze(-1), dash_dir, state.ent_dash_dir))
    state.ent_dash_speed.copy_(torch.where(can_dash, dash_speed, state.ent_dash_speed))
    state.ent_invuln_t.copy_(torch.where(can_dash, dash_duration, state.ent_invuln_t))
    state.ent_facing.copy_(torch.where(can_dash, geo.angle_of(dash_dir), state.ent_facing))
    state.ent_shots_fired.copy_(torch.where(can_dash, state.ent_shots_fired + 1, state.ent_shots_fired))

    row_mask = can_dash.unsqueeze(-1).expand(-1, -1, E)
    state.ent_dash_hits.copy_(torch.where(row_mask, torch.zeros_like(state.ent_dash_hits), state.ent_dash_hits))


def advance_dash(state, params, cfg):
    """MUTATES: ent_pos, ent_vel, ent_dash_t, ent_dash_hits. Returns (dmg_ent (N,E), dmg_by
    (N,E,E), dmg_box (N,B)) -- reported here, applied by combat.apply_damage (Step 14), same
    pattern as projectile damage."""
    E = state.ent_kind.shape[1]
    is_dashing = state.ent_dash_t > 0

    dash_radius = stats.gather_kind(params.dash_radius, state.ent_kind)

    # The last tick of a dash advances only the time that is actually LEFT, not a full dt.
    #
    # Without this every dash overshot by one whole tick. `dash_speed` is
    # `clipped_distance / dash_duration`, so travelling for a 7th tick when the duration only
    # covers 6 stretches the dash to 7/6 = 1.167x its configured distance -- and it happened on
    # every dash, because `dash_duration: 0.30` is not exactly representable: after six
    # subtractions of dt it leaves 1.19e-8 seconds, which is still `> 0`, so `is_dashing` was True
    # for one more tick.
    #
    # That is a correctness bug rather than a cosmetic one, because `start_dash` clips the dash
    # against terrain and backs the landing point off by `los_step_tiles + unit_radius`
    # specifically to GUARANTEE the body never lands inside a wall. Overshooting the clipped
    # distance by 16.7% silently spends that safety margin, and past it the entity is stuck --
    # `movement.resolve_move` rejects every escape step whose whole circle is not clear.
    # Found by tests/test_hero.py::test_a_charged_dash_into_a_wall_still_lands_the_body_clear_of_it
    # (Step D1 doubled the dash distance, which doubled the absolute overshoot and made it
    # reproducible against a wall).
    step_seconds = torch.clamp(state.ent_dash_t, max=cfg.dt)
    delta = state.ent_dash_dir * (state.ent_dash_speed * step_seconds).unsqueeze(-1)
    old_pos = state.ent_pos
    new_pos = old_pos + delta

    dmg = stats.effective_damage(state.ent_kind, state.ent_cubes, params)  # (N,E)

    p0 = old_pos.unsqueeze(2)          # (N,E,1,2) attacker axis
    p1 = new_pos.unsqueeze(2)          # (N,E,1,2)
    r = dash_radius.unsqueeze(2)       # (N,E,1)

    # --- units ---
    victim_pos = state.ent_pos.unsqueeze(1)  # (N,1,E,2)
    inside = geo.capsule_contains(p0, p1, r, victim_pos)  # (N,E,E)
    eye = torch.eye(E, dtype=torch.bool, device=state.ent_pos.device).unsqueeze(0)
    new_hit = (
        inside & ~eye
        & state.ent_alive.unsqueeze(1) & state.ent_alive.unsqueeze(2)
        & is_dashing.unsqueeze(2) & ~state.ent_dash_hits
    )
    dmg_by = torch.where(new_hit, dmg.unsqueeze(2), torch.zeros_like(new_hit, dtype=dmg.dtype))
    dmg_ent = dmg_by.sum(dim=1)

    # --- boxes ---
    box_pos = state.box_pos.unsqueeze(1)  # (N,1,B,2)
    box_inside = geo.capsule_contains(p0, p1, r, box_pos)  # (N,E,B)
    box_hit = box_inside & state.box_alive.unsqueeze(1) & is_dashing.unsqueeze(2)
    box_dmg = torch.where(box_hit, dmg.unsqueeze(2), torch.zeros_like(box_hit, dtype=dmg.dtype))
    dmg_box = box_dmg.sum(dim=1)

    # --- movement (dashers pass through units; no terrain re-resolve, no separation) ---
    state.ent_pos.copy_(torch.where(is_dashing.unsqueeze(-1), new_pos, old_pos))
    state.ent_vel.copy_(torch.where(is_dashing.unsqueeze(-1), delta / cfg.dt, state.ent_vel))

    # --- dash_t countdown + completion cleanup ---
    new_dash_t = torch.clamp(state.ent_dash_t - cfg.dt, min=0)
    state.ent_dash_t.copy_(torch.where(is_dashing, new_dash_t, state.ent_dash_t))
    finished = is_dashing & (new_dash_t <= 0)

    state.ent_dash_dir.copy_(torch.where(finished.unsqueeze(-1), torch.zeros_like(state.ent_dash_dir), state.ent_dash_dir))
    state.ent_dash_speed.copy_(torch.where(finished, torch.zeros_like(state.ent_dash_speed), state.ent_dash_speed))

    updated_hits = state.ent_dash_hits | new_hit
    finished_row = finished.unsqueeze(-1).expand(-1, -1, E)
    state.ent_dash_hits.copy_(torch.where(finished_row, torch.zeros_like(updated_hits), updated_hits))

    return dmg_ent, dmg_by, dmg_box
