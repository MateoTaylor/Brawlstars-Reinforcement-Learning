"""Hero action decoding, per-entity timers, the ability predicates (long dash, super, gadget) and
the kind-agnostic dash.

dash_t is owned by advance_dash alone: tick_timers does not decrement it, since doing both would
halve every dash against start_dash's `dash_speed = clipped_distance / dash_duration`.

ent_out_of_combat_t is a stopwatch counting UP, not a countdown: regen waits for it to reach
`regen_delay` (combat.apply_regen). Taking damage (combat.apply_damage) and any offensive action
-- attack, super or gadget (env._attack_phase) -- reset it to 0.
"""
import torch

from . import geometry as geo
from . import stats
from . import terrain

_EPS = 1e-6
# The attack column's values (config.EnvConfig.action_nvec): one column, one value.
ATTACK_NONE, ATTACK_FIRE, ATTACK_SUPER, ATTACK_GADGET, ATTACK_AUTO = 0, 1, 2, 3, 4
# A crate's collision radius: projectiles.BOX_RADIUS restated so this module need not import
# projectiles. tests/test_hero.py pins this copy at 0.5; keep the two equal.
_BOX_RADIUS = 0.5
# How far a clipped dash stops short of the contact `terrain.body_travel` finds, so float32
# rounding across advance_dash's increments cannot carry the body into the wall. See start_dash.
_WALL_CLEARANCE = 1e-3


def action_mask(state, params, cfg) -> dict:
    """Hero-only (entity index 0). {"move": (N, n_move_bins + 1) bool, "attack": (N,4) bool}, the
    attack half (N,5) under `cfg.auto_aim`.

    The attack column is `[no-fire, attack, SUPER, GADGET]`, plus `[AUTO]` under
    `action.auto_aim` (user decision, 2026-09-26): an attack whose dash is aimed for the policy
    at the nearest enemy or crate in reach (`auto_aim_target`), legal exactly when the plain
    attack is. These are VALUES of one column, not new columns, so `action.shape` stays (N,2) for
    every wrapper, `act_buf` and `env._held`, and one decision is at most one attack attempt --
    which is why a gadget and a dash cannot share one decision.
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
    # The gadget is a separate button on its own timer: deliberately NOT ANDed with `ready`, so it
    # is legal mid-dash, during the attack cooldown and on an empty clip, as in the game.
    gadget_ok = gadget_ready(state, params)[:, 0]
    no_fire_ok = torch.ones_like(fire_ok)
    columns = [no_fire_ok, fire_ok, super_ok, gadget_ok]
    if cfg.auto_aim:
        columns.append(fire_ok)
    attack = torch.stack(columns, dim=1)

    return {"move": move, "attack": attack}


def decode_action(action: torch.Tensor, state, params, cfg):
    """Hero-only. action: (N,2) i64. Returns (move_dir (N,2) f32, fire (N,) bool,
    super_fire (N,) bool, gadget_fire (N,) bool, auto (N,) bool).

    `action[:, 1]` is four-valued: 0 = nothing, 1 = attack, 2 = super, 3 = gadget, and
    five-valued under `cfg.auto_aim`, where 4 = auto-aimed attack. Every output is ANDed with
    the mask, so an illegal request of any kind is a silent no-op rather than an error. The
    values are mutually exclusive by construction -- one column, one value.

    `fire` is True for BOTH attack values (1 and 4), so everything downstream that means "the hero
    dashed" (ammo, cooldown, reveal, the reward's in-reach term, the cadence audit) is one code
    path. `auto` is read only by `env._attack_phase`, to swap in `auto_aim_target`'s direction
    before `start_dash`. Without the flag `auto` is all-False and a 4 is an illegal no-op."""
    move_bin = action[:, 0]
    is_idle = move_bin == 0
    dirs = geo.dir_from_bin(torch.clamp(move_bin - 1, min=0), cfg.n_move_bins)
    move_dir = torch.where(is_idle.unsqueeze(-1), torch.zeros_like(dirs), dirs)

    mask = action_mask(state, params, cfg)["attack"]
    attack = action[:, 1]
    fire = (attack == ATTACK_FIRE) & mask[:, ATTACK_FIRE]
    super_fire = (attack == ATTACK_SUPER) & mask[:, ATTACK_SUPER]
    gadget_fire = (attack == ATTACK_GADGET) & mask[:, ATTACK_GADGET]
    if cfg.auto_aim:
        auto = (attack == ATTACK_AUTO) & mask[:, ATTACK_AUTO]
        fire = fire | auto
    else:
        auto = torch.zeros_like(fire)

    return move_dir, fire, super_fire, gadget_fire, auto


def tick_timers(state, params, cfg) -> None:
    """MUTATES: ent_ammo, ent_attack_cd, ent_gadget_cd, ent_invuln_t, ent_reveal_t, ent_react_t,
    ent_out_of_combat_t, ent_attack_idle_t (NOT ent_dash_t -- see module docstring). Countdowns
    clamp at 0; the two stopwatches count up.

    Ammo (+dt/reload_seconds, clamped to max_ammo) accrues only while attack_cd is 0, so firing
    pauses the reload for `attack_cooldown` and sustained fire costs `attack_cooldown +
    reload_seconds` per shot. The gate is read BEFORE the decrement, so a cooldown of `k * dt`
    blocks exactly `k` ticks. This runs in phase 2, before attacks resolve in phase 6, so the
    tick an entity fires still banks one tick of reload -- a one-tick leak accepted as cheaper
    than reordering the tick.
    """
    reload_seconds = stats.gather_kind(params.reload_seconds, state.ent_kind)
    max_ammo = stats.gather_kind(params.max_ammo, state.ent_kind)

    reloading = state.ent_attack_cd <= 0
    gain = torch.where(reloading, cfg.dt / reload_seconds, torch.zeros_like(reload_seconds))
    state.ent_ammo.copy_(torch.clamp(state.ent_ammo + gain, max=max_ammo))
    state.ent_attack_cd.copy_(torch.clamp(state.ent_attack_cd - cfg.dt, min=0))
    # The gadget cooldown is independent of attack_cd: firing the gadget neither pauses the reload
    # nor is blocked by the dash -- it is a separate button on its own timer.
    state.ent_gadget_cd.copy_(torch.clamp(state.ent_gadget_cd - cfg.dt, min=0))
    state.ent_invuln_t.copy_(torch.clamp(state.ent_invuln_t - cfg.dt, min=0))
    state.ent_reveal_t.copy_(torch.clamp(state.ent_reveal_t - cfg.dt, min=0))
    state.ent_react_t.copy_(torch.clamp(state.ent_react_t - cfg.dt, min=0))
    state.ent_out_of_combat_t.copy_(state.ent_out_of_combat_t + cfg.dt)  # stopwatch, counts up
    # Also a stopwatch, reset ONLY by an attack or super (env._attack_phase) -- never by taking
    # damage or the gadget; see the field's note in core/state.py.
    state.ent_attack_idle_t.copy_(state.ent_attack_idle_t + cfg.dt)


def long_dash_ready(state, params) -> torch.Tensor:
    """(N,E) bool -- has this entity gone `long_dash_seconds` without attacking, so its next dash
    reaches `long_dash_multiplier` times as far? (CHARACTER_DETAILS.md: Mortis.)

    False for any kind that does not configure the ability (`long_dash_seconds` resolves to 0):
    gated on the threshold being set, not on a huge threshold never being reached.
    """
    threshold = stats.gather_kind(params.long_dash_seconds, state.ent_kind)
    return (threshold > 0) & (state.ent_attack_idle_t >= threshold)


def long_dash_charge_frac(state, params) -> torch.Tensor:
    """(N,E) f32 in [0,1] -- progress toward the long dash. 1.0 means ready; zero for kinds
    without the ability. The observation carries it beside the boolean so a policy can plan
    against "0.8 charged, hold off attacking a second longer".
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
    is a `configs/brawlers.yaml` block rather than a plumbing change."""
    return stats.gather_kind(params.super_charge_hits, state.ent_kind) > 0


def super_ready(state, params) -> torch.Tensor:
    """(N,E) bool -- charged enough to fire. False for any kind without a super."""
    needed = stats.gather_kind(params.super_charge_hits, state.ent_kind)
    return has_super(state, params) & (state.ent_super_charge >= needed.to(state.ent_super_charge.dtype))


def super_charge_frac(state, params) -> torch.Tensor:
    """(N,E) f32 in [0,1] -- progress toward the super. The observation carries it beside the
    boolean, as with `long_dash_charge_frac`: a flag alone gives a policy nothing to plan
    against, while a fraction alone hides the moment the ability becomes available."""
    needed = stats.gather_kind(params.super_charge_hits, state.ent_kind).to(torch.float32)
    frac = state.ent_super_charge.to(torch.float32) / torch.clamp(needed, min=_EPS)
    return torch.where(needed > 0, torch.clamp(frac, max=1.0), torch.zeros_like(frac))


def add_super_charge(state, hits: torch.Tensor, params) -> None:
    """MUTATES: ent_super_charge. `hits` is (N,E) -- how many qualifying hits each entity landed
    this tick. Clamped at `super_charge_hits`, so charge never banks past full.

    Only hits on living PLAYERS count. The caller (env._bookkeeping) derives `hits` from the
    attacker x victim damage matrix rather than a "did I deal damage" flag because box damage
    flows through a different path, and would otherwise let a hero farm his super off crates."""
    needed = stats.gather_kind(params.super_charge_hits, state.ent_kind).to(state.ent_super_charge.dtype)
    charged = state.ent_super_charge + hits.to(state.ent_super_charge.dtype)
    state.ent_super_charge.copy_(torch.where(has_super(state, params), torch.minimum(charged, needed), state.ent_super_charge))


def gadget_ready(state, params) -> torch.Tensor:
    """(N,E) bool -- may this entity throw its gadget right now? `alive & gadget_cd <= 0 & the
    kind HAS a gadget` and nothing else: not ammo, not `attack_cd`, not `dash_t`.

    The one definition shared by `action_mask` (the hero's legality column) and
    `env._attack_phase`'s safety net, so the two cannot drift. Every bot kind resolves
    `gadget_cooldown` to 0, which is what keeps a bot from ever firing one."""
    cooldown = stats.gather_kind(params.gadget_cooldown, state.ent_kind)
    return state.ent_alive & (state.ent_gadget_cd <= 0) & (cooldown > 0)


def gadget_charge_frac(state, params) -> torch.Tensor:
    """(N,E) f32 in [0,1] -- progress back to a charged gadget, `1 - gadget_cd / gadget_cooldown`:
    0.0 on the tick it is thrown, 1.0 once the timer has run out, and 0.0 for any kind without a
    gadget (`gadget_cooldown` 0, which is every bot kind). The observation pairs it with
    `gadget_ready` for the reason `super_charge_frac` gives.

    No `alive` term, like `super_charge_frac`: a dead hero still reads its timer's progress. The
    clamp only makes the declared range exact (`state.check_invariants` bounds `gadget_cd`)."""
    cooldown = stats.gather_kind(params.gadget_cooldown, state.ent_kind)
    frac = 1.0 - state.ent_gadget_cd / torch.clamp(cooldown, min=_EPS)
    return torch.where(cooldown > 0, torch.clamp(frac, 0.0, 1.0), torch.zeros_like(frac))


def gadget_target(state, vis: torch.Tensor, params, bank, cfg):
    """Where each entity's gadget spinner would fly THIS tick: `(dir (N,E,2) f32 unit vectors,
    travel (N,E) f32 tiles)`. Pure; `env._attack_phase` feeds the pair into
    `projectiles.spawn_gadget` for the entities that fire.

    `vis` is the FAIR `(N,E,E)` visibility (`bots/perception.visibility`: `vis[n,e,j]` = e sees
    j): bush concealment only, NOT the camera window the observation's `enemy_revealed` plane
    also applies (`core/camera.hero_view`), so the hero's spinner can aim at an unconcealed enemy
    that is off-camera. A bushed, unrevealed enemy is skipped for a revealed one further away.
    With nothing revealed it flies `gadget_range` along the facing; a revealed enemy standing ON
    the thrower takes the facing too, with travel 0 (it detonates in place), so `dir` is a unit
    vector on every row.

    Computed for every entity, read for the hero: bots have `gadget_range 0` and never fire one,
    and masking their rows out would cost the tensor ops it saved.

    `travel` is `min(gadget_range, distance to the nearest revealed enemy)` -- it lands ON a close
    enemy -- then clipped against `blocks_proj`. The spinner is an ARTILLERY shell, not stopped in
    flight, so this pre-clip is the only thing that keeps it from landing (and blasting) on the
    far side of a wall.
    """
    N, E = state.ent_kind.shape
    device = state.ent_pos.device

    # (N,E,E): e's candidate targets j. The alive AND is re-applied so a hand-built or stale `vis`
    # cannot aim at a corpse; the diagonal goes because vis[n,e,e] is True for every living e.
    eye = torch.eye(E, dtype=torch.bool, device=device).unsqueeze(0)
    revealed = vis & state.ent_alive.unsqueeze(1) & ~eye

    # diff[n,e,j] = pos_j - pos_e: the vector FROM the thrower TO the candidate.
    diff = state.ent_pos.unsqueeze(1) - state.ent_pos.unsqueeze(2)  # (N,E,E,2)
    dist = geo.safe_norm(diff, dim=-1)                                # (N,E,E)
    dist_eff = torch.where(revealed, dist, torch.full_like(dist, float("inf")))
    nearest = torch.argmin(dist_eff, dim=-1)                          # (N,E) i64
    has = revealed.any(dim=-1)                                        # (N,E), no host sync

    to_vec = diff.gather(2, nearest.view(N, E, 1, 1).expand(N, E, 1, 2)).squeeze(2)  # (N,E,2)
    nearest_dist = dist.gather(2, nearest.unsqueeze(-1)).squeeze(-1)                 # (N,E)
    facing_vec = geo.from_angle(state.ent_facing)
    # `normalize` of a zero vector is zero (its norm is clamped, not the output), so a target
    # coincident with the thrower falls back to the facing like "nothing revealed" does: the
    # unit-vector contract holds on every row, and travel is 0 either way.
    aimed = has & (nearest_dist > _EPS)
    direction = torch.where(aimed.unsqueeze(-1), geo.normalize(to_vec), facing_vec)

    gadget_range = stats.gather_kind(params.gadget_range, state.ent_kind)
    travel = torch.where(has, torch.minimum(gadget_range, nearest_dist), gadget_range)

    # Wall clip. `march` reports the first BLOCKED sample, at or past the true wall face, so
    # landing one sample (`los_step_tiles`) short of `hit_t` -- the back-off `step_projectiles`
    # applies -- keeps the detonation point (`prj_target`) on the thrower's side. The clamp covers
    # an entity closer than one sample to the wall. `gadget_range` is a per-kind tensor and march's
    # budget must be a Python scalar, so this pays the full `cfg.ray_steps` over an (N,E) grid.
    hit, _, hit_t = terrain.march(bank.blocks_proj, state.map_id, state.ent_pos, direction, travel, cfg)
    clipped = torch.clamp(hit_t - cfg.los_step_tiles, min=0.0)
    travel = torch.where(hit, torch.minimum(clipped, travel), travel)

    return direction, travel


def auto_aim_target(state, params, cfg):
    """Where the hero's AUTO-AIMED dash (attack value 4) points this tick: `(direction (N,2)
    f32 unit vectors, has_target (N,) bool)`. Pure. `env._attack_phase` swaps it in for the
    hero's dash direction on the rows whose action asked for it.

    The target is the nearest (by centre distance) alive enemy or unbroken crate within reach:
    the dash's current distance (long-dash multiplier included, as `start_dash` applies it) plus
    `dash_radius` plus the target's body (`unit_radius`, or `_BOX_RADIUS` for a crate). No
    priority between enemies and crates, and no visibility check, so a policy tracking an enemy
    off-screen can dash at it, as the game's bare tap does (user decision, 2026-09-26).

    With nothing in reach, or only a target coincident with the hero, `has_target` is False and
    `direction` is the facing; the caller then keeps the ordinary dash direction (the move bin,
    or the facing when idle), so the dash still goes.
    """
    hero_pos = state.ent_pos[:, 0]                                    # (N,2)
    dash_distance = stats.gather_kind(params.dash_distance, state.ent_kind)[:, 0]
    dash_distance = dash_distance * long_dash_scale(state, params)[:, 0]
    dash_radius = stats.gather_kind(params.dash_radius, state.ent_kind)[:, 0]
    base_reach = dash_distance + dash_radius                          # (N,)

    enemy_diff = state.ent_pos[:, 1:] - hero_pos.unsqueeze(1)         # (N,E-1,2)
    enemy_dist = geo.safe_norm(enemy_diff, dim=-1)                    # (N,E-1)
    enemy_reach = (base_reach + params.unit_radius).unsqueeze(-1)
    enemy_ok = state.ent_alive[:, 1:] & (enemy_dist <= enemy_reach)

    box_diff = state.box_pos - hero_pos.unsqueeze(1)                  # (N,B,2)
    box_dist = geo.safe_norm(box_diff, dim=-1)                        # (N,B)
    box_ok = state.box_alive & (box_dist <= (base_reach + _BOX_RADIUS).unsqueeze(-1))

    diff = torch.cat([enemy_diff, box_diff], dim=1)                   # (N,E-1+B,2)
    dist = torch.cat([enemy_dist, box_dist], dim=1)
    ok = torch.cat([enemy_ok, box_ok], dim=1)
    return _aim_at_nearest(state, diff, dist, ok)


def super_aim_target(state, params, cfg):
    """Where the hero's IDLE super (attack value 2 on move bin 0) points this tick: `(direction
    (N,2) f32 unit vectors, has_target (N,) bool)`. Pure. `env._attack_phase` aims the hero's bolt
    with it on the rows whose move bin is idle; a super fired while moving follows the move bin.

    The game's tap-to-fire (user decision, 2026-09-30): the nearest (by centre distance) alive
    enemy within the bolt's reach, `super_range + super_radius + unit_radius`, the farthest centre
    its straight flight can touch. `auto_aim_target`'s rule with the super's reach and without
    crates, which the bolt passes through without damaging; no visibility term, as there. The
    long dash does not stretch it.

    With no enemy in reach, or the nearest one coincident with the hero, `has_target` is False
    and `direction` is the facing, the idle dash's rule, so the bolt still flies: an idle bin's
    `move_dir` is (0, 0), and `projectiles.spawn_supers` launches a bolt along a zero aim that
    never moves and never expires.
    """
    hero_pos = state.ent_pos[:, 0]                                    # (N,2)
    super_range = stats.gather_kind(params.super_range, state.ent_kind)[:, 0]
    super_radius = stats.gather_kind(params.super_radius, state.ent_kind)[:, 0]

    diff = state.ent_pos[:, 1:] - hero_pos.unsqueeze(1)               # (N,E-1,2)
    dist = geo.safe_norm(diff, dim=-1)                                # (N,E-1)
    reach = (super_range + super_radius + params.unit_radius).unsqueeze(-1)
    ok = state.ent_alive[:, 1:] & (dist <= reach)
    return _aim_at_nearest(state, diff, dist, ok)


def _aim_at_nearest(state, diff, dist, ok):
    """The shared tail of `auto_aim_target` and `super_aim_target`: `(direction (N,2), has_target
    (N,))` toward the nearest candidate that `ok` (N,M) admits. `diff` (N,M,2) runs from the hero
    to each candidate and `dist` (N,M) is its length. A row whose nearest candidate is coincident
    with the hero, or that has none, gets no target and the hero's facing."""
    dist_eff = torch.where(ok, dist, torch.full_like(dist, float("inf")))
    nearest = torch.argmin(dist_eff, dim=-1)                          # (N,)
    N = diff.shape[0]
    to_vec = diff.gather(1, nearest.view(N, 1, 1).expand(N, 1, 2)).squeeze(1)  # (N,2)
    nearest_dist = dist.gather(1, nearest.unsqueeze(-1)).squeeze(-1)           # (N,)

    has_target = ok.any(dim=-1) & (nearest_dist > _EPS)
    facing_vec = geo.from_angle(state.ent_facing[:, 0])
    direction = torch.where(has_target.unsqueeze(-1), geo.normalize(to_vec), facing_vec)
    return direction, has_target


def start_dash(state, fire: torch.Tensor, move_dir: torch.Tensor, bank, params, cfg) -> None:
    """fire: (N,E) bool, move_dir: (N,E,2) f32 -- kind-agnostic, gated on dash_distance[kind] > 0.

    MUTATES: ent_ammo, ent_attack_cd, ent_dash_t, ent_dash_dir, ent_dash_speed, ent_dash_hits,
    ent_facing, ent_shots_fired.

    The dash grants NO invulnerability (user decision, 2026-09-25: it is an attack animation, and
    Mortis can be hit throughout it). Nothing seeds `ent_invuln_t`; the field stays so
    `hero.invuln` keeps its slot in every observation spec, reading False.
    """
    E = state.ent_kind.shape[1]

    dash_distance = stats.gather_kind(params.dash_distance, state.ent_kind)
    dash_duration = stats.gather_kind(params.dash_duration, state.ent_kind)
    attack_cooldown = stats.gather_kind(params.attack_cooldown, state.ent_kind)

    # LONG DASH: after `long_dash_seconds` without attacking, the next dash covers
    # `long_dash_multiplier` times its distance. `dash_duration` is deliberately NOT scaled, so a
    # long dash is also faster, as in the game; the body clip and `dash_speed = clipped /
    # duration` below are distance-agnostic.
    dash_distance = dash_distance * long_dash_scale(state, params)

    move_is_zero = (move_dir[..., 0] == 0) & (move_dir[..., 1] == 0)
    can_dash = fire & state.ent_alive & (dash_distance > 0)
    if cfg.dash_on_idle == "block":
        can_dash = can_dash & ~move_is_zero

    facing_vec = geo.from_angle(state.ent_facing)
    dash_dir = torch.where(move_is_zero.unsqueeze(-1), facing_vec, move_dir)

    # The dash stops where the BODY first touches a wall, with no slide along it (user's rule,
    # 2026-09-25: walls stop momentum, they do not redirect it). `terrain.body_travel` runs
    # `circle_blocked` along the line, so the landing is always a point walking can leave: a body
    # overlapping a wall is blocked in every direction, and `movement.resolve_move` would reject
    # every step. Clipping only the CENTRE line would clear the body head-on but not beside a
    # wall. `_WALL_CLEARANCE` keeps it a millitile short of the contact; the march is budgeted by
    # `params.dash_ray_tiles`, the spec's longest possible dash, not the full `cfg.ray_steps`.
    radius = params.unit_radius.unsqueeze(-1)  # (N,1), broadcasts against (N,E)
    clipped_distance = terrain.body_travel(
        bank.blocks_unit, state.map_id, state.ent_pos, dash_dir, dash_distance, radius, cfg,
        max_tiles=params.dash_ray_tiles, clearance=_WALL_CLEARANCE,
    )
    dash_speed = clipped_distance / torch.clamp(dash_duration, min=_EPS)

    state.ent_ammo.copy_(torch.where(can_dash, state.ent_ammo - 1.0, state.ent_ammo))
    state.ent_attack_cd.copy_(torch.where(can_dash, attack_cooldown, state.ent_attack_cd))
    state.ent_dash_t.copy_(torch.where(can_dash, dash_duration, state.ent_dash_t))
    state.ent_dash_dir.copy_(torch.where(can_dash.unsqueeze(-1), dash_dir, state.ent_dash_dir))
    state.ent_dash_speed.copy_(torch.where(can_dash, dash_speed, state.ent_dash_speed))
    state.ent_facing.copy_(torch.where(can_dash, geo.angle_of(dash_dir), state.ent_facing))
    state.ent_shots_fired.copy_(torch.where(can_dash, state.ent_shots_fired + 1, state.ent_shots_fired))

    row_mask = can_dash.unsqueeze(-1).expand(-1, -1, E)
    state.ent_dash_hits.copy_(torch.where(row_mask, torch.zeros_like(state.ent_dash_hits), state.ent_dash_hits))
    state.ent_dash_box_hits.masked_fill_(can_dash.unsqueeze(-1), False)


def advance_dash(state, params, cfg):
    """MUTATES: ent_pos, ent_vel, ent_dash_t, ent_dash_dir, ent_dash_speed, ent_dash_hits,
    ent_dash_box_hits. Returns (dmg_ent (N,E), dmg_by (N,E,E), dmg_box (N,B)) -- reported here,
    applied by combat.apply_damage, the same pattern as projectile damage.

    A unit is hit once per dash (`ent_dash_hits`). A crate is too under `boxes.dash_hits_once`
    (`ent_dash_box_hits`); with it off, a crate inside the capsule is hit on every tick."""
    E = state.ent_kind.shape[1]
    is_dashing = state.ent_dash_t > 0

    dash_radius = stats.gather_kind(params.dash_radius, state.ent_kind)

    # The last tick advances only the time LEFT, not a full dt. A `dash_duration` such as 0.30 is
    # not a float32-exact multiple of dt, so ~1e-8 s survives its last full tick; a whole extra
    # tick would stretch the dash (7/6x) past start_dash's body clip and into the wall, where
    # `movement.resolve_move` rejects every escape step. Pinned by tests/test_hero.py::
    # test_a_charged_dash_into_a_wall_still_lands_the_body_clear_of_it.
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
    if cfg.box_dash_hits_once:
        box_hit = box_hit & ~state.ent_dash_box_hits
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
    if cfg.box_dash_hits_once:
        state.ent_dash_box_hits.logical_or_(box_hit).masked_fill_(finished.unsqueeze(-1), False)

    return dmg_ent, dmg_by, dmg_box
