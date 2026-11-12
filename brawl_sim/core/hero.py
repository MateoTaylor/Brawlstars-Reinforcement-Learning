"""Hero action decoding, per-entity timers, the ability predicates (long dash, super, gadget), the
kind-agnostic dash and the CHARGE super that rides it (Bull: `start_charge`, `advance_dash`), and
the STOMP gadget (Bull's Stomper: `start_stomp`, `advance_stomp`).

dash_t is owned by advance_dash alone: tick_timers does not decrement it, since doing both would
halve every dash against start_dash's `dash_speed = clipped_distance / dash_duration`.

ent_out_of_combat_t is a stopwatch counting UP, not a countdown: regen waits for it to reach
`regen_delay` (combat.apply_regen). Taking damage (combat.apply_damage) and any offensive action
-- attack, super or gadget (env._attack_phase) -- reset it to 0.
"""
import torch

from ..constants import SuperType
from . import geometry as geo
from . import stats
from . import status
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

    # Stunned or airborne (core/status.py): no attack, super or gadget. The move half stays open,
    # and movement ignores it until the stun ends.
    free = ~status.disabled(state)[:, 0]
    ready = hero_alive & (hero_cd <= 0) & (hero_dash_t <= 0) & free
    fire_ok = ready & (hero_ammo >= 1.0)
    # A super costs CHARGE, not ammo -- an empty clip must not block it. It shares the cooldown and
    # the no-dashing rule, so it cannot be used to sidestep either.
    super_ok = ready & super_ready(state, params)[:, 0]
    # The gadget is a separate button on its own timer: deliberately NOT ANDed with `ready`, so it
    # is legal mid-dash, during the attack cooldown and on an empty clip, as in the game. A stun
    # still blocks it, and so does a charge super (`gadget_ready`).
    gadget_ok = gadget_ready(state, params)[:, 0] & free
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
    ent_out_of_combat_t, ent_attack_idle_t and the status effects' timers (`status.tick`), NOT
    ent_dash_t (see module docstring). Countdowns clamp at 0, the attack's, the gadget's and the
    status timers snapping to it (`status.count_down`); the two stopwatches count up.

    Ammo (+dt/reload_seconds, clamped to max_ammo) accrues only while attack_cd is 0, so firing
    pauses the reload for `attack_cooldown` and sustained fire costs `attack_cooldown +
    reload_seconds` per shot. The gate is read BEFORE the decrement, so a cooldown of `k * dt`
    blocks exactly `k` ticks. This runs in phase 2, before attacks resolve in phase 6, so the
    tick an entity fires still banks one tick of reload -- a one-tick leak accepted as cheaper
    than reordering the tick.

    Bull's star power, Berserker, multiplies the gain while his HP is below a threshold, read off
    this tick's HP before any of its damage (see the gain's comment).
    """
    reload_seconds = stats.gather_kind(params.reload_seconds, state.ent_kind)
    max_ammo = stats.gather_kind(params.max_ammo, state.ent_kind)

    reloading = state.ent_attack_cd <= 0
    gain = torch.where(reloading, cfg.dt / reload_seconds, torch.zeros_like(reload_seconds))
    # Berserker (user decision 2026-10-23, plan Q1): below `reload_boost_below_hp_frac` of max HP,
    # cubes' bonus included, the reload runs `reload_boost_mult` times as fast, and healing back to
    # the threshold ends it. Every other kind's threshold is 0, which no HP is below, so the where
    # hands their `gain` through untouched.
    boost_below = stats.gather_kind(params.reload_boost_below_hp_frac, state.ent_kind)
    boosted = state.ent_hp < boost_below * state.ent_max_hp
    gain = torch.where(boosted, gain * stats.gather_kind(params.reload_boost_mult, state.ent_kind), gain)
    state.ent_ammo.copy_(torch.clamp(state.ent_ammo + gain, max=max_ammo))
    # Snapped as the status timers are (user decision 2026-10-23): plain float32 subtraction
    # leaves ~1e-8 s after the fifth tick of 0.25, which held the shot and the reload of every
    # 0.25-s kind (Bull, sniper, artillery, rifle, spike) a sixth tick. Mortis's 0.35, Edgar's
    # 0.50 and Buzz's 1.00 reach 0 on their last tick either way.
    state.ent_attack_cd.copy_(status.count_down(state.ent_attack_cd, cfg.dt))
    # The gadget cooldown is independent of attack_cd: firing the gadget neither pauses the reload
    # nor is blocked by the dash -- it is a separate button on its own timer. It snaps to 0 the
    # same way: plain float32 subtraction leaves 1.8e-5 s of Bull's 20.0 after its 400th tick,
    # which would hold the gadget one tick longer, while Mortis's 18.0 reaches 0 on its 360th
    # either way.
    state.ent_gadget_cd.copy_(status.count_down(state.ent_gadget_cd, cfg.dt))
    state.ent_invuln_t.copy_(torch.clamp(state.ent_invuln_t - cfg.dt, min=0))
    state.ent_reveal_t.copy_(torch.clamp(state.ent_reveal_t - cfg.dt, min=0))
    state.ent_react_t.copy_(torch.clamp(state.ent_react_t - cfg.dt, min=0))
    state.ent_out_of_combat_t.copy_(state.ent_out_of_combat_t + cfg.dt)  # stopwatch, counts up
    # Also a stopwatch, reset ONLY by an attack or super (env._attack_phase) -- never by taking
    # damage or the gadget; see the field's note in core/state.py.
    state.ent_attack_idle_t.copy_(state.ent_attack_idle_t + cfg.dt)
    status.tick(state, cfg)


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
    kind HAS a gadget & not mid-charge`, and nothing else: not ammo, not `attack_cd`, not a
    dash's `dash_t`.

    The one definition shared by `action_mask` (the hero's legality column), the observation's
    `hero.gadget_ready` and `env._attack_phase`'s safety net, so the three cannot drift. Every bot
    kind resolves `gadget_cooldown` to 0, which is what keeps a bot from ever firing one.

    A CHARGE super holds the gadget through its whole wind-up and rush (`ent_dash_super`; user
    decision 2026-10-23, plan Q5), as `dash_t` holds the attack and the super; Mortis's dash does
    not. The term sits behind the charge's Python switch, so a spec with no CHARGE kind never
    computes it."""
    cooldown = stats.gather_kind(params.gadget_cooldown, state.ent_kind)
    ready = state.ent_alive & (state.ent_gadget_cd <= 0) & (cooldown > 0)
    if params.charge_ray_tiles > 0:
        ready = ready & ~state.ent_dash_super
    return ready


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


def weapon_reach(state, params, charged: bool) -> torch.Tensor:
    """(N,) f32 -- how far from the hero's centre his attack can touch, short of the target's own
    body (callers add `unit_radius` or `_BOX_RADIUS`).

    A kind with a dash (Mortis) reaches `dash_distance + dash_radius`, the dash scaled by the long
    dash when `charged` (auto-aim reads the dash `start_dash` would make now; `in_reach_radius`
    reads the uncharged one). A kind without one (Bull) fires a volley, which
    reaches `attack_range + proj_radius`: the farthest a pellet's edge travels (BULL_HERO_PLAN.md
    D5). A selection, not a blend, so a dash kind's reach is the same float it always was."""
    kind = state.ent_kind
    dash_distance = stats.gather_kind(params.dash_distance, kind)[:, 0]
    if charged:
        dash_distance = dash_distance * long_dash_scale(state, params)[:, 0]
    dash_reach = dash_distance + stats.gather_kind(params.dash_radius, kind)[:, 0]
    shot_reach = (stats.gather_kind(params.attack_range, kind)[:, 0]
                  + stats.gather_kind(params.proj_radius, kind)[:, 0])
    has_dash = stats.gather_kind(params.dash_distance, kind)[:, 0] > 0
    return torch.where(has_dash, dash_reach, shot_reach)


def in_reach_radius(state, params) -> torch.Tensor:
    """(N,) f32 -- the reward's `attack_in_reach` radius: how far an enemy's CENTRE may be from
    the hero's for his attack to count as in reach (env._attack_phase, and the audit's utilization
    in scripts/audit_attack_cadence.py, so the term pays for exactly what the audit measures).

    A kind with a dash (Mortis): the uncharged `weapon_reach` plus the enemy's body, 3.77 tiles,
    bit for bit the float it always was. A volley kind (Bull): the farthest an enemy aimed at dead
    centre still has MOST of the volley touching it (user decision 2026-10-23, BULL_HERO_PLAN.md
    §5.2), i.e. a strict majority of `proj_count` -- Bull's middle three of five, 5.42 tiles at his
    measured 5.2-tile range (5.61 at the old 7.0, where the pellet passed abeam before its end). His
    `weapon_reach` (5.9 to a centre) is where only the middle pellet can, and auto-aim keeps that.

    spawn_volley's fan puts pellet k at `spread * (k - (n-1)/2) / max(n-1, 1)`. Sorted outward,
    those offsets are 0, 1, 1, 2, 2, ... pellet gaps for an odd n and 1/2, 1/2, 3/2, 3/2, ... for
    an even one, so the outermost of the innermost n//2 + 1 sits ceil(n/2)//2 gaps out, plus half
    a gap when n is even. That pellet's path, a segment of `attack_range` from the hero's centre
    (projectile hits are swept against `proj_radius + unit_radius`), passes within `touch` of the
    enemy out to `touch / sin` while it is still abeam, else out to its end point."""
    kind = state.ent_kind
    dash_reach = weapon_reach(state, params, charged=False) + params.unit_radius
    count = torch.clamp(stats.gather_kind(params.proj_count, kind)[:, 0], min=1).to(torch.float32)
    spread = stats.gather_kind(params.proj_spread_rad, kind)[:, 0]
    gaps_out = torch.floor(torch.ceil(count / 2) / 2) + 0.5 * (torch.remainder(count, 2) == 0)
    theta = gaps_out * spread / torch.clamp(count - 1, min=1)
    attack_range = stats.gather_kind(params.attack_range, kind)[:, 0]
    touch = stats.gather_kind(params.proj_radius, kind)[:, 0] + params.unit_radius
    sin, cos = torch.sin(theta), torch.cos(theta)
    abeam = touch / torch.clamp(sin, min=_EPS)
    tip = attack_range * cos + torch.sqrt(torch.clamp(touch ** 2 - (attack_range * sin) ** 2, min=0.0))
    volley_reach = torch.where(attack_range * sin >= touch * cos, abeam, tip)
    has_dash = stats.gather_kind(params.dash_distance, kind)[:, 0] > 0
    return torch.where(has_dash, dash_reach, volley_reach)


def auto_aim_target(state, params, cfg):
    """Where the hero's AUTO-AIMED attack (attack value 4) points this tick: `(direction (N,2)
    f32 unit vectors, has_target (N,) bool)`. Pure. `env._attack_phase` swaps it in for the
    hero's dash direction, or his volley's aim for a kind without a dash, on the rows whose
    action asked for it.

    The target is the nearest (by centre distance) alive enemy or unbroken crate within reach:
    `weapon_reach` (the dash's current distance, long-dash multiplier included as `start_dash`
    applies it, plus `dash_radius`; or for Bull his range plus a pellet's radius) plus the
    target's body (`unit_radius`, or `_BOX_RADIUS` for a crate). No priority between enemies and
    crates, and no visibility check, so a policy tracking an enemy off-screen can attack it, as
    the game's bare tap does (user decision, 2026-09-26). An airborne enemy is untargetable
    (core/status.py) and skipped.

    With nothing in reach, or only a target coincident with the hero, `has_target` is False and
    `direction` is the facing; the caller then keeps the ordinary direction (the move bin, or the
    facing when idle), so the attack still goes.
    """
    hero_pos = state.ent_pos[:, 0]                                    # (N,2)
    base_reach = weapon_reach(state, params, charged=True)            # (N,)

    enemy_diff = state.ent_pos[:, 1:] - hero_pos.unsqueeze(1)         # (N,E-1,2)
    enemy_dist = geo.safe_norm(enemy_diff, dim=-1)                    # (N,E-1)
    enemy_reach = (base_reach + params.unit_radius).unsqueeze(-1)
    enemy_ok = state.ent_alive[:, 1:] & ~status.airborne(state)[:, 1:] & (enemy_dist <= enemy_reach)

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
    crates, which the bolt passes through without damaging; no visibility term, as there; and no
    airborne enemy (core/status.py). The long dash does not stretch it.

    A CHARGE super (Bull) hits on centres, as the dash does (`advance_dash`'s capsule), so its
    reach drops the `unit_radius`: `super_range + super_radius`, the farthest centre an
    unobstructed rush can touch.

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
    is_charge = stats.gather_kind(params.super_type, state.ent_kind)[:, 0] == int(SuperType.CHARGE)
    body = torch.where(is_charge, torch.zeros_like(params.unit_radius), params.unit_radius)
    reach = (super_range + super_radius + body).unsqueeze(-1)
    ok = state.ent_alive[:, 1:] & ~status.airborne(state)[:, 1:] & (dist <= reach)
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


def start_charge(state, fire: torch.Tensor, aim_dir: torch.Tensor, bank, params, cfg) -> None:
    """fire: (N,E) bool -- the entities whose CHARGE super fires this tick, already gated and
    booked by env._attack_phase (meter spent, cooldown, shot). aim_dir: (N,E,2), nonzero on every
    firing row (`super_dir` there). MUTATES: ent_dash_t, ent_dash_dir, ent_dash_speed,
    ent_dash_super, ent_dash_windup_t, ent_dash_hits, ent_dash_box_hits, ent_facing.

    Bull's Bulldozer (BULL_HERO_PLAN.md D2) rides the dash's state. So every `dash_t <= 0` gate
    (the action mask, walking, the attack phase's `can_*`) holds him for the whole charge, and the
    obs' `hero.dashing`, `hero.dash_t` and `hero.dash_dir` read as charging. The charge:
    - stands still for `super_windup_seconds`, then rushes along `aim_dir` at the FIXED
      `super_charge_speed`, so a charge cut short by a wall ends sooner rather than slower;
    - covers `super_range` tiles, or stops where the body first touches a wall or water (user
      decision 2026-10-23, D3: like the dash, with no pass-through and nothing destroyed). It uses
      `start_dash`'s `terrain.body_travel` march and wall clearance, budgeted by
      `params.charge_ray_tiles`.
    `ent_dash_t` is therefore the wind-up plus the rush, and `advance_dash` does the rest. A charge
    grants no invulnerability, as the dash grants none.
    """
    E = state.ent_kind.shape[1]

    charge_range = stats.gather_kind(params.super_range, state.ent_kind)
    speed = stats.gather_kind(params.super_charge_speed, state.ent_kind)
    windup = stats.gather_kind(params.super_windup_seconds, state.ent_kind)

    direction = geo.normalize(aim_dir)
    radius = params.unit_radius.unsqueeze(-1)  # (N,1), broadcasts against (N,E)
    rush = terrain.body_travel(
        bank.blocks_unit, state.map_id, state.ent_pos, direction, charge_range, radius, cfg,
        max_tiles=params.charge_ray_tiles, clearance=_WALL_CLEARANCE,
    )
    duration = windup + rush / torch.clamp(speed, min=_EPS)
    # A charge with no wind-up and no room to move would end before advance_dash saw it, leaving
    # its flag behind; validate() allows a 0 wind-up, so drop that one rather than start it.
    can_charge = fire & state.ent_alive & (duration > 0)

    state.ent_dash_t.copy_(torch.where(can_charge, duration, state.ent_dash_t))
    state.ent_dash_dir.copy_(torch.where(can_charge.unsqueeze(-1), direction, state.ent_dash_dir))
    state.ent_dash_speed.copy_(torch.where(can_charge, speed, state.ent_dash_speed))
    state.ent_dash_super.copy_(state.ent_dash_super | can_charge)
    state.ent_dash_windup_t.copy_(torch.where(can_charge, windup, state.ent_dash_windup_t))
    state.ent_facing.copy_(torch.where(can_charge, geo.angle_of(direction), state.ent_facing))

    row_mask = can_charge.unsqueeze(-1).expand(-1, -1, E)
    state.ent_dash_hits.copy_(torch.where(row_mask, torch.zeros_like(state.ent_dash_hits), state.ent_dash_hits))
    state.ent_dash_box_hits.masked_fill_(can_charge.unsqueeze(-1), False)


def advance_dash(state, params, cfg, bank=None):
    """MUTATES: ent_pos, ent_vel, ent_dash_t, ent_dash_dir, ent_dash_speed, ent_dash_hits,
    ent_dash_box_hits; under a charge also ent_dash_super, ent_dash_windup_t, ent_slow_window_t
    and its victims' ent_pos, ent_vel, ent_airborne_t, ent_slow_t and ent_slow_frac. Returns (dmg_ent (N,E), dmg_by
    (N,E,E), dmg_box (N,B)) -- reported here, applied by combat.apply_damage, the same pattern as
    projectile damage.

    A unit is hit once per dash (`ent_dash_hits`). A crate is too under `boxes.dash_hits_once`
    (`ent_dash_box_hits`); with it off, a crate inside the capsule is hit on every tick. An
    airborne unit (core/status.py) is passed through, neither hit nor marked, so a dash still
    inside it when it lands hits it then.

    A CHARGE super (`ent_dash_super`, from `start_charge`) differs from a dash as below. All of it
    sits behind a Python switch on `params.charge_ray_tiles`, which is 0 in every spec without a
    CHARGE kind, so a Mortis run never executes it. `bank` is needed for the push, and a spec with
    a CHARGE kind must pass it.
    - The first `ent_dash_windup_t` seconds stand still and hit nothing.
    - It hits on `super_radius` for the flat `super_damage` and passes crates untouched, as the
      bolt, the other super, does.
    - Each player it hits is pushed `super_knockback_tiles` straight away from the charger's
      centre at the moment of contact (user decision 2026-10-23): along the charge for a head-on
      hit, at an angle for a glancing one. Walls and water stop the push (`terrain.body_travel`),
      and the player is launched (airborne) for `super_airborne_seconds`.
    - On its last tick, everyone else alive within `super_slow_radius` of the landing, airborne
      or not, is slowed once (`status.apply_slow`; user decision 2026-10-23), and the charger's
      `ent_slow_window_t` starts the same `super_slow_seconds`, caught or not (D6).
    """
    E = state.ent_kind.shape[1]
    is_dashing = state.ent_dash_t > 0
    charging = params.charge_ray_tiles > 0  # a Python bool: does this spec have a CHARGE kind?
    if charging and bank is None:
        raise ValueError("advance_dash: a spec with a CHARGE super must pass `bank` for its knockback")

    dash_radius = stats.gather_kind(params.dash_radius, state.ent_kind)

    # The last tick advances only the time LEFT, not a full dt. A `dash_duration` such as 0.30 is
    # not a float32-exact multiple of dt, so ~1e-8 s survives its last full tick; a whole extra
    # tick would stretch the dash (7/6x) past start_dash's body clip and into the wall, where
    # `movement.resolve_move` rejects every escape step. Pinned by tests/test_hero.py::
    # test_a_charged_dash_into_a_wall_still_lands_the_body_clear_of_it.
    step_seconds = torch.clamp(state.ent_dash_t, max=cfg.dt)

    dmg = stats.effective_damage(state.ent_kind, state.ent_cubes, params)  # (N,E)
    hits_ok = is_dashing        # which dashers' capsules can hit a unit this tick
    box_dasher = is_dashing     # ... and a crate
    if charging:
        is_charge = is_dashing & state.ent_dash_super
        # The wind-up is the first stretch of dash_t: this tick's share of it stands still, and
        # only the rest of the tick moves. Distance therefore stays speed x rush seconds, the
        # clip start_charge made. A tick that does not move hits nothing: its capsule would be a
        # circle around the charger standing at his start.
        still = torch.minimum(state.ent_dash_windup_t, step_seconds)
        step_seconds = torch.where(is_charge, step_seconds - still, step_seconds)
        hits_ok = is_dashing & ~(is_charge & (step_seconds <= _EPS))
        box_dasher = is_dashing & ~is_charge
        dash_radius = torch.where(is_charge, stats.gather_kind(params.super_radius, state.ent_kind), dash_radius)
        dmg = torch.where(is_charge, stats.gather_kind(params.super_damage, state.ent_kind), dmg)

    delta = state.ent_dash_dir * (state.ent_dash_speed * step_seconds).unsqueeze(-1)
    old_pos = state.ent_pos
    new_pos = old_pos + delta

    p0 = old_pos.unsqueeze(2)          # (N,E,1,2) attacker axis
    p1 = new_pos.unsqueeze(2)          # (N,E,1,2)
    r = dash_radius.unsqueeze(2)       # (N,E,1)

    # --- units ---
    victim_pos = state.ent_pos.unsqueeze(1)  # (N,1,E,2)
    inside = geo.capsule_contains(p0, p1, r, victim_pos)  # (N,E,E)
    eye = torch.eye(E, dtype=torch.bool, device=state.ent_pos.device).unsqueeze(0)
    new_hit = (
        inside & ~eye
        & (state.ent_alive & ~status.airborne(state)).unsqueeze(1) & state.ent_alive.unsqueeze(2)
        & hits_ok.unsqueeze(2) & ~state.ent_dash_hits
    )
    dmg_by = torch.where(new_hit, dmg.unsqueeze(2), torch.zeros_like(new_hit, dtype=dmg.dtype))
    dmg_ent = dmg_by.sum(dim=1)

    if charging:
        # Where each hit made contact, read now: `old_pos` is `state.ent_pos` itself, which the
        # move below overwrites. The push runs from the charger's centre at that moment to the
        # victim's, falling back to the charge's own direction for a victim dead on the centre.
        # A graze the capsule test passes can round to no crossing at all; its contact is then
        # the closest approach, not the start.
        knocked = new_hit & is_charge.unsqueeze(2)                                  # (N,E,E)
        crossed, contact_t = geo.segment_circle_hit(p0, p1, victim_pos, r)           # (N,E,E)
        contact = torch.where(crossed.unsqueeze(-1), p0 + contact_t.unsqueeze(-1) * (p1 - p0),
                              geo.closest_point_on_segment(p0, p1, victim_pos))     # (N,E,E,2)
        away = victim_pos - contact
        dead_centre = geo.safe_norm(away, dim=-1) <= _EPS
        away = torch.where(dead_centre.unsqueeze(-1), state.ent_dash_dir.unsqueeze(2), away)
        knock_tiles = stats.gather_kind(params.super_knockback_tiles, state.ent_kind)  # (N,E)
        push = (torch.where(knocked.unsqueeze(-1), geo.normalize(away), torch.zeros_like(away))
                * knock_tiles.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)                # (N,E,2) per victim

    # --- boxes ---
    box_pos = state.box_pos.unsqueeze(1)  # (N,1,B,2)
    box_inside = geo.capsule_contains(p0, p1, r, box_pos)  # (N,E,B)
    box_hit = box_inside & state.box_alive.unsqueeze(1) & box_dasher.unsqueeze(2)
    if cfg.box_dash_hits_once:
        box_hit = box_hit & ~state.ent_dash_box_hits
    box_dmg = torch.where(box_hit, dmg.unsqueeze(2), torch.zeros_like(box_hit, dtype=dmg.dtype))
    dmg_box = box_dmg.sum(dim=1)

    # --- movement (dashers pass through units; no terrain re-resolve, no separation) ---
    state.ent_pos.copy_(torch.where(is_dashing.unsqueeze(-1), new_pos, old_pos))
    state.ent_vel.copy_(torch.where(is_dashing.unsqueeze(-1), delta / cfg.dt, state.ent_vel))

    if charging:
        # The push, clipped where the victim's body first touches a wall or water, as a dash is.
        # `ent_vel` takes it too, so it stays this tick's displacement over dt.
        push_dist = geo.safe_norm(push, dim=-1)
        push_dir = geo.normalize(push)
        travel = terrain.body_travel(
            bank.blocks_unit, state.map_id, state.ent_pos, push_dir, push_dist,
            params.unit_radius.unsqueeze(-1), cfg, max_tiles=params.knockback_ray_tiles,
            clearance=_WALL_CLEARANCE,
        )
        shove = push_dir * travel.unsqueeze(-1)
        was_knocked = knocked.any(dim=1)                                             # (N,E)
        state.ent_pos.copy_(torch.where(was_knocked.unsqueeze(-1), state.ent_pos + shove, state.ent_pos))
        state.ent_vel.copy_(torch.where(was_knocked.unsqueeze(-1), state.ent_vel + shove / cfg.dt, state.ent_vel))
        airborne_s = stats.gather_kind(params.super_airborne_seconds, state.ent_kind)  # the charger's
        status.apply_airborne(state, was_knocked,
                              torch.where(knocked, airborne_s.unsqueeze(2), torch.zeros_like(dmg_by)).amax(dim=1))

    # --- dash_t countdown + completion cleanup ---
    new_dash_t = torch.clamp(state.ent_dash_t - cfg.dt, min=0)
    state.ent_dash_t.copy_(torch.where(is_dashing, new_dash_t, state.ent_dash_t))
    finished = is_dashing & (new_dash_t <= 0)

    if charging:
        # The landing pulse, from where the charge ended, on everyone the push has already moved.
        # A slow is not a hit, so an airborne player is caught too: the Phase 5 Stomper reads a
        # slow on a target the charge just launched.
        landed = finished & is_charge & state.ent_alive                              # (N,E) charger
        gap = geo.safe_norm(state.ent_pos.unsqueeze(2) - state.ent_pos.unsqueeze(1), dim=-1)  # (N,E,E)
        slow_r = stats.gather_kind(params.super_slow_radius, state.ent_kind)
        caught = landed.unsqueeze(2) & (gap <= slow_r.unsqueeze(2)) & ~eye & state.ent_alive.unsqueeze(1)
        slow_s = stats.gather_kind(params.super_slow_seconds, state.ent_kind)
        slow_f = stats.gather_kind(params.super_slow_frac, state.ent_kind)
        zero = torch.zeros_like(gap)
        status.apply_slow(state, caught.any(dim=1),
                          torch.where(caught, slow_f.unsqueeze(2), zero).amax(dim=1),
                          torch.where(caught, slow_s.unsqueeze(2), zero).amax(dim=1))
        # The charger's copy of that slow's clock, on every landing whether it caught anyone or
        # not (the obs' `hero.slow_window`, the Stomper's "will this stomp stun?").
        state.ent_slow_window_t.copy_(torch.where(landed, slow_s, state.ent_slow_window_t))
        state.ent_dash_windup_t.copy_(torch.where(
            is_charge & ~finished, torch.clamp(state.ent_dash_windup_t - cfg.dt, min=0),
            torch.zeros_like(state.ent_dash_windup_t)))
        state.ent_dash_super.copy_(state.ent_dash_super & ~finished)

    state.ent_dash_dir.copy_(torch.where(finished.unsqueeze(-1), torch.zeros_like(state.ent_dash_dir), state.ent_dash_dir))
    state.ent_dash_speed.copy_(torch.where(finished, torch.zeros_like(state.ent_dash_speed), state.ent_dash_speed))

    updated_hits = state.ent_dash_hits | new_hit
    finished_row = finished.unsqueeze(-1).expand(-1, -1, E)
    state.ent_dash_hits.copy_(torch.where(finished_row, torch.zeros_like(updated_hits), updated_hits))
    if cfg.box_dash_hits_once:
        state.ent_dash_box_hits.logical_or_(box_hit).masked_fill_(finished.unsqueeze(-1), False)

    return dmg_ent, dmg_by, dmg_box


def start_stomp(state, fire: torch.Tensor, params) -> None:
    """fire: (N,E) bool -- the entities whose STOMP gadget fires this tick, already gated and
    booked by env._attack_phase (cooldown, reveal). MUTATES: ent_stomp_t.

    Bull's Stomper (BULL_HERO_PLAN.md §5.5) arms here and lands in `advance_stomp`
    `stomp_windup_seconds` later: 0.30 s, measured 2026-11-01 (tap to pulse 0.30-0.33 s over
    three stomps; the wiki said 0.25). Only the pulse
    waits. He walks and acts through the wind-up as usual, and the pulse lands around wherever
    he stands when it runs out."""
    windup = stats.gather_kind(params.stomp_windup_seconds, state.ent_kind)
    state.ent_stomp_t.copy_(torch.where(fire, windup, state.ent_stomp_t))


def advance_stomp(state, params, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    """MUTATES: ent_stomp_t, and the caught players' ent_slow_t, ent_slow_frac and ent_stun_t.
    Returns `(caught, lands)`: caught (N,E,E) bool, [n, e, j] = e's stomp landed on player j this
    tick; lands (N,E) bool, e's stomp landed this tick, on players or on nobody (the reward's
    `gadget_miss` is a landing that caught none).

    Counts every armed stomp down one tick (`status.count_down`, so a 0.30 s wind-up armed on
    tick t lands on tick t + 6) and lands those that reach 0, around the stomper's centre now.
    The pulse (user decision 2026-10-23: the current wiki's Stomper):
    - catches every other player alive within `stomp_radius` of that centre, on centres as the
      charge's landing pulse is. An airborne player is skipped, since D4 counts the air as
      neither slowed nor hittable, and crates are not players;
    - stuns for `stomp_stun_seconds` each caught player already slowed by anything, the charge's
      landing included, and leaves that slow running (`status.apply_stun`);
    - slows every other caught player by `stomp_slow_frac` for `stomp_slow_seconds`
      (`status.apply_slow`).
    It deals no damage. A stomper who dies in the wind-up lands nothing. Two pulses on one player
    in one tick (never today: only the hero stomps) take the largest of each number."""
    armed = state.ent_stomp_t > 0
    state.ent_stomp_t.copy_(status.count_down(state.ent_stomp_t, cfg.dt))
    lands = armed & (state.ent_stomp_t <= 0) & state.ent_alive                       # (N,E) stomper

    E = state.ent_kind.shape[1]
    eye = torch.eye(E, dtype=torch.bool, device=state.ent_pos.device).unsqueeze(0)
    gap = geo.safe_norm(state.ent_pos.unsqueeze(2) - state.ent_pos.unsqueeze(1), dim=-1)  # (N,E,E)
    radius = stats.gather_kind(params.stomp_radius, state.ent_kind)
    target = state.ent_alive & ~status.airborne(state)                                # (N,E) victim
    caught = lands.unsqueeze(2) & (gap <= radius.unsqueeze(2)) & ~eye & target.unsqueeze(1)

    # Read before either effect lands, so this pulse's own slow can never become its stun.
    slowed = (state.ent_slow_t > 0).unsqueeze(1)                                      # (N,1,E)
    stun, slow = caught & slowed, caught & ~slowed
    zero = torch.zeros_like(gap)
    stun_s = stats.gather_kind(params.stomp_stun_seconds, state.ent_kind)
    slow_s = stats.gather_kind(params.stomp_slow_seconds, state.ent_kind)
    slow_f = stats.gather_kind(params.stomp_slow_frac, state.ent_kind)
    status.apply_stun(state, stun.any(dim=1), torch.where(stun, stun_s.unsqueeze(2), zero).amax(dim=1))
    status.apply_slow(state, slow.any(dim=1),
                      torch.where(slow, slow_f.unsqueeze(2), zero).amax(dim=1),
                      torch.where(slow, slow_s.unsqueeze(2), zero).amax(dim=1))
    return caught, lands
