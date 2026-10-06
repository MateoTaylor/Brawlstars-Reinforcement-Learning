"""The bot phase's shared layer: `all_bot_intents`, the per-tick entry point env._bot_phase calls,
and the pieces bots/combat_rules.py (how a bot SHOOTS, by kind) and bots/personality.py (how it
MOVES, by personality) share: `BotIntent`, `Targeting`/`targeting()` (the effective aim target,
once per tick), `fire_gate`, the zone and loot (direction, weight) contributions for
`steering.combine`, and small gather/zone helpers.

`targeting()` also resolves the loot-box PSEUDO-TARGET (`cfg.bots_attack_boxes`), which is what
makes the user's rule "rush attacks lootboxes" real. Boxes go through the same bundle, so every
kind's fire rule applies to them unchanged (a cone swinger still needs the box inside its cone).

Loot pulls (user decision, 2026-09-25): the crate and cube pulls are UNIT directions that REPLACE
the personality's steering (bots/personality.movement), so their weights only compete with the
zone terms. They need no enemy target within _LOOT_ENEMY_FAR_TILES (a cube at the feet is
exempt), a clear walk, and loot out of the gas; they are off in RETREAT and HOLD_STILL, never pull
a CAMPER to a crate, and a bot shoots a crate only within _BOX_TARGET_TILES.
"""
from dataclasses import dataclass

import torch

from ..constants import Person
from ..core import geometry as geo
from ..core import stats
from ..core import terrain
from . import perception
from . import steering

# Loot farming (user decision, 2026-09-25). Live bots farm: a lobby's richest bot holds 7+ cubes
# by mid-match. These pulls, with the walk and zone gates below and the cube-first rule in
# bots/personality.movement, are sized to reproduce that (measurements: BRAWL_SIM_DESIGN.md §7).
_BOX_APPROACH_RADIUS = 20.0
_BOX_APPROACH_WEIGHT = 3.0
_BOX_TARGET_TILES = 5.0         # a bot SHOOTS a crate only this close (user decision, 2026-09-25):
                                # live bots break the crate beside them, not one across the map
_LOOT_ENEMY_FAR_TILES = 8.0     # an enemy target farther than this does not stop a loot pull
_CUBE_COLLECT_RADIUS = 12.0
_CUBE_COLLECT_WEIGHT = 4.0
_CUBE_AT_FEET_TILES = 2.0       # a cube this close is grabbed whoever is near (a kill's drop)
_LOOT_ZONE_MARGIN_TILES = 1.0   # loot this close to the gas, or in it, pulls no one
_ZONE_ESCAPE_WEIGHT = 3.0
_ZONE_AVOID_WEIGHT = 2.0
# Bounds on the KITE hold-distance multiplier 1 / aggression (see targeting).
_HOLD_SCALE_MIN = 0.6
_HOLD_SCALE_MAX = 1.4

@dataclass
class BotIntent:
    move_dir: torch.Tensor    # (N,E,2) length <= 1 (EMA-smoothed; the length is a speed throttle)
    fire: torch.Tensor        # (N,E) bool
    # (N,E) bool -- fire this entity's SUPER. All-False: no bot kind has a super
    # (`super_charge_hits` 0). The field keeps env._bot_phase -> _attack_phase plumbed, so a bot
    # super needs a brawlers.yaml block and a combat-rule decision, not new plumbing.
    super_fire: torch.Tensor
    aim_dir: torch.Tensor     # (N,E,2) unit
    aim_point: torch.Tensor   # (N,E,2)
    mode: torch.Tensor        # (N,E) i64 personality.Mode -- diagnostics only


@dataclass
class Targeting:
    """One tick's resolved aim target for every (N,E) entity. `has_enemy`/`enemy_*` describe the
    visible ENTITY target (state.ent_target, sticky, chosen by perception.select_target);
    `pos`/`vel`/`dist`/`los`/`has_target` describe the EFFECTIVE aim target: that enemy, or the
    nearest crate (`is_box`) within min(attack_range, _BOX_TARGET_TILES) when the bot has no
    enemy inside its attack range (never for a CAMPER). Enemy-relative movement steers off
    `has_enemy`/`enemy_pos`, the crate pull is box_contribution's, and fire/aim use the
    effective target."""
    idx: torch.Tensor          # (N,E) i64 entity index, clamped; meaningless where ~has_enemy
    has_enemy: torch.Tensor    # (N,E) bool
    enemy_pos: torch.Tensor    # (N,E,2)
    enemy_vel: torch.Tensor    # (N,E,2)
    pos: torch.Tensor          # (N,E,2) effective aim target
    vel: torch.Tensor          # (N,E,2) zero for a box
    dist: torch.Tensor         # (N,E) to the effective target
    has_target: torch.Tensor   # (N,E) bool -- enemy OR box
    is_box: torch.Tensor       # (N,E) bool
    los: torch.Tensor          # (N,E) bool physical wall LOS to the effective target
    seen_by_other: torch.Tensor  # (N,E) bool -- does ANY other entity see me? (Camper's gate)
    desired_range: torch.Tensor  # (N,E) kind's preferred distance, in tiles, <= fire_reach
    fire_reach: torch.Tensor     # (N,E) tiles: fire_range_fraction (0 read as 1.0) x attack_range


def strafe_sign(n_entities: int, device) -> torch.Tensor:
    """(E,) alternating +-1 by slot -- (-1)**entity_index, a fixed per-slot pattern that
    broadcasts against (N,E,2) directions in steering.strafe without needing its own N axis.
    Used by bots/personality.py's strafing modes to keep bots orbiting a shared target in
    opposite rotational senses rather than bunching up."""
    idx = torch.arange(n_entities, device=device)
    ones = torch.ones(n_entities, device=device)
    return torch.where(idx % 2 == 0, ones, -ones)


def gather_rows(source: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """source: (N,K,...), idx: (N,Q) i64 selecting along the K axis per query row. Returns
    (N,Q,...). Generalizes stats.gather_kind (2D only) to sources with trailing dims -- e.g.
    ent_pos (N,E,2) gathered by ent_target (N,E), or box_pos (N,B,2) by a nearest-box index."""
    idx_exp = idx
    while idx_exp.dim() < source.dim():
        idx_exp = idx_exp.unsqueeze(-1)
    idx_exp = idx_exp.expand(*idx.shape, *source.shape[idx.dim():])
    return torch.gather(source, 1, idx_exp)


def target_info(state):
    """Returns (target_idx_safe, has_target, target_pos, target_vel) -- ent_target clamped to
    a valid (if meaningless where invalid) index, plus its gathered position/velocity. Callers
    gate on has_target before trusting target_pos/target_vel."""
    has_target = state.ent_target >= 0
    target_idx_safe = torch.clamp(state.ent_target, min=0)
    target_pos = gather_rows(state.ent_pos, target_idx_safe)
    target_vel = gather_rows(state.ent_vel, target_idx_safe)
    return target_idx_safe, has_target, target_pos, target_vel


def targeting(state, vis: torch.Tensor, los: torch.Tensor, bank, params, cfg) -> Targeting:
    """Resolves every (N,E) entity's effective aim target. Reads `state.ent_target` (so
    perception.select_target must already have run this tick), `vis` (the (N,E,E) bush-only
    visibility) and `los` (the **(N,E)** wall LOS to each entity's own target,
    bots/perception.target_los).

    The loot-box pseudo-target needs `cfg.bots_attack_boxes` and is never adopted by a CAMPER
    (its whole behavior is to stay hidden); every other personality adopts one, so the hero does
    not get uncontested access to every cube on the map. An enemy target OUTSIDE the bot's own
    attack range does not block the box: the bot could not have fired at it this tick anyway
    (fire_gate's `dist <= attack_range`), so it shoots the crate beside it instead of holding its
    ammo while it walks. `has_enemy`/`enemy_pos` are unchanged, so movement still closes on the
    enemy."""
    idx, has_enemy, enemy_pos, enemy_vel = target_info(state)
    E = state.ent_pos.shape[1]
    device = state.ent_pos.device

    # (N,E): is anyone else currently able to see me? vis[n,i,j] is "i sees j", so this is an
    # any-reduce over the OBSERVER axis with the self-pair masked out (vis[n,i,i] is True for
    # every living entity and would otherwise make this trivially True for everyone).
    not_self = ~torch.eye(E, dtype=torch.bool, device=device).unsqueeze(0)
    seen_by_other = (vis & not_self).any(dim=1)

    attack_range = stats.gather_kind(params.attack_range, state.ent_kind)
    # The KITE hold distance scales by clamp(1 / aggression, 0.6, 1.4): an aggressive kiter holds
    # closer, a timid one farther. `aggression_of` reads 0 as 1.0, so a spec without the key holds
    # exactly `desired_range_fraction * attack_range`.
    hold_scale = torch.clamp(
        1.0 / stats.aggression_of(state.ent_kind, params), _HOLD_SCALE_MIN, _HOLD_SCALE_MAX,
    )
    # ... and never past the bot's own fire reach (user decision, 2026-09-21). The reach is
    # combat_rules.combat's own `range_ok` bound, so the cap and the fire gate cannot disagree
    # about where a bot can shoot. bots/personality.movement passes the same reach to
    # steering.maintain_range as the SEEK edge, because that edge is where an approaching kiter
    # parks.
    fire_fraction = stats.gather_kind(params.fire_range_fraction, state.ent_kind)
    fire_reach = torch.where(fire_fraction > 0, fire_fraction,
                             torch.ones_like(fire_fraction)) * attack_range
    box_idx, box_dist = perception.nearest_alive(state.box_pos, state.box_alive, state.ent_pos)
    box_pos = gather_rows(state.box_pos, box_idx)

    if cfg.bots_attack_boxes:
        # `box_alive` is False for every unused slot, so nearest_alive returns +inf there and the
        # range test rejects it -- no separate "does this env have any boxes left" check needed.
        #
        # A crate is a target only within _BOX_TARGET_TILES, or the bot's own attack range if
        # that is shorter (user decision, 2026-09-25): live bots break the crate beside them.
        enemy_far = has_enemy & (geo.dist(state.ent_pos, enemy_pos) > attack_range)
        crate_reach = torch.clamp(attack_range, max=_BOX_TARGET_TILES)
        is_box = (
            (~has_enemy | enemy_far) & (box_dist <= crate_reach)
            & (state.ent_person != int(Person.CAMPER))
        )
    else:
        is_box = torch.zeros_like(has_enemy)
    is_box3 = is_box.unsqueeze(-1)
    pos = torch.where(is_box3, box_pos, enemy_pos)
    vel = torch.where(is_box3, torch.zeros_like(enemy_vel), enemy_vel)
    has_target = has_enemy | is_box

    # LOS to the effective target: `los` is already the per-entity ray to its own enemy target
    # (bots/perception.target_los), and a box gets its own single (N,E) march.
    #
    # The box march is bounded by `params.attack_ray_tiles` for the same reason
    # combat.melee_hitscan bounds its cone march: a box further away than the shooter's own
    # attack_range is rejected by fire_gate's `dist <= attack_range` term, so a shortened ray
    # giving that box a wrong LOS answer cannot change any fire decision.
    los_enemy = los
    if cfg.bots_attack_boxes:
        los_box = terrain.line_of_sight(bank, state.map_id, state.ent_pos, box_pos, cfg,
                                        max_tiles=params.attack_ray_tiles)
        los_eff = torch.where(is_box, los_box, los_enemy)
    else:
        los_eff = los_enemy

    return Targeting(
        idx=idx, has_enemy=has_enemy, enemy_pos=enemy_pos, enemy_vel=enemy_vel,
        pos=pos, vel=vel, dist=geo.dist(state.ent_pos, pos), has_target=has_target,
        is_box=is_box, los=los_eff, seen_by_other=seen_by_other,
        desired_range=torch.minimum(
            stats.gather_kind(params.desired_range_fraction, state.ent_kind) * attack_range
            * hold_scale,
            fire_reach,
        ),
        fire_reach=fire_reach,
    )


def fire_gate(state, target_dist: torch.Tensor, has_target: torch.Tensor, attack_range: torch.Tensor) -> torch.Tensor:
    """(N,E) bool: alive & ammo>=1 & attack_cd<=0 & react_t<=0 & has_target & dist<=range.
    Decision-period gating is layered on top by all_bot_intents, not here."""
    return (
        state.ent_alive & (state.ent_ammo >= 1.0) & (state.ent_attack_cd <= 0)
        & (state.ent_react_t <= 0) & has_target & (target_dist <= attack_range)
    )


def zone_rect(state):
    """(zone_lo (N,1,2), zone_hi (N,1,2), rect_active (N,1) bool) -- the safe rect, shaped to
    broadcast against per-entity (N,E,...) tensors, plus the degenerate-rect guard every consumer
    needs: until core/zone.init_zone runs, state.zone_lo/zone_hi sit at allocate()'s zero-init,
    a zero-area rect that means "no zone active", not "the whole map is lethal"."""
    zone_lo = state.zone_lo.unsqueeze(1)
    zone_hi = state.zone_hi.unsqueeze(1)
    rect_active = (zone_hi[..., 0] > zone_lo[..., 0]) & (zone_hi[..., 1] > zone_lo[..., 1])
    return zone_lo, zone_hi, rect_active


def zone_clearance(state, cfg) -> torch.Tensor:
    """(N,E) tiles of room left before this entity is standing in the zone -- +inf where no zone
    is active, so every "am I under zone pressure" comparison is False by construction there."""
    zone_lo, zone_hi, rect_active = zone_rect(state)
    clearance = perception.zone_clearance(state.ent_pos, zone_lo, zone_hi)
    if not cfg.zone_enabled:
        return torch.full_like(clearance, float("inf"))
    return torch.where(rect_active, clearance, torch.full_like(clearance, float("inf")))


def zone_contribution(state, cfg):
    """(direction, weight): steering.escape_zone toward the safe rect, weight 3.0 exactly
    where perception.in_zone is True (and the rect is non-degenerate -- see zone_rect), else 0."""
    zero_dir = torch.zeros_like(state.ent_pos)
    zero_w = torch.zeros_like(state.ent_pos[..., 0])
    if not cfg.bots_avoid_zone:
        return zero_dir, zero_w

    zone_lo, zone_hi, rect_active = zone_rect(state)
    direction = steering.escape_zone(state.ent_pos, zone_lo, zone_hi)
    in_zone = perception.in_zone(state.ent_pos, zone_lo, zone_hi) & rect_active
    weight = torch.where(in_zone, torch.full_like(zero_w, _ZONE_ESCAPE_WEIGHT), zero_w)
    return direction, weight


def zone_avoid_contribution(state, cfg):
    """(direction, weight): a PREDICTIVE inward push, active while an entity is still safe but
    within `cfg.bots_zone_avoid_tiles` of the safe rect's edge, ramping linearly from 0 at that
    margin to _ZONE_AVOID_WEIGHT at the boundary itself.

    This is the "bots avoid the green zone and do not walk into it" rule (user's rule):
    escape_zone is zero everywhere INSIDE the rect, so `zone_contribution` alone only reacts once
    a bot is already taking damage. Aimed at the rect's center, which is inward and well-defined
    even near a corner, where "away from the nearest edge" is ambiguous."""
    zero_dir = torch.zeros_like(state.ent_pos)
    zero_w = torch.zeros_like(state.ent_pos[..., 0])
    if not cfg.bots_avoid_zone or cfg.bots_zone_avoid_tiles <= 0:
        return zero_dir, zero_w

    zone_lo, zone_hi, rect_active = zone_rect(state)
    center = (zone_lo + zone_hi) / 2.0  # (N,1,2), broadcasts against (N,E,2)
    direction = steering.seek(state.ent_pos, center)

    margin = cfg.bots_zone_avoid_tiles
    clearance = perception.zone_clearance(state.ent_pos, zone_lo, zone_hi)
    ramp = torch.clamp(1.0 - clearance / margin, min=0.0, max=1.0)
    inside = ~perception.in_zone(state.ent_pos, zone_lo, zone_hi)
    weight = torch.where(rect_active & inside, ramp * _ZONE_AVOID_WEIGHT, zero_w)
    return direction, weight


def _walk_clear(state, bank, cfg, target_pos: torch.Tensor, max_tiles: float) -> torch.Tensor:
    """(N,E) bool: the straight line from each entity to `target_pos` crosses no unit-blocking
    tile (wall or water). The loot pulls need it because bot movement has no pathfinding:
    terrain.resolve_move slides along a wall one axis at a time, so a pull aimed through a wall
    pins the bot against it.

    `max_tiles` is march's ray budget. Both callers pass their own pull radius and zero every
    pull farther than that, so the shortened ray never decides an answer that is used."""
    delta = target_pos - state.ent_pos
    hit, _, _ = terrain.march(bank.blocks_unit, state.map_id, state.ent_pos, delta,
                              geo.safe_norm(delta, dim=-1), cfg, max_tiles=max_tiles)
    return ~hit


def _loot_is_safe(state, cfg, target_pos: torch.Tensor) -> torch.Tensor:
    """(N,E) bool: `target_pos` sits at least _LOOT_ZONE_MARGIN_TILES inside the safe rect, or no
    zone is active. A loot pull is the only non-zone term while it is active, and it outweighs
    zone avoidance (2.0) up close, so without this a bot would walk into the gas after a crate
    or cube and stall there."""
    zone_lo, zone_hi, rect_active = zone_rect(state)
    room = perception.zone_clearance(target_pos, zone_lo, zone_hi)
    if not cfg.zone_enabled:
        return torch.ones_like(room, dtype=torch.bool)
    return ~rect_active | (room >= _LOOT_ZONE_MARGIN_TILES)


def box_contribution(state, bank, cfg):
    """(direction, weight): the UNIT direction to the nearest alive box, weight
    _BOX_APPROACH_WEIGHT, when cfg.bots_break_boxes and the box is within _BOX_APPROACH_RADIUS
    tiles, the bot is not a CAMPER, it has no enemy target or that target is more than
    _LOOT_ENEMY_FAR_TILES away, the walk to the box is clear (`_walk_clear`) and the box is out
    of the gas (`_loot_is_safe`).

    Campers are out because `targeting` never lets one shoot a crate: pulled anyway, a camper
    looking for a bush would park beside a crate it cannot break.

    bots/personality.movement drops this pull wherever cube_contribution is pulling (a cube on
    the ground comes first) and silences the personality's own steering while either pull is
    active, so the bot walks straight to the loot and nothing sums against it."""
    zero_dir = torch.zeros_like(state.ent_pos)
    zero_w = torch.zeros_like(state.ent_pos[..., 0])
    if not cfg.bots_break_boxes:
        return zero_dir, zero_w

    idx, dist = perception.nearest_alive(state.box_pos, state.box_alive, state.ent_pos)
    box_pos = gather_rows(state.box_pos, idx)
    direction = geo.normalize(steering.seek(state.ent_pos, box_pos))
    _, has_enemy, enemy_pos, _ = target_info(state)
    free = ~has_enemy | (geo.dist(state.ent_pos, enemy_pos) > _LOOT_ENEMY_FAR_TILES)
    gate = (
        free & (dist <= _BOX_APPROACH_RADIUS)
        & (state.ent_person != int(Person.CAMPER))
        & _walk_clear(state, bank, cfg, box_pos, _BOX_APPROACH_RADIUS)
        & _loot_is_safe(state, cfg, box_pos)
    )
    weight = torch.where(gate, torch.full_like(zero_w, _BOX_APPROACH_WEIGHT), zero_w)
    return direction, weight


def cube_contribution(state, bank, cfg):
    """(direction, weight): the UNIT direction to the nearest alive pickup, weight
    _CUBE_COLLECT_WEIGHT, when cfg.bots_collect_cubes and the pickup is within
    _CUBE_COLLECT_RADIUS tiles, the walk to it is clear, it is out of the gas, and either no
    enemy target is within _LOOT_ENEMY_FAR_TILES or the cube is within _CUBE_AT_FEET_TILES.

    The at-feet exception is how the cubes a kill drops get collected: they land where the
    fight was, and the next enemy is usually in sight. Farther cubes wait for the fight to end,
    since a pull toward them would walk an engaged bot away from its target.
    bots/personality.movement also drops this pull in RETREAT outright."""
    zero_dir = torch.zeros_like(state.ent_pos)
    zero_w = torch.zeros_like(state.ent_pos[..., 0])
    if not cfg.bots_collect_cubes:
        return zero_dir, zero_w

    idx, dist = perception.nearest_alive(state.pku_pos, state.pku_alive, state.ent_pos)
    pku_pos = gather_rows(state.pku_pos, idx)
    direction = geo.normalize(steering.seek(state.ent_pos, pku_pos))
    _, has_enemy, enemy_pos, _ = target_info(state)
    free = ~has_enemy | (geo.dist(state.ent_pos, enemy_pos) > _LOOT_ENEMY_FAR_TILES)
    gate = (
        (dist <= _CUBE_COLLECT_RADIUS)
        & (free | (dist <= _CUBE_AT_FEET_TILES))
        & _walk_clear(state, bank, cfg, pku_pos, _CUBE_COLLECT_RADIUS)
        & _loot_is_safe(state, cfg, pku_pos)
    )
    weight = torch.where(gate, torch.full_like(zero_w, _CUBE_COLLECT_WEIGHT), zero_w)
    return direction, weight


def all_bot_intents(state, vis, bank, params, cfg, gen) -> BotIntent:
    """MUTATES: ent_target (via perception.select_target), ent_move_smooth, and -- via
    bots/personality.movement -- ent_wander_dir/ent_wander_t/ent_hunt_seen/ent_hunt_t. Returns a
    freshly selected/smoothed BotIntent.

    1. perception.select_target, then the once-per-tick shared queries: `target_los` (N,E) and
       `targeting` (which resolves the loot-box pseudo-target on top of it).
    2. FIRE/AIM: one call to bots/combat_rules.combat, which resolves every kind's rule from
       per-kind params in a single pass over all (N,E) entities.
    3. MOVEMENT: bots/personality.movement computes move_dir once for all (N,E), selecting
       per-entity behavior off ent_person.
    4. Personality fire veto: CAMPER holds fire until something can actually see it, unless
       its kind's `aggression` is 1.25 or more.
    5. Decision period gates FIRE only (discrete: this tick must be entity `e`'s turn to
       reconsider firing -- `(step_count + e) % decision_period == 0`, staggering entities so
       they don't all decide in lockstep); reaction delay low-passes MOVEMENT only (continuous
       EMA into the persistent ent_move_smooth field via rate = clamp(dt/reaction_delay, 0, 1)).
       They are two different per-kind difficulty knobs (`decision_period_ticks` vs
       `reaction_delay`), so they get two different treatments rather than one gating both.
       ent_move_smooth (not the raw per-tick selection) is the returned move_dir, and its length
       is movement.apply_movement's speed throttle, so a bot that stops decelerates over the
       reaction delay instead of coasting at full speed (user decision, 2026-09-25).
    6. Zero fire/move_dir/aim_dir/aim_point for entity 0 (the hero) and all dead entities,
       applied to the OUTPUT only, after decision-period gating and move-smoothing.
       ent_move_smooth's underlying STATE for entity 0 / dead entities is left un-zeroed (it
       keeps tracking whatever garbage intent got selected for it, which is harmless):
       movement.apply_movement already multiplicatively masks out `~alive` entities via its own
       `active` gate, and the hero's movement comes from hero.decode_action, never from
       ent_move_smooth at all.
    """
    # Lazy, not module-level: both do `from . import policy as shared` to reach BotIntent /
    # Targeting / fire_gate / *_contribution / strafe_sign above, so an eager import here would be
    # circular. By the time this function is CALLED, this module has finished initializing (it is
    # what they imported first), so the import resolves cleanly.
    from . import combat_rules, personality

    # Bots act on a SIGHT-LIMITED view of `vis`, never on `vis` itself -- see
    # perception.bot_visibility for why. `vis` stays unclipped for the hero's observation, which
    # env.py builds separately.
    bot_vis = perception.bot_visibility(state, vis, cfg)
    perception.select_target(state, bot_vis, params, cfg)

    E = state.ent_pos.shape[1]
    device = state.ent_pos.device

    # (N,E): LOS from each entity to its own target only (perception.target_los).
    los = perception.target_los(state, bank, cfg)
    tgt = targeting(state, bot_vis, los, bank, params, cfg)

    fire, aim_dir, aim_point = combat_rules.combat(state, tgt, bank, params, cfg, gen)

    move_dir, mode = personality.movement(state, tgt, bank, params, cfg, gen)
    # The CAMPER veto lifts at aggression >= 1.25 (0 read as 1.0 by the helper).
    aggression = stats.aggression_of(state.ent_kind, params)
    fire = fire & personality.fire_allowed(state, tgt, aggression, cfg)

    # --- decision period: discrete fire-reconsideration gate, staggered by entity slot ---
    decision_period = torch.clamp(stats.gather_kind(params.decision_period, state.ent_kind), min=1)
    entity_idx = torch.arange(E, device=device, dtype=torch.int64).view(1, E)
    step_count = state.step_count.unsqueeze(-1).to(torch.int64)
    decision_tick = ((step_count + entity_idx) % decision_period) == 0
    fire = fire & decision_tick

    # --- reaction delay: continuous EMA low-pass of movement into ent_move_smooth ---
    reaction_delay = stats.gather_kind(params.reaction_delay, state.ent_kind)
    rate = torch.clamp(cfg.dt / torch.clamp(reaction_delay, min=1e-6), 0.0, 1.0)
    new_smooth = state.ent_move_smooth + rate.unsqueeze(-1) * (move_dir - state.ent_move_smooth)
    state.ent_move_smooth.copy_(new_smooth)
    move_dir = state.ent_move_smooth

    # --- zero entity 0 (hero) and all dead entities ---
    is_hero_slot = (entity_idx == 0)  # (1,E), broadcasts against (N,E)
    keep = state.ent_alive & ~is_hero_slot
    keep3 = keep.unsqueeze(-1)
    move_dir = torch.where(keep3, move_dir, torch.zeros_like(move_dir))
    fire = fire & keep
    aim_dir = torch.where(keep3, aim_dir, torch.zeros_like(aim_dir))
    aim_point = torch.where(keep3, aim_point, torch.zeros_like(aim_point))

    return BotIntent(move_dir=move_dir, fire=fire, super_fire=torch.zeros_like(fire),
                     aim_dir=aim_dir, aim_point=aim_point, mode=mode)
