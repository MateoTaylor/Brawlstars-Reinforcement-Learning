"""BotIntent, the building blocks every archetype (Steps 17-20) shares, and the dispatcher
that combines their outputs by kind (`all_bot_intents`, Step 21). See BRAWL_SIM_BUILD_PLAN.md
Steps 17-20's shared preamble, whose "Shared rules" paragraph names this file as home for both.

**Step 41 split combat from movement.** An archetype (`Kind`) now decides only how an entity
SHOOTS -- `combat()` in bots/combat_rules.py, returning fire/aim only -- and a personality
(`Person`, drawn independently per entity per reset) decides how it MOVES, in
bots/personality.py. Before this split, movement flavor was baked into each archetype's
`policy()`, which had two consequences worth recording:
  - Every archetype with no visible target converged on the SAME place (melee patrolled to the
    zone rect's center; artillery hugged cover; sniper/rifle held range on nobody), so a lobby of
    bots reliably collapsed into one scrum. An agent could farm that by sitting in a bush and
    letting the scrum resolve itself -- the exact degenerate policy this rewrite exists to break.
  - Movement is also where all the expensive work lives (tile scans, LOS probes), and computing
    it per-archetype meant paying for it 4x and throwing 3 away. It is now computed ONCE for all
    (N,E) entities regardless of archetype or personality.

**Step E1 then collapsed the four archetype modules into one.** Fire/aim used to be computed by
four separate `combat()` functions -- one per Kind, each run over all (N,E) entities with three of
the four results discarded -- and is now a single data-driven rule reading five per-kind fields off
SimParams (bots/combat_rules.py). Nothing about WHEN each brawler shoots changed; what changed is
that adding a sixth costs a configs/brawlers.yaml block instead of a fifth Python module and three
edits in this file.

Shared rules implemented here:
  - `BotIntent`: the (N,E)-batched return value of `all_bot_intents`.
  - `Targeting` / `targeting()`: one bundle carrying the effective aim target, its distance, and
    physical LOS to it, computed ONCE per tick (`raw_los` used to be computed twice -- once by
    the sniper rule and again by the rifle rule -- for the same answer). Since Step A2 that LOS
    is `perception.target_los`, an (N,E) per-entity ray, rather than the (N,E,E) all-pairs matrix
    this file used to build and then read a single column out of. It also resolves the loot-box
    PSEUDO-TARGET: with `cfg.bots_attack_boxes`, an entity with no visible enemy adopts the
    nearest in-range box as its aim target, which is what makes the user-specified "rush attacks
    lootboxes" real. Feeding it through the same bundle rather than special-casing it means every
    archetype's existing fire logic applies to boxes unchanged -- melee still needs the box inside
    its swing cone, artillery still needs no LOS to it, rifle still holds fire past
    `fire_range_fraction` of its range.
  - `fire_gate`: `alive & ammo>=1 & attack_cd<=0 & react_t<=0 & has_target & dist<=attack_range`.
    bots/combat_rules ANDs each kind's own conditions on top (LOS, range fraction, swing cone,
    lateral hold). AIM is not here: after Step E1 there is exactly one caller, and its three aim
    models share a `geo.lead_target` solve that splitting them across two modules would run
    twice.
  - `zone_contribution` / `zone_avoid_contribution` / `box_contribution` / `cube_contribution`:
    (direction, weight) pairs ready to hand straight to `steering.combine`. Universal -- they
    apply to every archetype AND every personality.
  - `Targeting.desired_range`: each archetype's preferred engagement distance, consumed only by
    the movement layer (bots/personality.py's RANGE mode). Was `RANGE_FRACTION_BY_KIND`, a Python
    tuple indexed by Kind carrying an `assert len(...) == N_KINDS`; Step E1 made it the per-kind
    `desired_range_fraction` param, so a sixth brawler no longer trips that assert.

Two decisions filled in beyond the plan's literal text:
  - `bots_avoid_zone` gates `zone_contribution` the same way `bots_break_boxes` /
    `bots_collect_cubes` gate their contributions. The Steps 17-20 preamble's "zone escape
    weight 3.0 when in_zone" line doesn't repeat the toggle name the way the box/cube lines do,
    but D14 lists fleeing the zone as toggleable exactly like the other two, and EnvConfig
    already carries `bots_avoid_zone` for it (Step 3) -- the omission reads as brevity, not an
    intentional exception.
  - `zone_contribution` treats a degenerate zone rect (`zone_hi <= zone_lo` on either axis) as
    "no zone active" rather than trusting `state.zone_lo`/`zone_hi` outright. Before
    `core/zone.py`'s `init_zone` (Step 23) ever runs, those fields sit at `allocate()`'s
    zero-init -- (0,0)-(0,0) -- which is a zero-area rect, not the full-map "everything is
    safe" rect a real reset will produce. Guarding on rect area keeps every archetype's zone
    steering well-defined today and costs nothing once Step 23 lands (a freshly-reset env's
    zone_lo/zone_hi legitimately spans the whole map, which independently yields `in_zone`
    False everywhere too).
  - Box/cube approach weight is fixed at 1.0, matching the personality-neutral baseline `seek`
    weight already used elsewhere (e.g. Step 19 melee's `seek` 1.0) -- the plan gives an exact
    weight for zone escape (3.0) but not for these two.
  - `strafe_sign` moved here from the melee archetype once Step 20 (rifle) needed the identical
    alternating-by-slot pattern -- promoted on second use, same as `projectiles.alloc_slots`/
    `_set_scalar`/`_set_vec2` were promoted for `combat.py` to reuse in Step 14.
"""
from dataclasses import dataclass

import torch

from ..constants import Person
from ..core import geometry as geo
from ..core import stats
from ..core import terrain
from . import perception
from . import steering

_BOX_APPROACH_RADIUS = 10.0
_BOX_APPROACH_WEIGHT = 1.0
_CUBE_COLLECT_RADIUS = 8.0
_CUBE_COLLECT_ENEMY_CLEARANCE = 6.0
_CUBE_COLLECT_WEIGHT = 1.0
_ZONE_ESCAPE_WEIGHT = 3.0
_ZONE_AVOID_WEIGHT = 2.0

@dataclass
class BotIntent:
    move_dir: torch.Tensor    # (N,E,2) unnormalized
    fire: torch.Tensor        # (N,E) bool
    # (N,E) bool -- fire this entity's SUPER. All-False today: no bot kind sets
    # `super_charge_hits`, so `hero.super_ready` is False for every one of them. The field exists
    # so that giving a bot a super is a brawlers.yaml block plus a decision in its combat rule,
    # rather than re-plumbing env._bot_phase -> _attack_phase (bot_overhaul.md D2).
    super_fire: torch.Tensor
    aim_dir: torch.Tensor     # (N,E,2) unit
    aim_point: torch.Tensor   # (N,E,2)
    mode: torch.Tensor        # (N,E) i64 personality.Mode -- diagnostics/rendering only


@dataclass
class Targeting:
    """One tick's resolved aim target for every (N,E) entity. `has_enemy`/`enemy_*` describe the
    visible ENTITY target (state.ent_target, sticky, chosen by perception.select_target);
    `pos`/`vel`/`dist`/`los`/`has_target` describe the EFFECTIVE aim target, which is that enemy
    when there is one and otherwise the nearest in-range loot box (`is_box`). Movement steers off
    `has_enemy`/`enemy_pos` -- a bot does not chase a box the way it chases a player, it just
    shoots one it happens to be standing near -- while fire/aim use the effective target."""
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
    desired_range: torch.Tensor  # (N,E) archetype's preferred distance, in tiles


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
    a valid (if meaningless where invalid) index, plus its gathered position/velocity. Every
    archetype needs this; callers gate on has_target before trusting target_pos/target_vel."""
    has_target = state.ent_target >= 0
    target_idx_safe = torch.clamp(state.ent_target, min=0)
    target_pos = gather_rows(state.ent_pos, target_idx_safe)
    target_vel = gather_rows(state.ent_vel, target_idx_safe)
    return target_idx_safe, has_target, target_pos, target_vel


def targeting(state, vis: torch.Tensor, los: torch.Tensor, bank, params, cfg) -> Targeting:
    """Resolves every (N,E) entity's effective aim target. Reads `state.ent_target` -- so
    perception.select_target must already have run this tick -- plus `vis`, the dense (N,E,E)
    bush-only visibility matrix, and `los`, an **(N,E)** per-entity physical LOS to that entity's
    own target (bots/perception.target_los; walls-only, see bots/perception.py's module docstring
    on why visibility and LOS are different questions).

    **`los` used to be the full (N,E,E) `raw_los` matrix, of which this function read exactly one
    column** (`torch.gather(los, 2, idx)` -- LOS to the entity's own target) and discarded the
    rest. Step A2 pushed that gather up into `target_los`, which computes only the (N,E) rays that
    were ever going to be read.

    The loot-box pseudo-target is gated on `cfg.bots_attack_boxes` AND on the personality: a
    CAMPER shoots nothing it isn't already committed to (its whole behavior is to stay hidden),
    so it never adopts a box. Every other personality does, which is what stops the hero from
    having uncontested access to every cube on the map -- previously bots only ever broke boxes by
    accident, with stray shots aimed at each other, since fire_gate has always required an ENTITY
    target."""
    idx, has_enemy, enemy_pos, enemy_vel = target_info(state)
    E = state.ent_pos.shape[1]
    device = state.ent_pos.device

    # (N,E): is anyone else currently able to see me? vis[n,i,j] is "i sees j", so this is an
    # any-reduce over the OBSERVER axis with the self-pair masked out (vis[n,i,i] is True for
    # every living entity and would otherwise make this trivially True for everyone).
    not_self = ~torch.eye(E, dtype=torch.bool, device=device).unsqueeze(0)
    seen_by_other = (vis & not_self).any(dim=1)

    attack_range = stats.gather_kind(params.attack_range, state.ent_kind)
    box_idx, box_dist = perception.nearest_alive(state.box_pos, state.box_alive, state.ent_pos)
    box_pos = gather_rows(state.box_pos, box_idx)

    if cfg.bots_attack_boxes:
        # `box_alive` is False for every unused slot, so nearest_alive returns +inf there and the
        # range test rejects it -- no separate "does this env have any boxes left" check needed.
        is_box = (
            ~has_enemy & (box_dist <= attack_range)
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
    # combat.melee_hitscan bounds its cone march (Step A1): a box further away than the shooter's
    # own attack_range is rejected by fire_gate's `dist <= attack_range` term, so a shortened ray
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
        desired_range=stats.gather_kind(params.desired_range_fraction, state.ent_kind)
        * attack_range,
    )


def fire_gate(state, target_dist: torch.Tensor, has_target: torch.Tensor, attack_range: torch.Tensor) -> torch.Tensor:
    """(N,E) bool: alive & ammo>=1 & attack_cd<=0 & react_t<=0 & has_target & dist<=range.
    Decision-period gating (Step 21) is layered on top by the dispatcher, not here."""
    return (
        state.ent_alive & (state.ent_ammo >= 1.0) & (state.ent_attack_cd <= 0)
        & (state.ent_react_t <= 0) & has_target & (target_dist <= attack_range)
    )


def zone_rect(state):
    """(zone_lo (N,1,2), zone_hi (N,1,2), rect_active (N,1) bool) -- the safe rect, shaped to
    broadcast against per-entity (N,E,...) tensors, plus the degenerate-rect guard every consumer
    needs (see module docstring on why trusting state.zone_lo/zone_hi outright is wrong before
    core/zone.init_zone has run)."""
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
    where perception.in_zone is True (and the rect is non-degenerate -- see module
    docstring), else 0."""
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

    This is the "bots avoid the green zone and do not walk into it" rule. `zone_contribution` on
    its own cannot express it: escape_zone is identically zero everywhere INSIDE the rect, so it
    only ever reacts once a bot is already taking damage. The two compose -- avoid keeps bots off
    the edge, escape drags them back if they end up over it anyway (which the zone SHRINKING onto
    a stationary bot still does, no matter how well it steers).

    Directed at the rect's center rather than straight away from the nearest edge: both are
    inward, and the center is always a valid destination even for an entity near a corner, where
    "away from the nearest edge" is ambiguous."""
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


def box_contribution(state, cfg):
    """(direction, weight): steering.seek the nearest alive box when cfg.bots_break_boxes, no
    visible enemy (ent_target < 0), and it's within 10 tiles."""
    zero_dir = torch.zeros_like(state.ent_pos)
    zero_w = torch.zeros_like(state.ent_pos[..., 0])
    if not cfg.bots_break_boxes:
        return zero_dir, zero_w

    idx, dist = perception.nearest_alive(state.box_pos, state.box_alive, state.ent_pos)
    box_pos = gather_rows(state.box_pos, idx)
    direction = steering.seek(state.ent_pos, box_pos)
    no_enemy = state.ent_target < 0
    gate = no_enemy & (dist <= _BOX_APPROACH_RADIUS)
    weight = torch.where(gate, torch.full_like(zero_w, _BOX_APPROACH_WEIGHT), zero_w)
    return direction, weight


def _nearest_other_alive_dist(state) -> torch.Tensor:
    """(N,E) distance to the nearest OTHER alive entity (self excluded), +inf if none."""
    E = state.ent_pos.shape[1]
    diff = state.ent_pos.unsqueeze(2) - state.ent_pos.unsqueeze(1)
    dist = geo.safe_norm(diff, dim=-1)
    not_self = ~torch.eye(E, dtype=torch.bool, device=state.ent_pos.device).unsqueeze(0)
    candidates = not_self & state.ent_alive.unsqueeze(1)
    dist_masked = torch.where(candidates, dist, torch.full_like(dist, float("inf")))
    return dist_masked.min(dim=-1).values


def cube_contribution(state, cfg):
    """(direction, weight): steering.seek the nearest alive pickup when
    cfg.bots_collect_cubes, it's within 8 tiles, and no (any, not just visible) enemy is
    within 6 tiles."""
    zero_dir = torch.zeros_like(state.ent_pos)
    zero_w = torch.zeros_like(state.ent_pos[..., 0])
    if not cfg.bots_collect_cubes:
        return zero_dir, zero_w

    idx, dist = perception.nearest_alive(state.pku_pos, state.pku_alive, state.ent_pos)
    pku_pos = gather_rows(state.pku_pos, idx)
    direction = steering.seek(state.ent_pos, pku_pos)
    enemy_dist = _nearest_other_alive_dist(state)
    gate = (dist <= _CUBE_COLLECT_RADIUS) & (enemy_dist > _CUBE_COLLECT_ENEMY_CLEARANCE)
    weight = torch.where(gate, torch.full_like(zero_w, _CUBE_COLLECT_WEIGHT), zero_w)
    return direction, weight


def all_bot_intents(state, vis, bank, params, cfg, gen) -> BotIntent:
    """MUTATES: ent_target (via perception.select_target), ent_move_smooth, and -- via
    bots/personality.movement -- ent_wander_dir/ent_wander_t/ent_hunt_seen/ent_hunt_t. Returns a
    freshly selected/smoothed BotIntent -- see BRAWL_SIM_BUILD_PLAN.md Step 21, as amended by
    Step 41's combat/movement split (module docstring).

    1. perception.select_target, then the once-per-tick shared queries: `target_los` (N,E) and
       `targeting` (which resolves the loot-box pseudo-target on top of it).
    2. FIRE/AIM: one call to bots/combat_rules.combat, which resolves every kind's rule from
       per-kind params in a single pass over all (N,E) entities (Step E1 -- this used to be four
       `combat()` calls with three of the four results thrown away).
    3. MOVEMENT: bots/personality.movement computes move_dir once for all (N,E), selecting
       per-entity behavior off ent_person. Nothing archetype-specific happens here anymore.
    4. Personality fire veto: CAMPER holds fire until its target can actually see it.
    5. Decision period gates FIRE only (discrete: this tick must be entity `e`'s turn to
       reconsider firing -- `(step_count + e) % decision_period == 0`, staggering entities so
       they don't all decide in lockstep); reaction delay low-passes MOVEMENT only (continuous
       EMA into the persistent ent_move_smooth field via rate = clamp(dt/reaction_delay, 0, 1)
       -- these are the two different per-archetype-difficulty mechanisms D19 lists
       (`decision_period_ticks` vs `reaction_delay`), so they get two different treatments
       rather than one gating both. ent_move_smooth (not the raw per-tick selection) is the
       actual returned move_dir, per the plan text.
    6. Zero fire/move_dir/aim_dir/aim_point for entity 0 (the hero) and all dead entities,
       applied to the OUTPUT only, after decision-period gating and move-smoothing -- matching
       the plan's own step ordering. ent_move_smooth's underlying STATE for entity 0 / dead
       entities is left un-zeroed (it keeps tracking whatever garbage intent got selected for it
       -- see bots/combat_rules.py's docstring on why that garbage is harmless):
       movement.apply_movement already multiplicatively masks out `~alive` entities via its own
       `active` gate, and the hero's movement comes from decode_action (Step 11), never from
       ent_move_smooth at all.
    """
    # Lazy, not module-level: both do `from . import policy as shared` to reach BotIntent /
    # Targeting / fire_gate / *_contribution / strafe_sign above, so an eager import here would be
    # circular. By the time this function is CALLED, this module has finished initializing (it is
    # what they imported first), so the import resolves cleanly.
    from . import combat_rules, personality

    # Bots act on a SIGHT-LIMITED view of `vis`, never on `vis` itself -- see
    # perception.bot_visibility for the measurements that made this necessary. `vis` stays
    # unclipped for the hero's observation, which env.py builds separately.
    bot_vis = perception.bot_visibility(state, vis, cfg)
    perception.select_target(state, bot_vis, cfg)

    E = state.ent_pos.shape[1]
    device = state.ent_pos.device

    # (N,E) rays to each entity's own target -- NOT the (N,E,E) raw_los matrix this used to build
    # and then read one column of. See perception.target_los (Step A2).
    los = perception.target_los(state, bank, cfg)
    tgt = targeting(state, bot_vis, los, bank, params, cfg)

    fire, aim_dir, aim_point = combat_rules.combat(state, tgt, bank, params, cfg, gen)

    move_dir, mode = personality.movement(state, tgt, bank, params, cfg, gen)
    fire = fire & personality.fire_allowed(state, tgt, cfg)

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
