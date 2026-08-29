"""Bot targeting and the observation's visibility annotations. See BRAWL_SIM_BUILD_PLAN.md
Step 15 / Notice 4: bush-hiding is the only thing that ever hides an entity -- terrain never
blocks sight, since the camera is fixed bird's-eye. visibility() (bush-only) drives BOTH bot
targeting and the hero's revealed_* observation flags; raw_los() (walls-only, via
terrain.line_of_sight) is a separate, purely physical query used for fire-gating and
obs["visibility"]["los"], never for targeting.

`bush_scan` (the tile search three of the five bot personalities need, Step 41) lives here
rather than in the sniper archetype, where its loop-per-probe ancestor `_nearest_bush` lived:
once Camper/Hunter/Trapper all needed it, it stopped being one archetype's private helper. It's
also the single most expensive thing the whole bot phase does, so it is deliberately structured
to be called ONCE per tick for all (N,E) entities and shared -- see its own docstring.
"""
from dataclasses import dataclass

import torch

from ..core import geometry as geo
from ..core import terrain

_EPS = 1e-6
_INF = float("inf")


def in_bush(state, bank) -> torch.Tensor:
    """(N,E) bool -- is this entity standing on a BUSH tile."""
    ix, iy = terrain.to_tile(state.ent_pos)
    map_id = terrain._broadcast_map_id(state.map_id, ix.shape)
    h, w = bank.is_bush.shape[1:]
    ix = torch.clamp(ix, 0, w - 1)
    iy = torch.clamp(iy, 0, h - 1)
    return bank.is_bush[map_id, iy, ix]


def visibility(state, bank, params, cfg) -> torch.Tensor:
    """(N,E,E) bool. vis[n,i,j] = i sees j. True iff i alive AND j alive AND (j not in bush OR
    dist <= bush_reveal_radius OR j.reveal_t > 0). No terrain lookup. vis[n,i,i] is True
    whenever i is alive (dist(i,i)=0 always satisfies the reveal-radius clause). Dead
    observers see nothing (alive_i gates every column of their row to False)."""
    bush_j = in_bush(state, bank).unsqueeze(1)  # (N,1,E)
    diff = state.ent_pos.unsqueeze(2) - state.ent_pos.unsqueeze(1)  # (N,E,E,2): i - j
    dist = geo.safe_norm(diff, dim=-1)  # (N,E,E)
    bush_reveal_radius = params.bush_reveal_radius.view(-1, 1, 1)
    reveal_t_j = state.ent_reveal_t.unsqueeze(1)  # (N,1,E)

    visible_despite_bush = (dist <= bush_reveal_radius) | (reveal_t_j > 0)
    bush_ok = (~bush_j) | visible_despite_bush

    alive_i = state.ent_alive.unsqueeze(2)
    alive_j = state.ent_alive.unsqueeze(1)
    return alive_i & alive_j & bush_ok


def bot_visibility(state, vis: torch.Tensor, cfg) -> torch.Tensor:
    """(N,E,E) bool -- `vis` narrowed to what a BOT is allowed to act on: the same bush rules, AND
    within `cfg.bots_sight_tiles`.

    **`visibility()` has no range limit at all**, by design: it answers "is this entity concealed",
    and concealment is a property of bushes, not of distance (the camera is a fixed bird's-eye
    view, so the hero's observation legitimately covers the whole map). Feeding that same matrix
    straight into bot targeting, which is what happened before Step 41, gave every bot perfect
    sight across a 60x60 map. Measured consequences on `bushy`:
      - 84.3% of live bot-ticks had an enemy locked, at a mean range of 22.7 tiles -- roughly three
        times the longest weapon in the game (8.67).
      - Bots were therefore always in an engage behavior. The specification's "when no enemies are
        visible, bots explore the map" fired on 7.7% of ticks and HUNTER's whole reason for
        existing fired on 3.6%. Bush-searching bots could not search, because they always had
        someone to shoot at instead.
    A sight limit is what makes every "nothing visible" branch in bots/personality.py reachable,
    and it is the difference between bots that converge on each other from across the map (which is
    exactly the scrum an agent learns to sit out) and bots that have to go looking.

    Deliberately NOT folded into `visibility()`: that function also produces the hero's
    `revealed_*` observation flags (core/observation.py), and clipping those to a bot's sight range
    would silently change the agent's observation contract while pretending to be a bot-AI change.
    """
    if cfg.bots_sight_tiles <= 0:
        return vis
    diff = state.ent_pos.unsqueeze(2) - state.ent_pos.unsqueeze(1)  # (N,E,E)
    return vis & (geo.safe_norm(diff, dim=-1) <= cfg.bots_sight_tiles)


def raw_los(state, bank, cfg) -> torch.Tensor:
    """(N,E,E) bool. Physical wall LOS only (terrain.line_of_sight via bank.blocks_proj),
    independent of bush -- NOT used for targeting, see module docstring.

    **The bot phase does not use this; `target_los` below does.** This full pairwise matrix is
    built once per DECISION for `obs["visibility"]["los"]`, which genuinely is an all-pairs,
    whole-map question. It used to be built once per SUB-TICK by bots/policy.all_bot_intents as
    well, where 90% of it was discarded -- see `target_los`."""
    E = state.ent_pos.shape[1]
    origin_b = state.ent_pos.unsqueeze(2).expand(-1, -1, E, -1)
    target_b = state.ent_pos.unsqueeze(1).expand(-1, E, -1, -1)
    return terrain.line_of_sight(bank, state.map_id, origin_b, target_b, cfg)


def target_los(state, bank, cfg) -> torch.Tensor:
    """(N,E) bool: physical wall LOS from each entity to ITS OWN CURRENT TARGET
    (`state.ent_target`). Requires `select_target` to have already run this tick.

    **This replaces an (N,E,E) `raw_los` the bot phase only ever read one column of.**
    bots/policy.targeting's single consumer was `torch.gather(los, 2, idx)` -- LOS to the entity's
    own target -- so the other E-1 columns per row were computed and thrown away. Measured
    1.391 -> 0.404 ms/tick at n_envs=1024, E=10 (bot_overhaul.md Step A2).

    **Rows with no target return a meaningless answer, and every consumer already drops them.**
    `ent_target` is -1 for "no target" and is clamped to 0 here, so such a row measures LOS to
    entity 0 at an arbitrary distance. bots/policy.targeting resolves those rows to `has_enemy =
    False`, and `fire_gate` requires `has_target`, so the value never reaches a fire decision --
    the same "compute for everyone, select later" discipline the archetype combat functions use.

    **The ray budget is `cfg.bots_sight_tiles`, and that bound is only valid because of how
    `select_target` works.** all_bot_intents passes it `bot_visibility(...)`, and BOTH of its
    paths are gated on that matrix -- the nearest-visible re-pick AND the stickiness check -- so a
    live `ent_target` is guaranteed to be within sight range THIS TICK, not merely to have been
    when it was acquired. A target that walks out of range is dropped, not held. `<= 0` disables
    the sight limit entirely, so the budget goes back to the full `cfg.ray_steps` with it.
    """
    idx = torch.clamp(state.ent_target, min=0)
    target_pos = torch.gather(state.ent_pos, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
    max_tiles = cfg.bots_sight_tiles if cfg.bots_sight_tiles > 0 else None
    return terrain.line_of_sight(bank, state.map_id, state.ent_pos, target_pos, cfg,
                                 max_tiles=max_tiles)


def team_id(kind: torch.Tensor, cfg) -> torch.Tensor:
    """(N,E) i64. Solo Showdown is full FFA (D14: bots fight each other too, not just the
    hero) -- there is no real team concept to encode. This returns `kind` itself, which is
    inert for any FFA targeting logic (nothing here compares team_id) and exists only as
    forward-compatible infrastructure for a possible future team mode."""
    return kind.clone()


def select_target(state, vis: torch.Tensor, cfg) -> None:
    """MUTATES ent_target. Sticky: keeps the current target as long as it's still alive and
    visible, only re-picking (nearest visible other entity) when it isn't -- this is what
    prevents oscillation between two ~equidistant targets.

    `vis` is whatever the caller says bots can see. bots/policy.all_bot_intents passes
    `bot_visibility(...)`, not the raw `visibility(...)`, so both the re-pick AND the stickiness
    check are bounded by `cfg.bots_sight_tiles` -- a target that walks out of sight must be
    dropped, not held forever, or the sight limit would only apply to acquiring targets and not to
    keeping them. ent_target < 0 means "no target";
    spawn/reset (Step 24) is responsible for initializing it that way, since allocate()'s
    blanket zero-init would otherwise leave it at entity index 0."""
    E = state.ent_pos.shape[1]
    device = state.ent_pos.device

    current = state.ent_target
    current_safe = torch.clamp(current, min=0)
    current_valid = (current >= 0) & torch.gather(vis, 2, current_safe.unsqueeze(-1)).squeeze(-1)

    diff = state.ent_pos.unsqueeze(2) - state.ent_pos.unsqueeze(1)
    dist = geo.safe_norm(diff, dim=-1)  # (N,E,E)
    not_self = ~torch.eye(E, dtype=torch.bool, device=device).unsqueeze(0)
    candidates = vis & not_self

    dist_for_min = torch.where(candidates, dist, torch.full_like(dist, float("inf")))
    nearest = torch.argmin(dist_for_min, dim=-1)
    has_candidate = candidates.any(dim=-1)
    picked = torch.where(has_candidate, nearest, torch.full_like(nearest, -1))

    state.ent_target.copy_(torch.where(current_valid, current, picked))


def incoming_threat(state, params, cfg) -> torch.Tensor:
    """(N,E,2). Not specified further by the plan beyond signature/shape -- this sums, over
    every live projectile not owned by the entity, a unit vector pointing from the
    projectile's current position toward the entity, weighted by 1/(1+closest_approach_dist)
    so nearby/imminent threats dominate. A flee steering behavior (Step 16) can use this
    vector directly."""
    E = state.ent_pos.shape[1]
    P = state.prj_pos.shape[1]
    device = state.ent_pos.device

    ent_pos = state.ent_pos.unsqueeze(2).expand(-1, -1, P, -1)   # (N,E,P,2)
    prj_pos = state.prj_pos.unsqueeze(1).expand(-1, E, -1, -1)   # (N,E,P,2)
    prj_vel = state.prj_vel.unsqueeze(1).expand(-1, E, -1, -1)   # (N,E,P,2)

    _, dist = geo.closest_approach(prj_pos, prj_vel, ent_pos)  # (N,E,P)

    owner = state.prj_owner.unsqueeze(1)  # (N,1,P)
    entity_idx = torch.arange(E, device=device).view(1, E, 1)
    not_owner = owner != entity_idx
    alive_p = state.prj_alive.unsqueeze(1)
    relevant = (not_owner & alive_p).expand(-1, E, -1)

    weight = torch.where(relevant, 1.0 / (1.0 + dist), torch.zeros_like(dist))
    direction = geo.normalize(ent_pos - prj_pos)
    return (direction * weight.unsqueeze(-1)).sum(dim=2)


@dataclass
class BushScan:
    """The nearest SAFE (well clear of the shrinking zone) bush tile for every (N,E) entity, from
    a LOCAL tile scan. This answers "where is the nearest cover I can duck into", which is what
    CAMPER and TRAPPER need. It deliberately does NOT answer "where should I go looking for
    someone" -- that is a map-scale question and lives in `hunt_waypoint`, for reasons its own
    docstring and maps/loader.bush_waypoints spell out.

    `found` is False where no acceptable bush exists inside the search radius, which is the
    "no non-green bushes available -> behave like RUSH" signal the personality layer keys off."""
    pos: torch.Tensor    # (N,E,2) tile center
    dist: torch.Tensor   # (N,E), +inf where ~found
    found: torch.Tensor  # (N,E) bool


@dataclass
class HuntTarget:
    """The nearest bush WAYPOINT this entity has not visited yet -- HUNTER's sweep destination and
    TRAPPER's idle relocation target. `idx` is the waypoint's index into bank.bush_wp, which is the
    bit position bots/personality.advance_hunt sets in `state.ent_hunt_seen` once the entity gets
    there (or gives up on it)."""
    pos: torch.Tensor    # (N,E,2)
    dist: torch.Tensor   # (N,E), +inf where ~found
    found: torch.Tensor  # (N,E) bool
    idx: torch.Tensor    # (N,E) i64, meaningless where ~found
    any_valid: torch.Tensor    # (N,E) bool -- usable waypoints exist at all (vs. all visited)
    n_unvisited: torch.Tensor  # (N,E) i64 -- how many are still unvisited, including this one


# (radius, device) -> (K,2) i64 tile offsets. Built once per distinct key, never per tick: the
# offsets are a pure function of the search radius, and materializing them per call would put a
# few hundred tiny kernel launches back into the hot path -- the exact cost this whole function
# was restructured to remove.
_BUSH_OFFSET_CACHE: dict = {}


def _bush_offsets(radius: int, device) -> torch.Tensor:
    """(K,2) i64 tile offsets: a square of side 2*radius+1 clipped to the inscribed circle,
    INCLUDING (0,0).

    Including the entity's own tile is a deliberate difference from the old sniper archetype's
    `_nearest_bush`,
    which this replaced: that function only ever fed a "drift toward cover" steering term, where
    seeking the tile you already occupy is a no-op, so excluding it cost nothing. It costs a great
    deal here. `found` is what tells CAMPER and TRAPPER that cover exists at all, and with (0,0)
    excluded a bot standing in an ISOLATED bush -- no second bush within the search radius --
    reported `found=False` and fell through to the RUSH fallback, walking straight out of the
    cover it was sitting in. It also broke HUNTER's arrival test, since a hunter that reached its
    target bush could no longer see the thing it had just arrived at.
    """
    key = (radius, device)
    cached = _BUSH_OFFSET_CACHE.get(key)
    if cached is None:
        pairs = [
            (dx, dy)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius
        ]
        # torch.full-per-scalar then stack, NOT torch.tensor([...], device=...) -- see
        # core/geometry.vec2's docstring on why the latter is a genuine host sync under CUDA.
        cached = torch.stack([
            torch.stack([
                torch.full((), dx, device=device, dtype=torch.int64),
                torch.full((), dy, device=device, dtype=torch.int64),
            ])
            for dx, dy in pairs
        ])  # (K,2)
        _BUSH_OFFSET_CACHE[key] = cached
    return cached


def bush_scan(
    pos: torch.Tensor, map_id: torch.Tensor, bank, cfg,
    zone_lo: torch.Tensor | None = None, zone_hi: torch.Tensor | None = None,
    zone_margin: float = 0.0,
) -> BushScan:
    """The nearest BUSH tile center within `cfg.bots_bush_search_tiles` tiles of every (N,E)
    entity, as a single batched `(N,E,K)` tile gather over K fixed offsets (K=49 at radius 4).

    **Call this ONCE per tick and share the result.** It is the single most expensive operation
    in the bot phase and every bush-using personality needs the same answer. Its predecessor
    (the sniper archetype's `_nearest_bush`, one Python iteration per offset) measured
    13.8 ms/tick at
    n_envs=1024 -- 43% of the entire bot phase -- purely in kernel-launch overhead from K
    separate tiny gathers; this batched form measured 1.07 ms for identical output
    (tests/test_perception.py::test_bush_scan_matches_a_naive_per_offset_reference checks it
    against an independent whole-map reference). That ~12.7 ms is what pays for the whole
    personality system.

    `zone_lo`/`zone_hi` are (N,1,2) and exclude candidate tiles inside the damaging shrunk-away
    area, plus -- with `zone_margin` > 0 -- any tile whose own `zone_clearance` is below that
    margin. The margin matters: excluding only tiles that are ALREADY lethal would let a bot walk
    to a bush one tile inside the boundary that the next shrink step swallows, and (worse) would
    let a camper fleeing the zone pick the bush it is already standing in as its destination, so
    it would never actually move. With the margin, any bush the scan returns is strictly further
    from the edge than the flee threshold, so "go to the nearest safe bush" is always genuine
    forward progress. Pass None (or a degenerate rect) to skip the exclusion entirely.

    """
    device = pos.device
    offsets = _bush_offsets(int(cfg.bots_bush_search_tiles), device)  # (K,2)

    ix, iy = terrain.to_tile(pos)  # (N,E)
    cand_ix = ix.unsqueeze(-1) + offsets[:, 0]  # (N,E,K)
    cand_iy = iy.unsqueeze(-1) + offsets[:, 1]
    out_of_bounds = terrain.oob(cand_ix, cand_iy, cfg)
    ix_safe = torch.clamp(cand_ix, 0, cfg.map_w - 1)
    iy_safe = torch.clamp(cand_iy, 0, cfg.map_h - 1)
    map_id_b = terrain._broadcast_map_id(map_id, ix_safe.shape)
    is_bush = bank.is_bush[map_id_b, iy_safe, ix_safe] & ~out_of_bounds  # (N,E,K)

    center = torch.stack(
        [cand_ix.to(pos.dtype) + 0.5, cand_iy.to(pos.dtype) + 0.5], dim=-1,
    )  # (N,E,K,2)

    if zone_lo is not None and zone_hi is not None:
        # Shape guard, not a debug_checks-gated assertion: this is a pure-Python check on .dim()
        # (no tensor read, no host sync, legal in the hot path), and getting it wrong is SILENT --
        # a (N,1,1,2) zone rect broadcasts the candidate mask up to rank 4, argmin then reduces the
        # wrong axis, and every entity gets another entity's answer with no error anywhere.
        if zone_lo.dim() != 3 or zone_hi.dim() != 3:
            raise ValueError(
                f"bush_scan expects (N,1,2) zone bounds (see bots/policy.zone_rect), got "
                f"{tuple(zone_lo.shape)} / {tuple(zone_hi.shape)}"
            )
        lo = zone_lo.unsqueeze(-2)  # (N,1,1,2), broadcasts against (N,E,K,2)
        hi = zone_hi.unsqueeze(-2)
        # Same degenerate-rect guard as bots/policy.zone_contribution: a zero-area rect means
        # "no zone active yet", not "the whole map is lethal".
        rect_active = (hi[..., 0] > lo[..., 0]) & (hi[..., 1] > lo[..., 1])
        lethal = in_zone(center, lo, hi) | (zone_clearance(center, lo, hi) < zone_margin)
        is_bush = is_bush & ~(lethal & rect_active)

    dist = geo.dist(pos.unsqueeze(-2), center)  # (N,E,K)
    masked = torch.where(is_bush, dist, torch.full_like(dist, _INF))
    best_pos, best_dist, found = _pick_min(center, masked)
    return BushScan(pos=best_pos, dist=best_dist, found=found)


def hunt_waypoint(
    pos: torch.Tensor, map_id: torch.Tensor, bank, cfg, seen: torch.Tensor,
    zone_lo: torch.Tensor | None = None, zone_hi: torch.Tensor | None = None,
    zone_margin: float = 0.0,
) -> HuntTarget:
    """The nearest bush waypoint (maps/loader.bush_waypoints) this entity has not yet visited.

    `seen` is (N,E) int64 used as a BITMASK -- bit `w` set means waypoint `w` has been searched.
    One scalar per entity holds the entity's complete search history, which is both smaller and
    more capable than the fixed-size ring of remembered POSITIONS this replaced: a ring of 4
    positions let a hunter re-search everywhere it had been 5 stops ago, and it could not express
    "I have now been everywhere" at all.

    Cost is a single (N,E,W) gather with W <= 63 regardless of map size -- see
    maps/loader.bush_waypoints for why this is both cheaper AND a genuine map-scale sweep, where
    growing `bush_scan`'s radius would have been neither.
    """
    device = pos.device
    waypoints = bank.bush_wp[map_id]        # (N,W,2)
    n_waypoints = bank.n_bush_wp[map_id]    # (N,)
    W = waypoints.shape[1]

    if W == 0:
        # A bank with no waypoint SLOTS at all. maps/loader always pads to MAX_BUSH_WAYPOINTS, so
        # real banks never land here (a bush-free map has W=63 slots with n_bush_wp=0, which the
        # `exists` mask below handles); hand-built test banks do. Guarded because torch.argmin
        # raises on a zero-length reduction axis rather than returning "nothing found".
        empty = torch.zeros(pos.shape[:-1], dtype=torch.bool, device=device)
        return HuntTarget(
            pos=torch.zeros_like(pos),
            dist=torch.full(pos.shape[:-1], _INF, device=device, dtype=pos.dtype),
            found=empty, idx=torch.zeros(pos.shape[:-1], dtype=torch.int64, device=device),
            any_valid=empty, n_unvisited=torch.zeros(pos.shape[:-1], dtype=torch.int64, device=device),
        )

    slot = torch.arange(W, device=device, dtype=torch.int64)
    exists = slot.view(1, 1, W) < n_waypoints.view(-1, 1, 1)     # (N,1,W) -> broadcasts over E
    visited = ((seen.unsqueeze(-1) >> slot) & 1).to(torch.bool)  # (N,E,W)

    wp = waypoints.unsqueeze(1)  # (N,1,W,2)
    if zone_lo is not None and zone_hi is not None:
        lo = zone_lo.unsqueeze(-2)
        hi = zone_hi.unsqueeze(-2)
        rect_active = (hi[..., 0] > lo[..., 0]) & (hi[..., 1] > lo[..., 1])
        lethal = in_zone(wp, lo, hi) | (zone_clearance(wp, lo, hi) < zone_margin)
        exists = exists & ~(lethal & rect_active)

    dist = geo.dist(pos.unsqueeze(-2), wp)  # (N,E,W)
    candidate = exists & ~visited
    masked = torch.where(candidate, dist, torch.full_like(dist, _INF))

    idx = torch.argmin(masked, dim=-1)
    best_dist = torch.gather(masked, -1, idx.unsqueeze(-1)).squeeze(-1)
    gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, 2)
    best_pos = torch.gather(wp.expand(-1, pos.shape[1], -1, -1), -2, gather_idx).squeeze(-2)
    return HuntTarget(
        pos=best_pos, dist=best_dist, found=torch.isfinite(best_dist), idx=idx,
        any_valid=exists.expand(-1, pos.shape[1], -1).any(dim=-1),
        n_unvisited=candidate.sum(dim=-1),
    )


def _pick_min(center: torch.Tensor, masked_dist: torch.Tensor):
    """center: (N,E,K,2), masked_dist: (N,E,K) with +inf for rejected candidates. Returns
    (pos (N,E,2), dist (N,E), found (N,E)). `pos` is an arbitrary candidate where nothing was
    found -- callers must gate on `found`, exactly as with nearest_alive."""
    idx = torch.argmin(masked_dist, dim=-1)  # (N,E)
    dist = torch.gather(masked_dist, -1, idx.unsqueeze(-1)).squeeze(-1)
    gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, 2)
    pos = torch.gather(center, -2, gather_idx).squeeze(-2)
    return pos, dist, torch.isfinite(dist)


def nearest_alive(points_pos: torch.Tensor, points_alive: torch.Tensor, from_pos: torch.Tensor):
    """points_pos: (N,K,2), points_alive: (N,K), from_pos: (N,Q,2). Returns (idx (N,Q) i64,
    dist (N,Q) f32) -- the nearest alive point to each query position (dist is +inf, idx is 0,
    where no point is alive)."""
    diff = from_pos.unsqueeze(2) - points_pos.unsqueeze(1)  # (N,Q,K,2)
    dist = geo.safe_norm(diff, dim=-1)  # (N,Q,K)
    dist_masked = torch.where(points_alive.unsqueeze(1), dist, torch.full_like(dist, float("inf")))
    idx = torch.argmin(dist_masked, dim=-1)  # (N,Q)
    min_dist = torch.gather(dist_masked, -1, idx.unsqueeze(-1)).squeeze(-1)
    return idx, min_dist


def in_zone(pos: torch.Tensor, zone_lo: torch.Tensor, zone_hi: torch.Tensor) -> torch.Tensor:
    """True where pos is OUTSIDE the safe rect [zone_lo, zone_hi] -- i.e. standing in the
    damaging shrunk-away area, matching the Brawl-Stars sense of "in the zone"."""
    return (
        (pos[..., 0] < zone_lo[..., 0]) | (pos[..., 0] > zone_hi[..., 0])
        | (pos[..., 1] < zone_lo[..., 1]) | (pos[..., 1] > zone_hi[..., 1])
    )


def nearest_safe_point(pos: torch.Tensor, zone_lo: torch.Tensor, zone_hi: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(pos[..., 0], zone_lo[..., 0], zone_hi[..., 0])
    y = torch.clamp(pos[..., 1], zone_lo[..., 1], zone_hi[..., 1])
    return torch.stack([x, y], dim=-1)


def zone_clearance(pos: torch.Tensor, zone_lo: torch.Tensor, zone_hi: torch.Tensor) -> torch.Tensor:
    """(...) tiles from pos to the NEAREST EDGE of the safe rect, measured from the inside:
    0 exactly on the boundary, growing toward the rect's middle, and 0 (clamped, not negative)
    anywhere already outside. This is the "how much room do I have left" number both universal
    zone avoidance (bots/policy.zone_avoid_contribution) and Camper's "will not leave its bush
    unless the green zone is 2 or fewer squares away" release condition are expressed in.

    Deliberately NOT `dist(pos, nearest_safe_point(pos))`, which is 0 everywhere INSIDE the rect
    and only grows once you're already taking damage -- useless for avoiding the zone before
    entering it, which is the entire point (D14's "bots avoid the zone")."""
    to_lo = pos - zone_lo
    to_hi = zone_hi - pos
    per_axis = torch.minimum(to_lo, to_hi)  # (...,2)
    return torch.clamp(torch.minimum(per_axis[..., 0], per_axis[..., 1]), min=0.0)
