"""Bot targeting, visibility, and the bush/zone queries bot movement shares.

Standing in a bush is the only thing that CONCEALS an entity; terrain never blocks sight.
visibility() (bush-only, whole-map) drives bot targeting (narrowed by bot_visibility's sight
range) and combat. The hero's observation is narrower: core/camera.hero_view = this matrix's hero
row AND the camera window, because the deployed detector only reports what is on screen. Physical
wall LOS is separate: target_los gates bot fire and raw_los feeds the observation; neither is used
for targeting.
"""
from dataclasses import dataclass

import torch

from ..core import geometry as geo
from ..core import stats
from ..core import terrain
from ..maps import nav

_EPS = 1e-6
_INF = float("inf")
# Entity slot 0 is always the hero (core/observation.py's `_HERO`, core/spawn's layout, and
# bots/policy.all_bot_intents' "zero entity 0" rule all rely on it). `hero_focus` discounts THIS
# COLUMN of the distance matrix -- it is an entity slot along the target axis, not a Kind value.
_HERO_SLOT = 0


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
    """(N,E,E) bool -- `vis` narrowed to what a BOT may act on: the same bush rules AND within
    cfg.bots_sight_tiles (<= 0 disables the limit).

    visibility() has no range limit by design: concealment is a property of bushes, not distance.
    Bots need one: with whole-map sight they almost always hold a target, the "nothing visible"
    branches in bots/personality.py (exploring, HUNTER's sweep) rarely fire, and the lobby
    converges into one scrum an agent learns to sit out. Not folded into visibility() because
    that matrix also feeds the hero's observation (core/observation.py, core/camera.hero_view).
    """
    if cfg.bots_sight_tiles <= 0:
        return vis
    diff = state.ent_pos.unsqueeze(2) - state.ent_pos.unsqueeze(1)  # (N,E,E)
    return vis & (geo.safe_norm(diff, dim=-1) <= cfg.bots_sight_tiles)


def raw_los(state, bank, cfg) -> torch.Tensor:
    """(N,E,E) bool. Physical wall LOS only (terrain.line_of_sight via bank.blocks_proj),
    independent of bush. Observation-only: env builds it when cfg.obs_include_raw_los, for
    obs["visibility"]["los"] and entities.los_from_hero. The bot phase uses the (N,E)
    target_los instead."""
    E = state.ent_pos.shape[1]
    origin_b = state.ent_pos.unsqueeze(2).expand(-1, -1, E, -1)
    target_b = state.ent_pos.unsqueeze(1).expand(-1, E, -1, -1)
    return terrain.line_of_sight(bank, state.map_id, origin_b, target_b, cfg)


def target_los(state, bank, cfg) -> torch.Tensor:
    """(N,E) bool: physical wall LOS from each entity to ITS OWN CURRENT TARGET
    (`state.ent_target`). Requires `select_target` to have already run this tick.

    Rows with no target (ent_target -1, clamped to 0) measure LOS to entity 0 and are meaningless:
    bots/policy.targeting resolves them to has_enemy=False, and fire_gate requires has_target.

    The ray budget is cfg.bots_sight_tiles (<= 0: the full cfg.ray_steps). That is valid only
    because both of select_target's paths, the re-pick AND the stickiness check, are gated on the
    sight-limited bot_visibility matrix, so a live target is within sight range THIS tick.
    """
    idx = torch.clamp(state.ent_target, min=0)
    target_pos = torch.gather(state.ent_pos, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
    max_tiles = cfg.bots_sight_tiles if cfg.bots_sight_tiles > 0 else None
    return terrain.line_of_sight(bank, state.map_id, state.ent_pos, target_pos, cfg,
                                 max_tiles=max_tiles)


def select_target(state, vis: torch.Tensor, params, cfg) -> None:
    """MUTATES ent_target. Sticky: keeps the current target while it is alive and in `vis`,
    re-picking the nearest visible other entity only when it isn't, which stops oscillation
    between two ~equidistant targets. bots/policy.all_bot_intents passes bot_visibility, so a
    target that walks out of sight is dropped, not held. ent_target < 0 means "no target";
    core/spawn sets it to -1 on reset, since allocate()'s zero-init would read as targeting
    entity 0.

    Hero focus: `params.hero_focus` in [0, 1], per kind, discounts the hero's distance by
    `(1 - hero_focus)` before the nearest pick, so at 0.5 a hero 7 tiles away (3.5 effective)
    beats a bot 4 tiles away and a hero 9 tiles away (4.5) does not. A discounted hero that WINS
    overrides a valid sticky target; otherwise "favour the hero" would only reach bots that were
    idle when the hero walked into view. The hero is a candidate only where `vis[:, e, 0]`, so a
    concealed or out-of-range hero is never picked. hero_focus 0 (the hero kind's own value) is
    the plain sticky rule bit for bit: the scale is exactly 1.0.
    """
    E = state.ent_pos.shape[1]
    device = state.ent_pos.device

    current = state.ent_target
    current_safe = torch.clamp(current, min=0)
    current_valid = (current >= 0) & torch.gather(vis, 2, current_safe.unsqueeze(-1)).squeeze(-1)

    diff = state.ent_pos.unsqueeze(2) - state.ent_pos.unsqueeze(1)
    dist = geo.safe_norm(diff, dim=-1)  # (N,E,E)
    not_self = ~torch.eye(E, dtype=torch.bool, device=device).unsqueeze(0)
    candidates = vis & not_self

    # Discount column `_HERO_SLOT` only (the hero as a TARGET), per observer's own kind. The
    # hero's own row (observer 0) gets the hero kind's 0 and is discarded downstream like every
    # other per-entity result computed for slot 0.
    focus = stats.gather_kind(params.hero_focus, state.ent_kind)  # (N,E)
    scale = torch.ones_like(dist)
    scale[:, :, _HERO_SLOT] = 1.0 - focus

    dist_eff = torch.where(candidates, dist * scale, torch.full_like(dist, float("inf")))
    nearest = torch.argmin(dist_eff, dim=-1)
    has_candidate = candidates.any(dim=-1)
    picked = torch.where(has_candidate, nearest, torch.full_like(nearest, -1))

    # Visible AND wins the discounted comparison -> overrides stickiness (see docstring).
    prefer_hero = has_candidate & (nearest == _HERO_SLOT) & (focus > 0)
    sticky_or_picked = torch.where(current_valid, current, picked)
    state.ent_target.copy_(
        torch.where(prefer_hero, torch.full_like(nearest, _HERO_SLOT), sticky_or_picked)
    )


@dataclass
class BushScan:
    """The nearest SAFE (well clear of the shrinking zone) bush tile for every (N,E) entity, from
    a LOCAL tile scan: "the nearest cover I can duck into", for CAMPER and TRAPPER. The map-scale
    "where do I go looking" question is `hunt_waypoint`'s.

    `found` is False where no acceptable bush exists inside the search radius: the "no non-green
    bushes available -> behave like RUSH" signal the personality layer keys off (user's rule)."""
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


# (radius, device) -> (K,2) i64 tile offsets, built once per key: materializing them per call
# would put a few hundred tiny kernel launches into the hot path.
_BUSH_OFFSET_CACHE: dict = {}


def _bush_offsets(radius: int, device) -> torch.Tensor:
    """(K,2) i64 tile offsets: a square of side 2*radius+1 clipped to the inscribed circle,
    INCLUDING (0,0). The entity's own tile must count: without it a bot standing in an isolated
    bush reports found=False, and the RUSH fallback walks it out of the cover it is in.
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

    Call this ONCE per tick and share the result: every bush-using personality needs the same
    answer. One batched gather rather than K small ones, because kernel-launch overhead dominates
    at this size (tests/test_perception.py::test_bush_scan_matches_a_naive_per_offset_reference
    checks it against an independent whole-map reference).

    `zone_lo`/`zone_hi` are (N,1,2) and exclude candidate tiles inside the damaging shrunk-away
    area, plus, with `zone_margin` > 0, any tile whose own `zone_clearance` is below that margin.
    Excluding only lethal tiles would send a bot to a bush the next shrink swallows, and let a
    camper fleeing the zone pick the bush it already stands in; with the margin, every bush
    returned is farther from the edge than the flee threshold, so reaching it is real progress.
    Under `cfg.bots_nav` a tile's clearance is counted along its way out (nav.centre_path_dip,
    the rule bots/policy.zone_clearance applies to the bot itself), so a bush in a pocket whose
    exit the gas is closing on is not picked. Pass None (or a degenerate rect) to skip the
    exclusion.
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
        # Always-on shape guard (a .dim() check, no host sync): a (N,1,1,2) rect would broadcast
        # the mask to rank 4 and argmin would silently reduce the wrong axis.
        if zone_lo.dim() != 3 or zone_hi.dim() != 3:
            raise ValueError(
                f"bush_scan expects (N,1,2) zone bounds (see bots/policy.zone_rect), got "
                f"{tuple(zone_lo.shape)} / {tuple(zone_hi.shape)}"
            )
        lo = zone_lo.unsqueeze(-2)  # (N,1,1,2), broadcasts against (N,E,K,2)
        hi = zone_hi.unsqueeze(-2)
        # Same degenerate-rect guard as bots/policy.zone_rect: a zero-area rect means "no zone
        # active yet", not "the whole map is lethal".
        rect_active = (hi[..., 0] > lo[..., 0]) & (hi[..., 1] > lo[..., 1])
        room = zone_clearance(center, lo, hi)
        if cfg.bots_nav:
            room = room - nav.centre_path_dip(bank, map_id, center, lo, hi, cfg)
        lethal = in_zone(center, lo, hi) | (room < zone_margin)
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

    `seen` is (N,E) int64 used as a BITMASK: bit `w` set means waypoint `w` has been searched, so
    one scalar holds the entity's whole search history, including "I have been everywhere".

    Cost is a single (N,E,W) gather with W <= 63 regardless of map size; see
    maps/loader.bush_waypoints for why this is a map-scale sweep where a larger `bush_scan`
    radius would not be.

    The zone exclusion is bush_scan's, the along-the-way-out clearance under `cfg.bots_nav`
    included.
    """
    device = pos.device
    waypoints = bank.bush_wp[map_id]        # (N,W,2)
    n_waypoints = bank.n_bush_wp[map_id]    # (N,)
    W = waypoints.shape[1]

    if W == 0:
        # A bank with no waypoint SLOTS at all: only hand-built test banks (maps/loader pads real
        # ones to MAX_BUSH_WAYPOINTS; a bush-free map has n_bush_wp=0, which `exists` handles).
        # torch.argmin raises on a zero-length axis rather than returning "nothing found".
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
        room = zone_clearance(wp, lo, hi)
        if cfg.bots_nav:
            room = room - nav.centre_path_dip(bank, map_id, wp, lo, hi, cfg)
        lethal = in_zone(wp, lo, hi) | (room < zone_margin)
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


def zone_clearance(pos: torch.Tensor, zone_lo: torch.Tensor, zone_hi: torch.Tensor) -> torch.Tensor:
    """(...) tiles from pos to the NEAREST EDGE of the safe rect, measured from the inside:
    0 exactly on the boundary, growing toward the rect's middle, and 0 (clamped, not negative)
    anywhere already outside. This is the "how much room do I have left" number both universal
    zone avoidance (bots/policy.zone_avoid_contribution) and Camper's "will not leave its bush
    unless the green zone is 2 or fewer squares away" release condition (user's rule) are
    expressed in.

    Measured from the inside on purpose: a distance TO the safe rect is 0 everywhere inside it
    and only grows once the entity is already taking damage, too late to avoid the zone."""
    to_lo = pos - zone_lo
    to_hi = zone_hi - pos
    per_axis = torch.minimum(to_lo, to_hi)  # (...,2)
    return torch.clamp(torch.minimum(per_axis[..., 0], per_axis[..., 1]), min=0.0)
