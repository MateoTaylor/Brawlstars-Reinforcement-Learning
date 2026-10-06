"""Projectile slot allocation, volley/super/gadget spawning, and the per-tick projectile pipeline.

MAX_PROJ_PER_ENTITY is a hardcoded Python int, not params.proj_count.max(): deriving it from a
tensor would force a host sync in the hot path (spawn_volley runs every tick). It must cover the
widest volley in brawlers.yaml (5 pellets). spawn_volley allocates a (N,E,K) grid with K = this
constant, so a kind with proj_count > K silently fires only K; config.validate's
`peak_projectile_demand` check is about buffer capacity and does NOT catch that, so
tests/test_projectiles.py asserts the relationship directly.

MAX_SPLITS sizes the ring table _spawn_splits reads its arms from and must cover the widest ring
in brawlers.yaml (Spike's 6). config.peak_projectile_demand budgets each kind against its OWN
`split_count`, and config.validate rejects a `split_count` above MAX_SPLITS rather than letting
the extra arms vanish the way an over-wide volley would.

BOX_RADIUS: SimParams has no box collision radius, so projectile-vs-box tests use a fixed 0.5
tiles (a box roughly fills one tile).
"""
import math

import torch

from ..constants import PROJ_CLASS_OF, Proj, ProjClass
from . import geometry as geo
from . import stats
from . import terrain

MAX_PROJ_PER_ENTITY = 5
BOX_RADIUS = 0.5
_EPS = 1e-6
# Relative velocity surplus a gadget spinner is thrown with so it PASSES its landing point on a
# deterministic tick -- see spawn_gadget. Dimensionless; 1e-4 of the travel distance.
_LANDING_OVERSHOOT = 1e-4

# constants.ProjClass values as plain ints, for comparing against the (N,P) int64 `prj_class`.
# Bare ints rather than the IntEnum members so the comparison never builds a temporary on device.
_PROJECTILE = int(ProjClass.PROJECTILE)
_ARTILLERY = int(ProjClass.ARTILLERY)
_HAZARD = int(ProjClass.HAZARD)
# The one Proj KIND step_projectiles treats specially: the gadget spinner never damages its owner
# and never charges a super, both decided per kind rather than per class so other ARTILLERY is
# unaffected (Grom's shell can still hurt Grom).
_GADGET_SPINNER = int(Proj.GADGET_SPINNER)

# The RING a detonating shell splits into: `split_count` unit vectors evenly spaced around the
# circle, starting at +x. Grom's 4 make a cross (exactly -- see the snap below), Spike's 6 a
# hexagonal star.
#
# WORLD-oriented, not relative to the shell's flight direction: the arms read as a fixed shape on
# the tile grid, predictable enough to dodge on sight, so where it is safe to stand is a property
# of the landing point alone.
#
# The COUNT is a per-kind param (`split_count`, a balance knob); the DIRECTIONS are derived from
# it, since every brawler that splits in the game splits evenly. MAX_SPLITS stays structural: it
# decides tensor shapes and how much of the projectile buffer one shell can claim.
MAX_SPLITS = 6


def _ring(count: int) -> tuple[tuple[float, float], ...]:
    """`count` evenly spaced unit vectors from +x, zero-padded out to MAX_SPLITS.

    Values below 1e-12 are snapped to exactly 0.0. Not cosmetic: at count 4 the raw `cos(pi/2)` is
    6.1e-17, which would send Grom's arms off the world axes by a hair; snapped, the count-4 ring
    is exactly `((1,0), (0,1), (-1,0), (0,-1))`."""
    out = []
    for k in range(MAX_SPLITS):
        if k >= count:
            out.append((0.0, 0.0))  # never spawned: _spawn_splits' demand is `count`
            continue
        angle = 2.0 * math.pi * k / count
        out.append(tuple(0.0 if abs(v) < 1e-12 else v for v in (math.cos(angle), math.sin(angle))))
    return tuple(out)


# Indexed by split_count, so row 4 is Grom's cross and row 6 is Spike's star. Row 0 is all zeros
# and is never read -- a kind with no splits has demand 0.
SPLIT_DIR_TABLE = tuple(_ring(c) for c in range(MAX_SPLITS + 1))

_PROJ_CLASS_CACHE: dict = {}


def _proj_class_table(device) -> torch.Tensor:
    """(len(Proj),) i64 view of constants.PROJ_CLASS_OF, built once per device. Cached and filled
    scalar-by-scalar for the same no-H2D-inside-step() reason as _split_dir_table below."""
    cached = _PROJ_CLASS_CACHE.get(device)
    if cached is None:
        cached = torch.stack([
            torch.full((), int(c), device=device, dtype=torch.int64) for c in PROJ_CLASS_OF
        ])
        _PROJ_CLASS_CACHE[device] = cached
    return cached


_SPLIT_DIR_CACHE: dict = {}


def _split_dir_table(device) -> torch.Tensor:
    """(MAX_SPLITS+1, MAX_SPLITS, 2) f32 view of SPLIT_DIR_TABLE, built once per device and then
    indexed by a (N,P) `split_count` to give each shell its own ring.

    Cached, and assembled from scalar fills rather than `torch.tensor(SPLIT_DIR_TABLE, ...)`, as
    bots/personality._weight_table is: _spawn_splits runs every tick inside step(), and a tensor
    built from a Python tuple there is a per-tick H2D copy AND a hard sync, which CONVENTIONS.md
    forbids. tests/test_integration.py::test_native_step_is_sync_free_on_cuda catches it."""
    cached = _SPLIT_DIR_CACHE.get(device)
    if cached is None:
        cached = torch.stack([
            torch.stack([
                torch.stack([torch.full((), c, device=device, dtype=torch.float32) for c in vec])
                for vec in ring
            ])
            for ring in SPLIT_DIR_TABLE
        ])
        _SPLIT_DIR_CACHE[device] = cached
    return cached


def alloc_slots(prj_alive: torch.Tensor, demand: torch.Tensor, max_per_entity: int):
    """prj_alive: (N,P) bool. demand: (N,E) i64 -- how many new slots each entity wants.
    Returns (idx, ok), both (N,E,max_per_entity): idx is a slot index (clamped into [0,P),
    meaningless where ok is False); ok says whether that claim actually landed on a free slot.
    Collision-free across all E shooters in the same env on the same tick. A full buffer
    silently drops the shot (ok=False) -- never overwrites a live projectile, never raises.
    """
    N, P = prj_alive.shape
    device = prj_alive.device

    free_rank = torch.cumsum((~prj_alive).to(torch.int64), dim=1) - 1  # (N,P)
    total_free = free_rank[:, -1] + 1  # (N,)

    offset = torch.cumsum(demand, dim=1) - demand  # exclusive cumsum, (N,E)

    k = torch.arange(max_per_entity, dtype=torch.int64, device=device).view(1, 1, -1)
    claimed_rank = offset.unsqueeze(-1) + k  # (N,E,max_per_entity)
    within_demand = k < demand.unsqueeze(-1)
    ok = within_demand & (claimed_rank < total_free.view(-1, 1, 1))

    claimed_rank_clamped = torch.clamp(claimed_rank, min=0, max=P - 1)
    idx = torch.searchsorted(free_rank, claimed_rank_clamped.reshape(N, -1), right=False)
    idx = torch.clamp(idx, max=P - 1).reshape(N, demand.shape[1], max_per_entity)

    return idx, ok


def _set_scalar(tensor: torch.Tensor, flat_idx: torch.Tensor, flat_valid: torch.Tensor, new_flat: torch.Tensor) -> None:
    """tensor[n, flat_idx[n,i]] = new_flat[n,i] wherever flat_valid[n,i], via a delta +
    scatter_add_ -- collisions among INVALID (masked-out) entries are harmless since their
    delta is zero regardless of which slot they land on."""
    is_bool = tensor.dtype == torch.bool
    work = tensor.to(torch.int64) if is_bool else tensor
    new_w = new_flat.to(torch.int64) if is_bool else new_flat
    old = work.gather(1, flat_idx)
    delta = torch.where(flat_valid, new_w - old, torch.zeros_like(old))
    work.scatter_add_(1, flat_idx, delta)
    if is_bool:
        tensor.copy_(work.to(torch.bool))


def _set_vec2(tensor: torch.Tensor, flat_idx: torch.Tensor, flat_valid: torch.Tensor, new_flat: torch.Tensor) -> None:
    idx_exp = flat_idx.unsqueeze(-1).expand(*flat_idx.shape, 2)
    old = tensor.gather(1, idx_exp)
    delta = torch.where(flat_valid.unsqueeze(-1), new_flat - old, torch.zeros_like(old))
    tensor.scatter_add_(1, idx_exp, delta)


def _write_slots(state, flat_idx: torch.Tensor, flat_valid: torch.Tensor, *, pos, vel, target,
                 dist_left, damage, radius, aoe, age, owner, kind, cls, pierce, alive) -> None:
    """MUTATES: all prj_*. Writes one fully-defined projectile into every slot flat_idx[n,i]
    whose flat_valid[n,i] is True. Each field is (N,M) -- (N,M,2) for the vec2s -- already
    flattened over whatever grid the caller allocated against (shooters x volley for
    spawn_volley, dying shots for _spawn_hazards). Every prj_ field is written, including
    the ones the new projectile's own physics will never read (a non-lobbed projectile's
    target), so a slot never carries a mix of its own values and its previous occupant's."""
    _set_vec2(state.prj_pos, flat_idx, flat_valid, pos)
    _set_vec2(state.prj_vel, flat_idx, flat_valid, vel)
    _set_vec2(state.prj_target, flat_idx, flat_valid, target)
    _set_scalar(state.prj_dist_left, flat_idx, flat_valid, dist_left)
    _set_scalar(state.prj_damage, flat_idx, flat_valid, damage)
    _set_scalar(state.prj_radius, flat_idx, flat_valid, radius)
    _set_scalar(state.prj_aoe, flat_idx, flat_valid, aoe)
    _set_scalar(state.prj_age, flat_idx, flat_valid, age)
    _set_scalar(state.prj_owner, flat_idx, flat_valid, owner)
    _set_scalar(state.prj_kind, flat_idx, flat_valid, kind)
    _set_scalar(state.prj_class, flat_idx, flat_valid, cls)
    _set_scalar(state.prj_pierce, flat_idx, flat_valid, pierce)
    _set_scalar(state.prj_alive, flat_idx, flat_valid, alive)


def spawn_volley(state, fire_mask: torch.Tensor, origin: torch.Tensor, aim_dir: torch.Tensor,
                  aim_point: torch.Tensor, kind: torch.Tensor, damage: torch.Tensor, params, cfg) -> None:
    """fire_mask/kind: (N,E). origin/aim_dir/aim_point/damage: (N,E,2) or (N,E) as shaped.
    MUTATES: all prj_*. proj_count[kind] projectiles in a symmetric fan spanning
    proj_spread_rad, evenly spaced (a lone projectile always goes straight down aim_dir,
    regardless of spread). lobbed = PROJ_CLASS_OF[proj_kind] is ARTILLERY. dist_left =
    attack_range, except for a fixed-flight-time lob (proj_flight_seconds > 0), which gets
    exactly the distance to its landing point -- see the velocity comment below.
    """
    N, E = fire_mask.shape
    K = MAX_PROJ_PER_ENTITY
    device = fire_mask.device

    proj_count = stats.gather_kind(params.proj_count, kind)
    proj_speed = stats.gather_kind(params.proj_speed, kind)
    proj_radius = stats.gather_kind(params.proj_radius, kind)
    proj_spread = stats.gather_kind(params.proj_spread_rad, kind)
    proj_kind_of_shot = stats.gather_kind(params.proj_kind, kind)
    attack_range = stats.gather_kind(params.attack_range, kind)
    aoe_radius = stats.gather_kind(params.aoe_radius, kind)
    flight_seconds = stats.gather_kind(params.proj_flight_seconds, kind)

    # Lead prediction (geo.lead_target) and artillery's positional aim noise (aim_noise_tiles) can
    # push aim_point past the map edge, and a lobbed shell flies a straight line to its target
    # with no wall/bounds clamping (it arcs OVER walls), so an off-map target would carry it
    # off-map mid-flight, which check_invariants flags. Clamping into a small inset (not the exact
    # edge: detonation triggers on the tick the shell passes the target, up to one tick's travel
    # beyond it) keeps the whole path in the convex map. Harmless for non-lobbed shots: their
    # prj_target is stored but never read for physics.
    margin = cfg.los_step_tiles
    aim_point = torch.stack([
        torch.clamp(aim_point[..., 0], margin, cfg.map_w - margin),
        torch.clamp(aim_point[..., 1], margin, cfg.map_h - margin),
    ], dim=-1)
    # A lobbed shell detonates when it passes aim_point (dot_val <= 0 in step_projectiles), so its
    # launch direction must point exactly at the clamped aim_point, or it can fly off the map
    # before ever "passing" it. Recomputed ONLY for lobbed shots: a non-lobbed aim_dir carries
    # angular firing noise (bots/combat_rules' LEAD model) that is independent of aim_point and
    # must not be overwritten. ARTILLERY vs PROJECTILE is decided by the weapon's proj_kind, via
    # constants.PROJ_CLASS_OF, since more than one weapon lobs.
    cls_new_1d = _proj_class_table(device)[proj_kind_of_shot]
    is_lobbed = cls_new_1d == _ARTILLERY
    aim_dir = torch.where(is_lobbed.unsqueeze(-1), geo.normalize(aim_point - origin), aim_dir)

    demand = torch.where(fire_mask, proj_count, torch.zeros_like(proj_count))
    idx, ok = alloc_slots(state.prj_alive, demand, K)  # (N,E,K)

    k_range = torch.arange(K, dtype=torch.float32, device=device).view(1, 1, K)
    count_f = proj_count.to(torch.float32).unsqueeze(-1)  # (N,E,1)
    denom = torch.clamp(count_f - 1.0, min=1.0)
    fan_offset = proj_spread.unsqueeze(-1) * (k_range - (count_f - 1.0) / 2.0) / denom  # (N,E,K)

    base_angle = geo.angle_of(aim_dir).unsqueeze(-1)  # (N,E,1)
    proj_dir = geo.from_angle(base_angle + fan_offset)  # (N,E,K,2)

    # Fixed flight TIME for a lobbed shell when proj_flight_seconds > 0 (Grom's "Watch This!"):
    # the speed puts it on its landing point in exactly that many seconds, so a 1-tile lob and a
    # max-range lob land at the same time -- a close shot is slow, a long one fast. proj_speed is
    # unused for these shells; for artillery it is the split shards' fallback speed
    # (_spawn_splits).
    to_aim = aim_point - origin
    timed_lob = is_lobbed & (flight_seconds > 0)
    vel_timed = (to_aim / torch.clamp(flight_seconds, min=_EPS).unsqueeze(-1)).unsqueeze(2)
    vel_fan = proj_dir * proj_speed.unsqueeze(-1).unsqueeze(-1)
    # dist_left is a range cap for everything else, but for a timed lob it is exactly the
    # distance the shell will cover: aim noise and lead prediction can put the landing point
    # past attack_range, and expiring there would detonate the shell short of the point its
    # whole arc was solved for.
    travel = torch.where(timed_lob, geo.safe_norm(to_aim, dim=-1), attack_range)

    pos_new = origin.unsqueeze(2).expand(N, E, K, 2)
    vel_new = torch.where(timed_lob.view(N, E, 1, 1), vel_timed, vel_fan)
    target_new = aim_point.unsqueeze(2).expand(N, E, K, 2)
    dist_left_new = travel.unsqueeze(-1).expand(N, E, K)
    damage_new = damage.unsqueeze(-1).expand(N, E, K)
    radius_new = proj_radius.unsqueeze(-1).expand(N, E, K)
    aoe_new = aoe_radius.unsqueeze(-1).expand(N, E, K)
    age_new = torch.zeros(N, E, K, device=device)
    owner_new = torch.arange(E, dtype=torch.int64, device=device).view(1, E, 1).expand(N, E, K)
    kind_new = proj_kind_of_shot.unsqueeze(-1).expand(N, E, K)
    cls_new = cls_new_1d.unsqueeze(-1).expand(N, E, K)
    alive_new = torch.ones(N, E, K, dtype=torch.bool, device=device)

    _write_slots(
        state, idx.reshape(N, E * K), ok.reshape(N, E * K),
        pos=pos_new.reshape(N, E * K, 2), vel=vel_new.reshape(N, E * K, 2),
        target=target_new.reshape(N, E * K, 2), dist_left=dist_left_new.reshape(N, E * K),
        damage=damage_new.reshape(N, E * K), radius=radius_new.reshape(N, E * K),
        aoe=aoe_new.reshape(N, E * K), age=age_new.reshape(N, E * K),
        owner=owner_new.reshape(N, E * K), kind=kind_new.reshape(N, E * K),
        cls=cls_new.reshape(N, E * K),
        pierce=torch.zeros(N, E * K, dtype=torch.bool, device=device),
        alive=alive_new.reshape(N, E * K),
    )


def spawn_supers(state, fire_mask: torch.Tensor, origin: torch.Tensor, aim_dir: torch.Tensor,
                 params, cfg) -> None:
    """fire_mask: (N,E) -- entities firing their super THIS tick. MUTATES: all prj_*.

    One PIERCING bolt per firing entity, using the `super_*` stat block (range, speed, damage,
    radius) rather than the weapon's. Separate from `spawn_volley` because a super shares none of
    its fan/spread/lob machinery: it is always exactly one projectile in one direction.

    Not hero-specific: `fire_mask` is (N,E) and every stat is gathered per kind, so a bot firing a
    super needs no change here.

    `aim_dir` must be nonzero on every firing row: `geo.normalize` keeps a zero vector zero, and a
    bolt with no velocity never moves and never runs out of `dist_left`, so it would sit where it
    spawned until reset. `env._attack_phase` passes `super_dir`, which is never zero.
    """
    N, E = fire_mask.shape
    device = fire_mask.device

    super_range = stats.gather_kind(params.super_range, state.ent_kind)
    super_speed = stats.gather_kind(params.super_proj_speed, state.ent_kind)
    super_damage = stats.gather_kind(params.super_damage, state.ent_kind)
    super_radius = stats.gather_kind(params.super_radius, state.ent_kind)

    demand = fire_mask.to(torch.int64)
    idx, ok = alloc_slots(state.prj_alive, demand, 1)  # (N,E,1)

    direction = geo.normalize(aim_dir)
    _write_slots(
        state, idx.reshape(N, E), ok.reshape(N, E),
        pos=origin,
        vel=direction * super_speed.unsqueeze(-1),
        target=origin,                        # unread: a super is not lobbed
        dist_left=super_range,
        damage=super_damage,
        radius=super_radius,
        aoe=torch.zeros(N, E, device=device),  # it hits what it passes through, not an area
        age=torch.zeros(N, E, device=device),
        owner=torch.arange(E, dtype=torch.int64, device=device).view(1, E).expand(N, E),
        kind=torch.full((N, E), int(Proj.SUPER_BOLT), dtype=torch.int64, device=device),
        cls=torch.full((N, E), _PROJECTILE, dtype=torch.int64, device=device),
        pierce=torch.ones(N, E, dtype=torch.bool, device=device),
        alive=torch.ones(N, E, dtype=torch.bool, device=device),
    )


def spawn_gadget(state, fire_mask: torch.Tensor, origin: torch.Tensor, direction: torch.Tensor,
                 travel: torch.Tensor, damage: torch.Tensor, params, cfg) -> None:
    """fire_mask: (N,E) -- entities throwing their gadget THIS tick. origin: (N,E,2). direction:
    (N,E,2) unit vectors and travel: (N,E) tiles, both straight from `hero.gadget_target`.
    damage: (N,E), from `stats.effective_gadget_damage`. MUTATES: all prj_*.

    One ARTILLERY-class `Proj.GADGET_SPINNER` per firing entity, a sibling of `spawn_supers`: it
    shares none of `spawn_volley`'s fan/spread machinery, is always exactly one projectile, and
    reads the `gadget_*` stat block rather than the weapon's.

    The slot is a timed lob: it lands on `target = origin + dir * travel` after exactly
    `gadget_flight_seconds`, whatever the distance, and `step_projectiles`' artillery path does
    the rest -- `arrived` (dot <= 0) once it passes the target, detonation ON the stored target,
    blast over every entity and box within `prj_aoe = gadget_radius`. An ARTILLERY shell ignores
    walls, units and boxes until it lands, which is why `gadget_target` clips `travel` against
    terrain BEFORE the throw. A zero `travel` (enemy on top of the thrower) is legal: `vel` is 0,
    `dot <= 0` holds on the first tick, and it detonates in place. `dist_left = travel + eps` so
    the range check can never fire a tick before `arrived` and move the detonation off target.

    `prj_radius` is 0: nothing collides with it in flight, and its only other reader,
    `_spawn_splits`' rim, never runs for a kind without `split_count`. Not hero-specific, like
    `spawn_supers`: every stat is gathered per kind.
    """
    N, E = fire_mask.shape
    device = fire_mask.device

    flight_seconds = stats.gather_kind(params.gadget_flight_seconds, state.ent_kind)
    gadget_radius = stats.gather_kind(params.gadget_radius, state.ent_kind)

    demand = fire_mask.to(torch.int64)
    idx, ok = alloc_slots(state.prj_alive, demand, 1)  # (N,E,1)

    # `direction` is used as given: `gadget_target` guarantees unit vectors on every row.
    target = origin + direction * travel.unsqueeze(-1)
    # `travel / flight`, plus one part in ten thousand. Without the nudge, `flight_seconds / dt`
    # ticks of `vel * dt` can sum to an ulp LESS than `travel` in float32, so the landing tick
    # would depend on the travel (and `dist_left = travel + eps` deliberately never fires first).
    # With it, `arrived` is true on exactly `ceil(flight / dt)` ticks for every travel, and
    # nothing observable changes: the detonation point is the stored `prj_target`, and the
    # in-flight position is off by at most 2e-4 tiles at max range.
    vel = direction * (travel * (1.0 + _LANDING_OVERSHOOT) / torch.clamp(flight_seconds, min=_EPS)).unsqueeze(-1)
    _write_slots(
        state, idx.reshape(N, E), ok.reshape(N, E),
        pos=origin,
        vel=vel,
        target=target,                         # READ: this is a lob, it detonates here
        dist_left=travel + _EPS,
        damage=damage,
        radius=torch.zeros(N, E, device=device),
        aoe=gadget_radius,
        age=torch.zeros(N, E, device=device),
        owner=torch.arange(E, dtype=torch.int64, device=device).view(1, E).expand(N, E),
        kind=torch.full((N, E), _GADGET_SPINNER, dtype=torch.int64, device=device),
        cls=torch.full((N, E), _ARTILLERY, dtype=torch.int64, device=device),
        pierce=torch.zeros(N, E, dtype=torch.bool, device=device),
        alive=torch.ones(N, E, dtype=torch.bool, device=device),
    )


def _spawn_splits(state, detonate: torch.Tensor, det_pos: torch.Tensor, params, cfg) -> None:
    """detonate: (N,P) bool -- shells that blew up THIS tick. det_pos: (N,P,2) where each did.
    MUTATES: all prj_*. Every detonating shell whose owner has split_distance/
    split_damage_fraction/split_count set hands `split_count` shards to free slots: ordinary
    non-lobbed projectiles (so walls, units and boxes all stop them) flying the evenly spaced ring
    of SPLIT_DIR_TABLE for split_distance tiles and carrying split_damage_fraction of the shell's
    damage each. For a kind with those params unset, demand is 0 and this is a no-op write.

    MUST be called after state.prj_alive reflects the detonations, so shards can reuse the
    slots their own shells just freed. A shell whose shards don't fit in the buffer simply
    loses them, the same way a volley into a full buffer is dropped (alloc_slots' ok=False) --
    the slot assignment below is alloc_slots', solved the other way round.
    """
    N, P = state.prj_pos.shape[:2]
    E = state.ent_pos.shape[1]
    K = MAX_SPLITS
    device = det_pos.device

    # The shard stats live on the OWNER's kind, not on the shell: a projectile carries no
    # back-reference to its archetype's row in SimParams, and prj_owner + ent_kind reconstruct
    # one for free (observation.py resolves owner_kind the same way). ent_kind outlives the
    # entity's death, so a shell whose shooter died mid-flight still splits correctly.
    owner_kind = torch.gather(state.ent_kind, 1, torch.clamp(state.prj_owner, min=0, max=E - 1))
    split_distance = stats.gather_kind(params.split_distance, owner_kind)
    split_fraction = stats.gather_kind(params.split_damage_fraction, owner_kind)
    # Clamped only to guard the ring-table index: config.validate already rejects a split_count
    # above MAX_SPLITS, but an out-of-bounds gather is a device-side assert that takes the whole
    # process down on CUDA.
    split_count = torch.clamp(stats.gather_kind(params.split_count, owner_kind), 0, K)

    # Shard speed is derived from how long the arms take to FINISH: CHARACTER_DETAILS gives Grom's
    # split strikes in seconds, so the config states that observable (`split_seconds`) and the
    # speed is `split_distance / split_seconds`. `split_seconds: 0` falls back to `proj_speed`.
    split_seconds = stats.gather_kind(params.split_seconds, owner_kind)
    timed_shard = split_seconds > 0
    shard_speed = torch.where(
        timed_shard,
        split_distance / torch.clamp(split_seconds, min=_EPS),
        stats.gather_kind(params.proj_speed, owner_kind),
    )

    splits = detonate & (split_distance > 0) & (split_fraction > 0) & (split_count > 0)
    zero_demand = torch.zeros_like(state.prj_owner)
    demand = torch.where(splits, split_count, zero_demand)  # (N,P)

    # Slot assignment, solved from the SLOT side: the r-th free slot goes to the shell whose demand
    # range covers r, as that shell's arm r - offset. Same slots and shards as `alloc_slots`, whose
    # (N,P,K) claims would be nearly all empty and each written through 13 (N,P*K) scatters, and a
    # full buffer drops the same tail of claims. Every field below is one (N,P) gather from the
    # claiming shell plus one masked write.
    free_rank = torch.cumsum((~state.prj_alive).to(torch.int64), dim=1) - 1  # (N,P)
    claimed = torch.cumsum(demand, dim=1)  # inclusive; p owns [claimed - demand, claimed)
    written = ~state.prj_alive & (free_rank < claimed[:, -1:])
    rank = torch.clamp(free_rank, min=0)
    shell = torch.clamp(torch.searchsorted(claimed, rank, right=True), max=P - 1)  # (N,P)
    arm = rank - (claimed.gather(1, shell) - demand.gather(1, shell))

    def from_shell(values: torch.Tensor) -> torch.Tensor:
        """(N,P) per-shell values -> the value of the shell that claimed each slot."""
        return values.gather(1, shell)

    # (N,P,2): the arm each slot receives, from its shell's OWN ring (split_count). An unwritten
    # slot's arm can be anything, hence the clamp; its direction is never stored.
    dirs = _split_dir_table(device)[from_shell(split_count), torch.clamp(arm, 0, K - 1)]

    # Shards start at the RIM of the blast, not its centre: anything the detonation damaged is
    # within prj_aoe of det_pos, and a shard starting (unit_radius + prj_radius) beyond that rim
    # and flying outward can never come within hit range of it. So the ground is partitioned --
    # the landing disc OR the arms, never both. Shards deal the shell's FULL damage
    # (split_damage_fraction 1.0), so an overlap would stack into a one-shot.
    #
    # A shell with NO blast (`aoe_radius: 0` -- Spike) keeps the formula on purpose: the ring then
    # sits at exactly the shard hit radius, the smallest ring that does not put every arm inside
    # a target standing on the landing point (a zero rim would land all `split_count` shards in
    # it). The cost is that a shot landing within a few thousandths of a tile of a body's centre
    # catches nobody, which LOB's aim noise makes vanishingly rare. tests/test_spike.py measures
    # the shard-count profile.
    rim = state.prj_aoe + params.unit_radius.view(N, 1) + state.prj_radius + _EPS  # (N,P)
    start = det_pos.gather(1, shell.unsqueeze(-1).expand(N, P, 2)) + dirs * from_shell(rim).unsqueeze(-1)
    # A shell landing near the map edge would put its outward shards off the map, which
    # check_invariants flags. Clamping into spawn_volley's landing inset keeps them legal; the
    # border is WALL on every map (maps/loader.validate_map), so a clamped shard dies on its first
    # march anyway.
    margin = cfg.los_step_tiles
    start = torch.stack([
        torch.clamp(start[..., 0], margin, cfg.map_w - margin),
        torch.clamp(start[..., 1], margin, cfg.map_h - margin),
    ], dim=-1)

    # Every value is read off the claiming shell BEFORE anything is written: a shard can land in
    # the slot its own shell (or another detonating shell) just freed.
    vel = dirs * from_shell(shard_speed).unsqueeze(-1)
    dist_left = from_shell(split_distance)
    damage = from_shell(state.prj_damage * split_fraction)
    radius = from_shell(state.prj_radius)
    owner = from_shell(state.prj_owner)
    # Same Proj kind as the shell that spawned them: they are the same weapon, and the observation
    # (projectiles.lobbed) and the viewer already tell arc from shard by class, so no extra enum
    # member widens every kind_onehot in the obs.
    kind = from_shell(state.prj_kind)

    # Every prj_ field is written, as `_write_slots` would, so a slot never carries a mix of its
    # own values and its previous occupant's.
    w = written.unsqueeze(-1)
    state.prj_pos.copy_(torch.where(w, start, state.prj_pos))
    state.prj_vel.copy_(torch.where(w, vel, state.prj_vel))
    state.prj_target.copy_(torch.where(w, start, state.prj_target))  # unread: not lobbed
    state.prj_dist_left.copy_(torch.where(written, dist_left, state.prj_dist_left))
    state.prj_damage.copy_(torch.where(written, damage, state.prj_damage))
    state.prj_radius.copy_(torch.where(written, radius, state.prj_radius))
    state.prj_aoe.masked_fill_(written, 0.0)  # only the shell has a blast
    state.prj_age.masked_fill_(written, 0.0)
    state.prj_owner.copy_(torch.where(written, owner, state.prj_owner))
    state.prj_kind.copy_(torch.where(written, kind, state.prj_kind))
    state.prj_class.masked_fill_(written, _PROJECTILE)
    state.prj_pierce.masked_fill_(written, False)
    state.prj_alive.masked_fill_(written, True)


def _spawn_hazards(state, died: torch.Tensor, death_pos: torch.Tensor, params, cfg) -> None:
    """died: (N,P) bool -- PROJECTILE-class projectiles that ended this tick. death_pos: (N,P,2).
    MUTATES: all prj_*. Every one whose OWNER's kind sets `on_hit_area_radius > 0` leaves a
    HAZARD-class projectile behind (Brock's lingering sphere).

    Same shape as `_spawn_splits`, for the same reasons: stats come from the owner's kind via
    `prj_owner` + `ent_kind` (so a Brock who dies mid-flight still leaves a sphere), slot
    allocation reuses `alloc_slots`, and it MUST run after `prj_alive` reflects the deaths so the
    sphere can claim the slot its own rocket just freed.

    Spawns on ANY death -- wall, unit, box, or running out of range -- not only a unit hit: a slow
    rocket often expires without hitting anything, and one that detonates on a wall still leaves
    a puddle in the game, so unit hits alone would remove most of Brock's area denial.

    `death_pos` is the end-of-tick position rather than the exact segment/circle intersection: at
    most one tick of travel off, small against the sphere, and exactly where the rocket is drawn
    when it dies.
    """
    N, P = state.prj_pos.shape[:2]
    E = state.ent_pos.shape[1]
    device = death_pos.device

    owner_kind = torch.gather(state.ent_kind, 1, torch.clamp(state.prj_owner, min=0, max=E - 1))
    radius = stats.gather_kind(params.on_hit_area_radius, owner_kind)
    # A FRACTION of the projectile's own damage, like `split_damage_fraction`, never an absolute
    # figure: `prj_damage` was set at spawn from `stats.effective_damage`, so the sphere scales
    # with the cube bonus, `enemy_damage_mult`, and the curriculum's per-tier damage multiplier
    # (training/curriculum.py scales `base_damage`).
    fraction = stats.gather_kind(params.on_hit_area_damage_fraction, owner_kind)
    damage = state.prj_damage * fraction

    spawns = died & (radius > 0)
    zero_demand = torch.zeros_like(state.prj_owner)
    demand = torch.where(spawns, torch.ones_like(zero_demand), zero_demand)  # (N,P)
    idx, ok = alloc_slots(state.prj_alive, demand, 1)  # (N,P,1)

    zeros = torch.zeros(N, P, device=device)
    _write_slots(
        state, idx.reshape(N, P), ok.reshape(N, P),
        pos=death_pos,
        vel=torch.zeros(N, P, 2, device=device),   # a hazard does not move
        target=death_pos,                          # unread; written so the slot is fully defined
        dist_left=zeros,                           # never expires by distance
        damage=damage,
        radius=zeros,                              # nothing collides with it
        aoe=radius,                                # the sphere's reach, reusing the AoE field
        age=zeros,                                 # the tick schedule is derived from this
        owner=state.prj_owner,
        # Same Proj kind as the rocket that left it -- a hazard is not a separate WEAPON, so the
        # observation and renderer still attribute it correctly. The CLASS is what differs.
        kind=state.prj_kind,
        cls=torch.full((N, P), _HAZARD, dtype=torch.int64, device=device),
        pierce=torch.zeros(N, P, dtype=torch.bool, device=device),
        alive=torch.ones(N, P, dtype=torch.bool, device=device),
    )


def _box_grid(state, cfg) -> torch.Tensor:
    """(N, H*W + 1) int16: the slot of the ALIVE box on each tile (cell `y * W + x`), B on a tile
    with none. The extra last cell is a sink: every dead box is scattered there with the value B,
    and out-of-map neighbours in `_box_candidates` point at it, so it always reads "no box".

    Rebuilt every tick rather than kept in the state: boxes never move, but they die, and a
    per-tick rebuild has no reset or autoreset path to keep in sync. int16 because it is the one
    (N, H*W) tensor here -- 29 MB at 4096 envs on a 60x60 map -- and B is far below 2**15.

    Relies on alive boxes sitting on distinct tiles (a shared tile would keep only one of them).
    `boxes.spawn_boxes` places them on distinct tile-centre spots, and `state.check_invariants`
    enforces it under `debug_checks`."""
    N, B = state.box_alive.shape
    H, W = cfg.map_h, cfg.map_w
    device = state.box_pos.device
    ix, iy = terrain.to_tile(state.box_pos)
    cell = torch.clamp(iy, 0, H - 1) * W + torch.clamp(ix, 0, W - 1)
    cell = torch.where(state.box_alive, cell, H * W)
    slot = torch.where(state.box_alive, torch.arange(B, device=device).view(1, B), B).to(torch.int16)
    grid = torch.full((N, H * W + 1), B, dtype=torch.int16, device=device)
    return grid.scatter_(1, cell, slot)


def _box_candidates(state, grid, centre: torch.Tensor, cells: int, cfg):
    """The box slots that `step_projectiles` tests each projectile slot against: every box on a
    tile within `cells` tiles (per axis) of the tile holding `centre` (N,P,2). Returns (slot
    (N,P,C) i64, pos (N,P,C,2), alive (N,P,C)), C = (2*cells + 1)**2; a candidate with no box
    has slot B and alive False.

    Exact for any box whose centre is within `cells` tiles of `centre`: a box on a tile outside
    the block is more than `cells` away along one axis. Each in-map tile appears once, so no box
    is ever counted twice.

    `grid` None is the dense fallback -- every box is a candidate, (N,1,B) broadcasting against
    the (N,P,...) callers -- for when the block would hold at least B tiles anyway (a small test
    config, or a spec whose reach is unbounded)."""
    N, B = state.box_alive.shape
    P = centre.shape[1]
    device = centre.device
    if grid is None:
        slot = torch.arange(B, device=device).view(1, 1, B).expand(N, P, B)
        return slot, state.box_pos.unsqueeze(1), state.box_alive.unsqueeze(1)

    H, W = cfg.map_h, cfg.map_w
    ix, iy = terrain.to_tile(centre)
    offsets = torch.arange(-cells, cells + 1, device=device)
    cx = torch.clamp(ix, 0, W - 1).unsqueeze(-1) + offsets  # (N,P,D)
    cy = torch.clamp(iy, 0, H - 1).unsqueeze(-1) + offsets
    in_map = ((cy >= 0) & (cy < H)).unsqueeze(-1) & ((cx >= 0) & (cx < W)).unsqueeze(-2)  # (N,P,D,D)
    cell = torch.where(in_map, cy.unsqueeze(-1) * W + cx.unsqueeze(-2), H * W).reshape(N, -1)

    slot = grid.gather(1, cell).to(torch.int64)  # (N, P*C)
    safe = torch.clamp(slot, max=B - 1)
    pos = state.box_pos.gather(1, safe.unsqueeze(-1).expand(N, safe.shape[1], 2))
    alive = state.box_alive.gather(1, safe) & (slot < B)
    return slot.view(N, P, -1), pos.view(N, P, -1, 2), alive.view(N, P, -1)


def _earliest_only(valid_hit: torch.Tensor, hit_t: torch.Tensor) -> torch.Tensor:
    """valid_hit/hit_t: (..., M). Keeps only the smallest-t True entry along the last dim."""
    t_for_argmin = torch.where(valid_hit, hit_t, torch.full_like(hit_t, float("inf")))
    earliest = torch.argmin(t_for_argmin, dim=-1, keepdim=True)
    has_any = valid_hit.any(dim=-1, keepdim=True)
    earliest_mask = torch.zeros_like(valid_hit).scatter_(-1, earliest, has_any)
    return valid_hit & earliest_mask


def step_projectiles(state, bank, params, cfg):
    """MUTATES: all prj_*. Returns (dmg_ent (N,E), dmg_by (N,E,E), dmg_box (N,B),
    heal_ent (N,E), charge_hit (N,E,E)) -- the first four reported here, applied by
    combat.apply_damage / combat.apply_heal.

    `charge_hit[n,a,v]` is "attacker a landed a SUPER-CHARGING hit on victim v this tick":
    `dmg_by > 0` with the gadget spinner's damage left out. `env._bookkeeping` counts charge from
    it because the spinner deals damage but does not charge the super, a rule that cannot be
    recovered from `dmg_by` once the per-projectile kinds have been summed away."""
    N, P = state.prj_pos.shape[:2]
    E = state.ent_pos.shape[1]
    B = state.box_pos.shape[1]
    device = state.prj_pos.device

    alive = state.prj_alive
    non_lobbed = state.prj_class == _PROJECTILE
    old_pos = state.prj_pos

    # (1) integrate, decrement dist_left, increment age
    step_delta = state.prj_vel * cfg.dt
    raw_new_pos = old_pos + step_delta
    step_dist = geo.safe_norm(step_delta, dim=-1)
    new_dist_left = state.prj_dist_left - step_dist
    new_age = state.prj_age + cfg.dt

    # (2) wall collision for non-lobbed via march, kill at hit point. march() samples every
    # los_step_tiles, so wall_hit_pos is the first BLOCKED sample, not the true wall boundary: AT
    # or (with float rounding) fractionally past the wall tile, which check_invariants flags as
    # out-of-bounds when that wall is the map border. Backing the kill position off by one
    # los_step_tiles along the travel direction keeps it on the known-safe side.
    move_dir = geo.normalize(step_delta)
    # Ray budget: `wall_hit` is only read for alive, non-lobbed, non-piercing slots (`wall_kill`
    # below), and none of those moves further in one tick than `params.shot_step_tiles` -- the
    # spec-derived bound on the fastest such shot (config.shot_step_tiles). Every march sample past
    # a slot's own step is masked out by march's `in_range` anyway, so the budget changes no
    # answer that is read, and the march takes 1 sample plus the endpoint instead of 48 plus it.
    wall_hit, wall_hit_pos, _ = terrain.march(bank.blocks_proj, state.map_id, old_pos, move_dir, step_dist, cfg,
                                              max_tiles=params.shot_step_tiles)
    safe_wall_hit_pos = wall_hit_pos - move_dir * cfg.los_step_tiles
    # A PIERCING bolt (the super) passes through walls: `wall_hit` is still computed -- the same
    # march either way -- but it neither kills the bolt nor clips its position.
    pierce = state.prj_pierce
    wall_kill = alive & non_lobbed & wall_hit & ~pierce
    pos_after_wall = torch.where(wall_kill.unsqueeze(-1), safe_wall_hit_pos, raw_new_pos)

    # (3) unit collision over (N,P,E), excluding owner and dead, earliest t wins
    p0 = old_pos.unsqueeze(2)              # (N,P,1,2)
    p1 = pos_after_wall.unsqueeze(2)       # (N,P,1,2)
    ent_pos_b = state.ent_pos.unsqueeze(1)  # (N,1,E,2)
    hit_r = params.unit_radius.view(-1, 1, 1) + state.prj_radius.unsqueeze(-1)  # (N,P,1)->(N,P,E)
    unit_hit, unit_hit_t = geo.segment_circle_hit(p0, p1, ent_pos_b, hit_r)  # (N,P,E)

    entity_idx = torch.arange(E, device=device).view(1, 1, E)
    not_owner = state.prj_owner.unsqueeze(-1) != entity_idx
    can_hit_unit = alive & non_lobbed & ~wall_kill
    valid_unit_hit = unit_hit & state.ent_alive.unsqueeze(1) & not_owner & can_hit_unit.unsqueeze(-1)

    # An ordinary projectile hits the FIRST thing on its path and dies. A piercing one hits
    # EVERYTHING it overlaps and flies on -- but only once per victim, which `prj_hits` remembers
    # (as `ent_dash_hits` does for dashes); otherwise it would re-damage a target on every tick it
    # stayed inside them.
    pierce_hit = valid_unit_hit & ~state.prj_hits
    final_unit_hit = torch.where(
        pierce.unsqueeze(-1), pierce_hit, _earliest_only(valid_unit_hit, unit_hit_t),
    )
    unit_kill = final_unit_hit.any(dim=-1) & ~pierce
    state.prj_hits.copy_(state.prj_hits | final_unit_hit)

    dmg_by_unit = torch.where(final_unit_hit, state.prj_damage.unsqueeze(-1), torch.zeros_like(unit_hit_t))
    dmg_by = torch.zeros(N, E, E, device=device)
    owner_exp_e = state.prj_owner.unsqueeze(-1).expand(-1, -1, E)
    dmg_by.scatter_add_(1, owner_exp_e, dmg_by_unit)

    # SUPER-CHARGE EXCLUSION: a second attacker x victim matrix that receives everything `dmg_by`
    # does EXCEPT damage dealt by a `GADGET_SPINNER` slot; `charge_hit` is `> 0` of this one. A
    # parallel accumulator, not a subtraction afterwards, because per-projectile identity is gone
    # once the (N,P,E) contributions are scattered into (N,E,E). Filtered by KIND, not class, so
    # other artillery still charges. The spinner only damages through the blast below; the
    # unit-hit term is masked too so the rule holds if it ever gets a body.
    not_gadget = (state.prj_kind != _GADGET_SPINNER).unsqueeze(-1)  # (N,P,1)
    dmg_by_charge = torch.zeros(N, E, E, device=device)
    dmg_by_charge.scatter_add_(1, owner_exp_e, torch.where(not_gadget, dmg_by_unit, torch.zeros_like(dmg_by_unit)))

    # (4) box collision, against the boxes near each slot only. Boxes never move, so a per-tile
    # lookup (`_box_grid`) finds every box a slot can touch this tick: its path's midpoint is
    # within `params.box_reach_tiles` of any box it hits, and of any box its blast in (5) reaches
    # (config.box_reach_tiles). That is a 3x3 block at the default spec -- (N,P,9) tests instead
    # of (N,P,B). Damage goes straight into per-box accumulators, so nothing here is (N,P,B).
    reach_cells = math.ceil(params.box_reach_tiles)
    grid = _box_grid(state, cfg) if (2 * reach_cells + 1) ** 2 < B else None
    cand_slot, cand_pos, cand_alive = _box_candidates(
        state, grid, 0.5 * (old_pos + pos_after_wall), reach_cells, cfg)
    box_hit_r = state.prj_radius.unsqueeze(-1) + BOX_RADIUS  # (N,P,1)->(N,P,C)
    box_hit, box_hit_t = geo.segment_circle_hit(p0, p1, cand_pos, box_hit_r)

    # A piercing bolt ignores boxes outright: damaging them would need a second (N,P,B) hit
    # memory to stop it re-hitting a crate every tick, and CHARACTER_DETAILS describes the super
    # only in terms of players.
    can_hit_box = alive & non_lobbed & ~wall_kill & ~unit_kill & ~pierce
    valid_box_hit = box_hit & cand_alive & can_hit_box.unsqueeze(-1)
    # The earliest hit wins, ties to the lowest slot: what `_earliest_only`'s argmin over all B
    # boxes picks, from candidates that are not in slot order.
    box_hit_t = torch.where(valid_box_hit, box_hit_t, float("inf"))
    earliest = valid_box_hit & (box_hit_t == box_hit_t.amin(dim=-1, keepdim=True))
    first_box = torch.where(earliest, cand_slot, B).amin(dim=-1)  # (N,P), B = no hit
    box_kill = first_box < B

    # (N,B+1) with a sink column for "no hit". Kept apart from the blast's accumulator below so
    # the float total is always (hits) + (blasts), in that order.
    dmg_box_hit = torch.zeros(N, B + 1, device=device).scatter_add_(
        1, first_box, torch.where(box_kill, state.prj_damage, torch.zeros_like(state.prj_damage)))

    # (5) artillery detonation: passed its landing point, or ran out of range
    to_target = state.prj_target - raw_new_pos
    dot_val = (to_target * state.prj_vel).sum(dim=-1)
    arrived = dot_val <= 0
    detonate = alive & (state.prj_class == _ARTILLERY) & (arrived | (new_dist_left <= 0))

    # A shell that arrived detonates ON its stored landing point, not where the tick that carried
    # it past ended (up to one tick of travel beyond): that overshoot is most of a landing-tile
    # blast and would bias the split ring away from the shooter. A shell that ran out of range
    # short of its target detonates where it is.
    det_pos = torch.where((detonate & arrived).unsqueeze(-1), state.prj_target, raw_new_pos)

    # (5b) HAZARD ticks. A lingering sphere damages everything in it on a schedule -- the SAME
    # "everyone within prj_aoe of this point takes prj_damage" query as the detonation above, so
    # it is folded into that computation rather than given its own (N,P,E) distance pass.
    #
    # The schedule is stateless, derived from `prj_age` crossing a multiple of the interval (as
    # core/melee_sweep.py does for sub-swings): nothing to keep in sync, and no tick dropped or
    # doubled when dt does not divide the interval. At interval 2.0 / ticks 2 a sphere fires at
    # age 2.0 and 4.0 and expires on the second.
    is_hazard = alive & (state.prj_class == _HAZARD)
    haz_owner_kind = torch.gather(state.ent_kind, 1, torch.clamp(state.prj_owner, min=0, max=E - 1))
    haz_interval = torch.clamp(stats.gather_kind(params.on_hit_area_interval, haz_owner_kind), min=_EPS)
    haz_n_ticks = stats.gather_kind(params.on_hit_area_ticks, haz_owner_kind).to(new_age.dtype)
    k_now = torch.floor(new_age / haz_interval)
    k_prev = torch.floor(state.prj_age / haz_interval)
    haz_fires = is_hazard & (k_now > k_prev) & (k_now <= haz_n_ticks)
    haz_expire = is_hazard & (k_now >= haz_n_ticks)

    blast = detonate | haz_fires
    # A hazard is stationary, so raw_new_pos IS its own position; det_pos therefore already holds
    # the right point for both cases.
    ent_dist = geo.safe_norm(det_pos.unsqueeze(2) - state.ent_pos.unsqueeze(1), dim=-1)  # (N,P,E)
    # A hazard never damages its owner, nor does the gadget spinner (thrown at an enemy standing
    # on Mortis, it lands at his own feet). Other artillery still can (a Grom can blow himself
    # up), hence the exclusion is by hazard CLASS and spinner KIND, not applied to every blast.
    entity_idx_e = torch.arange(E, device=device).view(1, 1, E)
    no_self = is_hazard | (state.prj_kind == _GADGET_SPINNER)
    owner_ok = ~no_self.unsqueeze(-1) | (state.prj_owner.unsqueeze(-1) != entity_idx_e)
    # `prj_aoe > 0` is what makes `aoe_radius: 0` (Spike) mean "no damage where it lands". Without
    # it a zero-radius blast still catches anything at EXACTLY the landing point (`ent_dist <= 0`
    # at coincidence), which is reachable: a LOB bot with no aim noise lands `det_pos` on a
    # stationary target's `ent_pos` bit for bit.
    has_blast = (state.prj_aoe > 0).unsqueeze(-1)
    ent_in_aoe = (
        (ent_dist <= state.prj_aoe.unsqueeze(-1)) & has_blast & state.ent_alive.unsqueeze(1)
        & blast.unsqueeze(-1) & owner_ok
    )
    dmg_by_aoe = torch.where(ent_in_aoe, state.prj_damage.unsqueeze(-1), torch.zeros_like(ent_dist))
    dmg_by.scatter_add_(1, owner_exp_e, dmg_by_aoe)
    # The blast is the spinner's only damage path, so this is the mask that keeps it from charging
    # the super; see the parallel accumulator's note at the unit-hit scatter above.
    dmg_by_charge.scatter_add_(1, owner_exp_e, torch.where(not_gadget, dmg_by_aoe, torch.zeros_like(dmg_by_aoe)))

    # Boxes in the blast: the same nearby-box lookup as (4), around the blast point.
    cand_slot, cand_pos, cand_alive = _box_candidates(state, grid, det_pos, reach_cells, cfg)
    box_dist = geo.safe_norm(det_pos.unsqueeze(2) - cand_pos, dim=-1)  # (N,P,C)
    box_in_aoe = (
        (box_dist <= state.prj_aoe.unsqueeze(-1)) & has_blast & cand_alive & blast.unsqueeze(-1)
    )
    dmg_box_aoe = torch.where(box_in_aoe, state.prj_damage.unsqueeze(-1), torch.zeros_like(box_dist))
    dmg_box_blast = torch.zeros(N, B + 1, device=device).scatter_add_(
        1, cand_slot.reshape(N, -1), dmg_box_aoe.reshape(N, -1))
    dmg_box = dmg_box_hit[:, :B] + dmg_box_blast[:, :B]  # (N,B)

    # (6) expiry at dist_left <= 0 (PROJECTILE class only; ARTILLERY's dist_left<=0 already ->
    # detonate, and a HAZARD has no distance to run out of -- it ends on its tick count instead).
    expire = alive & non_lobbed & (new_dist_left <= 0) & ~wall_kill & ~unit_kill & ~box_kill

    final_dead = wall_kill | unit_kill | box_kill | detonate | expire | haz_expire
    new_alive = alive & ~final_dead
    final_pos = torch.where((state.prj_class == _ARTILLERY).unsqueeze(-1), det_pos, pos_after_wall)

    state.prj_pos.copy_(torch.where(alive.unsqueeze(-1), final_pos, state.prj_pos))
    state.prj_dist_left.copy_(torch.where(alive, new_dist_left, state.prj_dist_left))
    state.prj_age.copy_(torch.where(alive, new_age, state.prj_age))
    state.prj_alive.copy_(new_alive)
    # Hit memory is cleared on DEATH rather than on spawn. A slot is only reallocated after it
    # dies (alloc_slots picks from ~prj_alive) or after a reset (state.zero_), so this suffices,
    # as one masked AND over (N,P,E) instead of a spawn-time gather/scatter, and no invalid claim
    # can clobber a live projectile's memory.
    state.prj_hits.copy_(state.prj_hits & new_alive.unsqueeze(-1))

    # Which PROJECTILE-class slots ended this tick, by ANY cause -- wall, unit, box, or range
    # expiry. Captured HERE, before either spawner runs: both write `prj_class` on the slots
    # these deaths just freed, so reading it afterwards would see the NEW occupant's class.
    died_projectile = (state.prj_class == _PROJECTILE) & (wall_kill | unit_kill | box_kill | expire)

    # (7) split shards. Deliberately AFTER prj_alive is updated: the shells that just detonated
    # have freed their own slots, and letting their shards claim them is what keeps one
    # artillery shot's total slot footprint at split_count rather than 1 + split_count.
    _spawn_splits(state, detonate, det_pos, params, cfg)

    # (8) lingering spheres, same "after prj_alive" reasoning: Brock's sphere reuses the slot his
    # own rocket just freed, so one Brock shot never occupies two slots at once.
    _spawn_hazards(state, died_projectile, pos_after_wall, params, cfg)

    # LIFESTEAL: `super_heal` per PLAYER a piercing bolt connected with this tick (boxes never
    # count: `final_unit_hit` is entity-only and supers ignore boxes). Reported like damage, not
    # applied here: this function computes and the caller commits (env._projectile_phase), through
    # `combat.apply_heal`'s alive gate.
    heal_per_hit = stats.gather_kind(params.super_heal, haz_owner_kind)  # (N,P), owner's kind
    heal_amount = final_unit_hit.sum(dim=-1).to(dmg_by.dtype) * heal_per_hit * pierce.to(dmg_by.dtype)
    heal_ent = torch.zeros(N, E, device=device, dtype=dmg_by.dtype)
    heal_ent.scatter_add_(1, state.prj_owner, heal_amount)

    dmg_ent = dmg_by.sum(dim=1)
    charge_hit = dmg_by_charge > 0
    return dmg_ent, dmg_by, dmg_box, heal_ent, charge_hit
