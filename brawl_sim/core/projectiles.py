"""Projectile slot allocation, volley spawning, and the per-tick projectile pipeline. See
BRAWL_SIM_BUILD_PLAN.md Step 13.

MAX_PROJ_PER_ENTITY is a hardcoded Python int, not derived from params.proj_count.max() --
deriving it from a tensor would force a host sync (.max().item()) inside the hot path
(spawn_volley runs every tick, phase 6). It is **5** as of Step B2, matching Shelly's 5-pellet
shotgun spread, the max across brawlers.yaml's roster (it was 3 for Bo's triple-shot). Bump this
if a future archetype ever fires more than 5 projectiles per volley -- and note the failure mode
if you forget: `spawn_volley` allocates a (N,E,K) grid with K = this constant, so a kind with
proj_count > K silently fires only K of its projectiles. config.validate's
`peak_projectile_demand` check does NOT catch that (it is about buffer capacity, not volley
width), so tests/test_projectiles.py asserts the relationship directly.

MAX_SPLITS is the second slot consumer: a detonating shell hands its payload to `split_count`
shards (_spawn_splits), so the buffer has to carry volleys AND one round of shards. It is **6** as
of Spike, the widest ring in brawlers.yaml (Grom's cross is 4). config.peak_projectile_demand
budgets each kind against its OWN `split_count`, not this ceiling -- this constant only sizes the
(N,P,K) grid _spawn_splits allocates against, and config.validate rejects a `split_count` above it
rather than letting the extra arms vanish the way an over-wide volley would.

BOX_RADIUS is a filled-in gap: there's no box-collision-radius field anywhere in SimParams,
so projectile-vs-box hit testing needs *some* fixed size. 0.5 tiles (a box roughly fills one
tile) is a reasonable placeholder, consistent with D05/D06 -- exact numbers don't matter yet.
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

# constants.ProjClass values as plain ints, for comparing against the (N,P) int64 `prj_class`.
# Bare ints rather than the IntEnum members so the comparison never builds a temporary on device.
_PROJECTILE = int(ProjClass.PROJECTILE)
_ARTILLERY = int(ProjClass.ARTILLERY)
_HAZARD = int(ProjClass.HAZARD)

# The RING a detonating shell splits into: `split_count` unit vectors evenly spaced around the
# circle, starting at +x. Grom's 4 make the original cross (exactly -- see the snap below), Spike's
# 6 make a hexagonal star.
#
# WORLD-oriented, not relative to the shell's own flight direction: the arms are meant to read as a
# fixed shape on the tile grid from the bird's-eye camera, which is also what makes them
# predictable enough to dodge on sight. Two shells landing on the same tile from opposite corners
# of the map produce the same star, so "where is it safe to stand once it lands" is a property of
# the landing point alone.
#
# The COUNT is a param (`split_count`, per kind) because it is a balance knob; the DIRECTIONS are
# derived from it rather than configured, because a ring that is not evenly spaced has no
# meaningful "count" and every brawler that splits in the real game splits evenly. What stays
# structural is MAX_SPLITS, which decides tensor shapes and how much of the projectile buffer one
# shell can claim.
MAX_SPLITS = 6


def _ring(count: int) -> tuple[tuple[float, float], ...]:
    """`count` evenly spaced unit vectors from +x, zero-padded out to MAX_SPLITS.

    Values below 1e-12 are snapped to exactly 0.0. That is not cosmetic: at count 4 the raw
    `cos(pi/2)` is 6.1e-17, so an un-snapped ring would send Grom's arms off the world axes by a
    hair and quietly turn every exact assertion about his cross into an approximate one. Snapping
    makes the derived count-4 ring bit-identical to the literal `((1,0), (0,1), (-1,0), (0,-1))`
    it replaced."""
    out = []
    for k in range(MAX_SPLITS):
        if k >= count:
            out.append((0.0, 0.0))  # never spawned: alloc_slots' demand is `count`, not MAX_SPLITS
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

    Cached, and assembled from scalar fills rather than written as
    `torch.tensor(SPLIT_DIR_TABLE, device=device)`, for exactly the reason
    bots/personality._weight_table is: _spawn_splits runs every tick inside step(), and building
    a tensor from a Python tuple there copies a host buffer to the GPU on every one of them --
    a per-tick H2D transfer AND a hard synchronization, which is the one thing CONVENTIONS.md
    forbids outright. tests/test_integration.py::test_native_step_is_sync_free_on_cuda catches
    it (it did catch it -- this function exists because of that failure)."""
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
    spawn_volley, shells x shards for _spawn_splits). Every prj_ field is written, including
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
    regardless of spread). lobbed = (proj_kind == ARTILLERY_SHELL). dist_left = attack_range,
    except for a fixed-flight-time lob (proj_flight_seconds > 0), which gets exactly the
    distance to its landing point -- see the velocity comment below.
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

    # Retroactive fix (found while building Step 29's end-to-end integration tests): lead
    # prediction (geo.lead_target) and artillery's positional aim noise (aim_noise_tiles) can
    # both push aim_point past the map edge, and lobbed shells fly a straight ballistic line to
    # their target with no wall/bounds clamping at all (Step 18: they arc OVER walls by
    # design) -- an off-map target sends the shell itself off-map mid-flight, which
    # check_invariants correctly flags. Clamping into a small inset of the map (not the exact
    # edge -- detonation triggers the tick the shell passes the target, which can overshoot it
    # by up to one tick's travel) guarantees every lobbed shell's straight-line path from an
    # always-in-map origin to an in-map target stays in-map (the map is a convex rectangle).
    # Harmless for non-lobbed shots: their prj_target is stored but never read for physics.
    margin = cfg.los_step_tiles
    aim_point = torch.stack([
        torch.clamp(aim_point[..., 0], margin, cfg.map_w - margin),
        torch.clamp(aim_point[..., 1], margin, cfg.map_h - margin),
    ], dim=-1)
    # A lobbed shell's own detonation check IS "have I passed aim_point" (dot_val <= 0 below,
    # phase 9) -- its launch direction must point exactly at the (now-clamped) aim_point to stay
    # geometrically consistent, or it can fly straight past the map before ever "passing" a
    # target that's no longer where its original (pre-clamp) direction was aimed. Recomputed
    # ONLY for lobbed shots: non-lobbed aim_dir carries genuine angular firing noise
    # (bots/combat_rules' LEAD model) that is intentionally independent of aim_point and must
    # not be
    # overwritten.
    # ARTILLERY vs PROJECTILE is still decided purely by the weapon's proj_kind, exactly as the
    # `prj_lobbed` bool was -- Step C3a is a rename of the representation, not a change of rule.
    # The rule now reads out of constants.PROJ_CLASS_OF rather than comparing against one enum
    # member, because more than one weapon lobs (see that table).
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

    # Fixed flight TIME for a lobbed shell (Grom's "Watch This!"), when proj_flight_seconds > 0:
    # the shell is given whatever speed puts it on its landing point in exactly that many
    # seconds, so a 1-tile lob and a max-range lob take the same time to land. A close shot is
    # therefore SLOW and trivially readable while a long one is fast, which is the opposite of
    # the constant-speed model (where closing distance was the way to shorten your reaction
    # window). proj_speed is unused for these shells -- for artillery it now describes the split
    # shards instead (_spawn_splits), which are ordinary constant-speed projectiles.
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

    One PIERCING bolt per firing entity, using the `super_*` stat block rather than the ordinary
    weapon stats: its own range, speed, damage and radius. Separate from `spawn_volley` because a
    super shares none of that function's fan/spread/lob machinery -- it is always exactly one
    projectile in one direction -- and folding it in would have meant branching every line of it.

    **Not hero-specific.** `fire_mask` is (N,E) and every stat is gathered per-kind, so a bot
    firing a super needs no change here at all (Step D2 / bot_overhaul.md D2).
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


def _spawn_splits(state, detonate: torch.Tensor, det_pos: torch.Tensor, params, cfg) -> None:
    """detonate: (N,P) bool -- shells that blew up THIS tick. det_pos: (N,P,2) where each did.
    MUTATES: all prj_*. Every detonating shell whose owner has split_distance/
    split_damage_fraction/split_count set hands `split_count` shards to free slots: ordinary
    non-lobbed projectiles (so walls, units and boxes all stop them) flying the evenly spaced ring
    of SPLIT_DIR_TABLE for split_distance tiles and carrying split_damage_fraction of the shell's
    damage each. An archetype with those params unset (every non-splitting kind, and any config
    predating them) detonates exactly as before -- demand is 0 and this whole function is a no-op
    write.

    MUST be called after state.prj_alive reflects the detonations, so shards can reuse the
    slots their own shells just freed. A shell whose shards don't fit in the buffer simply
    loses them (alloc_slots' ok=False), the same way a volley into a full buffer is dropped.
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
    # Clamped only as a guard against an out-of-range index into the ring table below.
    # config.validate already rejects a split_count above MAX_SPLITS at construction time, so this
    # never binds in practice -- but an out-of-bounds gather is a device-side assert that takes the
    # whole process down on CUDA, which is a bad way to learn about a typo in a YAML file.
    split_count = torch.clamp(stats.gather_kind(params.split_count, owner_kind), 0, K)

    # Shard speed is derived from how long the arms should take to FINISH, not configured directly
    # (Step C4). CHARACTER_DETAILS specifies Grom's split strikes in seconds ("~0.5 s to finish
    # their paths"), so expressing that as `split_distance / split_seconds` means the config states
    # the observable quantity and the speed falls out, rather than making whoever edits it solve
    # for a speed that produces the right duration.
    #
    # `split_seconds: 0` falls back to `proj_speed`, which is what every kind predating this field
    # resolves to and is the constant-speed behavior shards had before.
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
    idx, ok = alloc_slots(state.prj_alive, demand, K)  # (N,P,K)

    # (N,P,K,2): each shell gets the ring for its OWN split_count. Arms past that count are
    # (0,0) here and are never written -- alloc_slots' `ok` is False for them, since demand is
    # `split_count` rather than K.
    dirs = _split_dir_table(device)[split_count]

    # Shards start at the RIM of the blast that produced them, not at its centre. Anything the
    # detonation damaged is within prj_aoe of det_pos, and a shard that starts (unit_radius +
    # prj_radius) beyond that rim and flies straight outward can never come within hit range of
    # such a target: every point of its path is at least |start| - |target| >= hit_r away. So the
    # attack is a clean partition of the ground -- the landing disc OR one of the four arms, never
    # both -- instead of a centre pixel where all five payloads stack.
    #
    # **This geometry got MORE load-bearing in Step B1, not less.** It used to prevent a 3200 + 1600
    # overlap, which merely hit hard; since split_damage_fraction became 1.0 the shards deal the
    # shell's FULL damage, so the same overlap would now be 2080 + 2080 = 4160 from a single shell --
    # two thirds of a brawler's health on one landing tile. The partition is what keeps Grom's cross
    # "2080 wherever it catches you" rather than a centre one-shot.
    #
    # A shell with NO blast (`aoe_radius: 0` -- Spike) keeps the same formula, and the result is not
    # an accident worth "fixing" to 0. The ring then sits at exactly `unit_radius + prj_radius`,
    # which IS the shard hit radius, so it stays the smallest ring that does not put every arm on
    # top of a target standing on the landing point. Setting it to 0 there would make all
    # `split_count` shards spawn inside that target and all connect -- six times the shard damage
    # from one shell, a far worse outcome than the alternative it buys: a shot landing within a few
    # thousandths of a tile of dead centre catches nobody. That case needs the landing point to
    # coincide with a body to float precision, which LOB's positional aim noise makes vanishingly
    # rare, and it degrades toward MORE damage as the shot gets worse -- one arm at 0.7 tiles off,
    # three at 0.3. See tests/test_spike.py for the measured shard-count profile.
    rim = state.prj_aoe + params.unit_radius.view(N, 1) + state.prj_radius + _EPS  # (N,P)
    start = det_pos.unsqueeze(2) + dirs * rim.view(N, P, 1, 1)
    # A shell landing near the map edge would otherwise put its outward shards outside the map
    # entirely, which check_invariants flags. Clamping into the same inset spawn_volley clamps
    # landing points to keeps them legal; the border is WALL on every map (maps/loader.
    # validate_map), so a clamped shard dies on its first march anyway.
    margin = cfg.los_step_tiles
    start = torch.stack([
        torch.clamp(start[..., 0], margin, cfg.map_w - margin),
        torch.clamp(start[..., 1], margin, cfg.map_h - margin),
    ], dim=-1)

    zeros = torch.zeros(N, P, K, device=device)
    _write_slots(
        state, idx.reshape(N, P * K), ok.reshape(N, P * K),
        pos=start.reshape(N, P * K, 2),
        vel=(dirs * shard_speed.view(N, P, 1, 1)).reshape(N, P * K, 2),
        # Unread (shards are not lobbed), but written anyway so the slot is fully defined.
        target=start.reshape(N, P * K, 2),
        dist_left=split_distance.unsqueeze(-1).expand(N, P, K).reshape(N, P * K),
        damage=(state.prj_damage * split_fraction).unsqueeze(-1).expand(N, P, K).reshape(N, P * K),
        radius=state.prj_radius.unsqueeze(-1).expand(N, P, K).reshape(N, P * K),
        aoe=zeros.reshape(N, P * K),  # shards hit what they touch; only the shell has a blast
        age=zeros.reshape(N, P * K),
        owner=state.prj_owner.unsqueeze(-1).expand(N, P, K).reshape(N, P * K),
        # Same Proj kind as the shell that spawned them: they are the same weapon, and both the
        # observation (projectiles.lobbed) and the viewer already distinguish arc from shard
        # without needing a fifth enum member (which would widen every kind_onehot in the obs).
        kind=state.prj_kind.unsqueeze(-1).expand(N, P, K).reshape(N, P * K),
        cls=torch.full((N, P * K), int(ProjClass.PROJECTILE), dtype=torch.int64, device=device),
        pierce=torch.zeros(N, P * K, dtype=torch.bool, device=device),
        alive=torch.ones(N, P * K, dtype=torch.bool, device=device),
    )


def _spawn_hazards(state, died: torch.Tensor, death_pos: torch.Tensor, params, cfg) -> None:
    """died: (N,P) bool -- PROJECTILE-class projectiles that ended this tick. death_pos: (N,P,2).
    MUTATES: all prj_*. Every one whose OWNER's kind sets `on_hit_area_radius > 0` leaves a
    HAZARD-class projectile behind (Brock's lingering sphere, Step C3b).

    Same shape as `_spawn_splits`, and for the same reasons: stats come from the owner's kind via
    `prj_owner` + `ent_kind` (a projectile carries no back-reference to its archetype, and
    `ent_kind` outlives the entity's death, so a Brock who dies mid-flight still leaves a sphere);
    slot allocation reuses `alloc_slots`; and it MUST run after `prj_alive` reflects the deaths so
    the sphere can claim the slot its own rocket just freed.

    **Spawns on ANY death, not only a unit hit** (bot_overhaul.md assumption A2): wall, unit, box,
    or running out of range. A 5.33 tiles/s rocket at 8 tiles range expires without hitting
    anything fairly often, and a rocket that detonates on a wall still leaves a puddle in the real
    game -- restricting it to unit hits would remove most of Brock's area denial.

    `death_pos` is the projectile's end-of-tick position rather than the exact segment/circle
    intersection. At Brock's speed that is at most 0.27 tiles of error against a 0.75-tile sphere,
    and it is exactly where the projectile is drawn when it dies, so the sphere appears where the
    player saw the rocket land.
    """
    N, P = state.prj_pos.shape[:2]
    E = state.ent_pos.shape[1]
    device = death_pos.device

    owner_kind = torch.gather(state.ent_kind, 1, torch.clamp(state.prj_owner, min=0, max=E - 1))
    radius = stats.gather_kind(params.on_hit_area_radius, owner_kind)
    # A FRACTION of the projectile's own damage, exactly like `split_damage_fraction` -- never an
    # absolute figure.
    #
    # `prj_damage` was set at spawn from `stats.effective_damage`, so it already carries the cube
    # bonus, `enemy_damage_mult`, and the curriculum's per-tier damage multiplier
    # (training/curriculum.py scales `base_damage`). Reading a raw `on_hit_area_damage` param
    # instead made the sphere ignore ALL THREE: an "easy" tier halved Brock's rocket and left his
    # sphere at full strength, and power cubes doubled the rocket while the sphere stayed flat.
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
        dist_left=zeros,                           # never expires by distance -- see step_projectiles
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


def _earliest_only(valid_hit: torch.Tensor, hit_t: torch.Tensor) -> torch.Tensor:
    """valid_hit/hit_t: (..., M). Keeps only the smallest-t True entry along the last dim."""
    t_for_argmin = torch.where(valid_hit, hit_t, torch.full_like(hit_t, float("inf")))
    earliest = torch.argmin(t_for_argmin, dim=-1, keepdim=True)
    has_any = valid_hit.any(dim=-1, keepdim=True)
    earliest_mask = torch.zeros_like(valid_hit).scatter_(-1, earliest, has_any)
    return valid_hit & earliest_mask


def step_projectiles(state, bank, params, cfg):
    """MUTATES: all prj_*. Returns (dmg_ent (N,E), dmg_by (N,E,E), dmg_box (N,B),
    heal_ent (N,E)) -- all four reported here, applied by combat.apply_damage /
    combat.apply_heal (Step 14 / Step D2)."""
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

    # (2) wall collision for non-lobbed via march, kill at hit point
    # Retroactive fix (found while building Step 29's end-to-end integration tests, same root
    # cause as hero.start_dash's dash-clip fix): march() samples every los_step_tiles, so
    # wall_hit_pos is the first BLOCKED sample, not the true wall boundary -- it can land AT or
    # (with float rounding) fractionally past the wall tile itself, which check_invariants
    # correctly flags as out-of-bounds when that wall is the map's outer border. Backing the
    # kill position off by one los_step_tiles along the travel direction guarantees it lands on
    # the known-safe side of that gap.
    move_dir = geo.normalize(step_delta)
    wall_hit, wall_hit_pos, _ = terrain.march(bank.blocks_proj, state.map_id, old_pos, move_dir, step_dist, cfg)
    safe_wall_hit_pos = wall_hit_pos - move_dir * cfg.los_step_tiles
    # A PIERCING bolt (Step D2's super) passes through walls: `wall_hit` is still computed --
    # it costs the same march either way -- but it neither kills the projectile nor clips its
    # position, so the bolt flies on with `raw_new_pos`.
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
    # EVERYTHING it overlaps and flies on -- but only once per victim, which is what `prj_hits`
    # remembers. Without that memory a bolt would re-damage the same target on every tick it
    # stayed inside them: at 12 tiles/s and a 0.70 radius that is ~2 ticks, i.e. double damage,
    # and a slower bolt would be far worse. Exactly the problem `ent_dash_hits` solves for dashes.
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

    # (4) box collision over (N,P,B)
    box_pos_b = state.box_pos.unsqueeze(1)  # (N,1,B,2)
    box_hit_r = state.prj_radius.unsqueeze(-1) + BOX_RADIUS  # (N,P,1)->(N,P,B)
    box_hit, box_hit_t = geo.segment_circle_hit(p0, p1, box_pos_b, box_hit_r)

    # A piercing bolt ignores boxes outright. Letting it damage them would need a second
    # (N,P,B) hit-memory buffer to stop it re-hitting the same crate every tick, and
    # CHARACTER_DETAILS only describes the super in terms of players ('deals 1800 damage to
    # whatever it hits... heals for every hit to another player').
    can_hit_box = alive & non_lobbed & ~wall_kill & ~unit_kill & ~pierce
    valid_box_hit = box_hit & state.box_alive.unsqueeze(1) & can_hit_box.unsqueeze(-1)
    final_box_hit = _earliest_only(valid_box_hit, box_hit_t)
    box_kill = final_box_hit.any(dim=-1)

    dmg_box_unit = torch.where(final_box_hit, state.prj_damage.unsqueeze(-1), torch.zeros_like(box_hit_t))
    dmg_box = dmg_box_unit.sum(dim=1)  # (N,B)

    # (5) artillery detonation: passed its landing point, or ran out of range
    to_target = state.prj_target - raw_new_pos
    dot_val = (to_target * state.prj_vel).sum(dim=-1)
    arrived = dot_val <= 0
    detonate = alive & (state.prj_class == _ARTILLERY) & (arrived | (new_dist_left <= 0))

    # A shell that arrived detonates ON its stored landing point, not wherever the tick that
    # carried it past that point happened to end: detonation is only ever detected AFTER the
    # overshoot, up to one tick of travel (~0.4 tiles at max range) beyond the target. That was
    # invisible against a 1.5-tile blast, but it is most of a landing-TILE blast, and it would
    # make the split cross systematically lopsided away from the shooter. The dist_left branch
    # is left alone -- a shell that ran out of range short of its target genuinely did detonate
    # where it is, not where it was aimed.
    det_pos = torch.where((detonate & arrived).unsqueeze(-1), state.prj_target, raw_new_pos)

    # (5b) HAZARD ticks (Step C3b). A lingering sphere damages everything standing in it on a
    # schedule, which is the SAME "everyone within prj_aoe of this point takes prj_damage" query
    # the detonation above already performs -- so it is folded into that one computation rather
    # than given its own (N,P,E) distance pass. The extra cost is two boolean ops.
    #
    # The schedule is stateless, derived from `prj_age` crossing a multiple of the interval, the
    # same trick core/melee_sweep.py uses for Buzz's sub-swings: nothing to keep in sync, and no
    # tick dropped or doubled when dt does not divide the interval. At interval 2.0 / ticks 2 a
    # sphere fires at age 2.0 and 4.0 and expires on the second one (bot_overhaul.md D3).
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
    # D4: a hazard never damages its owner. Artillery detonations still can (a Grom can blow
    # himself up), which is pre-existing behavior this step deliberately leaves alone -- hence the
    # exclusion is conditioned on being a hazard rather than applied to every blast.
    entity_idx_e = torch.arange(E, device=device).view(1, 1, E)
    hazard_owner_ok = ~is_hazard.unsqueeze(-1) | (state.prj_owner.unsqueeze(-1) != entity_idx_e)
    # `prj_aoe > 0` is what makes "this shell does no damage where it lands" expressible in YAML
    # as `aoe_radius: 0` (Spike). Without it a zero-radius blast still catches anything at
    # EXACTLY the landing point, since `ent_dist <= 0` is true at coincidence -- and that is a
    # reachable state, not a float curiosity: a LOB bot with no aim noise aims at a stationary
    # target's exact position, so `det_pos` lands on `ent_pos` bit for bit and the shell that is
    # supposed to be harmless deals full damage to the one target it was aimed at.
    has_blast = (state.prj_aoe > 0).unsqueeze(-1)
    ent_in_aoe = (
        (ent_dist <= state.prj_aoe.unsqueeze(-1)) & has_blast & state.ent_alive.unsqueeze(1)
        & blast.unsqueeze(-1) & hazard_owner_ok
    )
    dmg_by_aoe = torch.where(ent_in_aoe, state.prj_damage.unsqueeze(-1), torch.zeros_like(ent_dist))
    dmg_by.scatter_add_(1, owner_exp_e, dmg_by_aoe)

    box_dist = geo.safe_norm(det_pos.unsqueeze(2) - state.box_pos.unsqueeze(1), dim=-1)  # (N,P,B)
    box_in_aoe = (
        (box_dist <= state.prj_aoe.unsqueeze(-1)) & has_blast & state.box_alive.unsqueeze(1)
        & blast.unsqueeze(-1)
    )
    dmg_box_aoe = torch.where(box_in_aoe, state.prj_damage.unsqueeze(-1), torch.zeros_like(box_dist))
    dmg_box = dmg_box + dmg_box_aoe.sum(dim=1)

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
    # Hit memory is cleared on DEATH rather than on spawn. A slot is only ever reallocated
    # after it dies (alloc_slots picks from ~prj_alive) or after a reset (state.zero_), so
    # this is sufficient -- and it is a single masked AND over (N,P,E) rather than the
    # gather/scatter a spawn-time clear would need, with no chance of an invalid claim
    # clobbering a live projectile's memory.
    state.prj_hits.copy_(state.prj_hits & new_alive.unsqueeze(-1))

    # Which PROJECTILE-class slots ended this tick, by ANY cause -- wall, unit, box, or range
    # expiry (assumption A2). **Captured HERE, before either spawner runs**: both write through
    # `_write_slots`, which sets `prj_class` on the slots these deaths just freed, so reading the
    # class afterwards would see the NEW occupant's class and mis-attribute the death.
    died_projectile = (state.prj_class == _PROJECTILE) & (wall_kill | unit_kill | box_kill | expire)

    # (7) the cross. Deliberately AFTER prj_alive is updated: the shells that just detonated
    # have freed their own slots, and letting their shards claim them is what keeps one
    # artillery shot's total slot footprint at split_count rather than 1 + split_count.
    _spawn_splits(state, detonate, det_pos, params, cfg)

    # (8) lingering spheres, same "after prj_alive" reasoning: Brock's sphere reuses the slot his
    # own rocket just freed, so one Brock shot never occupies two slots at once.
    _spawn_hazards(state, died_projectile, pos_after_wall, params, cfg)

    # LIFESTEAL (Step D2). `super_heal` per PLAYER a piercing bolt connected with this tick --
    # boxes contribute nothing, which falls out for free since `final_unit_hit` is entity-only and
    # supers do not interact with boxes at all.
    #
    # Reported like damage rather than applied here: this function's contract is that it computes
    # and the caller commits (see env._projectile_phase), and healing has to respect the same
    # "never touch a dead entity" rule that `combat.apply_damage` centralises.
    heal_per_hit = stats.gather_kind(params.super_heal, haz_owner_kind)  # (N,P), owner's kind
    heal_amount = final_unit_hit.sum(dim=-1).to(dmg_by.dtype) * heal_per_hit * pierce.to(dmg_by.dtype)
    heal_ent = torch.zeros(N, E, device=device, dtype=dmg_by.dtype)
    heal_ent.scatter_add_(1, state.prj_owner, heal_amount)

    dmg_ent = dmg_by.sum(dim=1)
    return dmg_ent, dmg_by, dmg_box, heal_ent
