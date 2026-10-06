"""Terrain queries against a MapBank. Generic geometric primitives -- this module doesn't know
about bots or targeting, only terrain. Point queries gather-index the bank's (M, H, W) tensors
directly (`bank_mask[map_id, iy, ix]`); nothing here ever materializes a per-env or per-map
(H, W) slice.

line_of_sight is the one exception to "generic": it hardcodes bank.blocks_proj as the blocking
mask. Only WALL blocks shots (constants.TILE_BLOCKS_PROJ), so "is there a clear physical shot"
and "is there a clear line of sight" are the same question, and every caller that needs the
physical-LOS answer gets it here: melee hit validation (combat.melee_hitscan), the bots' fire
gate and loot-box shot (bots/perception.target_los, bots/policy), and obs["visibility"]["los"]
(bots/perception.raw_los).
"""
import math

import torch

from . import geometry as geo


def _broadcast_map_id(map_id: torch.Tensor, leading_shape: torch.Size) -> torch.Tensor:
    """map_id's own shape must be a prefix of leading_shape (e.g. map_id=(N,) against a
    per-entity leading_shape=(N,E)) -- trailing dims are broadcast in."""
    extra = len(leading_shape) - map_id.dim()
    if extra > 0:
        map_id = map_id.reshape(map_id.shape + (1,) * extra)
    return map_id.expand(leading_shape)


def to_tile(pos: torch.Tensor):
    ix = torch.floor(pos[..., 0]).to(torch.int64)
    iy = torch.floor(pos[..., 1]).to(torch.int64)
    return ix, iy


def oob(ix: torch.Tensor, iy: torch.Tensor, cfg) -> torch.Tensor:
    return (ix < 0) | (ix >= cfg.map_w) | (iy < 0) | (iy >= cfg.map_h)


def sample(bank_mask: torch.Tensor, map_id: torch.Tensor, pos: torch.Tensor, cfg) -> torch.Tensor:
    """bank_mask: (M, H, W) bool, e.g. bank.blocks_unit. OOB positions return True (blocked)."""
    map_id = _broadcast_map_id(map_id, pos.shape[:-1])
    ix, iy = to_tile(pos)
    out = oob(ix, iy, cfg)
    ix_safe = torch.clamp(ix, 0, cfg.map_w - 1)
    iy_safe = torch.clamp(iy, 0, cfg.map_h - 1)
    return out | bank_mask[map_id, iy_safe, ix_safe]


# 8 fixed probe directions (bin 0 == angle 0).
_N_PROBES = 8


def circle_blocked(
    bank_mask: torch.Tensor, map_id: torch.Tensor, pos: torch.Tensor, radius: torch.Tensor, cfg,
) -> torch.Tensor:
    map_id = _broadcast_map_id(map_id, pos.shape[:-1])
    blocked = sample(bank_mask, map_id, pos, cfg)
    probe_idx = torch.arange(_N_PROBES, device=pos.device, dtype=torch.int64)
    probe_dirs = geo.dir_from_bin(probe_idx, _N_PROBES)  # (_N_PROBES, 2)
    for k in range(_N_PROBES):  # fixed compile-time constant, CONVENTIONS.md-sanctioned
        probe_pos = pos + radius.unsqueeze(-1) * probe_dirs[k]
        blocked = blocked | sample(bank_mask, map_id, probe_pos, cfg)
    return blocked


def resolve_move(
    bank_mask: torch.Tensor, map_id: torch.Tensor, pos: torch.Tensor, delta: torch.Tensor,
    radius: torch.Tensor, cfg,
) -> torch.Tensor:
    """Axis-separated sliding: try x alone, keep the move if legal; then y alone, from
    whatever x produced. Branch-free -- a blocked axis just keeps the old coordinate."""
    map_id = _broadcast_map_id(map_id, pos.shape[:-1])

    pos_x = torch.stack([pos[..., 0] + delta[..., 0], pos[..., 1]], dim=-1)
    blocked_x = circle_blocked(bank_mask, map_id, pos_x, radius, cfg)
    pos_after_x = torch.where(blocked_x.unsqueeze(-1), pos, pos_x)

    pos_y = torch.stack([pos_after_x[..., 0], pos_after_x[..., 1] + delta[..., 1]], dim=-1)
    blocked_y = circle_blocked(bank_mask, map_id, pos_y, radius, cfg)
    pos_after_y = torch.where(blocked_y.unsqueeze(-1), pos_after_x, pos_y)

    return pos_after_y


def ray_steps_for(max_tiles: float, cfg) -> int:
    """How many fixed samples a march needs to cover `max_tiles` tiles. At least 1 -- march's
    tensor build needs a non-empty sample axis, and the endpoint sample carries the answer for a
    degenerate budget anyway."""
    return max(1, int(math.ceil(max_tiles / cfg.los_step_tiles)))


def march(
    bank_mask: torch.Tensor, map_id: torch.Tensor, p0: torch.Tensor, dir: torch.Tensor,
    max_dist: torch.Tensor, cfg, max_tiles: float | None = None,
):
    """Uniform march: fixed samples at cfg.los_step_tiles spacing, masked beyond max_dist, PLUS
    one extra sample exactly at max_dist itself. Returns (hit, hit_pos, hit_t); hit_pos/hit_t are
    only meaningful where hit is True (a miss leaves them at the first sample point / one step,
    harmlessly).

    `max_tiles` is a per-call RAY BUDGET: the number of fixed samples becomes
    `ceil(max_tiles / los_step_tiles)` instead of `cfg.ray_steps` (`None`, the default).

    **The budget must be a Python scalar, not a tensor** -- it sizes a tensor dimension, so a
    tensor here would force a host sync in the hot path (CONVENTIONS.md). Callers that want to
    bound a march by a per-env/per-kind stat must derive a STATIC upper bound over that stat's
    whole configured range instead, as config.cone_ray_tiles, attack_ray_tiles, dash_ray_tiles
    and shot_step_tiles do. Why budget at all: every one of the `cfg.ray_steps` samples (48 at
    the defaults) is computed and gathered before `in_range` masks the ones past `max_dist`, so
    a short ray -- a melee cone needs 6 -- over a dense (N,E,E) pair matrix pays for all 48.

    **Passing a budget SHORTER than some element's own max_dist is legal but changes that
    element's answer**: samples between the budget and max_dist are never taken (only the
    endpoint is), so a wall there is not seen and `hit` comes back False. That is only sound when
    the caller independently discards those elements -- `combat.melee_hitscan` does (anything
    past the cone radius fails `in_cone`, which is ANDed with the LOS result). Do not pass a
    budget you have not checked that against.

    The endpoint sample is what lets a max_dist SHORTER than one los_step_tiles see a wall at
    all: `in_range` masks every fixed sample out, so without it projectiles.step_projectiles'
    per-tick wall check would miss for any shot slower than los_step_tiles/dt tiles/second, and
    that shot would fly through walls. It wins the "first hit" pick only when no fixed sample
    hit."""
    map_id = _broadcast_map_id(map_id, p0.shape[:-1])
    dir = geo.normalize(dir)
    steps = cfg.ray_steps if max_tiles is None else ray_steps_for(max_tiles, cfg)
    leading_shape = max_dist.shape

    fixed_dists = torch.arange(1, steps + 1, device=p0.device, dtype=p0.dtype) * cfg.los_step_tiles  # (steps,)
    fixed_dists = fixed_dists.view((1,) * len(leading_shape) + (steps,)).expand(*leading_shape, steps)
    max_dist_exp = max_dist.unsqueeze(-1)  # (..., 1)
    all_dists = torch.cat([fixed_dists, max_dist_exp], dim=-1)  # (..., steps+1)

    pts = p0.unsqueeze(-2) + dir.unsqueeze(-2) * all_dists.unsqueeze(-1)  # (..., steps+1, 2)

    blocked = sample(bank_mask, map_id, pts, cfg)  # (..., steps+1)
    in_range = all_dists <= max_dist_exp  # (..., steps+1); trivially True for the endpoint
    effective_blocked = blocked & in_range

    hit = effective_blocked.any(dim=-1)
    first_idx = torch.argmax(effective_blocked.to(torch.int64), dim=-1)  # first True index

    hit_t = torch.gather(all_dists, -1, first_idx.unsqueeze(-1)).squeeze(-1)
    idx_expand = first_idx.unsqueeze(-1).unsqueeze(-1).expand(*first_idx.shape, 1, 2)
    hit_pos = torch.gather(pts, -2, idx_expand).squeeze(-2)

    return hit, hit_pos, hit_t


# Bisection halvings `body_travel` spends inside the sample step where the body first touches: 4
# turn a 0.5-tile step into 1/32 of a tile, well under anything the policy or the camera resolves.
_BODY_REFINE_STEPS = 4


def body_travel(
    bank_mask: torch.Tensor, map_id: torch.Tensor, p0: torch.Tensor, dir: torch.Tensor,
    max_dist: torch.Tensor, radius: torch.Tensor, cfg, max_tiles: float | None = None,
    clearance: float = 0.0,
) -> torch.Tensor:
    """How far a body of `radius` can travel from `p0` along `dir` before it first touches
    `bank_mask`, capped at `max_dist`. Shapes as `march`, with `radius` broadcasting against
    `max_dist`; returns distances shaped like `max_dist`.

    The body test is `circle_blocked`, the one walking uses, taken at `p0`, every
    `los_step_tiles` along the line and at `max_dist` itself; inside the step where it first
    fails, `_BODY_REFINE_STEPS` bisections close in on the contact. The answer is the last
    distance tested clear (at a contact, backed off by `clearance` when that point is also
    clear), so a body moved there is never inside a wall and `resolve_move` can always walk it
    away. An unobstructed line returns `max_dist` exactly.

    A body that starts overlapping stays where it is (0). Nothing in the sim makes one: spawns
    are tile centres, and walking and dashing both refuse to end inside a wall. Letting one
    travel until it cleared would let it pass through a thin wall.

    `max_tiles` is a ray budget with `march`'s rules: a Python scalar that bounds `max_dist`
    over its whole configured range. Samples past it are never taken, so a longer `max_dist`
    is checked only at its endpoint.
    """
    lead = max_dist.shape
    map_id = _broadcast_map_id(map_id, lead)
    dir = geo.normalize(dir)
    steps = cfg.ray_steps if max_tiles is None else ray_steps_for(max_tiles, cfg)
    r = radius.expand(lead)

    def clear_at(dist):                                   # (..., S) distances -> (..., S) clear
        pts = p0.unsqueeze(-2) + dir.unsqueeze(-2) * dist.unsqueeze(-1)
        rr = r.unsqueeze(-1).expand(dist.shape)
        return ~circle_blocked(bank_mask, map_id.unsqueeze(-1).expand(dist.shape), pts, rr, cfg)

    fixed = torch.arange(0, steps + 1, device=p0.device, dtype=p0.dtype) * cfg.los_step_tiles
    fixed = fixed.view((1,) * len(lead) + (steps + 1,)).expand(*lead, steps + 1)
    dists = torch.cat([torch.minimum(fixed, max_dist.unsqueeze(-1)), max_dist.unsqueeze(-1)], dim=-1)
    clear = clear_at(dists)                               # sample 0 is the start itself

    # The first blocked sample after the start; every sample before it is clear.
    blocked = ~clear[..., 1:]
    hit = blocked.any(dim=-1)
    first = torch.argmax(blocked.to(torch.int64), dim=-1).unsqueeze(-1) + 1
    lo = torch.gather(dists, -1, first - 1).squeeze(-1)
    hi = torch.gather(dists, -1, first).squeeze(-1)
    for _ in range(_BODY_REFINE_STEPS):                   # fixed count: branch-free, no host sync
        mid = 0.5 * (lo + hi)
        ok = clear_at(mid.unsqueeze(-1)).squeeze(-1)
        lo = torch.where(ok, mid, lo)
        hi = torch.where(ok, hi, mid)

    travel = torch.where(hit, lo, max_dist)
    travel = torch.where(clear[..., 0], travel, torch.zeros_like(travel))
    if clearance > 0.0:
        backed = torch.clamp(travel - clearance, min=0.0)
        travel = torch.where(hit & clear_at(backed.unsqueeze(-1)).squeeze(-1), backed, travel)
    return travel


def line_of_sight(
    bank, map_id: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor, cfg,
    max_tiles: float | None = None,
) -> torch.Tensor:
    """`max_tiles` is march's per-call ray budget -- see its docstring, including the warning
    about what a budget shorter than the actual p0->p1 distance does to the answer."""
    delta = p1 - p0
    distance = geo.safe_norm(delta, dim=-1)
    hit, _, _ = march(bank.blocks_proj, map_id, p0, delta, distance, cfg, max_tiles=max_tiles)
    return ~hit
