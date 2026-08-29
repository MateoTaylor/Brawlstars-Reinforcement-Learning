"""Terrain queries against a MapBank (Step 6). Generic geometric primitives -- this module
doesn't know about bots or targeting, only terrain. Point queries gather-index the bank's
(M, H, W) tensors directly (`bank_mask[map_id, iy, ix]`); nothing here ever materializes a
per-env or per-map (H, W) slice.

line_of_sight is the one exception to "generic": it hardcodes bank.blocks_proj as the blocking
mask, per Notice 4 -- with only WALL left opaque (fence no longer blocks shots), "is there a
clear physical shot" and "is there a clear line of sight" are the same question, and this is
where every caller that needs the physical-LOS answer (melee hit validation, sniper/rifle
fire-gates, obs["visibility"]["los"]) gets it from.
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


# 8 fixed probe directions (bin 0 == angle 0), same "small fixed probe count" pattern as
# bots/steering.avoid_walls' 8 preallocated probe offsets (Step 16).
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
    `ceil(max_tiles / los_step_tiles)` instead of `cfg.ray_steps`. `None` (the default) keeps the
    full `cfg.ray_steps` budget and is bit-identical to this function before the argument existed.

    **The budget must be a Python scalar, not a tensor** -- it sizes a tensor dimension, so a
    tensor here would force a host sync in the hot path (CONVENTIONS.md). Callers that want to
    bound a march by a per-env/per-kind stat must derive a STATIC upper bound over that stat's
    whole configured range instead; see `config.cone_ray_tiles` for the one that exists today.

    **Why this matters.** `cfg.ray_steps` is `ceil(max_ray_tiles / los_step_tiles)` = 48 at the
    defaults, and every one of those 48 sample points is COMPUTED and gathered before `in_range`
    masks the ones past `max_dist`. A melee cone spans ~3 tiles and needs 6. Paying 48 for it,
    over a dense (N,E,E) pair matrix, every tick, was measured at 4.15 ms/tick at n_envs=1024
    versus 1.95 ms with a 6-step budget -- see bot_overhaul.md Step A1.

    **Passing a budget SHORTER than some element's own max_dist is legal but changes that
    element's answer**: samples past the budget are never taken, so a wall beyond it is not seen
    and `hit` comes back False. That is only sound when the caller independently discards those
    elements -- `combat.melee_hitscan` does (anything past the cone radius fails `in_cone`, which
    is ANDed with the LOS result). Do not pass a budget you have not checked that against.

    The extra endpoint sample is a retroactive fix (found while building Step 29's end-to-end
    integration tests): without it, any call whose max_dist is SMALLER than one los_step_tiles
    (e.g. projectiles.step_projectiles' per-tick wall check, for any projectile slower than
    los_step_tiles/dt tiles/second -- 10 tiles/s at the defaults) has every fixed sample masked
    out by `in_range` before a single one is ever checked, so march always reports a miss no
    matter what's actually there. A projectile below that speed threshold could fly straight
    through a wall, undetected, forever. The extra sample is harmless for the long-range LOS
    case (line_of_sight, hero.start_dash's dash clip): it only ever wins the "first hit" pick
    when none of the regular fixed samples did, i.e. exactly when it's needed."""
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
