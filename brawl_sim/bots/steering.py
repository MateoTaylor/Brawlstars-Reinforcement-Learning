"""Composable steering primitives for the bot archetypes (Steps 17-20). See
BRAWL_SIM_BUILD_PLAN.md Step 16.

Every primitive here is a pure function of plain (N,E,2)/(N,E) tensors -- none of them touch
SimState, params, or perception.select_target directly. That keeps them reusable both for
per-entity steering (archetypes call them with an entity's own position and its target's
gathered position) and for simple tests. Each primitive returns an UNNORMALIZED direction;
only combine() normalizes, once, after the weighted sum -- normalizing earlier would throw
away relative magnitude information (e.g. maintain_range's "how far outside the deadband")
that combine's weights are meant to balance against each other.

avoid_walls reuses the exact "8 fixed dir_from_bin offsets, Python for over a compile-time
constant" pattern terrain.circle_blocked already established (see the comment on
terrain._N_PROBES, which points back at this module) -- CONVENTIONS.md sanctions a fixed-count
Python loop like this since it doesn't depend on runtime tensor shapes.
"""
import torch

from ..core import geometry as geo
from ..core import terrain

_N_PROBES = 8


def seek(pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
    """(N,E,2) unnormalized direction from pos toward target_pos."""
    return target_pos - pos


def flee(pos: torch.Tensor, threat_pos: torch.Tensor) -> torch.Tensor:
    """(N,E,2) unnormalized direction from threat_pos toward pos (i.e. away from the threat)."""
    return pos - threat_pos


def strafe(pos: torch.Tensor, target_pos: torch.Tensor, sign) -> torch.Tensor:
    """(N,E,2). Direction perpendicular to the seek vector toward target_pos, scaled by
    `sign` (python +-1, or an (N,E) tensor such as Step 19's (-1)**entity_index)."""
    to_target = geo.normalize(target_pos - pos)
    perp = geo.perp(to_target)
    s = sign.unsqueeze(-1) if torch.is_tensor(sign) else sign
    return perp * s


def maintain_range(pos: torch.Tensor, target_pos: torch.Tensor, desired, deadband) -> torch.Tensor:
    """(N,E,2). Seeks (unnormalized diff toward target) when dist > desired+deadband, flees
    (negated diff) when dist < desired-deadband, zero inside the deadband. desired/deadband
    are python scalars or (N,E) tensors -- dist is deliberately kept (N,E) (no keepdim) so the
    too_far/too_close comparisons broadcast against them elementwise; keepdim's (N,E,1) would
    instead broadcast against an (N,E) desired as if E were a *second* entity axis, silently
    comparing the wrong pairs whenever E happens to also be a valid broadcast target (this bit
    a real caller -- see BRAWL_SIM_BUILD_PLAN.md Step 17's note)."""
    diff = target_pos - pos
    dist = geo.safe_norm(diff, dim=-1)
    too_far = (dist > (desired + deadband)).unsqueeze(-1)
    too_close = (dist < (desired - deadband)).unsqueeze(-1)
    return torch.where(too_far, diff, torch.where(too_close, -diff, torch.zeros_like(diff)))


def avoid_walls(pos: torch.Tensor, map_id: torch.Tensor, bank, probe_dist, cfg) -> torch.Tensor:
    """(N,E,2). Probes bank.blocks_unit at 8 fixed compass offsets scaled by probe_dist;
    returns the negated sum of each blocked probe's own direction, so a bot boxed in by walls
    on most sides is pushed toward whichever probe(s) remain open (a dead end pushes it back
    out the way it came)."""
    probe_idx = torch.arange(_N_PROBES, device=pos.device, dtype=torch.int64)
    probe_dirs = geo.dir_from_bin(probe_idx, _N_PROBES)  # (_N_PROBES, 2)
    push = torch.zeros_like(pos)
    for k in range(_N_PROBES):  # fixed compile-time constant, CONVENTIONS.md-sanctioned
        d = probe_dirs[k]
        probe_pos = pos + probe_dist * d
        blocked = terrain.sample(bank.blocks_unit, map_id, probe_pos, cfg)  # (N,E)
        push = push - torch.where(blocked.unsqueeze(-1), d.expand_as(pos), torch.zeros_like(pos))
    return push


def escape_zone(pos: torch.Tensor, zone_lo: torch.Tensor, zone_hi: torch.Tensor) -> torch.Tensor:
    """(N,E,2). Direction from pos to the nearest point inside [zone_lo, zone_hi]; zero when
    already inside. zone_lo/zone_hi are plain tensors (not read off state), matching
    perception.in_zone / nearest_safe_point (Step 15) -- core/zone.py (Step 23) is the eventual
    source of these bounds."""
    x = torch.clamp(pos[..., 0], zone_lo[..., 0], zone_hi[..., 0])
    y = torch.clamp(pos[..., 1], zone_lo[..., 1], zone_hi[..., 1])
    return torch.stack([x, y], dim=-1) - pos


def combine(*weighted) -> torch.Tensor:
    """weighted: any number of (direction (N,E,2), weight) pairs -- weight is either a python
    scalar or an (N,E) tensor. Weighted sum, then normalized once. geo.normalize clamps its
    denominator away from zero, so an all-zero sum (all weights zero, or directions that
    exactly cancel) returns the zero vector rather than NaN."""
    direction0, weight0 = weighted[0]
    w0 = weight0.unsqueeze(-1) if torch.is_tensor(weight0) else weight0
    total = direction0 * w0
    for direction, weight in weighted[1:]:
        w = weight.unsqueeze(-1) if torch.is_tensor(weight) else weight
        total = total + direction * w
    return geo.normalize(total)
