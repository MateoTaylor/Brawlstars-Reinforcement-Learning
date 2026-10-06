"""Composable steering primitives for bot movement (bots/personality.movement, bots/policy).

Each is a pure function of plain (N,E,2)/(N,E) tensors -- never SimState or params. seek, flee,
maintain_range and escape_zone return raw offsets (length = distance), strafe a unit vector. Only
combine() normalizes, once, after the weighted sum, so a seek/flee term weighs weight x distance
against the others.
"""
import torch

from ..core import geometry as geo


def seek(pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
    """(N,E,2) unnormalized direction from pos toward target_pos."""
    return target_pos - pos


def flee(pos: torch.Tensor, threat_pos: torch.Tensor) -> torch.Tensor:
    """(N,E,2) unnormalized direction from threat_pos toward pos (i.e. away from the threat)."""
    return pos - threat_pos


def strafe(pos: torch.Tensor, target_pos: torch.Tensor, sign) -> torch.Tensor:
    """(N,E,2). Direction perpendicular to the seek vector toward target_pos, scaled by
    `sign`: python +-1, or a tensor broadcastable to (N,E) such as bots/policy.strafe_sign's (E,)
    (-1)**slot."""
    to_target = geo.normalize(target_pos - pos)
    perp = geo.perp(to_target)
    s = sign.unsqueeze(-1) if torch.is_tensor(sign) else sign
    return perp * s


def maintain_range(pos: torch.Tensor, target_pos: torch.Tensor, desired, deadband,
                   max_dist=None) -> torch.Tensor:
    """(N,E,2). Seeks (unnormalized diff toward target) when dist > desired+deadband, flees
    (negated diff) when dist < desired-deadband, zero inside the deadband. desired/deadband are
    python scalars or (N,E) tensors. `dist` stays (N,E), no keepdim, on purpose: an (N,E,1) dist
    would broadcast against an (N,E) desired as if E were a second entity axis and silently
    compare the wrong pairs for some shapes instead of raising.

    `max_dist` (optional, scalar or (N,E)) also seeks whenever dist > max_dist, so the seek edge
    becomes min(desired + deadband, max_dist); the flee edge is untouched. The seek edge, not
    `desired`, is where an approaching entity stops, since inside the band only the caller's
    strafe is left and an orbit drifts outward. bots/personality.movement passes each bot's fire
    reach so no kiter parks out of its own range (user decision, 2026-09-21). If both edges hold
    at once (desired - deadband > max_dist) seeking wins and the entity jitters on max_dist, so
    callers keep desired <= max_dist, as bots/policy.targeting does."""
    diff = target_pos - pos
    dist = geo.safe_norm(diff, dim=-1)
    too_far = dist > (desired + deadband)
    if max_dist is not None:
        too_far = too_far | (dist > max_dist)
    too_far = too_far.unsqueeze(-1)
    too_close = (dist < (desired - deadband)).unsqueeze(-1)
    return torch.where(too_far, diff, torch.where(too_close, -diff, torch.zeros_like(diff)))


def escape_zone(pos: torch.Tensor, zone_lo: torch.Tensor, zone_hi: torch.Tensor) -> torch.Tensor:
    """(N,E,2). Direction from pos to the nearest point inside [zone_lo, zone_hi]; zero when
    already inside. zone_lo/zone_hi are plain tensors (not read off state), like
    perception.in_zone's."""
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
