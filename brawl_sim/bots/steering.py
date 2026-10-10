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


def range_band(pos: torch.Tensor, target_pos: torch.Tensor, desired, deadband, max_dist=None):
    """(too_far (N,E) bool, too_close (N,E) bool): which side of maintain_range's band each entity
    is on, with seeking winning where both edges hold (see maintain_range). Shared so that a
    caller choosing a path goal by the side (bots/personality.movement) cannot disagree with the
    steering it feeds."""
    dist = geo.safe_norm(target_pos - pos, dim=-1)
    too_far = dist > (desired + deadband)
    if max_dist is not None:
        too_far = too_far | (dist > max_dist)
    too_close = (dist < (desired - deadband)) & ~too_far
    return too_far, too_close


def maintain_range(pos: torch.Tensor, target_pos: torch.Tensor, desired, deadband,
                   max_dist=None, approach=None, retreat=None) -> torch.Tensor:
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
    callers keep desired <= max_dist, as bots/policy.targeting does.

    `approach` (optional, (N,E,2)) replaces the seek offset `target_pos - pos` while too far, and
    `retreat` (optional, (N,E,2)) the flee offset `pos - target_pos` while too close:
    bots/personality.movement passes bots/policy.path_toward's offsets under `bots.nav`, so a
    kiter closes along a walkable path and backs off along one (user decision, 2026-10-07)."""
    diff = target_pos - pos
    too_far, too_close = range_band(pos, target_pos, desired, deadband, max_dist)
    seek = diff if approach is None else approach
    flee = -diff if retreat is None else retreat
    return torch.where(too_far.unsqueeze(-1), seek,
                       torch.where(too_close.unsqueeze(-1), flee, torch.zeros_like(diff)))


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
