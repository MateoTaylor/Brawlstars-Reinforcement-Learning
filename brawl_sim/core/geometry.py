"""Batched 2D geometry. Every function is branch-free and broadcasts over arbitrary leading
dims -- vectors are (..., 2) with the last axis (x, y); angles are plain (...) tensors in
radians via atan2(dy, dx), range (-pi, pi] (CONVENTIONS.md). No host syncs anywhere here.
"""
import math

import torch

_EPS = 1e-8
_TWO_PI = 2.0 * math.pi


def vec2(x: float, y: float, device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """(2,) tensor built from two plain Python scalars -- NOT `torch.tensor([x, y],
    device=device)`, which is a real host sync (confirmed with `torch.cuda.
    set_sync_debug_mode("error")` while building env.py, Step 29): constructing a CUDA tensor
    directly from a Python list synchronizes, but `torch.full` with a scalar fill does not.
    Used anywhere a small map-size-shaped constant (map_w/map_h, a map center, ...) needs to
    become a device tensor inside the hot path."""
    return torch.stack([
        torch.full((), x, device=device, dtype=dtype),
        torch.full((), y, device=device, dtype=dtype),
    ])


def safe_norm(v: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return torch.clamp((v * v).sum(dim=dim, keepdim=keepdim), min=0.0).sqrt()


def normalize(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    n = torch.clamp(safe_norm(v, dim=dim, keepdim=True), min=_EPS)
    return v / n


def angle_of(v: torch.Tensor) -> torch.Tensor:
    return torch.atan2(v[..., 1], v[..., 0])


def from_angle(theta: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    # Maps to (-pi, pi], matching atan2's own range.
    return math.pi - torch.remainder(math.pi - theta, _TWO_PI)


def angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return wrap_angle(a - b)


def dir_from_bin(idx: torch.Tensor, n_bins: int) -> torch.Tensor:
    theta = idx.to(torch.float32) * (_TWO_PI / n_bins)
    return from_angle(theta)


def bin_from_dir(v: torch.Tensor, n_bins: int) -> torch.Tensor:
    theta = angle_of(v)
    step = _TWO_PI / n_bins
    idx = torch.round(theta / step)
    return torch.remainder(idx, n_bins).to(torch.int64)


def dist(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return safe_norm(p - q, dim=-1)


def dist2(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    d = p - q
    return (d * d).sum(dim=-1)


def rotate(v: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    x = v[..., 0] * c - v[..., 1] * s
    y = v[..., 0] * s + v[..., 1] * c
    return torch.stack([x, y], dim=-1)


def perp(v: torch.Tensor) -> torch.Tensor:
    # 90-degree CCW rotation, i.e. rotate(v, pi/2) -- (x, y) -> (-y, x).
    return torch.stack([-v[..., 1], v[..., 0]], dim=-1)


def point_in_cone(
    p: torch.Tensor, origin: torch.Tensor, facing: torch.Tensor,
    radius: torch.Tensor, half_angle: torch.Tensor,
) -> torch.Tensor:
    delta = p - origin
    d = safe_norm(delta, dim=-1)
    within_radius = d <= radius
    at_origin = d < _EPS
    ang = angle_of(delta)
    diff = torch.abs(angle_diff(ang, facing))
    within_angle = diff <= half_angle
    return within_radius & (within_angle | at_origin)


def segment_circle_hit(p0: torch.Tensor, p1: torch.Tensor, c: torch.Tensor, r: torch.Tensor):
    """Returns (hit, t): whether segment [p0, p1] intersects the circle (c, r), and the
    parametric t in [0, 1] of the first crossing (0 for a zero-length segment starting
    inside the circle)."""
    d = p1 - p0
    f = p0 - c
    a = (d * d).sum(dim=-1)
    b = 2.0 * (f * d).sum(dim=-1)
    cc = (f * f).sum(dim=-1) - r * r

    zero_len = a.abs() < _EPS
    a_safe = torch.where(zero_len, torch.ones_like(a), a)

    disc = b * b - 4.0 * a_safe * cc
    has_disc = disc >= 0.0
    sqrt_disc = torch.clamp(disc, min=0.0).sqrt()

    t1 = (-b - sqrt_disc) / (2.0 * a_safe)
    t2 = (-b + sqrt_disc) / (2.0 * a_safe)
    t1_valid = has_disc & (t1 >= 0.0) & (t1 <= 1.0)
    t2_valid = has_disc & (t2 >= 0.0) & (t2 <= 1.0)
    # p0 already inside/on the circle: both roots can fall outside [0, 1] (p1 also inside,
    # so the segment never crosses the boundary) even though the whole segment is a hit.
    starts_inside = cc <= 0.0
    t_line = torch.where(
        starts_inside, torch.zeros_like(t1),
        torch.where(t1_valid, t1, torch.where(t2_valid, t2, torch.zeros_like(t1))),
    )
    hit_line = starts_inside | t1_valid | t2_valid

    point_inside = cc <= 0.0
    hit = torch.where(zero_len, point_inside, hit_line)
    t = torch.where(zero_len, torch.zeros_like(t_line), t_line)
    return hit, t


def closest_point_on_segment(p0: torch.Tensor, p1: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    d = p1 - p0
    len2 = (d * d).sum(dim=-1, keepdim=True)
    len2_safe = torch.clamp(len2, min=_EPS)
    t = ((q - p0) * d).sum(dim=-1, keepdim=True) / len2_safe
    t = torch.clamp(t, 0.0, 1.0)
    return p0 + t * d  # d == 0 for a zero-length segment -> result is p0, no NaN


def capsule_contains(p0: torch.Tensor, p1: torch.Tensor, r: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    cp = closest_point_on_segment(p0, p1, q)
    return dist(cp, q) <= r


def closest_approach(p: torch.Tensor, v: torch.Tensor, q: torch.Tensor):
    """Time (clamped >= 0) and distance at the closest future approach of a point moving
    from `p` at constant velocity `v` to the stationary point `q`. Used for obs features
    like a projectile's time-to-closest/closest-dist to the hero."""
    r = q - p
    vv = (v * v).sum(dim=-1)
    vv_safe = torch.clamp(vv, min=_EPS)
    t_raw = (r * v).sum(dim=-1) / vv_safe
    zero_v = vv < _EPS
    t = torch.where(zero_v, torch.zeros_like(t_raw), torch.clamp(t_raw, min=0.0))
    closest_pos = p + t.unsqueeze(-1) * v
    return t, dist(closest_pos, q)


def lead_target(
    shooter: torch.Tensor, target: torch.Tensor, target_vel: torch.Tensor,
    proj_speed: torch.Tensor, fraction: torch.Tensor,
) -> torch.Tensor:
    """Aim point blending the target's current position (fraction=0) with a two-iteration
    fixed-point lead solution (fraction=1). Not an exact quadratic intercept solve -- stable
    and branch-free, which matters more here since bot aim already adds noise on top (D19)."""
    speed_safe = torch.clamp(proj_speed, min=_EPS)
    t = dist(shooter, target) / speed_safe
    predicted = target + target_vel * t.unsqueeze(-1)
    t = dist(shooter, predicted) / speed_safe
    predicted = target + target_vel * t.unsqueeze(-1)
    return target + fraction.unsqueeze(-1) * (predicted - target)
