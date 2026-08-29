import math

import torch

from brawl_sim.core import geometry as geo

BATCH = (64, 7)


def _rand_vec(*shape, scale=10.0):
    return (torch.rand(*shape, 2) - 0.5) * scale


# ---- safe_norm / normalize ------------------------------------------------

def test_normalize_unit_length_on_nonzero_input():
    v = _rand_vec(200)
    n = geo.normalize(v)
    lengths = geo.safe_norm(n)
    assert torch.allclose(lengths, torch.ones(200), atol=1e-5)


def test_normalize_zero_vector_no_nan():
    v = torch.zeros(50, 2)
    n = geo.normalize(v)
    assert not torch.any(torch.isnan(n))
    assert torch.allclose(n, torch.zeros(50, 2))


def test_safe_norm_matches_torch_norm():
    v = _rand_vec(100)
    assert torch.allclose(geo.safe_norm(v), v.norm(dim=-1), atol=1e-5)


# ---- angle_of / from_angle / wrap_angle / angle_diff -----------------------

def test_angle_of_from_angle_roundtrip():
    # exclude the exact +-pi boundary: it's a genuine branch-cut ambiguity (pi and -pi are
    # the same angle), not a roundtrip error, so compare via the wrapped difference instead.
    theta = torch.linspace(-math.pi + 0.01, math.pi - 0.01, 500)
    v = geo.from_angle(theta)
    theta2 = geo.angle_of(v)
    assert torch.allclose(geo.angle_diff(theta, theta2), torch.zeros_like(theta), atol=1e-5)


def test_wrap_angle_range_and_identity_inside_range():
    theta = torch.linspace(-math.pi + 1e-4, math.pi, 1000)
    wrapped = geo.wrap_angle(theta)
    assert torch.allclose(theta, wrapped, atol=1e-4)


def test_wrap_angle_out_of_range_values():
    theta = torch.tensor([3 * math.pi, -3 * math.pi, 2 * math.pi + 0.1, -2 * math.pi - 0.1])
    wrapped = geo.wrap_angle(theta)
    assert torch.all(wrapped > -math.pi - 1e-6)
    assert torch.all(wrapped <= math.pi + 1e-6)
    expected = torch.tensor([math.pi, math.pi, 0.1, -0.1])
    assert torch.allclose(wrapped, expected, atol=1e-4)


def test_angle_diff_wraps_correctly():
    a = torch.tensor([0.1])
    b = torch.tensor([-3.1])  # a - b = 3.2, should wrap to 3.2 - 2pi
    d = geo.angle_diff(a, b)
    expected = 0.1 - (-3.1) - 2 * math.pi
    assert torch.allclose(d, torch.tensor([expected]), atol=1e-4)
    assert torch.all(d > -math.pi - 1e-6) and torch.all(d <= math.pi + 1e-6)


# ---- dir_from_bin / bin_from_dir -------------------------------------------

def test_bin_roundtrip_all_k():
    n_bins = 16
    k = torch.arange(n_bins, dtype=torch.int64)
    v = geo.dir_from_bin(k, n_bins)
    k2 = geo.bin_from_dir(v, n_bins)
    assert torch.equal(k, k2)


def test_bin_from_dir_zero_vector_no_nan():
    v = torch.zeros(10, 2)
    idx = geo.bin_from_dir(v, 16)
    assert not torch.any(torch.isnan(idx.to(torch.float32)))
    assert torch.all((idx >= 0) & (idx < 16))


def test_dir_from_bin_bin_zero_is_angle_zero():
    v = geo.dir_from_bin(torch.tensor([0]), 16)
    assert torch.allclose(v, torch.tensor([[1.0, 0.0]]), atol=1e-5)


# ---- dist / dist2 / rotate / perp -------------------------------------------

def test_dist_matches_dist2_sqrt():
    p, q = _rand_vec(100), _rand_vec(100)
    assert torch.allclose(geo.dist(p, q) ** 2, geo.dist2(p, q), atol=1e-3)


def test_rotate_by_pi_over_2_equals_perp():
    v = _rand_vec(100)
    rotated = geo.rotate(v, torch.full((100,), math.pi / 2))
    assert torch.allclose(rotated, geo.perp(v), atol=1e-4)


def test_rotate_full_circle_is_identity():
    v = _rand_vec(50)
    rotated = geo.rotate(v, torch.full((50,), 2 * math.pi))
    assert torch.allclose(rotated, v, atol=1e-3)


# ---- point_in_cone -----------------------------------------------------------

def test_point_in_cone_directly_ahead():
    origin = torch.zeros(1, 2)
    facing = torch.zeros(1)
    p = torch.tensor([[1.0, 0.0]])
    assert bool(geo.point_in_cone(p, origin, facing, radius=torch.tensor([2.0]), half_angle=torch.tensor([0.5])))


def test_point_in_cone_behind_is_excluded():
    origin = torch.zeros(1, 2)
    facing = torch.zeros(1)
    p = torch.tensor([[-1.0, 0.0]])
    assert not bool(geo.point_in_cone(p, origin, facing, radius=torch.tensor([2.0]), half_angle=torch.tensor([0.5])))


def test_point_in_cone_at_origin_counts_regardless_of_angle():
    origin = torch.zeros(1, 2)
    facing = torch.zeros(1)
    p = torch.zeros(1, 2)
    assert bool(geo.point_in_cone(p, origin, facing, radius=torch.tensor([2.0]), half_angle=torch.tensor([0.01])))


# ---- segment_circle_hit ------------------------------------------------------

def test_segment_circle_hit_matches_brute_force():
    torch.manual_seed(0)
    n_candidates = 1500
    p0 = (torch.rand(n_candidates, 2) - 0.5) * 20
    p1 = p0 + (torch.rand(n_candidates, 2) - 0.5) * 20
    c = (torch.rand(n_candidates, 2) - 0.5) * 20
    r = torch.rand(n_candidates) * 5 + 0.1

    # independent reference (manual projection, not calling geometry.py) for filtering
    d = p1 - p0
    len2 = (d * d).sum(-1).clamp(min=1e-12)
    t_proj = (((c - p0) * d).sum(-1) / len2).clamp(0, 1)
    closest = p0 + t_proj.unsqueeze(-1) * d
    min_dist = (closest - c).norm(dim=-1)

    # drop near-tangent cases: ambiguous under any finite sampling resolution
    keep = (min_dist - r).abs() > 0.05
    p0, p1, c, r = p0[keep][:1000], p1[keep][:1000], c[keep][:1000], r[keep][:1000]
    n = p0.shape[0]
    assert n >= 500  # sanity: the filter shouldn't gut the sample

    hit, t = geo.segment_circle_hit(p0, p1, c, r)

    steps = 10000
    ts = torch.linspace(0, 1, steps)
    pts = p0.unsqueeze(1) + ts.view(1, -1, 1) * (p1 - p0).unsqueeze(1)
    dists = (pts - c.unsqueeze(1)).norm(dim=-1)
    inside = dists <= r.unsqueeze(1)
    brute_hit = inside.any(dim=1)
    first_idx = torch.argmax(inside.to(torch.int64), dim=1)
    brute_t = ts[first_idx]

    assert torch.equal(hit, brute_hit)
    matched = hit & brute_hit
    assert torch.all((t[matched] - brute_t[matched]).abs() < 1e-3)


def test_segment_circle_hit_zero_length_segment_no_nan():
    p0 = _rand_vec(50)
    p1 = p0.clone()
    c = _rand_vec(50)
    r = torch.rand(50) + 0.1
    hit, t = geo.segment_circle_hit(p0, p1, c, r)
    assert not torch.any(torch.isnan(t))
    inside = (p0 - c).norm(dim=-1) <= r
    assert torch.equal(hit, inside)
    assert torch.all(t == 0.0)


# ---- closest_point_on_segment / capsule_contains -----------------------------

def test_closest_point_on_segment_endpoints_and_midpoint():
    p0 = torch.tensor([[0.0, 0.0]])
    p1 = torch.tensor([[10.0, 0.0]])
    q_before = torch.tensor([[-5.0, 0.0]])
    q_after = torch.tensor([[15.0, 0.0]])
    q_mid = torch.tensor([[5.0, 3.0]])
    assert torch.allclose(geo.closest_point_on_segment(p0, p1, q_before), p0)
    assert torch.allclose(geo.closest_point_on_segment(p0, p1, q_after), p1)
    assert torch.allclose(geo.closest_point_on_segment(p0, p1, q_mid), torch.tensor([[5.0, 0.0]]))


def test_closest_point_on_segment_zero_length_no_nan():
    p0 = _rand_vec(50)
    p1 = p0.clone()
    q = _rand_vec(50)
    cp = geo.closest_point_on_segment(p0, p1, q)
    assert not torch.any(torch.isnan(cp))
    assert torch.allclose(cp, p0)


def test_capsule_contains_matches_radius_check():
    p0 = torch.tensor([[0.0, 0.0]])
    p1 = torch.tensor([[10.0, 0.0]])
    q_in = torch.tensor([[5.0, 1.0]])
    q_out = torch.tensor([[5.0, 3.0]])
    r = torch.tensor([2.0])
    assert bool(geo.capsule_contains(p0, p1, r, q_in))
    assert not bool(geo.capsule_contains(p0, p1, r, q_out))


# ---- closest_approach ---------------------------------------------------------

def test_closest_approach_head_on():
    p = torch.tensor([[0.0, 0.0]])
    v = torch.tensor([[1.0, 0.0]])
    q = torch.tensor([[5.0, 0.0]])
    t, d = geo.closest_approach(p, v, q)
    assert torch.allclose(t, torch.tensor([5.0]), atol=1e-4)
    assert torch.allclose(d, torch.tensor([0.0]), atol=1e-4)


def test_closest_approach_zero_velocity_no_nan():
    p = _rand_vec(50)
    v = torch.zeros(50, 2)
    q = _rand_vec(50)
    t, d = geo.closest_approach(p, v, q)
    assert not torch.any(torch.isnan(t)) and not torch.any(torch.isnan(d))
    assert torch.all(t == 0.0)
    assert torch.allclose(d, geo.dist(p, q), atol=1e-4)


def test_closest_approach_moving_away_clamps_to_zero():
    p = torch.tensor([[0.0, 0.0]])
    v = torch.tensor([[-1.0, 0.0]])  # moving away from q
    q = torch.tensor([[5.0, 0.0]])
    t, d = geo.closest_approach(p, v, q)
    assert torch.allclose(t, torch.tensor([0.0]), atol=1e-4)
    assert torch.allclose(d, torch.tensor([5.0]), atol=1e-4)


# ---- lead_target ---------------------------------------------------------------

def test_lead_target_fraction_zero_is_current_position():
    shooter = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[10.0, 0.0]])
    target_vel = torch.tensor([[0.0, 3.0]])
    proj_speed = torch.tensor([5.0])
    fraction = torch.tensor([0.0])
    aim = geo.lead_target(shooter, target, target_vel, proj_speed, fraction)
    assert torch.allclose(aim, target, atol=1e-4)


def test_lead_target_stationary_target_no_lead_needed():
    shooter = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[10.0, 0.0]])
    target_vel = torch.zeros(1, 2)
    proj_speed = torch.tensor([5.0])
    fraction = torch.tensor([1.0])
    aim = geo.lead_target(shooter, target, target_vel, proj_speed, fraction)
    assert torch.allclose(aim, target, atol=1e-4)


def test_lead_target_leads_a_moving_target_ahead_of_its_position():
    shooter = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[10.0, 0.0]])
    target_vel = torch.tensor([[0.0, 2.0]])  # moving perpendicular to the shot
    proj_speed = torch.tensor([5.0])
    fraction = torch.tensor([1.0])
    aim = geo.lead_target(shooter, target, target_vel, proj_speed, fraction)
    assert aim[0, 1] > 0.0  # aims ahead of the target's current y, in its direction of travel


def test_lead_target_zero_speed_no_nan():
    shooter = _rand_vec(20)
    target = _rand_vec(20)
    target_vel = _rand_vec(20)
    proj_speed = torch.zeros(20)
    fraction = torch.rand(20)
    aim = geo.lead_target(shooter, target, target_vel, proj_speed, fraction)
    assert not torch.any(torch.isnan(aim))


# ---- batched leading-dim smoke test (64, 7) -----------------------------------

def test_all_functions_support_64x7_leading_batch():
    p0 = _rand_vec(*BATCH)
    p1 = _rand_vec(*BATCH)
    c = _rand_vec(*BATCH)
    q = _rand_vec(*BATCH)
    v = _rand_vec(*BATCH)
    r = torch.rand(*BATCH) + 0.1
    theta = (torch.rand(*BATCH) - 0.5) * 4 * math.pi
    idx = torch.randint(0, 16, BATCH, dtype=torch.int64)
    fraction = torch.rand(*BATCH)
    proj_speed = torch.rand(*BATCH) + 1.0

    outputs = [
        geo.safe_norm(p0),
        geo.normalize(p0),
        geo.angle_of(p0),
        geo.from_angle(theta),
        geo.wrap_angle(theta),
        geo.angle_diff(theta, theta),
        geo.dir_from_bin(idx, 16),
        geo.bin_from_dir(p0, 16).to(torch.float32),
        geo.dist(p0, p1),
        geo.dist2(p0, p1),
        geo.rotate(p0, theta),
        geo.perp(p0),
        geo.point_in_cone(p0, p1, theta, r, r).to(torch.float32),
        geo.closest_point_on_segment(p0, p1, q),
        geo.capsule_contains(p0, p1, r, q).to(torch.float32),
        geo.closest_approach(p0, v, q)[0],
        geo.closest_approach(p0, v, q)[1],
        geo.lead_target(p0, p1, v, proj_speed, fraction),
    ]
    hit, t = geo.segment_circle_hit(p0, p1, c, r)
    outputs += [hit.to(torch.float32), t]

    for out in outputs:
        assert out.shape[: len(BATCH)] == BATCH
        assert not torch.any(torch.isnan(out))
