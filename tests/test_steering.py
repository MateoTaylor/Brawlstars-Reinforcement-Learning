import torch

from brawl_sim.config import load_config
from brawl_sim.constants import Tile, TILE_BLOCKS_UNIT
from brawl_sim.bots import steering

CONFIGS_DEFAULT = "configs/default.yaml"


class _FakeBank:
    def __init__(self, tiles):
        self.blocks_unit = TILE_BLOCKS_UNIT[tiles].unsqueeze(0)


def _grid(h, w, fill=Tile.FLOOR):
    t = torch.full((h, w), int(fill), dtype=torch.int64)
    t[0, :] = Tile.WALL
    t[-1, :] = Tile.WALL
    t[:, 0] = Tile.WALL
    t[:, -1] = Tile.WALL
    return t


def _cfg(map_h=20, map_w=20):
    return load_config(CONFIGS_DEFAULT, overrides={"world": {"map_h": map_h, "map_w": map_w}})


# ---- seek / flee -----------------------------------------------------------------

def test_seek_points_toward_target():
    pos = torch.tensor([[[0.0, 0.0]]])
    target = torch.tensor([[[5.0, 0.0]]])
    d = steering.seek(pos, target)
    assert torch.allclose(d, torch.tensor([[[5.0, 0.0]]]))


def test_flee_points_away_from_threat():
    pos = torch.tensor([[[0.0, 0.0]]])
    threat = torch.tensor([[[5.0, 0.0]]])
    d = steering.flee(pos, threat)
    assert torch.allclose(d, torch.tensor([[[-5.0, 0.0]]]))


# ---- strafe -----------------------------------------------------------------------

def test_strafe_is_perpendicular_to_seek():
    pos = torch.tensor([[[0.0, 0.0]]])
    target = torch.tensor([[[5.0, 0.0]]])
    d = steering.strafe(pos, target, sign=1.0)
    to_target = target - pos
    dot = (d * to_target).sum(dim=-1)
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-5)


def test_strafe_sign_flips_direction():
    pos = torch.tensor([[[0.0, 0.0]]])
    target = torch.tensor([[[5.0, 0.0]]])
    plus = steering.strafe(pos, target, sign=1.0)
    minus = steering.strafe(pos, target, sign=-1.0)
    assert torch.allclose(plus, -minus)


def test_strafe_per_entity_sign_tensor():
    pos = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    target = torch.tensor([[[5.0, 0.0], [5.0, 0.0]]])
    sign = torch.tensor([[1.0, -1.0]])
    d = steering.strafe(pos, target, sign)
    assert torch.allclose(d[0, 0], -d[0, 1])


# ---- maintain_range (acceptance) --------------------------------------------------

def test_maintain_range_moves_closer_when_too_far():
    pos = torch.tensor([[[0.0, 0.0]]])
    target = torch.tensor([[[12.0, 0.0]]])  # dist 12, desired 8
    d = steering.maintain_range(pos, target, desired=8.0, deadband=1.5)
    assert d[0, 0, 0].item() > 0  # moves toward target (closer)


def test_maintain_range_moves_away_when_too_close():
    pos = torch.tensor([[[0.0, 0.0]]])
    target = torch.tensor([[[3.0, 0.0]]])  # dist 3, desired 8
    d = steering.maintain_range(pos, target, desired=8.0, deadband=1.5)
    assert d[0, 0, 0].item() < 0  # moves away from target


def test_maintain_range_zero_inside_deadband():
    pos = torch.tensor([[[0.0, 0.0]]])
    target = torch.tensor([[[8.0, 0.0]]])  # dist == desired
    d = steering.maintain_range(pos, target, desired=8.0, deadband=1.5)
    assert torch.allclose(d, torch.zeros_like(d))


def test_maintain_range_multi_entity_per_row_tensor_desired():
    # Regression: with E>1 and desired/deadband as (N,E) tensors (as every real caller in
    # bots/personality.py passes them, gathered per-kind), a dist kept at (N,E,1) used to
    # broadcast against (N,E) as if E were a second entity axis -- silently comparing entity
    # 0's distance against entity 1's desired range whenever E happened to equal 2 (the
    # vector's own last-dim size), instead of raising a shape error. Two entities, two
    # different (desired, deadband) pairs, each picking out a DIFFERENT one of the too-far /
    # too-close / dead-band branches, is exactly the shape combination that hid it.
    pos = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    target = torch.tensor([[[12.0, 0.0], [3.0, 0.0]]])  # dist 12, dist 3
    desired = torch.tensor([[8.0, 8.0]])
    deadband = torch.tensor([[1.5, 1.5]])
    d = steering.maintain_range(pos, target, desired, deadband)
    assert d[0, 0, 0].item() > 0   # entity 0: too far -> seeks
    assert d[0, 1, 0].item() < 0   # entity 1: too close -> flees


# ---- avoid_walls (acceptance: escapes a dead end) --------------------------------

def test_avoid_walls_steers_out_of_dead_end():
    cfg = _cfg()
    tiles = _grid(20, 20)
    # carve a dead-end pocket open only to the west (-x): walls to north, south, east.
    tiles[9, 12] = Tile.WALL
    tiles[11, 12] = Tile.WALL
    tiles[10, 13] = Tile.WALL
    bank = _FakeBank(tiles)
    map_id = torch.zeros(1, dtype=torch.int64)

    pos = torch.tensor([[[12.5, 10.5]]])  # inside the pocket
    d = steering.avoid_walls(pos, map_id, bank, probe_dist=1.0, cfg=cfg)
    assert d[0, 0, 0].item() < 0  # pushed west, back out the open side
    assert not torch.any(torch.isnan(d))


def test_avoid_walls_zero_in_open_space():
    cfg = _cfg()
    tiles = _grid(20, 20)
    bank = _FakeBank(tiles)
    map_id = torch.zeros(1, dtype=torch.int64)
    pos = torch.tensor([[[10.5, 10.5]]])
    d = steering.avoid_walls(pos, map_id, bank, probe_dist=1.0, cfg=cfg)
    assert torch.allclose(d, torch.zeros_like(d))


# ---- escape_zone -------------------------------------------------------------------

def test_escape_zone_zero_when_inside():
    zone_lo = torch.tensor([[5.0, 5.0]])
    zone_hi = torch.tensor([[15.0, 15.0]])
    pos = torch.tensor([[10.0, 10.0]])
    d = steering.escape_zone(pos, zone_lo, zone_hi)
    assert torch.allclose(d, torch.zeros_like(d))


def test_escape_zone_points_inward_when_outside():
    zone_lo = torch.tensor([[5.0, 5.0]])
    zone_hi = torch.tensor([[15.0, 15.0]])
    pos = torch.tensor([[20.0, 10.0]])
    d = steering.escape_zone(pos, zone_lo, zone_hi)
    assert d[0, 0].item() < 0  # pulled back toward -x, into the rect


# ---- combine (acceptance: never NaN) -----------------------------------------------

def test_combine_weighted_sum_normalizes():
    a = torch.tensor([[[1.0, 0.0]]])
    b = torch.tensor([[[0.0, 1.0]]])
    d = steering.combine((a, 1.0), (b, 1.0))
    expected = torch.nn.functional.normalize(torch.tensor([[[1.0, 1.0]]]), dim=-1)
    assert torch.allclose(d, expected, atol=1e-5)


def test_combine_all_zero_input_no_nan():
    zero = torch.zeros(1, 1, 2)
    d = steering.combine((zero, 1.0), (zero, 0.5))
    assert not torch.any(torch.isnan(d))
    assert torch.allclose(d, torch.zeros_like(d))


def test_combine_all_zero_weights_no_nan():
    a = torch.tensor([[[1.0, 0.0]]])
    b = torch.tensor([[[0.0, 1.0]]])
    d = steering.combine((a, 0.0), (b, 0.0))
    assert not torch.any(torch.isnan(d))
    assert torch.allclose(d, torch.zeros_like(d))


def test_combine_per_entity_weight_tensor():
    a = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    weight = torch.tensor([[1.0, 0.0]])
    d = steering.combine((a, weight))
    assert d[0, 0, 0].item() > 0.0
    assert torch.allclose(d[0, 1], torch.zeros(2))


# ---- batched leading-dim smoke test -------------------------------------------------

def test_batched_smoke():
    cfg = _cfg()
    tiles = _grid(20, 20)
    bank = _FakeBank(tiles)
    map_id = torch.zeros(4, dtype=torch.int64)
    pos = torch.rand(4, 3, 2) * 16 + 2
    target = torch.rand(4, 3, 2) * 16 + 2
    zone_lo = torch.full((4, 2), 4.0)
    zone_hi = torch.full((4, 2), 16.0)

    s = steering.seek(pos, target)
    f = steering.flee(pos, target)
    st = steering.strafe(pos, target, sign=1.0)
    mr = steering.maintain_range(pos, target, desired=8.0, deadband=1.5)
    aw = steering.avoid_walls(pos, map_id, bank, probe_dist=1.0, cfg=cfg)
    ez = steering.escape_zone(pos, zone_lo.unsqueeze(1).expand(-1, 3, -1), zone_hi.unsqueeze(1).expand(-1, 3, -1))
    combined = steering.combine((s, 1.0), (f, 0.5), (st, 0.3), (mr, 1.0), (aw, 0.5), (ez, 3.0))

    for d in (s, f, st, mr, aw, ez, combined):
        assert d.shape == (4, 3, 2)
        assert not torch.any(torch.isnan(d))
