"""brawl_sim/core/camera.py: the camera window, the clamp and the hero's reveal.

The quad in configs/default.yaml is a number copied from the calibration; the first test here is
what keeps it honest, by re-deriving it from the shipped homography the way the deployed side
would see it.
"""
import numpy as np
import torch

from tests.bot_fixtures import FakeBank, cfg_and_params, fresh_state, grid
from brawl_sim.bots import perception
from brawl_sim.config import load_config
from brawl_sim.constants import Tile
from brawl_sim.core import camera

CONFIGS_DEFAULT = "configs/default.yaml"


def _cfg(map_h=60, map_w=60, **camera_overrides):
    overrides = {"world": {"map_h": map_h, "map_w": map_w}}
    if camera_overrides:
        overrides["camera"] = camera_overrides
    return load_config(CONFIGS_DEFAULT, overrides=overrides)


# ---- calibration parity ------------------------------------------------------------------

def test_quad_matches_the_shipped_homography():
    """The four corners of the viewport, through H_inv, relative to the hero's nominal anchor,
    must be the yaml's quad to 0.05 tiles. A recalibration that changes the field of view fails
    here instead of leaving the sim revealing the wrong ground."""
    from brawl_vision.camera import HERO_ANCHOR_TILES, load_camera_model
    m = load_camera_model()
    w, h = m.viewport
    corners = m.px_to_tile(np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64))
    anchor = m.px_to_tile(np.array([[w / 2, h / 2]], np.float64))[0] + np.array(HERO_ANCHOR_TILES)
    quad = np.array(_cfg().camera_quad)
    assert np.abs((corners - anchor) - quad).max() < 0.05, (corners - anchor)


# ---- clamp -------------------------------------------------------------------------------

def test_camera_tracks_the_hero_mid_map_and_clamps_near_the_edges():
    cfg = _cfg()
    heroes = torch.tensor([[30.0, 30.0], [3.0, 30.0], [58.0, 30.0], [30.0, 2.0], [30.0, 58.0]])
    cam = camera.camera_centre(heroes, cfg)
    expected = torch.tensor([[30.0, 30.0], [12.1, 30.0], [47.2, 30.0], [30.0, 8.9], [30.0, 54.5]])
    assert torch.allclose(cam, expected), cam


def test_a_map_narrower_than_the_onsets_pins_the_camera_at_the_midpoint():
    """20 wide: 20 - 12.8 = 7.2 < 12.1, so there is no tracking range on x at all."""
    cfg = _cfg(map_h=20, map_w=20)
    heroes = torch.tensor([[1.0, 1.0], [10.0, 10.0], [19.0, 19.0]])
    cam = camera.camera_centre(heroes, cfg)
    assert torch.allclose(cam[:, 0], torch.full((3,), 9.65)), cam
    assert torch.allclose(cam[:, 1], torch.tensor([8.9, 10.0, 14.5])), cam


def test_camera_constants_keep_the_hero_tensors_dtype_and_are_cached():
    cfg = _cfg(map_h=61, map_w=61)   # a key no other test in this process has built
    before = len(camera._cache)
    cam = camera.camera_centre(torch.zeros(2, 2, dtype=torch.float32), cfg)
    assert cam.dtype == torch.float32
    camera.camera_centre(torch.zeros(4, 2, dtype=torch.float32), cfg)
    camera.in_camera(torch.zeros(4, 3, 2, dtype=torch.float32), cfg)
    assert len(camera._cache) == before + 1


# ---- quad membership ---------------------------------------------------------------------

def test_quad_membership_hand_checked_points():
    cfg = _cfg()
    inside = torch.tensor([[0.0, 0.0], [12.0, 0.0], [0.0, -9.0], [0.0, 7.0], [-13.0, -10.0]])
    outside = torch.tensor([[16.0, 0.0], [0.0, 9.0], [14.0, 6.0], [-15.0, 0.0]])
    assert camera.in_camera(inside, cfg).all()
    assert not camera.in_camera(outside, cfg).any()
    # The corners themselves are on the boundary, which is inclusive.
    assert camera.in_camera(torch.tensor(cfg.camera_quad), cfg).all()


def test_in_camera_broadcasts_over_leading_dims():
    cfg = _cfg()
    rel = torch.zeros(3, 5, 2)
    rel[1, 2] = torch.tensor([40.0, 0.0])
    out = camera.in_camera(rel, cfg)
    assert out.shape == (3, 5)
    assert out.sum() == 14 and not out[1, 2]


def test_window_bbox_is_the_quads_extent():
    cfg = _cfg()
    lo, hi = camera.window_bbox(cfg, "cpu")
    assert torch.allclose(lo, torch.tensor([-14.11, -10.93]))
    assert torch.allclose(hi, torch.tensor([14.77, 7.45]))


# ---- hero_view ---------------------------------------------------------------------------

def _scene(n_enemies=2, bush_at=None):
    cfg, params, _ = cfg_and_params(n_enemies=n_enemies, map_h=60, map_w=60)
    tiles = grid(60, 60)
    if bush_at is not None:
        x, y = bush_at
        tiles[y, x] = Tile.BUSH
    bank = FakeBank(tiles)
    state = fresh_state(cfg, params)
    return cfg, params, bank, state


def _view(cfg, params, bank, state):
    vis = perception.visibility(state, bank, params, cfg)
    return camera.hero_view(state, vis, cfg)


def test_hero_view_is_the_window_around_a_tracking_camera():
    cfg, params, bank, state = _scene(n_enemies=4)
    state.ent_pos[0, 0] = torch.tensor([30.0, 30.0])
    state.ent_pos[0, 1] = torch.tensor([42.0, 30.0])   # 12 east: inside
    state.ent_pos[0, 2] = torch.tensor([46.0, 30.0])   # 16 east: outside
    state.ent_pos[0, 3] = torch.tensor([30.0, 21.0])   # 9 north: inside
    state.ent_pos[0, 4] = torch.tensor([30.0, 39.0])   # 9 south: outside
    seen = _view(cfg, params, bank, state)
    assert seen[0].tolist() == [True, True, False, True, False]


def test_hero_view_keeps_the_bush_rule():
    """Concealment still comes from `vis`: a bushed enemy inside the window is revealed only
    within `bush_reveal_radius` (2.0)."""
    for dist, expected in ((1.5, True), (3.0, False)):
        cfg, params, bank, state = _scene(n_enemies=1, bush_at=(33, 30))
        assert params.bush_reveal_radius[0].item() == 2.0
        state.ent_pos[0, 0] = torch.tensor([33.5 - dist, 30.5])
        state.ent_pos[0, 1] = torch.tensor([33.5, 30.5])
        assert bool(_view(cfg, params, bank, state)[0, 1]) is expected


def test_a_clamped_camera_sees_further_on_the_open_side():
    """Hero 2 tiles from the west edge: the camera stops at x = 12.1, so an enemy 20 tiles east
    of the hero is only 9.9 tiles from the camera centre and on screen."""
    cfg, params, bank, state = _scene(n_enemies=1)
    state.ent_pos[0, 0] = torch.tensor([2.0, 30.0])
    state.ent_pos[0, 1] = torch.tensor([22.0, 30.0])
    assert bool(_view(cfg, params, bank, state)[0, 1])


def test_a_dead_hero_sees_nothing():
    cfg, params, bank, state = _scene(n_enemies=2)
    state.ent_pos[0, 0] = torch.tensor([30.0, 30.0])
    state.ent_pos[0, 1] = torch.tensor([32.0, 30.0])
    state.ent_alive[0, 0] = False
    assert not _view(cfg, params, bank, state)[0].any()
