import matplotlib
matplotlib.use("Agg")  # headless: never open a GUI window/event loop during tests

import numpy as np
import pytest
import torch
import yaml
from matplotlib.colors import to_rgba

from brawl_sim.bots import perception
from brawl_sim.config import load_config
from brawl_sim.constants import Proj
from brawl_sim.core import stats
from brawl_sim.env import BrawlVecEnv
from brawl_sim.maps.loader import build_map_bank
from brawl_sim.render import ascii as ascii_render
from brawl_sim.render import viewer as viewer_mod
from scripts import record_rollout

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())


def _cfg(overrides=None):
    merged = dict(CONFIGS_TINY)
    if overrides:
        merged = {**merged, **overrides}
    return load_config(CONFIGS_DEFAULT, overrides=merged)


def _viewer_from_rollout(cfg=None, steps=5, seed=0, n_enemies=2):
    cfg = cfg or _cfg(overrides={"entities": {"n_enemies": n_enemies}})
    frames = record_rollout.record_rollout(cfg, steps=steps, seed=seed)
    bank = build_map_bank(cfg, device="cpu")
    return viewer_mod.ReplayViewer(frames, bank, cfg), frames, cfg, bank


class _FakeKeyEvent:
    def __init__(self, key):
        self.key = key


# ---- construction produces exactly n_entities per-slot artists --------------------------------

def test_construction_builds_one_artist_set_per_entity_slot():
    viewer, frames, cfg, bank = _viewer_from_rollout(n_enemies=3)
    n = cfg.n_entities
    assert len(viewer.body) == n
    assert len(viewer.hp_bg) == n
    assert len(viewer.hp_fill) == n
    assert len(viewer.ammo_text) == n
    assert len(viewer.reveal_ring) == n
    # One scatter artist per SPAWNABLE projectile kind -- i.e. every Proj except NONE (the "this
    # archetype has no projectile" sentinel, never reaches a live slot) and SUPER_BOLT (Mortis's
    # super, which gets its own radius-scaled circle pool -- `super_bolt_circles` -- below instead
    # of a fixed-size scatter dot, since its `prj_radius` actually matters to what's on screen).
    #
    # Derived from the enum, not written as a literal set: the literal silently became wrong the
    # moment Step B1 added Proj.SUPER_BOLT, and the failure read as "the viewer built the wrong
    # artists" rather than "a projectile kind was added". Derived, adding a Proj member is a
    # passing change here and FORGETTING to give it a colour is the failure -- which is the bug
    # actually worth catching, since viewer._PROJ_COLOR.get() would otherwise fall back silently.
    spawnable = {k for k in Proj if k != Proj.NONE and k != Proj.SUPER_BOLT}
    assert set(viewer.proj_scatter.keys()) == spawnable
    # SUPER_BOLT gets one circle per projectile slot instead, same pool size as prj_pos.
    assert len(viewer.super_bolt_circles) == frames["prj_pos"].shape[1]


# ---- frame 0: every alive entity gets a visible, correctly-positioned body --------------------

def test_frame_0_bodies_are_visible_and_positioned_at_ent_pos():
    viewer, frames, cfg, bank = _viewer_from_rollout(n_enemies=3)
    viewer._entity_sprite_imgs = {}  # exercise the circle fallback regardless of local assets/
    frame0 = {k: v[0] for k, v in frames.items()}
    assert np.all(frame0["ent_alive"])  # freshly reset -- everyone alive
    viewer._draw_frame(0)

    for e in range(cfg.n_entities):
        body = viewer.body[e]
        assert body.get_visible()
        assert body.center == (float(frame0["ent_pos"][e, 0]), float(frame0["ent_pos"][e, 1]))


def test_dead_entity_hides_all_its_artists():
    viewer, frames, cfg, bank = _viewer_from_rollout(n_enemies=2)
    viewer._entity_sprite_imgs = {}  # exercise the circle fallback regardless of local assets/
    frame0 = {k: v.copy() for k, v in frames.items()}
    frame0["ent_alive"][0][1] = False  # kill entity 1 on frame 0
    viewer.frames = frame0
    viewer._draw_frame(0)

    assert not viewer.body[1].get_visible()
    assert not viewer.hp_bg[1].get_visible()
    assert not viewer.hp_fill[1].get_visible()
    assert not viewer.ammo_text[1].get_visible()
    assert not viewer.reveal_ring[1].get_visible()
    # untouched entities are unaffected
    assert viewer.body[0].get_visible()


# ---- title text is produced by the exact same function render_ascii uses ----------------------

def test_title_matches_ascii_status_line_for_the_same_frame():
    viewer, frames, cfg, bank = _viewer_from_rollout(n_enemies=2)
    frame0 = {k: v[0] for k, v in frames.items()}
    assert viewer.title.get_text() == ascii_render.status_line(frame0)


# ---- reveal ring: los_to_hero True AND revealed_to_hero False -> ring visible ------------------

def _bush_scenario(map_names=("bushy",)):
    cfg = load_config(CONFIGS_DEFAULT, overrides={
        "world": {"maps": list(map_names), "map_selection": "fixed", "fixed_map": map_names[0]},
        "entities": {"n_enemies": 1},
    })
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0)
    env.reset()
    bank = build_map_bank(cfg, device="cpu")

    bush_mask = env.bank.is_bush[0]
    ys, xs = torch.nonzero(bush_mask, as_tuple=True)
    bush_pos = torch.stack([xs[0].float() + 0.5, ys[0].float() + 0.5])
    env.state.ent_pos[0, 1] = bush_pos
    # far enough that bush_reveal_radius (2.0 tiles, configs/default.yaml) doesn't kick in, but
    # still with a clear wall-free line of sight -- same placement test_ascii.py's bush test uses.
    env.state.ent_pos[0, 0] = bush_pos + torch.tensor([20.0, 0.0])
    env.state.ent_reveal_t[0, 1] = 0.0
    return env, bank, cfg


def _frame_from_env(env):
    from brawl_sim.core.state import snapshot as state_snapshot
    vis = perception.visibility(env.state, env.bank, env.params, env.cfg)
    los = perception.raw_los(env.state, env.bank, env.cfg)
    frame = state_snapshot(env.state, 0)
    frame["revealed_to_hero"] = vis[0, 0].numpy()
    frame["los_to_hero"] = los[0, 0].numpy()
    frame["max_ammo"] = stats.gather_kind(env.params.max_ammo, env.state.ent_kind)[0].numpy()
    frame["unit_radius"] = env.params.unit_radius[0].numpy()
    return frame


def test_reveal_ring_lights_up_for_a_bush_hidden_but_los_clear_bot():
    env, bank, cfg = _bush_scenario()
    frame = _frame_from_env(env)
    assert not bool(frame["revealed_to_hero"][1])  # bush-hidden
    assert bool(frame["los_to_hero"][1])  # no wall between hero and bot at this distance

    frames = {name: arr[None] for name, arr in frame.items()}
    viewer = viewer_mod.ReplayViewer(frames, bank, cfg)
    assert viewer.reveal_ring[1].get_visible()
    assert not viewer.reveal_ring[0].get_visible()  # never rings the hero itself


def test_reveal_ring_stays_off_when_los_is_blocked_by_a_wall():
    env, bank, cfg = _bush_scenario()
    frame = _frame_from_env(env)
    # force the "wall in the way" case regardless of the real geometry -- the ring must depend
    # on los_to_hero, not merely on revealed_to_hero being False.
    frame["revealed_to_hero"][1] = False
    frame["los_to_hero"][1] = False

    frames = {name: arr[None] for name, arr in frame.items()}
    viewer = viewer_mod.ReplayViewer(frames, bank, cfg)
    assert not viewer.reveal_ring[1].get_visible()


def test_v_key_toggles_reveal_ring_visibility():
    env, bank, cfg = _bush_scenario()
    frame = _frame_from_env(env)
    frames = {name: arr[None] for name, arr in frame.items()}
    viewer = viewer_mod.ReplayViewer(frames, bank, cfg)
    assert viewer.reveal_ring[1].get_visible()

    viewer.on_key(_FakeKeyEvent("v"))
    assert not viewer.reveal_ring[1].get_visible()

    viewer.on_key(_FakeKeyEvent("v"))
    assert viewer.reveal_ring[1].get_visible()


# ---- dashing gets a bright ring artist, independent of body vs. sprite rendering -----------------

def test_dashing_entity_gets_the_bright_dash_ring():
    viewer, frames, cfg, bank = _viewer_from_rollout(n_enemies=1)
    frame0 = {k: v[0].copy() for k, v in frames.items()}
    frame0["ent_dash_t"][0] = 0.5
    viewer.frames = {k: v[None] for k, v in frame0.items()}
    viewer._draw_frame(0)
    assert viewer.dash_ring[0].get_visible()
    assert viewer.dash_ring[0].get_edgecolor() == to_rgba(viewer_mod._DASH_RING_COLOR)

    frame0["ent_dash_t"][0] = 0.0
    viewer.frames = {k: v[None] for k, v in frame0.items()}
    viewer._draw_frame(0)
    assert not viewer.dash_ring[0].get_visible()


# ---- PNG sprites override the circle/scatter fallback iff a file was actually found -------------

def test_entity_falls_back_to_circle_when_no_sprite_for_its_kind():
    viewer, frames, cfg, bank = _viewer_from_rollout(n_enemies=1)
    viewer._entity_sprite_imgs = {}  # force the "no PNG on disk" case regardless of local assets/
    viewer._draw_frame(0)
    assert viewer.body[0].get_visible()
    assert not viewer.sprite[0].get_visible()


def test_entity_sprite_overrides_the_body_circle_when_present():
    viewer, frames, cfg, bank = _viewer_from_rollout(n_enemies=1)
    frame0 = {k: v[0] for k, v in frames.items()}
    kind0 = int(frame0["ent_kind"][0])
    fake_sprite = np.ones((4, 4, 4), dtype=np.float32)
    viewer._entity_sprite_imgs[kind0] = fake_sprite
    viewer._draw_frame(0)
    assert viewer.sprite[0].get_visible()
    assert not viewer.body[0].get_visible()
    assert np.array_equal(viewer.sprite[0].get_array(), fake_sprite)


def test_box_sprite_pool_used_instead_of_scatter_when_box_png_present(monkeypatch):
    fake_box_sprite = np.ones((4, 4, 4), dtype=np.float32)

    def fake_load_sprite(path):
        return fake_box_sprite if path.name == "box.png" else None

    monkeypatch.setattr(viewer_mod, "_load_sprite", fake_load_sprite)
    viewer, frames, cfg, bank = _viewer_from_rollout(steps=1)
    frame0 = {k: v[0].copy() for k, v in frames.items()}
    frame0["box_alive"][:] = False
    frame0["box_alive"][0] = True
    frame0["box_pos"][0] = [3.0, 4.0]
    viewer.frames = {k: v[None] for k, v in frame0.items()}
    viewer._draw_frame(0)

    assert not viewer.box_scatter.get_visible()
    assert viewer.box_sprites[0].get_visible()
    assert np.array_equal(viewer.box_sprites[0].get_array(), fake_box_sprite)
    assert not viewer.box_sprites[1].get_visible()


# ---- pause/scrub/speed key bindings --------------------------------------------------------------

def test_space_toggles_pause():
    viewer, *_ = _viewer_from_rollout(steps=5)
    assert not viewer.paused
    viewer.on_key(_FakeKeyEvent(" "))
    assert viewer.paused
    viewer.on_key(_FakeKeyEvent(" "))
    assert not viewer.paused


def test_arrow_keys_scrub_and_clamp_at_bounds():
    viewer, frames, cfg, bank = _viewer_from_rollout(steps=5)
    assert viewer.frame_idx == 0
    viewer.on_key(_FakeKeyEvent("left"))  # clamps at 0, doesn't go negative
    assert viewer.frame_idx == 0
    assert viewer.paused

    viewer.on_key(_FakeKeyEvent("right"))
    assert viewer.frame_idx == 1
    for _ in range(10):
        viewer.on_key(_FakeKeyEvent("right"))
    assert viewer.frame_idx == 4  # clamps at n_frames - 1


def test_speed_keys_double_and_halve():
    viewer, *_ = _viewer_from_rollout(steps=5)
    assert viewer.speed == 1.0
    viewer.on_key(_FakeKeyEvent("+"))
    assert viewer.speed == 2.0
    viewer.on_key(_FakeKeyEvent("-"))
    viewer.on_key(_FakeKeyEvent("-"))
    assert viewer.speed == 0.5


def test_g_key_toggles_crop_and_resets_limits_on_exit():
    viewer, *_ = _viewer_from_rollout(steps=5)
    assert not viewer.crop_to_view
    viewer.on_key(_FakeKeyEvent("g"))
    assert viewer.crop_to_view
    viewer.on_key(_FakeKeyEvent("g"))
    assert not viewer.crop_to_view
    assert viewer.ax.get_xlim() == (0, viewer.W)


def test_c_key_toggles_view_rect_visibility():
    viewer, *_ = _viewer_from_rollout(steps=5)
    assert not viewer.view_rect.get_visible()
    viewer.on_key(_FakeKeyEvent("c"))
    assert viewer.view_rect.get_visible()


# ---- zone bands partition the map correctly (no double coverage, no gaps) -----------------------

def test_zone_bands_never_cover_the_safe_rect():
    viewer, frames, cfg, bank = _viewer_from_rollout(steps=1)
    frame0 = {k: v[0].copy() for k, v in frames.items()}
    cx, cy = cfg.map_w / 2.0, cfg.map_h / 2.0
    frame0["zone_lo"] = np.array([cx - 5, cy - 5], dtype=np.float32)
    frame0["zone_hi"] = np.array([cx + 5, cy + 5], dtype=np.float32)
    viewer.frames = {k: v[None] for k, v in frame0.items()}
    viewer._draw_frame(0)

    top, bottom, left, right = viewer.zone_rects
    for rect in (top, bottom, left, right):
        rx0, ry0 = rect.get_x(), rect.get_y()
        rx1, ry1 = rx0 + rect.get_width(), ry0 + rect.get_height()
        # every band must not overlap the safe rect's interior
        overlaps = rx0 < cx + 5 and rx1 > cx - 5 and ry0 < cy + 5 and ry1 > cy - 5
        assert not overlaps, f"zone band ({rx0},{ry0},{rx1},{ry1}) overlaps the safe rect"


# ---- projectile scatter routes each kind to its own collection ---------------------------------

def test_projectiles_route_to_the_scatter_matching_their_kind():
    viewer, frames, cfg, bank = _viewer_from_rollout(steps=1)
    frame0 = {k: v[0].copy() for k, v in frames.items()}
    frame0["prj_alive"][:] = False
    frame0["prj_alive"][0] = True
    frame0["prj_kind"][0] = 2  # ARTILLERY_SHELL
    frame0["prj_pos"][0] = [7.0, 9.0]
    viewer.frames = {k: v[None] for k, v in frame0.items()}
    viewer._draw_frame(0)

    shell_offsets = viewer.proj_scatter[2].get_offsets()
    assert shell_offsets.shape == (1, 2)
    assert np.allclose(shell_offsets[0], [7.0, 9.0])
    for kind, scat in viewer.proj_scatter.items():
        if kind != 2:
            assert scat.get_offsets().shape[0] == 0


# ---- Mortis's super is drawn at its real prj_radius, not a fixed-size dot ----------------------

def test_super_bolt_circle_is_sized_to_prj_radius():
    viewer, frames, cfg, bank = _viewer_from_rollout(steps=1)
    frame0 = {k: v[0].copy() for k, v in frames.items()}
    frame0["prj_alive"][:] = False
    frame0["prj_alive"][0] = True
    frame0["prj_kind"][0] = int(Proj.SUPER_BOLT)
    frame0["prj_class"][0] = int(viewer_mod.ProjClass.PROJECTILE)
    frame0["prj_pos"][0] = [3.0, 4.0]
    frame0["prj_radius"][0] = 0.70
    viewer.frames = {k: v[None] for k, v in frame0.items()}
    viewer._draw_frame(0)

    circle = viewer.super_bolt_circles[0]
    assert circle.get_visible()
    assert np.allclose(circle.center, [3.0, 4.0])
    assert circle.get_radius() == pytest.approx(0.70)
    # every other slot in the pool stays hidden
    assert all(not c.get_visible() for c in viewer.super_bolt_circles[1:])
    # and it must not also show up as a fixed-size scatter dot
    assert Proj.SUPER_BOLT not in viewer.proj_scatter


# ---- CLI smoke: parses args, builds cfg/bank, constructs and "shows" without raising -----------

def test_main_cli_runs_end_to_end_under_the_agg_backend(tmp_path):
    npz_path = tmp_path / "rollout.npz"
    exit_code = record_rollout.main([
        "--preset", "configs/presets/debug_tiny.yaml",
        "--steps", "5", "--out", str(npz_path),
    ])
    assert exit_code == 0

    exit_code = viewer_mod.main([
        str(npz_path), "--preset", "configs/presets/debug_tiny.yaml",
    ])
    assert exit_code == 0
