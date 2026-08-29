import numpy as np
import yaml

from brawl_sim.config import load_config
from brawl_sim.maps.loader import build_map_bank
from brawl_sim.render.ascii import render_ascii
from scripts import record_rollout

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())


def _cfg(overrides=None):
    merged = dict(CONFIGS_TINY)
    if overrides:
        merged = {**merged, **overrides}
    return load_config(CONFIGS_DEFAULT, overrides=merged)


def test_record_rollout_returns_stacked_arrays_of_the_right_length():
    cfg = _cfg()
    frames = record_rollout.record_rollout(cfg, steps=15, seed=0)
    for name, arr in frames.items():
        assert arr.shape[0] == 15, f"{name}: expected 15 frames, got {arr.shape[0]}"
    assert "revealed_to_hero" in frames
    assert "max_ammo" in frames
    assert frames["ent_pos"].shape[1:] == (cfg.n_entities, 2)


def test_record_rollout_frame_0_is_the_post_reset_state():
    cfg = _cfg()
    frames = record_rollout.record_rollout(cfg, steps=5, seed=0)
    assert np.all(frames["step_count"][0] == 0)
    assert np.all(frames["ent_alive"][0])  # every entity starts alive


def test_record_rollout_is_deterministic_for_a_fixed_seed():
    cfg = _cfg()
    a = record_rollout.record_rollout(cfg, steps=10, seed=1, action_seed=2)
    b = record_rollout.record_rollout(cfg, steps=10, seed=1, action_seed=2)
    for name in a:
        assert np.array_equal(a[name], b[name]), f"{name} differs across identical seeds"


def test_record_rollout_rejects_non_positive_steps():
    cfg = _cfg()
    try:
        record_rollout.record_rollout(cfg, steps=0)
        assert False, "expected a ValueError"
    except ValueError:
        pass


# ---- integration: recorded frames render correctly via render_ascii --------------------------

def test_recorded_frames_render_via_ascii_in_all_three_modes():
    cfg = _cfg(overrides={"entities": {"n_enemies": 2}})
    frames = record_rollout.record_rollout(cfg, steps=10, seed=0)
    bank = build_map_bank(cfg, device="cpu")

    frame0 = {k: v[0] for k, v in frames.items()}
    for mode in ("world", "view", "agent"):
        out = render_ascii(frame0, bank, cfg, mode=mode)
        assert "\n\n" in out
        map_part, status = out.split("\n\n")
        expected_h = cfg.map_h if mode == "world" else cfg.view_h
        expected_w = cfg.map_w if mode == "world" else cfg.view_w
        lines = map_part.split("\n")
        assert len(lines) == expected_h
        assert all(len(line) == expected_w for line in lines)


def test_main_cli_writes_an_npz_file(tmp_path):
    out_path = tmp_path / "rollout.npz"
    exit_code = record_rollout.main([
        "--preset", "configs/presets/debug_tiny.yaml",
        "--steps", "10", "--out", str(out_path),
    ])
    assert exit_code == 0
    assert out_path.exists()
    with np.load(out_path) as data:
        assert data["ent_pos"].shape[0] == 10
        assert {"ent_pos", "ent_alive", "ent_kind", "revealed_to_hero", "max_ammo"} <= set(data.files)
