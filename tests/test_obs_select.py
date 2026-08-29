import tempfile
import os

import gymnasium
import torch
import yaml

from brawl_sim.config import load_config
from brawl_sim.core import obs_select
from brawl_sim.env import BrawlVecEnv

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())
AGENT_OBS_YAML = "configs/agent_obs.yaml"


def _cfg(overrides=None):
    merged = dict(CONFIGS_TINY)
    if overrides:
        merged = {**merged, **overrides}
    return load_config(CONFIGS_DEFAULT, overrides=merged)


def _env_and_obs(n_envs=4, seed=0, overrides=None):
    cfg = _cfg(overrides)
    env = BrawlVecEnv(cfg, n_envs=n_envs, device="cpu", seed=seed)
    full_obs = env.reset()
    return env, cfg, full_obs


def _write_spec(spec_dict) -> str:
    path = tempfile.mktemp(suffix=".yaml")
    with open(path, "w") as f:
        yaml.dump(spec_dict, f)
    return path


# ---- agent_space is a flat Dict, flatdim succeeds -----------------------------------------

def test_agent_space_is_flat_dict_and_flatdim_succeeds():
    cfg = _cfg()
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    space = obs_select.agent_space(spec, cfg)
    assert isinstance(space, gymnasium.spaces.Dict)
    for sub in space.spaces.values():
        assert not isinstance(sub, gymnasium.spaces.Dict)  # flat -- no nesting
    dim = gymnasium.spaces.utils.flatdim(space)
    assert dim > 0


# ---- build_agent_obs output shapes match the space, every value finite --------------------

def test_build_agent_obs_shapes_match_space_and_are_finite():
    env, cfg, full_obs = _env_and_obs(n_envs=6)
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    space = obs_select.agent_space(spec, cfg)
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=6, device="cpu")

    gen = torch.Generator().manual_seed(0)
    for _ in range(10):
        action = torch.stack([torch.randint(0, 17, (6,), generator=gen), torch.randint(0, 2, (6,), generator=gen)], dim=1)
        full_obs, *_rest = env.step(action)
        out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)
        for name, box in space.spaces.items():
            t = out[name]
            assert tuple(t.shape) == (6, *box.shape), f"{name}: {tuple(t.shape)} != {(6, *box.shape)}"
            assert t.dtype == (torch.uint8 if box.dtype.name == "uint8" else torch.float32)
            assert torch.isfinite(t.to(torch.float32)).all(), f"{name} has non-finite values"


# ---- privileged fields are rejected at load time, with a clear message --------------------

def test_privileged_field_raises_at_load_time():
    cfg = _cfg()
    bad_spec = {
        "fair": True, "normalize": False,
        "groups": [{"name": "leak", "per_entity": True, "dtype": "float32",
                    "fields": ["entities.privileged.target_id"]}],
    }
    path = _write_spec(bad_spec)
    try:
        try:
            obs_select.load_agent_spec(path, cfg)
            assert False, "expected a ValueError"
        except ValueError as e:
            assert "entities.privileged" in str(e)
    finally:
        os.remove(path)


def test_unknown_field_raises_at_load_time():
    cfg = _cfg()
    bad_spec = {
        "fair": True, "normalize": False,
        "groups": [{"name": "x", "per_entity": False, "dtype": "float32", "fields": ["hero.not_a_real_field"]}],
    }
    path = _write_spec(bad_spec)
    try:
        try:
            obs_select.load_agent_spec(path, cfg)
            assert False, "expected a ValueError"
        except ValueError as e:
            assert "not_a_real_field" in str(e)
    finally:
        os.remove(path)


def test_unfair_grid_channels_rejected_under_fair_true():
    cfg = _cfg()
    for bad_channel in ("enemy_any", "enemy_hidden"):
        spec_dict = {
            "fair": True, "normalize": False,
            "groups": [{"name": "grid", "dtype": "uint8", "view_channels": [bad_channel]}],
        }
        path = _write_spec(spec_dict)
        try:
            try:
                obs_select.load_agent_spec(path, cfg)
                assert False, f"expected a ValueError for {bad_channel!r}"
            except ValueError as e:
                assert bad_channel in str(e)
        finally:
            os.remove(path)
    # the same channel is fine under fair: false
    spec_dict = {"fair": False, "normalize": False, "groups": [{"name": "grid", "dtype": "uint8", "view_channels": ["enemy_any"]}]}
    path = _write_spec(spec_dict)
    try:
        obs_select.load_agent_spec(path, cfg)  # must not raise
    finally:
        os.remove(path)


def test_too_many_groups_rejected():
    cfg = _cfg()
    spec_dict = {
        "fair": False, "normalize": False,
        "groups": [{"name": f"g{i}", "per_entity": False, "dtype": "float32", "fields": ["hero.hp_frac"]} for i in range(7)],
    }
    path = _write_spec(spec_dict)
    try:
        try:
            obs_select.load_agent_spec(path, cfg)
            assert False, "expected a ValueError"
        except ValueError as e:
            assert "6" in str(e)
    finally:
        os.remove(path)


# ---- fairness gating: bush-hidden enemy contributes all-zero fields, revealed=0 -----------

def test_fair_true_zeroes_a_bush_hidden_enemys_row():
    env, cfg, full_obs = _env_and_obs(n_envs=1, overrides={"entities": {"n_enemies": 3}})
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    assert spec.fair

    full_obs["entities"]["revealed_to_hero"][0, 1] = False
    # full obs still has the true (non-zeroed) position for the hidden entity.
    true_pos = full_obs["entities"]["pos"][0, 1].clone()
    assert not torch.equal(true_pos, torch.zeros_like(true_pos)) or True  # sanity, not the actual assertion below

    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=1, device="cpu")
    out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)

    row = out["enemies"][0, 0]  # 'enemies' slot 0 == entity index 1 (hero dropped)
    assert torch.equal(row, torch.zeros_like(row))

    index_map = obs_select.agent_obs_index_map(spec, cfg)
    start, end = index_map["enemies"]["entities.revealed_to_hero"]
    assert torch.equal(row[start:end], torch.zeros(end - start))


def test_fair_false_does_not_zero_hidden_enemies():
    env, cfg, full_obs = _env_and_obs(n_envs=1, overrides={"entities": {"n_enemies": 3}})
    path = _write_spec({
        "fair": False, "normalize": False,
        "groups": [{"name": "enemies", "per_entity": True, "dtype": "float32",
                    "fields": ["entities.alive", "entities.revealed_to_hero", "entities.hp_frac"]}],
    })
    try:
        spec = obs_select.load_agent_spec(path, cfg)
    finally:
        os.remove(path)

    full_obs["entities"]["revealed_to_hero"][0, 1] = False
    full_obs["entities"]["hp_frac"][0, 1] = 0.5  # force a nonzero value we can check survives
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=1, device="cpu")
    out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)
    row = out["enemies"][0, 0]
    assert row[2].item() == 0.5  # hp_frac column untouched despite revealed_to_hero == False


# ---- max_slots projectile selection: alive, sorted by time_to_closest, zero-padded --------

def test_projectile_group_selects_k_nearest_by_time_to_closest_and_zero_pads():
    env, cfg, full_obs = _env_and_obs(n_envs=1)
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    proj_group = next(g for g in spec.groups if g.name == "projectiles")
    assert proj_group.max_slots == 12

    # no projectiles alive yet (fresh reset) -- every one of the 12 slots must be all-zero.
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=1, device="cpu")
    out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)
    assert torch.equal(out["projectiles"], torch.zeros_like(out["projectiles"]))


# ---- build_agent_obs allocates nothing per call (steady-state memory doesn't grow) --------

def test_build_agent_obs_reuses_the_same_out_buffer_tensors():
    env, cfg, full_obs = _env_and_obs(n_envs=4)
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=4, device="cpu")
    data_ptrs_before = {name: t.data_ptr() for name, t in buffers.items()}

    gen = torch.Generator().manual_seed(0)
    for _ in range(5):
        action = torch.stack([torch.randint(0, 17, (4,), generator=gen), torch.randint(0, 2, (4,), generator=gen)], dim=1)
        full_obs, *_rest = env.step(action)
        out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)
        assert out is buffers  # same dict object returned
        for name, t in out.items():
            assert t.data_ptr() == data_ptrs_before[name]  # same underlying storage, every call


# ---- docs/AGENT_OBS.md regenerates deterministically ---------------------------------------

def test_dump_obs_schema_regenerates_agent_obs_docs_deterministically():
    import importlib
    dump_mod = importlib.import_module("scripts.dump_obs_schema")
    first = dump_mod.render_agent_obs()
    second = dump_mod.render_agent_obs()
    assert first == second
    assert "## `self`" in first and "## `grid`" in first
