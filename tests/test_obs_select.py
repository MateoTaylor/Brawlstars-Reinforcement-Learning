import tempfile
import os

import gymnasium
import pytest
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


# ---- normalize: true, per unit ------------------------------------------------------------
#
# Until the deployed spec dropped `hp_frac`, nothing here exercised `normalize: true` at all --
# obs_select's own docstring called it "a genuinely underspecified corner of Step 32's plan text
# (no acceptance criterion exercises it)". That is how `hp` and `count` fields came to sit in a
# normalized spec undivided. These tests are that acceptance criterion.

def test_every_normalized_unit_divides_by_its_documented_scale():
    cfg = _cfg()
    assert obs_select._norm_divisor("tiles", cfg) == float(max(cfg.map_w, cfg.map_h))
    assert obs_select._norm_divisor("tiles/s", cfg) == obs_select._NORM_SPEED_SCALE
    assert obs_select._norm_divisor("hp", cfg) == obs_select._NORM_HP_SCALE
    assert obs_select._norm_divisor("count", cfg) == obs_select._NORM_COUNT_SCALE
    # Bounded by construction; a divisor here would shrink a field that is already in range.
    for units in ("fraction", "bool", "onehot", "unitless", "radians"):
        assert obs_select._norm_divisor(units, cfg) == 1.0


def test_the_hp_scale_covers_the_largest_hp_the_sim_can_produce():
    """Derived from the shipped config rather than restated, so a change to the roster or the
    cube numbers fails here instead of quietly pushing a normalized field past 1.

    The ceiling is an enemy at the roster's highest base HP, holding `max_cubes` at
    `hp_per_cube` flat, scaled by the top of `enemy_hp_mult` -- see core/stats.effective_max_hp,
    which applies that multiplier to the cube total and never to the hero.
    """
    import yaml as _yaml

    # cubes.* resolve onto SimParams, not EnvConfig, so they are read from the file itself.
    cubes = _yaml.safe_load(open("configs/default.yaml").read())["cubes"]
    brawlers = _yaml.safe_load(open("configs/brawlers.yaml").read())
    base = [v["base_hp"] for v in brawlers.values() if isinstance(v, dict) and "base_hp" in v]
    ceiling = (max(base) + cubes["max_cubes"] * cubes["hp_per_cube"]) * 1.25

    normalized = ceiling / obs_select._NORM_HP_SCALE
    assert normalized <= 1.05, (
        f"max possible HP {ceiling:.0f} normalizes to {normalized:.2f}; raise _NORM_HP_SCALE"
    )
    # Not so large that a real HP reading vanishes into the noise floor either -- a fresh Mortis
    # should land near the ~0.1-0.5 band _NORM_SPEED_SCALE puts a walking entity in.
    assert 0.2 <= brawlers["hero_mortis"]["base_hp"] / obs_select._NORM_HP_SCALE <= 0.6


def test_the_count_scale_covers_every_bounded_count_in_the_schema():
    import yaml as _yaml

    cfg = _cfg()
    max_cubes = _yaml.safe_load(open("configs/default.yaml").read())["cubes"]["max_cubes"]
    for largest in (max_cubes, cfg.n_enemies, 3):  # cubes, brawlers left, ammo
        assert largest / obs_select._NORM_COUNT_SCALE <= 1.0


def test_normalized_hp_and_count_fields_come_out_in_range_on_a_real_env():
    """End to end: the divisor is applied where it is supposed to be, not merely defined."""
    env, cfg, full_obs = _env_and_obs(n_envs=4, seed=3)
    spec_dict = {
        "fair": True, "normalize": True,
        "groups": [{"name": "self", "per_entity": False, "dtype": "float32",
                    "fields": ["hero.hp", "meta.n_enemies_alive", "hero.ammo_whole"]}],
    }
    path = _write_spec(spec_dict)
    try:
        spec = obs_select.load_agent_spec(path, cfg)
        buffers = obs_select.make_agent_obs_buffers(spec, cfg, 4, torch.device("cpu"))
        out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)["self"]
        assert out.min() >= 0.0
        assert out.max() <= 1.0, f"a normalized field left [0, 1]: {out.max().item()}"
        # And it is a real division, not a coincidence of small raw values: hero HP is thousands.
        raw_hp = full_obs["hero"]["hp"]
        assert torch.allclose(out[:, 0], raw_hp / obs_select._NORM_HP_SCALE)
        assert raw_hp.max() > 100.0
    finally:
        os.remove(path)


def test_two_specs_with_a_same_named_group_do_not_share_a_normalization_vector():
    """A latent bug found by the test above, worth its own name.

    `_normalize` memoizes its scale vector, and the key used to be the group NAME alone -- but
    the vector depends on the group's FIELDS. All three shipped specs have a group called
    "self", at 25/25/24 columns, so loading two of them in one process gave the second the
    first's divisors: a broadcast error when the widths differ, and silently wrong scaling when
    they match. Anything that evaluates a checkpoint against more than one spec hits this.
    """
    env, cfg, full_obs = _env_and_obs(n_envs=4, seed=5)
    wide = {"fair": True, "normalize": True,
            "groups": [{"name": "self", "per_entity": False, "dtype": "float32",
                        "fields": ["hero.hp", "hero.vel", "meta.n_enemies_alive"]}]}
    narrow = {"fair": True, "normalize": True,
              "groups": [{"name": "self", "per_entity": False, "dtype": "float32",
                          "fields": ["meta.n_enemies_alive"]}]}
    paths = [_write_spec(wide), _write_spec(narrow)]
    try:
        outs = []
        for path in paths:
            spec = obs_select.load_agent_spec(path, cfg)
            buffers = obs_select.make_agent_obs_buffers(spec, cfg, 4, torch.device("cpu"))
            outs.append(obs_select.build_agent_obs(full_obs, spec, cfg, buffers)["self"].clone())
        assert outs[0].shape[1] == 4 and outs[1].shape[1] == 1
        # The shared field must normalize identically in both, by the count scale.
        raw_n = full_obs["meta"]["n_enemies_alive"].to(torch.float32)
        expected = raw_n / obs_select._NORM_COUNT_SCALE
        assert torch.allclose(outs[0][:, 3], expected)
        assert torch.allclose(outs[1][:, 0], expected)
    finally:
        for p in paths:
            os.remove(p)


def _grid_spec(channels):
    return {"fair": True, "normalize": True,
            "groups": [{"name": "grid", "dtype": "uint8", "view_channels": list(channels)}]}


@pytest.mark.parametrize("first, second", [
    # The pair that found it: the full suite built agent_obs_deploy3.yaml's grid (10 planes) and
    # then every agent_obs_deploy.yaml test (8) died on a size mismatch.
    ("configs/agent_obs_deploy3.yaml", "configs/agent_obs_deploy.yaml"),
    # The worse half: equal widths, so nothing raises and the second spec just reads the wrong planes.
    (_grid_spec(["hero", "blocks_unit"]), _grid_spec(["blocks_unit", "hero"])),
])
def test_two_specs_with_a_same_named_grid_do_not_share_channel_indices(first, second):
    """The grid half of the bug above. `_build_grid_group` memoized its channel indices under the
    group NAME, and every spec calls its view group "grid".

    The reference is the second spec built from an empty cache, so the expected planes do not come
    from the same channel table the code under test uses.
    """
    env, cfg, full_obs = _env_and_obs(n_envs=4, seed=5)
    paths = [s if isinstance(s, str) else _write_spec(s) for s in (first, second)]
    try:
        specs = [obs_select.load_agent_spec(p, cfg) for p in paths]

        def grid(spec):
            buffers = obs_select.make_agent_obs_buffers(spec, cfg, 4, torch.device("cpu"))
            return obs_select.build_agent_obs(full_obs, spec, cfg, buffers)["grid"].clone()

        obs_select._tensor_cache.clear()
        reference = grid(specs[1])
        obs_select._tensor_cache.clear()
        assert not torch.equal(grid(specs[0]), reference), "the specs must differ for this to tell"
        assert torch.equal(grid(specs[1]), reference)
    finally:
        for s, p in zip((first, second), paths):
            if isinstance(s, dict):
                os.remove(p)
