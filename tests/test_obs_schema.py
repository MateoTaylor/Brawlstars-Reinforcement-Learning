import copy

import numpy as np
import pytest
import torch
import yaml

from brawl_sim.bots import perception
from brawl_sim.config import build_params, load_config
from brawl_sim.core import obs_schema as schema
from brawl_sim.core import observation as obs_mod
from brawl_sim.core import spawn
from brawl_sim.core.state import allocate
from brawl_sim.maps.loader import build_map_bank

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())


def _cfg_params_bank(n_envs=8, seed=0, overrides=None, tiny=True):
    merged = dict(CONFIGS_TINY) if tiny else {}
    if overrides:
        merged = {**merged, **overrides}
    cfg = load_config(CONFIGS_DEFAULT, overrides=merged or None)
    spec = {
        **yaml.safe_load(open(CONFIGS_DEFAULT).read()),
        **yaml.safe_load(open("configs/brawlers.yaml").read()),
    }
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    params = build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=spec)
    bank = build_map_bank(cfg, device="cpu")
    return cfg, params, bank, gen, spec


def _obs(n_envs=8, overrides=None, tiny=True):
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=n_envs, overrides=overrides, tiny=tiny)
    state = allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)
    mask = torch.ones(n_envs, dtype=torch.bool)
    spawn.reset_envs(state, mask, bank, params, cfg, gen, spec)
    vis = perception.visibility(state, bank, params, cfg)
    los = perception.raw_los(state, bank, cfg)
    obs = obs_mod.build_obs(state, bank, vis, los, params, cfg)
    return cfg, obs


# ---- set-comparison: build_obs output <-> obs_spec(cfg) (Step 26's core acceptance test) -----

def test_every_build_obs_field_is_in_schema_and_vice_versa_both_toggles_on():
    cfg, obs = _obs(overrides={"observation": {"include_world_grid": True, "include_privileged": True}})
    flat = schema._flatten(obs)
    spec = schema.obs_spec(cfg)
    assert set(flat.keys()) == set(spec.keys())


def test_every_build_obs_field_is_in_schema_and_vice_versa_both_toggles_off():
    cfg, obs = _obs(overrides={"observation": {"include_world_grid": False, "include_privileged": False}})
    flat = schema._flatten(obs)
    spec = schema.obs_spec(cfg)
    assert set(flat.keys()) == set(spec.keys())
    assert "world" not in flat
    assert not any(k.startswith("entities.privileged.") for k in flat)


def test_obs_spec_shapes_resolve_against_cfg():
    cfg, obs = _obs(overrides={"observation": {"include_world_grid": True}})
    spec = schema.obs_spec(cfg)
    assert spec["entities.pos"].shape == ("N", cfg.n_entities, 2)
    assert spec["projectiles.pos"].shape == ("N", cfg.max_projectiles, 2)
    assert spec["boxes.alive"].shape == ("N", cfg.max_boxes)
    assert spec["pickups.alive"].shape == ("N", cfg.max_pickups)
    # 12 base channels, then one enemy_hist plane per history slot (Step H2, history_frames 3).
    assert spec["view"].shape == ("N", 15, cfg.view_h, cfg.view_w)
    assert spec["world"].shape == ("N", 15, cfg.map_h, cfg.map_w)
    assert spec["action_mask.move"].shape == ("N", cfg.n_move_bins + 1)


# ---- validate_obs -----------------------------------------------------------------------

def test_validate_obs_passes_on_real_observation():
    cfg, obs = _obs(overrides={"observation": {"include_world_grid": True, "include_privileged": True}})
    schema.validate_obs(obs, cfg)  # must not raise

    cfg2, obs2 = _obs(overrides={"observation": {"include_world_grid": False, "include_privileged": False}})
    schema.validate_obs(obs2, cfg2)  # must not raise either


def test_validate_obs_fails_on_missing_field():
    cfg, obs = _obs()
    del obs["hero"]["pos"]
    try:
        schema.validate_obs(obs, cfg)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "hero.pos" in str(exc)


def test_validate_obs_fails_on_wrong_shape():
    cfg, obs = _obs()
    obs["hero"]["pos"] = obs["hero"]["pos"][:, :1]  # (N,1) instead of (N,2)
    try:
        schema.validate_obs(obs, cfg)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "hero.pos" in str(exc) and "shape" in str(exc)


def test_validate_obs_fails_on_wrong_dtype():
    cfg, obs = _obs()
    obs["hero"]["cubes"] = obs["hero"]["cubes"].to(torch.float32)
    try:
        schema.validate_obs(obs, cfg)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "hero.cubes" in str(exc) and "dtype" in str(exc)


def test_validate_obs_fails_on_nan():
    cfg, obs = _obs()
    obs["hero"]["hp"] = obs["hero"]["hp"].clone()
    obs["hero"]["hp"][0] = float("nan")
    try:
        schema.validate_obs(obs, cfg)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "hero.hp" in str(exc)


# ---- describe_obs -------------------------------------------------------------------------

def test_describe_obs_fits_on_one_screen_debug_tiny():
    cfg, obs = _obs(n_envs=8, overrides={"observation": {"include_world_grid": False}})
    text = schema.describe_obs(obs, env_index=0)
    lines = text.split("\n")
    assert len(lines) <= 60, f"describe_obs produced {len(lines)} lines, expected to fit one screen"
    assert all(len(line) <= 200 for line in lines)


def test_describe_obs_reflects_real_values():
    cfg, obs = _obs(n_envs=1)
    text = schema.describe_obs(obs, env_index=0)
    assert "hero" in text
    assert "entities" in text
    assert f"map={int(obs['meta']['map_id'][0])}" in text


# ---- to_numpy ------------------------------------------------------------------------------

def test_to_numpy_preserves_structure_and_values():
    cfg, obs = _obs(overrides={"observation": {"include_world_grid": True, "include_privileged": True}})
    np_obs = schema.to_numpy(obs)
    assert isinstance(np_obs["hero"]["pos"], np.ndarray)
    assert np.allclose(np_obs["hero"]["pos"], obs["hero"]["pos"].numpy())
    assert isinstance(np_obs["entities"]["privileged"]["target_id"], np.ndarray)
    assert np_obs["view"].dtype == np.uint8
    assert np_obs["view"].shape == tuple(obs["view"].shape)


# ---- dump_obs_schema.py regeneration -------------------------------------------------------

def test_render_is_deterministic():
    from scripts.dump_obs_schema import render
    a = render()
    b = render()
    assert a == b


def test_render_covers_every_schema_field():
    from scripts.dump_obs_schema import render
    text = render()
    for name in schema.OBS_SCHEMA:
        assert f"`{name}`" in text, f"{name} missing from generated docs"


# ---- the gadget rows (SIM_OVERHAUL Step G4) --------------------------------------------------

def test_the_gadget_rows_are_declared_as_the_plan_specifies():
    ready = schema.OBS_SCHEMA["hero.gadget_ready"]
    frac = schema.OBS_SCHEMA["hero.gadget_charge_frac"]
    assert (ready.shape, ready.dtype, ready.units, ready.range) == (("N",), "bool", "bool", None)
    assert (frac.shape, frac.dtype, frac.units, frac.range) == (("N",), "float32", "fraction", (0.0, 1.0))
    assert (ready.privileged, ready.conditional, frac.privileged, frac.conditional) == (False, None, False, None)


def test_the_hero_rows_are_in_build_obs_order_with_the_gadget_after_the_long_dash():
    """The module docstring's promise, which dump_obs_schema.py's top-to-bottom render relies on."""
    cfg, obs = _obs(n_envs=1)
    built = [k for k in schema._flatten(obs) if k.startswith("hero.")]
    declared = [k for k in schema.OBS_SCHEMA if k.startswith("hero.")]
    assert built == declared
    i = declared.index("hero.long_dash_frac")
    assert declared[i + 1:i + 4] == ["hero.gadget_ready", "hero.gadget_charge_frac", "hero.attack_idle_t"]


def test_validate_obs_requires_both_gadget_fields():
    for name in ("gadget_ready", "gadget_charge_frac"):
        cfg, obs = _obs(n_envs=2)
        del obs["hero"][name]
        with pytest.raises(ValueError, match=f"hero.{name}"):
            schema.validate_obs(obs, cfg)


def test_describe_obs_shows_the_gadget():
    cfg, obs = _obs(n_envs=1)
    assert "  gadget_ready: T" in schema.describe_obs(obs, env_index=0).split("\n")


# ---- the history rows (SIM_OVERHAUL Step H2) ---------------------------------------------------

_HIST_ROWS = ["hist.valid", "hist.move_onehot", "hist.attack_onehot", "hist.hp", "hist.ammo_frac",
              "hist.displacement"]


def test_the_hist_rows_are_declared_as_the_plan_specifies():
    rows = {name: (s.shape, s.dtype, s.units, s.range, s.privileged, s.conditional)
            for name, s in schema.OBS_SCHEMA.items() if name.startswith("hist.")}
    assert rows == {
        "hist.valid": (("N", "K"), "bool", "bool", None, False, None),
        "hist.move_onehot": (("N", "K", "MOVE"), "uint8", "onehot", (0, 1), False, None),
        "hist.attack_onehot": (("N", "K", 4), "uint8", "onehot", (0, 1), False, None),
        "hist.hp": (("N", "K"), "float32", "hp", (0.0, None), False, None),
        "hist.ammo_frac": (("N", "K"), "float32", "fraction", (0.0, 1.0), False, None),
        "hist.displacement": (("N", "K", 2), "float32", "tiles", None, False, None),
    }


def test_every_row_is_in_build_obs_order_with_hist_right_after_hero():
    """The module docstring's promise for the whole table, not only the hero rows: describe_obs
    and dump_obs_schema.py both render in OBS_SCHEMA's order."""
    cfg, obs = _obs(overrides={"observation": {"include_world_grid": True, "include_privileged": True}})
    declared = list(schema.obs_spec(cfg))
    assert list(schema._flatten(obs)) == declared
    i = declared.index("hero.rank")
    assert declared[i + 1:i + 8] == _HIST_ROWS + ["entities.alive"]


@pytest.mark.parametrize("frames, channels", [(3, 15), (1, 13), (4, 16)])
def test_k_and_c_resolve_from_history_frames(frames, channels):
    cfg, obs = _obs(n_envs=2, overrides={"observation": {"history_frames": frames, "include_world_grid": True}})
    spec = schema.obs_spec(cfg)
    assert spec["hist.valid"].shape == ("N", frames)
    assert spec["hist.move_onehot"].shape == ("N", frames, 17)
    assert spec["hist.displacement"].shape == ("N", frames, 2)
    assert spec["view"].shape == ("N", channels, 10, 14)
    assert spec["world"].shape == ("N", channels, 20, 20)
    schema.validate_obs(obs, cfg)  # and the builder agrees at every K


def test_validate_obs_requires_every_hist_field():
    for name in _HIST_ROWS:
        cfg, obs = _obs(n_envs=2)
        del obs["hist"][name.split(".", 1)[1]]
        with pytest.raises(ValueError, match=name):
            schema.validate_obs(obs, cfg)


def test_the_docs_put_hist_between_hero_and_entities_and_define_k_and_c():
    from scripts.dump_obs_schema import render
    text = render()
    assert text.index("## `hero`") < text.index("## `hist`") < text.index("## `entities`")
    assert "`K` = history_frames" in text
    assert "`C` = 12 + history_frames" in text
