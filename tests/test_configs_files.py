"""Exercises the actual files under configs/ against brawl_sim.config, per Step 4's acceptance
criteria. Step 3's tests/test_config.py already covers the loader machinery in isolation with
inline fixtures; this file is the end-to-end check that the real files parse correctly.
"""
from pathlib import Path

import pytest
import torch
import yaml

from brawl_sim.config import (
    build_params,
    load_config,
    load_randomization,
    apply_randomization,
    validate,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
PRESETS = CONFIGS / "presets"


def _merged_spec():
    default = yaml.safe_load((CONFIGS / "default.yaml").read_text())
    brawlers = yaml.safe_load((CONFIGS / "brawlers.yaml").read_text())
    return {**default, **brawlers}


def _build_and_validate(overrides=None, n_envs=8):
    cfg = load_config(CONFIGS / "default.yaml", overrides=overrides)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    params = build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=_merged_spec())
    validate(cfg, params)
    return cfg, params


def test_default_config_loads_and_validates():
    _build_and_validate()


@pytest.mark.parametrize("preset_name", ["debug_tiny", "no_zone", "single_archetype"])
def test_preset_loads_and_validates(preset_name):
    overrides = yaml.safe_load((PRESETS / f"{preset_name}.yaml").read_text())
    _build_and_validate(overrides=overrides)


def test_debug_tiny_yields_three_entities():
    overrides = yaml.safe_load((PRESETS / "debug_tiny.yaml").read_text())
    cfg, _ = _build_and_validate(overrides=overrides)
    assert cfg.n_entities == 3
    assert cfg.map_h == 20 and cfg.map_w == 20
    assert cfg.view_h == 10 and cfg.view_w == 14
    assert cfg.max_episode_steps == 300
    assert cfg.zone_enabled is False
    assert cfg.obs_include_world_grid is False


def test_single_archetype_preset_pins_sniper():
    overrides = yaml.safe_load((PRESETS / "single_archetype.yaml").read_text())
    cfg, _ = _build_and_validate(overrides=overrides)
    assert cfg.randomize_enemy_types is False
    assert cfg.fixed_enemy_types == ("sniper",) * cfg.n_enemies


def test_randomization_file_ships_fully_commented():
    spec = load_randomization(CONFIGS / "randomization.yaml")
    assert spec == {}


def test_uncommenting_one_randomization_line_only_adds_variation():
    cfg = load_config(CONFIGS / "default.yaml")
    base_spec = _merged_spec()

    gen_a = torch.Generator(device="cpu")
    gen_a.manual_seed(42)
    baseline = build_params(cfg, n_envs=64, device="cpu", gen=gen_a, spec=base_spec)
    assert torch.all(baseline.enemy_hp_mult == baseline.enemy_hp_mult[0])

    # Simulate uncommenting exactly the entities.enemy_hp_mult line from the real file.
    raw_text = (CONFIGS / "randomization.yaml").read_text()
    target = "# entities.enemy_hp_mult:       {low: 0.8, high: 1.25}"
    assert target in raw_text, "randomization.yaml's worked example changed out from under this test"
    uncommented_text = raw_text.replace(target, target.lstrip("# "), 1)

    # Parse the mutated text the same way load_randomization does, without needing a temp file.
    from brawl_sim.config import _parse_range_entry  # test-only introspection

    raw = yaml.safe_load(uncommented_text)
    randomization = {k: _parse_range_entry(v) for k, v in raw.items()}
    assert randomization == {"entities.enemy_hp_mult": {"low": 0.8, "high": 1.25, "mode": "additive"}}

    merged = apply_randomization(dict(base_spec), randomization)
    gen_b = torch.Generator(device="cpu")
    gen_b.manual_seed(42)
    randomized = build_params(cfg, n_envs=64, device="cpu", gen=gen_b, spec=merged)

    # The one uncommented field now varies per env...
    assert not torch.all(randomized.enemy_hp_mult == randomized.enemy_hp_mult[0])
    assert randomized.enemy_hp_mult.min() >= 0.8 and randomized.enemy_hp_mult.max() <= 1.25
    # ...and nothing else changed: every other field matches the baseline build exactly
    # (same seed, same spec apart from the one overridden leaf).
    assert torch.equal(randomized.base_hp, baseline.base_hp)
    assert torch.equal(randomized.move_speed, baseline.move_speed)
    assert torch.equal(randomized.enemy_damage_mult, baseline.enemy_damage_mult)
    assert torch.equal(randomized.zone_start_time, baseline.zone_start_time)


def test_agent_obs_yaml_loads_as_a_real_agent_spec():
    # Step 4 only needed the file to exist and parse as a placeholder; Step 32 replaced it with
    # the real schema (brawl_sim.core.obs_select) -- this is the end-to-end check for that file,
    # the same role every other test in this module plays for configs/default.yaml et al.
    from brawl_sim.core.obs_select import load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    spec = load_agent_spec(CONFIGS / "agent_obs.yaml", cfg)
    assert spec.fair is True
    assert len(spec.groups) >= 1
    assert {g.name for g in spec.groups} >= {"self", "enemies", "projectiles", "zone", "grid"}
