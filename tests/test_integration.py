"""End-to-end integration tests spanning multiple modules. See BRAWL_SIM_BUILD_PLAN.md Step 41.

**Not every bullet in Step 41's own list gets a NEW test here.** Three of the eleven (autoreset
boundary, dash-overrides-walk, the SB3 terminal_observation/TimeLimit.truncated/episode
contract) already have thorough, dedicated coverage elsewhere (`tests/test_autoreset.py`,
`tests/test_env.py::test_dashing_tick_shows_only_dash_displacement_not_walk` +
`tests/test_hero.py`, `tests/test_sb3_vecenv.py`) -- duplicating them here would just be a
second copy that can drift, not more safety. This file adds ONE light end-to-end check for
autoreset's boundary anyway (below), specifically because it's the one property the OTHER eight
tests in this file lean on implicitly (a wrong autoreset boundary would silently corrupt the
determinism/isolation/config-override tests too), but does not re-derive
`test_autoreset.py`'s full per-env-isolation matrix. The other seven bullets (determinism,
isolation, randomization end-to-end, fairness gating via the REAL shipped `configs/
agent_obs.yaml`, exhaustive privileged-field isolation, combined config overrides, sync-free
native stepping) had real, documented gaps against genuine multi-module/end-to-end coverage --
see BRAWL_SIM_BUILD_PLAN.md Step 41's implementation notes for the specific gap each one closes.
"""
import dataclasses

import pytest
import torch
import yaml

from brawl_sim.config import load_config, load_randomization
from brawl_sim.core import obs_select
from brawl_sim.core.obs_schema import OBS_SCHEMA
from brawl_sim.env import BrawlVecEnv

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())
AGENT_OBS_YAML = "configs/agent_obs.yaml"
RANDOMIZATION_YAML = "configs/randomization.yaml"


def _cfg(overrides=None):
    merged = dict(CONFIGS_TINY)
    if overrides:
        merged = {**merged, **overrides}
    return load_config(CONFIGS_DEFAULT, overrides=merged)


def _random_action(n_envs, gen):
    move = torch.randint(0, 17, (n_envs,), generator=gen)
    fire = torch.randint(0, 2, (n_envs,), generator=gen)
    return torch.stack([move, fire], dim=1)


# ---- determinism: two independent envs, same seed, same actions -> bit-identical -------------

def test_same_seed_same_actions_produces_bit_identical_trajectories():
    cfg = _cfg(overrides={"entities": {"n_enemies": 3}})
    env_a = BrawlVecEnv(cfg, n_envs=6, device="cpu", seed=7, verbose=False)
    env_b = BrawlVecEnv(cfg, n_envs=6, device="cpu", seed=7, verbose=False)
    env_a.reset()
    env_b.reset()

    action_gen = torch.Generator().manual_seed(123)
    actions = [_random_action(6, action_gen) for _ in range(20)]

    for name in env_a.state.__slots__:
        assert torch.equal(getattr(env_a.state, name), getattr(env_b.state, name)), f"{name} differs after reset"

    for action in actions:
        env_a.step(action.clone())
        env_b.step(action.clone())
        for name in env_a.state.__slots__:
            ta, tb = getattr(env_a.state, name), getattr(env_b.state, name)
            assert torch.equal(ta, tb), f"{name} diverged between two identically-seeded envs"


# ---- isolation: perturbing env 0's actions never touches envs 1..N-1 --------------------------

def test_perturbing_only_env_0s_actions_leaves_every_other_env_bit_identical():
    cfg = _cfg(overrides={"entities": {"n_enemies": 3}})
    n_envs = 5
    control = BrawlVecEnv(cfg, n_envs=n_envs, device="cpu", seed=3, verbose=False)
    perturbed = BrawlVecEnv(cfg, n_envs=n_envs, device="cpu", seed=3, verbose=False)
    control.reset()
    perturbed.reset()

    gen = torch.Generator().manual_seed(99)
    for _ in range(15):
        action = _random_action(n_envs, gen)
        control.step(action.clone())
        perturbed_action = action.clone()
        perturbed_action[0] = torch.tensor([16, 1]) if perturbed_action[0, 0] != 16 else torch.tensor([0, 0])
        perturbed.step(perturbed_action)

    for name in control.state.__slots__:
        c, p = getattr(control.state, name), getattr(perturbed.state, name)
        assert torch.equal(c[1:], p[1:]), f"{name}: envs 1..N-1 diverged after only env 0's action changed"


# ---- autoreset boundary: the one light check this file adds on top of test_autoreset.py -------

def test_final_observation_holds_the_pre_reset_state_not_the_post_reset_one():
    cfg = _cfg(overrides={"entities": {"n_enemies": 1}})
    env = BrawlVecEnv(cfg, n_envs=2, device="cpu", seed=0, verbose=False)
    env.reset()
    env.state.ent_hp[0, 0] = 0.0
    env.state.ent_last_hit_by[0, 0] = -1

    obs, reward, terminated, truncated, info = env.step(torch.zeros(2, 2, dtype=torch.int64))
    assert bool(terminated[0])
    # the returned obs is the FRESH episode's (hero alive again); final_observation is the DEATH tick's
    assert bool(obs["hero"]["alive"][0])
    assert not bool(info["final_observation"]["hero"]["alive"][0])
    assert float(info["final_observation"]["hero"]["hp"][0]) == 0.0


# ---- randomization: a ranged param varies across envs and resamples on reset via the public API --

def test_randomized_param_varies_across_envs_and_resamples_on_reset():
    """configs/randomization.yaml ships with every line commented out (its own docstring: a
    template to uncomment from, not a default). load_randomization(RANDOMIZATION_YAML) on the
    real file returns {} -- correct, but useless for this test. Uncommenting one of its own
    real lines by hand (matching tests/test_configs_files.py's own established approach for
    the same file) proves the end-to-end path works without asserting on a moving/absent
    default."""
    cfg = load_config(CONFIGS_DEFAULT, overrides=CONFIGS_TINY)
    assert load_randomization(RANDOMIZATION_YAML) == {}  # documents the "ships empty" fact above
    randomization = {"entities.enemy_hp_mult": {"low": 0.8, "high": 1.25, "mode": "additive"}}

    env = BrawlVecEnv(cfg, n_envs=64, device="cpu", seed=0, randomization=randomization, verbose=False)
    env.reset()

    dotted, _ = next(iter(randomization.items()))
    attr = dotted.split(".")[-1] if "." in dotted else dotted
    # PER_KIND_FIELDS are (N,K); PER_ENV_FIELDS are (N,) -- both vary across the N axis either way.
    values = getattr(env.params, attr)
    assert values.shape[0] == 64
    flat = values.reshape(64, -1)
    assert not torch.all(flat == flat[0]), f"{dotted} did not vary across envs -- randomization isn't wired up"

    before = flat.clone()
    mask = torch.zeros(64, dtype=torch.bool)
    mask[:10] = True
    env.reset(mask)
    after = getattr(env.params, attr).reshape(64, -1)
    assert not torch.all(after[:10] == before[:10]), f"{dotted} did not resample on a masked reset"
    assert torch.all(after[10:] == before[10:]), f"{dotted} changed for envs NOT in the reset mask"


# ---- fairness gating, through the REAL shipped configs/agent_obs.yaml -------------------------

def _bush_hidden_env():
    cfg = load_config(CONFIGS_DEFAULT, overrides={
        "world": {"maps": ["bushy"], "map_selection": "fixed", "fixed_map": "bushy"},
        "entities": {"n_enemies": 1},
    })
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0, verbose=False)
    obs = env.reset()

    bush_mask = env.bank.is_bush[0]
    ys, xs = torch.nonzero(bush_mask, as_tuple=True)
    bush_pos = torch.stack([xs[0].float() + 0.5, ys[0].float() + 0.5])
    env.state.ent_pos[0, 1] = bush_pos
    env.state.ent_pos[0, 0] = bush_pos + torch.tensor([20.0, 0.0])
    env.state.ent_reveal_t[0, 1] = 0.0
    obs, *_ = env.step(torch.zeros(1, 2, dtype=torch.int64))
    assert not bool(obs["entities"]["revealed_to_hero"][0, 1])  # sanity: really hidden
    return env, cfg, obs


def test_shipped_agent_obs_yaml_zeroes_the_hidden_bots_row_when_fair():
    env, cfg, obs = _bush_hidden_env()
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    assert spec.fair is True  # configs/agent_obs.yaml's own shipped default
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, 1, "cpu")
    agent_obs = obs_select.build_agent_obs(obs, spec, cfg, buffers)
    assert torch.all(agent_obs["enemies"][0, 0] == 0.0), "hidden bot's row must be all-zero under fair=true"


def test_same_spec_with_fair_false_does_not_zero_the_hidden_bots_row():
    env, cfg, obs = _bush_hidden_env()
    fair_spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    unfair_spec = dataclasses.replace(fair_spec, fair=False)
    buffers = obs_select.make_agent_obs_buffers(unfair_spec, cfg, 1, "cpu")
    agent_obs = obs_select.build_agent_obs(obs, unfair_spec, cfg, buffers)
    assert torch.any(agent_obs["enemies"][0, 0] != 0.0), "fair=false must NOT zero the hidden bot's row"


# ---- privileged isolation: every entities.privileged.* field, not just one -------------------

def test_every_privileged_field_is_individually_rejected_by_load_agent_spec(tmp_path):
    privileged_fields = [name for name, spec in OBS_SCHEMA.items() if spec.privileged]
    assert len(privileged_fields) > 0, "obs_schema has no privileged fields -- this test would pass vacuously"

    for field in privileged_fields:
        raw = {
            "fair": True, "normalize": False,
            "groups": [{"name": "leak_test", "per_entity": True, "fields": [field]}],
        }
        path = tmp_path / "leaky_agent_obs.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="privileged"):
            obs_select.load_agent_spec(path, _cfg())


def test_real_agent_obs_yaml_never_produces_a_privileged_field_in_its_output():
    env, cfg, obs = _bush_hidden_env()
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    for group in spec.groups:
        for field in group.fields:
            assert not field.startswith("entities.privileged."), f"group {group.name!r} exposes {field!r}"


# ---- combined config overrides: every knob the plan names, changed together, still runs -------

def test_every_named_config_override_can_be_combined_and_the_env_still_runs():
    cfg = load_config(CONFIGS_DEFAULT, overrides={
        "entities": {"n_enemies": 2},
        "view": {"height": 8, "width": 10},
        # map_h/map_w must match the chosen map CSV's real dimensions (loader.load_map_csv
        # enforces this) -- "blank" is deliberately sized 20x20, same combo debug_tiny.yaml uses.
        "world": {"map_h": 20, "map_w": 20, "maps": ["blank"], "map_selection": "fixed", "fixed_map": "blank"},
        "zone": {"enabled": False},
        "sim": {"action_latency_seconds": 0.1, "max_episode_steps": 100},
        "observation": {"include_world_grid": False},
    })
    assert cfg.n_enemies == 2 and cfg.view_h == 8 and cfg.view_w == 10
    assert cfg.map_h == 20 and cfg.map_w == 20 and cfg.zone_enabled is False
    assert cfg.action_latency_seconds == 0.1 and cfg.obs_include_world_grid is False

    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    unfair_spec = dataclasses.replace(spec, fair=False)  # agent_obs.fair, the one non-EnvConfig knob

    env = BrawlVecEnv(cfg, n_envs=4, device="cpu", seed=0, verbose=False)
    obs = env.reset()
    assert "world" not in obs  # include_world_grid=False actually took effect
    buffers = obs_select.make_agent_obs_buffers(unfair_spec, cfg, 4, "cpu")
    obs_select.build_agent_obs(obs, unfair_spec, cfg, buffers)

    gen = torch.Generator().manual_seed(0)
    for _ in range(30):
        env.step(_random_action(4, gen))
    assert True  # ran 30 ticks with every named knob touched at once without raising


# ---- sync-free native stepping (CUDA only -- set_sync_debug_mode is a CUDA concept) ------------

def test_native_step_is_sync_free_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = _cfg(overrides={"entities": {"n_enemies": 3}})
    env = BrawlVecEnv(cfg, n_envs=64, device="cuda", seed=0, verbose=False)
    env.reset()
    gen = torch.Generator(device="cuda").manual_seed(0)
    action = _random_action(64, None).to("cuda")

    # a couple of untimed warmup steps first -- lazy first-call allocations are expected to
    # sync, that's not what this test is checking.
    for _ in range(3):
        env.step(action)
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(100):
            env.step(action)
    finally:
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("default")
