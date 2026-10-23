import numpy as np
import yaml

from brawl_sim.config import _deep_merge, load_config
from brawl_sim.core import obs_select
from brawl_sim.core.reward import ExampleReward
from brawl_sim.env import BrawlVecEnv
from brawl_sim.wrappers.gym_single import BrawlGymEnv

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())
AGENT_OBS_YAML = "configs/agent_obs.yaml"


def _make(seed=0, reward_fn=None, overrides=None):
    merged = dict(CONFIGS_TINY)
    if overrides:
        merged = _deep_merge(merged, overrides)  # the preset's other keys in a section stay
    cfg = load_config(CONFIGS_DEFAULT, overrides=merged)
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=seed)
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    return BrawlGymEnv(env, spec, reward_fn or ExampleReward())


# ---- the time limit ends an episode as truncated, obs dtypes match the space ---------------

def test_the_time_limit_ends_an_episode_truncated_and_obs_dtypes_match_space():
    # 2026-10-20: this stepped up to 300 random actions and accepted either ending, so a wrapper
    # that reported the clock as `terminated` passed, after up to 60 full steps of the tiny
    # preset. The clock now starts one tick short, and the one step must come back truncated.
    env = _make()
    obs, info = env.reset()
    assert info == {}
    for name, box in env.observation_space.spaces.items():
        assert obs[name].shape == box.shape
        assert obs[name].dtype == box.dtype

    env.env.state.step_count.fill_(env.env.cfg.max_episode_steps - 1)
    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.int64))
    for name, box in env.observation_space.spaces.items():
        assert obs[name].shape == box.shape
        assert obs[name].dtype == box.dtype
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert truncated and not terminated


# ---- no autoreset: obs is the terminal tick's own state, not a fresh episode ----------------

def test_no_autoreset_obs_reflects_the_terminal_tick_not_a_fresh_episode():
    env = _make()
    env.reset()
    env.env.state.ent_hp[0, 0] = 0.0
    env.env.state.ent_last_hit_by[0, 0] = -1

    obs, reward, terminated, truncated, info = env.step(np.array([0, 0]))
    assert terminated
    assert obs["self"][6] == 0.0  # self group column 6 == hero.hp_frac (dead -> 0)


def test_step_after_done_without_reset_raises():
    env = _make()
    env.reset()
    env.env.state.ent_hp[0, 0] = 0.0
    env.env.state.ent_last_hit_by[0, 0] = -1

    obs, reward, terminated, truncated, info = env.step(np.array([0, 0]))
    assert terminated
    try:
        env.step(np.array([0, 0]))
        assert False, "expected a RuntimeError"
    except RuntimeError as e:
        assert "reset()" in str(e)

    # reset() clears the flag and lets stepping resume normally.
    env.reset()
    env.step(np.array([0, 0]))  # must not raise


def test_env_autoreset_flag_is_forced_off():
    env = _make()
    assert env.env.autoreset is False


# ---- reset(seed=...) reseeds the underlying generator ---------------------------------------

def test_reset_seed_reproduces_the_same_trajectory():
    env_a = _make(seed=1)
    obs_a, _ = env_a.reset(seed=42)
    actions = [env_a.action_space.sample() for _ in range(10)]
    traj_a = [obs_a["self"].copy()]
    for a in actions:
        obs_a, *_ = env_a.step(a)
        traj_a.append(obs_a["self"].copy())

    env_b = _make(seed=99)  # different construction seed -- reset(seed=42) must still win
    obs_b, _ = env_b.reset(seed=42)
    traj_b = [obs_b["self"].copy()]
    for a in actions:
        obs_b, *_ = env_b.step(a)
        traj_b.append(obs_b["self"].copy())

    for a, b in zip(traj_a, traj_b):
        assert np.allclose(a, b)
