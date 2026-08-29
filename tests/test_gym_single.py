import numpy as np
import yaml
from stable_baselines3.common.env_checker import check_env

from brawl_sim.config import load_config
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
        merged = {**merged, **overrides}
    cfg = load_config(CONFIGS_DEFAULT, overrides=merged)
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=seed)
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    return BrawlGymEnv(env, spec, reward_fn or ExampleReward())


# ---- check_env passes with no errors -------------------------------------------------------

def test_check_env_passes():
    env = _make()
    check_env(env, warn=True)  # raises on any failure


# ---- 300-step random episode terminates correctly, obs dtypes match the space --------------

def test_random_episode_terminates_and_obs_dtypes_match_space():
    env = _make()
    obs, info = env.reset()
    assert info == {}
    for name, box in env.observation_space.spaces.items():
        assert obs[name].shape == box.shape
        assert obs[name].dtype == box.dtype

    rng = np.random.default_rng(0)
    terminated = truncated = False
    steps = 0
    for _ in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        for name, box in env.observation_space.spaces.items():
            assert obs[name].shape == box.shape
            assert obs[name].dtype == box.dtype
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        if terminated or truncated:
            break

    assert terminated or truncated
    assert steps <= 300


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
    rng = np.random.default_rng(0)
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
