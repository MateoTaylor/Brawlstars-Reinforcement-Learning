"""BrawlGymEnv: a standard single-environment `gymnasium.Env` wrapping `BrawlVecEnv(n_envs=1)`.

**Validation and debugging only.** gymnasium's `step`/`reset` contract needs plain Python
floats/bools and numpy arrays, so this file and `wrappers/sb3_vecenv.py` are the two places host
transfers are allowed; nothing else should need one.

**No autoreset.** A single-env `gymnasium.Env` must not reset inside `step()`, so `__init__` sets
`env.autoreset = False` on the `BrawlVecEnv` it is given (and takes over `env.reward_fn`, as
`BrawlSB3VecEnv` does). Calling `step()` after `terminated | truncated` without a `reset()`
raises `RuntimeError` rather than simulating a dead hero or a finished episode.
"""
import gymnasium as gym
import numpy as np
import torch

from ..core import obs_select
from ..env import BrawlVecEnv


class BrawlGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env: BrawlVecEnv, spec: obs_select.AgentObsSpec, reward_fn) -> None:
        if env.n_envs != 1:
            raise ValueError(
                f"BrawlGymEnv wraps exactly one env; got env.n_envs={env.n_envs}. "
                "Construct BrawlVecEnv(..., n_envs=1) for this wrapper."
            )
        env.reward_fn = reward_fn  # same reward-ownership takeover as BrawlSB3VecEnv
        env.autoreset = False      # see module docstring
        self.env = env
        self.cfg = env.cfg
        self.agent_spec = spec

        self.observation_space = obs_select.agent_space(spec, env.cfg)
        self.action_space = gym.spaces.MultiDiscrete(list(env.cfg.action_nvec))
        self.render_mode = None

        self._device_buffers = obs_select.make_agent_obs_buffers(spec, env.cfg, 1, env.device)
        self._done = True  # must call reset() before the first step(), like any fresh gym.Env

    def _agent_obs_np(self, full_obs: dict) -> dict:
        agent_obs = obs_select.build_agent_obs(full_obs, self.agent_spec, self.cfg, self._device_buffers)
        return {name: t[0].to("cpu").numpy() for name, t in agent_obs.items()}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.env.gen.manual_seed(seed)
        full_obs = self.env.reset()
        self._done = False
        return self._agent_obs_np(full_obs), {}

    def step(self, action):
        if self._done:
            raise RuntimeError(
                "step() called after terminated/truncated was True without an intervening "
                "reset() -- gymnasium single envs don't autoreset. Call reset() first."
            )
        action_t = torch.as_tensor(np.asarray(action), dtype=torch.int64, device=self.env.device).view(1, 2)
        full_obs, reward_t, terminated_t, truncated_t, info = self.env.step(action_t)

        terminated = bool(terminated_t.item())
        truncated = bool(truncated_t.item())
        self._done = terminated or truncated

        obs_np = self._agent_obs_np(full_obs)
        reward = float(reward_t.item())
        return obs_np, reward, terminated, truncated, {}

    def render(self):
        raise NotImplementedError(
            "Rendering isn't implemented for BrawlGymEnv; use BrawlVecEnv.snapshot(0) for "
            "per-tick CPU/numpy state instead."
        )

    def close(self) -> None:
        pass  # BrawlVecEnv owns no external resources (files/sockets/subprocesses) to release
