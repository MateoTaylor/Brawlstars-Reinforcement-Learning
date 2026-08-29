"""BrawlGymEnv: a standard single-environment `gymnasium.Env` wrapping `BrawlVecEnv(n_envs=1)`.
See BRAWL_SIM_BUILD_PLAN.md Step 34.

**Validation and debugging only** -- one env, host transfers, `.item()`/`.cpu()`/`.numpy()`
calls throughout. `wrappers/sb3_vecenv.py` (Step 33) called itself "the ONLY place in the repo
where host transfers... are permitted"; that claim was true when it was written but is
superseded here -- this file has the exact same obligation for the same reason (gymnasium's
`Env.step`/`reset` contract requires plain Python floats/bools and numpy arrays, not torch
tensors) and is the second, equally legitimate place it happens. Neither file's existence makes
the other's discipline optional; nothing outside these two files should ever need a host
transfer.

**No autoreset.** Standard single-env `gymnasium.Env`s are contractually forbidden from
resetting inside `step()` -- the caller must see `terminated`/`truncated` and call `reset()`
itself. `BrawlGymEnv.__init__` enforces this by setting `env.autoreset = False` on the
`BrawlVecEnv` it's given (Step 34's own addition to `env.py`; see that module's docstring) --
the SAME "wrapper takes ownership of an attribute on construction" pattern
`BrawlSB3VecEnv.__init__` already established for `env.reward_fn` (Step 33), not a new idiom.
Once `terminated | truncated` is `True`, calling `step()` again without an intervening
`reset()` raises `RuntimeError` rather than silently continuing to simulate a hero that's
already dead (or an episode that's already timed out) -- a debugging tool should fail loudly on
caller misuse, not produce quietly-meaningless data.
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
        env.reward_fn = reward_fn  # same reward-ownership takeover as BrawlSB3VecEnv, Step 33
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
                "reset() -- gymnasium single envs don't autoreset (Step 34). Call reset() first."
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
