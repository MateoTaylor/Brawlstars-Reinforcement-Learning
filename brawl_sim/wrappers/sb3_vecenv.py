"""BrawlSB3VecEnv: a `stable_baselines3.common.vec_env.VecEnv` adapter around `BrawlVecEnv`.

**One `step()` here is one AGENT DECISION, i.e. `cfg.action_repeat` sim ticks** (see `env.py`).
Everything this file counts or transfers is per decision: `info["episode"]["l"]`,
`self._step_count`/`info_every`, the host transfers and the `action_masks()` loop. That is the
performance argument for action repeat: the per-env Python work SB3 forces at this boundary is
paid once per decision, not once per tick.

**Host transfers and per-env Python loops are allowed here and in `wrappers/gym_single.py`, and
nowhere else.** This is the seam where the device-resident sim meets SB3's
python-object-per-env contract; keep every new host transfer deliberate and counted
(`self.last_step_host_transfers`).

**Reward ownership.** `__init__` takes over `env.reward_fn`, overwriting it with the `reward_fn`
passed in here. `BrawlVecEnv.step()`'s phase-16 `_observe` call is then the one place it runs,
on the right (pre-autoreset) full observation, and `EpisodeStats` accumulates the same reward
SB3 sees. Calling it here as well would double the work; calling it only here would leave
`EpisodeStats`, and so `info["episode"]["r"]`, summing the wrapped env's default `ZeroReward`.
An `env` built with its own `reward_fn` does not keep it.

**`action_masks()` has TWO call paths, both required by `sb3_contrib.MaskablePPO`:** its masking
probe `has_attr("action_masks")` goes through `get_attr`, which checks THIS wrapper before the
underlying env; and rollout collection calls `get_action_masks(env)`, i.e.
`env.env_method("action_masks")` + `np.stack`, which `env_method` special-cases. The latter is a
per-env Python loop on every rollout step.
"""
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv

from ..core import obs_select
from .episode_stats import EpisodeStats

_INFO_MODES = ("minimal", "episode", "full")


class BrawlSB3VecEnv(VecEnv):
    def __init__(
        self, env, spec: obs_select.AgentObsSpec, reward_fn,
        info_mode: str = "minimal", info_every: int = 1,
    ) -> None:
        if info_mode not in _INFO_MODES:
            raise ValueError(f"info_mode must be one of {_INFO_MODES}, got {info_mode!r}")
        if info_every < 1:
            raise ValueError(f"info_every must be >= 1, got {info_every}")

        env.reward_fn = reward_fn  # see module docstring -- deliberate takeover
        self.env = EpisodeStats(env)
        self.cfg = env.cfg
        self.agent_spec = spec
        self.info_mode = info_mode
        self.info_every = info_every
        self.device = env.device
        self.num_envs = env.n_envs

        observation_space = obs_select.agent_space(spec, env.cfg)
        action_space = gym.spaces.MultiDiscrete(list(env.cfg.action_nvec))
        super().__init__(self.num_envs, observation_space, action_space)

        self._pinned = self.device.type == "cuda"
        self._device_buffers = obs_select.make_agent_obs_buffers(spec, env.cfg, self.num_envs, self.device)
        self._host_buffers = self._make_host_buffers(self._device_buffers)
        # A SEPARATE buffer set for terminal_observation (built from info["final_observation"]):
        # reusing `_device_buffers`/`_host_buffers` would clobber the observation `step_wait`
        # just returned, since both are persistent, in-place-written storage.
        self._terminal_device_buffers = obs_select.make_agent_obs_buffers(spec, env.cfg, self.num_envs, self.device)
        self._terminal_host_buffers = self._make_host_buffers(self._terminal_device_buffers)

        self._reward_host = torch.empty(self.num_envs, dtype=torch.float32, pin_memory=self._pinned)
        self._terminated_host = torch.empty(self.num_envs, dtype=torch.bool, pin_memory=self._pinned)
        self._truncated_host = torch.empty(self.num_envs, dtype=torch.bool, pin_memory=self._pinned)

        self._pending_actions = None
        self._step_count = 0
        self._last_full_obs = None  # for action_masks(), set by reset()/step_wait()
        self.last_step_host_transfers = 0

    @property
    def reward_fn(self):
        """Proxies to the underlying BrawlVecEnv's `reward_fn` (the module docstring's takeover),
        so `get_attr`/`set_attr("reward_fn", ...)`, which check THIS wrapper first, read and write
        the value `step()` actually uses rather than an inert copy."""
        return self.env.env.reward_fn

    @reward_fn.setter
    def reward_fn(self, value) -> None:
        self.env.env.reward_fn = value

    def _make_host_buffers(self, device_buffers: dict) -> dict:
        return {
            name: torch.empty(t.shape, dtype=t.dtype, pin_memory=self._pinned)
            for name, t in device_buffers.items()
        }

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _copy_to_host(self, agent_obs: dict, host_buffers: dict) -> dict:
        """Issues one non_blocking `.copy_` per group (queued on the current CUDA stream), then
        the CALLER is responsible for one `_sync()` before reading `.numpy()` off the result --
        batching every queued copy behind a single sync (rather than one sync per group) is the
        whole point of `non_blocking=True` here."""
        for name, t in agent_obs.items():
            host_buffers[name].copy_(t, non_blocking=True)
            self.last_step_host_transfers += 1
        return {name: host_buffers[name].numpy() for name in agent_obs}

    # ---- VecEnv surface ---------------------------------------------------------------

    def reset(self):
        full_obs = self.env.reset()
        self._last_full_obs = full_obs
        self.last_step_host_transfers = 0
        agent_obs = obs_select.build_agent_obs(full_obs, self.agent_spec, self.cfg, self._device_buffers)
        obs_np = self._copy_to_host(agent_obs, self._host_buffers)
        self._sync()
        return obs_np

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)

    def step_wait(self):
        full_obs, reward_t, terminated, truncated, info = self.env.step(self._pending_actions)
        self._step_count += 1
        self._last_full_obs = full_obs
        self.last_step_host_transfers = 0

        agent_obs = obs_select.build_agent_obs(full_obs, self.agent_spec, self.cfg, self._device_buffers)
        obs_np = self._copy_to_host(agent_obs, self._host_buffers)

        self._reward_host.copy_(reward_t, non_blocking=True)
        self._terminated_host.copy_(terminated, non_blocking=True)
        self._truncated_host.copy_(truncated, non_blocking=True)
        self.last_step_host_transfers += 3
        self._sync()

        reward_np = self._reward_host.numpy()
        terminated_np = self._terminated_host.numpy()
        truncated_np = self._truncated_host.numpy()
        dones_np = terminated_np | truncated_np  # SB3 has one done bool, not two

        infos = self._build_infos(info, terminated_np, truncated_np, full_obs)
        return obs_np, reward_np, dones_np, infos

    def close(self) -> None:
        pass  # BrawlVecEnv owns no external resources (files/sockets/subprocesses) to release

    def get_attr(self, attr_name, indices=None):
        """Checks THIS wrapper first, then the underlying `BrawlVecEnv`. Required:
        `sb3_contrib.MaskablePPO` detects masking via `has_attr("action_masks")`, which calls
        `get_attr`, and `action_masks` exists only on this class; looking at the env first would
        make MaskablePPO refuse to train."""
        obj = self if hasattr(self, attr_name) else self.env.env
        value = getattr(obj, attr_name)
        return [value] * len(self._resolve_indices(indices))

    def set_attr(self, attr_name, value, indices=None) -> None:
        """Mirrors get_attr's "check this wrapper first" precedence."""
        obj = self if hasattr(self, attr_name) else self.env.env
        setattr(obj, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        """Only `"action_masks"` is supported: `sb3_contrib`'s `get_action_masks(env)` calls
        `env.env_method("action_masks")` and `np.stack`s the per-sub-env results, so this returns
        the rows of the fused `(num_envs, sum(cfg.action_nvec))` mask. Every other method has no
        single sub-env to run on."""
        if method_name == "action_masks":
            # Index INTO the mask rather than testing `i in idx` per row: with indices=None
            # (every MaskablePPO call) that test is an O(num_envs^2) scan.
            mask = self.action_masks()
            return [mask[i] for i in self._resolve_indices(indices)]
        raise NotImplementedError(
            f"BrawlSB3VecEnv fuses every env into one BrawlVecEnv on-device batch, not n "
            f"independent sub-envs -- there is no single sub-env to call {method_name!r} on. "
            "Use get_attr/set_attr for shared config values instead."
        )

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._resolve_indices(indices))

    def render(self, mode: str | None = None):
        raise NotImplementedError(
            "Rendering isn't implemented for the fused BrawlSB3VecEnv; use "
            "BrawlVecEnv.snapshot(env_index) for per-env CPU/numpy state instead."
        )

    def seed(self, seed: int | None = None):
        """Reseeds the ONE shared `torch.Generator` the whole fused batch draws from, and
        returns `[seed] * num_envs` per `VecEnv`'s contract. SB3 calls `env.seed(seed)` whenever
        `PPO`/`MaskablePPO` gets `seed=`, so refusing would make a seeded model unbuildable.

        Sub-envs cannot be seeded independently: every env draws from one interleaved stream, so
        trajectories are reproducible only for a fixed `(seed, n_envs, device, actions)`.
        """
        if seed is not None:
            self.env.env.gen.manual_seed(int(seed))
        return [seed] * self.num_envs

    def _resolve_indices(self, indices) -> list:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    # ---- action masking (sb3_contrib.MaskablePPO) --------------------------------------

    def action_masks(self) -> np.ndarray:
        """(num_envs, sum(cfg.action_nvec)) bool = [move_mask (17), attack_mask (4: none, attack,
        super, gadget; 5 with the auto-aimed attack under `action.auto_aim`)] -- what MaskablePPO
        expects, sliced per dimension. The widths come from the obs, never from a literal here.
        Reads the LAST obs this wrapper produced (reset() or step_wait()); MaskablePPO calls this
        right after an obs, before the matching action, so it is never stale in normal use."""
        am = self._last_full_obs["action_mask"]
        mask = torch.cat([am["move"], am["attack"]], dim=-1)
        return mask.to("cpu").numpy()

    # ---- infos --------------------------------------------------------------------------

    def _build_infos(self, info: dict, terminated_np, truncated_np, full_obs: dict) -> list:
        infos = [{} for _ in range(self.num_envs)]
        done_idx = np.nonzero(terminated_np | truncated_np)[0]
        if len(done_idx) == 0:
            self._maybe_attach_full_diagnostics(infos, full_obs)
            return infos

        # terminal_observation must be the FINISHED episode's agent obs, built from
        # info["final_observation"] (the pre-autoreset clone), never the just-reset `obs`
        # step_wait already transferred. Only paid on steps with >= 1 done env.
        terminal_agent_obs = obs_select.build_agent_obs(
            info["final_observation"], self.agent_spec, self.cfg, self._terminal_device_buffers
        )
        terminal_np = self._copy_to_host(terminal_agent_obs, self._terminal_host_buffers)

        want_episode = self.info_mode in ("episode", "full")
        if want_episode:
            final_len = info["final_episode_length"].to("cpu").numpy()
            final_ret = info["final_episode_return"].to("cpu").numpy()
            # `hero_rank` is 0-INDEXED (0 = won the match; core/events.py subtracts 1 from
            # core/observation.compute_rank). Read off the PRE-autoreset `final_info`: a done
            # env's live `hero_rank` already describes its replacement episode. It gets its own
            # top-level key because VecMonitor overwrites `info["episode"]` wholesale.
            final_rank = info["final_info"]["hero_rank"].to("cpu").numpy()
            # Per-episode hero totals for `training/callbacks.TrainingMonitorCallback`. These obs
            # fields are CUMULATIVE counters that `core/state.zero_` resets only on spawn, so the
            # pre-autoreset `final_observation` holds the finished episode's totals. A plain
            # `.to("cpu")` per field: the pinned non_blocking buffers are for the every-step
            # transfers, already synced before `_build_infos` runs.
            final_hero = info["final_observation"]["hero"]
            final_kills = final_hero["kills"].to("cpu").numpy()
            final_dmg_dealt = final_hero["damage_dealt"].to("cpu").numpy()
            final_dmg_taken = final_hero["damage_taken"].to("cpu").numpy()
            final_cubes = final_hero["cubes"].to("cpu").numpy()
            final_shots = final_hero["shots_fired"].to("cpu").numpy()

        for i in done_idx.tolist():
            infos[i]["terminal_observation"] = {name: arr[i] for name, arr in terminal_np.items()}
            infos[i]["TimeLimit.truncated"] = bool(truncated_np[i] and not terminated_np[i])
            if want_episode:
                infos[i]["episode"] = {"r": float(final_ret[i]), "l": int(final_len[i])}
                infos[i]["outcome"] = {"rank": int(final_rank[i]), "won": bool(final_rank[i] == 0)}
                infos[i]["episode_stats"] = {
                    "kills": int(final_kills[i]), "damage_dealt": float(final_dmg_dealt[i]),
                    "damage_taken": float(final_dmg_taken[i]), "cubes": int(final_cubes[i]),
                    "shots_fired": int(final_shots[i]),
                }

        self._maybe_attach_full_diagnostics(infos, full_obs)
        return infos

    def _maybe_attach_full_diagnostics(self, infos: list, full_obs: dict) -> None:
        """`info_mode="full"`: a small per-env diagnostic snapshot every `info_every`-th step, for
        EVERY env -- not a fixed schema (nothing downstream depends on its keys), just enough to
        eyeball training health. Its per-env Python loop makes it diagnostics-only, never the
        training default."""
        if self.info_mode != "full" or self._step_count % self.info_every != 0:
            return
        hero, meta = full_obs["hero"], full_obs["meta"]
        hp_frac = hero["hp_frac"].to("cpu").numpy()
        pos = hero["pos"].to("cpu").numpy()
        alive = hero["alive"].to("cpu").numpy()
        step_count = meta["step_count"].to("cpu").numpy()
        n_alive = meta["n_alive"].to("cpu").numpy()
        for i in range(self.num_envs):
            infos[i]["full"] = {
                "hero_hp_frac": float(hp_frac[i]), "hero_pos": pos[i].tolist(),
                "hero_alive": bool(alive[i]), "step_count": int(step_count[i]),
                "n_alive": int(n_alive[i]),
            }
