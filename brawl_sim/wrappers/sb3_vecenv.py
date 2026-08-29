"""BrawlSB3VecEnv: a `stable_baselines3.common.vec_env.VecEnv` adapter around `BrawlVecEnv`.
See BRAWL_SIM_BUILD_PLAN.md Step 33.

**One `step()` here is one AGENT DECISION, i.e. `cfg.action_repeat` sim ticks** (see `env.py`'s
own docstring). Everything this file counts or transfers is therefore per decision, not per tick:
`info["episode"]["l"]`, `self._step_count`/`info_every`, the host transfers below, and the
`action_masks()` python loop. That is the whole performance argument for action repeat -- the
per-`n_envs` Python work this module is forced to do at the SB3 boundary is paid once per
decision instead of once per tick, so raising `action_repeat` amortizes it down proportionally
without changing a single line here.

**This is one of exactly two places in the repo where device->host transfers (`.cpu()`/
`.numpy()`) and per-env Python loops over `n_envs` are permitted** -- the other is
`wrappers/gym_single.py` (Step 34), for the identical reason (gymnasium's own `Env` contract
also requires plain Python/numpy, not torch tensors). This docstring originally claimed "the
ONLY place"; that was true until Step 34 and is corrected here rather than left stale. Every
OTHER module (`core/`, `bots/`, `env.py`, `wrappers/episode_stats.py`, `core/obs_select.py`)
still stays torch-native and device-resident, with no python-level per-env loop -- that
discipline is what makes `n_envs=4096` fast. This file is the seam where the sim meets a
CPU-bound, python-object-per-env, MANY-envs-at-once library (SB3) and MUST cross both boundaries
to satisfy that library's contract; `gym_single.py` crosses the same boundaries for a single env
instead. Keep every new host transfer here deliberate and counted
(`self.last_step_host_transfers`); nothing upstream of either file should ever need one.

**Reward ownership (a resolved design tension, read this before changing either side).** The
plan's own text describes `step_wait` as the thing that "calls `reward_fn(full_obs, info,
cfg)`". Taken completely literally, that would mean EITHER (a) `reward_fn` gets called twice
per tick -- once inside `BrawlVecEnv.step()` via its own (default `ZeroReward`) `self.reward_fn`
at Section 4 phase 16, once again here -- wasted compute, or (b) it's called only here, in which
case `EpisodeStats` (which accumulates whatever `env.step()`'s OWN first return value is) would
silently track the WRONG reward (the wrapped env's `ZeroReward`, always 0) while SB3 sees a
DIFFERENT, correct one -- `infos[i]["episode"]["r"]` would then always read 0, breaking
`VecMonitor`/the SB3 logger, exactly the thing Constraint 4 asks this file to get right.
**The fix: `BrawlSB3VecEnv.__init__` takes ownership of `env.reward_fn`, overwriting it with the
`reward_fn` passed in here.** `BrawlVecEnv.step()`'s own phase-16 `_observe` call
(`core/reward.py`'s documented call site, already handling the pre-reset-obs subtlety a naive
second call site would have to duplicate) then IS the one and only place `reward_fn` runs, on
exactly the right (pre-autoreset) full observation, and its returned `reward` tensor is used
directly as SB3's per-step reward -- no second call, no redundant compute, and `EpisodeStats`'
accumulation is automatically correct. If you construct `env` yourself with a non-default
`reward_fn` and expect it to survive unwrapped construction here, it won't -- this is a
deliberate, documented takeover, not an oversight.

**`action_masks()` has TWO call paths, both required** -- found by actually running
`sb3_contrib.MaskablePPO.learn()` against this wrapper, not from the plan text alone (which only
specifies the direct method). (1) `sb3_contrib`'s own masking-support probe,
`VecEnv.has_attr("action_masks")`, goes through `get_attr`, which -- like `set_attr` --
deliberately checks THIS wrapper before the underlying `BrawlVecEnv`, since `action_masks` is
defined here, not there. (2) Rollout collection itself calls `get_action_masks(env)`, which for
any `VecEnv` calls `env.env_method("action_masks")` and `np.stack()`s the result -- NOT
`env.action_masks()` directly. `env_method` special-cases exactly this one method name, and
otherwise still raises for everything else (there genuinely is no per-sub-env method to call on
a fused batch). This makes `env_method("action_masks", indices=...)` an unavoidable per-`n_envs`
Python loop (`list(self.action_masks())`) on every rollout step -- the same category of cost the
plan's own `info_mode="full"` warning already accepts for diagnostics, just now also paid
whenever `sb3_contrib.MaskablePPO` (the plan's own recommended algorithm) collects a rollout.
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
        # A SEPARATE buffer set for terminal_observation (built from info["final_observation"],
        # a genuinely different tensor than the just-transferred regular `obs` -- reusing
        # `_device_buffers`/`_host_buffers` here would clobber the observation `step_wait` just
        # returned, since both are the same persistent, in-place-written storage.
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
        """Proxies straight through to the underlying BrawlVecEnv's `reward_fn` -- there is no
        separate copy to fall out of sync. `__init__` set `env.reward_fn = reward_fn` (the
        deliberate takeover, see module docstring) once, and this property means `get_attr`/
        `set_attr("reward_fn", ...)` (which check THIS wrapper before the underlying env, see
        `get_attr`) transparently read/write that SAME underlying value rather than a second,
        inert copy that `step_wait` would never actually look at."""
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
        dones_np = terminated_np | truncated_np  # Constraint 1: SB3 has one bool, not two

        infos = self._build_infos(info, terminated_np, truncated_np, full_obs)
        return obs_np, reward_np, dones_np, infos

    def close(self) -> None:
        pass  # BrawlVecEnv owns no external resources (files/sockets/subprocesses) to release

    def get_attr(self, attr_name, indices=None):
        """Checks THIS wrapper first, falling back to the underlying `BrawlVecEnv` -- not the
        other way around. This is required, not just a style choice: `sb3_contrib.MaskablePPO`
        detects action-masking support via `VecEnv.has_attr("action_masks")`, which calls
        `get_attr` and checks for `AttributeError` -- `action_masks` is a method on THIS class,
        not on `BrawlVecEnv`, so looking there first would (and, before this fix, did) make
        `is_masking_supported` return False and MaskablePPO refuse to train."""
        obj = self if hasattr(self, attr_name) else self.env.env
        value = getattr(obj, attr_name)
        return [value] * len(self._resolve_indices(indices))

    def set_attr(self, attr_name, value, indices=None) -> None:
        """Mirrors get_attr's "check this wrapper first" precedence."""
        obj = self if hasattr(self, attr_name) else self.env.env
        setattr(obj, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        """`"action_masks"` is a REAL, expected call here, not a hypothetical: `sb3_contrib`'s
        `get_action_masks(env)` calls `env.env_method("action_masks")` (not `env.action_masks()`
        directly) and does `np.stack(...)` over the result, exactly the "one result per
        sub-env" shape a real multi-process VecEnv would produce. `self.action_masks()` already
        returns the FUSED `(num_envs, 19)` array; `list(...)` over it yields exactly that
        per-index row list, so stacking it back reproduces the same array -- found by an actual
        MaskablePPO.learn() smoke test, not anticipated from the plan text alone. Every other
        method name still legitimately has no single sub-env to call it on."""
        if method_name == "action_masks":
            # Index INTO the mask by idx rather than enumerating every row and testing
            # `i in idx`: with the default indices=None (every real MaskablePPO call --
            # get_action_masks never passes a subset), the old `in`-a-list check was an
            # accidental O(num_envs^2) scan. Profiling a real training run showed this alone
            # costing ~8% of total wall-clock at n_envs=4096.
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
        returns `[seed] * num_envs` per `VecEnv`'s contract.

        This used to raise `NotImplementedError` on the grounds that there are no independent
        per-sub-env seeds to set. That reasoning is still true -- and is why the returned list is
        `num_envs` copies of one value rather than `num_envs` distinct ones -- but refusing
        outright turned out to break the plan's own recommended entry point: passing `seed=` to
        `PPO`/`MaskablePPO` makes SB3's `BaseAlgorithm.set_random_seed` call `env.seed(seed)`
        unconditionally, so a seeded model could not be constructed against this env at all.
        `BrawlVecEnv` owns exactly one `Generator` and `manual_seed` on it is a genuine reseed,
        so honoring the call is both possible and more useful than refusing it.

        What you still cannot do is seed sub-envs independently: every env in the batch is drawn
        from a single interleaved stream, so trajectories are reproducible only for a fixed
        `(seed, n_envs, device, actions)` -- the same caveat `BrawlVecEnv` already documents.
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
        """(num_envs, 19) bool = [move_mask (17), attack_mask (2)] -- what MaskablePPO expects.
        Reads the LAST obs this wrapper produced (reset() or step_wait()); action_masks() is
        always called by MaskablePPO right after an obs, before the matching action, so this is
        never stale in normal use."""
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

        # Constraint 3: terminal_observation must be the FINISHED episode's agent obs -- built
        # from info["final_observation"] (Step 29's pre-autoreset clone), never the just-reset
        # `obs` step_wait already transferred above. Only paid for on ticks with >= 1 done env.
        terminal_agent_obs = obs_select.build_agent_obs(
            info["final_observation"], self.agent_spec, self.cfg, self._terminal_device_buffers
        )
        terminal_np = self._copy_to_host(terminal_agent_obs, self._terminal_host_buffers)

        want_episode = self.info_mode in ("episode", "full")
        if want_episode:
            final_len = info["final_episode_length"].to("cpu").numpy()
            final_ret = info["final_episode_return"].to("cpu").numpy()
            # `hero_rank` is 0-INDEXED (0 = won the match) -- core/events.py subtracts 1 from
            # core/observation.compute_rank's 1-indexed placement. Read off the PRE-autoreset
            # `final_info`, never the live `info`: by the time step_wait returns, a done env's
            # own `hero_rank` already describes its freshly spawned replacement episode.
            # Surfaced under its own top-level key rather than folded into `episode`, because
            # VecMonitor overwrites `info["episode"]` wholesale and would drop it.
            final_rank = info["final_info"]["hero_rank"].to("cpu").numpy()
            # Per-episode hero totals for `training/callbacks.TrainingMonitorCallback` --
            # `obs["hero"]["kills"/"damage_dealt"/"damage_taken"/"cubes"/"shots_fired"]` are
            # CUMULATIVE counters that `core/state.zero_` resets only on spawn (core/observation.
            # py's own field comments), so reading them off `final_observation` (the pre-autoreset
            # clone) gives the finished episode's own final totals, not a delta. A single plain
            # `.to("cpu")` per field, matching `_maybe_attach_full_diagnostics`' own precedent for
            # infrequent (done-tick-only) transfers -- not the pinned non_blocking buffer path,
            # which is reserved for the ALWAYS-paid per-step transfers this method already issued
            # (and already synced) before `_build_infos` was ever called.
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
        """`info_mode="full"`: a small per-env diagnostic snapshot on every `info_every`-th
        tick, for EVERY env (not just done ones) -- deliberately NOT a fixed part of the
        schema (nothing downstream depends on its exact keys), just enough to eyeball training
        health without a separate logging path. This is the one place in this file with an
        unavoidable per-env python loop; the plan's own warning applies -- diagnostics only,
        never the training-time default."""
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
