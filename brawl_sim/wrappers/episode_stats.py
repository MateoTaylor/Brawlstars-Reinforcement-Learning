"""EpisodeStats: per-env episode length/return tracking around BrawlVecEnv.

On-device and sync-free like `env.py`: host transfers belong only in `wrappers/sb3_vecenv.py` and
`wrappers/gym_single.py`. `BrawlSB3VecEnv` wraps `EpisodeStats` and turns its two fields into
SB3's host-side `info["episode"] = {"r": ..., "l": ...}` dicts.

Mirrors `env.py`'s `final_observation`/`final_info` pattern: `info["final_episode_length"]` and
`info["final_episode_return"]` are DENSE `(N,)` tensors, meaningful only where this step's
`terminated | truncated` is True (elsewhere they are the in-progress totals, not a sentinel).
`BrawlVecEnv.step()` has already autoreset a finished env when it returns, so these are this
wrapper's own counters, captured before it zeroes them for done envs.
"""
import torch


class EpisodeStats:
    """Wraps a `BrawlVecEnv`, forwarding `reset`/`step` unchanged except for adding
    `final_episode_length` (i64) and `final_episode_return` (f32) to `info`. Does not alter
    `obs`, `reward`, `terminated` or `truncated`, and does not touch `self.env`'s own state, so
    it is safe to construct around an already-stepped env.

    **`final_episode_length` counts DECISIONS, not sim ticks**: one per `step()`, each covering
    `cfg.action_repeat` ticks (env.py `_run_decision`). That is SB3's convention for
    `info["episode"]["l"]`, so `rollout/ep_len_mean` is in decisions; multiply by
    `cfg.action_repeat` for ticks or by `cfg.agent_dt` for game seconds. `final_episode_return`
    needs no such caveat: the reward is already summed over the decision's sub-ticks, so the
    return is invariant to the decision rate."""

    def __init__(self, env) -> None:
        self.env = env
        n_envs, device = env.n_envs, env.device
        self._length = torch.zeros(n_envs, dtype=torch.int64, device=device)
        self._return = torch.zeros(n_envs, dtype=torch.float32, device=device)

    def reset(self, reset_mask: torch.Tensor | None = None) -> dict:
        obs = self.env.reset(reset_mask)
        if reset_mask is None:
            reset_mask = torch.ones(self.env.n_envs, dtype=torch.bool, device=self.env.device)
        self._length.copy_(torch.where(reset_mask, torch.zeros_like(self._length), self._length))
        self._return.copy_(torch.where(reset_mask, torch.zeros_like(self._return), self._return))
        return obs

    def step(self, action: torch.Tensor, override: torch.Tensor | None = None):
        obs, reward, terminated, truncated, info = self.env.step(action, override)
        done = terminated | truncated

        length = self._length + 1
        ret = self._return + reward
        info["final_episode_length"] = length
        info["final_episode_return"] = ret

        self._length = torch.where(done, torch.zeros_like(length), length)
        self._return = torch.where(done, torch.zeros_like(ret), ret)
        return obs, reward, terminated, truncated, info
