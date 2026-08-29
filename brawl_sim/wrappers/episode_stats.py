"""EpisodeStats: per-env episode length/return tracking around BrawlVecEnv. See
BRAWL_SIM_BUILD_PLAN.md Step 30.

Stays entirely on-device, torch-native, sync-free -- same discipline as `env.py` itself. This
is deliberate: per the architecture note at the top of the plan, host transfers are legal in
exactly one place, `wrappers/sb3_vecenv.py` (Step 33), and nowhere else -- including here, even
though this module lives under `wrappers/`. `BrawlSB3VecEnv` wraps `EpisodeStats` (not the bare
`BrawlVecEnv`) and is the thing that eventually turns `final_episode_length`/
`final_episode_return` into SB3's host-side `info["episode"] = {"r": ..., "l": ...}` dicts.

Mirrors the exact `final_observation`/`final_info` pattern `env.py` already established for
autoreset (Step 29): `info["final_episode_length"]` / `info["final_episode_return"]` are DENSE
`(N,)` tensors, valid for every env every tick, but only *meaningful* where this tick's
`terminated | truncated` was True -- for a still-running env they're just that env's
in-progress totals, not a sentinel. Same reasoning as `final_observation`: `BrawlVecEnv.step()`
already autoresets internally, so by the time `step()` returns, `state.step_count` etc. for a
just-finished env already reflect the NEW episode -- these two fields are this wrapper's own
parallel counters, captured immediately after `env.step()` returns and BEFORE they get zeroed
for done envs, for exactly the same reason `env.py` clones its own final_observation/final_info
before autoreset would otherwise stomp them.
"""
import torch


class EpisodeStats:
    """Wraps a `BrawlVecEnv`, forwarding `reset`/`step` unchanged except for adding
    `final_episode_length` (i64) and `final_episode_return` (f32) to `info`. Does not alter
    `obs`, `reward`, `terminated`, or `truncated` in any way, and does not touch `self.env`'s
    own state -- safe to construct around an already-stepped env.

    **`final_episode_length` counts DECISIONS, not sim ticks** -- it increments once per `step()`
    call, and one of those covers `cfg.action_repeat` ticks (env.py `_run_decision`). That is
    SB3's own convention for `info["episode"]["l"]`, which is what this eventually becomes, and it
    is why `rollout/ep_len_mean` reads ~600 rather than ~3000 at the shipped `action_repeat: 5`.
    Multiply by `cfg.action_repeat` for sim ticks, or by `cfg.agent_dt` for seconds of game time.
    `final_episode_return` needs no such caveat: `reward` is already summed over the decision's
    sub-ticks by the reward function, so the accumulated return is in absolute units and is
    invariant to the decision rate."""

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
