"""Reward functions. See BRAWL_SIM_BUILD_PLAN.md Step 28.

D09/N05: the simulator itself always returns zero reward (`ZeroReward`, wired as `env.py`'s
default `reward_fn`, Step 29). Anything real gets computed at the SB3 adapter boundary (Step
33) from `obs` (Step 25) + `info` (Step 27) -- both already carry everything any reward
function could need (positions, HP, kills, damage attribution, rank, termination cause, ...),
so **no reward requires touching the simulator itself**. `RewardFn` is a `typing.Protocol`
(structural, not an ABC) specifically so a reward function can be a plain function, a lambda, or
any callable with the right signature -- no inheritance, no boilerplate, and `BrawlVecEnv`/
`BrawlSB3VecEnv` (Steps 29/33) can accept and call it with zero knowledge of its concrete type.
"""
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class RewardFn(Protocol):
    def __call__(self, obs: dict, info: dict, cfg) -> torch.Tensor: ...  # (N,) f32


class ZeroReward:
    """Default `RewardFn` (D09/N05). Always returns zeros, `(N,) float32`, on whatever device
    `info` is on. Preallocates that zeros tensor once per distinct `(n_envs, device)` pair seen
    and returns the SAME tensor object on every subsequent call for that pair -- "allocates
    nothing per step" after the first call for a given shape/device, matching every other
    hot-path buffer in this codebase. The returned tensor is shared across calls: callers must
    treat it as read-only (mutating it in place would corrupt every future step's "reward")."""

    def __init__(self) -> None:
        self._cache_key: tuple | None = None
        self._zeros: torch.Tensor | None = None

    def __call__(self, obs: dict, info: dict, cfg) -> torch.Tensor:
        n_envs = info["time"].shape[0]
        device = info["time"].device
        key = (n_envs, device)
        if key != self._cache_key:
            self._zeros = torch.zeros(n_envs, dtype=torch.float32, device=device)
            self._cache_key = key
        return self._zeros


class ExampleReward:
    """NOT tuned, NOT recommended -- a syntactically valid worked example showing how to read
    obs + info, nothing more. Terminal-only: +1 for the hero being the last one alive, -1 for
    the hero being dead, 0 on every non-terminal tick (including a truncation that catches the
    hero alive but not alone -- a timeout is neither a win nor a death). Delete or replace; it
    exists so the SB3 smoke test (Step 36) has something to run.

    `info["terminated"]` is already exactly "hero dead OR hero last alive" (core/events.py,
    Step 27) and those two conditions are mutually exclusive and exhaustive within
    `terminated`, so `hero_alive` alone is enough to tell them apart -- no need to separately
    check `n_alive`."""

    def __call__(self, obs: dict, info: dict, cfg) -> torch.Tensor:
        terminated = info["terminated"]
        hero_alive = obs["hero"]["alive"]
        reward = torch.zeros_like(terminated, dtype=torch.float32)
        reward = torch.where(terminated & hero_alive, torch.ones_like(reward), reward)
        reward = torch.where(terminated & ~hero_alive, -torch.ones_like(reward), reward)
        return reward
