"""Reward functions. The simulator itself returns zero reward: `ZeroReward` is `BrawlVecEnv`'s
default `reward_fn`. A real reward (training/reward.ShapedReward) is a `reward_fn` over `obs` +
`info`, which already carry everything one needs, so **no reward requires touching the
simulator**. `BrawlSB3VecEnv` (and wrappers/gym_single.py) installs it on the env, whose
`_observe` calls it once per decision. `RewardFn` is a structural `typing.Protocol`, so any
callable with the right signature works: a plain function, a lambda, a class.
"""
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class RewardFn(Protocol):
    def __call__(self, obs: dict, info: dict, cfg) -> torch.Tensor: ...  # (N,) f32


class ZeroReward:
    """Default `RewardFn`: zeros, `(N,) float32`, on `info`'s device. The tensor is allocated
    when `(n_envs, device)` changes and the SAME object is returned on every other call, so
    callers must treat it as read-only (an in-place write would corrupt every later reward)."""

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
    """NOT tuned, NOT recommended: a worked example of reading obs + info, used by
    scripts/sb3_smoke.py, scripts/benchmark.py and tests. Terminal-only: +1 when the hero is
    the last one alive, -1 when it is dead, 0 otherwise (a truncation that catches the hero
    alive but not alone is neither a win nor a death).

    `info["terminated"]` is exactly "hero dead OR hero last alive" (events.compute_done), so the
    hero's alive flag tells the two apart. It reads `obs["hero"]["alive"]`, the post-decision
    value, not the latched `info["hero_alive"]`: with action_repeat > 1, a win followed by a
    death later in the same decision scores -1."""

    def __call__(self, obs: dict, info: dict, cfg) -> torch.Tensor:
        terminated = info["terminated"]
        hero_alive = obs["hero"]["alive"]
        reward = torch.zeros_like(terminated, dtype=torch.float32)
        reward = torch.where(terminated & hero_alive, torch.ones_like(reward), reward)
        reward = torch.where(terminated & ~hero_alive, -torch.ones_like(reward), reward)
        return reward
