"""Hyperparameter schedules in the exact shape Stable-Baselines3 wants.

SB3 accepts either a float or a `Callable[[float], float]` for `learning_rate` and `clip_range`,
and calls it with **`progress_remaining`**, which runs 1.0 at the first step down to 0.0 at
`total_timesteps` -- i.e. it counts DOWN, not up. Getting that direction backwards silently
trains with a rising learning rate, which is the single easiest way to make a PPO run diverge
late; every function here is written in terms of `t = 1 - progress_remaining` (elapsed
fraction, counting up) to keep the intent visible at the point of use.

`progress_remaining` is clamped to [0, 1] on the way in: SB3 computes it as
`1 - num_timesteps / total_timesteps`, and `num_timesteps` can legitimately overshoot
`total_timesteps` by up to one rollout (it advances `n_envs` at a time), which would otherwise
push an exponential schedule below its own floor on the last update.
"""
import math

from .config import ScheduleConfig


def constant_schedule(value: float):
    def _schedule(progress_remaining: float) -> float:
        return value
    return _schedule


def linear_schedule(initial: float, final: float):
    """`initial` -> `final`, straight line in elapsed timesteps."""
    def _schedule(progress_remaining: float) -> float:
        t = _elapsed(progress_remaining)
        return initial + t * (final - initial)
    return _schedule


def cosine_schedule(initial: float, final: float):
    """`initial` -> `final` on a half cosine: flat at both ends, steepest in the middle. Decays
    more slowly than linear early on, which tends to suit PPO's noisy early updates."""
    def _schedule(progress_remaining: float) -> float:
        t = _elapsed(progress_remaining)
        return final + 0.5 * (initial - final) * (1.0 + math.cos(math.pi * t))
    return _schedule


def exponential_schedule(initial: float, final: float):
    """`initial` -> `final` at a constant decay RATIO per unit of progress. Requires
    `final > 0` (enforced in ScheduleConfig): a geometric decay never reaches zero."""
    ratio = final / initial

    def _schedule(progress_remaining: float) -> float:
        t = _elapsed(progress_remaining)
        return initial * (ratio ** t)
    return _schedule


def _elapsed(progress_remaining: float) -> float:
    return 1.0 - min(1.0, max(0.0, float(progress_remaining)))


_BUILDERS = {
    "constant": lambda c: constant_schedule(c.initial),
    "linear": lambda c: linear_schedule(c.initial, c.final),
    "cosine": lambda c: cosine_schedule(c.initial, c.final),
    "exponential": lambda c: exponential_schedule(c.initial, c.final),
}


def make_schedule(cfg: ScheduleConfig):
    """ScheduleConfig -> the callable SB3 expects. `constant` ignores `final` entirely."""
    return _BUILDERS[cfg.schedule](cfg)


def describe(cfg: ScheduleConfig) -> str:
    """One-line human summary for the run banner, e.g. `linear 3.0e-04 -> 1.0e-05`."""
    if cfg.schedule == "constant":
        return f"constant {cfg.initial:.3g}"
    return f"{cfg.schedule} {cfg.initial:.3g} -> {cfg.final:.3g}"
