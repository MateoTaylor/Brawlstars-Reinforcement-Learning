"""Hyperparameter schedules in the exact shape Stable-Baselines3 wants.

SB3 calls a `learning_rate` / `clip_range` callable with `progress_remaining`, which counts DOWN
from 1.0 at the first step to 0.0 at `total_timesteps`; getting that backwards silently trains
with a rising learning rate. Every function here works in `t = 1 - progress_remaining` (elapsed
fraction, counting up) to keep the intent visible.

`progress_remaining` is clamped to [0, 1]: `num_timesteps` can overshoot `total_timesteps` by up
to one rollout, which would otherwise push an exponential schedule below its floor.
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


def over_segment(schedule, start_progress: float):
    """`schedule` squeezed into the last `start_progress` of SB3's progress, for `scripts/train.py
    --resume --restart-schedules`. A resumed model counts progress over ALL its timesteps, so a
    fine-tune's first update arrives at `progress_remaining = total / (done + total)`, not 1.0;
    squeezed, the schedule still starts at its `initial` there and reaches `final` at the end. It
    is saved inside the model, so a plain `--resume` after a crash continues it."""
    def _schedule(progress_remaining: float) -> float:
        return schedule(progress_remaining / start_progress)
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
