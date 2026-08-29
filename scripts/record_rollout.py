"""Records a BrawlVecEnv(n_envs=1) rollout to a stacked-array .npz, for offline replay via
render/ascii.py (Step 37) or render/viewer.py (Step 38). See BRAWL_SIM_BUILD_PLAN.md Step 37;
extended in Step 38 with two more optional extra fields the viewer needs, and in Step 39 with
`extra_fields`'s rename to public (see its own docstring).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from brawl_sim.bots import perception
from brawl_sim.config import load_config
from brawl_sim.core import stats
from brawl_sim.core.state import snapshot as state_snapshot
from brawl_sim.env import BrawlVecEnv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def extra_fields(env: BrawlVecEnv) -> dict:
    """Optional extra keys, none part of raw SimState (Step 9) -- every consumer (`render_ascii`
    Step 37, `render.viewer` Step 38, `play_manual` Step 39) treats them as OPTIONAL, so adding
    more here over time is safe and doesn't require touching any renderer's required-field
    contract. Public (no leading underscore, renamed in Step 39) since `play_manual.py` now
    imports this directly rather than keeping a third hand-written copy -- same treatment
    `render/ascii.py`'s `status_line` got in Step 38 for the same reason.

    `revealed_to_hero`/`max_ammo`: see render/ascii.py's module docstring (Step 37).

    `los_to_hero` (added Step 38, for the viewer's reveal overlay): `bots.perception.raw_los`'s
    hero row -- physical wall-only LOS, independent of bush-hiding (Notice 4). The viewer's 'v'
    overlay specifically highlights entities that are `los_to_hero` (no wall in the way) but NOT
    `revealed_to_hero` (bush-hidden) -- i.e. bush-hiding is the only reason they're not visible,
    as opposed to a wall being in the way, which `render_ascii` doesn't distinguish (it only
    ever draws what IS visible, never annotates *why* something isn't).

    `unit_radius` (added Step 38, for sizing the viewer's entity circles): a `SimParams`
    PER_ENV field (config.py PER_ENV_FIELDS) -- one scalar per env, shared by every entity in
    that env, not per-kind. Recomputed every frame like everything else here, since an autoreset
    mid-recording resamples it."""
    vis = perception.visibility(env.state, env.bank, env.params, env.cfg)
    los = perception.raw_los(env.state, env.bank, env.cfg)
    revealed_to_hero = vis[0, 0].detach().cpu().numpy()
    los_to_hero = los[0, 0].detach().cpu().numpy()
    max_ammo = stats.gather_kind(env.params.max_ammo, env.state.ent_kind)[0].detach().cpu().numpy()
    unit_radius = env.params.unit_radius[0].detach().cpu().numpy()
    return {
        "revealed_to_hero": revealed_to_hero,
        "max_ammo": max_ammo,
        "los_to_hero": los_to_hero,
        "unit_radius": unit_radius,
    }


def record_rollout(cfg, steps: int, seed: int = 0, device: str = "cpu", action_seed: int = 0) -> dict:
    """`n_envs=1`, `steps` frames: frame 0 is the post-`reset()` state; frame `i` (`i>0`) is the
    state after the `i`-th SIM TICK, driven by random actions (this script is a visual-debugging
    aid, not a policy evaluation tool -- Step 39's `play_manual.py` is where a human/policy
    drives the hero). Returns `{field_name: (steps, ...) numpy array}`, one array per
    `core.state.snapshot()` field plus the extras from `extra_fields`.

    **`steps` is sim ticks, not `step()` calls.** One `env.step()` is one agent decision covering
    `cfg.action_repeat` ticks, so frames are captured through `BrawlVecEnv.tick_hook` (which
    fires once per sub-tick) rather than around the step loop. That keeps a recording at the
    sim's own 20 Hz regardless of the decision rate -- the same thing `scripts/watch.py` does,
    and what the replay viewer expects: sampling at the decision boundary instead would drop
    four of every five ticks and make projectiles jump between frames."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    env = BrawlVecEnv(cfg, n_envs=1, device=device, seed=seed)
    env.reset()
    gen = torch.Generator(device=device).manual_seed(action_seed)

    def _frame(e) -> dict:
        frame = state_snapshot(e.state, 0)
        frame.update(extra_fields(e))
        return frame

    frames = [_frame(env)]
    # Keeps capturing past `steps` within the final decision (the hook can't stop mid-decision);
    # the slice below trims that tail, so the caller always gets exactly `steps` frames.
    env.tick_hook = lambda e: frames.append(_frame(e))
    try:
        while len(frames) < steps:
            move = torch.randint(0, env.cfg.n_move_bins + 1, (1,), generator=gen, device=device)
            fire = torch.randint(0, 2, (1,), generator=gen, device=device)
            env.step(torch.stack([move, fire], dim=1))
    finally:
        env.tick_hook = None

    frames = frames[:steps]
    return {name: np.stack([f[name] for f in frames], axis=0) for name in frames[0]}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--preset", default=None, help="optional configs/presets/*.yaml overrides file")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out", default="rollout.npz")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    overrides = None
    if args.preset:
        import yaml
        overrides = yaml.safe_load(Path(args.preset).read_text())
    cfg = load_config(args.config, overrides=overrides)

    frames = record_rollout(cfg, steps=args.steps, seed=args.seed, device=args.device)
    np.savez(args.out, **frames)
    print(f"wrote {args.out} ({args.steps} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
