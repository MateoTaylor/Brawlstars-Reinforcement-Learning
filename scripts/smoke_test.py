"""For every world-config variant, builds `BrawlVecEnv(n_envs=8)`, resets, and runs 500 random
steps with `debug_checks=True`, asserting a battery of invariants. See BRAWL_SIM_BUILD_PLAN.md
Step 41.

**"every config in configs/ and configs/presets/" is read as "every world-config variant," not
literally every `.yaml` file in those two directories.** `configs/brawlers.yaml`,
`configs/agent_obs.yaml`, and `configs/randomization.yaml` aren't `EnvConfig` sources at all --
they're auxiliary specs (brawler stats, agent-observation field selection, randomization
ranges) that get merged into or layered alongside a real world config, never loaded as one
themselves. `load_config()` would not even error if pointed at one of them directly: any
dotted key it doesn't recognize as an `EnvConfig` field is silently skipped (`_dget`'s own
`default=_MISSING` short-circuit), so "testing" them that way would exercise nothing new while
looking like extra coverage. Concretely this script builds: `configs/default.yaml` alone, plus
`configs/default.yaml` + each `configs/presets/*.yaml` overlay -- exactly how every preset
file's own docstring already says it's meant to be used (`load_config(default_path,
overrides=yaml.safe_load(preset_path))`).

**`core.state.check_invariants` already runs automatically inside every `BrawlVecEnv.step()`/
`reset()` call once `cfg.debug_checks=True`** (`env.py`'s own `if self.cfg.debug_checks:
check_invariants(...)`), and already covers: NaN/Inf on every float field, position bounds,
`0 <= hp <= max_hp`, no dead-with-nonzero-hp, `cubes <= max_cubes`, `dash_t <= dash_duration`,
live projectile count `<= max_projectiles`, `act_head` bounds, `env_idx` identity. Setting
`debug_checks=True` on every config this script builds IS how the plan's own "no NaN/Inf;
positions in bounds; ... 0 <= hp <= max_hp; live projectiles <= max_projectiles; cubes <=
max_cubes" acceptance items are satisfied -- this script does not re-implement any of them.

**What this script adds on top** (things `check_invariants` structurally can't check, since
they need either cross-tick history or calls it has no access to): alive-never-resurrects-
mid-episode (tracked via `env.step()`'s own public `terminated`/`truncated` return values, a
black-box behavioral check of the ENV'S CONTRACT -- deliberately not read off
`SimState.step_count`/internal bookkeeping, which would only prove the sim is internally
self-consistent, not that its public autoreset contract is honored), `validate_obs` every tick,
and every completed episode's length staying within `max_episode_steps + 1` (tracked the same
black-box way, via a counter this script owns, not the sim's).
"""
import sys
from pathlib import Path

import torch
import yaml

from brawl_sim.config import load_config
from brawl_sim.core.obs_schema import validate_obs
from brawl_sim.env import BrawlVecEnv

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
PRESETS_DIR = CONFIGS_DIR / "presets"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"

N_ENVS = 8
N_STEPS = 500


def config_variants() -> list:
    """[(label, overrides_dict_or_None), ...] -- see module docstring for why this is "every
    world-config variant" rather than every yaml file under configs/."""
    variants = [("default", None)]
    for preset_path in sorted(PRESETS_DIR.glob("*.yaml")):
        variants.append((preset_path.stem, yaml.safe_load(preset_path.read_text())))
    return variants


def _random_action(env: BrawlVecEnv) -> torch.Tensor:
    move = torch.randint(0, env.cfg.n_move_bins + 1, (env.n_envs,), device=env.device)
    fire = torch.randint(0, 2, (env.n_envs,), device=env.device)
    return torch.stack([move, fire], dim=1)


def _with_debug_checks(overrides: dict | None) -> dict:
    merged = dict(overrides) if overrides else {}
    merged["engine"] = {**merged.get("engine", {}), "debug_checks": True}
    return merged


def run_smoke(label: str, overrides: dict | None, device: str, n_envs: int = N_ENVS, steps: int = N_STEPS, seed: int = 0) -> None:
    """Raises on the first violation found -- no partial-success return value, same "smoke
    test" contract `scripts/sb3_smoke.py` (Step 36) already uses."""
    cfg = load_config(DEFAULT_CONFIG_PATH, overrides=_with_debug_checks(overrides))
    env = BrawlVecEnv(cfg, n_envs=n_envs, device=device, seed=seed, verbose=False)

    obs = env.reset()
    validate_obs(obs, cfg)
    prev_alive = env.state.ent_alive.clone()
    episode_step = torch.zeros(n_envs, dtype=torch.int64, device=env.device)

    for tick in range(steps):
        obs, reward, terminated, truncated, info = env.step(_random_action(env))
        validate_obs(obs, cfg)  # check_invariants already ran inside env.step() itself

        done = terminated | truncated
        # `episode_step` counts DECISIONS (one per env.step()), so the cap it is checked against
        # is max_agent_steps, not max_episode_steps -- those differ by cfg.action_repeat.
        episode_step = torch.where(done, torch.zeros_like(episode_step), episode_step + 1)
        if torch.any(episode_step > cfg.max_agent_steps + 1):
            raise AssertionError(
                f"[{label}] tick {tick}: an episode ran longer than max_agent_steps + 1 "
                f"({cfg.max_agent_steps + 1}) decisions without terminating/truncating"
            )

        new_alive = env.state.ent_alive  # POST-autoreset -- a done env's alive is a FRESH episode's
        resurrected = (~prev_alive) & new_alive & (~done).unsqueeze(-1)
        if torch.any(resurrected):
            bad = torch.nonzero(resurrected, as_tuple=False)[0].tolist()
            raise AssertionError(
                f"[{label}] tick {tick}: entity {bad[1]} in env {bad[0]} came back alive "
                "mid-episode (not via autoreset)"
            )
        prev_alive = new_alive.clone()


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--n-envs", type=int, default=N_ENVS)
    p.add_argument("--steps", type=int, default=N_STEPS)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    variants = config_variants()
    print(f"=== smoke_test: device={args.device} n_envs={args.n_envs} steps={args.steps} "
          f"({len(variants)} config variant(s)) ===")
    for label, overrides in variants:
        run_smoke(label, overrides, args.device, n_envs=args.n_envs, steps=args.steps, seed=args.seed)
        print(f"  [{label}] OK")
    print("=== ALL CONFIG VARIANTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
