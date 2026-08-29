"""End-to-end SB3 plumbing validation. See BRAWL_SIM_BUILD_PLAN.md Step 36.

**This is not a training run** -- `total_timesteps` defaults are small enough that no agent
here learns anything useful. The point is proving every interface built in Steps 32-35 actually
connects: `check_env`, `BrawlSB3VecEnv`, `VecMonitor`, `MaskablePPO` AND plain `PPO`, action
masking, and a save/load round trip, all in one script, on whichever `device` is passed in.

Deliberately touches nothing under `brawl_sim/core/` -- every piece this script exercises
(`BrawlGymEnv`, `BrawlSB3VecEnv`, `BrawlFeaturesExtractor`/`default_policy_kwargs`) already
existed from Steps 32-35; this step is integration proof, not new simulator surface.

`run_smoke(...)` is the reusable entry point (small, overridable `n_envs`/`total_timesteps` for
`tests/test_sb3_integration.py` to call cheaply); `main()` is the CLI, using the plan's own
`n_steps=64, batch_size=128, total_timesteps=2048` defaults and printing observed steps/sec so
the SB3 boundary cost is visible from the first run, per the plan's own request.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecMonitor

from brawl_sim.config import load_config
from brawl_sim.core import obs_select
from brawl_sim.core.reward import ExampleReward
from brawl_sim.env import BrawlVecEnv
from brawl_sim.wrappers.gym_single import BrawlGymEnv
from brawl_sim.wrappers.sb3_features import default_policy_kwargs
from brawl_sim.wrappers.sb3_vecenv import BrawlSB3VecEnv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_AGENT_OBS_PATH = REPO_ROOT / "configs" / "agent_obs.yaml"


def _make_venv(cfg, agent_spec, device: str, n_envs: int, seed: int) -> VecMonitor:
    env = BrawlVecEnv(cfg, n_envs=n_envs, device=device, seed=seed)
    return VecMonitor(BrawlSB3VecEnv(env, agent_spec, ExampleReward(), info_mode="episode"))


def run_smoke(
    cfg, agent_spec, device: str = "cpu", n_envs: int = 8,
    n_steps: int = 64, batch_size: int = 128, total_timesteps: int = 2048,
    verbose: int = 0, save_dir=None, log=print,
) -> dict:
    """Runs all 8 plan steps in order, returns a dict of {step_name: result/timing}. Raises on
    any failure (shape/dtype/space errors, a save/load mismatch, ...) -- there is no
    partial-success return value, matching "this is not a training run, it's a plumbing check"."""
    results: dict = {}
    policy_kwargs = default_policy_kwargs(agent_spec, cfg)

    # 1. check_env(BrawlGymEnv(cfg)) -> no errors
    gym_env = BrawlGymEnv(BrawlVecEnv(cfg, n_envs=1, device=device, seed=0), agent_spec, ExampleReward())
    check_env(gym_env, warn=True)
    results["check_env"] = "OK"
    log("[1/8] check_env OK")

    # 2-3. venv = BrawlSB3VecEnv(...); venv = VecMonitor(venv)
    venv = _make_venv(cfg, agent_spec, device, n_envs, seed=1)
    log("[2/8][3/8] BrawlSB3VecEnv + VecMonitor constructed OK")

    # 4-5. MaskablePPO, learn
    model = MaskablePPO(
        "MultiInputPolicy", venv, policy_kwargs=policy_kwargs,
        n_steps=n_steps, batch_size=batch_size, verbose=verbose, device=device,
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.perf_counter() - t0
    results["maskable_ppo_steps_per_sec"] = total_timesteps / elapsed
    log(f"[4/8][5/8] MaskablePPO.learn completed in {elapsed:.1f}s "
        f"({results['maskable_ppo_steps_per_sec']:.1f} steps/sec)")

    # 6. predict with action_masks -> valid actions
    obs = venv.reset()
    action_masks = venv.action_masks()
    action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
    assert action.shape == (n_envs, len(cfg.action_nvec))
    for row in action:
        assert venv.action_space.contains(row.astype(venv.action_space.dtype))
    results["predict_masked"] = "OK"
    log("[6/8] masked predict() produced valid actions")

    # 7. save / load round trip -> identical predictions
    save_dir = Path(save_dir) if save_dir is not None else Path(tempfile.mkdtemp())
    save_path = save_dir / "sb3_smoke_maskable_ppo.zip"
    model.save(save_path)
    loaded = MaskablePPO.load(save_path, env=venv, device=device)
    action_reloaded, _ = loaded.predict(obs, deterministic=True, action_masks=action_masks)
    if not np.array_equal(action, action_reloaded):
        raise AssertionError("save/load round trip produced a different action from the same obs")
    results["save_load_roundtrip"] = "OK"
    log("[7/8] save/load round trip: identical predictions")

    # 8. repeat 4-5 with plain PPO (no masking)
    venv_unmasked = _make_venv(cfg, agent_spec, device, n_envs, seed=2)
    model_unmasked = PPO(
        "MultiInputPolicy", venv_unmasked, policy_kwargs=policy_kwargs,
        n_steps=n_steps, batch_size=batch_size, verbose=verbose, device=device,
    )
    t0 = time.perf_counter()
    model_unmasked.learn(total_timesteps=total_timesteps)
    elapsed_unmasked = time.perf_counter() - t0
    results["plain_ppo_steps_per_sec"] = total_timesteps / elapsed_unmasked
    log(f"[8/8] plain PPO.learn completed in {elapsed_unmasked:.1f}s "
        f"({results['plain_ppo_steps_per_sec']:.1f} steps/sec)")

    obs_unmasked = venv_unmasked.reset()
    action_unmasked, _ = model_unmasked.predict(obs_unmasked, deterministic=True)
    assert action_unmasked.shape == (n_envs, len(cfg.action_nvec))
    results["predict_unmasked"] = "OK"

    return results


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--agent-obs", default=str(DEFAULT_AGENT_OBS_PATH))
    p.add_argument("--preset", default=None, help="optional configs/presets/*.yaml overrides file")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--total-timesteps", type=int, default=2048)
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    overrides = None
    if args.preset:
        import yaml
        overrides = yaml.safe_load(Path(args.preset).read_text())
    cfg = load_config(args.config, overrides=overrides)
    agent_spec = obs_select.load_agent_spec(args.agent_obs, cfg)

    print(f"=== sb3_smoke: device={args.device} n_envs={args.n_envs} "
          f"total_timesteps={args.total_timesteps} ===")
    results = run_smoke(
        cfg, agent_spec, device=args.device, n_envs=args.n_envs,
        n_steps=args.n_steps, batch_size=args.batch_size,
        total_timesteps=args.total_timesteps, verbose=args.verbose,
    )
    print("=== ALL 8 STEPS PASSED ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
