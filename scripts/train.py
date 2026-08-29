"""PPO training entry point.

    python scripts/train.py                                  # configs/train.yaml as-is
    python scripts/train.py --set run.n_envs=1024 --set ppo.n_steps=256
    python scripts/train.py --smoke                          # 30-second wiring check on CPU
    python scripts/train.py --resume runs/mortis_ppo-.../checkpoints/model_4000000_steps.zip

Every knob lives in configs/train.yaml (see that file's own comments); `--set` patches single
dotted keys on top of it for one-off sweeps without editing the file. The merged config is
archived into the run directory, so a run is always reproducible from its own output.

Run layout:

    runs/<name>-<timestamp>/
      train.yaml        the exact merged config this run used
      curriculum.json   current stage + full transition history, rewritten on every stage change
      logs/progress.csv every logged scalar, one row per dump -- plot this
      logs/log.txt      the same tables the console printed
      logs/events.*     TensorBoard, if the `tensorboard` package is installed
      logs/eval_<tier>/ per-difficulty event files -- overlaid on one eval/win_rate chart
      checkpoints/      periodic model .zip + VecNormalize .pkl
      best_model.zip    best mean stationary-eval win rate so far
      final_model.zip   + final_vecnormalize.pkl

View progress with `tensorboard --logdir runs`, or read `logs/progress.csv` directly
(`pandas.read_csv`) -- the CSV is written whether or not TensorBoard is available.

Four scalar groups answer four different questions, all in the same log:
  train/*        SB3's own loss/entropy/KL stats, PLUS (added here) a continuous, never-reset
                 win rate + episode breakdown (kills, damage, cubes) vs whatever the hero is
                 training against RIGHT NOW -- moves with the curriculum, one section in
                 TensorBoard, "how is training going" at a glance
  rollout/       SB3's own ep_rew_mean / ep_len_mean
  curriculum/    the curriculum's OWN decision variable -- resets to empty on every stage change
  eval/*         stationary -- fixed difficulty, fixed seed, comparable ACROSS THE WHOLE RUN
"""
import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from stable_baselines3.common.callbacks import CheckpointCallback

# Support `python scripts/train.py` from a source checkout without requiring
# `pip install -e .` first.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brawl_sim.training import schedules
from brawl_sim.training.builder import build_run, make_logger, tensorboard_available
from brawl_sim.training.callbacks import (
    CurriculumCallback, TierEvalCallback, TrainingMonitorCallback, VecNormalizeCheckpoint,
)
from brawl_sim.training.config import deep_merge, load_train_config, parse_overrides
from brawl_sim.training.evaluation import TierEvaluator

DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "train.yaml"

# --smoke: the smallest run that still exercises every moving part (curriculum hook and an
# actual stage transition, action masking, both schedules, VecNormalize, checkpointing) end to
# end. Not a training run. `presets/debug_tiny.yaml` is layered in as env_overrides specifically
# so episodes cap at 300 steps instead of 3000 -- without it, no episode would finish inside the
# smoke budget and the curriculum half would never be exercised at all.
_SMOKE = {
    "run": {"device": "cpu", "n_envs": 8, "total_timesteps": 8192, "name": "smoke",
            "checkpoint_every_steps": 4096, "tensorboard": False},
    "ppo": {"n_steps": 64, "batch_size": 128, "n_epochs": 2},
    "curriculum": {"window_episodes": 8, "min_episodes_at_stage": 8},
    "eval": {"every_timesteps": 4096, "episodes_per_tier": 2},
}
_SMOKE_PRESET = REPO_ROOT / "configs" / "presets" / "debug_tiny.yaml"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_TRAIN_CONFIG), help="training config YAML")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted override, repeatable (e.g. --set ppo.n_steps=256)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU run that proves the wiring; overrides run/ppo/curriculum sizes")
    p.add_argument("--resume", default=None, metavar="MODEL_ZIP",
                   help="continue from a saved model; loads the sibling VecNormalize stats if present")
    p.add_argument("--curriculum-stage", type=int, default=None, metavar="N",
                   help="start (or resume) at this 0-indexed stage instead of stage 0")
    p.add_argument("--out-dir", default=None, help="override run.out_dir")
    p.add_argument("--no-stamp", action="store_true",
                   help="use runs/<name> verbatim instead of runs/<name>-<timestamp>")
    return p.parse_args(argv)


def _make_run_dir(tcfg, out_dir: str | None, stamp: bool) -> Path:
    root = Path(out_dir) if out_dir else Path(tcfg.run.out_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    name = tcfg.run.name
    if stamp:
        name = f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def _banner(tcfg, parts, run_dir: Path) -> None:
    env_cfg = parts["env_cfg"]
    sim = parts["sim"]
    print("=" * 78)
    print(f"  brawl-sim PPO  |  {tcfg.run.name}")
    print("=" * 78)
    print(f"  run dir        {run_dir}")
    print(f"  algo           {tcfg.run.algo}   device {sim.device}   seed {tcfg.run.seed}")
    print(f"  envs           {sim.n_envs}  x  {tcfg.ppo.n_steps} steps "
          f"= {tcfg.run.n_envs * tcfg.ppo.n_steps:,} transitions/rollout")
    print(f"  total          {tcfg.run.total_timesteps:,} timesteps")
    print(f"  learning rate  {schedules.describe(tcfg.learning_rate)}")
    print(f"  clip range     {schedules.describe(tcfg.clip_range)}")
    print(f"  world          {env_cfg.map_h}x{env_cfg.map_w}, {env_cfg.n_enemies} enemies, "
          f"zone {'on' if env_cfg.zone_enabled else 'off'}")
    # Both units, on purpose: nearly every "why is this number 5x off" question about a run comes
    # from reading a decision count as a tick count or vice versa (see configs/default.yaml's
    # sim.action_repeat comment). `total` above is in DECISIONS, like every SB3 timestep.
    print(f"  timing         {1 / env_cfg.dt:.0f} Hz sim / {1 / env_cfg.agent_dt:.4g} Hz decisions "
          f"(action_repeat {env_cfg.action_repeat})   episode cap "
          f"{env_cfg.max_agent_steps} decisions = {env_cfg.max_episode_steps} ticks "
          f"= {env_cfg.max_episode_steps * env_cfg.dt:.0f}s")
    norm = [k for k, on in (("obs", tcfg.normalize.obs), ("reward", tcfg.normalize.reward)) if on]
    print(f"  normalize      {', '.join(norm) if norm else 'off'}")
    print(f"  logs           {run_dir / 'logs'}  (progress.csv, log.txt"
          f"{', tensorboard' if tcfg.run.tensorboard and tensorboard_available() else ''})")
    if tcfg.eval.enabled:
        tiers = tcfg.eval.tiers or tuple(tcfg.curriculum.tiers)
        print(f"  eval           every {tcfg.eval.every_timesteps:,} steps | "
              f"{tcfg.eval.episodes_per_tier} matches x {len(tiers)} tiers "
              f"({', '.join(tiers)}) | seed {tcfg.eval.seed}")
    else:
        print("  eval           DISABLED (no stationary win-rate signal will be logged)")
    if parts["curriculum"] is None:
        print("  curriculum     DISABLED (bots use configs/brawlers.yaml verbatim)")
    else:
        m = parts["curriculum"]
        print(f"  curriculum     {m.n_stages} stages, starting at "
              f"{m.stage_index + 1}/{m.n_stages} {m.stage.name!r}")
        for i, stage in enumerate(m.cfg.stages):
            mix = ", ".join(f"{n} {w:.0%}" for n, w in stage.tier_weights.items() if w > 0)
            gate = ("terminal" if stage.advance_win_rate is None
                    else f"advance at {stage.advance_win_rate:.0%} win rate")
            marker = ">" if i == m.stage_index else " "
            print(f"                 {marker} {i}. {stage.name:<12} {mix:<40} ({gate})")
    print("=" * 78)


def main(argv=None) -> int:
    args = _parse_args(argv)

    overrides = parse_overrides(args.overrides)
    if args.smoke:
        smoke = deep_merge(_SMOKE, {"run": {"env_overrides": yaml.safe_load(_SMOKE_PRESET.read_text())}})
        # Applied UNDER the explicit --set overrides, so `--smoke --set run.n_envs=16` works.
        overrides = deep_merge(smoke, overrides)
    tcfg = load_train_config(args.config, overrides=overrides)

    run_dir = _make_run_dir(tcfg, args.out_dir, stamp=not (args.no_stamp or args.smoke))
    (run_dir / "train.yaml").write_text(yaml.safe_dump(tcfg.raw, sort_keys=False))

    model, venv, parts = build_run(tcfg)

    curriculum = parts["curriculum"]
    if curriculum is not None and args.curriculum_stage is not None:
        curriculum.set_stage(args.curriculum_stage)

    if args.resume:
        model = _resume(model, venv, tcfg, Path(args.resume), curriculum, run_dir,
                        stage_pinned=args.curriculum_stage is not None)

    # After any resume: `_resume` returns a DIFFERENT model object (algo.load builds a new one),
    # and a logger installed on the pre-resume model would be silently discarded with it.
    model.set_logger(make_logger(tcfg, run_dir / "logs"))

    _banner(tcfg, parts, run_dir)

    callbacks = []
    if tcfg.run.info_mode in ("episode", "full"):
        # Always on, independent of curriculum/eval: the continuous train/* win-rate + episode
        # breakdown described in TrainingMonitorCallback's own docstring. Reuses
        # curriculum.window_episodes as its rolling window size purely as a shared "how many
        # recent episodes" knob -- it works identically whether curriculum.enabled is true or not.
        callbacks.append(TrainingMonitorCallback(
            window_episodes=tcfg.curriculum.window_episodes, verbose=tcfg.run.verbose,
        ))
    else:
        print(f"[train] run.info_mode={tcfg.run.info_mode!r}: no per-episode outcome data, so "
              "train/win_rate and friends will not be logged. Use 'episode' or 'full' to get them.")

    evaluator = None
    if tcfg.eval.enabled:
        evaluator = TierEvaluator(tcfg, verbose=tcfg.run.verbose)
        callbacks.append(TierEvalCallback(
            evaluator, tcfg.eval.every_timesteps, run_dir / "logs",
            at_start=tcfg.eval.at_start, best_model_path=run_dir / "best_model.zip",
            verbose=tcfg.run.verbose,
        ))
    if curriculum is not None:
        callbacks.append(CurriculumCallback(
            curriculum, tcfg.curriculum, reward_fn=parts["reward_fn"],
            state_path=run_dir / "curriculum.json", verbose=tcfg.run.verbose,
        ))
    if tcfg.run.checkpoint_every_steps > 0:
        # CheckpointCallback counts CALLS (one per vec-env step), not timesteps, so the interval
        # is divided by n_envs to make `checkpoint_every_steps` mean what it says.
        save_freq = max(1, tcfg.run.checkpoint_every_steps // tcfg.run.n_envs)
        callbacks.append(CheckpointCallback(
            save_freq=save_freq, save_path=str(run_dir / "checkpoints"),
            name_prefix="model", verbose=tcfg.run.verbose,
        ))
        if tcfg.normalize.enabled:
            callbacks.append(VecNormalizeCheckpoint(
                save_freq, run_dir / "checkpoints", verbose=tcfg.run.verbose,
            ))

    t0 = time.perf_counter()
    interrupted = False
    try:
        model.learn(
            total_timesteps=tcfg.run.total_timesteps,
            callback=callbacks or None,
            log_interval=tcfg.run.log_interval,
            reset_num_timesteps=not args.resume,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupted] saving current model before exit...")

    elapsed = time.perf_counter() - t0
    model.save(run_dir / "final_model.zip")
    if tcfg.normalize.enabled:
        venv.save(str(run_dir / "final_vecnormalize.pkl"))
    venv.close()
    if evaluator is not None:
        evaluator.close()

    steps = model.num_timesteps
    print(f"\n[done{' (interrupted)' if interrupted else ''}] {steps:,} timesteps in "
          f"{elapsed / 60:.1f} min ({steps / max(elapsed, 1e-9):,.0f} steps/sec)")
    print(f"[done] model -> {run_dir / 'final_model.zip'}")
    if curriculum is not None:
        print(f"[done] finished at curriculum stage {curriculum.stage_index + 1}/"
              f"{curriculum.n_stages}: {curriculum.stage.name!r}")
        print(f"[done] curriculum history -> {run_dir / 'curriculum.json'}")
    return 0


def _resume(model, venv, tcfg, model_path: Path, curriculum, run_dir: Path, stage_pinned: bool = False):
    """Loads `model_path`'s weights and optimizer state into the freshly built run, plus the
    sibling `*vecnormalize*.pkl` if one is there.

    Resuming the CURRICULUM is deliberately explicit rather than automatic: the stage lives in
    the source run's `curriculum.json`, which is copied forward and applied here, but
    `--curriculum-stage` (applied before this function runs) always wins. Silently resuming a
    finetune at stage 0 against a policy that had already reached the final stage is the kind of
    mistake that costs a day of GPU time before anyone notices the win rate looks too good."""
    algo_cls = type(model)
    loaded = algo_cls.load(str(model_path), env=venv, device=tcfg.run.device)
    print(f"[resume] loaded {model_path} at {loaded.num_timesteps:,} timesteps")

    stats = sorted(model_path.parent.glob("*vecnormalize*.pkl"))
    if stats and tcfg.normalize.enabled:
        from stable_baselines3.common.vec_env import VecNormalize
        restored = VecNormalize.load(str(stats[-1]), venv.venv)
        restored.training = True
        loaded.set_env(restored)
        print(f"[resume] loaded VecNormalize stats from {stats[-1]}")
    elif tcfg.normalize.enabled:
        print(f"[resume] WARNING: normalize is on but no *vecnormalize*.pkl was found next to "
              f"{model_path.name}; the policy will see differently-scaled inputs than it "
              "trained on until the fresh running statistics converge")

    source_state = model_path.parent.parent / "curriculum.json"
    if curriculum is not None and source_state.exists():
        if stage_pinned:
            print(f"[resume] --curriculum-stage given; ignoring the stage recorded in "
                  f"{source_state} and staying at {curriculum.stage.name!r}")
        else:
            import json
            curriculum.load_state_dict(json.loads(source_state.read_text()))
            print(f"[resume] curriculum resumed at stage {curriculum.stage_index + 1}/"
                  f"{curriculum.n_stages}: {curriculum.stage.name!r}")
        shutil.copy(source_state, run_dir / "curriculum.json")
    elif curriculum is not None:
        print(f"[resume] no curriculum.json found at {source_state}; starting at stage "
              f"{curriculum.stage_index + 1}/{curriculum.n_stages}: {curriculum.stage.name!r}")
    return loaded


if __name__ == "__main__":
    sys.exit(main())
