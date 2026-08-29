"""Watch a trained model play one match, in the matplotlib viewer.

    python scripts/watch.py runs/mortis_ppo-.../best_model.zip
    python scripts/watch.py runs/.../final_model.zip --tier easy
    python scripts/watch.py runs/.../checkpoints/model_4000000_steps.zip --tier elite --episodes 5
    python scripts/watch.py runs/.../best_model.zip --ascii          # terminal, no GUI
    python scripts/watch.py runs/.../best_model.zip --save match.npz # replay later

The policy drives Mortis against bots pinned to ONE difficulty tier (`--tier`, default `hard`),
using the same `FixedTierHook` `training/evaluation.py` uses -- so what you watch at `--tier hard`
is exactly the population `eval/win_rate_hard` scores.

Controls in the viewer come from `render/viewer.py`'s `ReplayViewer`: space pauses, left/right
scrub, and the match is a scrubbable recording rather than a live stream, so you can step back
through the moment the hero died instead of watching it once at 20 Hz.

**The run's own `train.yaml` is found automatically** next to the model (or one directory up, for
a checkpoint) and used to rebuild the environment. That matters for more than convenience:
`map_id` indexes `cfg.map_names`, and the observation layout must match what the policy was
trained on, so a mismatched config renders the wrong map or fails to feed the network at all.
`--train-config` overrides the search.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from brawl_sim.config import load_config
from brawl_sim.constants import DeathCause
from brawl_sim.core import obs_select
from brawl_sim.core.state import snapshot as state_snapshot
from brawl_sim.env import BrawlVecEnv
from brawl_sim.training.builder import REPO_ROOT, _resolve, build_spec
from brawl_sim.training.config import load_train_config
from brawl_sim.training.curriculum import FixedTierHook
from brawl_sim.training.reward import ShapedReward
from brawl_sim.wrappers.sb3_vecenv import BrawlSB3VecEnv

try:
    from scripts.record_rollout import extra_fields
except ModuleNotFoundError:
    # `python scripts/watch.py` puts scripts/ -- not the repo root -- on sys.path, so the
    # `scripts.` namespace package isn't importable that way; under pytest the repo root IS on
    # the path and the qualified form is the one that resolves. Support both entry points.
    from record_rollout import extra_fields

_DEATH_CAUSE_NAMES = {int(DeathCause.ALIVE): "alive", int(DeathCause.COMBAT): "combat",
                      int(DeathCause.ZONE): "zone"}


def find_train_config(model_path: Path) -> Path:
    """`runs/x/final_model.zip` -> `runs/x/train.yaml`; `runs/x/checkpoints/model_N.zip` ->
    `runs/x/train.yaml`. Raises with both candidates named rather than silently falling back to
    `configs/train.yaml`, whose `env_overrides` may not be the ones the model trained under."""
    candidates = [model_path.parent / "train.yaml", model_path.parent.parent / "train.yaml"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no train.yaml found next to {model_path.name} (looked in "
        f"{', '.join(str(c) for c in candidates)}). Pass --train-config explicitly -- it must be "
        "the config this model was trained with, or the observation layout won't match."
    )


def load_model(model_path: Path, algo: str, device: str):
    if algo == "maskable_ppo":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(str(model_path), device=device), True
    from stable_baselines3 import PPO
    return PPO.load(str(model_path), device=device), False


def build_watch_env(tcfg, tier: str, seed: int, device: str, map_name: str | None = None):
    """An `n_envs=1` env pinned to `tier`, with autoreset OFF so a finished match FREEZES on its
    terminal state instead of silently starting a new episode mid-recording.

    `map_name`, if given, pins `world.map_selection` to `fixed` -- overriding whatever the run's
    own `train.yaml` used (typically `uniform`, i.e. a random map each episode) -- so `--map`
    reliably shows the requested map instead of just biasing the odds."""
    overrides = dict(tcfg.run.env_overrides or {})
    if map_name is not None:
        overrides["world"] = {**overrides.get("world", {}), "map_selection": "fixed", "fixed_map": map_name}
    env_cfg = load_config(_resolve(tcfg.run.env_config), overrides=overrides or None)
    agent_spec = obs_select.load_agent_spec(_resolve(tcfg.run.agent_obs), env_cfg)
    sim = BrawlVecEnv(
        env_cfg, n_envs=1, device=device, seed=seed,
        reward_fn=ShapedReward(tcfg.reward, track_terms=False),
        spec=build_spec(tcfg), verbose=False, autoreset=False,
    )
    if tier is not None:
        if not tcfg.curriculum.tiers:
            raise ValueError("--tier needs `curriculum.tiers` defined in the training config")
        sim.params_hook = FixedTierHook.uniform(tcfg.curriculum.tiers, sim.device, 1, tier)
    venv = BrawlSB3VecEnv(sim, agent_spec, sim.reward_fn, info_mode="episode")
    return sim, venv, env_cfg


def maybe_wrap_vecnormalize(venv, model_path: Path, tcfg, verbose: bool = True):
    """Applies the run's saved observation statistics, if it used any.

    Only load-bearing when the run trained with `normalize.obs: true` -- but in that case a
    policy fed raw observations behaves like a differently-trained (and much worse) one, which
    is very easy to misread as "the checkpoint is bad" rather than "the viewer is misconfigured".
    """
    if not tcfg.normalize.obs:
        return venv
    candidates = sorted(model_path.parent.glob("*vecnormalize*.pkl")) or \
        sorted(model_path.parent.parent.glob("*vecnormalize*.pkl"))
    if not candidates:
        if verbose:
            print("[watch] WARNING: this run normalized observations but no *vecnormalize*.pkl "
                  "was found; the policy will see unnormalized inputs and play worse than it "
                  "really is.")
        return venv
    from stable_baselines3.common.vec_env import VecNormalize
    wrapped = VecNormalize.load(str(candidates[-1]), venv)
    wrapped.training = False
    wrapped.norm_reward = False
    if verbose:
        print(f"[watch] loaded observation statistics from {candidates[-1].name}")
    return wrapped


def play_match(model, venv, sim, uses_masks: bool, deterministic: bool, max_steps: int):
    """Runs one episode, returning `(frames, summary)`. `frames` is the stacked-array dict
    `ReplayViewer`/`render_ascii` consume -- built from the same
    `state_snapshot` + `record_rollout.extra_fields` pair a recorded rollout would produce, so
    this is the renderer's normal input, not a special case.

    **One frame per SIM TICK (20 Hz), not per decision.** `max_steps` is in decisions
    (`cfg.max_agent_steps`) and each `venv.step()` advances the world `cfg.action_repeat` ticks,
    so sampling at the step boundary would produce a 4 Hz replay in which projectiles teleport
    between frames. `BrawlVecEnv.tick_hook` fires after every sub-tick, which is where the
    frames are actually captured -- the replay is at the sim's own rate no matter what the
    decision rate is, and looks identical at `action_repeat=1` and `5`.

    The hook is uninstalled in a `finally`, so an exception mid-match can't leave a snapshotting
    callback (and its host syncs) attached to an env someone else goes on to use."""
    obs = venv.reset()
    frames = [_frame(sim)]
    summary = {"steps": 0, "terminated": False, "truncated": False,
               "ticks_per_step": sim.cfg.action_repeat, "dt": sim.cfg.dt}

    sim.tick_hook = lambda env: frames.append(_frame(env))
    try:
        _drive_match(model, venv, uses_masks, deterministic, max_steps, obs, summary)
    finally:
        sim.tick_hook = None

    summary.update(_hero_stats(sim))
    stacked = {name: np.stack([f[name] for f in frames], axis=0) for name in frames[0]}
    return stacked, summary


def _drive_match(model, venv, uses_masks, deterministic, max_steps, obs, summary) -> None:
    """The rollout loop itself. Frames are appended by `play_match`'s tick_hook, not here."""
    for _ in range(max_steps):
        kwargs = {"action_masks": venv.action_masks()} if uses_masks else {}
        action, _ = model.predict(obs, deterministic=deterministic, **kwargs)
        obs, reward, dones, infos = venv.step(action)
        summary["steps"] += 1
        if dones[0]:
            info = infos[0]
            summary["terminated"] = not info.get("TimeLimit.truncated", False)
            summary["truncated"] = bool(info.get("TimeLimit.truncated", False))
            summary["rank"] = int(info["outcome"]["rank"])
            summary["won"] = bool(info["outcome"]["won"])
            summary["return"] = float(info["episode"]["r"])
            break


def _frame(sim) -> dict:
    frame = state_snapshot(sim.state, 0)
    frame.update(extra_fields(sim))
    return frame


def _hero_stats(sim) -> dict:
    state = sim.state
    cause = int(state.ent_death_cause[0, 0])
    return {
        "hp": float(state.ent_hp[0, 0]),
        "max_hp": float(state.ent_max_hp[0, 0]),
        "alive": bool(state.ent_alive[0, 0]),
        "kills": int(state.ent_kills[0, 0]),
        "cubes": int(state.ent_cubes[0, 0]),
        "damage_dealt": float(state.ent_damage_dealt[0, 0]),
        "damage_taken": float(state.ent_damage_taken[0, 0]),
        "shots_fired": int(state.ent_shots_fired[0, 0]),
        "enemies_left": int(state.n_alive[0]) - int(state.ent_alive[0, 0]),
        "death_cause": _DEATH_CAUSE_NAMES.get(cause, str(cause)),
    }


def print_summary(summary: dict, tier: str, index: int, total: int) -> None:
    outcome = "WON" if summary.get("won") else ("timeout" if summary["truncated"] else "died")
    label = f"match {index + 1}/{total}" if total > 1 else "match"
    # summary["steps"] counts DECISIONS; ticks and seconds are derived from the env's own
    # action_repeat/dt rather than a hardcoded 0.05, which was only ever right at action_repeat=1.
    ticks = summary["steps"] * summary["ticks_per_step"]
    print(f"\n  {label} vs {tier} bots: {outcome}"
          f"   rank {summary.get('rank', '?')}   {summary['steps']} decisions / {ticks} ticks "
          f"({ticks * summary['dt']:.0f}s)")
    print(f"    hp {summary['hp']:.0f}/{summary['max_hp']:.0f}"
          f"   kills {summary['kills']}   cubes {summary['cubes']}"
          f"   enemies left {summary['enemies_left']}")
    print(f"    damage dealt {summary['damage_dealt']:.0f}"
          f"   taken {summary['damage_taken']:.0f}"
          f"   shots {summary['shots_fired']}"
          f"   {'death: ' + summary['death_cause'] if not summary['alive'] else 'survived'}"
          f"   return {summary.get('return', float('nan')):.2f}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="path to a saved model .zip")
    p.add_argument("--tier", default="hard",
                   help="difficulty tier to play against (default: hard); 'none' uses "
                        "configs/brawlers.yaml verbatim with no tier multipliers")
    p.add_argument("--map", default=None,
                   help="pin a specific map (default: whatever the run's train.yaml uses, "
                        "usually a random pick from world.maps each episode); must be one of "
                        "world.maps in the resolved env config, e.g. open, bushy, walled, "
                        "skull_creek, feast_or_famine, scorched_stone, island_invasion")
    p.add_argument("--train-config", default=None,
                   help="training config the model was trained with (default: auto-detect)")
    p.add_argument("--episodes", type=int, default=1,
                   help="play N matches; the viewer shows the LAST one, stats print for all")
    p.add_argument("--seed", type=int, default=0, help="env seed; change it for a different match")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="cpu is usually faster for a single env (default: cpu)")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking the argmax")
    p.add_argument("--fps", type=int, default=20, help="viewer playback rate (sim runs at 20 Hz)")
    p.add_argument("--ascii", action="store_true", help="print the final frame as text, no GUI")
    p.add_argument("--save", default=None, help="also write the rollout to this .npz")
    p.add_argument("--video", default=None,
                   help="also write the LAST match to this .mp4 or .gif, e.g. --video match.mp4 "
                        "(see brawl_sim.render.viewer.ReplayViewer.save for what .mp4 needs)")
    p.add_argument("--no-view", action="store_true", help="stats only; skip the viewer entirely")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[watch] no such model: {model_path}", file=sys.stderr)
        return 1

    train_config = Path(args.train_config) if args.train_config else find_train_config(model_path)
    tcfg = load_train_config(train_config)
    tier = None if args.tier == "none" else args.tier
    print(f"[watch] {model_path.name}  vs  {tier or 'brawlers.yaml (no tier)'} bots"
          f"   [config: {train_config}]")

    model, uses_masks = load_model(model_path, tcfg.run.algo, args.device)

    frames = summary = None
    for i in range(args.episodes):
        sim, venv, env_cfg = build_watch_env(tcfg, tier, args.seed + i, args.device, map_name=args.map)
        venv = maybe_wrap_vecnormalize(venv, model_path, tcfg, verbose=(i == 0))
        _check_spaces(model, venv, train_config)
        frames, summary = play_match(
            model, venv, sim, uses_masks,
            deterministic=not args.stochastic, max_steps=env_cfg.max_agent_steps,
        )
        print_summary(summary, tier or "default", i, args.episodes)

    if args.save:
        np.savez_compressed(args.save, **frames)
        print(f"\n[watch] rollout -> {args.save}  "
              f"(replay with: python -m brawl_sim.render.viewer {args.save})")

    if args.ascii:
        from brawl_sim.render.ascii import render_ascii, status_line
        last = {name: arr[-1] for name, arr in frames.items()}
        print()
        print(status_line(last))
        print(render_ascii(last, sim.bank, env_cfg))
    elif args.video or not args.no_view:
        from brawl_sim.render.viewer import ReplayViewer
        viewer = ReplayViewer(frames, sim.bank, env_cfg, fps=args.fps)
        if args.video:
            viewer.save(args.video, fps=args.fps)
            print(f"[watch] rollout -> {args.video}")
        if not args.no_view:
            print("\n[watch] opening viewer -- space pauses, left/right scrub, q quits")
            viewer.show()
    return 0


def _check_spaces(model, venv, train_config: Path) -> None:
    """A model trained under a different `agent_obs.yaml`/`env_overrides` has a different input
    layout. Caught here with a pointer at the likely cause, rather than as a bare shape error
    from inside the policy's first Linear layer."""
    if model.observation_space != venv.observation_space:
        raise SystemExit(
            f"observation space mismatch: the model expects\n  {model.observation_space}\n"
            f"but {train_config} builds\n  {venv.observation_space}\n"
            "This config is not the one the model was trained with -- pass the right "
            "--train-config (the run directory's own archived train.yaml)."
        )


if __name__ == "__main__":
    sys.exit(main())
