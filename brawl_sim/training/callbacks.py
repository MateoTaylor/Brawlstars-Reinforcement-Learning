"""Training callbacks: curriculum advancement, stationary eval, and continuous training-side
logging.

`CurriculumCallback` is the host-side half of the curriculum: `training/curriculum.py` applies
stage N on device inside `reset_envs`, and this decides when N becomes N+1, because that needs
episode OUTCOMES, which exist only after `BrawlSB3VecEnv` crosses to the host.

**The win signal.** `BrawlSB3VecEnv._build_infos` attaches `info["outcome"] = {"rank", "won"}`
(plus `info["episode_stats"]`) to every env that finished this step; that needs `info_mode`
`"episode"` or `"full"`, which `TrainConfig` validation enforces for the curriculum. `won` means
the hero was the last one standing: a timeout with the hero alive but not alone is not a win, or
the agent could advance the curriculum by learning to hide.

**Reading the logs.** Overlay `curriculum/stage_index` on `rollout/ep_rew_mean`: a shaped
reward's level is only comparable WITHIN a stage. `curriculum/tier_spawned_*` (what the sim
rolled) should match `tier_weight_*` (what the config asked for) to within sampling noise.
`curriculum/win_rate` is the curriculum's decision variable and is reset on every stage change,
so a strong easy-stage window cannot clear the next stage; `train/*` (`TrainingMonitorCallback`)
is the continuous view. Compare runs on `eval/*`, never on `train/win_rate`.

After a transition the window also skips each env's first finish, whose bots were drawn from the
previous stage; `curriculum/envs_on_previous_stage` counts the envs still waiting for one.
"""
import json
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from . import evaluation
from .config import CurriculumConfig
from .curriculum import CurriculumManager


class TrainingMonitorCallback(BaseCallback):
    """Continuous, never-reset win rate and play breakdown against whatever the hero trains on
    right now. Unlike `curriculum/win_rate` it never resets, and unlike stationary eval it follows
    the curriculum.

    Logged under `train/`, every rollout, over the last `window_episodes` finished TRAINING
    episodes (`scripts/train.py` passes `curriculum.window_episodes`):

      train/win_rate            fraction of the window the hero was last alive
      train/mean_rank           mean 0-indexed placement (0 = won)
      train/episodes_total      cumulative count since training started, never reset
      train/episodes_window     how many of the last `window_episodes` are filled in right now
      train/kills_mean          mean hero kills per finished episode
      train/damage_dealt_mean   mean hero damage dealt per finished episode
      train/damage_taken_mean   mean hero damage taken per finished episode
      train/cubes_mean          mean power cubes collected per finished episode
      train/shots_fired_mean    mean shots fired per finished episode

    Reads `info["outcome"]` / `info["episode_stats"]`, which need `run.info_mode` `"episode"` or
    `"full"`; under `"minimal"` `scripts/train.py` does not install this callback and prints a
    note instead.
    """

    def __init__(self, window_episodes: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._window_size = window_episodes
        self._won = deque(maxlen=window_episodes)
        self._ranks = deque(maxlen=window_episodes)
        self._kills = deque(maxlen=window_episodes)
        self._damage_dealt = deque(maxlen=window_episodes)
        self._damage_taken = deque(maxlen=window_episodes)
        self._cubes = deque(maxlen=window_episodes)
        self._shots_fired = deque(maxlen=window_episodes)
        self.episodes_total = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            outcome = info.get("outcome")
            if outcome is None:
                continue
            self._won.append(bool(outcome["won"]))
            self._ranks.append(int(outcome["rank"]))
            self.episodes_total += 1
            stats = info.get("episode_stats")
            if stats is not None:
                self._kills.append(stats["kills"])
                self._damage_dealt.append(stats["damage_dealt"])
                self._damage_taken.append(stats["damage_taken"])
                self._cubes.append(stats["cubes"])
                self._shots_fired.append(stats["shots_fired"])
        return True

    def _on_rollout_end(self) -> None:
        rec = self.logger.record
        rec("train/win_rate", _mean(self._won, default=0.0))
        rec("train/mean_rank", _mean(self._ranks, default=float("nan")))
        rec("train/episodes_total", self.episodes_total)
        rec("train/episodes_window", len(self._won))
        rec("train/kills_mean", _mean(self._kills, default=float("nan")))
        rec("train/damage_dealt_mean", _mean(self._damage_dealt, default=float("nan")))
        rec("train/damage_taken_mean", _mean(self._damage_taken, default=float("nan")))
        rec("train/cubes_mean", _mean(self._cubes, default=float("nan")))
        rec("train/shots_fired_mean", _mean(self._shots_fired, default=float("nan")))


def _mean(values, default: float) -> float:
    return (sum(values) / len(values)) if values else default


class CurriculumCallback(BaseCallback):
    """Tracks a rolling win rate and steps `manager` through its stages.

    `state_path`, when given, receives a JSON file rewritten on every stage change (and at the end
    of training) with the current stage and the transition history, so a resumed run picks up at
    the right difficulty.
    """

    def __init__(
        self, manager: CurriculumManager, cfg: CurriculumConfig,
        reward_fn=None, state_path=None, verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.manager = manager
        self.cfg = cfg
        self.reward_fn = reward_fn
        self.state_path = Path(state_path) if state_path is not None else None

        self._window = deque(maxlen=cfg.window_episodes)   # bool: did the hero win?
        self._ranks = deque(maxlen=cfg.window_episodes)    # int: 0-indexed placement
        self.episodes_at_stage = 0
        self.total_episodes = 0
        self._stage_start_timesteps = 0
        self.history: list[dict] = []
        # Per env: were the bots of the episode it is playing now drawn from the PREVIOUS stage?
        # See the module docstring. Sized on the first step, from the infos list (one dict per env).
        self._stale: np.ndarray | None = None

    # ---- SB3 hooks ---------------------------------------------------------------------

    def _on_training_start(self) -> None:
        self._stage_start_timesteps = self.num_timesteps
        self._banner("curriculum start")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        if self._stale is None or len(self._stale) != len(infos):
            self._stale = np.zeros(len(infos), dtype=bool)
        for i, info in enumerate(infos):
            outcome = info.get("outcome")
            if outcome is None:
                continue
            self.total_episodes += 1
            if self._stale[i]:
                # Its bots came from the previous stage. The env reset on this same step, under
                # the current stage, so its next episode counts.
                self._stale[i] = False
                continue
            self._window.append(bool(outcome["won"]))
            self._ranks.append(int(outcome["rank"]))
            self.episodes_at_stage += 1
        self._maybe_transition()
        return True

    def _on_rollout_end(self) -> None:
        self._log()

    def _on_training_end(self) -> None:
        self._write_state()
        self._banner("curriculum end")

    # ---- decision ----------------------------------------------------------------------

    @property
    def win_rate(self) -> float:
        return (sum(self._window) / len(self._window)) if self._window else 0.0

    @property
    def mean_rank(self) -> float:
        return (sum(self._ranks) / len(self._ranks)) if self._ranks else float("nan")

    def _maybe_transition(self) -> None:
        stage = self.manager.stage
        # Both gates must pass: enough episodes SINCE THE LAST TRANSITION (so a stage can't be
        # cleared on evidence gathered against the previous, easier stage) and a full-enough
        # window to measure a rate over at all.
        have_evidence = (
            self.episodes_at_stage >= self.cfg.min_episodes_at_stage
            and len(self._window) >= self.cfg.min_episodes_at_stage
        )
        win_rate = self.win_rate

        if have_evidence and self.cfg.demote_win_rate is not None and win_rate < self.cfg.demote_win_rate:
            if self.manager.demote():
                self._transitioned("demoted", win_rate)
                return

        if stage.advance_win_rate is None:
            return  # terminal stage

        if have_evidence and win_rate >= stage.advance_win_rate:
            if self.manager.advance():
                self._transitioned("advanced", win_rate)
            return

        budget = self.cfg.max_timesteps_at_stage
        if budget is not None and (self.num_timesteps - self._stage_start_timesteps) >= budget:
            if self.manager.advance():
                self._transitioned("force-advanced (stage timestep budget exhausted)", win_rate)

    def _transitioned(self, how: str, win_rate: float) -> None:
        entry = {
            "timesteps": int(self.num_timesteps),
            "stage_index": self.manager.stage_index,
            "stage_name": self.manager.stage.name,
            "how": how,
            "win_rate_at_transition": round(win_rate, 4),
            "episodes_at_previous_stage": self.episodes_at_stage,
            "total_episodes": self.total_episodes,
        }
        self.history.append(entry)
        # Both the window and the per-stage episode count reset: outcomes scored against the
        # previous difficulty say nothing about the new one, and carrying them over would let a
        # strong easy-stage window instantly clear the next stage too. The same holds for the
        # episodes still in flight, which is what marking every env stale is for.
        self._window.clear()
        self._ranks.clear()
        self.episodes_at_stage = 0
        if self._stale is not None:
            self._stale[:] = True
        self._stage_start_timesteps = self.num_timesteps
        self._write_state()
        if self.verbose:
            print(
                f"\n{'=' * 78}\n"
                f"  CURRICULUM {how.upper()} -> stage {self.manager.stage_index + 1}/"
                f"{self.manager.n_stages}: {self.manager.stage.name!r}\n"
                f"  at {self.num_timesteps:,} timesteps | win rate {win_rate:.1%} | "
                f"mixture {self._weights_str()}\n"
                f"{'=' * 78}\n"
            )

    # ---- logging -----------------------------------------------------------------------

    def _log(self) -> None:
        m = self.manager
        rec = self.logger.record
        rec("curriculum/stage_index", m.stage_index)
        rec("curriculum/stage_name", m.stage.name)
        rec("curriculum/stage_progress", m.stage_index / max(1, m.n_stages - 1))
        rec("curriculum/win_rate", self.win_rate)
        rec("curriculum/mean_rank", self.mean_rank)
        rec("curriculum/episodes_at_stage", self.episodes_at_stage)
        rec("curriculum/episodes_total", self.total_episodes)
        rec("curriculum/window_filled", len(self._window))
        rec("curriculum/envs_on_previous_stage",
            0 if self._stale is None else int(self._stale.sum()))
        target = m.stage.advance_win_rate
        rec("curriculum/advance_threshold", -1.0 if target is None else target)

        for name, weight in m.tier_weights().items():
            rec(f"curriculum/tier_weight_{name}", weight)
        for name, frac in m.tier_spawn_fractions().items():
            rec(f"curriculum/tier_spawned_{name}", frac)

        if self.reward_fn is not None:
            for name, value in self.reward_fn.term_means().items():
                rec(f"reward_terms/{name}", value)

    def _weights_str(self) -> str:
        return ", ".join(f"{n} {w:.0%}" for n, w in self.manager.tier_weights().items() if w > 0)

    def _banner(self, title: str) -> None:
        if not self.verbose:
            return
        m = self.manager
        print(f"[{title}] stage {m.stage_index + 1}/{m.n_stages} {m.stage.name!r} "
              f"({self._weights_str()}) | advance at "
              f"{'n/a (terminal)' if m.stage.advance_win_rate is None else f'{m.stage.advance_win_rate:.0%} win rate'}")

    def _write_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **self.manager.state_dict(),
            "n_stages": self.manager.n_stages,
            "timesteps": int(self.num_timesteps),
            "total_episodes": self.total_episodes,
            "win_rate": round(self.win_rate, 4),
            "history": self.history,
        }
        self.state_path.write_text(json.dumps(payload, indent=2))


class TierEvalCallback(BaseCallback):
    """Runs `TierEvaluator` every `eval.every_timesteps` and logs the result two ways:

    1. Named scalars on the run's logger (`eval/win_rate_easy`, ...), so they also land in
       `logs/progress.csv` and survive without TensorBoard.
    2. One event-file writer per tier in `logs/eval_<tier>/`, all writing the SAME tag
       (`eval/win_rate`). TensorBoard treats each subdirectory as a run and overlays identical
       tags, which draws one chart with a line per tier; separate scalar names cannot.

    Also saves `best_model.zip` whenever the mean win rate improves.

    **Training maps vs holdout maps.** `evaluator` scores the maps the run trains on, so every key
    above is a TRAINING-map number. `holdout_evaluator`, when given, is its twin on
    `eval.holdout_maps` (never in the training rotation), scored at the same moments:
      - `eval/holdout_win_rate_<tier>`   per tier, next to `eval/win_rate_<tier>`
      - `eval/holdout_win_rate`          mean over tiers, the mirror of `eval/win_rate_mean`
      - `eval/holdout_gap`               `win_rate_mean - holdout_win_rate`. The map sets are not
                                         equally hard, so read its TREND against the `at_start`
                                         row, not its sign: a widening gap is map memorization.
      - tag `eval/holdout_win_rate` in each `logs/eval_<tier>/`, the per-tier overlay for the
        holdout maps. The run logger's mean shares that tag, so the chart has one extra line.
    `best_model.zip` is selected on the training-map mean ONLY: picking it by holdout score would
    fit the selection to the holdout maps.
    """

    def __init__(
        self, evaluator, every_timesteps: int, log_dir, at_start: bool = True,
        best_model_path=None, verbose: int = 1, holdout_evaluator=None,
    ) -> None:
        super().__init__(verbose)
        self.evaluator = evaluator
        self.holdout_evaluator = holdout_evaluator
        self.every_timesteps = every_timesteps
        self.log_dir = Path(log_dir)
        self.at_start = at_start
        self.best_model_path = Path(best_model_path) if best_model_path is not None else None
        self.best_mean_win_rate = -1.0
        self.history: list[dict] = []
        self._writers: dict = {}
        self._next_eval_at = 0

    # ---- SB3 hooks ---------------------------------------------------------------------

    def _on_training_start(self) -> None:
        self._next_eval_at = self.num_timesteps if self.at_start else self.num_timesteps + self.every_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval_at:
            # Advance from the CURRENT step, not by adding to the old target: rollouts advance
            # num_timesteps by n_envs at a time, so a fixed grid would fire twice in a row
            # whenever a rollout overshoots by more than one interval.
            self._next_eval_at = self.num_timesteps + self.every_timesteps
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        self._evaluate()   # a final measurement at the end of the run
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    # ---- the evaluation ----------------------------------------------------------------

    def _evaluate(self) -> None:
        self._sync_normalization(self.evaluator)
        results = self.evaluator.evaluate(self.model)
        mean_win_rate = evaluation.mean_win_rate(results)

        for name, r in results.items():
            for metric in evaluation.METRICS:
                self.logger.record(f"eval/{metric}_{name}", r[metric])
            self._tier_writer(name).add_scalar("eval/win_rate", r["win_rate"], self.num_timesteps)
            self._tier_writer(name).add_scalar("eval/mean_rank", r["mean_rank"], self.num_timesteps)
            self._tier_writer(name).add_scalar("eval/mean_reward", r["mean_reward"], self.num_timesteps)
            self._tier_writer(name).add_scalar("eval/gadgets_used", r["gadgets_used"],
                                               self.num_timesteps)
        self.logger.record("eval/win_rate_mean", mean_win_rate)

        entry = {"timesteps": int(self.num_timesteps),
                 "mean_win_rate": round(mean_win_rate, 4),
                 "tiers": {n: round(r["win_rate"], 4) for n, r in results.items()}}

        # Best-model selection reads the TRAINING-map mean and nothing below this block: see the
        # class docstring for why the holdout score must never pick the checkpoint.
        if mean_win_rate > self.best_mean_win_rate:
            self.best_mean_win_rate = mean_win_rate
            self._save_best()
        self.logger.record("eval/best_win_rate_mean", self.best_mean_win_rate)

        if self.verbose:
            print(f"\n[eval @ {self.num_timesteps:,} steps]  "
                  f"{evaluation.summary_line(results)}   (mean {mean_win_rate:.1%})")
            print(evaluation.format_table(results) + "\n")

        if self.holdout_evaluator is not None:
            entry.update(self._evaluate_holdout(mean_win_rate))
        self.history.append(entry)

    def _evaluate_holdout(self, training_mean_win_rate: float) -> dict:
        """Scores the holdout maps and logs them; returns the two history fields."""
        self._sync_normalization(self.holdout_evaluator)
        results = self.holdout_evaluator.evaluate(self.model)
        holdout_mean = evaluation.mean_win_rate(results)

        for name, r in results.items():
            self.logger.record(f"eval/holdout_win_rate_{name}", r["win_rate"])
            self._tier_writer(name).add_scalar("eval/holdout_win_rate", r["win_rate"], self.num_timesteps)
        self.logger.record("eval/holdout_win_rate", holdout_mean)
        self.logger.record("eval/holdout_gap", training_mean_win_rate - holdout_mean)

        if self.verbose:
            maps = ", ".join(self.holdout_evaluator.map_names)
            print(f"[eval holdout: {maps}]  {evaluation.summary_line(results)}   "
                  f"(mean {holdout_mean:.1%}, training maps {training_mean_win_rate:.1%})\n")
        return {"holdout_mean_win_rate": round(holdout_mean, 4),
                "holdout_tiers": {n: round(r["win_rate"], 4) for n, r in results.items()}}

    def _sync_normalization(self, evaluator) -> None:
        """Copies the training env's observation-normalization statistics onto `evaluator`'s env.

        Only matters when `normalize.obs` is on, but then skipping it feeds the policy
        differently-scaled observations and the eval score looks like a training failure. Reward
        statistics are deliberately NOT synced: eval reports raw returns.

        Copied directly, NOT through SB3's `sync_envs_normalization`, which asserts both wrapper
        stacks have the same depth: training is `VecNormalize(VecMonitor(env))`
        (builder.build_env), an evaluator is `VecNormalize(env)`."""
        train_env = self.model.get_vec_normalize_env()
        if train_env is None or not getattr(train_env, "norm_obs", False):
            return
        from stable_baselines3.common.vec_env import VecNormalize
        if not isinstance(evaluator.venv, VecNormalize):
            evaluator.venv = VecNormalize(
                evaluator.venv, training=False, norm_obs=True, norm_reward=False,
                clip_obs=train_env.clip_obs, norm_obs_keys=list(train_env.obs_rms.keys()),
            )
        evaluator.venv.obs_rms = deepcopy(train_env.obs_rms)

    def _tier_writer(self, tier: str):
        """One `SummaryWriter` per tier, each in its own subdirectory -- see the class docstring
        for why the overlay needs separate directories rather than separate tags."""
        if tier not in self._writers:
            from torch.utils.tensorboard import SummaryWriter
            self._writers[tier] = SummaryWriter(str(self.log_dir / f"eval_{tier}"))
        return self._writers[tier]

    def _save_best(self) -> None:
        if self.best_model_path is None:
            return
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.best_model_path)
        venv = self.model.get_vec_normalize_env()
        if venv is not None:
            venv.save(str(self.best_model_path.with_name("best_vecnormalize.pkl")))
        if self.verbose:
            print(f"[eval] new best mean win rate {self.best_mean_win_rate:.1%} "
                  f"-> {self.best_model_path}")


class VecNormalizeCheckpoint(BaseCallback):
    """Saves `VecNormalize`'s running statistics next to each model checkpoint, on the same
    cadence: SB3's own `save_vecnormalize` option no-ops silently if the wrapper isn't found, and a
    normalized policy is meaningless without its statistics.

    The names sort lexicographically (`_final` last, `9..._steps` after `45..._steps`) and
    `scripts/train.py --resume` loads the last-sorted `*vecnormalize*.pkl` beside the zip, so
    resume from a folder inside the run holding only the matching pair, with
    `--set run.total_timesteps=<remaining>`.
    """

    def __init__(self, save_freq: int, save_path, name: str = "vecnormalize", verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name = name

    def _on_step(self) -> bool:
        if self.save_freq <= 0 or self.n_calls % self.save_freq != 0:
            return True
        self._save(f"{self.name}_{self.num_timesteps}_steps.pkl")
        return True

    def _on_training_end(self) -> None:
        self._save(f"{self.name}_final.pkl")

    def _save(self, filename: str) -> None:
        venv = self.model.get_vec_normalize_env()
        if venv is None:
            return
        self.save_path.mkdir(parents=True, exist_ok=True)
        venv.save(str(self.save_path / filename))
        if self.verbose:
            print(f"[checkpoint] VecNormalize stats -> {self.save_path / filename}")
