"""Training callbacks: curriculum advancement, stationary eval, and continuous training-side
logging -- the things that make a run's progress legible.

`CurriculumCallback` is the host-side half of the curriculum -- `training/curriculum.py` owns
"apply stage N to the sim", this owns "decide when N should become N+1". The split exists
because the decision needs episode OUTCOMES, which only exist as Python objects after
`BrawlSB3VecEnv` has crossed the device->host boundary, while the application has to stay
device-resident inside `reset_envs`.

**The win signal.** `BrawlSB3VecEnv._build_infos` attaches `info["outcome"] = {"rank", "won"}`
(plus `info["episode_stats"]` -- kills/damage/cubes/shots, see `TrainingMonitorCallback`) to every
env that finished this tick (requires `info_mode` `"episode"` or `"full"`; `TrainConfig`
validation enforces that for the curriculum). `rank` is 0-indexed off
`info["final_info"]["hero_rank"]`, so `won` means the hero was genuinely the last one standing --
a timeout that catches the hero alive but not alone is not a win, which is the behavior you want:
otherwise the agent can advance the curriculum by learning to hide.

**What gets logged, and why so much of it.** Every scalar below is prefixed `curriculum/`, so a
TensorBoard run shows the whole difficulty trajectory next to the reward curve. In particular
`curriculum/stage_index` is the line to overlay on `rollout/ep_rew_mean`: a shaped reward's
absolute level is only comparable WITHIN a stage, and a reward drop right after a stage bump is
expected progress, not a regression. `curriculum/tier_spawned_*` reports what the sim actually
rolled (as opposed to `tier_weight_*`, what the config asked for) -- they should agree to within
sampling noise, and if they don't, the mixture is not doing what you think it is.

**`curriculum/win_rate` is NOT a general-purpose training metric** -- it's `CurriculumCallback`'s
own decision variable, and it is DELIBERATELY reset to empty on every stage transition (a strong
easy-stage window must not carry over and instantly clear the next, harder stage). That makes it
unsuitable as "how is training going" at a glance: it drops to 0%/unfilled right after every
advance, which reads like a regression but isn't one. `TrainingMonitorCallback` exists
specifically to answer that different question -- a continuous, never-reset view under `train/`,
present even when the curriculum is disabled entirely.
"""
import json
from collections import deque
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from . import evaluation
from .config import CurriculumConfig
from .curriculum import CurriculumManager


class TrainingMonitorCallback(BaseCallback):
    """Continuous, never-reset visibility into the TRAINING distribution itself -- everything SB3
    doesn't already give you for free.

    SB3's own logger already reports `train/*` (losses, `learning_rate`, `clip_range`,
    `approx_kl`, `explained_variance`), `rollout/*` (`ep_rew_mean`, `ep_len_mean` -- via
    `VecMonitor`), and `time/*` (`fps`, `total_timesteps`, `iterations`). This callback adds the
    one thing missing from that picture: **a win rate and a "how is the hero actually playing"
    breakdown, measured against whatever the hero is training against right now**, independent of
    `CurriculumCallback`'s own `curriculum/win_rate` (which is a DECISION variable, cleared to
    empty on every stage transition -- see module docstring) and independent of stationary eval
    (which is deliberately a DIFFERENT, fixed population, precisely so it can't answer "how is
    training going right now").

    Logged under `train/`, every rollout, over a rolling window of the last `window_episodes`
    finished TRAINING episodes (defaults to `curriculum.window_episodes`, the same "how many
    recent episodes" knob the curriculum already exposes -- reused rather than duplicated as a
    second config field for the same concept):

      train/win_rate            fraction of the window the hero was last alive
      train/mean_rank           mean 0-indexed placement (0 = won)
      train/episodes_total      cumulative count since training started, never reset
      train/episodes_window     how many of the last `window_episodes` are filled in right now
      train/kills_mean          mean hero kills per finished episode
      train/damage_dealt_mean   mean hero damage dealt per finished episode
      train/damage_taken_mean   mean hero damage taken per finished episode
      train/cubes_mean          mean power cubes collected per finished episode
      train/shots_fired_mean    mean shots fired per finished episode

    Reads `info["outcome"]`/`info["episode_stats"]` (`wrappers/sb3_vecenv.py`'s own per-done-env
    fields) -- both require `run.info_mode` `"episode"` or `"full"`. Under `"minimal"` this
    callback still runs but every metric above stays at its empty-window default (0.0 win rate,
    NaN mean_rank, zero episode counts) since neither info key ever appears; `scripts/train.py`
    prints a note rather than silently shipping a blank chart.
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

    `state_path`, when given, receives a small JSON file rewritten on every stage change (and at
    the end of training) holding the current stage plus the full transition history -- so a
    resumed run picks up at the right difficulty, and so a finished run can say what it actually
    trained against without re-reading the TensorBoard log.
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

    # ---- SB3 hooks ---------------------------------------------------------------------

    def _on_training_start(self) -> None:
        self._stage_start_timesteps = self.num_timesteps
        self._banner("curriculum start")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            outcome = info.get("outcome")
            if outcome is None:
                continue
            self._window.append(bool(outcome["won"]))
            self._ranks.append(int(outcome["rank"]))
            self.episodes_at_stage += 1
            self.total_episodes += 1
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
        # strong easy-stage window instantly clear the next stage too.
        self._window.clear()
        self._ranks.clear()
        self.episodes_at_stage = 0
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
    """Runs `TierEvaluator` every `eval.every_timesteps` and logs the result two ways.

    **1. Named scalars on the run's own logger** -- `eval/win_rate_easy`, `eval/win_rate_hard`,
    ... These appear as separate TensorBoard charts AND as columns in `logs/progress.csv`, so the
    numbers survive without TensorBoard installed.

    **2. A per-tier event-file writer** in `logs/eval_<tier>/`, each writing the SAME tag
    (`eval/win_rate`). TensorBoard treats each subdirectory as its own "run" and overlays
    identical tags from different runs on one chart -- so this produces a single `eval/win_rate`
    plot with one colored line per difficulty, labelled `eval_easy`, `eval_hard`, and so on.
    That comparison (is the easy-vs-hard gap closing?) is the thing you actually want to look at,
    and separate scalar names alone cannot draw it. The two paths are complementary, not
    redundant: (1) is machine-readable and always present, (2) is the readable picture.

    Also tracks the best mean win rate seen and saves `best_model.zip` when it improves --
    stationary eval is the only signal in the run that "best" can honestly be defined against.
    """

    def __init__(
        self, evaluator, every_timesteps: int, log_dir, at_start: bool = True,
        best_model_path=None, verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.evaluator = evaluator
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
        self._sync_normalization()
        results = self.evaluator.evaluate(self.model)

        win_rates = [r["win_rate"] for r in results.values()]
        mean_win_rate = float(sum(win_rates) / len(win_rates)) if win_rates else 0.0

        for name, r in results.items():
            for metric in evaluation.METRICS:
                self.logger.record(f"eval/{metric}_{name}", r[metric])
            self._tier_writer(name).add_scalar("eval/win_rate", r["win_rate"], self.num_timesteps)
            self._tier_writer(name).add_scalar("eval/mean_rank", r["mean_rank"], self.num_timesteps)
            self._tier_writer(name).add_scalar("eval/mean_reward", r["mean_reward"], self.num_timesteps)
        self.logger.record("eval/win_rate_mean", mean_win_rate)

        self.history.append({"timesteps": int(self.num_timesteps),
                             "mean_win_rate": round(mean_win_rate, 4),
                             "tiers": {n: round(r["win_rate"], 4) for n, r in results.items()}})

        if mean_win_rate > self.best_mean_win_rate:
            self.best_mean_win_rate = mean_win_rate
            self._save_best()
        self.logger.record("eval/best_win_rate_mean", self.best_mean_win_rate)

        if self.verbose:
            print(f"\n[eval @ {self.num_timesteps:,} steps]  "
                  f"{evaluation.summary_line(results)}   (mean {mean_win_rate:.1%})")
            print(evaluation.format_table(results) + "\n")

    def _sync_normalization(self) -> None:
        """Copies the training env's observation-normalization statistics onto the eval env.

        Only matters when `normalize.obs` is on -- but when it is, skipping this silently feeds
        the policy differently-scaled observations than it trained on, and the eval score becomes
        meaningless in a way that looks like a training failure. Reward statistics are
        deliberately NOT synced: eval reports raw returns."""
        train_env = self.model.get_vec_normalize_env()
        if train_env is None or not getattr(train_env, "norm_obs", False):
            return
        from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization
        if not isinstance(self.evaluator.venv, VecNormalize):
            self.evaluator.venv = VecNormalize(
                self.evaluator.venv, training=False, norm_obs=True, norm_reward=False,
                clip_obs=train_env.clip_obs, norm_obs_keys=list(train_env.obs_rms.keys()),
            )
        sync_envs_normalization(train_env, self.evaluator.venv)

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
    """Saves `VecNormalize`'s running statistics next to each model checkpoint.

    SB3's own `CheckpointCallback` has a `save_vecnormalize` option, but it only fires on its own
    schedule and silently no-ops if the wrapper isn't found -- which is exactly the failure that
    surfaces months later as "my loaded model performs nothing like it did in training", because
    a `VecNormalize`-trained policy is meaningless without the observation/reward statistics it
    was normalized against. This saves them unconditionally on the same cadence and says so.
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
