"""TierEvaluator: stationary, per-difficulty evaluation of a policy.

**Why this exists.** Everything else logged during training is measured against the TRAINING
distribution, which the curriculum deliberately makes harder over time -- so a flat
`rollout/ep_rew_mean` is genuinely ambiguous between "the policy stopped improving" and "the
policy improved exactly as fast as the bots did". This module fixes the difficulty and the
scenarios, so `eval/win_rate_hard` at 2M steps and at 20M steps mean the same thing.

**One env, every tier.** Rather than building one environment per difficulty, a single
`BrawlVecEnv` of `len(tiers) * episodes_per_tier` envs is partitioned into contiguous blocks and
`FixedTierHook` pins each block to one tier -- envs `[0:k)` easy, `[k:2k)` medium, and so on.
One batched rollout scores every difficulty at once, which is the whole reason this is cheap
enough to run periodically: the GPU cost of 4 tiers is the cost of 1, times a wider batch.

**Reproducibility is the point.** The eval env's generator is reseeded to the SAME value before
every evaluation, so each run replays an identical set of maps, spawn positions, and bot stats.
Differences between two evals are then attributable to the policy and nothing else. That seed
defaults to something unrelated to `run.seed`, so the evaluation scenarios are not a subset of
the ones training happened to see.

**Only the first episode per env slot counts.** Autoreset means a slot that finishes early
starts another episode and would otherwise be over-represented (short episodes = deaths, so
counting every finish would bias the win rate DOWN). Each slot contributes exactly one episode;
the rollout runs until every slot has finished once, bounded by `max_episode_steps`.

**Which maps (SIM_OVERHAUL M4).** By default the eval env is built from `run.env_config` +
`run.env_overrides`, i.e. the SAME resolved `world.maps` the training env draws from -- so every
`eval/*` number is a TRAINING-MAP number. With configs/train.yaml that is the fourteen maps its
`run.env_overrides.world.maps` lists, not configs/default.yaml's sixteen. `TierEvaluator(tcfg,
maps=...)` builds the holdout twin instead: identical in every other respect (tiers, episode
count, seed, reward, spec), but its map bank holds ONLY the named maps. `build_evaluators`
returns the pair a run needs. Training-map win rate minus holdout win rate is the map-overfitting
measurement; it is only meaningful because `validate_train_config` guarantees the two map sets
are disjoint.
"""
from dataclasses import replace

import numpy as np
import torch

from ..config import load_config
from ..core import obs_select
from ..env import BrawlVecEnv
from ..wrappers.sb3_vecenv import BrawlSB3VecEnv
from .config import TrainConfig, deep_merge, holdout_env_overrides
from .curriculum import FixedTierHook
from .reward import ShapedReward

# Per-tier metric names, in the order `summary_line` prints them.
METRICS = ("win_rate", "mean_rank", "mean_ep_length", "mean_reward", "gadgets_used")

# The attack column's gadget value: [no-fire, attack, super, gadget] (core/hero.decode_action).
_GADGET = 3


class TierEvaluator:
    """Builds the pinned eval env once and scores a policy against every tier on demand.

    Construction is deliberately eager (the env is allocated up front, not per evaluation): at
    `episodes_per_tier=32` over 4 tiers that's 128 envs held for the life of the run, which costs
    well under the VRAM headroom `benchmark.py` measured, and rebuilding it every 500k steps
    would re-pay map-bank construction and buffer allocation for nothing.

    `maps=None` evaluates on the run's own (training) maps. `maps=("a", "b")` replaces the env's
    `world.maps` with exactly those, drawn uniformly -- the holdout evaluator. Everything else is
    built from the same `tcfg`, so the two evaluators differ in the map bank and nothing else.
    """

    def __init__(self, tcfg: TrainConfig, tiers=None, device=None, verbose: int = 0,
                 maps=None) -> None:
        from .builder import build_spec, _resolve   # local: avoid a circular import at module load

        if maps:
            # Patched into `run.env_overrides` rather than into the EnvConfig alone, because the
            # env is built from TWO views of that dict (`load_config` and `build_spec` below) and
            # they must agree. `replace`, not `with_overrides`: this derived config is by
            # construction one validate_train_config refuses (its world.maps ARE the holdout).
            # `deep_merge` copies every dict it descends into, so the run's own config is not
            # written to.
            tcfg = replace(tcfg, run=replace(tcfg.run, env_overrides=deep_merge(
                tcfg.run.env_overrides or {}, holdout_env_overrides(maps))))
        # Stored AFTER the patch: `self.tcfg` is the config THIS evaluator's env was built from,
        # so it agrees with `self.env_cfg` / `self.map_names` and with `build_spec(self.tcfg)`.
        # For the holdout twin that is the derived config, not the run's.
        self.tcfg = tcfg
        self.verbose = verbose
        self.tier_names = tuple(tiers or tcfg.eval.tiers or tcfg.curriculum.tiers)
        if not self.tier_names:
            raise ValueError("TierEvaluator needs at least one tier to evaluate")
        self.episodes_per_tier = tcfg.eval.episodes_per_tier
        self.n_envs = len(self.tier_names) * self.episodes_per_tier
        self.uses_masks = tcfg.run.algo == "maskable_ppo"
        self.seed = tcfg.eval.seed

        device = device or tcfg.run.device
        env_cfg = load_config(_resolve(tcfg.run.env_config), overrides=tcfg.run.env_overrides or None)
        self.env_cfg = env_cfg
        self.map_names = env_cfg.map_names   # what this evaluator's bank holds, for the banner/log
        # DECISIONS, not sim ticks: `evaluate`'s loop drives `venv.step()`, and one of those
        # covers `action_repeat` ticks. Using max_episode_steps here would spin the eval rollout
        # action_repeat times longer than any episode can possibly last.
        self.max_steps = env_cfg.max_agent_steps
        agent_spec = obs_select.load_agent_spec(_resolve(tcfg.run.agent_obs), env_cfg)

        sim = BrawlVecEnv(
            env_cfg, n_envs=self.n_envs, device=device, seed=self.seed,
            reward_fn=ShapedReward(tcfg.reward, track_terms=False),
            # NO `randomization=`: per-env stat jitter is a training-time regularizer. Leaving it
            # on here would add variance to the one measurement that exists to be low-variance.
            spec=build_spec(tcfg), verbose=False,
        )
        # Block assignment: env i belongs to tier i // episodes_per_tier.
        assignment = torch.arange(self.n_envs, device=sim.device) // self.episodes_per_tier
        # The eval hook is pinned and shared, and is NEVER the training CurriculumManager -- that
        # is exactly what makes this measurement stationary.
        self.hook = FixedTierHook(
            {name: tcfg.curriculum.tiers[name] for name in self.tier_names}, sim.device, assignment
        )
        sim.params_hook = self.hook
        self.sim = sim
        self.venv = BrawlSB3VecEnv(sim, agent_spec, sim.reward_fn, info_mode="episode")
        self._tier_of_env = assignment.cpu().numpy()

    # ---- evaluation ------------------------------------------------------------------------

    def evaluate(self, model, deterministic: bool | None = None) -> dict:
        """Returns `{tier_name: {win_rate, mean_rank, mean_ep_length, mean_reward, gadgets_used,
        episodes}}`.

        `model` is any SB3 algorithm; action masks are passed only when the run's algo is
        `maskable_ppo` (plain `PPO.predict` has no `action_masks` parameter and would raise).

        `gadgets_used` is gadget throws per episode (SIM_OVERHAUL_STEPS.md Step I2). Nothing in the
        sim state counts throws, so this loop counts the ones it sends, by the sim's own rule:
        `core/hero.decode_action` throws on attack value 3 where the attack mask allows it, and
        `env._held` clears the fire column after a decision's first tick, so a decision throws at
        most once. The mask read here is the one the policy was given. The sim derives its own
        after ticking the timers, and the two differ only on the tick before a charge completes,
        where the policy's copy still says not ready. A `maskable_ppo` policy cannot press there,
        so for it the count is exact.
        """
        deterministic = self.tcfg.eval.deterministic if deterministic is None else deterministic
        self.sim.gen.manual_seed(self.seed)   # identical scenarios on every call -- see docstring
        obs = self.venv.reset()

        n = self.n_envs
        recorded = np.zeros(n, dtype=bool)
        won = np.zeros(n, dtype=bool)
        rank = np.zeros(n, dtype=np.int64)
        length = np.zeros(n, dtype=np.int64)
        ret = np.zeros(n, dtype=np.float64)
        gadgets = np.zeros(n, dtype=np.int64)
        gadget_col = self.env_cfg.action_nvec[0] + _GADGET   # the fused mask is [move, attack]

        for _ in range(self.max_steps):
            masks = self.venv.action_masks()
            kwargs = {"action_masks": masks} if self.uses_masks else {}
            action, _ = model.predict(obs, deterministic=deterministic, **kwargs)
            # Counted before this step's dones are recorded: a slot's last decision belongs to its
            # first episode, and every decision after it to the autoreset one, which must not count.
            gadgets += (action[:, 1] == _GADGET) & masks[:, gadget_col] & ~recorded
            obs, _, dones, infos = self.venv.step(action)
            for i in np.nonzero(dones & ~recorded)[0]:
                outcome, episode = infos[i]["outcome"], infos[i]["episode"]
                recorded[i] = True
                won[i] = outcome["won"]
                rank[i] = outcome["rank"]
                length[i] = episode["l"]
                ret[i] = episode["r"]
            if recorded.all():
                break

        return self._summarize(recorded, won, rank, length, ret, gadgets)

    def _summarize(self, recorded, won, rank, length, ret, gadgets) -> dict:
        out = {}
        for t, name in enumerate(self.tier_names):
            block = (self._tier_of_env == t) & recorded
            count = int(block.sum())
            if count == 0:
                # Only reachable if NO episode in this tier finished within max_episode_steps,
                # which the sim's own truncation makes impossible -- reported rather than
                # silently producing a nan-filled row.
                out[name] = {m: float("nan") for m in METRICS} | {"episodes": 0}
                continue
            out[name] = {
                "win_rate": float(won[block].mean()),
                "mean_rank": float(rank[block].mean()),
                "mean_ep_length": float(length[block].mean()),
                "mean_reward": float(ret[block].mean()),
                "gadgets_used": float(gadgets[block].mean()),
                "episodes": count,
            }
        return out

    def close(self) -> None:
        self.venv.close()


def build_evaluators(tcfg: TrainConfig, device=None, verbose: int = 0):
    """`(training_map_evaluator, holdout_evaluator_or_None)` for a run.

    The second is built only when `eval.holdout_maps` names something, and is the first one's
    twin on those maps (see the module docstring). It costs what the first costs: a second env of
    `len(tiers) * episodes_per_tier` slots held for the life of the run, and a second rollout per
    evaluation, so eval wall-clock roughly doubles."""
    evaluator = TierEvaluator(tcfg, device=device, verbose=verbose)
    holdout = None
    if tcfg.eval.has_holdout:
        holdout = TierEvaluator(tcfg, device=device, verbose=verbose, maps=tcfg.eval.holdout_maps)
    return evaluator, holdout


def mean_win_rate(results: dict) -> float:
    """Unweighted mean of the per-tier win rates -- `eval/win_rate_mean` and
    `eval/holdout_win_rate` are both this, over their own evaluator's results."""
    win_rates = [r["win_rate"] for r in results.values()]
    return float(sum(win_rates) / len(win_rates)) if win_rates else 0.0


def summary_line(results: dict) -> str:
    """One-line human summary, e.g. `easy 41% | medium 22% | hard 9% | elite 3%`."""
    return " | ".join(f"{name} {r['win_rate']:.0%}" for name, r in results.items())


def format_table(results: dict) -> str:
    header = (f"  {'tier':<10} {'win rate':>9} {'mean rank':>10} {'mean len':>9} {'mean rew':>9} "
              f"{'gadgets':>8} {'n':>5}")
    rows = [
        f"  {name:<10} {r['win_rate']:>8.1%} {r['mean_rank']:>10.2f} "
        f"{r['mean_ep_length']:>9.0f} {r['mean_reward']:>9.2f} {r['gadgets_used']:>8.2f} "
        f"{r['episodes']:>5d}"
        for name, r in results.items()
    ]
    return "\n".join([header, *rows])
