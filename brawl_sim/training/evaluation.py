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
"""
import numpy as np
import torch

from ..config import load_config
from ..core import obs_select
from ..env import BrawlVecEnv
from ..wrappers.sb3_vecenv import BrawlSB3VecEnv
from .config import TrainConfig
from .curriculum import FixedTierHook
from .reward import ShapedReward

# Per-tier metric names, in the order `summary_line` prints them.
METRICS = ("win_rate", "mean_rank", "mean_ep_length", "mean_reward")


class TierEvaluator:
    """Builds the pinned eval env once and scores a policy against every tier on demand.

    Construction is deliberately eager (the env is allocated up front, not per evaluation): at
    `episodes_per_tier=32` over 4 tiers that's 128 envs held for the life of the run, which costs
    well under the VRAM headroom `benchmark.py` measured, and rebuilding it every 500k steps
    would re-pay map-bank construction and buffer allocation for nothing.
    """

    def __init__(self, tcfg: TrainConfig, tiers=None, device=None, verbose: int = 0) -> None:
        from .builder import build_spec, _resolve   # local: avoid a circular import at module load

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
        """Returns `{tier_name: {win_rate, mean_rank, mean_ep_length, mean_reward}}`.

        `model` is any SB3 algorithm; action masks are passed only when the run's algo is
        `maskable_ppo` (plain `PPO.predict` has no `action_masks` parameter and would raise).
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

        for _ in range(self.max_steps):
            kwargs = {"action_masks": self.venv.action_masks()} if self.uses_masks else {}
            action, _ = model.predict(obs, deterministic=deterministic, **kwargs)
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

        return self._summarize(recorded, won, rank, length, ret)

    def _summarize(self, recorded, won, rank, length, ret) -> dict:
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
                "episodes": count,
            }
        return out

    def close(self) -> None:
        self.venv.close()


def summary_line(results: dict) -> str:
    """One-line human summary, e.g. `easy 41% | medium 22% | hard 9% | elite 3%`."""
    return " | ".join(f"{name} {r['win_rate']:.0%}" for name, r in results.items())


def format_table(results: dict) -> str:
    header = f"  {'tier':<10} {'win rate':>9} {'mean rank':>10} {'mean len':>9} {'mean rew':>9} {'n':>5}"
    rows = [
        f"  {name:<10} {r['win_rate']:>8.1%} {r['mean_rank']:>10.2f} "
        f"{r['mean_ep_length']:>9.0f} {r['mean_reward']:>9.2f} {r['episodes']:>5d}"
        for name, r in results.items()
    ]
    return "\n".join([header, *rows])
