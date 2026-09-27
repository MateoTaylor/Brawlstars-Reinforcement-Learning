"""How often does the gas take a kill the hero made, and how much blocked damage is the hero
charged for? (design doc §9 entry 22; both are 0 since 2026-09-25 and this keeps them there)

Wraps `core.combat.apply_damage` and rolls a checkpoint through `TierEvaluator` (deterministic
as deployed, CPU, under the NEXT run's gas by default):

  * gas-stolen kills: at the zone phase, an entity at 0 HP that is still `ent_alive` (deaths
    resolve phases later) with a real last hitter; before the fix the zone overwrote that last
    hitter with -1 and the finisher lost the credit;
  * i-frame charge: combat damage aimed at the hero while `ent_invuln_t > 0`. Before the fix it
    was zeroed for HP but the raw matrix still fed `damage_taken_tick`, priced at -1e-4 per HP;
    the dash grants no i-frames any more, so the counter must read 0.

Usage (from the repo root, the venv's python, CPU only -- the GPU is the trainer's):

    .venv/Scripts/python.exe scripts/probes/kill_credit_measure.py
    .venv/Scripts/python.exe scripts/probes/kill_credit_measure.py --tier hard --episodes 64

Reads the run's `train.yaml`, `best_model.zip` and the agent spec it names; writes nothing.
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brawl_sim.constants import DeathCause                                   # noqa: E402
from brawl_sim.core import combat                                            # noqa: E402
from brawl_sim.training.config import deep_merge, load_train_config          # noqa: E402
from brawl_sim.training.evaluation import TierEvaluator                      # noqa: E402
from sb3_contrib import MaskablePPO                                          # noqa: E402

_DEFAULT_RUN = "runs/mortis_deploy5-20260924-090436"
_NEXT_GAS = {"zone": {"step_seconds": 5.0}, "sim": {"max_episode_steps": 3700}}
C = {"stolen_any": 0, "stolen_hero": 0, "stolen_from_hero_victim": 0,
     "hero_blocked": 0.0, "hero_raw": 0.0, "hero_blocked_events": 0, "hero_hit_events": 0}
ACTIVE = None  # (N,) bool of envs still in their first episode

_orig = combat.apply_damage


def _wrap(state, dmg, cause, attacker, params, cfg):
    live = ACTIVE if ACTIVE is not None else torch.ones(dmg.shape[0], dtype=torch.bool)
    if int(cause) == int(DeathCause.ZONE):
        stolen = ((state.ent_hp <= 0) & state.ent_alive & (state.ent_last_hit_by >= 0)
                  & (dmg > 0) & live[:, None])
        C["stolen_any"] += int(stolen.sum())
        C["stolen_hero"] += int((stolen & (state.ent_last_hit_by == 0)).sum())
        C["stolen_from_hero_victim"] += int(stolen[:, 0].sum())
    else:
        blocked = (state.ent_invuln_t[:, 0] > 0) & live
        d = dmg[:, 0]
        C["hero_blocked"] += float(d[blocked].sum())
        C["hero_raw"] += float(d[live].sum())
        C["hero_blocked_events"] += int(((d > 0) & blocked).sum())
        C["hero_hit_events"] += int(((d > 0) & live).sum())
    return _orig(state, dmg, cause, attacker, params, cfg)


combat.apply_damage = _wrap


def measure(base, model, tier, episodes, gas):
    global ACTIVE
    overrides = deep_merge(base.run.env_overrides or {}, gas)
    tcfg = replace(base, run=replace(base.run, env_overrides=overrides),
                   eval=replace(base.eval, episodes_per_tier=episodes, tiers=(tier,)))
    ev = TierEvaluator(tcfg, tiers=(tier,), device="cpu")
    venv, n = ev.venv, ev.n_envs
    ev.sim.gen.manual_seed(ev.seed)
    obs = venv.reset()
    recorded = np.zeros(n, bool)
    ACTIVE = torch.ones(n, dtype=torch.bool)
    kills = np.zeros(n, np.int64)
    won = np.zeros(n, bool)
    dmg_taken = np.zeros(n)
    t0 = time.time()
    for _ in range(ev.max_steps):
        masks = venv.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        obs, _, dones, infos = venv.step(action)
        for i in np.nonzero(dones & ~recorded)[0]:
            recorded[i] = True
            kills[i] = infos[i]["episode_stats"]["kills"]
            dmg_taken[i] = infos[i]["episode_stats"]["damage_taken"]
            won[i] = infos[i]["outcome"]["won"]
        ACTIVE = torch.from_numpy(~recorded)
        if recorded.all():
            break
    ev.close()
    print(f"tier {tier}  {int(recorded.sum())} eps  win {won.mean():.2f}  ({time.time() - t0:.0f} s)")
    print(f"  hero kills credited: {kills.sum()} ({kills.mean():.2f}/ep); kills the gas took from the "
          f"hero: {C['stolen_hero']} ({C['stolen_hero'] / max(kills.sum() + C['stolen_hero'], 1):.3f} "
          f"of the hero's finishes); gas-stolen kills of any attacker: {C['stolen_any']}; hero deaths "
          f"mis-credited to the gas: {C['stolen_from_hero_victim']}")
    print(f"  hero raw incoming combat damage {C['hero_raw']:.0f} over {C['hero_hit_events']} hit-ticks; "
          f"of which inside dash i-frames {C['hero_blocked']:.0f} "
          f"({C['hero_blocked'] / max(C['hero_raw'], 1):.3f}) over {C['hero_blocked_events']} hit-ticks; "
          f"reward charged for blocked hits {C['hero_blocked'] * 1e-4 / n:.3f} per episode "
          f"(damage taken {np.mean(dmg_taken):.0f}/ep)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--run", default=_DEFAULT_RUN)
    ap.add_argument("--checkpoint", default=None, help="defaults to <run>/best_model.zip")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--tier", default="elite")
    ap.add_argument("--gas", choices=("next", "trained"), default="next")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    run = ROOT / args.run
    base = load_train_config(run / "train.yaml", check_holdout=False)
    model = MaskablePPO.load(args.checkpoint or run / "best_model.zip", device="cpu")
    measure(base, model, args.tier, args.episodes, _NEXT_GAS if args.gas == "next" else {})


if __name__ == "__main__":
    main()
