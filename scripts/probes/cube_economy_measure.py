"""Cube economy in the sim for a checkpoint under the CURRENT sim (design doc §9 entry 21).

Rolls a checkpoint through `TierEvaluator` (the same scenarios in every cell), deterministic as
deployed, on the CPU, under the NEXT run's gas (185 s episode, 5.0 s per tile; `--gas trained`
keeps the run's own schedule), and reports per cell:

  * a time series over envs whose hero is still alive at t: the hero's cubes, the mean and the
    richest living bot's cubes, how often the richest holds 7 or more, crates broken so far,
    players alive, cubes lying on the ground;
  * at the hero's death: its own cubes, its killer's cubes, the richest living bot's cubes;
  * who collected the cubes: the hero's share of every cube picked up while it lived;
  * the win rate against the hero's cubes at 60 s;
  * how often a bot pulled toward a crate did not move (the wall-grinding measure that motivated
    bots/policy._walk_clear).

Cells: `repo` is the sim as configured (every crate spot filled since 2026-09-25); `old` is the
crate count deploy5 trained on (8 per match, 16 / 32 slot limits) with the bots as they are now.
The bot pulls themselves are code, not config, so the pre-2026-09-25 bots cannot be selected
here; their numbers are in the design doc entry.

Usage (from the repo root, the venv's python, CPU only -- the GPU is the trainer's):

    .venv/Scripts/python.exe scripts/probes/cube_economy_measure.py
    .venv/Scripts/python.exe scripts/probes/cube_economy_measure.py --tier hard --episodes 64
    .venv/Scripts/python.exe scripts/probes/cube_economy_measure.py --cells old repo

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
from brawl_sim.bots import policy as bot_policy                        # noqa: E402
from brawl_sim.constants import Person                                 # noqa: E402
from brawl_sim.training.config import deep_merge, load_train_config    # noqa: E402
from brawl_sim.training.evaluation import TierEvaluator                # noqa: E402
from sb3_contrib import MaskablePPO                                    # noqa: E402

_DEFAULT_RUN = "runs/mortis_deploy5-20260924-090436"
_NEXT_GAS = {"zone": {"step_seconds": 5.0}, "sim": {"max_episode_steps": 3700}}
_CELLS = {
    "repo": {},
    "old": {"boxes": {"n_boxes": 8}, "limits": {"max_boxes": 16, "max_pickups": 32}},
}
_BINS = (15, 30, 45, 60, 90, 120, 150)


def _run_cell(name, overrides, base, model, tier):
    t0 = time.time()
    tcfg = replace(base, run=replace(base.run, env_overrides=overrides))
    ev = TierEvaluator(tcfg, tiers=(tier,), device="cpu")
    venv, n = ev.venv, ev.n_envs
    ev.sim.gen.manual_seed(ev.seed)
    obs = venv.reset()
    st = ev.sim.state
    E = st.ent_cubes.shape[1]

    recorded = np.zeros(n, bool)
    rank = np.zeros(n, np.int64)
    won = np.zeros(n, bool)
    length = np.zeros(n, np.int64)
    crates0 = st.box_alive.sum(1).cpu().numpy().astype(np.int64)
    series = {b: [] for b in _BINS}
    seen_bin = np.zeros((n, len(_BINS)), bool)
    at60 = np.full(n, -1, np.int64)
    death = []
    hero_gain = np.zeros(n, np.int64)
    bot_gain = np.zeros(n, np.int64)
    pulled_n = stalled_n = 0

    for _ in range(ev.max_steps):
        # .copy(): on the CPU these are views of the live tensors, and the step (and its
        # autoreset) rewrites them in place.
        t = st.time.cpu().numpy().copy()
        cubes = st.ent_cubes.cpu().numpy().copy()
        alive = st.ent_alive.cpu().numpy().copy()
        last_hit = st.ent_last_hit_by[:, 0].cpu().numpy().copy()
        pos = st.ent_pos.cpu().numpy().copy()
        broken = crates0 - st.box_alive.sum(1).cpu().numpy()
        live = ~recorded & alive[:, 0]
        ground = np.where(st.pku_alive.cpu().numpy(), st.pku_cubes.cpu().numpy(), 0).sum(1)

        bpos = st.box_pos.cpu().numpy()
        balive = st.box_alive.cpu().numpy()
        dd = np.linalg.norm(pos[:, :, None, :] - bpos[:, None, :, :], axis=-1)
        dd = np.where(balive[:, None, :], dd, np.inf).min(-1)
        person = st.ent_person.cpu().numpy()
        mobile = (person != int(Person.CAMPER)) & (person != int(Person.TRAPPER))
        pulled = (st.ent_target.cpu().numpy() < 0) & (dd <= bot_policy._BOX_APPROACH_RADIUS) & mobile
        pulled[:, 0] = False

        bot_alive = alive[:, 1:]
        max_bot = np.where(bot_alive, cubes[:, 1:], -1).max(1)
        mean_bot = np.where(bot_alive, cubes[:, 1:], 0).sum(1) / np.maximum(bot_alive.sum(1), 1)
        for j, b in enumerate(_BINS):
            hit = live & ~seen_bin[:, j] & (t >= b)
            for i in np.nonzero(hit)[0]:
                series[b].append((cubes[i, 0], mean_bot[i], max_bot[i], broken[i], alive[i].sum(),
                                  ground[i]))
            seen_bin[:, j] |= hit
        at60 = np.where(live & (at60 < 0) & (t >= 60), cubes[:, 0], at60)

        masks = venv.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        obs, _, dones, infos = venv.step(action)
        for i in np.nonzero(dones & ~recorded)[0]:
            recorded[i] = True
            won[i] = infos[i]["outcome"]["won"]
            rank[i] = infos[i]["outcome"]["rank"]
            length[i] = infos[i]["episode"]["l"]
            if not won[i] and length[i] < ev.max_steps:
                k = int(last_hit[i])
                killer = int(cubes[i, k]) if 0 < k < E else -1
                death.append((cubes[i, 0], killer, max_bot[i], alive[i].sum()))
        moved = np.linalg.norm(st.ent_pos.cpu().numpy() - pos, axis=-1)
        pull_live = pulled & alive & st.ent_alive.cpu().numpy() & ~recorded[:, None] & ~dones[:, None]
        pulled_n += int(pull_live.sum())
        stalled_n += int((pull_live & (moved < 0.1) & (dd > 1.5)).sum())
        # Gains across this decision, for envs whose first episode is still running after it.
        d = st.ent_cubes.cpu().numpy() - cubes
        gained = np.where((d > 0) & alive & st.ent_alive.cpu().numpy(), d, 0)
        hero_gain += np.where(~recorded, gained[:, 0], 0)
        bot_gain += np.where(~recorded, gained[:, 1:].sum(1), 0)
        if recorded.all():
            break
    ev.close()

    print(f"\n=== {name}  tier {tier}  {int(recorded.sum())} eps  win {won[recorded].mean():.2f}  "
          f"rank {rank[recorded].mean():.2f}  median {np.median(length[recorded]) * ev.sim.cfg.agent_dt:.0f} s  "
          f"crates/match {crates0.mean():.1f}  ({time.time() - t0:.0f} s wall)")
    print("  t s | envs | hero cubes | bot mean | bot max | max>=7 | crates broken | alive | on ground")
    for b in _BINS:
        rows = np.array(series[b], float)
        if len(rows) == 0:
            continue
        print(f"  {b:4d} | {len(rows):4d} | {rows[:, 0].mean():10.2f} | {rows[:, 1].mean():8.2f} | "
              f"{rows[:, 2].mean():7.2f} | {(rows[:, 2] >= 7).mean():6.2f} | {rows[:, 3].mean():13.1f} | "
              f"{rows[:, 4].mean():5.1f} | {rows[:, 5].mean():9.1f}")
    dr = np.array(death, float)
    if len(dr):
        kd = dr[dr[:, 1] >= 0]
        k_mean = kd[:, 1].mean() if len(kd) else float("nan")
        k_rich = (kd[:, 1] >= 7).mean() if len(kd) else float("nan")
        print(f"  hero deaths {len(dr)}: hero cubes {dr[:, 0].mean():.2f}, killer cubes {k_mean:.2f} "
              f"(>=7: {k_rich:.2f}, n {len(kd)}), richest bot {dr[:, 2].mean():.2f}, alive {dr[:, 3].mean():.1f}")
    print(f"  mobile-bot decisions pulled to a crate {pulled_n}, of which stalled > 1.5 tiles out "
          f"{stalled_n / max(pulled_n, 1):.3f}")
    tot = hero_gain.sum() + bot_gain.sum()
    print(f"  cubes collected while the hero lived: hero {hero_gain.sum()}, bots {bot_gain.sum()}, "
          f"hero share {hero_gain.sum() / max(tot, 1):.3f}; hero per episode {hero_gain.mean():.2f}")
    ok = at60 >= 0
    for lo, hi in ((0, 1), (2, 3), (4, 6), (7, 99)):
        m = ok & (at60 >= lo) & (at60 <= hi)
        if m.any():
            print(f"  hero cubes at 60 s {lo}-{hi if hi < 99 else '+'}: {int(m.sum()):3d} eps, "
                  f"win {won[m].mean():.2f}, rank {rank[m].mean():.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=_DEFAULT_RUN)
    ap.add_argument("--checkpoint", default=None, help="defaults to <run>/best_model.zip")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--tier", default="elite")
    ap.add_argument("--cells", nargs="+", default=list(_CELLS), choices=list(_CELLS))
    ap.add_argument("--gas", choices=("next", "trained"), default="next")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    run = ROOT / args.run
    base = load_train_config(run / "train.yaml", check_holdout=False)
    base = replace(base, eval=replace(base.eval, episodes_per_tier=args.episodes, tiers=(args.tier,)))
    model = MaskablePPO.load(args.checkpoint or run / "best_model.zip", device="cpu")
    gas = _NEXT_GAS if args.gas == "next" else {}
    for name in args.cells:
        overrides = deep_merge(deep_merge(base.run.env_overrides or {}, gas), _CELLS[name])
        _run_cell(name, overrides, base, model, args.tier)


if __name__ == "__main__":
    main()
