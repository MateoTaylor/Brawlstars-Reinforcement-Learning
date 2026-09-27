"""Where do the loot pulls put bots that should be doing something else? (design doc §9 entry 22)

Wraps `bots.personality.movement` and the two loot contributions in `bots.policy`, rolls a
checkpoint through `TierEvaluator` (deterministic as deployed, CPU, under the NEXT run's gas by
default) and counts, per decision of a living bot:

  * campers: share of decisions in a bush; share of TO_BUSH decisions with a crate pull; share
    "parked" (TO_BUSH, a crate within 1.5 tiles, not in a bush) and the longest park;
  * engaged mobile bots (enemy target within 8 tiles): share with a cube pull, the cube's mean
    distance, how many of those pulls point away from the enemy or reach past 4 and 8 tiles;
  * CLOSE / HOLD_RANGE and RETREAT decisions: share with a cube pull and, for RETREAT, with a
    crate pull, and how many of those move toward the enemy;
  * any bot standing on an unbroken crate with the pull live and no fire target;
  * per personality: share of decisions with a crate pull / a cube pull / in a bush.

The pull weights are read where `bots.policy` produces them, BEFORE `personality.movement`
zeroes them for RETREAT and for a crate under a live cube pull, so a "RETREAT cube-pulled" share
here is the share of retreat decisions where the pull WOULD be live; the move direction is the
final one, after that zeroing. The pre-2026-09-25 pulls are code, not config, so they cannot be
selected here; their numbers are in the design doc entry.

Usage (from the repo root, the venv's python, CPU only -- the GPU is the trainer's):

    .venv/Scripts/python.exe scripts/probes/bot_pull_measure.py
    .venv/Scripts/python.exe scripts/probes/bot_pull_measure.py --tier hard --episodes 64

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
from brawl_sim.bots import personality, perception, policy as bot_policy   # noqa: E402
from brawl_sim.constants import Person                                      # noqa: E402
from brawl_sim.training.config import deep_merge, load_train_config         # noqa: E402
from brawl_sim.training.evaluation import TierEvaluator                     # noqa: E402
from sb3_contrib import MaskablePPO                                         # noqa: E402

_DEFAULT_RUN = "runs/mortis_deploy5-20260924-090436"
_NEXT_GAS = {"zone": {"step_seconds": 5.0}, "sim": {"max_episode_steps": 3700}}
Mode = personality.Mode
CAP = {}

_orig_movement = personality.movement
_orig_box = bot_policy.box_contribution
_orig_cube = bot_policy.cube_contribution


def _movement_wrap(state, tgt, bank, params, cfg, gen):
    move_dir, mode = _orig_movement(state, tgt, bank, params, cfg, gen)
    CAP["tgt"] = tgt
    CAP["move"] = move_dir.clone()
    CAP["mode"] = mode.clone()
    CAP["in_bush"] = perception.in_bush(state, bank).clone()
    return move_dir, mode


def _box_wrap(state, bank, cfg):
    d, w = _orig_box(state, bank, cfg)
    CAP["box_w"] = w.clone()
    return d, w


def _cube_wrap(state, bank, cfg):
    d, w = _orig_cube(state, bank, cfg)
    CAP["cube_w"] = w.clone()
    # Distance at the tick itself: after the step the pickup may already be collected.
    _, dist = perception.nearest_alive(state.pku_pos, state.pku_alive, state.ent_pos)
    CAP["cube_dist"] = dist.clone()
    return d, w


personality.movement = _movement_wrap
bot_policy.box_contribution = _box_wrap
bot_policy.cube_contribution = _cube_wrap


class Tally:
    def __init__(self):
        self.c = {}

    def add(self, key, mask, value=None):
        n = int(mask.sum())
        s = float(value[mask].sum()) if value is not None and n else 0.0
        a, b = self.c.get(key, (0, 0.0))
        self.c[key] = (a + n, b + s)

    def n(self, key):
        return self.c.get(key, (0, 0.0))[0]

    def mean(self, key):
        a, b = self.c.get(key, (0, 0.0))
        return b / max(a, 1)

    def frac(self, key, base):
        return self.n(key) / max(self.n(base), 1)


def measure(base, model, tier, episodes, gas):
    t0 = time.time()
    overrides = deep_merge(base.run.env_overrides or {}, gas)
    tcfg = replace(base, run=replace(base.run, env_overrides=overrides),
                   eval=replace(base.eval, episodes_per_tier=episodes, tiers=(tier,)))
    ev = TierEvaluator(tcfg, tiers=(tier,), device="cpu")
    venv, n = ev.venv, ev.n_envs
    ev.sim.gen.manual_seed(ev.seed)
    obs = venv.reset()
    st = ev.sim.state
    E = st.ent_pos.shape[1]
    recorded = np.zeros(n, bool)
    won = np.zeros(n, bool)
    T = Tally()
    parked_run = np.zeros((n, E), np.int64)
    parked_max = np.zeros((n, E), np.int64)
    prev_pos = st.ent_pos.numpy().copy()
    prev_mode = None

    for _ in range(ev.max_steps):
        masks = venv.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        obs, _, dones, infos = venv.step(action)
        for i in np.nonzero(dones & ~recorded)[0]:
            recorded[i] = True
            won[i] = infos[i]["outcome"]["won"]
        if "tgt" not in CAP:
            continue
        # Captured at the last sim tick of this step; state is post-step (same tick, after moves).
        tgt = CAP["tgt"]
        mode = CAP["mode"].numpy()
        move = CAP["move"].numpy()
        in_bush = CAP["in_bush"].numpy()
        box_w = CAP["box_w"].numpy()
        cube_w = CAP["cube_w"].numpy()
        alive = st.ent_alive.numpy().copy()
        person = st.ent_person.numpy()
        pos = st.ent_pos.numpy()
        has_enemy = tgt.has_enemy.numpy()
        enemy_pos = tgt.enemy_pos.numpy()
        seek = enemy_pos - pos
        seek_n = seek / np.maximum(np.linalg.norm(seek, axis=-1, keepdims=True), 1e-6)
        enemy_dist = np.linalg.norm(seek, axis=-1)
        toward_enemy = (move * seek_n).sum(-1)
        bpos = st.box_pos.numpy()
        balive = st.box_alive.numpy()
        dd = np.linalg.norm(pos[:, :, None, :] - bpos[:, None, :, :], axis=-1)
        crate_dist = np.where(balive[:, None, :], dd, np.inf).min(-1)
        cube_dist = CAP["cube_dist"].numpy()

        live = alive & ~recorded[:, None] & ~dones[:, None]
        live[:, 0] = False
        camper = live & (person == int(Person.CAMPER))
        mobile = live & (mode != int(Mode.HOLD_STILL))
        engaged = mobile & has_enemy & (enemy_dist <= 8.0)
        fighting = live & np.isin(mode, [int(Mode.CLOSE), int(Mode.HOLD_RANGE)])
        retreat = live & (mode == int(Mode.RETREAT))
        # A bot in HOLD_STILL at this decision and the last one, displaced over the agent step:
        # before the intent-length throttle the smoothed intent decayed while the bot walked on
        # at full speed along its old heading (0.47 of HOLD_STILL ticks moved at hard).
        if prev_mode is not None:
            held = live & (mode == int(Mode.HOLD_STILL)) & (prev_mode == int(Mode.HOLD_STILL))
            step_move = np.linalg.norm(pos - prev_pos, axis=-1)
            T.add("held", held)
            T.add("held_moved", held & (step_move > 0.25), step_move)
        prev_pos = pos.copy()
        prev_mode = mode

        T.add("bot", live)
        T.add("camper", camper)
        T.add("camper_in_bush", camper & in_bush)
        T.add("camper_tobush", camper & (mode == int(Mode.TO_BUSH)))
        T.add("camper_tobush_boxpull", camper & (mode == int(Mode.TO_BUSH)) & (box_w > 0))
        parked = camper & (mode == int(Mode.TO_BUSH)) & (crate_dist <= 1.5) & ~in_bush
        T.add("camper_parked", parked)
        parked_run = np.where(parked, parked_run + 1, 0)
        parked_max = np.maximum(parked_max, parked_run)
        T.add("engaged", engaged)
        T.add("engaged_cubepull", engaged & (cube_w > 0), cube_dist)
        T.add("engaged_cubepull_away", engaged & (cube_w > 0) & (toward_enemy < 0))
        T.add("engaged_cubepull_far4", engaged & (cube_w > 0) & (cube_dist > 4.0))
        T.add("engaged_cubepull_far8", engaged & (cube_w > 0) & (cube_dist > 8.0))
        T.add("engaged_cubepull_away_far4",
              engaged & (cube_w > 0) & (toward_enemy < 0) & (cube_dist > 4.0))
        T.add("engaged_boxpull", engaged & (box_w > 0))
        T.add("fighting", fighting)
        T.add("fighting_cubepull", fighting & (cube_w > 0), cube_dist)
        T.add("fighting_cubepull_away", fighting & (cube_w > 0) & (toward_enemy < 0))
        T.add("retreat", retreat)
        T.add("retreat_cubepull", retreat & (cube_w > 0), cube_dist)
        T.add("retreat_cubepull_toward", retreat & (cube_w > 0) & (toward_enemy > 0.5))
        T.add("retreat_toward", retreat & (toward_enemy > 0.5))
        T.add("retreat_boxpull", retreat & (box_w > 0), crate_dist)
        T.add("retreat_boxpull_toward", retreat & (box_w > 0) & (toward_enemy > 0.5))
        for p in Person:
            m = live & (person == int(p))
            T.add(f"{p.name}", m)
            T.add(f"{p.name}_boxpull", m & (box_w > 0))
            T.add(f"{p.name}_cubepull", m & (cube_w > 0))
            T.add(f"{p.name}_in_bush", m & in_bush)
        # a bot standing on a crate it is not breaking: crate within 1.5 tiles, pull on, no target
        T.add("boxpull_close_notarget",
              live & (box_w > 0) & (crate_dist <= 1.5) & ~tgt.has_target.numpy())
        if recorded.all():
            break
    ev.close()
    agent_dt = ev.sim.cfg.agent_dt
    print(f"\n=== tier {tier}  {int(recorded.sum())} eps  win {won[recorded].mean():.2f}  "
          f"({time.time() - t0:.0f} s wall)")
    print(f"  bot decisions {T.n('bot')}")
    print(f"  CAMPER: {T.n('camper')} decisions, in bush {T.frac('camper_in_bush', 'camper'):.3f}, "
          f"TO_BUSH {T.frac('camper_tobush', 'camper'):.3f}, of which crate-pulled "
          f"{T.frac('camper_tobush_boxpull', 'camper_tobush'):.3f}; parked at a crate "
          f"{T.frac('camper_parked', 'camper'):.3f} of camper time, longest park "
          f"{parked_max.max() * agent_dt:.0f} s")
    print(f"  ENGAGED mobile (enemy <= 8): {T.n('engaged')} decisions, cube-pulled "
          f"{T.frac('engaged_cubepull', 'engaged'):.3f} (cube {T.mean('engaged_cubepull'):.1f} tiles "
          f"away; moving away from the enemy {T.frac('engaged_cubepull_away', 'engaged_cubepull'):.2f}), "
          f"crate-pulled {T.frac('engaged_boxpull', 'engaged'):.3f}")
    print(f"    of the engaged cube pulls: cube > 4 tiles "
          f"{T.frac('engaged_cubepull_far4', 'engaged_cubepull'):.2f}, > 8 tiles "
          f"{T.frac('engaged_cubepull_far8', 'engaged_cubepull'):.2f}; away from the enemy AND > 4 tiles "
          f"{T.frac('engaged_cubepull_away_far4', 'engaged_cubepull'):.2f}")
    print(f"  CLOSE/HOLD_RANGE: {T.n('fighting')} decisions, cube-pulled "
          f"{T.frac('fighting_cubepull', 'fighting'):.3f} (cube {T.mean('fighting_cubepull'):.1f} tiles; "
          f"away {T.frac('fighting_cubepull_away', 'fighting_cubepull'):.2f})")
    print(f"  RETREAT: {T.n('retreat')} decisions, cube pull live {T.frac('retreat_cubepull', 'retreat'):.3f} "
          f"(cube {T.mean('retreat_cubepull'):.1f} tiles; moving toward the enemy "
          f"{T.frac('retreat_cubepull_toward', 'retreat_cubepull'):.2f}), crate pull live "
          f"{T.frac('retreat_boxpull', 'retreat'):.3f} (crate {T.mean('retreat_boxpull'):.1f} tiles; toward "
          f"the enemy {T.frac('retreat_boxpull_toward', 'retreat_boxpull'):.2f}); all retreat decisions "
          f"toward the enemy {T.frac('retreat_toward', 'retreat'):.3f}")
    print(f"  standing on an unbroken crate with the pull on and no target: "
          f"{T.frac('boxpull_close_notarget', 'bot'):.3f} of bot decisions")
    print(f"  HOLD_STILL bots displaced > 0.25 tile over an agent step: "
          f"{T.frac('held_moved', 'held'):.3f} of {T.n('held')} held decisions "
          f"(mean displacement when moved {T.mean('held_moved'):.2f} tiles)")
    print("  person   | decisions | crate pull | cube pull | in bush")
    for p in Person:
        k = p.name
        print(f"  {k:8s} | {T.n(k):9d} | {T.frac(k + '_boxpull', k):10.3f} | "
              f"{T.frac(k + '_cubepull', k):9.3f} | {T.frac(k + '_in_bush', k):7.3f}")


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
