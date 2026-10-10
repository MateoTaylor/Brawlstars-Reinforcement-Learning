"""The five sim issues seen in scripts/watch.py on 2026-10-06, measured in the sim itself
(SIM_ISSUES_PLAN.md).

watch.py builds the training env from the run's own train.yaml and changes three things: one
pinned tier, no zone jitter, argmax actions. This probe runs a checkpoint both ways, so a finding
that shows in both belongs to the sim, not the viewer:

  --mode watch   watch.py's own env (`build_watch_env`: n_envs=1, autoreset off), seeds
                 seed..seed+episodes-1, argmax -- seed 0 is the match `watch.py --tier X` shows
  --mode train   one batched env with the run's randomization.yaml, SAMPLED actions, one episode
                 per env

Per issue it reports:
  1  bot deaths by personality and cause; a zone death's last 3 s (stalled = pushing, not
     moving) and mode; stalls by mode, and how many happen under the zone's inward pull; KITE
     ticks holding range on a target it has no line of sight to
  2  the late game: seconds with <= 3 bots left, who they are, how they die
  3  cubes the hero spends >= 5 s within 2 tiles of without taking; whether the cube shares his
     tile out of reach (the grid draws both in one cell)
  4  crate HP one hero dash removes: scripted (no policy) and in the rollouts
  5  hero move-bin changes and reversals, idling, path efficiency, attacks taken when one is in
     reach (and argmax's split vote: no attack chosen though values 1 and 4 together outweigh
     it), the policy's own top-2 move probabilities, and bot hit rates after a reversal

Usage (repo root, the venv's python, CPU):

    .venv/Scripts/python.exe scripts/probes/sim_issues_probe.py \
        runs/mortis_ppo-20260930-182748/best_model.zip --tier expert --mode watch --episodes 8
    .venv/Scripts/python.exe scripts/probes/sim_issues_probe.py \
        runs/mortis_ppo-20260930-182748/best_model.zip --tier expert --mode train --envs 16

`--set KEY=VALUE` (repeatable) overrides the run's own train.yaml, as `scripts/train.py --set`
does, so a sim fix can be measured on an old checkpoint before anything trains on it:

    --set run.env_overrides.bots.nav=true --set run.env_overrides.boxes.dash_hits_once=true

Writes nothing.
"""
import argparse
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from brawl_sim.bots import perception as bot_perception                 # noqa: E402
from brawl_sim.bots import policy as bot_policy                         # noqa: E402
from brawl_sim.bots.personality import Mode                             # noqa: E402
from brawl_sim.config import load_config                                # noqa: E402
from brawl_sim.constants import DeathCause, Person                      # noqa: E402
from brawl_sim.core import obs_select, terrain                          # noqa: E402
from brawl_sim.env import BrawlVecEnv                                   # noqa: E402
from brawl_sim.training.builder import _resolve, build_spec             # noqa: E402
from brawl_sim.training.config import load_train_config, parse_overrides  # noqa: E402
from brawl_sim.training.curriculum import FixedTierHook                 # noqa: E402
from brawl_sim.training.reward import ShapedReward                      # noqa: E402
from brawl_sim.wrappers.sb3_vecenv import BrawlSB3VecEnv                # noqa: E402
from watch import (build_watch_env, find_train_config, load_model,  # noqa: E402
                   maybe_wrap_vecnormalize, watch_env_overrides)

_STALL_SPEED = 0.5      # tiles/s: slower than this while the smoothed intent is > 0.5 = stalled
_NEAR_CUBE = 2.0        # tiles
_NEAR_CUBE_SECONDS = 5.0
_REVERSAL_BINS = 6      # >= 135 degrees between consecutive move bins
_ENEMY_NEAR = 8.0       # tiles: "an enemy is in play" for the jitter contexts
_THREAT_TILES = 4.0     # an enemy projectile this close to the hero
_DASH_REACH = 2.67      # Mortis's uncharged dash, tiles
# A crate the dash can hit lies within dash_radius (0.70) of its path, and the hero's per-tick
# positions sit up to 0.89 tiles apart on a charged dash: 0.70 + 0.89 / 2, rounded up.
_DASH_PATH_TILES = 1.2
_PERSON = {int(p): p.name for p in Person}
_MODE = {int(m): m.name for m in Mode}

# Every bot intent and targeting bundle, as computed this tick. env._bot_phase calls
# `policy.all_bot_intents`, which calls `targeting` through its own module globals, so wrapping
# both module attributes sees each tick's values without touching the sim.
_CAPTURE = {}
_ORIG = {"targeting": bot_policy.targeting, "all_bot_intents": bot_policy.all_bot_intents}


def _targeting(*args, **kwargs):
    _CAPTURE["tgt"] = _ORIG["targeting"](*args, **kwargs)
    return _CAPTURE["tgt"]


def _intents(*args, **kwargs):
    _CAPTURE["intent"] = _ORIG["all_bot_intents"](*args, **kwargs)
    return _CAPTURE["intent"]


def _np(t, dtype=None):
    a = t.detach().to("cpu", copy=True).numpy()
    return a.astype(dtype) if dtype is not None else a


class Recorder:
    """Per-tick world rows (tick_hook) and per-decision rows (the drive loop), stacked later."""

    def __init__(self):
        self.ticks = defaultdict(list)
        self.decisions = defaultdict(list)

    @property
    def n_ticks(self) -> int:
        return len(self.ticks["pos"])

    def on_tick(self, env):
        s, intent, tgt = env.state, _CAPTURE["intent"], _CAPTURE["tgt"]
        lo, hi, active = bot_policy.zone_rect(s)
        row = self.ticks
        row["pos"].append(_np(s.ent_pos))
        row["alive"].append(_np(s.ent_alive))
        row["person"].append(_np(s.ent_person, np.int8))
        row["cause"].append(_np(s.ent_death_cause, np.int8))
        row["last_hit"].append(_np(s.ent_last_hit_by, np.int8))
        row["mode"].append(_np(intent.mode, np.int8))
        row["fire"].append(_np(intent.fire))
        row["move_norm"].append(_np(intent.move_dir.norm(dim=-1), np.float16))
        row["has_enemy"].append(_np(tgt.has_enemy))
        row["tgt"].append(_np(tgt.idx, np.int8))
        row["los"].append(_np(tgt.los))
        row["is_box"].append(_np(tgt.is_box))
        row["tgt_dist"].append(_np(tgt.dist, np.float16))
        row["in_gas"].append(_np(bot_perception.in_zone(s.ent_pos, lo, hi) & active))
        row["clear"].append(_np(bot_policy.zone_clearance(s, env.cfg, env.bank), np.float16))
        row["dash_t"].append(_np(s.ent_dash_t[:, 0]))
        row["cubes"].append(_np(s.ent_cubes[:, 0]))
        row["dealt"].append(_np(s.ent_damage_dealt[:, 0]))
        row["taken"].append(_np(s.ent_damage_taken[:, 0]))
        row["hero_hit_by"].append(_np(s.ent_last_hit_by[:, 0], np.int8))
        row["ammo"].append(_np(s.ent_ammo, np.float16))
        row["reach"].append(_np(tgt.fire_reach, np.float16))
        row["box_hp"].append(_np(s.box_hp))
        row["box_alive"].append(_np(s.box_alive))
        row["box_pos"].append(_np(s.box_pos, np.float16))
        row["pku_pos"].append(_np(s.pku_pos))
        row["pku_alive"].append(_np(s.pku_alive))
        row["zone_lo"].append(_np(s.zone_lo))
        row["zone_hi"].append(_np(s.zone_hi))
        row["map_id"].append(_np(s.map_id))

    def on_decision(self, env, action, masks, probs, aprobs):
        s = env.state
        hero = s.ent_pos[:, :1]
        prj_d = (s.prj_pos - hero).norm(dim=-1)
        threat = (s.prj_alive & (s.prj_owner != 0) & (prj_d <= _THREAT_TILES)).any(-1)
        d = self.decisions
        d["tick"].append(self.n_ticks)
        d["action"].append(np.asarray(action).copy())
        d["masks"].append(np.asarray(masks).copy())
        d["probs"].append(probs)
        d["aprobs"].append(aprobs)
        d["hero_view"].append(_np(env._obs_hero_view))
        d["threat"].append(_np(threat))
        d["ammo"].append(_np(s.ent_ammo[:, 0]))

    def stacked(self):
        return ({k: np.stack(v) for k, v in self.ticks.items()},
                {k: np.stack(v) for k, v in self.decisions.items()})


def _policy_probs(model, obs, masks) -> tuple[np.ndarray, np.ndarray]:
    """((B, move bins), (B, attack values)) -- the policy's own masked distributions: the move
    column for the top-2 margin, the attack column for argmax's split vote between values 1 and 4."""
    with torch.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_t, action_masks=masks)
        return (dist.distributions[0].probs.cpu().numpy(),
                dist.distributions[1].probs.cpu().numpy())


def _drive(model, venv, sim, rec, deterministic, max_decisions):
    """Runs every env to its first episode end; returns (end tick per env, outcome per env)."""
    n = sim.n_envs
    end = np.full(n, -1)
    outcome = [None] * n
    obs = venv.reset()
    sim.tick_hook = rec.on_tick
    try:
        for _ in range(max_decisions):
            masks = venv.action_masks()
            probs, aprobs = _policy_probs(model, obs, masks)
            action, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
            rec.on_decision(sim, action, masks, probs, aprobs)
            obs, _, dones, infos = venv.step(action)
            for i in np.flatnonzero(dones):
                if end[i] < 0:
                    end[i] = rec.n_ticks
                    outcome[i] = {"won": bool(infos[i]["outcome"]["won"]),
                                  "rank": int(infos[i]["outcome"]["rank"]),
                                  "truncated": bool(infos[i].get("TimeLimit.truncated", False))}
            if (end >= 0).all():
                break
    finally:
        sim.tick_hook = None
    end[end < 0] = rec.n_ticks
    return end, outcome


def _episodes(ticks, decs, end, outcome):
    """One dict of (T, ...) arrays per env, cut at that env's first episode end."""
    out = []
    for i in range(len(end)):
        t_end = int(end[i])
        ep = {k: v[:t_end, i] for k, v in ticks.items()}
        keep = decs["tick"][:, 0] < t_end if decs["tick"].ndim > 1 else decs["tick"] < t_end
        ep_dec = {k: (v[keep, i] if k != "tick" else v[keep]) for k, v in decs.items()}
        ep["dec"] = ep_dec
        ep["outcome"] = outcome[i]
        out.append(ep)
    return out


# ------------------------------------------------------------------------------------------
# issue 1 + 2: bot deaths, stalls, KITE without LOS, the late game
# ------------------------------------------------------------------------------------------

def _speed(pos, dt):
    v = np.zeros(pos.shape[:-1], np.float32)
    v[1:] = np.linalg.norm(pos[1:] - pos[:-1], axis=-1) / dt
    return v


def bot_deaths(ep, dt, map_w, map_h):
    alive, pos = ep["alive"], ep["pos"]
    speed = _speed(pos, dt)
    rows = []
    for e in range(1, alive.shape[1]):
        dead = np.flatnonzero(~alive[:, e])
        if not alive[0, e] or dead.size == 0:
            continue
        t = int(dead[0])
        w = slice(max(0, t - 60), t)
        stalled = (ep["move_norm"][w, e] > 0.5) & (speed[w, e] < _STALL_SPEED)
        x, y = pos[t - 1, e]
        before = max(0, t - 200)
        rows.append({
            "t": t * dt, "e": e, "person": _PERSON[int(ep["person"][t - 1, e])],
            "cause": int(ep["cause"][t, e]), "killer": int(ep["last_hit"][t, e]),
            "stalled_3s": float(stalled.mean()) if stalled.size else 0.0,
            "moved_3s": float(np.linalg.norm(pos[t - 1, e] - pos[w.start, e])),
            "gas_5s": float(ep["in_gas"][max(0, t - 100):t, e].mean()),
            "clear_10s_before": float(ep["clear"][before, e]),
            "mode_3s": Counter(_MODE[int(m)] for m in ep["mode"][w, e]).most_common(2),
            "border": float(min(x, map_w - x, y, map_h - y)),
        })
    return rows


def stall_share(ep, dt):
    """(person -> [alive ticks, stalled ticks]) over the whole episode, bots only."""
    speed = _speed(ep["pos"], dt)
    stalled = (ep["move_norm"] > 0.5) & (speed < _STALL_SPEED) & ep["alive"]
    out = defaultdict(lambda: np.zeros(2))
    for e in range(1, ep["alive"].shape[1]):
        p = _PERSON[int(ep["person"][0, e])]
        out[p] += (ep["alive"][1:, e].sum(), stalled[1:, e].sum())
    return out


def stall_by_mode(ep, dt, avoid_tiles):
    """(mode -> Counter(alive, stalled, zone_pull)) over bot ticks: which steering the stalls
    happen under, and how many of them with the zone's inward pull on (clearance < avoid_tiles)."""
    speed = _speed(ep["pos"], dt)
    stalled = (ep["move_norm"] > 0.5) & (speed < _STALL_SPEED) & ep["alive"]
    pull = ep["clear"] < avoid_tiles
    out = defaultdict(Counter)
    for e in range(1, ep["alive"].shape[1]):
        a, s, m, z = ep["alive"][1:, e], stalled[1:, e], ep["mode"][1:, e], pull[1:, e]
        for mode_id in np.unique(m[a]):
            sel = a & (m == mode_id)
            c = out[_MODE[int(mode_id)]]
            c["alive"] += int(sel.sum())
            c["stalled"] += int((sel & s).sum())
            c["zone_pull"] += int((sel & s & z).sum())
    return out


def kite_no_los(ep, dt):
    """KITE ticks in HOLD_RANGE on an entity target, and why each one is not a shot: no LOS,
    past fire reach, out of ammo, or none of those (cooldown, decision period, lateral hold)."""
    speed = _speed(ep["pos"], dt)
    kite = (ep["person"] == int(Person.KITE)) & ep["alive"]
    kite[:, 0] = False
    hold = kite & (ep["mode"] == int(Mode.HOLD_RANGE)) & ep["has_enemy"] & ~ep["is_box"]
    fire = ep["fire"] & hold
    no_los = hold & ~fire & ~ep["los"]
    far = hold & ~fire & ep["los"] & (ep["tgt_dist"].astype(np.float32) > ep["reach"].astype(np.float32))
    dry = hold & ~fire & ~no_los & ~far & (ep["ammo"].astype(np.float32) < 1.0)
    other = hold & ~fire & ~no_los & ~far & ~dry
    out = {"kite_alive": int(kite.sum()), "hold": int(hold.sum()), "fire": int(fire.sum()),
           "no_los": int(no_los.sum()), "far": int(far.sum()), "dry": int(dry.sum()),
           "other": int(other.sum()),
           "no_los_standing": int((no_los & (speed < _STALL_SPEED)).sum()),
           "no_los_on_hero": int((no_los & (ep["tgt"] == 0)).sum())}
    # Longest unbroken run of no-LOS holding per kiter, in ticks.
    longest = 0
    for e in range(1, no_los.shape[1]):
        run = 0
        for v in no_los[:, e]:
            run = run + 1 if v else 0
            longest = max(longest, run)
    out["no_los_longest"] = longest
    return out


def late_game(ep, deaths, dt):
    bots_left = ep["alive"][:, 1:].sum(1)
    hero_alive = ep["alive"][:, 0]
    T = len(bots_left)
    idx = np.flatnonzero((bots_left <= 3) & hero_alive)
    if idx.size == 0:
        return None
    t4 = int(idx[0])
    survivors = [_PERSON[int(ep["person"][t4, e])] for e in range(1, ep["alive"].shape[1])
                 if ep["alive"][t4, e]]
    after = [d for d in deaths if d["t"] > t4 * dt]
    dealt = ep["dealt"]
    # Longest stretch after t4 with no hero damage dealt and no bot death.
    events = np.zeros(T, bool)
    events[1:] = np.diff(dealt) > 0
    for d in after:
        events[min(T - 1, int(round(d["t"] / dt)))] = True
    gaps, last = [], t4
    for t in np.flatnonzero(events[t4:]) + t4:
        gaps.append(t - last)
        last = t
    gaps.append(T - last)
    return {
        "t4": t4 * dt, "phase_s": (T - t4) * dt, "survivors": survivors,
        "hero_dealt": float(dealt[-1] - dealt[t4]),
        "deaths": [(d["person"], "zone" if d["cause"] == int(DeathCause.ZONE)
                    else ("hero" if d["killer"] == 0 else "bot")) for d in after],
        "longest_quiet_s": max(gaps) * dt,
    }


# ------------------------------------------------------------------------------------------
# issue 3: cubes the hero hangs around without taking
# ------------------------------------------------------------------------------------------

def cube_runs(ep, dt, pickup_radius, max_cubes):
    alive, pos = ep["pku_alive"], ep["pku_pos"]
    hero, hero_alive, cubes = ep["pos"][:, 0], ep["alive"][:, 0], ep["cubes"]
    T, P = alive.shape
    out = []
    for p in range(P):
        a = alive[:, p].astype(np.int8)
        starts = np.flatnonzero(np.diff(np.r_[0, a]) == 1)
        stops = np.flatnonzero(np.diff(np.r_[a, 0]) == -1) + 1
        for t0, t1 in zip(starts, stops):
            c = pos[t0, p]
            d = np.linalg.norm(hero[t0:t1] - c, axis=-1)
            near = (d <= _NEAR_CUBE) & hero_alive[t0:t1]
            if not near.any():
                taken_by = "other" if t1 < T else "left"
                out.append({"near_s": 0.0, "taken_by": taken_by})
                continue
            same_tile = (np.floor(hero[t0:t1]) == np.floor(c)).all(-1) & (d > pickup_radius)
            lo, hi = ep["zone_lo"][t0:t1], ep["zone_hi"][t0:t1]
            cube_in_gas = ((c < lo) | (c > hi)).any(-1)
            if t1 < T:
                hero_took = cubes[t1] > cubes[t1 - 1] and np.linalg.norm(hero[t1] - c) <= pickup_radius + 0.2
                taken_by = "hero" if hero_took else "other"
            else:
                taken_by = "left"
            first_near = int(np.flatnonzero(near)[0])
            out.append({
                "p": p, "t0": t0 * dt, "t1": t1 * dt, "pos": c.round(2).tolist(),
                "near_s": float(near.sum() * dt), "min_d": float(d[hero_alive[t0:t1]].min()),
                "same_tile_out_of_reach_s": float((same_tile & near).sum() * dt),
                "in_gas_while_near": float(cube_in_gas[near].mean()),
                "capped_while_near": float((cubes[t0:t1] >= max_cubes)[near].mean()),
                "latency_s": (t1 - t0 - first_near) * dt if taken_by == "hero" else None,
                "taken_by": taken_by, "map_id": int(ep["map_id"][t0]),
                "window": (t0 + first_near, t1),
            })
    return out


def standable_fraction(bank, cfg, map_id, c, radius, unit_radius) -> float:
    """Share of points within `radius` of cube `c` where a hero body fits (circle_blocked)."""
    r = np.sqrt(np.linspace(0.0, 1.0, 6))[:, None] * radius
    th = np.linspace(0, 2 * np.pi, 12, endpoint=False)[None, :]
    pts = np.stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th)], -1).reshape(-1, 2)
    pos = torch.as_tensor(pts, dtype=torch.float32).unsqueeze(0)
    mid = torch.tensor([map_id])
    rad = torch.full(pos.shape[:-1], unit_radius)
    blocked = terrain.circle_blocked(bank.blocks_unit, mid, pos, rad, cfg)
    return float((~blocked).float().mean())


# ------------------------------------------------------------------------------------------
# issue 4: crate HP per dash
# ------------------------------------------------------------------------------------------

def scripted_dash_vs_crate(hits_once):
    """HP one uncharged hero dash takes off a crate, for crates at several spots on and beside
    the path, under the run's own `boxes.dash_hits_once`. Built on tests/test_hero.py's own
    fixtures (an open 20x20 floor)."""
    from brawl_sim.constants import Kind
    from brawl_sim.core import hero
    from tests.test_hero import _bank_from_grid, _cfg_and_params, _fresh_state, _grid
    cfg, params = _cfg_and_params(extra_overrides={"boxes": {"dash_hits_once": hits_once}})
    rows = []
    for ahead in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        for side in (0.0, 0.4, 0.7):
            state = _fresh_state(cfg)
            state.ent_pos[0, 0] = torch.tensor([5.0, 10.0])
            state.box_pos[0, 0] = torch.tensor([5.0 + ahead, 10.0 + side])
            state.box_alive[0, 0] = True
            state.box_hp[0, 0] = 1e9
            state.box_max_hp[0, 0] = 1e9
            bank = _bank_from_grid(_grid(20, 20))
            fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
            fire[0, 0] = True
            move = torch.zeros(1, cfg.n_entities, 2)
            move[0, 0] = torch.tensor([1.0, 0.0])
            hero.start_dash(state, fire, move, bank, params, cfg)
            n = int(round(params.dash_duration[0, int(Kind.HERO_MORTIS)].item() / cfg.dt)) + 1
            hits, total = 0, 0.0
            for _ in range(n):
                _, _, dmg_box = hero.advance_dash(state, params, cfg)
                hits += int(dmg_box[0, 0] > 0)
                total += float(dmg_box[0, 0])
            rows.append((ahead, side, hits, total))
    return rows, float(params.base_damage[0, int(Kind.HERO_MORTIS)])


def rollout_dashes(ep, decs, crate_hp):
    """Per hero dash in the rollout: the most HP it took off one crate, and whether it broke a
    crate that was at full HP when the dash started. Gadget decisions are left out (its spinner
    also damages crates).

    Only crates the dash passed: within `_DASH_PATH_TILES` of the hero on some tick of the
    window. The window alone took every crate on the map, and bots break crates too (2026-10-07:
    two full crates 20 and 54 tiles from the hero, shot by bots in the first 2 s of a match, read
    as one-dash breaks)."""
    dash_t, hp, alive = ep["dash_t"], ep["box_hp"], ep["box_alive"]
    T = len(dash_t)
    gadget_ticks = set()
    for k, t in enumerate(decs["tick"]):
        if decs["action"][k][1] == 3:
            gadget_ticks.update(range(int(t), int(t) + 10))
    out = []
    starts = np.flatnonzero((dash_t[1:] > 0) & (dash_t[:-1] <= 0)) + 1
    for t in starts:
        if t in gadget_ticks or t + 7 >= T:
            continue
        before, after = hp[t - 1], hp[t + 7]
        hero = ep["pos"][t - 1:t + 8, 0].astype(np.float64)                      # (9, 2)
        crate = ep["box_pos"][t - 1].astype(np.float64)                           # (B, 2)
        passed = (np.linalg.norm(crate[:, None] - hero[None], axis=-1).min(-1) <= _DASH_PATH_TILES)
        was = alive[t - 1] & passed
        lost = np.where(was, before - np.where(alive[t + 7], after, 0.0), 0.0)
        if lost.max() <= 0:
            continue
        full_broken = bool((was & ~alive[t + 7] & (before >= crate_hp - 1e-3)).any())
        out.append((float(lost.max()), full_broken))
    return out


# ------------------------------------------------------------------------------------------
# issue 5: hero movement
# ------------------------------------------------------------------------------------------

def _bin_gap(a, b):
    g = np.abs(a - b) % 16
    return np.minimum(g, 16 - g)


def _decision_context(ep, max_cubes):
    """Per decision: threat (an enemy projectile within 4 tiles) > enemy (one in view within 8) >
    capped_cube (hero at the cube cap, a live cube within 3) > seen_far (one in view, farther) >
    alone. Read off the tick before the decision, i.e. the state its observation showed."""
    dec = ep["dec"]
    ticks = np.maximum(np.asarray(dec["tick"]).astype(int) - 1, 0)
    pos = ep["pos"][ticks, 0]
    d_enemy = np.linalg.norm(ep["pos"][ticks, 1:] - pos[:, None], axis=-1)
    seen = np.asarray(dec["hero_view"])[:, 1:] & ep["alive"][ticks, 1:]
    d_cube = np.linalg.norm(ep["pku_pos"][ticks] - pos[:, None], axis=-1)
    cube_near = (ep["pku_alive"][ticks] & (d_cube <= 3.0)).any(1)
    capped = ep["cubes"][ticks] >= max_cubes
    ctx = np.full(len(ticks), "alone", dtype=object)
    ctx[seen.any(1)] = "seen_far"
    ctx[capped & cube_near] = "capped_cube"
    ctx[(seen & (d_enemy <= _ENEMY_NEAR)).any(1)] = "enemy"
    ctx[np.asarray(dec["threat"]).astype(bool)] = "threat"
    in_reach = (seen & (d_enemy <= _DASH_REACH)).any(1)
    return ctx, in_reach


def jitter(ep, max_cubes):
    dec = ep["dec"]
    ticks = np.asarray(dec["tick"]).astype(int)
    if ticks.size < 3:
        return None
    act = np.asarray(dec["action"]).reshape(len(ticks), -1)
    move, att = act[:, 0], act[:, 1]
    hero_alive = ep["alive"][np.maximum(ticks - 1, 0), 0]
    pos = ep["pos"][np.maximum(ticks - 1, 0), 0]
    ctx, in_reach = _decision_context(ep, max_cubes)
    attack_legal = np.asarray(dec["masks"])[:, 17 + 1].astype(bool)   # value 1, the aimed dash
    probs = np.asarray(dec["probs"])
    aprobs = np.asarray(dec["aprobs"])
    # p(an attack that dashes at the enemy): value 1, plus value 4 on an auto-aim run.
    p_att = aprobs[:, 1] + (aprobs[:, 4] if aprobs.shape[1] > 4 else 0.0)

    rows = defaultdict(Counter)
    for k in range(1, len(ticks)):
        if not (hero_alive[k] and hero_alive[k - 1]):
            continue
        for c in (rows[ctx[k]], rows["all"]):
            c["n"] += 1
            c["idle"] += int(move[k] == 0)
            if move[k] and move[k - 1]:
                g = _bin_gap(move[k], move[k - 1])
                c["pairs"] += 1
                c["change"] += int(g > 0)
                c["reverse"] += int(g >= _REVERSAL_BINS)
                if k >= 2 and move[k - 2] and move[k] == move[k - 2] and g >= 4:
                    c["aba"] += 1
            c["attack"] += int(att[k] in (1, 2, 3, 4))
            if attack_legal[k] and in_reach[k]:
                c["chance"] += 1
                c["chance_taken"] += int(att[k] in (1, 4))
                c["chance_p"] += float(p_att[k])
                # A split vote: no attack was the single likeliest value, yet 1 and 4 together
                # outweighed it, so argmax passed on an attack the policy mostly wanted.
                c["chance_split"] += int(att[k] == 0 and p_att[k] > aprobs[k, 0])
            order = np.argsort(probs[k])[::-1]
            c["p_top"] += float(probs[k][order[0]])
            c["margin"] += float(probs[k][order[0]] - probs[k][order[1]])
            if move[k] and move[k - 1] and _bin_gap(move[k], move[k - 1]) >= _REVERSAL_BINS:
                # How sure was the policy of the reversal it made? (argmax-flip dithering would
                # show a small margin here.)
                c["rev_margin"] += float(probs[k][order[0]] - probs[k][order[1]])
    # Path efficiency over 2 s (8 decisions) windows where the hero lives throughout.
    eff = []
    for k in range(0, len(ticks) - 8, 4):
        if not hero_alive[k:k + 9].all():
            continue
        steps = np.linalg.norm(np.diff(pos[k:k + 9], axis=0), axis=-1).sum()
        if steps > 0.5:
            eff.append(np.linalg.norm(pos[k + 8] - pos[k]) / steps)
    return rows, eff


def bot_shots_at_hero(ep):
    """Every bot shot aimed at the hero, and whether that bot hit him within 1.5 s, by what the
    hero did in the 0.5 s after the shot: reversed his move bin, held it, or idled."""
    dec = ep["dec"]
    ticks = np.asarray(dec["tick"]).astype(int)
    move = np.asarray(dec["action"]).reshape(len(ticks), -1)[:, 0]
    taken, hit_by = ep["taken"], ep["hero_hit_by"]
    T = len(taken)
    got_hit = np.zeros(T, bool)
    got_hit[1:] = np.diff(taken) > 0
    shots = np.argwhere(ep["fire"] & (ep["tgt"] == 0) & ~ep["is_box"] & ep["alive"][:, :1])
    out = Counter()
    for t, b in shots:
        if b == 0 or t + 30 >= T or not ep["alive"][t, 0]:
            continue
        ks = np.flatnonzero((ticks > t) & (ticks <= t + 10))
        prev = np.flatnonzero(ticks <= t)
        if not ks.size or not prev.size:
            continue
        seq = np.r_[move[prev[-1]], move[ks]]
        rev = any(a and c and _bin_gap(a, c) >= _REVERSAL_BINS for a, c in zip(seq[1:], seq[:-1]))
        what = "reversed" if rev else ("idle" if not seq.any() else "held/turned")
        hit = bool((got_hit[t + 1:t + 31] & (hit_by[t + 1:t + 31] == b)).any())
        out[f"{what}:n"] += 1
        out[f"{what}:hit"] += int(hit)
    return out


# ------------------------------------------------------------------------------------------

def _build_train_env(tcfg, tier, seed, n_envs, device):
    overrides = watch_env_overrides(tcfg)
    tcfg = replace(tcfg, run=replace(tcfg.run, env_overrides=overrides))
    env_cfg = load_config(_resolve(tcfg.run.env_config), overrides=overrides or None)
    spec = obs_select.load_agent_spec(_resolve(tcfg.run.agent_obs), env_cfg)
    rand = str(_resolve(tcfg.run.randomization)) if tcfg.run.randomization else None
    sim = BrawlVecEnv(env_cfg, n_envs=n_envs, device=device, seed=seed,
                      reward_fn=ShapedReward(tcfg.reward, track_terms=False), randomization=rand,
                      spec=build_spec(tcfg), verbose=False, autoreset=True)
    sim.params_hook = FixedTierHook.uniform(tcfg.curriculum.tiers, sim.device, n_envs, tier)
    return sim, BrawlSB3VecEnv(sim, spec, sim.reward_fn, info_mode="episode"), env_cfg


def _collect(args, tcfg, model):
    episodes, sims = [], []
    if args.mode == "watch":
        for i in range(args.episodes):
            sim, venv, env_cfg = build_watch_env(tcfg, args.tier, args.seed + i, "cpu")
            venv = maybe_wrap_vecnormalize(venv, Path(args.model), tcfg, verbose=False)
            rec = Recorder()
            end, outcome = _drive(model, venv, sim, rec, True, env_cfg.max_agent_steps)
            ticks, decs = rec.stacked()
            for ep in _episodes(ticks, decs, end, outcome):
                ep["seed"] = args.seed + i
                episodes.append(ep)
            sims.append(sim)
            print(f"  seed {args.seed + i}: {outcome[0]}  ({end[0] * env_cfg.dt:.0f} s)",
                  flush=True)
    else:
        sim, venv, env_cfg = _build_train_env(tcfg, args.tier, args.seed, args.envs, "cpu")
        rec = Recorder()
        end, outcome = _drive(model, venv, sim, rec, False, env_cfg.max_agent_steps + 5)
        ticks, decs = rec.stacked()
        for i, ep in enumerate(_episodes(ticks, decs, end, outcome)):
            ep["seed"] = f"{args.seed}/env{i}"
            episodes.append(ep)
        sims.append(sim)
    return episodes, sims[-1], env_cfg


def _report(episodes, sim, cfg, tcfg):
    dt = cfg.dt
    n = len(episodes)
    won = sum(ep["outcome"]["won"] for ep in episodes if ep["outcome"])
    print(f"\n{n} episodes, hero won {won}")

    # ---------------- 1. deaths, stalls, KITE ----------------
    deaths = []
    stall = defaultdict(lambda: np.zeros(2))
    stall_mode = defaultdict(Counter)
    kite = Counter()
    for ep in episodes:
        rows = bot_deaths(ep, dt, cfg.map_w, cfg.map_h)
        for r in rows:
            r["seed"] = ep["seed"]
        ep["deaths"] = rows
        deaths += rows
        for p, v in stall_share(ep, dt).items():
            stall[p] += v
        for m, c in stall_by_mode(ep, dt, cfg.bots_zone_avoid_tiles).items():
            stall_mode[m].update(c)
        kite.update(kite_no_los(ep, dt))
    print("\n[1] bot deaths by personality (zone / combat by hero / combat by bot)")
    by_p = defaultdict(Counter)
    for d in deaths:
        kind = "zone" if d["cause"] == int(DeathCause.ZONE) else ("hero" if d["killer"] == 0 else "bot")
        by_p[d["person"]][kind] += 1
    for p in sorted(by_p):
        c = by_p[p]
        tot = sum(c.values())
        print(f"    {p:8s} {tot:4d} deaths: zone {c['zone']:3d} ({c['zone'] / tot:5.1%})  "
              f"hero {c['hero']:3d}  bot {c['bot']:3d}")
    zone = [d for d in deaths if d["cause"] == int(DeathCause.ZONE)]
    if zone:
        stuck = [d for d in zone if d["stalled_3s"] >= 0.5]
        print(f"    zone deaths: {len(zone)}; stalled >= half of the last 3 s: {len(stuck)} "
              f"({len(stuck) / len(zone):.0%}); moved < 1 tile in the last 3 s: "
              f"{sum(d['moved_3s'] < 1.0 for d in zone)}")
        print(f"    median border distance at a zone death {np.median([d['border'] for d in zone]):.1f}"
              f" tiles; median clearance 10 s before {np.median([d['clear_10s_before'] for d in zone]):.1f}")
        modes = Counter(m for d in zone for m, _ in d["mode_3s"][:1])
        print(f"    last-3 s majority mode of zone deaths: {dict(modes)}")
        for d in zone[:6]:
            print(f"      seed {d['seed']} t={d['t']:5.1f}s {d['person']:8s} stalled {d['stalled_3s']:.0%}"
                  f" moved {d['moved_3s']:.1f} border {d['border']:.1f} modes {d['mode_3s']}")
    print("    stalled share of alive ticks (pushing, not moving), by personality:")
    for p in sorted(stall):
        a, s = stall[p]
        print(f"      {p:8s} {s / max(a, 1):6.1%}")
    all_stalled = max(sum(c["stalled"] for c in stall_mode.values()), 1)
    print("    stalls by mode: share of all stalled ticks, stalled share of the mode's own ticks, "
          "and stalled ticks with the zone pull on:")
    for m in sorted(stall_mode, key=lambda k: -stall_mode[k]["stalled"]):
        c = stall_mode[m]
        print(f"      {m:10s} {c['stalled'] / all_stalled:5.1%} of stalls  {c['stalled'] / max(c['alive'], 1):5.1%}"
              f" of its ticks  zone pull on {c['zone_pull'] / max(c['stalled'], 1):5.1%}")
    if kite["hold"]:
        h = kite["hold"]
        print(f"    KITE: {kite['kite_alive'] * dt:.0f} s alive, {h * dt:.0f} s holding range on an "
              f"enemy ({h / max(kite['kite_alive'], 1):.0%}); {kite['fire']} shots = "
              f"{kite['fire'] / (h * dt):.2f}/s while holding")
        print(f"      not-shooting ticks while holding: no LOS {kite['no_los'] / h:.0%}, past fire "
              f"reach {kite['far'] / h:.0%}, no ammo {kite['dry'] / h:.0%}, cooldown/decision/"
              f"lateral {kite['other'] / h:.0%}")
        print(f"      no-LOS holding: standing {kite['no_los_standing'] / max(kite['no_los'], 1):.0%}, "
              f"on the hero {kite['no_los_on_hero'] / max(kite['no_los'], 1):.0%}, longest unbroken "
              f"{kite['no_los_longest'] * dt:.1f} s")

    # ---------------- 2. late game ----------------
    print("\n[2] late game (from the first tick with <= 3 bots alive and the hero alive)")
    lates = [late_game(ep, ep["deaths"], dt) for ep in episodes]
    lates = [x for x in lates if x]
    if lates:
        phase = np.array([x["phase_s"] for x in lates])
        ep_len = np.array([len(ep["alive"]) * dt for ep in episodes])
        print(f"    reached by {len(lates)}/{n}; phase length median {np.median(phase):.0f} s "
              f"(mean {phase.mean():.0f} s) of median episode {np.median(ep_len):.0f} s")
        surv = Counter(p for x in lates for p in x["survivors"])
        print(f"    who is left: {dict(surv)}")
        how = Counter(h for x in lates for _, h in x["deaths"])
        howp = Counter(f"{p}:{h}" for x in lates for p, h in x["deaths"])
        print(f"    how they die: {dict(how)}")
        print(f"    by personality: {dict(howp)}")
        quiet = np.array([x["longest_quiet_s"] for x in lates])
        print(f"    longest stretch with no hero damage and no death: median {np.median(quiet):.0f} s,"
              f" max {quiet.max():.0f} s; hero damage in the phase median "
              f"{np.median([x['hero_dealt'] for x in lates]):.0f}")

    # ---------------- 3. cubes ----------------
    print("\n[3] cubes")
    radius = float(sim.params.pickup_radius.flatten()[0])
    unit_r = float(sim.params.unit_radius.flatten()[0])
    max_cubes = int(sim.params.max_cubes.flatten()[0])
    capped_at = [np.flatnonzero(ep["cubes"] >= max_cubes) for ep in episodes]
    capped_at = [c[0] * dt for c in capped_at if c.size]
    print(f"    hero reached the {max_cubes}-cube cap in {len(capped_at)}/{n} episodes"
          + (f", median at {np.median(capped_at):.0f} s" if capped_at else ""))
    all_runs, flagged = [], []
    for ep in episodes:
        runs = cube_runs(ep, dt, radius, max_cubes)
        for r in runs:
            r["seed"] = ep["seed"]
            r["ep"] = ep
        all_runs += runs
        flagged += [r for r in runs if r["near_s"] >= _NEAR_CUBE_SECONDS and r["taken_by"] != "hero"]
    near = [r for r in all_runs if r["near_s"] > 0]
    took = [r for r in near if r["taken_by"] == "hero"]
    print(f"    cubes the hero came within {_NEAR_CUBE} tiles of: {len(near)}; he took {len(took)}; "
          f"latency (first near -> taken) median {np.median([r['latency_s'] for r in took]) if took else float('nan'):.1f} s, "
          f"90th pct {np.percentile([r['latency_s'] for r in took], 90) if took else float('nan'):.1f} s")
    st = sum(r["same_tile_out_of_reach_s"] for r in near)
    print(f"    seconds the hero shared a cube's tile but stood out of reach: {st:.0f} s total")
    print(f"    near >= {_NEAR_CUBE_SECONDS:.0f} s and NOT taken by the hero: {len(flagged)}")
    for r in flagged[:10]:
        stand = standable_fraction(sim.bank, cfg, r["map_id"], np.array(r["pos"]), radius, unit_r)
        ep = r["ep"]
        w0, w1 = r["window"]
        dec = ep["dec"]
        ticks = np.asarray(dec["tick"]).astype(int)
        sel = (ticks >= w0) & (ticks < w1)
        moves = np.stack(dec["action"]).reshape(len(ticks), -1)[sel, 0]
        rev = np.mean([_bin_gap(a, b) >= _REVERSAL_BINS for a, b in zip(moves[1:], moves[:-1])
                       if a and b]) if sel.sum() > 2 else float("nan")
        print(f"      seed {r['seed']} cube {r['pos']} alive {r['t0']:.0f}-{r['t1']:.0f} s near "
              f"{r['near_s']:.1f} s min_d {r['min_d']:.2f} same-tile-out-of-reach "
              f"{r['same_tile_out_of_reach_s']:.1f} s in-gas {r['in_gas_while_near']:.0%} "
              f"capped {r['capped_while_near']:.0%} standable {stand:.0%} reversals {rev:.0%} "
              f"-> {r['taken_by']}")
    if flagged:
        print(f"    of those, hero at the cap for most of the time near: "
              f"{sum(r['capped_while_near'] > 0.5 for r in flagged)}/{len(flagged)}")

    # ---------------- 4. crates ----------------
    print("\n[4] crate HP per hero dash")
    rows, base = scripted_dash_vs_crate(cfg.box_dash_hits_once)
    crate_hp = float(sim.params.box_hp.flatten()[0])
    rule = "once per dash" if cfg.box_dash_hits_once else "every tick"
    print(f"    scripted, base damage {base:.0f}, crate hp {crate_hp:.0f}, hit {rule}: "
          "(tiles ahead, tiles aside) -> ticks hit, HP taken")
    print("      " + "  ".join(f"({a:.1f},{s:.1f})->{h}x {t:.0f}" for a, s, h, t in rows))
    dashes = [x for ep in episodes for x in rollout_dashes(ep, ep["dec"], crate_hp)]
    if dashes:
        lost = np.array([x[0] for x in dashes])
        print(f"    rollouts: {len(dashes)} dashes touched a crate; HP taken from one crate "
              f"median {np.median(lost):.0f}, max {lost.max():.0f}; a full-HP crate broken by one "
              f"dash: {sum(x[1] for x in dashes)}")

    # ---------------- 5. jitter ----------------
    print("\n[5] hero movement, per decision (0.25 s)")
    tot = defaultdict(Counter)
    shots = Counter()
    effs = []
    for ep in episodes:
        res = jitter(ep, max_cubes)
        if res:
            rows, eff = res
            for k, c in rows.items():
                tot[k].update(c)
            effs += eff
        shots.update(bot_shots_at_hero(ep))
    for k in ("all", "alone", "seen_far", "capped_cube", "enemy", "threat"):
        c = tot[k]
        if not c["n"]:
            continue
        pairs = max(c["pairs"], 1)
        print(f"    {k:11s} n={c['n']:6d} ({c['n'] / max(tot['all']['n'], 1):4.0%})  idle "
              f"{c['idle'] / c['n']:4.0%}  change {c['change'] / pairs:4.0%}  REVERSE "
              f"{c['reverse'] / pairs:4.0%}  A-B-A {c['aba'] / pairs:4.0%}  attack "
              f"{c['attack'] / c['n']:4.0%}  dash when legal+in reach "
              f"{c['chance_taken'] / max(c['chance'], 1):4.0%} (n={c['chance']})  p(top) "
              f"{c['p_top'] / c['n']:.2f}  margin {c['margin'] / c['n']:.2f}  margin at a reversal "
              f"{c['rev_margin'] / max(c['reverse'], 1):.2f}")
    c = tot["all"]
    if c["chance"]:
        print(f"    attack chances (legal, enemy in view within dash reach): {c['chance']}; mean "
              f"p(attack 1 or 4) {c['chance_p'] / c['chance']:.2f}; passed by a split vote "
              f"(no attack chosen though p(1)+p(4) > p(0)): {c['chance_split'] / c['chance']:.0%}")
    if effs:
        effs = np.array(effs)
        print(f"    path efficiency over 2 s windows (net / walked): median {np.median(effs):.2f}, "
              f"share below 0.3: {np.mean(effs < 0.3):.0%}")
    if shots:
        print("    bot shots aimed at the hero -> hit within 1.5 s, by the hero's next 0.5 s:")
        for what in ("reversed", "held/turned", "idle"):
            k = shots[f"{what}:n"]
            if k:
                print(f"      {what:12s} {k:6d} shots, hit {shots[f'{what}:hit'] / k:5.1%}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--tier", default="expert")
    p.add_argument("--mode", choices=("watch", "train"), default="watch")
    p.add_argument("--episodes", type=int, default=8, help="watch mode: matches, seeds seed..")
    p.add_argument("--envs", type=int, default=16, help="train mode: batched envs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override the run's train.yaml, as scripts/train.py --set does "
                        "(e.g. run.env_overrides.bots.nav=true); repeatable")
    args = p.parse_args(argv)

    model_path = Path(args.model)
    tcfg = load_train_config(find_train_config(model_path), overrides=parse_overrides(args.set),
                             check_holdout=False)
    model, _ = load_model(model_path, tcfg.run.algo, "cpu")
    bot_policy.targeting, bot_policy.all_bot_intents = _targeting, _intents
    t0 = time.time()
    try:
        episodes, sim, cfg = _collect(args, tcfg, model)
    finally:
        bot_policy.targeting = _ORIG["targeting"]
        bot_policy.all_bot_intents = _ORIG["all_bot_intents"]
    print(f"[probe] {args.mode} mode, tier {args.tier}: {time.time() - t0:.0f} s of rollouts")
    _report(episodes, sim, cfg, tcfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
