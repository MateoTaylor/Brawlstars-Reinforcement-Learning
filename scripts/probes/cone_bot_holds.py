"""How often do Buzz and Edgar hold fire against the hero in contact, in the env training runs?
(BULL_HERO_PLAN.md §5.5, "Found in passing")

`combat_rules`' cone gate tests `ent_facing`, which `movement.apply_movement` copies from a bot's
smoothed steering whenever it is nonzero, and bodies never block each other. So a cone bot that is
not walking at its target has it outside its cone, and holds fire. This probe counts how often that
happens in the training env: configs/train.yaml as `training/builder.build_env` makes it (the map
pool, the roster weights, randomization.yaml, the gas pace, 3700-tick episodes), at one
curriculum stage's tier mixture.

Hooks, all read-only: `combat_rules.combat`, `personality.fire_allowed`,
`policy.all_bot_intents` and `core.combat.dominant_attacker` are wrapped through their module
attributes. The combat wrapper recomputes the rule's gates without drawing from the RNG, and raises
if their AND differs from the rule's own `fire`. Every combat damage source hands its attacker x
victim matrix to `dominant_attacker`, so the damage a bot deals the hero is exact, not the last
hitter's. With the hooks installed and the tick hook recording, hero_seam_digest.py still gives
both of its baselines.

  collect  one rollout of `--decisions` per env (740 is one whole match): `--hero`, `--driver`,
           `--stage`. Writes one .npz of per-tick rows, every alive Buzz or Edgar while the hero
           plays, to --out.
             policy  --run's best_model.zip, the checkpoint configs/deployment.yaml serves, its
                     actions SAMPLED as in training. Its best_vecnormalize.pkl is loaded too: a
                     pass-through for the deploy6 run, which trained on raw obs. For Bull it is a
                     stand-in: no Bull policy exists, and the D1 row swap keeps the obs widths.
             random  uniform over each action column's unmasked values: an untrained policy.
  report   reads every .npz in --out. Per hero x driver, the stages pool by their share of the
           400M run (83M each for stages 0-3, 68M for the elite stage); the stall spreads and the
           breakdowns pool unweighted.

  contact  the bot's effective target is the hero, not a crate, within its attack_range
  ready    contact, on its decision tick, with every other gate open: ammo, cooldown, reaction,
           line of sight, fire range, the lateral hold, the camper veto, not stunned
  held     ready, with the hero outside the cone. Held TIME counts each held decision tick as
           its decision period, so it is comparable across tiers; the share of ready ticks is not
           (a stalled bot is ready on every decision tick, a swinging one waits out its cooldown).
  stall    consecutive held ticks with no swing between, timed first to last plus one period

The same counts against other bots ("vs other bots") use the bot's target when that is a bot.

Usage (from the repo root, the venv's python, CPU only -- the GPU is the trainer's). The plan's
numbers are stages 0-4 with `--driver policy` for each hero, plus `--driver random` at stage 0,
each at `--seed` 0 and 1: 24 collects of about 7 min each, which can run side by side at the
default `--threads 2` (8 at a time on 16 cores). The report pools every .npz in --out, so it pools
the seeds too.

    .venv/Scripts/python.exe scripts/probes/cone_bot_holds.py collect --hero mortis --stage 4
    .venv/Scripts/python.exe scripts/probes/cone_bot_holds.py collect --hero bull --driver random
    .venv/Scripts/python.exe scripts/probes/cone_bot_holds.py report

Writes only under --out (default runs/probes/cone_bot_holds, ignored with the rest of runs/).
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brawl_sim.bots import combat_rules, personality                    # noqa: E402
from brawl_sim.bots import policy as bot_policy                         # noqa: E402
from brawl_sim.bots.personality import Mode                             # noqa: E402
from brawl_sim.constants import Kind, Person                            # noqa: E402
from brawl_sim.core import combat as core_combat                        # noqa: E402
from brawl_sim.core import geometry as geo                              # noqa: E402
from brawl_sim.core import stats                                        # noqa: E402
from brawl_sim.training.builder import build_env                        # noqa: E402
from brawl_sim.training.config import load_train_config                 # noqa: E402

KINDS = {int(Kind.BOT_MELEE): "Buzz", int(Kind.BOT_EDGAR): "Edgar"}
STAGE_DECISIONS = {0: 83, 1: 83, 2: 83, 3: 83, 4: 68}   # millions, configs/train.yaml's 400M run
STILL = 0.05                                            # tiles/s
_ORIG = {"combat": combat_rules.combat, "fire_allowed": personality.fire_allowed,
         "intents": bot_policy.all_bot_intents, "dominant": core_combat.dominant_attacker}
_CAPTURE = {}


def _install(on: bool) -> None:
    combat_rules.combat = _combat if on else _ORIG["combat"]
    personality.fire_allowed = _fire_allowed if on else _ORIG["fire_allowed"]
    bot_policy.all_bot_intents = _intents if on else _ORIG["intents"]
    core_combat.dominant_attacker = _dominant_attacker if on else _ORIG["dominant"]


def _combat(state, tgt, bank, params, cfg, gen):
    fire, aim_dir, aim_point = _ORIG["combat"](state, tgt, bank, params, cfg, gen)
    kind, pos = state.ent_kind, state.ent_pos
    attack_range = stats.gather_kind(params.attack_range, kind)
    los_ok = tgt.los | ~(stats.gather_kind(params.fire_needs_los, kind) > 0)
    frac = stats.gather_kind(params.fire_range_fraction, kind)
    range_ok = tgt.dist <= torch.where(frac > 0, frac, torch.ones_like(frac)) * attack_range
    arc = stats.gather_kind(params.attack_arc_rad, kind)
    half = (stats.gather_kind(params.hitscan_sweep_rad, kind) + arc) / 2.0
    cone_ok = geo.point_in_cone(tgt.pos, pos, state.ent_facing, attack_range, half) | (arc <= 0)
    lateral = (tgt.vel * geo.perp(geo.normalize(tgt.pos - pos))).sum(dim=-1).abs()
    limit = stats.gather_kind(params.fire_lateral_speed_limit, kind)
    lateral_hold = ((limit > 0) & (lateral > limit) & (
        tgt.dist > stats.gather_kind(params.fire_lateral_hold_range_fraction, kind) * attack_range))
    gate = bot_policy.fire_gate(state, tgt.dist, tgt.has_target, attack_range)
    if not torch.equal(gate & los_ok & range_ok & cone_ok & ~lateral_hold, fire):
        raise AssertionError("the recomputed gates do not AND to combat_rules.combat's fire")
    E = pos.shape[1]
    period = torch.clamp(stats.gather_kind(params.decision_period, kind), min=1)
    step = state.step_count.unsqueeze(-1).to(torch.int64)
    to_target = tgt.pos - pos
    off = torch.remainder(state.ent_facing - torch.atan2(to_target[..., 1], to_target[..., 0])
                          + math.pi, 2 * math.pi) - math.pi
    _CAPTURE["pre"] = {
        "kind": kind.clone(), "alive": state.ent_alive.clone(), "person": state.ent_person.clone(),
        "tgt_hero": tgt.has_enemy & (tgt.idx == 0) & ~tgt.is_box,
        "tgt_bot": tgt.has_enemy & (tgt.idx != 0) & ~tgt.is_box,
        "dist": tgt.dist.clone(), "has_target": tgt.has_target.clone(),
        "ammo_ok": state.ent_ammo >= 1.0, "cd_ok": state.ent_attack_cd <= 0,
        "react_ok": state.ent_react_t <= 0, "in_range": tgt.dist <= attack_range,
        "los_ok": los_ok, "range_ok": range_ok, "cone_ok": cone_ok, "lat_ok": ~lateral_hold,
        "comb_fire": fire.clone(), "stunned": (state.ent_stun_t > 0) | (state.ent_airborne_t > 0),
        "dp": period.clone(),
        "dtick": ((step + torch.arange(E, dtype=torch.int64).view(1, E)) % period) == 0,
        "face_off": off, "speed": state.ent_vel.norm(dim=-1),
        "shots0": state.ent_shots_fired.clone(),
    }
    return fire, aim_dir, aim_point


def _fire_allowed(state, tgt, aggression, cfg):
    allowed = _ORIG["fire_allowed"](state, tgt, aggression, cfg)
    _CAPTURE["veto_ok"] = allowed.clone()
    return allowed


def _intents(*args, **kwargs):
    intent = _ORIG["intents"](*args, **kwargs)
    _CAPTURE["fire"], _CAPTURE["mode"] = intent.fire.clone(), intent.mode.clone()
    return intent


def _dominant_attacker(dmg_by):
    # Called once per combat damage source per tick (env.py: melee, dash, projectiles) with
    # (N,E,E) attacker x victim damage; victim column 0 is the hero.
    hit = dmg_by[:, :, 0]
    _CAPTURE["hit_hero"] = hit.clone() if "hit_hero" not in _CAPTURE else _CAPTURE["hit_hero"] + hit
    return _ORIG["dominant"](dmg_by)


class Recorder:
    """One row per tick per alive Buzz/Edgar while the hero plays, read at the end of the tick
    (`BrawlVecEnv.tick_hook`) from what the wrappers caught in phase 4 of the same tick."""

    def __init__(self, n_envs):
        self.rows = defaultdict(list)
        self.episode = np.zeros(n_envs, np.int32)
        self.tick = self.hero_ticks = self.missing = 0

    def on_tick(self, env):
        self.tick += 1
        pre, hit_hero = _CAPTURE.pop("pre", None), _CAPTURE.pop("hit_hero", None)
        if pre is None or hit_hero is None:
            self.missing += 1
            return
        s, alive, kind = env.state, pre["alive"], pre["kind"]
        playing = alive[:, 0] & (alive.sum(dim=1) >= 2)
        self.hero_ticks += int(playing.sum())
        cone_kind = torch.zeros_like(alive)
        for kid in KINDS:
            cone_kind |= kind == kid
        rows = alive & cone_kind & playing.unsqueeze(1)
        if not bool(rows.any()):
            return
        n_idx, e_idx = torch.nonzero(rows, as_tuple=True)
        cols = {key: pre[key][rows] for key in (
            "kind", "person", "tgt_hero", "tgt_bot", "dist", "has_target", "ammo_ok", "cd_ok",
            "react_ok", "in_range", "los_ok", "range_ok", "cone_ok", "lat_ok", "comb_fire",
            "stunned", "dp", "dtick", "face_off")}
        cols.update({
            "n": n_idx, "e": e_idx, "mode": _CAPTURE["mode"][rows], "fire": _CAPTURE["fire"][rows],
            "veto_ok": _CAPTURE["veto_ok"][rows], "bot_speed": pre["speed"][rows],
            "hero_speed": pre["speed"][:, 0:1].expand_as(rows)[rows],
            "shots": (s.ent_shots_fired - pre["shots0"])[rows],
            # What this bot dealt the hero this tick: its entry in the melee, dash and projectile
            # phases' attacker x victim matrices, summed. Overkill counts, as in each phase's
            # dmg_ent; the gas has no attacker and never enters.
            "hero_hit": hit_hero[rows],
        })
        for key, value in cols.items():
            self.rows[key].append(value.numpy())
        self.rows["ep"].append(self.episode[n_idx.numpy()])
        self.rows["tick"].append(np.full(len(n_idx), self.tick, np.int32))


def _random_action(masks, nvec, rng):
    out = np.zeros((masks.shape[0], len(nvec)), np.int64)
    start = 0
    for column, n in enumerate(nvec):
        out[:, column] = (rng.random((masks.shape[0], n)) * masks[:, start:start + n]).argmax(axis=1)
        start += n
    return out


def collect(args) -> int:
    torch.set_num_threads(args.threads)
    run = {"device": "cpu", "n_envs": args.envs, "seed": args.seed, "verbose": 0}
    if args.hero != "mortis":
        run["env_overrides"] = {"entities": {"hero": args.hero}}
    # batch_size only has to fit one rollout for validation; nothing here trains.
    tcfg = load_train_config(args.train_config,
                             {"run": run, "ppo": {"batch_size": min(8192, 128 * args.envs)}})
    # No fresh VecNormalize (build_env's would be a training one): the policy driver loads the
    # run's saved statistics below.
    tcfg = replace(tcfg, normalize=replace(tcfg.normalize, obs=False, reward=False))
    venv, parts = build_env(tcfg, n_envs=args.envs, verbose=False)
    sim, curriculum, cfg = parts["sim"], parts["curriculum"], parts["env_cfg"]
    if cfg.hero != args.hero:
        raise AssertionError(f"the env plays {cfg.hero}, not {args.hero}")
    curriculum.set_stage(args.stage)
    model = None
    if args.driver == "policy":
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import VecNormalize
        venv = VecNormalize.load(str(args.run / "best_vecnormalize.pkl"), venv)
        venv.training, venv.norm_reward = False, False
        model = MaskablePPO.load(str(args.run / "best_model.zip"), device="cpu")
    # After the load: MaskablePPO.load reseeds the global RNGs with the run's own training seed.
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    rec = Recorder(args.envs)
    _install(True)
    # Each env's FIRST match, as TierEvaluator counts them: every one ends inside a whole match's
    # decisions, where the matches after it count only if they end early, which favours deaths.
    first = {"won": [None] * args.envs, "rank": [None] * args.envs, "decisions": [None] * args.envs}
    length = np.zeros(args.envs, np.int64)
    episodes = 0
    t0 = time.time()
    try:
        obs = venv.reset()                       # re-spawns every env under the stage's tiers
        _CAPTURE.clear()                         # hit_hero sums, so nothing may carry over
        sim.tick_hook = rec.on_tick
        for _ in range(args.decisions):
            masks = venv.action_masks()
            if model is not None:
                action, _ = model.predict(obs, deterministic=False, action_masks=masks)
            else:
                action = _random_action(masks, venv.action_space.nvec, rng)
            obs, _, dones, infos = venv.step(action)
            length += 1
            for i in np.flatnonzero(dones):
                episodes += 1
                if rec.episode[i] == 0:
                    first["won"][i] = bool(infos[i]["outcome"]["won"])
                    first["rank"][i] = int(infos[i]["outcome"]["rank"])
                    first["decisions"][i] = int(length[i])
            length[dones] = 0
            rec.episode[dones] += 1
    finally:
        sim.tick_hook = None
        _install(False)
    if rec.missing:
        raise AssertionError(f"{rec.missing} ticks ran without the combat or damage wrapper")
    meta = {"hero": args.hero, "driver": args.driver, "stage": args.stage,
            "stage_name": curriculum.stage.name, "envs": args.envs, "decisions": args.decisions,
            "seed": args.seed, "dt": cfg.dt, "hero_ticks": rec.hero_ticks,
            "episodes_finished": episodes, "first_won": first["won"],
            "first_rank": first["rank"], "first_decisions": first["decisions"],
            "tier_spawns": curriculum.tier_spawn_fractions(reset=False),
            "seconds": round(time.time() - t0, 1)}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.hero}_{args.driver}_s{args.stage}_seed{args.seed}.npz"
    np.savez_compressed(path, meta=json.dumps(meta),
                        **{key: np.concatenate(value) for key, value in rec.rows.items()})
    print(json.dumps(meta))
    return 0


# ------------------------------------------------------------------------------------------

def _ready(r, target):
    return (target & r["in_range"] & r["dtick"] & r["ammo_ok"] & r["cd_ok"] & r["react_ok"]
            & r["los_ok"] & r["range_ok"] & r["lat_ok"] & r["veto_ok"] & ~r["stunned"])


def _masks(r, k, target):
    """Contact, ready and held rows of the kind mask `k` against `target`."""
    ready = k & _ready(r, target)
    return k & target & r["in_range"], ready, ready & ~r["cone_ok"]


def _stalls(r, target, dt):
    """Seconds per stall: one bot's consecutive held ticks in unbroken contact with `target`,
    ended by a fire decision, a shot, or a gap in contact."""
    ready = _ready(r, target)
    held, fired = ready & ~r["cone_ok"], ready & r["cone_ok"]
    key = r["n"].astype(np.int64) * 1_000_000 + r["ep"].astype(np.int64) * 100 + r["e"]
    idx = np.flatnonzero(target & r["in_range"])
    out, start, last, last_i, prev = [], None, None, None, (None, None)
    for i in idx[np.lexsort((r["tick"][idx], key[idx]))]:
        k, t = key[i], r["tick"][i]
        if (k, t - 1) != prev or fired[i] or r["shots"][i] > 0:
            if start is not None:
                out.append((last - start + r["dp"][last_i]) * dt)
            start = None
        if held[i]:
            start = t if start is None else start
            last, last_i = t, i
        prev = (k, t)
    if start is not None:
        out.append((last - start + r["dp"][last_i]) * dt)
    return np.asarray(out)


def _tally(meta, r):
    """Counts, per kind, for one run: summable across runs."""
    dt = meta["dt"]
    still, d = r["hero_speed"] < STILL, r["dist"]
    out = {"_minutes": meta["hero_ticks"] * dt / 60.0}
    for kid, name in KINDS.items():
        k = r["kind"] == kid
        c = {"bots": len(set(zip(r["n"][k], r["ep"][k], r["e"][k])))}
        for side, target in (("hero", r["tgt_hero"]), ("bots", r["tgt_bot"])):
            contact, ready, held = _masks(r, k, target)
            c[f"{side}_contact_s"] = contact.sum() * dt
            c[f"{side}_ready"] = ready.sum()
            c[f"{side}_held"] = held.sum()
            c[f"{side}_held_s"] = (r["dp"][held] * dt).sum()
        contact, ready, held = _masks(r, k, r["tgt_hero"])
        c["still_s"] = (contact & still).sum() * dt
        for label, m in (("still", still), ("moving", ~still), ("d0", d < 0.5),
                         ("d1", (d >= 0.5) & (d < 1.0)), ("d2", d >= 1.0),
                         ("parked", r["bot_speed"] < STILL)):
            c[f"ready_{label}"] = (ready & m).sum()
            c[f"held_{label}"] = (held & m).sum()
        c["shots"] = r["shots"][contact].sum()
        c["damage"] = r["hero_hit"][contact].sum()
        c["damage_still"] = r["hero_hit"][contact & still].sum()
        out[name] = c
    return out


def _first(meta):
    """Each env's first match: (recorded, won, mean place). The info's rank is 0-indexed, the
    place is not: 1 is the win."""
    won = [w for w in meta["first_won"] if w is not None]
    places = [r + 1 for r in meta["first_rank"] if r is not None]
    return len(won), sum(won), sum(places) / len(places) if places else float("nan")


def _pct(a, b):
    return f"{100.0 * a / b:.1f} %" if b else "-"


def _spread(a):
    return f"{np.median(a):.1f} / {np.percentile(a, 90):.1f} / {a.max():.1f} s" if a.size else "-"


def report(args) -> int:
    runs = []
    for path in sorted(args.out.glob("*.npz")):
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        r = {key: z[key] for key in z.files if key != "meta"}
        gate = r["ammo_ok"] & r["cd_ok"] & r["react_ok"] & r["has_target"] & r["in_range"]
        if not np.array_equal(gate & r["los_ok"] & r["range_ok"] & r["cone_ok"] & r["lat_ok"],
                              r["comb_fire"]) or not np.array_equal(
                r["comb_fire"] & r["veto_ok"] & r["dtick"], r["fire"]):
            raise AssertionError(f"{path.name}: the recorded gates do not reproduce the fire rule")
        runs.append((meta, r))
    if not runs:
        print(f"no .npz in {args.out}")
        return 1
    groups = defaultdict(list)
    for meta, r in runs:
        groups[(meta["hero"], meta["driver"])].append((meta, r))
    labels = ["bots seen", "contact, s per hero-minute", "hero still, share of contact",
              "held time / contact time", "held, share of ready ticks",
              "  hero still / moving", "  distance <0.5 / 0.5-1 / >=1",
              "  bot itself parked, share of held", "  |facing - bearing| median",
              "stalls median / p90 / longest", "stalls >= 3 s",
              "swings per contact-second", "hero damage per contact-second, HP",
              "  hero still / moving",
              "vs other bots: held time / contact", "vs other bots: stalls", "vs other bots: >= 10 s"]
    for (hero, driver), items in sorted(groups.items()):
        per_stage = defaultdict(int)
        for m, _ in items:
            per_stage[m["stage"]] += 1
        total = sum(STAGE_DECISIONS[s] for s in per_stage)
        weights = [len(items) * STAGE_DECISIONS[m["stage"]] / total / per_stage[m["stage"]]
                   for m, _ in items]
        tallies = [_tally(m, r) for m, r in items]
        minutes = sum(w * t["_minutes"] for w, t in zip(weights, tallies))
        firsts = [_first(m) for m, _ in items]
        win_rate = sum(w * won / n for w, (n, won, _) in zip(weights, firsts) if n) / len(items)
        place = sum(w * mean for w, (_, _, mean) in zip(weights, firsts)) / len(items)
        print(f"\n=== {hero} / {driver}: stages {sorted(per_stage)}, {minutes:.0f} hero-minutes; "
              f"first matches won {100 * win_rate:.1f} % ({sum(f[1] for f in firsts)} of "
              f"{sum(f[0] for f in firsts)}), mean place {place:.2f}")
        cols = {}
        for kid, name in KINDS.items():
            c = defaultdict(float)
            for w, t in zip(weights, tallies):
                for key, value in t[name].items():
                    c[key] += w * value
            subs = [(m, {key: value[r["kind"] == kid] for key, value in r.items()}) for m, r in items]
            stalls = np.concatenate([_stalls(s, s["tgt_hero"], m["dt"]) for m, s in subs])
            bot_stalls = np.concatenate([_stalls(s, s["tgt_bot"], m["dt"]) for m, s in subs])
            off = np.concatenate([np.abs(np.degrees(s["face_off"][_ready(s, s["tgt_hero"])
                                                                  & ~s["cone_ok"]]))
                                  for _, s in subs])
            contact_s = max(c["hero_contact_s"], 1e-9)
            still_s = max(c["still_s"], 1e-9)
            moving_s = max(c["hero_contact_s"] - c["still_s"], 1e-9)
            cols[name] = [
                f"{c['bots']:.0f}",
                f"{c['hero_contact_s'] / minutes:.2f}",
                _pct(c["still_s"], c["hero_contact_s"]),
                _pct(c["hero_held_s"], c["hero_contact_s"]),
                f"{_pct(c['hero_held'], c['hero_ready'])} of {c['hero_ready']:.0f}",
                f"{_pct(c['held_still'], c['ready_still'])} / {_pct(c['held_moving'], c['ready_moving'])}",
                " / ".join(_pct(c[f"held_d{i}"], c[f"ready_d{i}"]) for i in range(3)),
                _pct(c["held_parked"], c["hero_held"]),
                f"{np.median(off):.0f} deg" if off.size else "-",
                _spread(stalls), f"{(stalls >= 3.0).sum()}",
                f"{c['shots'] / contact_s:.2f}",
                f"{c['damage'] / contact_s:.0f}",
                f"{c['damage_still'] / still_s:.0f} / {(c['damage'] - c['damage_still']) / moving_s:.0f}",
                _pct(c["bots_held_s"], c["bots_contact_s"]),
                _spread(bot_stalls), f"{(bot_stalls >= 10.0).sum()}",
            ]
        print(f"  {'':36s}" + "".join(f"{name:>30s}" for name in KINDS.values()))
        for i, label in enumerate(labels):
            print(f"  {label:36s}" + "".join(f"{cols[name][i]:>30s}" for name in KINDS.values()))

    print("\nHeld time / hero contact time, and each env's first match, per run:")
    for meta, r in runs:
        t = _tally(meta, r)
        n, won, place = _first(meta)
        print(f"  {meta['hero']:6s} {meta['driver']:6s} stage {meta['stage']} seed {meta['seed']}  "
              + "   ".join(
            f"{name} {_pct(t[name]['hero_held_s'], t[name]['hero_contact_s'])} of "
            f"{t[name]['hero_contact_s']:.0f} s" for name in KINDS.values())
            + f"   | won {won} of {n}, mean place {place:.2f}")

    print("\nHeld share of hero-contact ready ticks by personality, steering mode and decision "
          "period (stages pooled unweighted, cells of 20+ ready ticks):")
    cells = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for meta, r in runs:
        ready = _ready(r, r["tgt_hero"])
        for kid, name in KINDS.items():
            for j in np.flatnonzero(ready & (r["kind"] == kid)):
                for label in (Person(int(r["person"][j])).name, Mode(int(r["mode"][j])).name,
                              f"period {float(r['dp'][j]):g}"):
                    cell = cells[(meta["hero"], meta["driver"], name)][label]
                    cell[0] += 1
                    cell[1] += int(not r["cone_ok"][j])
    for key, by in sorted(cells.items()):
        print(f"  {' / '.join(key)}: " + ", ".join(
            f"{label} {_pct(h, n)} of {n}" for label, (n, h) in sorted(by.items()) if n >= 20))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("collect", "report"))
    p.add_argument("--hero", choices=("mortis", "bull"), default="mortis")
    p.add_argument("--driver", choices=("policy", "random"), default="policy")
    p.add_argument("--stage", type=int, default=0, help="curriculum stage index, 0-4")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--decisions", type=int, default=740, help="per env; 740 is one whole match")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--train-config", type=Path, default=ROOT / "configs" / "train.yaml")
    p.add_argument("--run", type=Path, default=ROOT / "runs" / "mortis_ppo-20261012-082348",
                   help="the policy driver's run folder")
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "probes" / "cone_bot_holds")
    args = p.parse_args(argv)
    return collect(args) if args.mode == "collect" else report(args)


if __name__ == "__main__":
    raise SystemExit(main())
