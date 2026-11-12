"""Bit-exact fingerprint of a configs/train.yaml rollout: the check that a Bull change left Mortis
unchanged (BULL_HERO_PLAN.md, every phase's "the Mortis digest matches").

Builds the training env (CPU, 64 envs, seed 0), steps 100 decisions (500 ticks) of seeded random
actions and hashes the obs, reward and done of every step, then every SimState and SimParams tensor
at the end. Two runs of the same code give the same file, so any difference is a behaviour change.
It also counts non-finite obs values, which is how a Mortis-only field divided by Bull's zeros
would show up under `--hero bull`.

Usage (from the repo root, the venv's python, CPU only -- the GPU is the trainer's):

    .venv/Scripts/python.exe scripts/probes/hero_seam_digest.py --out before.json
    # ... change the code ...
    .venv/Scripts/python.exe scripts/probes/hero_seam_digest.py --out after.json --compare before.json
    .venv/Scripts/python.exe scripts/probes/hero_seam_digest.py --out bull.json --hero bull

Take the baseline from the code BEFORE the change: the digest pins whatever the code does, so a
baseline from a tree that already has the change cannot fail. Mortis's digest was
d53b1c4140510e7d through Phase 3; the zero-width cone fix (2026-10-23, BULL_HERO_PLAN.md §5.3)
moved it on purpose to 5df44a1488f7da5e, and Bull's to 7c781a0dd35248c6. Phase 4's charge super
left Mortis's unchanged and moved Bull's to ec7997b246e9d0af. Phase 5's Stomper and Berserker left
Mortis's unchanged again and moved Bull's to 6ab2df8a244c9861. With that kit switched off
(`hero_bull`'s `gadget_cooldown` and `reload_boost_below_hp_frac` zeroed through
`run.env_overrides`), the same rollout still gives ec7997b246e9d0af. Snapping `ent_attack_cd` to
whole ticks (2026-10-24, the user's call) moved both on purpose, through every 0.25-s cooldown:
Mortis's to 247884ee3f0cf382 and Bull's to 483861f81de74216. With the old plain clamp patched back
over that one line, the same rollout gives the two before it bit for bit. Phase 6a's slow window
(`ent_slow_window_t`, 2026-10-27) left both unchanged; its new tensor is listed. Setting
`hero_bull.reload_seconds` to the measured 1.6 (2026-10-28, the user's call; `bot_bull` keeps
2.0) left Mortis's unchanged and moved Bull's to 367cb1d78c40591e. Setting `hero_bull`'s
`attack_range` 5.2, `proj_speed` 9.0 and `super_knockback_tiles` 2.5 (2026-11-01, measured, the
user's call; `bot_bull` keeps 7.0 and 6.4) left Mortis's unchanged and moved Bull's to
82bc81256f947e00. With the three old values put back through `run.env_overrides`, the same rollout
gives 367cb1d78c40591e bit for bit. Giving `bot_bull` the same range and pellet speed (the same
day, the user's call) moved both on purpose: Mortis's to 5b469682fe6833dc and Bull's to
285f0ed3e35482ee. With `bot_bull`'s 7.0 and 6.4 put back the same way, they give 247884ee3f0cf382
and 82bc81256f947e00 bit for bit, and with `projectiles.box_cells` swapped back to a plain ceil
(25 box candidates per shot instead of 9), the new two again. Setting both blocks' `move_speed`
2.68 (measured; `bot_bull` was 2.17) and `hero_bull`'s `stomp_windup_seconds` 0.30 and
`stomp_stun_seconds` 1.4 (measured; the wiki's 0.25 and 1.0), the same day and the user's call,
moved both on purpose: Mortis's to 48e1a2b59a31e97d and Bull's to 5d4250ff5c7ac420. With the
four old values put back through `run.env_overrides`, they give 5b469682fe6833dc and
285f0ed3e35482ee bit for bit. The miss penalties (2026-11-02, the user's call; BULL_HERO_PLAN.md
§5.8) left the sim alone: Mortis's stays 48e1a2b59a31e97d with no tensor moved, and Bull's moves
through its reward only, to b8b6b69df80c322d. With `reward.attack_miss` and `reward.gadget_miss`
zeroed, the same rollout gives 5d4250ff5c7ac420 bit for bit. `state.hero_volley_hit` is listed.
Raising `attack_miss` to -0.10 (2026-11-03, the user's call) left Mortis's alone again and moved
Bull's to 21c617decff340fb, no tensor moved; -0.05 put back gives b8b6b69df80c322d bit for bit.

A tensor the baseline does not have (a new SimState field, such as Phase 3's status timers) is
listed, not compared: the run still counts as identical only if every step and every tensor the
two digests share match. A tensor the baseline has and this run lacks is a difference.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brawl_sim.training.builder import build_env              # noqa: E402
from brawl_sim.training.config import load_train_config       # noqa: E402

N_ENVS = 64
N_DECISIONS = 100


def _h(x) -> str:
    if isinstance(x, dict):
        return hashlib.sha256("".join(f"{k}:{_h(v)}" for k, v in sorted(x.items())).encode()).hexdigest()[:16]
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


def _tensors(obj) -> dict:
    names = vars(obj) if hasattr(obj, "__dict__") else obj.__slots__
    return {k: _h(getattr(obj, k)) for k in names if isinstance(getattr(obj, k, None), torch.Tensor)}


def digest(hero: str | None) -> dict:
    overrides = {"run": {"device": "cpu", "n_envs": N_ENVS, "seed": 0, "verbose": 0}}
    if hero:
        overrides["run"]["env_overrides"] = {"entities": {"hero": hero}}
    tcfg = load_train_config(ROOT / "configs" / "train.yaml", overrides)
    venv, parts = build_env(tcfg, n_envs=N_ENVS, verbose=False)
    sim = parts["sim"]

    rng = np.random.default_rng(0)
    obs = venv.reset()
    steps, nonfinite = [_h(obs)], 0
    for _ in range(N_DECISIONS):
        act = np.stack([rng.integers(0, n, size=N_ENVS) for n in venv.action_space.nvec], axis=1)
        obs, rew, done, _info = venv.step(act)
        nonfinite += sum(int((~np.isfinite(v)).sum()) for v in obs.values() if v.dtype.kind == "f")
        steps.append(_h({"obs": _h(obs), "rew": _h(rew), "done": _h(done)}))
    return {"hero": hero or "mortis", "steps": steps, "state": _tensors(sim.state),
            "params": _tensors(sim.params), "nonfinite_obs": nonfinite,
            "obs_shapes": {k: list(v.shape) for k, v in obs.items()}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--hero", default=None, help="entities.hero override (default: the config's)")
    p.add_argument("--compare", type=Path, default=None, help="a baseline digest to diff against")
    args = p.parse_args()

    d = digest(args.hero)
    args.out.write_text(json.dumps(d, indent=1))
    print(f"{args.out}: hero {d['hero']}, {len(d['steps'])} steps, {len(d['state'])} state + "
          f"{len(d['params'])} params tensors, {d['nonfinite_obs']} non-finite obs, final {d['steps'][-1]}")
    if args.compare is None:
        return 0
    base = json.loads(args.compare.read_text())
    if base.get("hero", "mortis") != d["hero"]:
        print(f"note: comparing hero {d['hero']} against a {base.get('hero', 'mortis')} baseline")
    first = next((i for i, (a, b) in enumerate(zip(base["steps"], d["steps"])) if a != b), None)
    groups = ("state", "params")
    moved = [f"{g}.{k}" for g in groups for k in sorted(set(base[g]) & set(d[g])) if base[g][k] != d[g][k]]
    gone = [f"{g}.{k}" for g in groups for k in sorted(set(base[g]) - set(d[g]))]
    added = [f"{g}.{k}" for g in groups for k in sorted(set(d[g]) - set(base[g]))]
    if added:
        print(f"new since the baseline, not compared: {added}")
    if first is None and not moved and not gone and len(base["steps"]) == len(d["steps"]):
        print("BIT-IDENTICAL")
        return 0
    print(f"DIFFERENT: first step {first}; {len(moved)} tensors moved: {moved[:12]}; gone: {gone[:12]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
