"""Wall-push measurement (BRAWL_DEPLOYMENT_DESIGN.md §6.16) for a checkpoint under the CURRENT sim.

Rolls a checkpoint in the sim over four cells -- {training maps, holdout maps} x {dead-bin mask
off, on} -- 96 elite episodes per cell by default, deterministic as deployed, on the CPU, and
reports the §6.16 table: the fraction of decisions whose chosen move bin cannot move the hero
(a WALL-PUSH decision), how many runs of those last 2 s or more per episode (a STALL), the
longest such run, and the win rate.

Two masks, one truth. The truth for "this bin is dead" is the sim's own collision rule,
`terrain.resolve_move` on `bank.blocks_unit` with the hero's real radius and one tick of its real
speed, evaluated for all 16 bins at once (`core/movement.py` walks with exactly that call). The
mask the "on" cells AND into the policy's action mask is the DEPLOYED one, `move_mask.legal_move_bins`
on the agent's own `blocks_unit` grid plane with `hero_pos - observation._view_origin` as the
position, because that is what the BlueStacks loop hands the policy (`loop.py`, `policy.dead_bin_mask`).
Every decision also records whether the two agree, so the table doubles as a check of the deployed
mask against the sim it was written to mirror.

A decision counts as a wall-push only while the hero is alive and not mid-dash (movement is skipped
in both, `apply_movement`'s `active`), the chosen bin is not idle, and the truth says the bin is
dead. Stalls are maximal runs of consecutive wall-push decisions of at least `2.0 / cfg.agent_dt`
decisions (8 at the default 4 Hz), counted once per run.

Usage (from the repo root, the venv's python, CPU only -- the GPU is the trainer's):

    .venv/Scripts/python.exe scripts/probes/wall_push_measure.py
    .venv/Scripts/python.exe scripts/probes/wall_push_measure.py --run runs/<deploy5 run> --episodes 96
    .venv/Scripts/python.exe scripts/probes/wall_push_measure.py --cells training:off holdout:on

Reads the run's `train.yaml`, `best_model.zip` (or `--checkpoint`) and the agent spec it names;
writes nothing.
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brawl_deployment.move_mask import legal_move_bins                 # noqa: E402
from brawl_sim.core import geometry as geo, observation, stats, terrain   # noqa: E402
from brawl_sim.core import obs_select                                  # noqa: E402
from brawl_sim.training.builder import _resolve                        # noqa: E402
from brawl_sim.training.config import load_train_config                # noqa: E402
from brawl_sim.training.evaluation import TierEvaluator                # noqa: E402
from sb3_contrib import MaskablePPO                                    # noqa: E402

_HERO = 0
_MOVED_TILES = 1e-6          # `move_mask._MOVED_TILES`: below this a float32 slide is noise
_DEFAULT_RUN = "runs/mortis_deploy4-20260921-185945"
_CELLS = ("training:off", "training:on", "holdout:off", "holdout:on")


def _blocks_plane(tcfg, env_cfg) -> int:
    """Index of the `blocks_unit` plane inside the agent's grid group -- what `loop.py` calls
    `self._blocks_plane`. Read from the run's own spec so a spec that reorders channels cannot
    silently point this at another plane."""
    spec = obs_select.load_agent_spec(_resolve(tcfg.run.agent_obs), env_cfg)
    for g in spec.groups:
        if g.view_channels is not None:
            if "blocks_unit" not in g.view_channels:
                raise ValueError(f"grid group {g.name!r} has no blocks_unit plane: {g.view_channels}")
            return g.view_channels.index("blocks_unit")
    raise ValueError("the agent spec has no grid group")


def _truth_legal(sim, cfg, dirs16: torch.Tensor) -> torch.Tensor:
    """(N, n_move_bins + 1) bool -- the sim's own answer to "does this bin move the hero one
    tick", idle always True. Same call, radius and step as `core/movement.apply_movement`."""
    st, params = sim.state, sim.params
    hero_pos = st.ent_pos[:, _HERO]                                         # (N,2)
    speed = stats.effective_speed(st.ent_kind, params)[:, _HERO]            # (N,)
    delta = dirs16.unsqueeze(0) * (speed * cfg.dt)[:, None, None]           # (N,16,2)
    pos = hero_pos.unsqueeze(1).expand(-1, dirs16.shape[0], -1)             # (N,16,2)
    radius = params.unit_radius.unsqueeze(-1)                               # (N,1)
    final = terrain.resolve_move(sim.bank.blocks_unit, st.map_id, pos, delta, radius, cfg)
    moved = (final - pos).abs().amax(dim=-1) > _MOVED_TILES                 # (N,16)
    idle = torch.ones(moved.shape[0], 1, dtype=torch.bool, device=moved.device)
    return torch.cat([idle, moved], dim=1)


def _deployed_legal(sim, cfg, grid: np.ndarray, plane: int) -> np.ndarray:
    """(N, n_move_bins + 1) bool -- `legal_move_bins` per env on the agent's own grid plane,
    positioned the way `loop.py` positions it: `hero_pos - GridBuilder.origin_tile(hero_pos)`,
    which in the sim is `hero_pos - observation._view_origin`."""
    st, params = sim.state, sim.params
    hero_pos = st.ent_pos[:, _HERO]
    origin = observation._view_origin(st, cfg).to(hero_pos.dtype)
    hero_xy = (hero_pos - origin).cpu().numpy()                             # (N,2) (column,row)
    step = (stats.effective_speed(st.ent_kind, params)[:, _HERO] * cfg.dt).cpu().numpy()
    radius = params.unit_radius.cpu().numpy()
    out = np.ones((hero_xy.shape[0], cfg.n_move_bins + 1), dtype=bool)
    for i in range(hero_xy.shape[0]):
        out[i] = legal_move_bins(grid[i, plane], hero_xy[i], radius=float(radius[i]),
                                 step=float(step[i]), n_bins=cfg.n_move_bins)
    return out


def measure_cell(tcfg, model, *, maps, mask_on: bool, device: str, verbose: bool = True) -> dict:
    ev = TierEvaluator(tcfg, tiers=("elite",), device=device, maps=maps)
    sim, venv, cfg = ev.sim, ev.venv, ev.sim.cfg
    plane = _blocks_plane(tcfg, ev.env_cfg)
    n = ev.n_envs
    n_move = cfg.n_move_bins + 1
    stall_len = int(round(2.0 / cfg.agent_dt))
    dirs16 = geo.dir_from_bin(torch.arange(cfg.n_move_bins, device=sim.device), cfg.n_move_bins)

    sim.gen.manual_seed(ev.seed)          # the evaluator's own scenarios, identical per cell
    obs = venv.reset()

    recorded = np.zeros(n, dtype=bool)
    won = np.zeros(n, dtype=bool)
    rank = np.zeros(n, dtype=np.int64)
    length = np.zeros(n, dtype=np.int64)
    run_len = np.zeros(n, dtype=np.int64)
    longest = np.zeros(n, dtype=np.int64)
    stalls = np.zeros(n, dtype=np.int64)
    acc = dict(decisions=0, walking=0, wall_push=0, wall_push_wedged=0, changed=0, disagree=0,
               idle=0, masked_dead=0, wedged=0, stuck=0, wedge_onsets=0, wedge_onsets_dash=0)
    was_wedged = np.zeros(n, dtype=bool)
    since_dash = np.full(n, 10**6, dtype=np.int64)
    t0 = time.time()
    steps = 0
    for steps in range(1, ev.max_steps + 1):
        masks = venv.action_masks()                                         # (n, 21) bool
        live = ~recorded
        truth = _truth_legal(sim, cfg, dirs16).cpu().numpy()                # (n, 17)
        deployed = _deployed_legal(sim, cfg, obs["grid"], plane)            # (n, 17)
        st = sim.state
        dashing = (st.ent_dash_t[:, _HERO] > 0).cpu().numpy()
        walking = st.ent_alive[:, _HERO].cpu().numpy() & ~dashing & live
        # WEDGED: the hero's own footprint at full radius already overlaps a wall, which only a
        # dash can produce (`start_dash` marches the CENTRE, so a dash along a wall face can land
        # the body inside it); STUCK: every one of the 16 bins is dead, the sim's one-way trap.
        wedged = terrain.circle_blocked(sim.bank.blocks_unit, st.map_id, st.ent_pos[:, _HERO],
                                        sim.params.unit_radius, cfg).cpu().numpy() & live
        stuck = ~truth[:, 1:].any(axis=1) & live
        since_dash = np.where(dashing, 0, since_dash + 1)
        onset = wedged & ~was_wedged
        was_wedged = wedged

        plain, _ = model.predict(obs, deterministic=True, action_masks=masks)
        if mask_on:
            masked = masks.copy()
            masked[:, :n_move] &= deployed
            action, _ = model.predict(obs, deterministic=True, action_masks=masked)
        else:
            action = plain
        mb = action[:, 0]
        dead_choice = walking & (mb > 0) & ~truth[np.arange(n), mb]

        acc["decisions"] += int(live.sum())
        acc["walking"] += int(walking.sum())
        acc["wall_push"] += int(dead_choice.sum())
        acc["wall_push_wedged"] += int((dead_choice & wedged).sum())
        acc["wedged"] += int(wedged.sum())
        acc["stuck"] += int(stuck.sum())
        acc["wedge_onsets"] += int(onset.sum())
        acc["wedge_onsets_dash"] += int((onset & (since_dash <= 1)).sum())
        acc["changed"] += int((live & (action[:, 0] != plain[:, 0])).sum())
        acc["disagree"] += int((live & (truth != deployed).any(axis=1)).sum())
        acc["idle"] += int((walking & (mb == 0)).sum())
        # Decisions where the DEPLOYED mask would have vetoed the plain choice -- what the mask
        # changes on the live path, whichever cell this is.
        acc["masked_dead"] += int((walking & (plain[:, 0] > 0)
                                   & ~deployed[np.arange(n), plain[:, 0]]).sum())

        run_len = np.where(dead_choice, run_len + 1, 0)
        longest = np.maximum(longest, run_len)
        stalls += run_len == stall_len

        obs, _, dones, infos = venv.step(action)
        for i in np.nonzero(dones & ~recorded)[0]:
            recorded[i] = True
            won[i] = infos[i]["outcome"]["won"]
            rank[i] = infos[i]["outcome"]["rank"]
            length[i] = infos[i]["episode"]["l"]
        run_len[dones] = 0
        was_wedged[dones] = False
        since_dash[dones] = 10**6
        if recorded.all():
            break
        if verbose and steps % 100 == 0:
            print(f"    step {steps:4d}  live {int((~recorded).sum()):3d}  "
                  f"wall-push {acc['wall_push'] / max(acc['decisions'], 1):.4f}  "
                  f"{time.time() - t0:5.0f} s", flush=True)

    venv.close()
    dec = max(acc["decisions"], 1)
    return {
        "maps": "holdout" if maps else "training",
        "mask": "on" if mask_on else "off",
        "episodes": int(recorded.sum()),
        "unfinished": int((~recorded).sum()),
        "decisions": acc["decisions"],
        "wall_push_frac": acc["wall_push"] / dec,
        "wall_push": acc["wall_push"],
        "wall_push_free_frac": (acc["wall_push"] - acc["wall_push_wedged"]) / dec,
        "wall_push_wedged_frac": acc["wall_push_wedged"] / dec,
        "wedged_frac": acc["wedged"] / dec,
        "stuck_frac": acc["stuck"] / dec,
        "wedge_onsets": acc["wedge_onsets"],
        "wedge_onsets_after_dash": acc["wedge_onsets_dash"],
        "stalls_per_episode": float(stalls.sum() / n),
        "stalls": int(stalls.sum()),
        "longest_stall": int(longest.max()),
        "stall_len": stall_len,
        "win_rate": float(won[recorded].mean()) if recorded.any() else float("nan"),
        "mean_rank": float(rank[recorded].mean()) if recorded.any() else float("nan"),
        "mean_len": float(length[recorded].mean()) if recorded.any() else float("nan"),
        "idle_frac": acc["idle"] / max(acc["walking"], 1),
        "changed_frac": acc["changed"] / dec,
        "masked_dead_frac": acc["masked_dead"] / dec,
        "mask_disagree_frac": acc["disagree"] / dec,
        "steps": steps,
        "seconds": time.time() - t0,
        "map_names": list(ev.map_names),
    }


def _table(rows) -> str:
    head = ("| maps     | mask | wall-push decisions | of which wedged | stalls >= 2 s / episode "
            "| longest stall | win  | mean rank | decisions |\n"
            "|----------|------|---------------------|-----------------|------------------------"
            "|---------------|------|-----------|-----------|")
    body = [
        f"| {r['maps']:<8} | {r['mask']:<4} | {r['wall_push_frac']:<19.3f} | "
        f"{r['wall_push_wedged_frac']:<15.3f} | {r['stalls_per_episode']:<22.2f} | "
        f"{r['longest_stall']:<13d} | {r['win_rate']:<4.2f} | {r['mean_rank']:<9.2f} | "
        f"{r['decisions']:<9d} |"
        for r in rows
    ]
    wedge = ("\n| maps     | mask | wedged decisions | stuck (all 16 bins dead) | wedge onsets "
             "| onsets within 1 decision of a dash | mask != sim truth |\n"
             "|----------|------|------------------|--------------------------|--------------"
             "|------------------------------------|-------------------|")
    body2 = [
        f"| {r['maps']:<8} | {r['mask']:<4} | {r['wedged_frac']:<16.3f} | {r['stuck_frac']:<24.3f} | "
        f"{r['wedge_onsets']:<12d} | {r['wedge_onsets_after_dash']:<34d} | "
        f"{r['mask_disagree_frac']:<17.3f} |"
        for r in rows
    ]
    return "\n".join([head, *body, wedge, *body2])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=_DEFAULT_RUN, help="run directory (train.yaml + checkpoint)")
    ap.add_argument("--checkpoint", default="best_model.zip")
    ap.add_argument("--episodes", type=int, default=96, help="elite episodes per cell")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cells", nargs="*", default=list(_CELLS),
                    help="subset of training:off training:on holdout:off holdout:on")
    ap.add_argument("--json", default=None, help="also write the rows as JSON lines here")
    args = ap.parse_args(argv)

    run = (ROOT / args.run) if not Path(args.run).is_absolute() else Path(args.run)
    tcfg = load_train_config(run / "train.yaml", check_holdout=False)
    tcfg = replace(tcfg, eval=replace(tcfg.eval, episodes_per_tier=args.episodes, tiers=("elite",)))
    model = MaskablePPO.load(run / args.checkpoint, device=args.device)
    print(f"run {run.name}  checkpoint {args.checkpoint}  agent_obs {tcfg.run.agent_obs}  "
          f"episodes/cell {args.episodes}  device {args.device}  holdout {tcfg.eval.holdout_maps}")

    rows = []
    for cell in args.cells:
        maps_name, mask_name = cell.split(":")
        maps = tcfg.eval.holdout_maps if maps_name == "holdout" else None
        if maps_name == "holdout" and not maps:
            print(f"  {cell}: the run names no holdout maps, skipped")
            continue
        print(f"  cell {cell}", flush=True)
        r = measure_cell(tcfg, model, maps=maps, mask_on=(mask_name == "on"), device=args.device)
        rows.append(r)
        print(f"    done: {r['episodes']} episodes ({r['unfinished']} unfinished) in {r['steps']} "
              f"steps, {r['seconds']:.0f} s; wall-push {r['wall_push_frac']:.4f}, stalls/ep "
              f"{r['stalls_per_episode']:.2f}, longest {r['longest_stall']}, win {r['win_rate']:.3f}, "
              f"idle {r['idle_frac']:.3f}, changed {r['changed_frac']:.4f}, deployed-mask vetoes "
              f"{r['masked_dead_frac']:.4f}, mask!=truth {r['mask_disagree_frac']:.4f}; wedged "
              f"{r['wedged_frac']:.4f} (stuck {r['stuck_frac']:.4f}, onsets {r['wedge_onsets']}, "
              f"{r['wedge_onsets_after_dash']} right after a dash), wall-push while free "
              f"{r['wall_push_free_frac']:.4f}", flush=True)
        if args.json:
            with open(args.json, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r) + "\n")

    print()
    print(_table(rows))
    n = args.episodes
    print(f"\n{n} elite episodes per cell; SE of a win rate near 0.25 ~ {np.sqrt(0.25 * 0.75 / n):.3f}; "
          f"a stall is >= {rows[0]['stall_len'] if rows else '?'} consecutive wall-push decisions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
