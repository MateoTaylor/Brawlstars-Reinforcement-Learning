"""How much of the deploy4 hero observation comes from OUTSIDE the 21x13 camera window.

Rolls the deploy4 checkpoint in the sim (elite tier, its training maps, CPU) and, on every
decision, compares what the observation carried against what a camera-limited observer could
have supplied: revealed enemies beyond the window, projectile slots taken by off-screen
projectiles, zone margins reporting gas that is not on screen, and how rarely the bush rule
(the only concealment the sim has) actually hides anyone.

Since 2026-09-24 the hero's reveal is `core/camera.hero_view` (concealment AND the
camera quad), so the enemy block must read 0 off-screen by construction; it stays as the
regression check, and the window here is the quad, not the 21x13 crop.
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brawl_sim.training.config import load_train_config          # noqa: E402
from brawl_sim.training.evaluation import TierEvaluator           # noqa: E402
from brawl_sim.core import camera, observation, geometry as geo    # noqa: E402
from sb3_contrib import MaskablePPO                                # noqa: E402

EPS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800
device = "cpu"
run = ROOT / (sys.argv[3] if len(sys.argv) > 3 else "runs/mortis_deploy4-20260921-185945")  # argv: [episodes] [max_steps] [run dir]
tcfg = load_train_config(run / "train.yaml", check_holdout=False)
tcfg = replace(tcfg, eval=replace(tcfg.eval, episodes_per_tier=EPS, tiers=("elite",)))
model = MaskablePPO.load(run / "best_model.zip", device=device)
ev = TierEvaluator(tcfg, tiers=("elite",), device=device, maps=None)
sim, venv = ev.sim, ev.venv
cfg = sim.cfg
sim.gen.manual_seed(ev.seed)
obs = venv.reset()
n = ev.n_envs
K = 12
HW, HH = cfg.view_w // 2, cfg.view_h // 2          # 10, 6
print(f"envs {n}  steps<= {MAX_STEPS}  view {cfg.view_w}x{cfg.view_h}  horizon "
      f"{cfg.zone_margin_horizon_tiles}  bush_reveal {float(sim.params.bush_reveal_radius.mean()):.1f}")

recorded = np.zeros(n, bool)
acc = dict(dec=0, any_rev=0, any_rev_in=0, any_alive_in=0, n_rev=0.0, n_rev_in=0.0,
           n_rev_out=0.0, n_hidden_bush=0.0, n_alive=0.0, nearest_rev_beyond=0,
           hp_of_offscreen=0, prj_alive=0.0, prj_in=0.0, prj_shown=0.0, prj_slots_out=0.0,
           prj_zero_ttc_out=0.0, prj_starved=0, zone_margin_unseen=0, zone_any_margin=0,
           zone_gas_in_window=0)
nearest = []
t0 = time.time()
for step in range(min(ev.max_steps, MAX_STEPS)):
    masks = venv.action_masks()
    action, _ = model.predict(obs, deterministic=True, action_masks=masks)
    st = sim.state
    live = torch.as_tensor(~recorded)
    L = int(live.sum())
    if L == 0:
        break
    # -- enemies: what the observation carried vs what the window contains ------------------
    vis = sim._obs_hero_view[:, 1:]                  # hero sees enemy j (bush rule AND on screen)
    alive = st.ent_alive[:, 1:]
    revealed = vis & alive
    origin = observation._view_origin(st, cfg)
    hero = st.ent_pos[:, 0:1]
    cam = camera.camera_centre(st.ent_pos[:, 0], cfg).unsqueeze(1)
    in_view = camera.in_camera(st.ent_pos[:, 1:] - cam, cfg)   # the quad the screen shows
    dist = (st.ent_pos[:, 1:] - hero).norm(dim=-1)
    rev_in, rev_out = revealed & in_view, revealed & ~in_view
    acc["dec"] += L
    acc["any_rev"] += int((revealed.any(1) & live).sum())
    acc["any_rev_in"] += int((rev_in.any(1) & live).sum())
    acc["any_alive_in"] += int(((alive & in_view).any(1) & live).sum())
    acc["n_rev"] += float(revealed.sum(1)[live].sum())
    acc["n_rev_in"] += float(rev_in.sum(1)[live].sum())
    acc["n_rev_out"] += float(rev_out.sum(1)[live].sum())
    acc["n_alive"] += float(alive.sum(1)[live].sum())
    acc["n_hidden_bush"] += float((alive & ~vis).sum(1)[live].sum())
    dmin = torch.where(revealed, dist, torch.full_like(dist, float("inf"))).min(1).values
    has = revealed.any(1) & live
    nearest.extend(dmin[has].tolist())
    acc["nearest_rev_beyond"] += int((has & ~rev_in.any(1)).sum())   # revealed, none on screen
    # -- projectiles: top-12 by ttc over the whole map, THEN the in_view mask -----------------
    p_alive = st.prj_alive
    p_in = observation._in_window(st.prj_pos, origin, cfg.view_h, cfg.view_w)
    ttc, _ = geo.closest_approach(st.prj_pos, st.prj_vel, hero.expand_as(st.prj_pos))
    prio = torch.where(p_alive, ttc, torch.full_like(ttc, float("inf")))
    _, idx = torch.topk(prio, k=min(K, prio.shape[1]), largest=False, dim=-1)
    top_alive = torch.gather(p_alive, 1, idx)
    top_in = torch.gather(p_in, 1, idx)
    top_ttc = torch.gather(ttc, 1, idx)
    shown = torch.as_tensor(obs["projectiles"][:, :, 0] > 0).sum(1)
    n_in = (p_alive & p_in).sum(1)
    acc["prj_alive"] += float(p_alive.sum(1)[live].sum())
    acc["prj_in"] += float(n_in[live].sum())
    acc["prj_shown"] += float(shown[live].sum())
    acc["prj_slots_out"] += float((top_alive & ~top_in).sum(1)[live].sum())
    acc["prj_zero_ttc_out"] += float((top_alive & ~top_in & (top_ttc <= 0)).sum(1)[live].sum())
    acc["prj_starved"] += int(((shown < torch.clamp(n_in, max=K)) & live).sum())
    # -- zone: a margin under the horizon while no gas cell is inside the window --------------
    hx, hy = st.ent_pos[:, 0, 0], st.ent_pos[:, 0, 1]
    m = torch.stack([hx - st.zone_lo[:, 0], st.zone_hi[:, 0] - hx,
                     hy - st.zone_lo[:, 1], st.zone_hi[:, 1] - hy], dim=-1)   # W, E, N, S
    H = float(cfg.zone_margin_horizon_tiles)
    ext = torch.tensor([HW, HW, HH, HH], dtype=m.dtype)     # window half-extent per side, cells
    informative = (m.abs() < H)                              # the sim reports a real number
    beyond_screen = (m.abs() > ext + 1.0)                    # and that edge is not on screen
    acc["zone_any_margin"] += int((informative.any(1) & live).sum())
    acc["zone_margin_unseen"] += int(((informative & beyond_screen).any(1) & live).sum())
    # any window cell outside the rect = gas visible on screen
    ox, oy = origin[:, 0].to(m.dtype), origin[:, 1].to(m.dtype)
    gas_on_screen = ((ox < st.zone_lo[:, 0]) | (ox + cfg.view_w > st.zone_hi[:, 0]) |
                     (oy < st.zone_lo[:, 1]) | (oy + cfg.view_h > st.zone_hi[:, 1]))
    acc["zone_gas_in_window"] += int((gas_on_screen & live).sum())

    obs, _, dones, infos = venv.step(action)
    for i in np.nonzero(dones & ~recorded)[0]:
        recorded[i] = True
    if step % 200 == 199:
        print(f"  step {step + 1}  live {L}  {time.time() - t0:.0f}s", flush=True)

D = acc["dec"]
near = np.array(nearest)
print(f"\ndecisions {D}  ({time.time() - t0:.0f}s)")
print("ENEMIES (hero reveal = bush concealment AND the camera quad, since C3)")
print(f"  >=1 enemy revealed            {acc['any_rev'] / D:.3f}")
print(f"  >=1 revealed INSIDE window    {acc['any_rev_in'] / D:.3f}")
print(f"  >=1 alive inside window       {acc['any_alive_in'] / D:.3f}")
print(f"  revealed but none on screen   {acc['nearest_rev_beyond'] / D:.3f}   (policy told about enemies a camera would not show)")
print(f"  mean revealed / in-window / off-screen per decision  {acc['n_rev'] / D:.2f} / {acc['n_rev_in'] / D:.2f} / {acc['n_rev_out'] / D:.2f}")
print(f"  share of revealed rows that are off-screen           {acc['n_rev_out'] / max(acc['n_rev'], 1):.3f}")
print(f"  mean alive enemies / not revealed (bush or off screen) {acc['n_alive'] / D:.2f} / {acc['n_hidden_bush'] / D:.3f}")
if len(near):
    print(f"  nearest revealed enemy, tiles: p25 {np.percentile(near, 25):.1f}  p50 {np.percentile(near, 50):.1f}  p75 {np.percentile(near, 75):.1f}  p90 {np.percentile(near, 90):.1f}")
print("PROJECTILES (top-12 by ttc over the whole map, then in_view mask)")
print(f"  mean alive on map / alive in window / rows the policy saw  {acc['prj_alive'] / D:.2f} / {acc['prj_in'] / D:.2f} / {acc['prj_shown'] / D:.2f}")
print(f"  mean top-12 slots held by OFF-screen projectiles           {acc['prj_slots_out'] / D:.2f}  (of which ttc==0 receding/static: {acc['prj_zero_ttc_out'] / D:.2f})")
print(f"  decisions where an on-screen projectile was dropped        {acc['prj_starved'] / D:.3f}")
print("ZONE (exact rect margins clamped at the horizon)")
print(f"  a margin under the horizon           {acc['zone_any_margin'] / D:.3f}")
print(f"  ... while that edge is off screen    {acc['zone_margin_unseen'] / D:.3f}")
print(f"  gas actually inside the window       {acc['zone_gas_in_window'] / D:.3f}")
ev.close()
