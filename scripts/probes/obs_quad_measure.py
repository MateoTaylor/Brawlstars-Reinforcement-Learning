"""What the 21x13 rectangle would discard compared with the camera's full ground quad.

Same rollout as obs_leak_measure.py (deploy4 checkpoint, elite tier, CPU). Per decision, every
alive, bush-revealed enemy is classified by where it stands relative to the hero: inside the
21x13 grid rectangle, inside the camera quad but outside the rectangle (split by where), or
outside the quad. The quad corners are hero-relative tiles from the calibrated homography.
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
from brawl_sim.core import observation                              # noqa: E402
from sb3_contrib import MaskablePPO                                # noqa: E402

# hero-relative (dx, dy) tiles: top-left, top-right, bottom-right, bottom-left
QUAD = np.array([[-14.11, -10.59], [14.77, -10.93], [11.73, 7.45], [-11.55, 7.01]], np.float32)


def in_quad(rel: torch.Tensor) -> torch.Tensor:
    """(..., 2) hero-relative -> (...) bool, convex quad by four half-planes (clockwise on screen)."""
    inside = torch.ones(rel.shape[:-1], dtype=torch.bool)
    for i in range(4):
        a, b = QUAD[i], QUAD[(i + 1) % 4]
        ex, ey = b[0] - a[0], b[1] - a[1]
        cross = ex * (rel[..., 1] - a[1]) - ey * (rel[..., 0] - a[0])
        inside &= cross >= 0
    return inside


EPS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
run = ROOT / "runs" / "mortis_deploy4-20260921-185945"
tcfg = load_train_config(run / "train.yaml", check_holdout=False)
tcfg = replace(tcfg, eval=replace(tcfg.eval, episodes_per_tier=EPS, tiers=("elite",)))
model = MaskablePPO.load(run / "best_model.zip", device="cpu")
ev = TierEvaluator(tcfg, tiers=("elite",), device="cpu", maps=None)
sim, venv, cfg = ev.sim, ev.venv, ev.sim.cfg
sim.gen.manual_seed(ev.seed)
obs = venv.reset()
n = ev.n_envs
recorded = np.zeros(n, bool)
c = dict(dec=0, rect=0.0, quad_only=0.0, above=0.0, below=0.0, side=0.0, outside=0.0,
         any_rect=0, any_quad=0)
t0 = time.time()
for step in range(ev.max_steps):
    masks = venv.action_masks()
    action, _ = model.predict(obs, deterministic=True, action_masks=masks)
    st = sim.state
    live = torch.as_tensor(~recorded)
    if not live.any():
        break
    revealed = sim._obs_vis[:, 0, 1:] & st.ent_alive[:, 1:]
    origin = observation._view_origin(st, cfg)
    in_rect = observation._in_window(st.ent_pos[:, 1:], origin, cfg.view_h, cfg.view_w)
    rel = st.ent_pos[:, 1:] - st.ent_pos[:, 0:1]
    q = in_quad(rel)
    quad_only = revealed & q & ~in_rect
    above = quad_only & (rel[..., 1] < -(cfg.view_h // 2 + 0.5))
    below = quad_only & (rel[..., 1] > (cfg.view_h // 2 + 0.5))
    side = quad_only & ~above & ~below
    L = int(live.sum())
    c["dec"] += L
    c["rect"] += float((revealed & in_rect).sum(1)[live].sum())
    c["quad_only"] += float(quad_only.sum(1)[live].sum())
    c["above"] += float(above.sum(1)[live].sum())
    c["below"] += float(below.sum(1)[live].sum())
    c["side"] += float(side.sum(1)[live].sum())
    c["outside"] += float((revealed & ~q).sum(1)[live].sum())
    c["any_rect"] += int(((revealed & in_rect).any(1) & live).sum())
    c["any_quad"] += int(((revealed & q).any(1) & live).sum())
    obs, _, dones, infos = venv.step(action)
    for i in np.nonzero(dones & ~recorded)[0]:
        recorded[i] = True

D = c["dec"]
on_screen = c["rect"] + c["quad_only"]
print(f"decisions {D}  ({time.time() - t0:.0f}s)")
print(f"revealed enemies per decision: in 21x13 rect {c['rect'] / D:.3f} | on screen but outside the rect "
      f"{c['quad_only'] / D:.3f} (above {c['above'] / D:.3f}, below {c['below'] / D:.3f}, sides {c['side'] / D:.3f}) "
      f"| off screen {c['outside'] / D:.3f}")
print(f"share of ON-SCREEN sightings the rectangle discards: {c['quad_only'] / max(on_screen, 1):.3f}")
print(f">=1 enemy on screen: rect {c['any_rect'] / D:.3f}   quad {c['any_quad'] / D:.3f}")
ev.close()
