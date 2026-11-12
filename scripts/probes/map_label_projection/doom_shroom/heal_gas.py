"""Does the live gas path read the heal pad, or anything else on the map, as gas? CPU only.

    CLIP=bluestacks-doom-shroom-north-center .venv/Scripts/python.exe heal_gas.py [STEP]

Runs the loop's own `detect_zone` on the clip's registered frames (every STEP-th, through the fixed
HUD plan, as the projection does) and maps each view cell to its map tile with the image
registration's A and the clip's S. Two thresholds matter, and they are different tests:

- `zone.at_least(0.05)`: what the live loop refuses before the classifier's reading reaches the
  occupancy map or the localizer, and what the projection refuses as `GAS_FRACTION`.
- `zone.cells` (fraction >= `zone_min_cell_fraction`, 0.6): what the sticky `GasMap` latches into
  the policy's `in_zone`.

Prints a timeline with the heal pad's highest share per tile, then the textured tiles (and any near
the pad) a GasMap would have latched, each with the first frame that latched it. The numbers in
MAP_ONBOARDING.md (Doom Shroom, second recording) came from STEP 10 on both clips.
"""
import json
import sys
from collections import Counter

import numpy as np

from clip_paths import CLIP, COVERAGE, REPO, scale

sys.path.insert(0, str(REPO))
from brawl_vision.config import load_vision_config  # noqa: E402
from brawl_vision.terrain.labeling import default_hud, hud_plans, read_label_frames  # noqa: E402
from brawl_vision.terrain.zone import detect_zone  # noqa: E402

STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 10
HEAL = {(49, 29), (49, 30), (50, 29), (50, 30)}
S = scale()
cov = {int(k): v for k, v in json.load(open(COVERAGE)).items()}
tex = np.array([l.rstrip("\n").split(",") for l in
                open(REPO / "scripts/probes/map_label_projection/doom_shroom/textures.csv")
                if l.strip("\n")])
hud = default_hud(CLIP)
plan = hud_plans()[hud]
cfg = load_vision_config()
cols, rows = plan.size_tiles
print(f"clip {CLIP}, hud {hud}, S {S}, min_cell_fraction {cfg.zone_min_cell_fraction}")

frames = [f for f in sorted(cov) if f % STEP == 0]
images = read_label_frames(CLIP, set(frames), plan.viewport)
latched = {}          # tile -> first frame zone.cells put it in
peak = (0.0, None, None)
print(f"{'frame':>5} {'t':>6} {'cells':>5} {'>=.05':>5}  heal max frac (per tile)       textures under cells")
for f in frames:
    if f not in images:
        continue
    A = cov[f]["A"]
    z = detect_zone(plan.rectify(images[f]), plan, cfg)
    jj, ii = np.mgrid[0:rows, 0:cols]
    tx = np.floor((ii + 0.5) / S[0] + A[0]).astype(int)
    ty = np.floor((jj + 0.5) / S[1] + A[1]).astype(int)
    heal = {}
    for (x, y) in sorted(HEAL):
        m = (tx == x) & (ty == y) & z.observed
        if m.any():
            heal[(x, y)] = float(z.cell_fraction[m].max())
            if heal[(x, y)] > peak[0]:
                peak = (heal[(x, y)], (x, y), f)
    under = Counter()
    for j, i in zip(*np.nonzero(z.cells)):
        x, y = int(tx[j, i]), int(ty[j, i])
        name = str(tex[y, x]) if 0 <= x < 60 and 0 <= y < 60 else "off-map"
        under[name or "plain"] += 1
        latched.setdefault((x, y), f)
    hs = " ".join(f"{x},{y}:{v:.2f}" for (x, y), v in heal.items()) or "-"
    print(f"{f:5d} {f / 30:6.2f} {int(z.cells.sum()):5d} {int(z.at_least(0.05).sum()):5d}  "
          f"{hs:30s} {dict(under)}")

print("\nGasMap would latch these tiles (first frame), textured or near the heal pad:")
for (x, y), f in sorted(latched.items(), key=lambda kv: kv[1]):
    name = tex[y, x] if 0 <= x < 60 and 0 <= y < 60 else "off-map"
    if name or (47 <= x <= 52 and 27 <= y <= 32):
        print(f"  ({x},{y}) {name or 'plain'} from frame {f} ({f / 30:.2f} s)")
on_map = [f for (x, y), f in latched.items() if 0 <= x < 60 and 0 <= y < 60]
print(f"\n{len(latched)} tiles latched in all; first latch at frame "
      f"{min(latched.values()) if latched else '-'}, first on the map at frame "
      f"{min(on_map) if on_map else '-'}")
print("heal pad peak:", f"{peak[0]:.2f} on {peak[1]} at frame {peak[2]}" if peak[1] else "never in view")
