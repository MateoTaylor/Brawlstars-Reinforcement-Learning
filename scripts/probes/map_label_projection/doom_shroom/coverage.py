"""Per registered frame: registration consistency, and what map ground it sees, by class and by
texture (floor specials included), so training and held-out frames can be picked on disjoint ground
covering everything. Writes the clip's coverage: `$CLIP`'s, by default the first clip's, to
runs/map_checks/doom_shroom/coverage.json (clip_paths.py).

    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/probes/map_label_projection/doom_shroom/coverage.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from brawl_deployment.perception.known_map import MAPS_DIR, read_chars          # noqa: E402
from brawl_vision.terrain import label_projection as lp                       # noqa: E402
from brawl_vision.terrain.labeling import hud_plans, observed_cells           # noqa: E402
from clip_paths import COVERAGE, WORK                                         # noqa: E402

MAP = "doom_shroom"
SCR = Path(__file__).resolve().parent
work = WORK
reg = np.load(work / "reg.npz", allow_pickle=True)
reg2 = np.load(work / "reg_at_scale.npz", allow_pickle=True)
S = tuple(json.loads(str(reg2["meta"]))["S"])
track = lp.Track(reg["pos"], reg["seg"], reg2["idx"], reg2["A"], S)
chars = read_chars(MAPS_DIR / f"{MAP}.csv")
tex = np.array([l.rstrip("\n").split(",") for l in open(SCR / "textures.csv") if l.strip("\n")])
TEX = sorted(set(tex.ravel()) - {""})
plan = hud_plans()["emulator"]
jj, ii = np.nonzero(observed_cells(plan))
where = {int(i): k for k, i in enumerate(reg2["idx"])}
frames = [int(f) for f in reg2["idx"] if track.covers(int(f))]
out = {}
print(f"S = {S}; {len(frames)} registered frames in the matched segments")
print("frame   t(s)  seg  A               dA     NCC    seen: . # b ~   " + " ".join(t[:6] for t in TEX))
for f in frames:
    A = track(f)
    own = reg2["A"][where[f]]
    dA = float(np.abs(A - own).max())
    x = np.floor(A[0] + (ii + 0.5) / S[0]).astype(int)
    y = np.floor(A[1] + (jj + 0.5) / S[1]).astype(int)
    ok = (x >= 0) & (x < 60) & (y >= 0) & (y < 60)
    tiles = set(zip(x[ok].tolist(), y[ok].tolist()))
    c = Counter(chars[ty, tx] for tx, ty in tiles)
    t = Counter(tex[ty, tx] for tx, ty in tiles if tex[ty, tx])
    out[f] = dict(A=A.tolist(), dA=dA, ncc=float(reg2["score"][where[f]]), seg=int(reg["seg"][f]),
                  tiles=sorted(tiles), counts={k: c.get(k, 0) for k in ".#b~"}, tex=dict(t))
    if f % 30 == 0:
        print(f"{f:5d} {f/30:6.1f}  {reg['seg'][f]:3d}  ({A[0]:6.2f},{A[1]:6.2f})  {dA:5.3f}  {out[f]['ncc']:.3f}  "
              f"{c.get('.',0):4d} {c.get('#',0):3d} {c.get('b',0):3d} {c.get('~',0):3d}   "
              + " ".join(f"{t.get(k, 0):6d}" for k in TEX))
json.dump(out, open(COVERAGE, "w"))
