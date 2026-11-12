"""Per candidate frame: registration consistency, and what map ground it sees (by class and wall
texture), so training and held-out frames can be picked on disjoint ground covering everything."""
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

SCR = Path(__file__).resolve().parent
work = REPO / "runs/map_checks/truth/projection/bluestacks-truth-west-upper"
reg = np.load(work / "reg.npz", allow_pickle=True)
reg2 = np.load(work / "reg_at_scale.npz", allow_pickle=True)
S = tuple(json.loads(str(reg2["meta"]))["S"])
track = lp.Track(reg["pos"], reg["seg"], reg2["idx"], reg2["A"], S)
chars = read_chars(MAPS_DIR / "truth.csv")
tex = np.array([l.rstrip("\n").split(",") for l in open(SCR / "textures.csv") if l.strip("\n")])
plan = hud_plans()["emulator"]
obs = observed_cells(plan)
cols, rows = plan.size_tiles
jj, ii = np.nonzero(obs)
where = {int(i): k for k, i in enumerate(reg2["idx"])}
skip = [(2640, 2830), (2400, 2470)]
frames = [int(f) for f in reg2["idx"] if track.covers(int(f)) and reg["seg"][f] in (7, 16)
          and not any(a <= f <= b for a, b in skip)]
out = {}
print("frame   t(s)  seg  A               dA(track-own)  NCC   hero~tile   seen: . # b ~   stone crate barrel mailbox tele")
for f in frames:
    if not (470 <= f <= 802 or 1383 <= f <= 3425):
        continue
    A = track(f)
    own = reg2["A"][where[f]]
    dA = np.abs(A - own).max()
    x = np.floor(A[0] + (ii + 0.5) / S[0]).astype(int)
    y = np.floor(A[1] + (jj + 0.5) / S[1]).astype(int)
    ok = (x >= 0) & (x < 60) & (y >= 0) & (y < 60)
    tiles = set(zip(x[ok].tolist(), y[ok].tolist()))
    c = Counter(chars[ty, tx] for tx, ty in tiles)
    t = Counter(tex[ty, tx] for tx, ty in tiles if tex[ty, tx])
    hero = (A[0] + 15 / S[0], A[1] + 11.5 / S[1])
    out[f] = dict(A=A.tolist(), dA=float(dA), ncc=float(reg2["score"][where[f]]), tiles=sorted(tiles),
                  counts={k: c.get(k, 0) for k in ".#b~"}, tex=dict(t))
    if f % 30 == 0:
        print(f"{f:5d} {f/30:6.1f}  {reg['seg'][f]:3d}  ({A[0]:6.2f},{A[1]:6.2f})  {dA:6.3f}  {reg2['score'][where[f]]:.3f}  "
              f"({hero[0]:4.1f},{hero[1]:4.1f})   {c.get('.',0):4d} {c.get('#',0):3d} {c.get('b',0):3d} {c.get('~',0):3d}   "
              f"{t.get('stone',0):4d} {t.get('crate',0):4d} {t.get('barrel',0):4d} {t.get('mailbox',0):4d} {t.get('teleporter', t.get('tele',0)):4d}")
json.dump(out, open(REPO / "runs" / "map_checks" / "truth" / "coverage.json", "w"))
print(sorted(set(tex.ravel())))
