"""Extra TRAINING frames: the most observed non-R textured tiles (mailbox first), away from the
held-out window, the bad windows and the existing frames."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(REPO))
from brawl_vision.terrain.labeling import hud_plans, observed_cells
SCR = Path(__file__).resolve().parent
cov = {int(k): v for k, v in json.load(open(REPO / "runs" / "map_checks" / "truth" / "coverage.json")).items()}
tex = np.array([l.rstrip("\n").split(",") for l in open(SCR / "textures.csv") if l.strip("\n")])
obs = observed_cells(hud_plans()["emulator"]); rows, cols = obs.shape
S = (1.022, 1.004)
train = [530, 570, 610, 650, 690, 730, 1460, 1510, 1560, 1600, 1640, 1680, 1720, 1760, 1800, 2490, 2580, 2850,
         2880, 2910, 2950, 3000, 3030, 3060, 3100, 3140, 3220, 3250, 3280, 3320]
bad = [(0, 528), (770, 99999 if False else 802), (803, 1382), (1815, 2315), (2390, 2480), (2630, 2840), (3326, 99999)]
W = {"mailbox": 4, "barrel": 2, "crate": 1, "teleporter": 1}
rows_ = []
for f, v in cov.items():
    if any(a <= f <= b for a, b in bad) or v["dA"] > 0.05 or min(abs(f - g) for g in train) < 12:
        continue
    A = v["A"]; hit = []
    for x, y in v["tiles"]:
        t = tex[y, x]
        if t in W and not (x <= 28 and y >= 29):
            i, j = (x + 0.5 - A[0]) * S[0], (y + 0.5 - A[1]) * S[1]
            if 0.5 <= i < cols - 0.5 and 0.5 <= j < rows - 0.5 and obs[int(j), int(i)]:
                hit.append(t)
    rows_.append((sum(W[t] for t in hit), f, hit))
rows_.sort(key=lambda r: -r[0])
picked = []
for s, f, hit in rows_:
    if s >= 6 and all(abs(f - p) >= 14 for _, p, _ in picked):
        picked.append((s, f, hit))
    if len(picked) == 20:
        break
picked.sort(key=lambda r: r[1])
from collections import Counter
tot = Counter()
for s, f, hit in picked:
    tot.update(hit); print(f, s, dict(Counter(hit)))
print("frames:", " ".join(str(f) for _, f, _ in picked)); print("tile-views:", dict(tot))
