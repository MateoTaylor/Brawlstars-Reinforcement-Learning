"""Extra held-out frames: ones whose view holds the most textured R tiles (x 0-28, y 29-59)."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

SCR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
cov = {int(k): v for k, v in json.load(open(REPO / "runs" / "map_checks" / "truth" / "coverage.json")).items()}
tex = np.array([l.rstrip("\n").split(",") for l in open(SCR / "textures.csv") if l.strip("\n")])
train = [530, 570, 610, 650, 690, 730, 1460, 1510, 1560, 1600, 1640, 1680, 1720, 1760, 1800, 2490, 2580, 2850,
         2880, 2910, 2950, 3000, 3030, 3060, 3100, 3140, 3220, 3250, 3280, 3320]
held = [1830, 1870, 1910, 1940, 1980, 2010, 2090, 2220, 2260, 2300]
bad = [(0, 520), (2390, 2480), (2630, 2840), (3340, 99999)]
W = {"teleporter": 3, "barrel": 3, "mailbox": 2, "crate": 2}
inR = lambda x, y: x <= 28 and y >= 29
held_tiles = Counter()
for f in held:
    for x, y in cov[f]["tiles"]:
        if inR(x, y) and tex[y, x] in W:
            held_tiles[(x, y)] += 1
print("R texture tiles in view of the current held-out frames:", len(held_tiles))
rows = []
for f, v in cov.items():
    if any(a <= f <= b for a, b in bad) or v["dA"] > 0.06 or min(abs(f - g) for g in train + held) < 12:
        continue
    t = [(x, y) for x, y in v["tiles"] if inR(x, y) and tex[y, x] in W]
    new = [p for p in t if p not in held_tiles]
    score = sum(W[tex[y, x]] for x, y in new) + 0.25 * sum(W[tex[y, x]] for x, y in t)
    rows.append((score, f, v["A"], v["dA"], v["ncc"], Counter(tex[y, x] for x, y in t), len(new)))
rows.sort(key=lambda r: -r[0])
picked = []
for r in rows:
    if all(abs(r[1] - p[1]) >= 25 for p in picked):
        picked.append(r)
    if len(picked) == 14:
        break
for s, f, A, dA, ncc, c, n in sorted(picked, key=lambda r: r[1]):
    print(f"{f:5d}  score {s:5.1f}  A ({A[0]:6.2f},{A[1]:6.2f})  dA {dA:.3f}  ncc {ncc:.3f}  new {n:2d}  {dict(c)}")
