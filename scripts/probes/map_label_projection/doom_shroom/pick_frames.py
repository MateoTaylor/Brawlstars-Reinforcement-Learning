"""Training and held-out frames for Doom Shroom's projected labels, split by ground (region R).

R is the west strip, map x 0-13, without the bounce pad (7-8, 15-16), plus the planter patch
(21-23, 33-35), so that every texture the clip shows is in both sets but the one bounce pad, which
trains. Held-out frames are those whose view holds the most R tiles, weighted toward the rare
textures, at least GAP frames apart; training frames are every STEP frames over the match that
are at least NEAR frames from a held-out one. Both skip frames whose track and own registration
differ by more than 0.06 tile (dA), and keep to FIRST-LAST: after the gate, before the defeat.

    python pick_frames.py          prints both lists and the --leave-row strings for each set
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

SCR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
cov = {int(k): v for k, v in json.load(open(REPO / "runs" / "map_checks" / "doom_shroom" / "coverage.json")).items()}
tex = np.array([l.rstrip("\n").split(",") for l in open(SCR / "textures.csv") if l.strip("\n")])

FIRST, LAST = 530, 2350          # after the gate (17.27 s, frame 518) to before "Defeated" (~80 s)
STEP, GAP, NEAR, N_HELD = 40, 30, 12, 12
W = {"mailbox": 3, "gray": 3, "barrel": 2, "planter": 3, "fence": 1, "speed": 1, "pink_plain": 1,
     "pink_cross": 0.5}


def in_r(x, y):
    if 7 <= x <= 8 and 15 <= y <= 16:
        return False
    return x <= 13 or (21 <= x <= 23 and 33 <= y <= 35)


def leave_rows(keep_r: bool):
    """--leave-row specs that leave R (keep_r False: training) or everything but R (held-out)."""
    out = []
    for y in range(60):
        cols = [x for x in range(60) if in_r(x, y) != keep_r]
        runs, start = [], None
        for x in range(61):
            on = x < 60 and x in cols
            if on and start is None:
                start = x
            if not on and start is not None:
                runs.append((start, x - 1))
                start = None
        out += [f"{y}:{a}-{b}" for a, b in runs]
    return out


ok = [f for f in sorted(cov) if FIRST <= f <= LAST and cov[f]["dA"] <= 0.06]
scored = []
for f in ok:
    t = [(x, y) for x, y in cov[f]["tiles"] if in_r(x, y)]
    w = sum(W.get(tex[y, x], 0) for x, y in t)
    scored.append((len(t) + 4 * w, f, len(t), Counter(tex[y, x] for x, y in t if tex[y, x])))
scored.sort(key=lambda r: -r[0])
held = []
for s, f, n, c in scored:
    if n >= 120 and all(abs(f - g) >= GAP for g in held):
        held.append(f)
    if len(held) == N_HELD:
        break
held.sort()
train = []
for f in ok:
    if all(abs(f - g) >= NEAR for g in held) and (not train or f - train[-1] >= STEP):
        train.append(f)

for name, frames, keep in (("held-out", held, True), ("training", train, False)):
    cells = Counter()
    for f in frames:
        for x, y in cov[f]["tiles"]:
            if in_r(x, y) == keep and tex[y, x]:
                cells[tex[y, x]] += 1
    print(f"{name}: {len(frames)} frames: {' '.join(map(str, frames))}")
    print(f"   textured tile-views in its ground: {dict(sorted(cells.items()))}")
print("\nheld-out --leave-row:", " ".join(leave_rows(True)))
print("\ntraining --leave-row:", " ".join(leave_rows(False)))
