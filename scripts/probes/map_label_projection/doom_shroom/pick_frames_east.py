"""Training and held-out frames for the second Doom Shroom clip, split by ground (region R2).

The first clip's split (pick_frames.py) holds out region R, the west strip. Its training labels
cover the ground the first clip saw outside R, map x 0-49. This clip, from the north-centre spawn,
reaches the east, so its held-out set is the east strip R2, which no label from either clip trains
on. Its training frames leave both R and R2 `?`. R2 starts at x 52 (`R2_X0`) so that the textures
the first clip never labelled, the orange barriers and the heal pad at x 47-51, are all training
ground. From x 50, the held-out set took 8 of the 11 barriers and training kept 3.

Held-out frames are those whose view holds the most R2 tiles, weighted toward the rare textures,
at least GAP frames apart. Training frames are every STEP frames over the match that are at least
NEAR frames from a held-out one. Both skip frames whose track and own registration differ by more
than 0.06 tile (dA), and keep to FIRST-LAST, after the gate and before the defeat.

The shipped lists are this script's with one change, which
tests/fixtures/vision/labels_doom_shroom_east_test/README.md records: three held-out frames the gas
had emptied were replaced, and two training frames that then sat too near were dropped.

    CLIP=bluestacks-doom-shroom-north-center python pick_frames_east.py
"""
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("CLIP", "bluestacks-doom-shroom-north-center")
from clip_paths import COVERAGE  # noqa: E402

SCR = Path(__file__).resolve().parent
cov = {int(k): v for k, v in json.load(open(COVERAGE)).items()}
tex = np.array([l.rstrip("\n").split(",") for l in open(SCR / "textures.csv") if l.strip("\n")])

FIRST, LAST = int(os.environ.get("FIRST", 570)), int(os.environ.get("LAST", 2700))
STEP, GAP, NEAR = 40, 30, 12
N_HELD = int(os.environ.get("N_HELD", 12))
X0 = int(os.environ.get("R2_X0", 52))      # R2 = map x X0-59
W = {"orange": 4, "heal": 4, "mailbox": 3, "gray": 3, "barrel": 2, "planter": 3, "fence": 1,
     "speed": 1, "pink_plain": 1, "pink_cross": 0.5}


def in_r(x, y):
    """The first clip's held-out region R, as pick_frames.py draws it."""
    if 7 <= x <= 8 and 15 <= y <= 16:
        return False
    return x <= 13 or (21 <= x <= 23 and 33 <= y <= 35)


def in_r2(x, y):
    return x >= X0


def runs_of(cols):
    out, start = [], None
    for x in range(61):
        on = x < 60 and x in cols
        if on and start is None:
            start = x
        if not on and start is not None:
            out.append((start, x - 1))
            start = None
    return out


def leave_rows(held: bool):
    """--leave-row specs: held-out leaves everything but R2; training leaves R and R2."""
    out = []
    for y in range(60):
        if held:
            cols = {x for x in range(60) if not in_r2(x, y)}
        else:
            cols = {x for x in range(60) if in_r(x, y) or in_r2(x, y)}
        out += [f"{y}:{a}-{b}" for a, b in runs_of(cols)]
    return out


ok = [f for f in sorted(cov) if FIRST <= f <= LAST and cov[f]["dA"] <= 0.06]
scored = []
for f in ok:
    t = [(x, y) for x, y in cov[f]["tiles"] if in_r2(x, y)]
    w = sum(W.get(tex[y, x], 0) for x, y in t)
    scored.append((len(t) + 4 * w, f, len(t)))
scored.sort(key=lambda r: -r[0])
held = []
for s, f, n in scored:
    if n >= 60 and all(abs(f - g) >= GAP for g in held):
        held.append(f)
    if len(held) == N_HELD:
        break
held.sort()
train = []
for f in ok:
    if all(abs(f - g) >= NEAR for g in held) and (not train or f - train[-1] >= STEP):
        train.append(f)

print(f"\n=== {os.environ['CLIP']}: R2 = map x {X0}-59, frames {FIRST}-{LAST}, {len(ok)} usable")
for name, frames, mine in (("held-out", held, lambda x, y: in_r2(x, y)),
                           ("training", train, lambda x, y: not in_r(x, y) and not in_r2(x, y))):
    cells, tiles = Counter(), set()
    for f in frames:
        for x, y in cov[f]["tiles"]:
            if mine(x, y):
                tiles.add((x, y))
                if tex[y, x]:
                    cells[tex[y, x]] += 1
    print(f"{name}: {len(frames)} frames: {' '.join(map(str, frames))}")
    print(f"   {len(tiles)} tiles; textured tile-views in its ground: {dict(sorted(cells.items()))}")
print("\nheld-out --leave-row:", " ".join(leave_rows(True)))
print("\ntraining --leave-row:", " ".join(leave_rows(False)))
