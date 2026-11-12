"""How far each traced vote's crop sat off its tile, from trace_deposits.py's output and the image
registration:
    python crop_offset.py TRACE.txt

The loop re-aims each frame onto the localizer's sub-tile position (`RectifyPlan.registered`), so a
deposited cell holds exactly the pixels of the tile the localizer believes in. The image
registration says where that tile really is. coverage.json's offset A, interpolated between
registered frames as loc_vs_image.py does, puts the tile's centre at view coordinate
(x + 0.5 - A) * S in the fixed plan's cells, and the re-aim f moves it to (x + 0.5 - A) * S + f.
The crop's own centre is at i + 0.5. Their difference d, in tiles per axis, is how far the crop sat
off the tile. The crop carries 1 - |d| of the tile and |d| of the neighbour on d's side (+x east,
+y south), and with |d| > 0.5 it is mostly the neighbour. d holds the localizer's error and, toward
the view's sides, the rectification's scale error: S = 1.022 in x is about 0.3 tile at the view's
left and right edges.

Per tile it prints the votes by class in three bands of max(|dx|, |dy|), then each vote as time,
class, (dx, dy) and view cell i,j. A vote inside a registration gap (the bad windows in README.md)
has no d.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
S = (1.022, 1.004)
cov = {int(k): v["A"] for k, v in
       json.load(open(REPO / "runs" / "map_checks" / "truth" / "coverage.json")).items()}
frames = sorted(cov)
BANDS = ((0.25, "on the tile, |d| <= 0.25"), (0.5, "leaning, 0.25 < |d| <= 0.5"),
         (float("inf"), "mostly a neighbour, |d| > 0.5"))
VOTE = re.compile(r"([\d.]+):(.)(?:\(([^)]*)\))?@j(-?\d+)i(-?\d+)f([+-][\d.]+),([+-][\d.]+)")


def registration(t):
    """A at clip time t, between registered frames at most 20 frames apart, else None."""
    f = t * 30
    lo = max((g for g in frames if g <= f), default=None)
    hi = min((g for g in frames if g > f), default=None)
    if lo is None or hi is None or hi - lo > 20:
        return None
    w = (f - lo) / (hi - lo)
    return tuple(cov[lo][k] + w * (cov[hi][k] - cov[lo][k]) for k in (0, 1))


if len(sys.argv) != 2:
    sys.exit(__doc__)
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
lines = text[text.index("votes on the traced tiles"):].splitlines()
print(lines[0])
for k, line in enumerate(lines):
    if re.match(r"\(\d+, \d+\): no votes", line):
        print(line)
    m = re.match(r"\((\d+), (\d+)\): \d+ votes", line)
    if not m:
        continue
    x, y = int(m.group(1)), int(m.group(2))
    bands = defaultdict(Counter)
    detail = []
    for t, c, _w, j, i, fx, fy in VOTE.findall(lines[k + 1]):
        t, j, i, fx, fy = float(t), int(j), int(i), float(fx), float(fy)
        A = registration(t)
        if A is None:
            bands["unregistered"][c] += 1
            detail.append(f"{t:.1f}{c}@{i},{j}")
            continue
        dx = ((i + 0.5) - ((x + 0.5 - A[0]) * S[0] + fx)) / S[0]
        dy = ((j + 0.5) - ((y + 0.5 - A[1]) * S[1] + fy)) / S[1]
        band = next(name for lim, name in BANDS if max(abs(dx), abs(dy)) <= lim)
        bands[band][c] += 1
        detail.append(f"{t:.1f}{c}({dx:+.2f},{dy:+.2f})@{i},{j}")
    print(f"({x},{y}) {line.split(': ', 1)[1]}")
    for name in [n for _, n in BANDS] + ["unregistered"]:
        if bands.get(name):
            print(f"    {name:>30}: " + " ".join(f"{c}{n}" for c, n in bands[name].most_common()))
    print("    " + " ".join(detail))
