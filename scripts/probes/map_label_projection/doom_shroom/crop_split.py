"""Each traced vote's crop offset d, split into the localizer's error and the scale term:
    python crop_split.py TRACE.txt CSV

TRACE.txt is trace_deposits.py's output and CSV the replay it traced. d is crop_offset.py's: how far
the crop sat off its tile, by the image registration, + east / + south. The loop maps view
coordinates to map tiles at unit scale. The image registration maps them at scale S. So, with
u = i + 0.5 - f the crop's view coordinate before the re-aim f,

    d     = A - A_loop + u (1/S - 1)
    loc   = d - (u - c)(1/S - 1)   the localizer's error at the view's centre c
    scale = (u - c)(1/S - 1)       zero at the centre, about 0.3 tile in x at the view's sides

c is the camera model's viewport centre in the fixed plan's cells. A positive loc means the loop
put the view's centre west (or north) of where the registration puts it, so every crop sat east
(or south) of its tile by loc, plus its own scale term.

The last column is K = (loc_off - img_off) + loc, with loc_off - img_off as loc_vs_image.py
computes it. It comes out nearly constant over the clip: within 0.11 tile of (+13.93, +10.08) over
the shipped replay's 376 traced votes. So K minus loc_vs_image.py's output reads as the
localizer's error at the centre.
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from brawl_vision.camera import build_rectify_plan, load_camera_model  # noqa: E402
from clip_paths import COVERAGE, scale  # noqa: E402

S = scale()
_model = load_camera_model()
_plan = build_rectify_plan(_model)
C = tuple(float(v) for v in _model.px_to_tile(np.array([[_model.viewport[0] / 2,
                                                          _model.viewport[1] / 2]], float))[0]
          - np.asarray(_plan.origin_tile, float))
cov = {int(k): v["A"] for k, v in json.load(open(COVERAGE)).items()}
frames = sorted(cov)
VOTE = re.compile(r"([\d.]+):(.)(?:\(([^)]*)\))?@j(-?\d+)i(-?\d+)f([+-][\d.]+),([+-][\d.]+)")


def g(r, k):
    try:
        return float(r[k])
    except ValueError:
        return math.nan


def registration(t):
    """A at clip time t, between registered frames at most 20 frames apart, else None."""
    f = t * 30
    lo = max((q for q in frames if q <= f), default=None)
    hi = min((q for q in frames if q > f), default=None)
    if lo is None or hi is None or hi - lo > 20:
        return None
    w = (f - lo) / (hi - lo)
    return tuple(cov[lo][k] + w * (cov[hi][k] - cov[lo][k]) for k in (0, 1))


if len(sys.argv) != 3:
    sys.exit(__doc__)
rows = sorted((r for r in csv.DictReader(open(sys.argv[2]))
               if not math.isnan(g(r, "hero_odo_x")) and r["map_state"] in ("fixed", "confirm")),
              key=lambda r: g(r, "t"))


def between(t):
    a = next((r for r in reversed(rows) if g(r, "t") <= t), None)
    b = next((r for r in rows if g(r, "t") > t), None)
    return (a, b) if a is not None and b is not None else None


def loc_minus_img(t, A):
    """loc_vs_image.py's loc_off - img_off, at a vote's own time."""
    a, b = between(t)
    w = (t - g(a, "t")) / (g(b, "t") - g(a, "t"))
    lerp = lambda k: g(a, k) + w * (g(b, k) - g(a, k))  # noqa: E731
    ix = A[0] + lerp("hero_offset_x") / S[0] - 30 - lerp("hero_odo_x")
    iy = A[1] + lerp("hero_offset_y") / S[1] - 30 - lerp("hero_odo_y")
    return g(a, "map_dx") - ix, g(a, "map_dy") - iy


text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
lines = text[text.index("votes on the traced tiles"):].splitlines()
print(f"view centre c = ({C[0]:.2f}, {C[1]:.2f}), S = {S}")
for k, line in enumerate(lines):
    m = re.match(r"\((\d+), (\d+)\): \d+ votes", line)
    if not m:
        continue
    x, y = int(m.group(1)), int(m.group(2))
    print(f"({x},{y}) {line.split(': ', 1)[1]}")
    print("       t c   view    d             loc           scale         K")
    for t, c, _w, j, i, fx, fy in VOTE.findall(lines[k + 1]):
        t, j, i, fx, fy = float(t), int(j), int(i), float(fx), float(fy)
        A = registration(t)
        if A is None or between(t) is None:
            print(f"   {t:5.2f} {c}  {i:2d},{j:<2d}  unregistered")
            continue
        dx = ((i + 0.5) - ((x + 0.5 - A[0]) * S[0] + fx)) / S[0]
        dy = ((j + 0.5) - ((y + 0.5 - A[1]) * S[1] + fy)) / S[1]
        sx = ((i + 0.5 - fx) - C[0]) * (1 / S[0] - 1)
        sy = ((j + 0.5 - fy) - C[1]) * (1 / S[1] - 1)
        lx, ly = loc_minus_img(t, A)
        print(f"   {t:5.2f} {c}  {i:2d},{j:<2d}  ({dx:+.2f},{dy:+.2f})  ({dx - sx:+.2f},{dy - sy:+.2f})"
              f"  ({sx:+.2f},{sy:+.2f})  ({lx + dx - sx:+.2f},{ly + dy - sy:+.2f})")
