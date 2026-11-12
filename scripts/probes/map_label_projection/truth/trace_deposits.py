"""Which ticks put which votes on given map tiles, in replay_clip.py's label check.
    python trace_deposits.py X,Y [X,Y ...] -- CLIP [replay_clip.py args]

Runs scripts/probes/replay_clip.py unchanged. For every vote the label check's map takes on those
tiles, it prints the clip time, the class, the vote's weight (below 1 on a one-wide gap), the view
cell it came from (column i, row j, j = 0 at the top) and the frame's re-aim f (x, then y), then a
summary per tile.

The loop re-aims each frame onto the localizer's sub-tile position before classifying it
(`RectifyPlan.registered`): the content moves by f, the fractional part of the view's world
position, so every cell is exactly a tile the localizer believes in and the deposit's rounding is
exact. A localizer error of d tiles therefore moves every crop d off its tile, by a fraction, and
the crop carries 1 - |d| of the tile and |d| of a neighbour. The view cell alone cannot show d.
crop_offset.py turns i, j and f into it with the image registration.

$OVERLAY=off runs the odometry at `overlay_share: 1.0`, the estimator from before 2026-11-05, the
way scripts/probes/odometry_overlay/replay_overlay.py does.
"""
import dataclasses
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "probes"))
sys.path.insert(0, str(REPO))
import replay_clip as rc                                    # noqa: E402
import brawl_vision.terrain.odometry as odometry            # noqa: E402
from brawl_vision.terrain.labeling import CLASSES          # noqa: E402

if "--" not in sys.argv[1:]:
    sys.exit(__doc__)
cut = sys.argv.index("--")
TILES = [tuple(int(v) for v in s.split(",")) for s in sys.argv[1:cut]]
ARGS = sys.argv[cut + 1:]
CH = {"FLOOR": ".", "WALL": "#", "BUSH": "b", "WATER": "~", "FENCE": "f"}
NAMES = [CH[c.name] for c in CLASSES]
MAP_ROWS = MAP_COLS = 60          # KnownMap tiles; LabelCheck.report centres them the same way

if os.environ.get("OVERLAY") == "off":
    _update = odometry.Odometry.update

    def _update_off(self, *args, **kwargs):
        if self.cfg.odometry_overlay_share != 1.0:
            self.cfg = dataclasses.replace(self.cfg, odometry_overlay_share=1.0)
        return _update(self, *args, **kwargs)
    odometry.Odometry.update = _update_off

votes_on = defaultdict(list)
_attach = rc.LabelCheck.attach


def attach(self, loop):
    _attach(self, loop)
    m = self.map
    r0, c0 = m.height // 2 - MAP_ROWS // 2, m.width // 2 - MAP_COLS // 2
    rr = np.array([r0 + y for _, y in TILES])
    cc = np.array([c0 + x for x, _ in TILES])
    inner = m.update

    def update(cells, world, plan, **kw):
        before = (m.votes[rr, cc].copy() if m.votes is not None
                  else np.zeros((len(TILES), len(CLASSES)), np.float32))
        out = inner(cells, world, plan, **kw)
        if m.votes is None:
            return out
        gained = m.votes[rr, cc] - before
        if not gained.any():
            return out
        px, py = world.position_tiles
        col0 = int(round(plan.origin_tile[0] + px)) - m.origin[0]
        row0 = int(round(plan.origin_tile[1] + py)) - m.origin[1]
        # The re-aim `registered` applied: the fixed plan's origin minus this one's.
        fx, fy = (float(b) - float(o) for b, o in zip(loop.vision.plan.origin_tile, plan.origin_tile))
        t = loop._match_clock()
        for k, tile in enumerate(TILES):
            for c in np.flatnonzero(gained[k] > 0):
                votes_on[tile].append((t, NAMES[c], float(gained[k, c]),
                                       int(rr[k] - row0), int(cc[k] - col0), fx, fy))
        return out
    m.update = update


rc.LabelCheck.attach = attach
code = rc.main(ARGS)
print(f"\nvotes on the traced tiles ({'overlay rule off' if os.environ.get('OVERLAY') == 'off' else 'as shipped'})")
for tile in TILES:
    v = votes_on.get(tile, [])
    if not v:
        print(f"{tile}: no votes")
        continue
    count = Counter(n for _, n, *_ in v)
    print(f"{tile}: {len(v)} votes, " + " ".join(f"{n}{k}" for n, k in count.most_common()) +
          f", {v[0][0]:.2f}-{v[-1][0]:.2f} s")
    print("     " + " ".join(f"{t:.2f}:{n}{'' if w == 1 else f'({w:g})'}@j{j}i{i}f{fx:+.3f},{fy:+.3f}"
                             for t, n, w, j, i, fx, fy in v))
sys.exit(code)
