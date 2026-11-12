"""Score replay seen-maps (../truth/replay_seen.py npz) against rockwall_brawl.csv, on the held-out
region R (pick_frames.in_r; no training label ever touched it) and, for reference, on the ground
outside it. Per class, per texture (cube and cone, read as wall or not), and the bush/wall
confusion this palette risks. Spawn tiles score as floor. A copy of Doom Shroom's, with R imported
rather than restated.

    python region_score.py NAME=seen.npz [NAME=seen.npz ...] [--broken C,R@F ...]

A seen map gathers the whole match, and the map label is the ground as it starts, so a cell that
breaks reads floor late and would score as a miss. `--broken` takes map_project_labels.py's list
(the held-out README's $BROKEN) and leaves those cells out of every count, whatever their frame.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from brawl_deployment.perception.known_map import MAPS_DIR, read_chars          # noqa: E402
from brawl_vision.terrain.label_projection import parse_broken                  # noqa: E402
from pick_frames import TEX as tex, in_r                                        # noqa: E402

label = read_chars(MAPS_DIR / "rockwall_brawl.csv")
label = np.where(label == "S", ".", label)
R = np.array([[in_r(x, y) for x in range(60)] for y in range(60)])
BLOCK = set("#~")
TEXTURES = ("cube", "cone")
ap = argparse.ArgumentParser()
ap.add_argument("maps", nargs="+", metavar="NAME=seen.npz")
ap.add_argument("--broken", nargs="+", default=[], metavar="C,R@F")
a = ap.parse_args()
kept = np.ones((60, 60), bool)
for (x, y), _ in parse_broken(a.broken):
    kept[y, x] = False
if a.broken:
    print(f"{(~kept).sum()} broken cells left out:", " ".join(f"({x},{y}){label[y, x]}"
                                                         for y, x in np.argwhere(~kept)))
for spec in a.maps:
    name, path = spec.split("=", 1)
    seen = np.load(path)["seen"]
    for region, mask in (("held-out R", R & kept), ("training ground", ~R & kept)):
        s = mask & (seen != "?")
        t, v = label[s], seen[s]
        agree = (t == v).mean()
        tb, vb = np.isin(t, list(BLOCK)), np.isin(v, list(BLOCK))
        tp = (tb & vb).sum()
        f1 = 2 * tp / max(tb.sum() + vb.sum(), 1)
        print(f"\n== {name}, {region}: {s.sum()} tiles seen of {mask.sum()}; 5-class {agree:.3f}; "
              f"blocking F1 {f1:.3f} (recall {tp / max(tb.sum(), 1):.3f}, precision {tp / max(vb.sum(), 1):.3f})")
        for c, cname in ((".", "floor"), ("#", "wall"), ("b", "bush"), ("~", "water")):
            k = t == c
            if k.sum():
                got = Counter(v[k])
                print(f"   {cname:6s} n {k.sum():4d}  read as: " + ", ".join(f"{g} {n}" for g, n in got.most_common()))
        for texture in TEXTURES:
            k = s & (tex == texture)
            if k.sum():
                want = label[k][0]
                got = Counter(seen[k])
                wrong = [f"({x},{y})={seen[y, x]}" for y, x in np.argwhere(k) if seen[y, x] != want]
                print(f"   {want}/{texture:5s} {int(k.sum()):3d} of {int((mask & (tex == texture)).sum()):3d} tiles seen, "
                      f"read {want} {got.get(want, 0) / k.sum():.3f}" + (f"  misses {' '.join(wrong[:12])}" if wrong else ""))
        k = s & (label == "b")
        if k.sum():
            wrong = [f"({x},{y})={seen[y, x]}" for y, x in np.argwhere(k) if seen[y, x] != "b"]
            print(f"   bush misses: {' '.join(wrong[:16])}" if wrong else "   bush: no misses")
