"""Score replay seen-maps (../truth/replay_seen.py npz) against doom_shroom.csv, on the held-out
region R (pick_frames.py's: map x 0-13 without the bounce pad, plus the planter patch 21-23, 33-35;
no training label ever touched it) and, for reference, on the ground outside it. Per class, per
texture (walls and floor specials, read as their label class or not), and bush/water confusion.
Spawn tiles score as floor. Truth's copy scores wall textures only.

    python region_score.py [--r2] NAME=seen.npz [NAME=seen.npz ...]

`--r2` also scores region R2 (pick_frames_east.py's: map x 52-59), the second recording's held-out
strip, and the training ground becomes what is in neither. Without it the output is as it was for
the first recording, where the training ground was everything outside R.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from brawl_deployment.perception.known_map import MAPS_DIR, read_chars          # noqa: E402

SCR = Path(__file__).resolve().parent
tex =np.array([l.rstrip("\n").split(",") for l in open(SCR / "textures.csv") if l.strip("\n")])
label = read_chars(MAPS_DIR / "doom_shroom.csv")
label = np.where(label == "S", ".", label)
yy, xx = np.mgrid[0:60, 0:60]
R = ((xx <= 13) | ((xx >= 21) & (xx <= 23) & (yy >= 33) & (yy <= 35))) & ~(
    (xx >= 7) & (xx <= 8) & (yy >= 15) & (yy <= 16))
BLOCK = set("#~")
TEXTURES = ("pink_cross", "pink_plain", "gray", "barrel", "planter", "mailbox", "orange",
            "speed", "fence", "bounce", "heal", "teleporter")
args = sys.argv[1:]
R2 = xx >= 52
regions = (("held-out R", R), ("training ground", ~R))
if args[:1] == ["--r2"]:
    args = args[1:]
    regions = (("held-out R", R), ("held-out R2", R2), ("training ground", ~R & ~R2))
for spec in args:
    name, path = spec.split("=", 1)
    seen = np.load(path)["seen"]
    for region, mask in regions:
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
                print(f"   {want}/{texture:10s} {int(k.sum()):3d} of {int((mask & (tex == texture)).sum()):3d} tiles seen, "
                      f"read {want} {got.get(want, 0) / k.sum():.3f}" + (f"  misses {' '.join(wrong[:12])}" if wrong else ""))
