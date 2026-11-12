"""Score replay seen-maps (replay_seen.py npz) against truth.csv, on the held-out region R (map x 0-28,
y 29-59: no training label ever touched it) and, for reference, on the training ground outside it.
Per class (fence merged into wall), per wall texture, and bush/water confusion.

    python region_score.py NAME=seen.npz [NAME=seen.npz ...]
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from brawl_deployment.perception.known_map import MAPS_DIR, read_chars          # noqa: E402

SCR = Path(__file__).resolve().parent
tex = np.array([l.rstrip("\n").split(",") for l in open(SCR / "textures.csv") if l.strip("\n")])
truth = np.where(read_chars(MAPS_DIR / "truth.csv") == "f", "#", read_chars(MAPS_DIR / "truth.csv"))
yy, xx = np.mgrid[0:60, 0:60]
R = (xx <= 28) & (yy >= 29)
BLOCK = set("#~")
for spec in sys.argv[1:]:
    name, path = spec.split("=", 1)
    z = np.load(path)
    seen = np.where(z["seen"] == "f", "#", z["seen"])
    for region, mask in (("held-out R", R), ("training ground", ~R)):
        s = mask & (seen != "?")
        t, v = truth[s], seen[s]
        agree = (t == v).mean()
        tb, vb = np.isin(t, list(BLOCK)), np.isin(v, list(BLOCK))
        tp = (tb & vb).sum()
        f1 = 2 * tp / max(tb.sum() + vb.sum(), 1)
        print(f"\n== {name}, {region}: {s.sum()} tiles seen of {mask.sum()}; 5-class {agree:.3f}; "
              f"blocking F1 {f1:.3f} (recall {tp / max(tb.sum(), 1):.3f}, precision {tp / max(vb.sum(), 1):.3f})")
        for c, label in ((".", "floor"), ("#", "wall"), ("b", "bush"), ("~", "water")):
            k = t == c
            if k.sum():
                got = Counter(v[k])
                print(f"   {label:6s} n {k.sum():4d}  read as: " + ", ".join(f"{g} {n}" for g, n in got.most_common()))
        for texture in ("stone", "crate", "barrel", "mailbox", "teleporter"):
            k = s & (tex == texture)
            if k.sum():
                got = Counter(seen[k])
                wrong = [f"({x},{y})={seen[y, x]}" for y, x in np.argwhere(k) if seen[y, x] != "#"]
                print(f"   wall/{texture:10s} {int(k.sum()):3d} of {int((mask & (tex == texture)).sum()):3d} tiles seen, "
                      f"read # {got.get('#', 0) / k.sum():.3f}" + (f"  misses {' '.join(wrong[:12])}" if wrong else ""))
