"""Labelled cells per wall texture in projected label folders (clip filter), and the tiles both share."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SCR = Path(__file__).resolve().parent
tex = np.array([l.rstrip("\n").split(",") for l in open(SCR / "textures.csv") if l.strip("\n")])
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from brawl_deployment.perception.known_map import MAPS_DIR, read_chars          # noqa: E402
chars = read_chars(MAPS_DIR / "truth.csv")
tiles = {}
for d in sys.argv[1:]:
    c, per_tex, seen_tex, n = Counter(), Counter(), Counter(), 0
    T = set()
    for p in sorted(Path(d).glob("bluestacks-truth-*.json")):
        g = json.loads(p.read_text())
        notes = g["notes"]
        S = [float(v) for v in re.search(r"scale \(([-\d.]+), ([-\d.]+)\)", notes).groups()]
        A = [float(v) for v in re.search(r"A=\(([-\d.]+), ([-\d.]+)\)", notes).groups()]
        ch = np.array([list(r) for r in g["grid"]])
        n += 1
        for j, i in zip(*np.nonzero(ch != "?")):
            x, y = int(np.floor(A[0] + (i + 0.5) / S[0])), int(np.floor(A[1] + (j + 0.5) / S[1]))
            c[ch[j, i]] += 1
            T.add((x, y))
            if ch[j, i] == "#":
                per_tex[tex[y, x] or "(none)"] += 1
    tiles[d] = T
    tt = Counter(tex[y, x] for x, y in T if tex[y, x])
    print(f"{d}: {n} frames, cells {dict(c)}\n   wall cells by texture {dict(per_tex)}\n   distinct tiles {len(T)}, wall tiles by texture {dict(tt)}")
ds = list(tiles)
if len(ds) == 2:
    print("shared tiles:", len(tiles[ds[0]] & tiles[ds[1]]))
