"""Every truth-wall tile a seen map reads as non-blocking, per model, with its row, texture and view confidence."""
import sys
from collections import Counter
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(REPO))
from brawl_deployment.perception.known_map import MAPS_DIR, read_chars
SCR = Path(__file__).resolve().parent
tex = np.array([l.rstrip("\n").split(",") for l in open(SCR / "textures.csv") if l.strip("\n")])
truth = read_chars(MAPS_DIR / "truth.csv")
yy, xx = np.mgrid[0:60, 0:60]; R = (xx <= 28) & (yy >= 29)
for spec in sys.argv[1:]:
    name, path = spec.split("=", 1); z = np.load(path); seen = z["seen"]; conf = z["conf"]
    miss = (truth == "#") & np.isin(seen, list(".b")) 
    fp = (truth == ".") & np.isin(seen, list("#~"))
    ys, xs = np.where(miss)
    print(f"\n== {name}: {miss.sum()} wall tiles read non-blocking; {fp.sum()} floor tiles read blocking")
    print("   misses by map row band:", dict(sorted(Counter(('R' if R[y,x] else '') + f"y{(y//4)*4:02d}-{(y//4)*4+3:02d}" for y, x in zip(ys, xs)).items())))
    print("   misses:", " ".join(f"({x},{y}){tex[y,x][:2]}{seen[y,x]}:{conf[y,x]:.2f}" for y, x in zip(ys, xs)))
    ys, xs = np.where(fp)
    print("   floor read blocking by row band:", dict(sorted(Counter(f"y{(y//4)*4:02d}" for y, x in zip(ys, xs)).items())))
