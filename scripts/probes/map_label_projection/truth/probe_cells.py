"""For given map tiles, each model's per-frame class at that tile over every registered frame
(coverage.json, every 10th frame, bad windows skipped), with the cell's row j in the view (0 = top).
    python probe_cells.py name=model.pt [name=model.pt ...] -- x,y x,y ..."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(REPO))
from brawl_vision.terrain.labeling import CLASSES, hud_plans, observed_cells, read_label_frames
from brawl_vision.terrain.classifier import TerrainClassifier
SCR = Path(__file__).resolve().parent
cut = sys.argv.index("--")
models = {s.split("=", 1)[0]: TerrainClassifier.load(s.split("=", 1)[1]) for s in sys.argv[1:cut]}
tiles = [tuple(int(v) for v in s.split(",")) for s in sys.argv[cut + 1:]]
CH = {"FLOOR": ".", "WALL": "#", "BUSH": "b", "WATER": "~", "FENCE": "f"}
chars = np.array([CH[t.name] for t in CLASSES])
cov = {int(k): v for k, v in json.load(open(REPO / "runs" / "map_checks" / "truth" / "coverage.json")).items()}
bad = [(0, 520), (2390, 2480), (2630, 2840), (3340, 99999)]
S = (1.022, 1.004)
plan = hud_plans()["emulator"]; obs = observed_cells(plan); cols, rows = plan.size_tiles
frames = [f for f in sorted(cov) if not any(lo <= f <= hi for lo, hi in bad)]
images = read_label_frames("bluestacks-truth-west-upper", set(frames), plan.viewport)
hits = defaultdict(list)
for f in frames:
    A = cov[f]["A"]
    want = []
    for x, y in tiles:
        i, j = int((x + 0.5 - A[0]) * S[0]), int((y + 0.5 - A[1]) * S[1])
        if 0 <= i < cols and 0 <= j < rows and obs[j, i]:
            want.append((x, y, i, j))
    if not want:
        continue
    rect = plan.rectify(images[f])
    preds = {n: m.predict(rect)[0] for n, m in models.items()}
    for x, y, i, j in want:
        hits[(x, y)].append((f, i, j, {n: chars[p[j, i]] for n, p in preds.items()}))
print(f"view grid {cols} x {rows} cells; j = row in the view, 0 = top")
for t in tiles:
    h = hits.get(t, [])
    if not h:
        print(f"{t}: never in view"); continue
    summary = "  ".join(f"{n}: " + " ".join(f"{c}{k}" for c, k in Counter(r[3][n] for r in h).most_common()) for n in models)
    js = Counter(r[2] for r in h)
    print(f"{t}: {len(h)} frames, view rows {dict(sorted(js.items()))}  |  {summary}")
    print("     " + " ".join(f"f{r[0]}:j{r[2]}i{r[1]}:" + "".join(r[3][n] for n in models) for r in h))
