"""Contact sheets of disputed tiles: map-image crop | rectified game-frame crop, the tile outlined.
    python crops_disputed.py LOG OUT_PREFIX   (LOG holds replay_clip's 'for a look in map_label.py' line)"""
import json, re, sys
from pathlib import Path
import cv2
import numpy as np
REPO = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(REPO))
from brawl_vision.terrain.labeling import hud_plans, observed_cells, read_label_frames
SCR = Path(__file__).resolve().parent
log, out = sys.argv[1], sys.argv[2]
line = [l for l in open(log, encoding="utf-8", errors="replace") if "for a look in map_label.py" in l][0]
cells = [(int(c), int(r), a, b) for c, r, a, b in re.findall(r"\((\d+), (\d+)\) = (.), (.)", line)]
tex = np.array([l.rstrip("\n").split(",") for l in open(SCR / "textures.csv") if l.strip("\n")])
cov = {int(k): v for k, v in json.load(open(REPO / "runs" / "map_checks" / "truth" / "coverage.json")).items()}
S = (1.022, 1.004)
plan = hud_plans()["emulator"]; obs = observed_cells(plan); ppt = plan.pixels_per_tile
cols, rows = plan.size_tiles
bad = [(0, 520), (2390, 2480), (2630, 2840), (3340, 99999)]
lat = json.load(open(REPO / "brawl_deployment/data/maps/truth.json"))
mimg = cv2.imread(str(REPO / "brawl_deployment/data/maps/truth.png"))
pick = {}
for x, y, a, b in cells:
    best = None
    for f, v in cov.items():
        if any(lo <= f <= hi for lo, hi in bad):
            continue
        A = v["A"]
        i, j = (x + 0.5 - A[0]) * S[0], (y + 0.5 - A[1]) * S[1]
        if not (2.5 <= i < cols - 2.5 and 2.5 <= j < rows - 2.5) or not obs[int(j), int(i)]:
            continue
        d = abs(i - cols / 2) + abs(j - rows * 0.55)
        if best is None or d < best[0]:
            best = (d, f)
    pick[(x, y)] = best[1] if best else None
frames = sorted({f for f in pick.values() if f is not None})
images = read_label_frames("bluestacks-truth-west-upper", frames, plan.viewport)
rects = {f: plan.rectify(images[f]) for f in frames}
Z = 210
tiles = []
for x, y, a, b in cells:
    f = pick[(x, y)]
    px, py = lat["pitch_px"]; ox, oy = lat["origin_px"]
    m = mimg[int(oy + py * (y - 2)):int(oy + py * (y + 3)), int(ox + px * (x - 2)):int(ox + px * (x + 3))]
    m = cv2.resize(m, (Z, Z), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(m, (2 * Z // 5, 2 * Z // 5), (3 * Z // 5, 3 * Z // 5), (0, 255, 255), 2)
    if f is None:
        g = np.zeros((Z, Z, 3), np.uint8)
    else:
        A = cov[f]["A"]
        x0, x1 = (x - A[0]) * S[0] * ppt, (x + 1 - A[0]) * S[0] * ppt
        y0, y1 = (y - A[1]) * S[1] * ppt, (y + 1 - A[1]) * S[1] * ppt
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        h = 2.5 * ppt
        r = rects[f]
        c = r[int(max(cy - h, 0)):int(cy + h), int(max(cx - h, 0)):int(cx + h)].copy()
        sx, sy = Z / c.shape[1], Z / c.shape[0]
        ox0, oy0 = max(cx - h, 0), max(cy - h, 0)
        g = cv2.resize(c, (Z, Z))
        cv2.rectangle(g, (int((x0 - ox0) * sx), int((y0 - oy0) * sy)), (int((x1 - ox0) * sx), int((y1 - oy0) * sy)), (0, 255, 255), 2)
    pair = np.full((Z + 26, 2 * Z + 6, 3), 255, np.uint8)
    pair[26:, :Z] = m; pair[26:, Z + 6:] = g
    cv2.putText(pair, f"({x},{y}) label {a} seen {b} {tex[y, x] or ''} f{f}", (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    tiles.append(pair)
per = 12
for s in range(0, len(tiles), per):
    chunk = tiles[s:s + per]
    while len(chunk) % 3:
        chunk.append(np.full_like(tiles[0], 255))
    rows_ = [np.hstack([np.pad(t, ((4, 4), (4, 4), (0, 0)), constant_values=255) for t in chunk[k:k + 3]]) for k in range(0, len(chunk), 3)]
    cv2.imwrite(f"{out}_{s // per}.jpg", np.vstack(rows_), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{out}_{s // per}.jpg", len(chunk))
