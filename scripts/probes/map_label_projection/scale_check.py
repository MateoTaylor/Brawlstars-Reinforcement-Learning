"""Best uniform scale of rectified frames against a map image: scale_check.py <map> <clip> <frames...>"""
import json, sys
from pathlib import Path
import cv2, numpy as np
from common import MAP_DIR  # first: puts the repo on sys.path
from brawl_vision.terrain.labeling import hud_plans, read_label_frames
MAP, CLIP, idx = sys.argv[1], sys.argv[2], [int(a) for a in sys.argv[3:]]
plan = hud_plans()["emulator"]
meta = json.loads((MAP_DIR / f"{MAP}.json").read_text())
img = cv2.imread(str(MAP_DIR / f"{MAP}.png"))
PAD = 14
def canvas_at(scale):
    (px, py), (mx, my) = meta["pitch_px"], meta["origin_px"]
    A = np.array([[scale / px, 0, -scale * mx / px], [0, scale / py, -scale * my / py]], np.float64)
    A[:, 2] += A[:, :2] @ np.array([0.5, 0.5]) - 0.5 + PAD * scale
    n = (60 + 2 * PAD) * scale
    return cv2.warpAffine(img, A, (n, n), flags=cv2.INTER_AREA, borderValue=(0, 0, 0)).astype(np.float32)
C24 = canvas_at(24)
mask48 = cv2.erode(plan.valid.astype(np.uint8), np.ones((9, 9), np.uint8))
fr = read_label_frames(CLIP, idx, plan.viewport)
for i, im in sorted(fr.items()):
    rect = plan.rectify(im).astype(np.float32)
    out = []
    for s in np.arange(0.96, 1.051, 0.01):
        t = cv2.resize(rect, None, fx=0.5 / s, fy=0.5 / s, interpolation=cv2.INTER_AREA)
        m = (cv2.resize(mask48.astype(np.float32), (t.shape[1], t.shape[0]), interpolation=cv2.INTER_AREA) > 0.99).astype(np.uint8)
        r = np.nan_to_num(cv2.matchTemplate(C24, t, cv2.TM_CCOEFF_NORMED, mask=m), nan=-1, posinf=-1, neginf=-1)
        out.append((round(float(s), 2), round(float(r.max()), 3)))
    best = max(out, key=lambda z: z[1])
    print(i, "best", best, " 1.00:", dict(out)[1.0], " all", out, flush=True)
