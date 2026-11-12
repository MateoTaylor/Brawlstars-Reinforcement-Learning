"""Odometry over every frame + masked NCC of every 10th rectified frame against the map image.

Writes reg.npz: per frame odometry position/status/segment, and per sampled frame the NCC camera
position P (map tiles; rectified pixel (0,0) sits at world origin_tile + P), its score and the
best score more than 2 tiles away (uniqueness)."""
import json, sys, time
from pathlib import Path
import cv2, numpy as np
from common import CLIP, MAP, MAP_DIR, WORK  # first: puts the repo on sys.path
from brawl_vision.config import load_vision_config
from brawl_vision.clips import load_bounds
from brawl_vision.gameplay import walk_frames
from brawl_vision.terrain.labeling import hud_plans, find_clip
from brawl_vision.terrain.odometry import Odometry
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 10
plan = hud_plans()["emulator"]
cfg = load_vision_config()
odo = Odometry(plan, cfg)
ppt = plan.pixels_per_tile
ox, oy = plan.origin_tile

meta = json.loads((MAP_DIR / f"{MAP}.json").read_text())
img = cv2.imread(str(MAP_DIR / f"{MAP}.png"))
PAD = 14
def canvas_at(scale):
    (px, py), (mx, my) = meta["pitch_px"], meta["origin_px"]
    A = np.array([[scale / px, 0, -scale * mx / px], [0, scale / py, -scale * my / py]], np.float64)
    A[:, 2] += A[:, :2] @ np.array([0.5, 0.5]) - 0.5 + PAD * scale
    n = (60 + 2 * PAD) * scale
    return cv2.warpAffine(img, A, (n, n), flags=cv2.INTER_AREA if scale < 16 else cv2.INTER_CUBIC,
                          borderValue=(0, 0, 0)).astype(np.float32)
C12, C48 = canvas_at(12), canvas_at(48)
mask48 = cv2.erode(plan.valid.astype(np.uint8), np.ones((9, 9), np.uint8))
mask12 = cv2.resize(mask48, (plan.size_px[0] // 4, plan.size_px[1] // 4), interpolation=cv2.INTER_AREA)
mask12 = (mask12 > 0).astype(np.uint8)

def peak_sub(res, x, y):
    def para(a, b, c):
        d = a - 2 * b + c
        return 0.0 if d >= 0 else 0.5 * (a - c) / d
    h, w = res.shape
    dx = para(res[y, x - 1], res[y, x], res[y, x + 1]) if 0 < x < w - 1 else 0.0
    dy = para(res[y - 1, x], res[y, x], res[y + 1, x]) if 0 < y < h - 1 else 0.0
    return x + dx, y + dy

def register(rect):
    t12 = cv2.resize(rect, (rect.shape[1] // 4, rect.shape[0] // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    r = cv2.matchTemplate(C12, t12, cv2.TM_CCOEFF_NORMED, mask=mask12)
    r = np.nan_to_num(r, nan=-1, posinf=-1, neginf=-1)
    _, best, _, (x, y) = cv2.minMaxLoc(r)
    far = r.copy(); far[max(0, y - 24):y + 25, max(0, x - 24):x + 25] = -1
    second = float(far.max())
    # refine at 48 px/tile in a +/-1.5 tile window
    X0, Y0 = x * 4 - 72, y * 4 - 72
    th, tw = rect.shape[:2]
    X0c, Y0c = max(0, X0), max(0, Y0)
    sub = C48[Y0c:Y0c + th + 144, X0c:X0c + tw + 144]
    r2 = cv2.matchTemplate(sub, rect.astype(np.float32), cv2.TM_CCOEFF_NORMED, mask=mask48)
    r2 = np.nan_to_num(r2, nan=-1, posinf=-1, neginf=-1)
    _, best2, _, (x2, y2) = cv2.minMaxLoc(r2)
    sx, sy = peak_sub(r2, x2, y2)
    cx, cy = X0c + sx, Y0c + sy                       # canvas px of rectified (0,0)
    P = (cx / 48 - PAD - ox, cy / 48 - PAD - oy)
    return P, float(best2), float(best), second

path = find_clip(CLIP)
bounds = load_bounds(path)
n = bounds["usable"][1] + 1
pos = np.full((n, 2), np.nan); status = np.empty(n, "<U9"); seg = np.zeros(n, int); inl = np.zeros(n)
reg_idx, reg_P, reg_s48, reg_s12, reg_2nd = [], [], [], [], []
t0 = time.time()
for i, im in walk_frames(path, tuple(bounds["content_box"]), range(n)):
    if im.shape[1::-1] != tuple(plan.viewport):
        im = cv2.resize(im, plan.viewport, interpolation=cv2.INTER_AREA)
    rect = plan.rectify(im)
    o = odo.update(rect)
    pos[i] = o.position_tiles; status[i] = o.status; seg[i] = o.segment; inl[i] = o.inlier_ratio
    if i % STEP == 0:
        P, s48, s12, s2 = register(rect)
        reg_idx.append(i); reg_P.append(P); reg_s48.append(s48); reg_s12.append(s12); reg_2nd.append(s2)
    if i % 300 == 0:
        print(f"{i}/{n}  {time.time() - t0:.0f}s  odo {o.status} seg {o.segment} pos {o.position_tiles}", flush=True)
WORK.mkdir(parents=True, exist_ok=True)
np.savez(WORK / "reg.npz", pos=pos, status=status, seg=seg, inl=inl, reg_idx=np.array(reg_idx),
         reg_P=np.array(reg_P), reg_s48=np.array(reg_s48), reg_s12=np.array(reg_s12), reg_2nd=np.array(reg_2nd))
print("done", time.time() - t0)
