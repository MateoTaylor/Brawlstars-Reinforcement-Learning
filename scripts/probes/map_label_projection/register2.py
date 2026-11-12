"""Re-register sampled frames at the measured scale, plus 3x3 block residuals.

For frame f: template = rectified frame resized by (1/sx, 1/sy) to true 48 px/tile, matched in a
+/-1.5 tile window of the 48 px/tile canvas around the scale-1 estimate. A = world of rect px (0,0).
Blocks: each of a 3x3 grid of template blocks matched locally (+/-0.75 tile) -> residual shift."""
import json, sys
from pathlib import Path
import cv2, numpy as np
from common import CLIP, MAP, MAP_DIR, WORK  # first: puts the repo on sys.path
from brawl_vision.clips import load_bounds
from brawl_vision.gameplay import walk_frames
from brawl_vision.terrain.labeling import hud_plans, find_clip
SX, SY = float(sys.argv[1]), float(sys.argv[2])
STEP = int(sys.argv[3]) if len(sys.argv) > 3 else 10
plan = hud_plans()["emulator"]
ox, oy = plan.origin_tile
meta = json.loads((MAP_DIR / f"{MAP}.json").read_text())
img = cv2.imread(str(MAP_DIR / f"{MAP}.png"))
PAD = 14
(px, py), (mx, my) = meta["pitch_px"], meta["origin_px"]
A_ = np.array([[48 / px, 0, -48 * mx / px], [0, 48 / py, -48 * my / py]], np.float64)
A_[:, 2] += A_[:, :2] @ np.array([0.5, 0.5]) - 0.5 + PAD * 48
N = (60 + 2 * PAD) * 48
C48 = cv2.warpAffine(img, A_, (N, N), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0)).astype(np.float32)
d = np.load(WORK / "reg.npz"); pos0 = d["pos"]; ri0 = list(d["reg_idx"]); P0 = d["reg_P"]
valid8 = cv2.erode(plan.valid.astype(np.uint8), np.ones((9, 9), np.uint8))
tw, th = plan.size_px
TW, TH = int(round(tw / SX)), int(round(th / SY))
mask_s = (cv2.resize(valid8.astype(np.float32), (TW, TH), interpolation=cv2.INTER_AREA) > 0.99).astype(np.uint8)

def peak_sub(res, x, y):
    def para(a, b, c):
        dd = a - 2 * b + c
        return 0.0 if dd >= 0 else 0.5 * (a - c) / dd
    h, w = res.shape
    dx = para(res[y, x - 1], res[y, x], res[y, x + 1]) if 0 < x < w - 1 else 0.0
    dy = para(res[y - 1, x], res[y, x], res[y + 1, x]) if 0 < y < h - 1 else 0.0
    return x + dx, y + dy

def match(tmpl, mask, X, Y, rad):
    """Best canvas px of tmpl's (0,0) within +/-rad px of (X, Y)."""
    X0, Y0 = int(round(X)) - rad, int(round(Y)) - rad
    sub = C48[max(0, Y0):Y0 + tmpl.shape[0] + 2 * rad, max(0, X0):X0 + tmpl.shape[1] + 2 * rad]
    r = np.nan_to_num(cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED, mask=mask), nan=-1, posinf=-1, neginf=-1)
    _, best, _, (x, y) = cv2.minMaxLoc(r)
    sx, sy = peak_sub(r, x, y)
    return max(0, X0) + sx, max(0, Y0) + sy, float(best)

def register(rect, guess_A):
    t = cv2.resize(rect, (TW, TH), interpolation=cv2.INTER_AREA).astype(np.float32)
    X, Y = (guess_A[0] + PAD) * 48, (guess_A[1] + PAD) * 48
    cx, cy, sc = match(t, mask_s, X, Y, 72)
    A = (cx / 48 - PAD, cy / 48 - PAD)
    blocks = []
    bw, bh = TW // 3, TH // 3
    for by in range(3):
        for bx in range(3):
            tb = t[by * bh:(by + 1) * bh, bx * bw:(bx + 1) * bw]
            mb = mask_s[by * bh:(by + 1) * bh, bx * bw:(bx + 1) * bw]
            if mb.mean() < 0.5:
                blocks.append((np.nan, np.nan, np.nan)); continue
            bxx, byy, bs = match(np.ascontiguousarray(tb), np.ascontiguousarray(mb), cx + bx * bw, cy + by * bh, 36)
            blocks.append(((bxx - cx - bx * bw) / 48, (byy - cy - by * bh) / 48, bs))
    return A, sc, blocks

if __name__ == "__main__":
    path = find_clip(CLIP); bounds = load_bounds(path)
    idx = [i for i in ri0 if i >= 430 and i % STEP == 0]
    out_i, out_A, out_s, out_b = [], [], [], []
    for i, im in walk_frames(path, tuple(bounds["content_box"]), idx):
        im = cv2.resize(im, plan.viewport, interpolation=cv2.INTER_AREA)
        rect = plan.rectify(im)
        P = P0[ri0.index(i)]
        # scale-1 estimate refers to the frame centre roughly: A_guess so the centres agree
        cen = np.array([tw / 2, th / 2]) / 48
        guess = (P[0] + ox + cen[0] - cen[0] / SX, P[1] + oy + cen[1] - cen[1] / SY)
        A, sc, blocks = register(rect, guess)
        out_i.append(i); out_A.append(A); out_s.append(sc); out_b.append(blocks)
    np.savez(WORK / f"reg2_{SX}_{SY}.npz", idx=np.array(out_i), A=np.array(out_A), score=np.array(out_s), blocks=np.array(out_b))
    print("done", len(out_i))
