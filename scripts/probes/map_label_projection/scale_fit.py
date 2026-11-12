"""Measure the rectified-px-per-world-tile scale S from odometry vs map-image NCC (reads WORK/reg.npz).

Over the longest odometry segment, regress the scale-1 NCC camera position P (map tiles) on the
odometry position (rectified tiles): P = pos / S + c, so S = 1 / slope per axis. Starfruit, 143
samples with NCC score > 0.4: x slope 0.9784 -> S 1.0221 (resid sd 0.157 tile), y slope 0.9962 ->
S 1.0038 (resid sd 0.117). register2.py and make_labels.py use S = (1.022, 1.004).
"""
import numpy as np

from common import WORK

d = np.load(WORK / "reg.npz")
pos, seg, ri, P, s48 = d["pos"], d["seg"], d["reg_idx"], d["reg_P"], d["reg_s48"]
good = s48 > 0.4
segment = int(np.bincount(seg[ri][good]).argmax())
k = good & (seg[ri] == segment)
o, p = pos[ri][k], P[k]
for ax, name in ((0, "x"), (1, "y")):
    A = np.vstack([o[:, ax], np.ones(k.sum())]).T
    (m, c), *_ = np.linalg.lstsq(A, p[:, ax], rcond=None)
    r = p[:, ax] - (m * o[:, ax] + c)
    print(f"{name}: slope {m:.4f} => S {1 / m:.4f}  intercept {c:.3f}  resid sd {r.std():.3f}  "
          f"max {np.abs(r).max():.3f}  n {int(k.sum())} (segment {segment})")
