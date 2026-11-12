"""Alignment check for projected labels: one model's single-frame reading against each label shifted
by (di, dj) cells. Right labels agree best at (0, 0); a registration a cell off peaks elsewhere.

    python label_shift.py --labels DIR --model PT
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from brawl_vision.terrain.classifier import TerrainClassifier                           # noqa: E402
from brawl_vision.terrain.labeling import CLASSES, hud_plans, load_label_dir, read_label_frames  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--model", required=True)
a = ap.parse_args()
NAMES = [t.name for t in CLASSES]
blocking = np.array([n in ("WALL", "WATER", "FENCE") for n in NAMES])
plans = hud_plans()
grids = load_label_dir(a.labels, plans["phone"])
model = TerrainClassifier.load(a.model, "cpu")
R = 2
agree = np.zeros((2 * R + 1, 2 * R + 1))
total = np.zeros_like(agree)
tp = np.zeros_like(agree); fp = np.zeros_like(agree); fn = np.zeros_like(agree)
by_clip = {}
for g in grids:
    by_clip.setdefault(g.clip, []).append(g)
for clip, gs in by_clip.items():
    plan = plans[gs[0].hud]
    frames = read_label_frames(clip, {g.frame for g in gs}, plan.viewport)
    for g in gs:
        cells, _ = model.predict(plan.rectify(frames[g.frame]), plan)
        lab = g.as_class_index()
        rows, cols = lab.shape
        for dj in range(-R, R + 1):
            for di in range(-R, R + 1):
                # label cell (j, i) compared with prediction cell (j + dj, i + di)
                l = lab[max(0, -dj):rows - max(0, dj), max(0, -di):cols - max(0, di)]
                p = cells[max(0, dj):rows - max(0, -dj) if -dj > 0 else rows, max(0, di):cols - max(0, -di) if -di > 0 else cols]
                p = cells[max(0, dj):max(0, dj) + l.shape[0], max(0, di):max(0, di) + l.shape[1]]
                ok = l >= 0
                agree[dj + R, di + R] += (l[ok] == p[ok]).sum()
                total[dj + R, di + R] += ok.sum()
                lb, pb = blocking[l[ok]], blocking[p[ok]]
                tp[dj + R, di + R] += (lb & pb).sum()
                fp[dj + R, di + R] += (~lb & pb).sum()
                fn[dj + R, di + R] += (lb & ~pb).sum()
print(f"{a.model} on {a.labels} ({len(grids)} frames): agreement / blocking F1 with the label shifted "
      f"(rows dj = prediction row offset, cols di)")
for dj in range(-R, R + 1):
    print(f"  dj {dj:+d}  " + "  ".join(
        f"{agree[dj + R, di + R] / total[dj + R, di + R]:.3f}/{2 * tp[dj + R, di + R] / (2 * tp[dj + R, di + R] + fp[dj + R, di + R] + fn[dj + R, di + R]):.3f}"
        for di in range(-R, R + 1)))
