"""Draw LabelGrids over their rectified frames: class tint + letter on labelled cells, '?' dimmed.
usage: python review.py LABELDIR OUTDIR [aux]

With a third argument, cells make_labels.py refused draw a diagonal: white = under a detection or
transient, yellow = gas (from WORK/lab/aux_<frame>.npz)."""
import sys
from pathlib import Path
import cv2, numpy as np
from common import WORK  # first: puts the repo on sys.path
from brawl_vision.terrain.labeling import LabelGrid, hud_plans, read_label_frames

COL = {".": (200, 200, 200), "#": (40, 40, 255), "b": (40, 230, 40), "~": (255, 170, 0), "f": (0, 230, 255)}

def draw(rect, grid, ppt=48):
    out = rect.copy()
    dim = (rect * 0.35).astype(np.uint8)
    rows, cols = grid.chars.shape
    for j in range(rows):
        for i in range(cols):
            c = grid.chars[j, i]
            x0, y0 = i * ppt, j * ppt
            if c == "?":
                out[y0:y0 + ppt, x0:x0 + ppt] = dim[y0:y0 + ppt, x0:x0 + ppt]
                continue
            col = COL[c]
            tile = out[y0:y0 + ppt, x0:x0 + ppt]
            if c != ".":
                tile[:] = (0.72 * tile + 0.28 * np.array(col)).astype(np.uint8)
            cv2.rectangle(out, (x0 + 1, y0 + 1), (x0 + ppt - 2, y0 + ppt - 2), col, 1)
            if c != ".":
                cv2.putText(out, c, (x0 + 15, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                cv2.putText(out, c, (x0 + 15, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    return out

if __name__ == "__main__":
    labdir, outdir = Path(sys.argv[1]), Path(sys.argv[2]); outdir.mkdir(parents=True, exist_ok=True)
    plan = hud_plans()["emulator"]
    grids = [LabelGrid.load(p, plan) for p in sorted(labdir.glob("*.json"))]
    frames = read_label_frames(grids[0].clip, [g.frame for g in grids], plan.viewport)
    for g in grids:
        rect = plan.rectify(frames[g.frame])
        img = draw(rect, g)
        aux = WORK / "lab" / f"aux_{g.frame}.npz"
        if len(sys.argv) > 3 and aux.exists():
            a = np.load(aux)
            for key, col in (("occ", (255, 255, 255)), ("gas", (0, 255, 255))):
                for j, i in zip(*np.nonzero(a[key])):
                    cv2.line(img, (i * 48 + 4, j * 48 + 4), (i * 48 + 44, j * 48 + 44), col, 1)
        cv2.imwrite(str(outdir / f"review_f{g.frame}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(g.frame, g.count())
