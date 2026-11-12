"""Classifier labels for the Starfruit Supernova clip, projected from the map label.

Each frame's registration: A = world tile of rectified px (0,0), at scale S (rect px per world
px is S). A comes from the map-image NCC (register2.py), smoothed through odometry: the offset
A - pos/S is a running median over +/-3 samples (+/-30 frames), so one bad NCC peak cannot move a
label.

A cell's label is the majority GROUND-FOOTPRINT class of the world area under it, kept only when
that class holds >= PURITY of the cell at every one of 9 shifts of +/-DELTA tiles (registration
error), and only for classes the classifier has (meeples and off-map never label). Then the
labels README's refusals: unobserved/HUD cells, gas, and cells a brawler, crate, cube or
projectile covers (the detector boxes already take in the name, HP bar and ammo bar), plus the
brawlers-left text, the kill card and cells that change against the frames 8 either side.

usage: python make_labels.py OUTDIR FRAME [FRAME ...]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from common import CLIP, WORK  # first: puts the repo on sys.path
from project import class_grid, NAMES, IDX  # noqa: E402
from brawl_vision.config import load_vision_config  # noqa: E402
from brawl_vision.terrain.labeling import (LabelGrid, hud_plans, read_label_frames,  # noqa: E402
                                           observed_cells)
from brawl_vision.terrain.zone import detect_zone  # noqa: E402

S = np.array([1.022, 1.004])          # scale_fit.py
REG2 = WORK / "reg2_1.022_1.004.npz"  # register2.py 1.022 1.004 10
PURITY, DELTA, SUB = 0.60, 0.12, 16
LABELLABLE = set(".#b~f")
GAS_FRACTION = 0.05
OCCLUDED_FRACTION = 0.10


def smoothed_A():
    r = np.load(WORK / "reg.npz")
    d = np.load(REG2)
    idx, A = d["idx"], d["A"]
    off = A - r["pos"][idx] / S
    sm = np.array([np.median(off[max(0, k - 3):k + 4], axis=0) for k in range(len(idx))])
    pos = r["pos"]

    def at(frame):
        # offset interpolated between the 10-frame samples, then the frame's own odometry
        ox = np.interp(frame, idx, sm[:, 0])
        oy = np.interp(frame, idx, sm[:, 1])
        return pos[frame] / S + np.array([ox, oy])
    return at


def fractions_all(g, A, size_tiles, shift=(0.0, 0.0)):
    """(rows, cols, len(NAMES)) class shares of each cell's world window, by SUBxSUB sampling."""
    cols, rows = size_tiles
    u = (np.arange(SUB) + 0.5) / SUB
    jj, ii = np.mgrid[0:rows, 0:cols]
    x = A[0] + shift[0] + (ii[..., None, None] + u[None, None, None, :]) / S[0]
    y = A[1] + shift[1] + (jj[..., None, None] + u[None, None, :, None]) / S[1]
    x = np.broadcast_to(x, (rows, cols, SUB, SUB))
    y = np.broadcast_to(y, (rows, cols, SUB, SUB))
    tx, ty = np.floor(x).astype(int), np.floor(y).astype(int)
    inside = (tx >= 0) & (tx < 60) & (ty >= 0) & (ty < 60)
    k = np.full(x.shape, IDX["X"], np.int8)
    k[inside] = g[ty[inside], tx[inside]]
    out = np.stack([(k == c).mean(axis=(2, 3)) for c in range(len(NAMES))], axis=-1)
    return out


def robust_chars(g, A, size_tiles):
    cols, rows = size_tiles
    agree = np.ones((rows, cols), bool)
    best = None
    for dx in (-DELTA, 0.0, DELTA):
        for dy in (-DELTA, 0.0, DELTA):
            f = fractions_all(g, A, size_tiles, (dx, dy))
            k = f.argmax(-1)
            ok = f.max(-1) >= PURITY
            if best is None:
                best = k
            agree &= ok & (k == best)
    chars = np.full((rows, cols), "?", "<U1")
    for c in LABELLABLE:
        chars[agree & (best == IDX[c])] = c
    return chars


_DETECTORS = None


def detectors(cfg):
    global _DETECTORS
    if _DETECTORS is None:
        from brawl_vision.object_detection.detector import ObjectDetector
        from brawl_vision.object_detection.projectile_detection.detect import ProjectileDetector
        _DETECTORS = (ObjectDetector.from_config(cfg), ProjectileDetector.from_config(cfg))
    return _DETECTORS


# RECT px added around each detection's rectified bounding box: (sides, above, below). The boxes
# already take in the name tag, HP and ammo bars (frame 600: the hero's box tops out at the name
# tag, rect y 356 vs 362; a crate's at its "5000", 378 vs 382), so this is only a margin.
RECT_PAD = {"brawler": (14, 14, 8), "Power Cube Box": (10, 10, 6), "other": (8, 8, 8)}
# The "Brawlers left: 10" readout runs past the emulator mask's brawlers_left rect on these 1440p
# captures (text to x~455 against the mask's 400), so this clip gets a wider one. Viewport px.
TEXT_BOX = (0, 0, 480, 115)
# Likewise the kill card: its names reach x~447 and its drop shadow y~262 (frames 910/1000/1840/2140)
# against the mask's kill_feed x<=440, y<=252.
KILL_BOX = (0, 110, 470, 270)


def _rect_box(plan, xyxy):
    x0, y0, x1, y1 = xyxy
    q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(q, plan.M).reshape(-1, 2)
    return q


def _coverage(plan, polys):
    cols, rows = plan.size_tiles
    ppt = plan.pixels_per_tile
    mask = np.zeros((rows * ppt, cols * ppt), np.float32)
    for p in polys:
        if np.isfinite(p).all():
            cv2.fillConvexPoly(mask, np.clip(p, -1e4, 1e4).round().astype(np.int32), 1.0)
    return cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_AREA)


def occluders(image, plan, cfg):
    ents, projs = detectors(cfg)
    boxes = [("brawler", d) for d in ents.predict(image)]
    boxes += [(d.label if d.label in RECT_PAD else "other", d) for d in projs.predict(image)]
    polys = []
    for kind, d in boxes:
        q = _rect_box(plan, d.xyxy)
        if not np.isfinite(q).all():
            continue
        side, above, below = RECT_PAD[kind]
        x0, y0 = q.min(0)
        x1, y1 = q.max(0)
        polys.append(np.array([[x0 - side, y0 - above], [x1 + side, y0 - above],
                               [x1 + side, y1 + below], [x0 - side, y1 + below]]))
    return _coverage(plan, polys) > OCCLUDED_FRACTION, boxes


# Transient content the detectors miss (shot splashes, damage numbers, shells in flight): compare the
# frame with frames +/-NEIGHBOUR, shifted by the odometry delta (static ground lines up to a median
# max-channel difference of 2). A cell is dropped when more than TRANSIENT_FRACTION of its pixels
# differ by more than TRANSIENT_DIFF from EITHER neighbour. Grass sway and water ripple stay under it
# (measured on frame 640); a splash reads 24-100 %. A full-screen flash (a super, frame 722) fails
# most of the view, so pick a different frame rather than raise the threshold.
NEIGHBOUR, TRANSIENT_DIFF, TRANSIENT_FRACTION, MIN_ALIGNED_PX = 8, 40, 0.12, 400


def transient_cells(frame, rect, neighbours, plan, pos):
    cols, rows = plan.size_tiles
    ppt = plan.pixels_per_tile
    worst = np.zeros((rows, cols))
    here = rect.astype(np.int16)
    for n, img in neighbours.items():
        d = (pos[n] - pos[frame]) * ppt
        M = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
        size = (rect.shape[1], rect.shape[0])
        moved = cv2.warpAffine(plan.rectify(img), M, size).astype(np.int16)
        seen = plan.valid & (cv2.warpAffine(plan.valid.astype(np.uint8), M, size) > 0)
        big = (np.abs(moved - here).max(-1) > TRANSIENT_DIFF) & seen
        cnt = seen.reshape(rows, ppt, cols, ppt).sum((1, 3))
        hit = big.reshape(rows, ppt, cols, ppt).sum((1, 3))
        frac = np.where(cnt >= MIN_ALIGNED_PX, hit / np.maximum(cnt, 1), 0.0)
        worst = np.maximum(worst, frac)
    return worst > TRANSIENT_FRACTION


def label_frame(frame, image, plan, cfg, g, A, neighbours, pos):
    rect = plan.rectify(image)
    chars = robust_chars(g, A, plan.size_tiles)
    n_proj = int((chars != "?").sum())
    unobs = ~observed_cells(plan)
    gas = detect_zone(rect, plan, cfg).at_least(GAS_FRACTION)
    gas = cv2.dilate(gas.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    occ, raw = occluders(image, plan, cfg)
    text = _coverage(plan, [_rect_box(plan, TEXT_BOX), _rect_box(plan, KILL_BOX)]) > 0.02
    moving = transient_cells(frame, rect, neighbours, plan, pos) & ~unobs
    for m in (unobs, gas, occ, text, moving):
        chars[m] = "?"
    stats = {"projected": n_proj, "unobserved": int(unobs.sum()), "gas": int(gas.sum()),
             "occluded": int(occ.sum()), "text": int(text.sum()), "transient": int(moving.sum()),
             "boxes": [(k, d.label, round(d.confidence, 2)) for k, d in raw],
             "labelled": int((chars != "?").sum())}
    return rect, chars, gas, occ | moving, stats


NOTE = ("Machine-made 2026-11-04, not hand-drawn: projected from "
        "brawl_deployment/data/maps/starfruit_supernova.csv. Registration: map-image NCC at scale "
        "(1.022, 1.004) smoothed through odometry, A=({ax:.3f}, {ay:.3f}) = world tile of rect px "
        "(0,0). A cell is its majority ground-footprint class when that class holds >= 0.60 of it "
        "under every +/-0.12-tile shift; meeple cells (map rows 26/33, wall-or-fence unsettled), "
        "off-map, gas (plus one cell), HUD, the brawlers-left text and kill card (both wider here than the mask), cells under detected "
        "brawlers/crates/cubes/projectiles, and cells that change against the frames 8 either side "
        "(splashes, damage numbers) are left '?'.")


def main(outdir, frames):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (WORK / "lab").mkdir(parents=True, exist_ok=True)
    plan = hud_plans()["emulator"]
    cfg = load_vision_config()
    g = class_grid()
    at = smoothed_A()
    pos = np.load(WORK / "reg.npz")["pos"]
    wanted = sorted({x for f in frames for x in (f - NEIGHBOUR, f, f + NEIGHBOUR)})
    images = read_label_frames(CLIP, wanted, plan.viewport)
    cols, rows = plan.size_tiles
    report = {}
    for f in frames:
        A = at(f)
        nb = {n: images[n] for n in (f - NEIGHBOUR, f + NEIGHBOUR)}
        rect, chars, gas, occ, stats = label_frame(f, images[f], plan, cfg, g, A, nb, pos)
        grid = LabelGrid(clip=CLIP, frame=int(f), origin_tile=tuple(int(round(v)) for v in plan.origin_tile),
                         size_tiles=(cols, rows), pixels_per_tile=plan.pixels_per_tile,
                         chars=chars, notes=NOTE.format(ax=A[0], ay=A[1]), hud="emulator")
        grid.save(outdir / f"{CLIP}_f{f}.json")
        np.savez(WORK / "lab" / f"aux_{f}.npz", gas=gas, occ=occ, A=A)
        cv2.imwrite(str(WORK / "lab" / f"rect_{f}.png"), rect)
        report[int(f)] = stats | {"A": [round(float(A[0]), 3), round(float(A[1]), 3)],
                                  "count": grid.count()}
        print(f, json.dumps(report[int(f)]))
    return report


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
