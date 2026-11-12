"""Ground that breaks during the match: each map tile a label calls wall or bush, scored in every
label frame, to find the cells `map_project_labels.py --broken` has to name.

The projection labels from the map, the ground as the match starts (its docstring, **Broken
cells**). A cone or cube that a super or a shot breaks leaves floor, and so does a bush. This reads
each projected label of `$CLIP` (training and held-out) with its frame. It takes each map tile
under a `#` or `b` cell centre that the map gives the same class, and scores the middle of that
tile (INNER of it each way) in the rectified frame. It scores the tile rather than the cell
because a cell can sit half a tile off its tile and take in a neighbour. A cell can even be
labelled `#` with its centre on floor, when walls hold 60 % of it. Only pixels the plan's `valid`
backs count (inside the viewport, off the HUD): a tile not wholly backed is not scored, since a
bush half off the frame's edge reads as floor.

- **Floor share:** the share of pixels inside the frame's floor colour spread, which is the 80th
  percentile of the floor cells' distance from their median. Standing walls and bushes read near
  0, and a broken cone or bush 0.5-1. A broken cube's rubble is darker than the floor and can
  read under 0.1, as (26,32) does at frame 3420 (0.09), so cubes are caught by the bright share.
- **A wall's bright share:** the share of pixels whose brightest channel is over BRIGHT_LEVEL.
  Rockwall Brawl's cubes are white-blue and its floor is dark blue, so a broken cube reads near 0
  ((26,32) at 3420: 0.08).
  A broken cone can keep some brightness from its debris, so walls are scored on the floor share
  as well.
- **A bush's colour:** where the tile's mean sits between the frame's floor and bush medians, from
  0 = floor to 1 = bush.

The lines are this palette's (the 2026-11-11 survey). A map with dark walls needs its own.

The numbers don't settle it alone. A wall's top face leans about 0.9 tile north, and further
toward the view's edges, so a broken cone beside a cube can keep some of the cube's brightness:
(41,25) at frame 3510 reads 0.21 against the line's 0.25. The see-through bush around the hero
reads as floor for a frame. (These numbers are the first projection's, before --broken left the
cells '?'.) For each tile with a
view past the line, the probe prints every view in frame order. It ends `stays from F` when the tile
stays past the line to its last view, and F is then the frame to give `--broken`. Otherwise it ends
`comes back`. With --sheet, every such tile gets a row of crops, 5 x 5 tiles around it with the tile
boxed: its first view, its last view inside the line, its first and last views past it, and its
last view. Settle each by eye there.

    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe \\
        scripts/probes/map_label_projection/rockwall_brawl/broken_cells.py [--sheet SHEET.png] [LABEL_DIR ...]

LABEL_DIR defaults to the training folder and labels_rockwall_brawl_test.

Run it on the first projection, name what broke with --broken, project again, and run it once
more. On the final labels, a tile that stays past the line is a cell still to name, or a lean to
rule out on the sheet.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from brawl_deployment.perception.known_map import MAPS_DIR, read_chars          # noqa: E402
from brawl_vision.terrain.labeling import LabelGrid, hud_plans, read_label_frames  # noqa: E402
from clip_paths import CLIP                                                         # noqa: E402

LABEL_DIRS = [REPO / "tests/fixtures/vision/labels", REPO / "tests/fixtures/vision/labels_rockwall_brawl_test"]
BRIGHT_LEVEL = 150
WALL_LINE, WALL_FLOOR_LINE = 0.25, 0.5     # a wall is past the line under the first or over the second
BUSH_LINE, COLOUR_LINE = 0.5, 0.35         # a bush is past it over the first or under the second
INNER = 0.5          # the middle half of each tile, each way
Q = 2                # scored at 1/Q scale, 24 px per tile at 48
MIN_BUSHES = 5       # a frame with fewer bush tiles gives no bush colour median


def registration(notes):
    A = re.search(r"A=\(([-\d.]+), ([-\d.]+)\)", notes).groups()
    S = re.search(r"scale \(([\d.]+), ([\d.]+)\)", notes).groups()
    return tuple(map(float, A)), tuple(map(float, S))


def tile_of(A, S, j, i):
    return int(np.floor(A[0] + (i + 0.5) / S[0])), int(np.floor(A[1] + (j + 0.5) / S[1]))


def tile_pixels(img, ok, A, S, tile, ppt):
    """The middle INNER of map `tile` in `img` (`ppt` px per rectified cell) as (n, 3), or None
    unless all of it is world-backed (`ok`, the plan's `valid` at this scale)."""
    (x, y), m = tile, (1 - INNER) / 2
    x0, x1 = ((np.array([x + m, x + 1 - m]) - A[0]) * S[0] * ppt).round().astype(int)
    y0, y1 = ((np.array([y + m, y + 1 - m]) - A[1]) * S[1] * ppt).round().astype(int)
    h, w = img.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 - x0 < 3 or y1 - y0 < 3 or not ok[y0:y1, x0:x1].all():
        return None
    return img[y0:y1, x0:x1].reshape(-1, 3)


def score(grids, images, plan, chars):
    """{tile: [(frame, class, floor share, bright share or colour)]} for the tiles under `#` and `b`
    cell centres that the map label `chars` gives the same class. A bush's colour is NaN in a frame
    with under MIN_BUSHES bush tiles."""
    q = plan.pixels_per_tile // Q
    h, w = plan.valid.shape
    ok = cv2.resize(plan.valid.astype(np.float32), (w // Q, h // Q), interpolation=cv2.INTER_AREA) > 0.999
    views = defaultdict(list)
    for g in grids:
        A, S = registration(g.notes)
        rect = plan.rectify(images[g.frame]).astype(np.float32)
        small = cv2.resize(rect, (rect.shape[1] // Q, rect.shape[0] // Q), interpolation=cv2.INTER_AREA)
        floor = [small[j * q:(j + 1) * q, i * q:(i + 1) * q][ok[j * q:(j + 1) * q, i * q:(i + 1) * q]]
                 for j, i in zip(*np.nonzero(g.chars == "."))]
        floor = [c for c in floor if len(c)]
        fl = np.concatenate(floor)
        med = np.median(fl, axis=0)
        spread = np.percentile(np.abs(fl - med).sum(1), 80)
        fmed = np.median([c.mean(0) for c in floor], axis=0)
        tiles = {}
        for c in "#b":
            for j, i in zip(*np.nonzero(g.chars == c)):
                x, y = tile_of(A, S, j, i)
                if 0 <= x < chars.shape[1] and 0 <= y < chars.shape[0] and chars[y, x] == c:
                    tiles[(x, y)] = c
        crops = {t: tile_pixels(small, ok, A, S, t, q) for t in tiles}
        bush = [crops[t].mean(0) for t, c in tiles.items() if c == "b" and crops[t] is not None]
        bmed = np.median(bush, axis=0) if len(bush) >= MIN_BUSHES else None
        for t, c in tiles.items():
            px = crops[t]
            if px is None:
                continue
            share = float((np.abs(px - med).sum(1) <= spread).mean())
            if c == "#":
                second = float((px.max(1) > BRIGHT_LEVEL).mean())
            elif bmed is not None:
                db, df = np.abs(px.mean(0) - bmed).sum(), np.abs(px.mean(0) - fmed).sum()
                second = float(df / (db + df + 1e-6))
            else:
                second = np.nan
            views[t].append((g.frame, c, share, second))
    return views


def past_line(cls, share, second):
    if cls == "#":
        return second < WALL_LINE or share > WALL_FLOOR_LINE
    return share > BUSH_LINE or second < COLOUR_LINE      # a NaN colour is never under the line


def crop(rect, A, S, tile, ppt):
    """5 x 5 tiles of `rect` around map `tile`, the tile boxed."""
    (x, y), h = tile, int(2.5 * ppt)
    cx, cy = (x + 0.5 - A[0]) * S[0] * ppt, (y + 0.5 - A[1]) * S[1] * ppt
    pad = cv2.copyMakeBorder(rect, h, h, h, h, cv2.BORDER_CONSTANT)
    c = pad[int(cy):int(cy) + 2 * h, int(cx):int(cx) + 2 * h].copy()
    x0, y0 = int(h + (x - A[0]) * S[0] * ppt - cx), int(h + (y - A[1]) * S[1] * ppt - cy)
    cv2.rectangle(c, (x0, y0), (x0 + int(S[0] * ppt), y0 + int(S[1] * ppt)), (0, 255, 255), 2)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", type=Path, default=LABEL_DIRS)
    ap.add_argument("--sheet", help="write the flagged tiles' crops here")
    a = ap.parse_args()
    grids = sorted((LabelGrid.load(p) for d in a.dirs for p in d.glob(f"{CLIP}_f*.json")),
                   key=lambda g: g.frame)
    plan = hud_plans()["emulator"]
    images = read_label_frames(CLIP, [g.frame for g in grids], plan.viewport)
    views = score(grids, images, plan, read_chars(MAPS_DIR / "rockwall_brawl.csv"))
    n_wall = sum(1 for vs in views.values() if vs[0][1] == "#")
    print(f"{CLIP}: {len(grids)} label frames; {n_wall} wall and {len(views) - n_wall} bush tiles "
          f"labelled. Each view: frame:floor share/{{bright share (wall) or colour (bush)}}, * = past the line")
    flagged, stay = [], []
    for tile, vs in sorted(views.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        vs.sort()
        past = [past_line(*v[1:]) for v in vs]
        if not any(past):
            continue
        k = next((k for k in range(len(vs)) if all(past[k:])), None)
        print(f"({tile[0]:2d},{tile[1]:2d}) {vs[0][1]} {len(vs):3d} views, "
              + (f"stays from {vs[k][0]}" if k is not None else "comes back") + ": "
              + " ".join(f"{f}:{s:.2f}/{c:.2f}" + ("*" if p else "") for (f, _, s, c), p in zip(vs, past)))
        flagged.append((tile, vs, past))
        if k is not None:
            stay.append(f"{tile[0]},{tile[1]}@{vs[k][0]}")
    print(f"{len(flagged)} tiles past the line in some view; {len(stay)} stay to their last view"
          + (": --broken " + " ".join(stay) if stay else ""))

    if a.sheet and flagged:
        by_frame = {g.frame: g for g in grids}
        ppt = plan.pixels_per_tile
        rows = []
        for tile, vs, past in flagged:
            first = past.index(True)
            inside = [k for k in range(first) if not past[k]]
            pick = sorted({0, *inside[-1:], first, max(k for k, p in enumerate(past) if p), len(vs) - 1})
            tiles = []
            for k in pick:
                f = vs[k][0]
                A, S = registration(by_frame[f].notes)
                c = crop(plan.rectify(images[f]), A, S, tile, ppt)
                text = f"{tile[0]},{tile[1]} f{f}" + (" *" if past[k] else "")
                for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                    cv2.putText(c, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, thick)
                tiles.append(c)
            tiles += [np.zeros_like(tiles[0])] * (5 - len(tiles))
            rows.append(np.hstack(tiles))
        cv2.imwrite(a.sheet, np.vstack(rows))
        print("sheet ->", a.sheet)


if __name__ == "__main__":
    main()
