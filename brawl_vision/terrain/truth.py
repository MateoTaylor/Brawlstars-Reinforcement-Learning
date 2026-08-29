"""Authoring and scoring a hand-verified `<clip>.grid.csv`. See Terrain_Perception_Build_Plan.md
Phases I and L.

**The blocking artifact for two acceptance criteria, and the only one a person has to make.** Phase
I wants to know how much of its accumulated map is *right*, and Phase L's third bullet wants the
same for a whole clip. Both need an answer key: the real terrain, typed out, one character per
tile, in the map-CSV legend the simulator already uses.

Writing one from memory is miserable and error-prone, which is most of why it had not been done.
So this generates the two halves of the job:

* **A reference image** -- Phase L's mosaic, cropped to a region the camera actually covered, drawn
  at a readable scale with the tile grid and coordinates on it. Every square in that picture is one
  character in the file, and the labels down the side are the CSV's own line numbers.
* **A blank CSV** of exactly matching size, every cell `?`.

**The template is blank, never pre-filled with the classifier's guess.** A truth file seeded from
the thing it is grading is not truth; it measures how well someone spots errors in a prediction
they were shown, which is a different and much easier question. `?` cells are skipped on both sides
of the comparison, so leaving one blank is free -- anything ambiguous, under a loot box, under a
brawler, half off the edge, stays `?` and costs nothing.

**The region is chosen by coverage, not by cropping the middle.** The largest rectangle where every
tile was actually painted by the mosaic is the largest rectangle a person can actually read, and a
template containing cells the camera barely saw is a template with unanswerable questions in it.
"""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from brawl_sim.constants import CHAR_TO_TILE

from .evaluate import Mosaic, Window
from .occupancy import compare_to_truth

UNLABELLED = "?"
VALID_CHARS = set(CHAR_TO_TILE) | {UNLABELLED}

_REF_SCALE = 44          # reference-image pixels per tile; big enough to read a tile's material
_MARGIN_PX = 46          # room for the coordinate labels
_MAJOR = 5               # heavier gridline and a number every this many tiles
_MIN_COVERAGE = 0.85     # a tile counts as covered when this fraction of its pixels were painted


@dataclass
class Template:
    """A region of the world to hand-label, and where it sits."""
    clip: str
    origin_tile: tuple[int, int]        # world tile (x, y) of the template's cell (0, 0)
    size_tiles: tuple[int, int]         # (cols, rows)
    window: Window                      # the same region as occupancy-grid indices
    coverage: float                     # mean mosaic coverage over the region

    @property
    def cols(self) -> int:
        return self.size_tiles[0]

    @property
    def rows(self) -> int:
        return self.size_tiles[1]


class Plate:
    """A MEDIAN composite of the rectified frames, in the world frame. The thing you trace from.

    **The averaging mosaic is the wrong reference for reading terrain, and that is not a tuning
    complaint.** Phase L's mosaic exists to make odometry drift visible, and it does that by
    averaging every frame -- which is exactly what blends eight hundred frames of brawlers,
    projectiles, damage numbers and gas into the ground beneath them. The first template generated
    from it was a uniform smear: 100% "coverage" and not one tile you could name.

    A per-pixel median fixes it for the same reason it works as a background plate anywhere else.
    Terrain is what is there in most frames; everything that moves is in a minority of them at any
    given world point, and a median discards minorities instead of mixing them in.

    **Which `K` samples matters as much as taking a median of them, and the obvious choice is
    wrong.** Keeping the last `K` writes round-robin sounds like a spread and is not: for a pixel
    written 400 times, the surviving slots are writes 392-400, which at 60 fps are a seventh of a
    second apart. A brawler standing still for that long occupies every slot and *becomes* the
    median. The samples have to be spread over the pixel's whole history, and the standard way to
    do that in one pass without knowing the history's length up front is **reservoir sampling**:
    the n-th sample replaces a uniformly chosen slot with probability `K/n`, which leaves every
    slot holding a uniform draw from everything that pixel ever saw.

    A pixel's first sample fills every slot, so the median is always over `K` real values and never
    over zeros left in unwritten slots -- which would drag every partially-covered pixel to black.
    """

    def __init__(self, mosaic: Mosaic, slots: int = 12, seed: int = 0):
        self.mosaic = mosaic
        self.slots = slots
        self.stack = np.zeros((slots, mosaic.h, mosaic.w, 3), np.uint8)
        self.count = np.zeros((mosaic.h, mosaic.w), np.int32)
        self._rng = np.random.default_rng(seed)

    def add(self, rect: np.ndarray, position: tuple[float, float]) -> None:
        placed = self.mosaic.place(rect, position)
        if placed is None:
            return
        patch, valid, dst, src = placed
        m = valid[src][..., 0]
        if not m.any():
            return
        sub = patch[src]
        ys, xs = np.nonzero(m)
        dy, dx = dst[0].start, dst[1].start
        gy, gx = ys + dy, xs + dx
        seen = self.count[gy, gx]

        first = seen == 0
        if first.any():
            fy, fx = gy[first], gx[first]
            self.stack[:, fy, fx] = sub[ys[first], xs[first]]

        rest = ~first
        if rest.any():
            n = seen[rest]
            # Reservoir: draw j uniformly in [0, n]; keep the sample only when it lands in a slot.
            j = self._rng.integers(0, n + 1)
            take = j < self.slots
            if take.any():
                sel = np.nonzero(rest)[0][take]
                self.stack[j[take], gy[sel], gx[sel]] = sub[ys[sel], xs[sel]]
        self.count[gy, gx] = seen + 1

    def image(self) -> np.ndarray:
        """(h, w, 3) uint8 RGB. Unseen pixels are black."""
        med = np.median(self.stack, axis=0).astype(np.uint8)
        med[self.count == 0] = 0
        return med

    def coverage(self) -> np.ndarray:
        """(rows, cols) fraction of each tile's pixels that were ever written."""
        s = self.mosaic.scale
        rows, cols = self.mosaic.window.rows, self.mosaic.window.cols
        seen = (self.count > 0).astype(np.float32)
        return seen[:rows * s, :cols * s].reshape(rows, s, cols, s).mean(axis=(1, 3))


def tile_coverage(mosaic: Mosaic) -> np.ndarray:
    """(rows, cols) fraction of each tile's pixels that the mosaic actually painted."""
    s = mosaic.scale
    count = (mosaic.count[..., 0] > 0).astype(np.float32)
    rows, cols = mosaic.window.rows, mosaic.window.cols
    return count[:rows * s, :cols * s].reshape(rows, s, cols, s).mean(axis=(1, 3))


def largest_covered_rect(covered: np.ndarray, max_cols: int, max_rows: int) -> Window:
    """Largest all-covered axis-aligned rectangle, then trimmed to the cap about its centre.

    Standard largest-rectangle-in-a-histogram sweep. Choosing by coverage rather than cropping the
    middle is the difference between a template someone can fill in and one with unanswerable cells
    in it -- the camera's footprint is a trapezoid, so the edges of any observed region are thin
    and streaky however long the clip ran.
    """
    h = np.zeros(covered.shape[1], np.int64)
    best = (0, 0, 0, 0, 0)                     # area, rows, cols, r_bottom, c_left
    for r in range(covered.shape[0]):
        h = np.where(covered[r], h + 1, 0)
        stack: list[tuple[int, int]] = []
        for c in range(len(h) + 1):
            cur = int(h[c]) if c < len(h) else 0
            start = c
            while stack and stack[-1][1] >= cur:
                start, height = stack.pop()
                area = height * (c - start)
                if area > best[0]:
                    best = (area, height, c - start, r, start)
            stack.append((start, cur))
    _, rows, cols, r_bot, c_left = best
    if rows == 0 or cols == 0:
        raise ValueError("no fully covered region -- the clip observed nothing cleanly")
    r0, c0 = r_bot + 1 - rows, c_left
    if cols > max_cols:
        c0 += (cols - max_cols) // 2
        cols = max_cols
    if rows > max_rows:
        r0 += (rows - max_rows) // 2
        rows = max_rows
    return Window(row0=r0, row1=r0 + rows, col0=c0, col1=c0 + cols)


def build_template(clip: str, plate: "Plate", grid_origin: tuple[int, int],
                   max_cols: int = 24, max_rows: int = 18,
                   min_coverage: float = _MIN_COVERAGE) -> Template:
    mosaic = plate.mosaic
    cov = plate.coverage()
    local = largest_covered_rect(cov >= min_coverage, max_cols, max_rows)
    # `local` indexes the mosaic canvas; shift it into occupancy-grid indices, then into world tiles.
    win = Window(row0=mosaic.window.row0 + local.row0, row1=mosaic.window.row0 + local.row1,
                 col0=mosaic.window.col0 + local.col0, col1=mosaic.window.col0 + local.col1)
    gx, gy = grid_origin
    return Template(
        clip=clip,
        origin_tile=(win.col0 + gx, win.row0 + gy),
        size_tiles=(win.cols, win.rows),
        window=win,
        coverage=float(cov[local.row0:local.row1, local.col0:local.col1].mean()),
    ), local


def reference_image(plate: "Plate", local: Window) -> np.ndarray:
    """The mosaic region blown up with the tile grid and coordinates drawn on it.

    The numbers down the left are the CSV's line numbers and the numbers along the top are its
    field positions, both 0-based, so "the cell at row 7 column 12" means one thing in the picture
    and the file. Getting that correspondence wrong is the only way to fill in a template
    perfectly and still score zero.
    """
    scale = plate.mosaic.scale
    crop = plate.image()[local.row0 * scale:local.row1 * scale,
                         local.col0 * scale:local.col1 * scale]
    # BGR out, because `cv2.imwrite` writes what it is given as BGR. The plate is RGB -- the same
    # convention `video.VideoSink` needs -- and the first version of this file skipped the
    # conversion, which turned the map's red terrain blue and made every tile unreadable.
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    rows, cols = local.rows, local.cols
    canvas = np.zeros((rows * scale + _MARGIN_PX, cols * scale + _MARGIN_PX, 3), np.uint8)
    canvas[:] = 24
    canvas[_MARGIN_PX:, _MARGIN_PX:] = crop

    for c in range(cols + 1):
        x = _MARGIN_PX + c * scale
        major = c % _MAJOR == 0
        cv2.line(canvas, (x, _MARGIN_PX), (x, canvas.shape[0] - 1),
                 (255, 255, 0) if major else (150, 150, 150), 2 if major else 1)
    for r in range(rows + 1):
        y = _MARGIN_PX + r * scale
        major = r % _MAJOR == 0
        cv2.line(canvas, (_MARGIN_PX, y), (canvas.shape[1] - 1, y),
                 (255, 255, 0) if major else (150, 150, 150), 2 if major else 1)
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    for c in range(cols):
        cv2.putText(canvas, str(c), (_MARGIN_PX + c * scale + 6, _MARGIN_PX - 12),
                    font, fs, (255, 255, 255) if c % _MAJOR == 0 else (150, 150, 150), 1,
                    cv2.LINE_AA)
    for r in range(rows):
        cv2.putText(canvas, str(r), (6, _MARGIN_PX + r * scale + scale // 2 + 5),
                    font, fs, (255, 255, 255) if r % _MAJOR == 0 else (150, 150, 150), 1,
                    cv2.LINE_AA)
    return canvas


def write_template(template: Template, out_csv: Path, overwrite: bool = False) -> None:
    """A CSV of `?`, one row per line. Refuses to clobber work already done."""
    out_csv = Path(out_csv)
    if out_csv.exists() and not overwrite:
        raise FileExistsError(
            f"{out_csv} already exists. That file is hand-written work; pass overwrite=True only "
            f"if you mean to throw it away."
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for _ in range(template.rows):
            w.writerow([UNLABELLED] * template.cols)


def write_sidecar(template: Template, out_json: Path) -> None:
    """Where the template sits, so scoring does not have to rediscover it by search.

    `compare_to_truth` can find the alignment on its own, but only within +/-6 tiles and only by
    picking whichever offset scores best -- which quietly turns a position measurement into a
    fitted parameter. Recording the origin means the search confirms an answer instead of choosing
    one.
    """
    Path(out_json).write_text(json.dumps({
        "clip": template.clip,
        "origin_tile": list(template.origin_tile),
        "size_tiles": list(template.size_tiles),
        "grid_window": [template.window.row0, template.window.row1,
                        template.window.col0, template.window.col1],
        "mosaic_coverage": round(template.coverage, 4),
        "legend": {c: t.name for c, t in CHAR_TO_TILE.items()} | {UNLABELLED: "UNLABELLED"},
    }, indent=2) + "\n", encoding="utf-8")


def load_grid_csv(path) -> np.ndarray:
    """(rows, cols) of single characters. Raises on anything not in the legend.

    **Two spellings of a row are accepted, because one of them is far easier to type.** The written
    form is comma-separated (`.,.,#,#`), which keeps the `.csv` extension honest and opens in a
    spreadsheet. But a person filling 400 cells against a picture is counting columns, and in the
    comma form column N sits at character 2N -- so the file does not line up with the reference
    image and losing your place costs a row. The compact form (`..##`) is one character per tile,
    which lines up in any monospace editor exactly as the picture does.

    Both are the same legend and the same file; strip the commas or don't. Detection is by
    content, not by a flag: a row containing a comma is comma-separated, otherwise every character
    is a cell.

    Strict on what a cell may contain, though: a stray space or a lowercase `W` would otherwise be
    compared as a mismatch, and the resulting accuracy would be a typo count wearing the
    classifier's name.
    """
    rows = []
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        if "," in raw:
            cells = [c.strip() for c in next(csv.reader([raw]))]
        else:
            cells = [c for c in raw.strip() if not c.isspace()]
        for col_no, cell in enumerate(cells):
            if cell not in VALID_CHARS:
                raise ValueError(
                    f"{path}:{line_no} column {col_no}: {cell!r} is not one of "
                    f"{sorted(VALID_CHARS)}"
                )
        rows.append(cells)
    if not rows:
        raise ValueError(f"{path} has no rows")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{path} has ragged rows: widths {sorted(widths)}")
    return np.array(rows, dtype="<U1")


def progress(truth: np.ndarray) -> dict:
    """How much of a template is filled in, and with what. For telling someone where they are."""
    total = truth.size
    done = int((truth != UNLABELLED).sum())
    counts = {c: int((truth == c).sum()) for c in sorted(CHAR_TO_TILE) if (truth == c).any()}
    return {"cells": total, "filled": done, "fraction": done / total if total else 0.0,
            "counts": counts}


def score(predicted_chars: np.ndarray, truth: np.ndarray, window: Window,
          max_shift: int = 6) -> dict:
    """`compare_to_truth` over the template's own region of an accumulated map.

    `predicted_chars` is the whole occupancy grid as characters (`OccupancyMap.to_chars()`); the
    window crops it to what the truth file covers. Cropping here rather than asking the caller to
    do it keeps the two halves of the alignment -- which region, and which offset -- in one place.
    """
    pred = predicted_chars[window.row0:window.row1, window.col0:window.col1]
    if pred.shape != truth.shape:
        raise ValueError(
            f"truth is {truth.shape} but its recorded window is {pred.shape}. The CSV has been "
            f"resized since the template was generated, so the two no longer describe the same "
            f"region of the world."
        )
    out = compare_to_truth(pred, truth, max_shift=max_shift)
    out["wall_vs_fence"] = _wall_vs_fence(pred, truth)
    return out


def _wall_vs_fence(pred: np.ndarray, truth: np.ndarray) -> float | None:
    """Accuracy restricted to cells the truth calls WALL or FENCE, reported on its own per Phase H.

    Both block movement and only WALL blocks shots, so a fence read as a wall tells the policy it
    has cover from a shot that is about to hit it. That error is invisible in an overall accuracy
    dominated by floor.
    """
    sel = np.isin(truth, ["#", "f"])
    if not sel.any():
        return None
    return float((pred[sel] == truth[sel]).mean())
