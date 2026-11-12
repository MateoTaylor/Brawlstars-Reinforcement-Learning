"""Terrain-classifier labels projected from a map label, instead of drawn by hand.

MAP_ONBOARDING.md, step M3b. Once a map's 60 x 60 label (`brawl_deployment/data/maps/<name>.csv`)
is right, a recording on that map can be labelled from it, frame by frame: register the frame
against the map image, give each cell the class of the ground under it, and refuse whatever the
labels README (`tests/fixtures/vision/labels/README.md`) refuses. `scripts/map_project_labels.py`
runs that over a clip. This module holds every part that needs no video, so a test can pin them
without footage. First used on Starfruit Supernova, 2026-11-04
(`scripts/probes/map_label_projection/README.md` has that run's numbers).

**Registration.** Call A the map tile under rectified pixel (0, 0), and S the rectified pixels per
true tile divided by `plan.pixels_per_tile`, per axis.

- Odometry gives a smooth track, but in the camera model's scale.
- Masked NCC of the rectified frame against the map image gives an absolute position at a chosen
  scale: `MapMatcher.coarse` at scale 1, `MapMatcher.at_scale` at S.
- `fit_scale` regresses the scale-1 NCC position on odometry over the best odometry segment, so
  S = 1 / slope per axis, rounded to `SCALE_DECIMALS`. BlueStacks captures come out near
  (1.022, 1.004): the camera model was calibrated on the phone, and the emulator rectifies 2.2 %
  too wide in x.
- The fit is only as good as the odometry. A static overlay that odometry locks onto bends it.
  On Truth, 2026-11-05, a "SHOWDOWN!" banner froze odometry for 3 s while the camera moved
  6 tiles, and the fit read (0.939, 0.935). `block_stretch` measures the scale from the image
  instead, out of stage 2's block residuals. There it showed a 6-7 % stretch, which was gone at
  (1.022, ~1.01); `map_project_labels.py --scale` sets S when the two disagree.
- `Track` gives each frame A = pos / S + offset. The offset, A_ncc - pos / S, is a running median
  over `2 * SMOOTH_HALF + 1` samples of the frame's own odometry segment, so one bad NCC peak cannot
  move a label, and an odometry reset cannot blend two segments' offsets.

**The label rule** (`robust_chars`). A cell takes the majority class of the map area under it.

- The area is the ground footprint. A wall's top face shows about 0.9 tile north, in the floor
  cell above it, and that cell stays floor.
- The class must hold at least `PURITY` of the cell under all 9 shifts of +/-`DELTA` tile, the
  registration error.
- Only `. # b ~ f` label. A spawn is floor. Off-map, a `?` in the map label, and the cells a run
  asks to leave (Starfruit's meeple figures, wall or fence unsettled) never label.
- The map label is the ground at the start of the match. A wall or bush that a super or a shot
  destroys shows floor from then on, so a cell a run names broken (`parse_broken`) labels from the
  map until the frame it was seen broken, and never after (`grid_at`). First needed on Rockwall
  Brawl, 2026-11-11, whose clip lost 8 destructible cones, 4 rock cubes and 4 bushes.

**The refusals** (`label_cells`), the labels README's, as on the first run:

- HUD and unobserved cells (`observed_cells`);
- gas: `detect_zone` at `GAS_FRACTION` coverage, grown by one cell;
- detector boxes (brawlers, crates, cubes, projectiles), mapped into the rectified frame with a
  small margin (`RECT_PAD`); the boxes already take in the name tag, HP bar and ammo bar;
- the brawlers-left text and the kill card, which on the 1440p emulator captures run past the HUD
  mask (`TEXT_BOX`, `KILL_BOX`);
- cells that change against the frames `NEIGHBOUR` either side, shifted by the odometry delta:
  splashes, damage numbers, shells in flight (`transient_cells`).
"""
import warnings
from dataclasses import dataclass

import cv2
import numpy as np

from ..camera import RectifyPlan
from .labeling import UNLABELLED, observed_cells

# ---------------------------------------------------------------------------
# the map label as classes
# ---------------------------------------------------------------------------

# Grid classes by index: the five terrain classes, each labelled as itself, then two that never
# label. LEAVE is a cell the map label holds as `?` or a run asks to leave; OFF_MAP is outside it.
TERRAIN = (".", "#", "b", "~", "f")
LEAVE, OFF_MAP = len(TERRAIN), len(TERRAIN) + 1
N_CLASSES = OFF_MAP + 1
SPAWN = "S"


def class_grid(chars, leave=()) -> np.ndarray:
    """(rows, cols) int8 class indices for a map label's characters (`known_map.read_chars`).

    A spawn is floor. `?` (a cell the label never settled) and every (col, row) in `leave` are
    LEAVE, so no classifier label is ever drawn from them."""
    chars = np.asarray(chars)
    unknown = sorted(set(np.unique(chars).tolist()) - set(TERRAIN) - {SPAWN, UNLABELLED})
    if unknown:
        raise ValueError(f"the map label holds {unknown}; a deploy map label is only "
                         f"{''.join(TERRAIN)}{SPAWN}{UNLABELLED}")
    g = np.full(chars.shape, LEAVE, np.int8)
    for k, c in enumerate(TERRAIN):
        g[chars == c] = k
    g[chars == SPAWN] = TERRAIN.index(".")
    rows, cols = g.shape
    for c, r in leave:
        if not (0 <= c < cols and 0 <= r < rows):
            raise ValueError(f"cell to leave ({c},{r}) is off the {cols} x {rows} map")
        g[r, c] = LEAVE
    return g


def parse_leave(cells=(), row_runs=()) -> list[tuple[int, int]]:
    """Cells to leave `?`, as sorted (col, row): `cells` are "C,R" strings, `row_runs` are
    "R:C0-C1,C2-C3,C4" (inclusive column runs on map row R), the easy way to list a row of
    sprites."""
    out = set()
    for tok in cells:
        parts = tok.split(",")
        if len(parts) != 2 or not all(p.strip().lstrip("-").isdigit() for p in parts):
            raise ValueError(f"cell {tok!r}: expected C,R, for example 3,26")
        out.add((int(parts[0]), int(parts[1])))
    for tok in row_runs:
        row, sep, runs = tok.partition(":")
        if not sep or not row.strip().isdigit() or not runs:
            raise ValueError(f"row run {tok!r}: expected R:C0-C1,C2-C3, for example 26:0-3,8-14")
        for run in runs.split(","):
            a, _, b = run.partition("-")
            if not a.strip().isdigit() or not (b.strip().isdigit() or b == ""):
                raise ValueError(f"row run {tok!r}: {run!r} is not C or C0-C1")
            a, b = int(a), int(b) if b else int(a)
            if b < a:
                raise ValueError(f"row run {tok!r}: {run!r} runs backwards")
            out.update((c, int(row)) for c in range(a, b + 1))
    return sorted(out, key=lambda cr: (cr[1], cr[0]))


def parse_broken(specs=()) -> list[tuple[tuple[int, int], int]]:
    """Map cells that break during the match, as ((col, row), frame), sorted by row then column.

    Each spec is "C,R@F": cell (C, R) is broken by frame F. Only labelled frames count, so F can be
    any frame after the last labelled one that shows the cell standing, up to the first that shows
    it broken. `grid_at` applies them."""
    out = {}
    for tok in specs:
        cell, sep, frame = tok.partition("@")
        parts = cell.split(",")
        if not (sep and frame.strip().isdigit() and len(parts) == 2
                and all(p.strip().isdigit() for p in parts)):
            raise ValueError(f"broken cell {tok!r}: expected C,R@F, for example 41,33@2100")
        key = (int(parts[0]), int(parts[1]))
        if key in out:
            raise ValueError(f"broken cell {key[0]},{key[1]} is given twice")
        out[key] = int(frame)
    return sorted(out.items(), key=lambda kv: (kv[0][1], kv[0][0]))


def grid_at(g, broken, frame: int) -> np.ndarray:
    """`class_grid`'s `g` as the ground stands at `frame`: a copy, with each `broken` cell
    (`parse_broken`) whose frame is `frame` or earlier set to LEAVE. A broken cell off the map
    raises whatever the frame, so one call before the slow part checks them all."""
    out = np.array(g, copy=True)
    rows, cols = out.shape
    for (c, r), f in broken:
        if not (0 <= c < cols and 0 <= r < rows):
            raise ValueError(f"broken cell ({c},{r}) is off the {cols} x {rows} map")
        if f <= frame:
            out[r, c] = LEAVE
    return out


# ---------------------------------------------------------------------------
# registration: the map image as an NCC canvas
# ---------------------------------------------------------------------------

PAD = 14             # tiles of black around the map on every canvas, so a view off the edge fits
COARSE = 4           # the coarse search runs at plan.pixels_per_tile // COARSE px per tile
ERODE_PX = 9         # the HUD mask is eroded this much before matching, off the buttons' edges
NCC_GOOD = 0.4       # a scale-1 NCC score above this is a trusted absolute position
SCALE_DECIMALS = 3   # S is used rounded: 0.001 is far below the DELTA registration error
SMOOTH_HALF = 3      # samples either side in the offset's running median


def map_canvas(image, pitch_px, origin_px, map_tiles, ppt: int) -> np.ndarray:
    """The map image resampled to `ppt` px per tile and padded by PAD tiles, float32: canvas
    pixel ((col + PAD) * ppt, (row + PAD) * ppt) is tile (col, row)'s top-left corner.

    `pitch_px` and `origin_px` are the lattice `map_label.py` saved in `<name>.json`."""
    (px, py), (mx, my) = pitch_px, origin_px
    A = np.array([[ppt / px, 0, -ppt * mx / px], [0, ppt / py, -ppt * my / py]], np.float64)
    A[:, 2] += A[:, :2] @ np.array([0.5, 0.5]) - 0.5 + PAD * ppt
    cols, rows = map_tiles
    size = ((cols + 2 * PAD) * ppt, (rows + 2 * PAD) * ppt)
    flags = cv2.INTER_AREA if ppt < 16 else cv2.INTER_CUBIC
    return cv2.warpAffine(image, A, size, flags=flags, borderValue=(0, 0, 0)).astype(np.float32)


def peak_sub(res, x, y):
    """Sub-pixel peak of `res` at integer (x, y), by a parabola through each axis' neighbours."""
    def para(a, b, c):
        d = a - 2 * b + c
        return 0.0 if d >= 0 else 0.5 * (a - c) / d
    h, w = res.shape
    dx = para(res[y, x - 1], res[y, x], res[y, x + 1]) if 0 < x < w - 1 else 0.0
    dy = para(res[y - 1, x], res[y, x], res[y + 1, x]) if 0 < y < h - 1 else 0.0
    return x + dx, y + dy


def _ncc(canvas, tmpl, mask):
    r = cv2.matchTemplate(canvas, tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
    return np.nan_to_num(r, nan=-1, posinf=-1, neginf=-1)


class MapMatcher:
    """Masked NCC of rectified frames against one map image: at scale 1 over the whole map, and at
    a fitted scale near a guess. Holds two canvases (the fine one is about 200 MB at 48 px/tile)."""

    def __init__(self, image, pitch_px, origin_px, map_tiles, plan: RectifyPlan):
        ppt = plan.pixels_per_tile
        if ppt % COARSE:
            raise ValueError(f"pixels_per_tile {ppt} is not a multiple of {COARSE}")
        self.ppt, self.cppt = ppt, ppt // COARSE
        self.origin = plan.origin_tile
        self.size_px = plan.size_px
        self.fine = map_canvas(image, pitch_px, origin_px, map_tiles, ppt)
        self.coarse_canvas = map_canvas(image, pitch_px, origin_px, map_tiles, self.cppt)
        self.valid = cv2.erode(plan.valid.astype(np.uint8), np.ones((ERODE_PX, ERODE_PX), np.uint8))
        w, h = plan.size_px
        small = cv2.resize(self.valid, (w // COARSE, h // COARSE), interpolation=cv2.INTER_AREA)
        self.coarse_mask = (small > 0).astype(np.uint8)
        self._scaled = {}

    def coarse(self, rect):
        """Scale-1 registration of a rectified frame over the whole map.

        Returns (P, score, coarse_score, runner_up): rectified (0, 0) sits at map tile
        plan.origin_tile + P; the runner-up is the best coarse score over 2 tiles from the peak,
        so score against it says how unique the match is."""
        ppt, c = self.ppt, COARSE
        t = cv2.resize(rect, (rect.shape[1] // c, rect.shape[0] // c),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        r = _ncc(self.coarse_canvas, t, self.coarse_mask)
        _, best, _, (x, y) = cv2.minMaxLoc(r)
        far, ex = r.copy(), 2 * self.cppt
        far[max(0, y - ex):y + ex + 1, max(0, x - ex):x + ex + 1] = -1
        runner_up = float(far.max())
        win = ppt * 3 // 2                       # refine at full scale within +/-1.5 tiles
        X0, Y0 = x * c - win, y * c - win
        th, tw = rect.shape[:2]
        X0c, Y0c = max(0, X0), max(0, Y0)
        sub = self.fine[Y0c:Y0c + th + 2 * win, X0c:X0c + tw + 2 * win]
        r2 = _ncc(sub, rect.astype(np.float32), self.valid)
        _, best2, _, (x2, y2) = cv2.minMaxLoc(r2)
        sx, sy = peak_sub(r2, x2, y2)
        cx, cy = X0c + sx, Y0c + sy               # canvas px of rectified (0, 0)
        ox, oy = self.origin
        P = (cx / ppt - PAD - ox, cy / ppt - PAD - oy)
        return P, float(best2), float(best), runner_up

    def _match(self, tmpl, mask, X, Y, rad):
        """Best canvas px of tmpl's (0, 0) within +/-rad px of (X, Y), and its score."""
        X0, Y0 = int(round(X)) - rad, int(round(Y)) - rad
        sub = self.fine[max(0, Y0):Y0 + tmpl.shape[0] + 2 * rad,
                        max(0, X0):X0 + tmpl.shape[1] + 2 * rad]
        r = _ncc(sub, tmpl, mask)
        _, best, _, (x, y) = cv2.minMaxLoc(r)
        sx, sy = peak_sub(r, x, y)
        return max(0, X0) + sx, max(0, Y0) + sy, float(best)

    def _scaled_mask(self, S):
        key = (float(S[0]), float(S[1]))
        if key not in self._scaled:
            tw, th = self.size_px
            TW, TH = int(round(tw / key[0])), int(round(th / key[1]))
            m = cv2.resize(self.valid.astype(np.float32), (TW, TH), interpolation=cv2.INTER_AREA)
            self._scaled[key] = (TW, TH, (m > 0.99).astype(np.uint8))
        return self._scaled[key]

    def at_scale(self, rect, S, P):
        """Re-registration at scale S, near the scale-1 estimate P, plus 3 x 3 block residuals.

        The frame is resized by 1/S to true tile size and matched within +/-1.5 tiles. Returns
        (A, score, blocks): A is the map tile of rectified (0, 0); blocks holds each ninth of the
        frame matched on its own within +/-0.75 tile, as (dx, dy, score) in tiles, NaN where the
        HUD covers most of it. A wrong S shows as opposite residuals in the outer blocks."""
        ppt, (tw, th) = self.ppt, self.size_px
        SX, SY = S
        TW, TH, mask = self._scaled_mask(S)
        ox, oy = self.origin
        # The scale-1 estimate agrees with the truth near the frame centre, not at (0, 0).
        cen = np.array([tw / 2, th / 2]) / ppt
        guess = (P[0] + ox + cen[0] - cen[0] / SX, P[1] + oy + cen[1] - cen[1] / SY)
        t = cv2.resize(rect, (TW, TH), interpolation=cv2.INTER_AREA).astype(np.float32)
        win = ppt * 3 // 2
        X, Y = (guess[0] + PAD) * ppt, (guess[1] + PAD) * ppt
        cx, cy, score = self._match(t, mask, X, Y, win)
        A = (cx / ppt - PAD, cy / ppt - PAD)
        blocks = []
        bw, bh = TW // 3, TH // 3
        for by in range(3):
            for bx in range(3):
                tb = t[by * bh:(by + 1) * bh, bx * bw:(bx + 1) * bw]
                mb = mask[by * bh:(by + 1) * bh, bx * bw:(bx + 1) * bw]
                if mb.mean() < 0.5:
                    blocks.append((np.nan, np.nan, np.nan))
                    continue
                bxx, byy, bs = self._match(np.ascontiguousarray(tb), np.ascontiguousarray(mb),
                                           cx + bx * bw, cy + by * bh, win // 2)
                blocks.append(((bxx - cx - bx * bw) / ppt, (byy - cy - by * bh) / ppt, bs))
        return A, score, blocks


# ---------------------------------------------------------------------------
# registration: the scale fit and the per-frame track
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScaleFit:
    """`fit_scale`'s answer. `S` is what the labels use (rounded); `raw` is 1 / slope."""
    S: tuple[float, float]
    raw: tuple[float, float]
    slope: tuple[float, float]
    intercept: tuple[float, float]
    resid_sd: tuple[float, float]     # tiles
    resid_max: tuple[float, float]    # tiles
    segment: int
    n: int

    def describe(self) -> str:
        return "\n".join(
            f"{ax}: slope {self.slope[k]:.4f} => S {self.raw[k]:.4f} (used {self.S[k]:.3f})  "
            f"intercept {self.intercept[k]:.3f}  resid sd {self.resid_sd[k]:.3f}  "
            f"max {self.resid_max[k]:.3f} tile"
            for k, ax in enumerate("xy")) + f"\n{self.n} samples, odometry segment {self.segment}"


def fit_scale(pos, seg, idx, P, score, min_samples: int = 10) -> ScaleFit:
    """S per axis from odometry against scale-1 NCC, over the odometry segment holding the most
    samples with an NCC score above NCC_GOOD.

    Within one segment the NCC position is P = pos / S + c (odometry reads rectified tiles, the map
    reads true ones), so S = 1 / slope of a least-squares line per axis.

    `pos` (n_frames, 2) and `seg` (n_frames,) are odometry per frame; `idx`, `P` (n, 2) and
    `score` are the sampled frames and their `MapMatcher.coarse` positions and scores."""
    idx, P, score = np.asarray(idx), np.asarray(P), np.asarray(score)
    good = score > NCC_GOOD
    if not good.any():
        raise ValueError(f"no sampled frame matches the map image above NCC {NCC_GOOD}: the wrong "
                         f"map, or a clip that isn't in a match")
    segment = int(np.bincount(seg[idx][good]).argmax())
    k = good & (seg[idx] == segment)
    if k.sum() < min_samples:
        raise ValueError(f"only {int(k.sum())} samples match the map in odometry segment "
                         f"{segment}; the fit needs {min_samples}")
    o, p = pos[idx][k], P[k]
    out = {name: [] for name in ("slope", "intercept", "sd", "max")}
    for ax in (0, 1):
        A = np.vstack([o[:, ax], np.ones(k.sum())]).T
        (m, c), *_ = np.linalg.lstsq(A, p[:, ax], rcond=None)
        r = p[:, ax] - (m * o[:, ax] + c)
        for name, v in zip(out, (m, c, r.std(), np.abs(r).max())):
            out[name].append(float(v))
    raw = tuple(1 / m for m in out["slope"])
    return ScaleFit(S=tuple(round(s, SCALE_DECIMALS) for s in raw), raw=raw,
                    slope=tuple(out["slope"]), intercept=tuple(out["intercept"]),
                    resid_sd=tuple(out["sd"]), resid_max=tuple(out["max"]),
                    segment=segment, n=int(k.sum()))


def segments_to_register(seg, idx, score) -> list[int]:
    """Odometry segments worth re-registering at the fitted scale: those with at least one median
    window (2 * SMOOTH_HALF + 1) of samples scoring above NCC_GOOD. Every sample of such a segment
    is registered, whatever its own score, since the median absorbs a bad one. The intro pan and
    loading screens before a match make segments of their own, and fall short of this."""
    good = np.asarray(score) > NCC_GOOD
    counts = np.bincount(np.asarray(seg)[np.asarray(idx)][good])
    return [int(s) for s in np.nonzero(counts >= 2 * SMOOTH_HALF + 1)[0]]


def block_stretch(blocks, S, size_tiles) -> tuple[float, float]:
    """The scale error in stage 2's 3 x 3 block residuals (`MapMatcher.at_scale`), per axis.

    The value is the median over samples of (the last outer block's shift minus the first's) over
    the distance between their centres, which is 2/3 of the frame resized to true tiles. It is
    about 0 when S is right. It is negative when S is too small: the frame, resized by 1 / S, comes
    out too big, so its left blocks shift right and its right blocks shift left to match.
    S * (1 - k) is the scale the blocks point to. The blocks match within +/-0.75 tile, so a
    scale a few percent off reads short, and from far away it only gives the direction.

    `blocks` (n, 9, 3) holds (dx, dy, score) in tiles, row-major by block, with NaN where the HUD
    covers a block. `size_tiles` is the rectified frame's (cols, rows)."""
    b = np.asarray(blocks, np.float64).reshape(-1, 3, 3, 3)       # sample, by, bx, (dx, dy, score)
    span = 2 / 3 * np.asarray(size_tiles, np.float64) / np.asarray(S, np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # all-NaN columns under the HUD
        cols = np.nanmean(b[..., 0], axis=1)                       # (n, 3): dx per block column
        rows = np.nanmean(b[..., 1], axis=2)                       # (n, 3): dy per block row
        return (float(np.nanmedian((cols[:, 2] - cols[:, 0]) / span[0])),
                float(np.nanmedian((rows[:, 2] - rows[:, 0]) / span[1])))


class Track:
    """A per frame: the frame's own odometry, plus the NCC offset of its odometry segment.

    `pos`, `seg` per frame; `idx`, `A` the samples registered at scale S (`MapMatcher.at_scale`),
    in frame order. Each segment's offset A - pos / S is a running median over 2 * SMOOTH_HALF + 1
    of its samples, interpolated between them (and held flat past the first and last)."""

    def __init__(self, pos, seg, idx, A, S):
        self.pos, self.seg = np.asarray(pos), np.asarray(seg)
        self.S = np.asarray(S, np.float64)
        idx, A = np.asarray(idx), np.asarray(A, np.float64)
        self.segments = {}
        for s in np.unique(self.seg[idx]):
            k = self.seg[idx] == s
            ii = idx[k]
            off = A[k] - self.pos[ii] / self.S
            h = SMOOTH_HALF
            sm = np.array([np.median(off[max(0, j - h):j + h + 1], axis=0) for j in range(len(ii))])
            self.segments[int(s)] = (ii, sm)

    def span(self, segment: int) -> tuple[int, int]:
        """First and last frame of `segment`."""
        frames = np.nonzero(self.seg == segment)[0]
        return int(frames[0]), int(frames[-1])

    def covers(self, frame: int) -> bool:
        return 0 <= frame < len(self.seg) and int(self.seg[frame]) in self.segments

    def __call__(self, frame: int) -> np.ndarray:
        if not self.covers(frame):
            spans = ", ".join(f"{a}-{b}" for a, b in map(self.span, self.segments))
            raise ValueError(f"frame {frame} is outside every registered odometry segment "
                             f"(frames {spans})")
        ii, sm = self.segments[int(self.seg[frame])]
        ox = np.interp(frame, ii, sm[:, 0])
        oy = np.interp(frame, ii, sm[:, 1])
        return self.pos[frame] / self.S + np.array([ox, oy])


# ---------------------------------------------------------------------------
# the label rule
# ---------------------------------------------------------------------------

PURITY, DELTA, SUB = 0.60, 0.12, 16    # majority share, registration error (tiles), samples/axis


def fractions(g, A, S, size_tiles, shift=(0.0, 0.0)) -> np.ndarray:
    """(rows, cols, N_CLASSES) class shares of the map area under each cell, from SUB x SUB
    samples: cell (row j, col i) spans map x from A[0] + i / S[0] to A[0] + (i + 1) / S[0]."""
    cols, rows = size_tiles
    u = (np.arange(SUB) + 0.5) / SUB
    jj, ii = np.mgrid[0:rows, 0:cols]
    x = A[0] + shift[0] + (ii[..., None, None] + u[None, None, None, :]) / S[0]
    y = A[1] + shift[1] + (jj[..., None, None] + u[None, None, :, None]) / S[1]
    x = np.broadcast_to(x, (rows, cols, SUB, SUB))
    y = np.broadcast_to(y, (rows, cols, SUB, SUB))
    tx, ty = np.floor(x).astype(int), np.floor(y).astype(int)
    map_rows, map_cols = g.shape
    inside = (tx >= 0) & (tx < map_cols) & (ty >= 0) & (ty < map_rows)
    k = np.full(x.shape, OFF_MAP, np.int8)
    k[inside] = g[ty[inside], tx[inside]]
    return np.stack([(k == c).mean(axis=(2, 3)) for c in range(N_CLASSES)], axis=-1)


def robust_chars(g, A, S, size_tiles) -> np.ndarray:
    """(rows, cols) '<U1': each cell's majority class where it holds >= PURITY of the cell under
    all 9 shifts of +/-DELTA tile and is a terrain class, else '?'."""
    cols, rows = size_tiles
    agree = np.ones((rows, cols), bool)
    best = None
    for dx in (-DELTA, 0.0, DELTA):
        for dy in (-DELTA, 0.0, DELTA):
            f = fractions(g, A, S, size_tiles, (dx, dy))
            k = f.argmax(-1)
            ok = f.max(-1) >= PURITY
            if best is None:
                best = k
            agree &= ok & (k == best)
    chars = np.full((rows, cols), UNLABELLED, "<U1")
    for k, c in enumerate(TERRAIN):
        chars[agree & (best == k)] = c
    return chars


# ---------------------------------------------------------------------------
# the refusals
# ---------------------------------------------------------------------------

GAS_FRACTION = 0.05        # detect_zone(...).at_least: any tint refuses the cell...
GAS_GROW = 3               # ...and the cells around it (a 3 x 3 dilation)
OCCLUDED_FRACTION = 0.10   # share of a cell under a padded detector box that refuses it
TEXT_FRACTION = 0.02       # share of a cell under TEXT_BOX or KILL_BOX that refuses it

# RECT px added around each detection's rectified bounding box: (sides, above, below). The boxes
# already take in the name tag, HP and ammo bars (Starfruit frame 600: the hero's box tops out at
# the name tag, rect y 356 vs 362; a crate's at its "5000", 378 vs 382), so this is only a margin.
# Entity-detector boxes are all "brawler"; projectile-detector boxes go by label.
RECT_PAD = {"brawler": (14, 14, 8), "Power Cube Box": (10, 10, 6), "other": (8, 8, 8)}
# The "Brawlers left: 10" readout runs past the emulator mask's brawlers_left rect on the 1440p
# captures (text to x~455 against the mask's 400), hence a wider box. Viewport px (x0, y0, x1, y1).
TEXT_BOX = (0, 0, 480, 115)
# Likewise the kill card: its names reach x~447 and its drop shadow y~262 (Starfruit frames
# 910/1000/1840/2140) against the mask's kill_feed x<=440, y<=252.
KILL_BOX = (0, 110, 470, 270)

# Transient content the detectors miss (shot splashes, damage numbers, shells in flight): compare
# the frame with frames +/-NEIGHBOUR, shifted by the odometry delta (static ground lines up to a
# median max-channel difference of 2). A cell is dropped when more than TRANSIENT_FRACTION of its
# pixels differ by more than TRANSIENT_DIFF from EITHER neighbour. Grass sway and water ripple stay
# under it (measured on Starfruit frame 640); a splash reads 24-100 %. A full-screen flash (a
# super, Starfruit frame 722) fails most of the view, so pick another frame rather than raise it.
NEIGHBOUR, TRANSIENT_DIFF, TRANSIENT_FRACTION, MIN_ALIGNED_PX = 8, 40, 0.12, 400


def rect_box(plan: RectifyPlan, xyxy) -> np.ndarray:
    """A viewport-px box's four corners in rectified px."""
    x0, y0, x1, y1 = xyxy
    q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(q, plan.M).reshape(-1, 2)


def coverage(plan: RectifyPlan, polys) -> np.ndarray:
    """(rows, cols) share of each cell inside any of the convex rectified-px polygons."""
    cols, rows = plan.size_tiles
    ppt = plan.pixels_per_tile
    mask = np.zeros((rows * ppt, cols * ppt), np.float32)
    for p in polys:
        if np.isfinite(p).all():
            cv2.fillConvexPoly(mask, np.clip(p, -1e4, 1e4).round().astype(np.int32), 1.0)
    return cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_AREA)


def box_cells(plan: RectifyPlan, boxes) -> np.ndarray:
    """(rows, cols) bool: cells under a detector box. `boxes` are (kind, viewport xyxy), kind a
    RECT_PAD key; each box's rectified bounding rectangle grows by its kind's pad."""
    polys = []
    for kind, xyxy in boxes:
        q = rect_box(plan, xyxy)
        if not np.isfinite(q).all():
            continue
        side, above, below = RECT_PAD[kind]
        x0, y0 = q.min(0)
        x1, y1 = q.max(0)
        polys.append(np.array([[x0 - side, y0 - above], [x1 + side, y0 - above],
                               [x1 + side, y1 + below], [x0 - side, y1 + below]]))
    return coverage(plan, polys) > OCCLUDED_FRACTION


def text_cells(plan: RectifyPlan) -> np.ndarray:
    """(rows, cols) bool: cells under the brawlers-left text or the kill card."""
    return coverage(plan, [rect_box(plan, TEXT_BOX), rect_box(plan, KILL_BOX)]) > TEXT_FRACTION


def grown_gas(gassed) -> np.ndarray:
    """`detect_zone(...).at_least(GAS_FRACTION)` grown by one cell each way."""
    kernel = np.ones((GAS_GROW, GAS_GROW), np.uint8)
    return cv2.dilate(np.asarray(gassed).astype(np.uint8), kernel).astype(bool)


def transient_cells(rect, neighbours, plan: RectifyPlan) -> np.ndarray:
    """(rows, cols) bool: cells that change against any neighbour frame.

    `neighbours` are (rectified neighbour frame, its odometry position minus this frame's, in
    tiles). Each is shifted onto this frame, and a cell is transient when more than
    TRANSIENT_FRACTION of the pixels both frames see differ by more than TRANSIENT_DIFF in some
    channel. A cell with under MIN_ALIGNED_PX such pixels is not judged."""
    cols, rows = plan.size_tiles
    ppt = plan.pixels_per_tile
    worst = np.zeros((rows, cols))
    here = rect.astype(np.int16)
    size = (rect.shape[1], rect.shape[0])
    for other, delta in neighbours:
        d = np.asarray(delta) * ppt
        M = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
        moved = cv2.warpAffine(other, M, size).astype(np.int16)
        seen = plan.valid & (cv2.warpAffine(plan.valid.astype(np.uint8), M, size) > 0)
        big = (np.abs(moved - here).max(-1) > TRANSIENT_DIFF) & seen
        cnt = seen.reshape(rows, ppt, cols, ppt).sum((1, 3))
        hit = big.reshape(rows, ppt, cols, ppt).sum((1, 3))
        frac = np.where(cnt >= MIN_ALIGNED_PX, hit / np.maximum(cnt, 1), 0.0)
        worst = np.maximum(worst, frac)
    return worst > TRANSIENT_FRACTION


@dataclass
class FrameLabels:
    """One frame's projected labels. `refused` maps each refusal to its (rows, cols) mask."""
    chars: np.ndarray
    projected: int                     # cells the label rule labelled, before any refusal
    refused: dict

    def stats(self) -> dict:
        return {"projected": self.projected} | {k: int(m.sum()) for k, m in self.refused.items()} \
            | {"labelled": int((self.chars != UNLABELLED).sum())}


def label_cells(g, A, S, plan: RectifyPlan, gassed, boxes, transient) -> FrameLabels:
    """The label rule, then every refusal: unobserved cells, gas grown by one, detector `boxes`
    ((kind, viewport xyxy)), the HUD text and kill card, and `transient` cells
    (`transient_cells`). `gassed` is `detect_zone(rect, plan, cfg).at_least(GAS_FRACTION)`."""
    chars = robust_chars(g, A, S, plan.size_tiles)
    projected = int((chars != UNLABELLED).sum())
    unobserved = ~observed_cells(plan)
    refused = {"unobserved": unobserved, "gas": grown_gas(gassed),
               "occluded": box_cells(plan, boxes), "text": text_cells(plan),
               "transient": np.asarray(transient) & ~unobserved}
    for mask in refused.values():
        chars[mask] = UNLABELLED
    return FrameLabels(chars, projected, refused)


# ---------------------------------------------------------------------------
# the record each label carries
# ---------------------------------------------------------------------------

def note(map_csv: str, date: str, S, A, leave_why: str = "", broken=()) -> str:
    """The `notes` text of a projected `LabelGrid`: what it was made from and how. `broken` holds
    the (col, row) map cells this frame leaves because they are broken by then (`grid_at`)."""
    leave = f"{leave_why}, " if leave_why else ""
    gone = (f"map cells broken by this frame ({' '.join(f'{c},{r}' for c, r in broken)}), "
            if len(broken) else "")
    return (f"Machine-made {date}, not hand-drawn: projected from {map_csv}. Registration: "
            f"map-image NCC at scale ({S[0]:.3f}, {S[1]:.3f}) smoothed through odometry, "
            f"A=({A[0]:.3f}, {A[1]:.3f}) = world tile of rect px (0,0). A cell is its majority "
            f"ground-footprint class when that class holds >= {PURITY:.2f} of it under every "
            f"+/-{DELTA:.2f}-tile shift; {leave}{gone}off-map, gas (plus one cell), HUD, the "
            f"brawlers-left text and kill card (both wider here than the mask), cells under "
            f"detected brawlers/crates/cubes/projectiles, and cells that change against the frames "
            f"{NEIGHBOUR} either side (splashes, damage numbers) are left '?'.")


def labelled_tiles(chars, A, S) -> set[tuple[int, int]]:
    """Map tiles (col, row) under the centres of a frame's labelled cells. Two label sets that
    share none of these share no labelled ground: the held-out rule."""
    j, i = np.nonzero(np.asarray(chars) != UNLABELLED)
    x = np.floor(A[0] + (i + 0.5) / S[0]).astype(int)
    y = np.floor(A[1] + (j + 0.5) / S[1]).astype(int)
    return set(zip(x.tolist(), y.tolist()))
