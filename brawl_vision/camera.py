"""Camera geometry: HUD masking (Phase B), the ground-plane homography (Phase C), and
rectification to a top-down tile grid (Phase D). See Terrain_Perception_Build_Plan.md.

Everything here operates on the NORMALIZED viewport (Section 3) — `capture.normalize_viewport`'s
output, not a raw capture. That is why the mask is stored in normalized `[0, 1]` coordinates:
it then survives both a resolution change and the training-cave-vs-Showdown framing difference,
because both sources land on the same viewport before this module sees them.

**What counts as HUD here: SCREEN-ANCHORED UI, and nothing else.** The distinction is not
cosmetic. Anything that follows the world — other players' nameplates, loot boxes, the auto-aim
reticle, and *the local player's own nameplate* — moves relative to the screen and cannot be
masked by a static rectangle. Those are the entity chunk's problem, and the occupancy layer
already has the right answer for them (an occluded cell is unobserved; cast no vote). Measured on
`zone_grows_from_east.mp4`, the local player's nameplate wanders 330 px horizontally — 4.3 tiles
— so masking it statically would cost a 515x205 px hole in the most valuable part of the frame
and still not cover it.

**Downstream wants the BOOLEAN mask, not blacked-out pixels, and Phase F especially.** Zeroing a
UI region paints a hard-edged black rectangle that is *identical between consecutive frames*.
`cv2.phaseCorrelate` would see that as strong static structure and bias the estimated shift
toward zero — i.e. toward "the camera did not move" — which is precisely the failure odometry
cannot afford. `apply()` exists for visualization and for stages that genuinely want pixels
removed; `bool_at()` is what Phase F and Phase H should consume.
"""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import HOMOGRAPHY_PATH, HUD_MASK_PATH


@dataclass(frozen=True)
class HudRect:
    """One UI region, in NORMALIZED viewport coordinates: `x`/`y` are the top-left corner and
    `w`/`h` the extent, all as fractions of the normalized viewport in `[0, 1]`."""
    name: str
    x: float
    y: float
    w: float
    h: float

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) inclusive pixel bounds, clipped into the frame."""
        x0 = max(0, min(width - 1, int(round(self.x * width))))
        y0 = max(0, min(height - 1, int(round(self.y * height))))
        x1 = max(x0, min(width - 1, int(round((self.x + self.w) * width)) - 1))
        y1 = max(y0, min(height - 1, int(round((self.y + self.h) * height)) - 1))
        return x0, y0, x1, y1


@dataclass(frozen=True)
class HudMask:
    rects: tuple[HudRect, ...]
    notes: str = ""

    def bool_at(self, shape) -> np.ndarray:
        """(H, W) bool — True where the pixel is UI and must not be treated as world.

        `shape` may be an image's `.shape` or a plain `(h, w)`. This is the form Phases F and H
        should consume; see the module docstring on why blacking pixels out instead would bias
        phase correlation toward zero motion.
        """
        h, w = shape[0], shape[1]
        mask = np.zeros((h, w), dtype=bool)
        for rect in self.rects:
            x0, y0, x1, y1 = rect.pixels(w, h)
            mask[y0:y1 + 1, x0:x1 + 1] = True
        return mask

    def apply(self, image: np.ndarray, fill=0) -> np.ndarray:
        """A COPY of `image` with every UI region filled. For debug overlays and for stages that
        genuinely want the pixels gone — not for odometry (see the module docstring)."""
        out = image.copy()
        h, w = out.shape[0], out.shape[1]
        for rect in self.rects:
            x0, y0, x1, y1 = rect.pixels(w, h)
            out[y0:y1 + 1, x0:x1 + 1] = fill
        return out

    def coverage(self, shape) -> float:
        """Fraction of the frame the mask removes. Worth asserting a ceiling on: the mask is
        subtracted from every downstream stage's evidence, so an over-broad one silently starves
        classification rather than failing."""
        return float(self.bool_at(shape).mean())


def load_hud_mask(path=HUD_MASK_PATH) -> HudMask:
    raw = json.loads(Path(path).read_text())
    rects = tuple(
        HudRect(name=r["name"], x=float(r["x"]), y=float(r["y"]),
                w=float(r["w"]), h=float(r["h"]))
        for r in raw["rects"]
    )
    if not rects:
        raise ValueError(f"{path} declares no rects")
    return HudMask(rects=rects, notes=raw.get("notes", ""))


# ---------------------------------------------------------------------------
# Phase C -- the homography
# ---------------------------------------------------------------------------

# Output scale of the rectified patch, in pixels per tile. A free choice, not a measurement: the
# native on-screen pitch is ~77 px/tile at frame centre, so 48 downsamples ~1.6x. That is ample
# for telling floor/wall/bush/water/fence apart -- the classifier needs texture, not detail -- and
# it shrinks the rectified patch (and therefore Phase H's cost) quadratically. Raise it if Phase H
# ever turns out to be starved of signal; nothing else depends on the value.
PIXELS_PER_TILE = 48

# Below this, |H[2, :2]| is indistinguishable from zero and the camera is orthographic-oblique
# rather than perspective. Measured on real footage it is ~1.2e-02, three orders above this.
_AFFINE_EPS = 1e-5


@dataclass(frozen=True)
class CameraModel:
    """Ground-plane tile coordinates <-> normalized-viewport pixels.

    `H` maps GROUND-plane tile (x, y) to pixels. That choice matters: wall blocks are ~0.87 tiles
    tall in projected terms, so a fit to their crisp top faces describes a plane a whole cell away
    from the one that actually blocks movement. See `solve_camera_model` for how the ground plane
    is recovered from the wall-top fit.

    `H_top` is kept alongside it for diagnostics and because Phase H may want it: a wall's visible
    appearance lives on that plane even though its collision footprint lives on the ground.
    """
    H: np.ndarray
    H_inv: np.ndarray
    H_top: np.ndarray
    pixels_per_tile: int
    viewport: tuple[int, int]          # (w, h) of the normalized viewport this was fit in
    is_affine: bool
    parallax_tiles: float              # mean ground-vs-top offset, in tiles
    source_frame: str = ""
    notes: str = ""

    def tile_to_px(self, tiles: np.ndarray) -> np.ndarray:
        t = np.asarray(tiles, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(t, self.H).reshape(-1, 2)

    def px_to_tile(self, px: np.ndarray) -> np.ndarray:
        p = np.asarray(px, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H_inv).reshape(-1, 2)

    def check_viewport(self, shape) -> None:
        """Raise if `shape` is not the viewport this model was fit in. A homography applied at a
        different framing produces a plausible-looking, uniformly wrong rectification -- the
        expensive kind of wrong, because nothing errors."""
        h, w = shape[0], shape[1]
        if (w, h) != tuple(self.viewport):
            raise ValueError(
                f"camera model was calibrated for a {self.viewport[0]}x{self.viewport[1]} "
                f"viewport, got {w}x{h}. Normalize the frame first (capture.normalize_viewport)."
            )


def _read_csv(path):
    return [r for r in csv.DictReader(open(path)) if not next(iter(r.values())).startswith("#")]


def solve_camera_model(points_csv, height_csv, viewport, pixels_per_tile=PIXELS_PER_TILE,
                       exclude_pairs=(), source_frame="") -> tuple["CameraModel", dict]:
    """Fit the ground-plane homography. Returns `(model, report)`; the report carries the
    residuals and diagnostics a caller should look at before trusting the result.

    **Two stages, because the crisp features and the wanted plane are not the same plane.**

    1. Fit `H_top` to the annotated wall-block TOP corners. Those are the only unambiguous
       features in the frame -- the floor of both maps is effectively textureless, with no tile
       seams to click at all.
    2. Transfer that fit down to the ground with a **planar homology**. Two parallel planes seen
       by one camera are related by a homology: centre at the vertical vanishing point (recovered
       by intersecting the annotated vertical edges), axis at the plane family's horizon (the
       image of the tile plane's line at infinity, i.e. the third row of `H_top^-1`), and a single
       free ratio `mu` fit to the top/base pairs.

    **Why not fit the top->base pixel map as a plain homography?** It has 8 degrees of freedom
    against the homology's 5, and with a handful of noisy short segments it overfits badly:
    measured leave-one-out error was 33 px for the unconstrained fit against 5 px for the
    homology, on the same data. The constraint is doing real work.

    `exclude_pairs` drops annotation mistakes. A vertical pair is only meaningful if both clicks
    lie on the SAME vertical edge; a segment cutting diagonally across a block face looks
    plausible in isolation and is detectable only against its neighbours -- two edges a few dozen
    pixels apart must be near-parallel, so a wildly different slope is an error, not a measurement.
    """
    rows = _read_csv(points_csv)
    tiles = np.array([[float(r["tx"]), float(r["ty"])] for r in rows], np.float64)
    px = np.array([[float(r["px"]), float(r["py"])] for r in rows], np.float64)
    if len(tiles) < 6:
        raise ValueError(f"{points_csv}: need >=6 correspondences, got {len(tiles)}")

    H_top, _ = cv2.findHomography(tiles, px, method=0)
    if H_top is None:
        raise ValueError(f"{points_csv}: homography solve failed (degenerate points?)")
    proj = cv2.perspectiveTransform(tiles.reshape(-1, 1, 2), H_top).reshape(-1, 2)
    grid_res = np.linalg.norm(proj - px, axis=1)

    pairs = {}
    for r in _read_csv(height_csv):
        pairs.setdefault(r["pair"], {})[r["kind"]] = [float(r["px"]), float(r["py"])]
    keys = [k for k in sorted(pairs, key=lambda s: int(s)) if k not in set(exclude_pairs)]
    if len(keys) < 3:
        raise ValueError(f"{height_csv}: need >=3 usable height pairs, got {len(keys)}")

    lines = np.array([np.cross(np.append(pairs[k]["top"], 1.0), np.append(pairs[k]["base"], 1.0))
                      for k in keys])
    V = np.linalg.svd(lines)[2][-1]
    V = V / V[2]
    horizon = np.linalg.inv(H_top)[2, :].copy()
    horizon = horizon / np.linalg.norm(horizon[:2])

    tops = np.array([pairs[k]["top"] for k in keys], np.float64)
    bases = np.array([pairs[k]["base"] for k in keys], np.float64)

    def homology(mu):
        return np.eye(3) + (mu - 1.0) * np.outer(V, horizon) / (horizon @ V)

    def base_err(mu):
        q = homology(mu) @ np.hstack([tops, np.ones((len(keys), 1))]).T
        return np.linalg.norm((q[:2] / q[2]).T - bases, axis=1)

    lo, hi = 0.5, 2.0                       # ternary search; base_err is unimodal in mu
    for _ in range(90):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if base_err(a).mean() < base_err(b).mean():
            hi = b
        else:
            lo = a
    mu = (lo + hi) / 2

    H = homology(mu) @ H_top
    H = H / H[2, 2]
    H_inv = np.linalg.inv(H)

    probe = np.array([[x, y] for x in (0.15, 0.5, 0.85) for y in (0.2, 0.5, 0.85)]) * \
        np.array(viewport, np.float64)
    on_top = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), np.linalg.inv(H_top)).reshape(-1, 2)
    on_gnd = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
    parallax = on_gnd - on_top

    model = CameraModel(
        H=H, H_inv=H_inv, H_top=H_top, pixels_per_tile=pixels_per_tile,
        viewport=(int(viewport[0]), int(viewport[1])),
        is_affine=bool(np.abs(H[2, :2]).max() < _AFFINE_EPS),
        parallax_tiles=float(parallax[:, 1].mean()),
        source_frame=source_frame,
    )
    report = {
        "n_grid_points": int(len(tiles)),
        "grid_residual_px": {"mean": float(grid_res.mean()), "median": float(np.median(grid_res)),
                             "max": float(grid_res.max())},
        "height_pairs_used": keys,
        "height_pairs_excluded": list(exclude_pairs),
        "base_residual_px": {"mean": float(base_err(mu).mean()), "max": float(base_err(mu).max())},
        "vanishing_point": [float(V[0]), float(V[1])],
        "mu": float(mu),
        "H_perspective_row": [float(H[2, 0]), float(H[2, 1])],
        "parallax_tiles": {"mean": float(parallax[:, 1].mean()),
                           "min": float(parallax[:, 1].min()), "max": float(parallax[:, 1].max())},
    }
    return model, report


def save_camera_model(model: CameraModel, report: dict, path=HOMOGRAPHY_PATH) -> None:
    doc = {
        "_comment": [
            "GROUND-plane homography: H maps tile (x,y) -> pixels of the NORMALIZED viewport",
            "(brawl_vision.capture.normalize_viewport output), NOT a raw capture frame.",
            "H_top is the wall-top plane the corners were actually clicked on; H is that fit",
            "transferred down to the ground by a planar homology. See solve_camera_model.",
        ],
        "H": model.H.tolist(), "H_inv": model.H_inv.tolist(), "H_top": model.H_top.tolist(),
        "pixels_per_tile": model.pixels_per_tile,
        "viewport": list(model.viewport),
        "is_affine": model.is_affine,
        "parallax_tiles": model.parallax_tiles,
        "source_frame": model.source_frame,
        "notes": model.notes,
        "report": report,
    }
    Path(path).write_text(json.dumps(doc, indent=2) + "\n")


def load_camera_model(path=HOMOGRAPHY_PATH) -> CameraModel:
    raw = json.loads(Path(path).read_text())
    H = np.array(raw["H"], np.float64)
    return CameraModel(
        H=H, H_inv=np.array(raw["H_inv"], np.float64), H_top=np.array(raw["H_top"], np.float64),
        pixels_per_tile=int(raw["pixels_per_tile"]), viewport=tuple(raw["viewport"]),
        is_affine=bool(raw["is_affine"]), parallax_tiles=float(raw["parallax_tiles"]),
        source_frame=raw.get("source_frame", ""), notes=raw.get("notes", ""),
    )


# ---------------------------------------------------------------------------
# Phase D -- rectification (inverse perspective mapping)
# ---------------------------------------------------------------------------

# The rectified window is derived from the camera model rather than configured, so it cannot drift
# out of sync with a refit homography. It is the tile-space bounding box of the viewport's ground
# footprint, snapped OUTWARD to whole tiles -- outward because the snap must never crop real world,
# and to whole tiles because that is what keeps integer tile coordinates on rectified-pixel
# multiples of `pixels_per_tile` (see `RectifyPlan.tile_to_rect`).
#
# Measured on the shipped calibration: origin tile (-8, -8), 30 x 19 tiles, 1440 x 912 px.

# A homography is only usable over the half-plane on one side of its horizon; across it, points
# project through infinity and come back mirrored. Here the horizon sits ~4550 px outside the
# nearest viewport corner, so the whole window is safely on one side -- but that is a property of
# this camera, not a law, so it is checked rather than assumed.
_HORIZON_EPS = 1e-6


@dataclass(frozen=True)
class RectifyPlan:
    """Everything needed to warp normalized-viewport frames to a fixed top-down tile grid.

    Built once per camera model and reused for every frame: the warp matrix and the validity mask
    are both static, because the homography is static. Only `rectify()` runs per frame.

    **The rectified frame is CAMERA-RELATIVE, not world-anchored.** A camera translating parallel
    to the ground changes the world->screen homography to `H . translate(-d)`, so rectifying with
    the fixed calibration `H` leaves ground content shifted rigidly by `d` and otherwise identical.
    That is precisely why Phase F can recover camera motion with plain 2D phase correlation, and
    why this stage must come before it. It also means tile (0, 0) here is "wherever the calibration
    frame's origin corner was", not a map position -- anchoring to the map is Phase I's job.
    """
    M: np.ndarray                       # normalized-viewport px -> rectified px
    M_inv: np.ndarray                   # rectified px -> normalized-viewport px
    origin_tile: tuple[int, int]        # tile coords of rectified pixel (0, 0)
    size_tiles: tuple[int, int]
    size_px: tuple[int, int]            # (w, h)
    pixels_per_tile: int
    viewport: tuple[int, int]
    valid: np.ndarray                   # (h, w) bool -- rectified pixels backed by real world

    def rectify(self, image: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """Warp one normalized-viewport frame to the top-down tile grid.

        The HUD is deliberately NOT blacked out here even though `valid` knows where it is; see
        the module docstring. Downstream stages take the pixels and consult `valid` separately.
        """
        if image.shape[1] != self.viewport[0] or image.shape[0] != self.viewport[1]:
            raise ValueError(
                f"rectify() expects the {self.viewport[0]}x{self.viewport[1]} normalized viewport, "
                f"got {image.shape[1]}x{image.shape[0]}. Run capture.normalize_viewport first."
            )
        return cv2.warpPerspective(image, self.M, self.size_px, flags=interpolation)

    def tile_to_rect(self, tiles: np.ndarray) -> np.ndarray:
        t = np.asarray(tiles, np.float64).reshape(-1, 2)
        return (t - np.asarray(self.origin_tile, np.float64)) * self.pixels_per_tile

    def rect_to_tile(self, rect: np.ndarray) -> np.ndarray:
        r = np.asarray(rect, np.float64).reshape(-1, 2)
        return r / self.pixels_per_tile + np.asarray(self.origin_tile, np.float64)


def build_rectify_plan(model: CameraModel, hud: HudMask | None = None,
                       pad_tiles: int = 0) -> RectifyPlan:
    """Derive the rectification window from `model`.

    `hud`, when given, is folded into `valid` so downstream stages get one mask covering both
    reasons a rectified pixel is meaningless: it fell outside the viewport, or it landed on UI.
    Both are static, so this costs nothing per frame.

    `pad_tiles` widens the window beyond the visible footprint. It is 0 by default -- padding buys
    only known-invalid pixels -- and exists for Phase I, which may want margin to accumulate into.
    """
    w, h = model.viewport
    corners = model.px_to_tile(np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64))
    lo = np.floor(corners.min(0)).astype(int) - pad_tiles
    hi = np.ceil(corners.max(0)).astype(int) + pad_tiles
    ppt = model.pixels_per_tile
    size_tiles = (int(hi[0] - lo[0]), int(hi[1] - lo[1]))
    size_px = (size_tiles[0] * ppt, size_tiles[1] * ppt)

    # tile -> rectified px. Integer entries only, which is what makes tile corners land exactly on
    # pixel multiples of ppt rather than a fraction of a pixel off.
    S = np.array([[ppt, 0.0, -ppt * lo[0]],
                  [0.0, ppt, -ppt * lo[1]],
                  [0.0, 0.0, 1.0]], np.float64)
    M = S @ model.H_inv
    M_inv = np.linalg.inv(M)

    # Horizon guard: every corner of the window must project to the same side of infinity.
    w_px, h_px = size_px
    probe = np.array([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]], np.float64)
    ws = (M_inv @ np.vstack([probe.T, np.ones(4)]))[2]
    if np.abs(ws).min() < _HORIZON_EPS or len(set(np.sign(ws))) > 1:
        raise ValueError(
            "the rectification window crosses the camera's horizon, so part of it would be a "
            "mirrored projection of world behind the camera. The calibration is probably wrong: "
            f"homogeneous w at the window corners was {ws.tolist()}."
        )

    inside = np.full((h, w), 255, np.uint8)
    if hud is not None:
        inside[hud.bool_at((h, w))] = 0
    valid = cv2.warpPerspective(inside, M, size_px, flags=cv2.INTER_NEAREST, borderValue=0) > 127

    return RectifyPlan(
        M=M, M_inv=M_inv, origin_tile=(int(lo[0]), int(lo[1])), size_tiles=size_tiles,
        size_px=size_px, pixels_per_tile=ppt, viewport=(int(w), int(h)), valid=valid,
    )
