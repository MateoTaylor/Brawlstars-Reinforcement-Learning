"""Poison-gas detection. See Terrain_Perception_Build_Plan.md Phase G.

The gas has to stay legible to a human player whatever map is loaded underneath, which is what
makes a colour threshold the right tool: the overlay is deliberately high-contrast and consistent
across skins in a way terrain is not.

**Hue alone does not work, and the map that proves it is the one that was missing.** On
`showdown_alternate_map` the bushes are yellow-green with a median hue of 60 -- the same hue as the
gas. A hue-only band swallows the whole bush strip there (11.5% of the frame flagged against 4.0%
for the shipped band). What separates them is that gas is markedly **brighter and less saturated**:
the gas is a bright wash laid over the terrain, while foliage is dark, saturated and textured. So
the threshold is carried by S and V, with H only excluding the non-green half of the wheel.

**This runs every frame and is never cached.** Unlike terrain, the zone genuinely changes, and a
cell that was clear thirty seconds ago says nothing about now.

**Phase I is the consumer, and it needs cells, not pixels.** Gas tints the terrain beneath it, so a
gas-covered cell is a cell whose colours have been shifted by a full-screen effect, and gas never
leaves, so votes on it would outvote the real terrain in time. `ZoneMask.cells` is the per-cell
boolean that stage should consult, and it is deliberately computed against each cell's VALID
pixels only -- a cell half-covered by the HUD must not be called clear just because the visible
half happens to be.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from ..camera import RectifyPlan
from ..config import VisionConfig


@dataclass(frozen=True)
class ZoneMask:
    """`pixels` is at the rectified patch's resolution; `cells` is on the tile grid.

    `observed` marks cells with enough real world in them to judge at all. A cell that is mostly
    outside the viewport or under the HUD is neither gassed nor clear -- it is unknown, and
    collapsing that into "clear" is how a stage silently starts trusting the frame edge.
    """
    pixels: np.ndarray          # (h, w) bool
    cells: np.ndarray           # (tiles_y, tiles_x) bool
    cell_fraction: np.ndarray   # (tiles_y, tiles_x) float -- gassed share of each cell
    observed: np.ndarray        # (tiles_y, tiles_x) bool
    coverage: float             # gassed share of the observed area, for logging and tests

    @property
    def gassed_cells(self) -> int:
        return int(self.cells.sum())

    def at_least(self, fraction: float) -> np.ndarray:
        """Cells at least `fraction` gassed. Phase I should use a value well BELOW
        `min_cell_fraction`, because the two consumers want opposite errors: `cells` answers "is
        this cell in the zone" for the agent, where a wrong answer either way costs the same,
        while Phase I uses it to REFUSE to vote -- and there, missing gas casts a wrong vote on
        every frame for the rest of the match while over-flagging only slows the map filling in.
        Any tinting at all is reason enough to abstain."""
        return self.observed & (self.cell_fraction >= fraction)


def _per_cell_fraction(mask: np.ndarray, valid: np.ndarray, tiles: tuple[int, int], ppt: int):
    """Reduce a pixel mask to per-cell fractions of each cell's VALID pixels."""
    tw, th = tiles
    blocks = lambda a: a.reshape(th, ppt, tw, ppt).sum(axis=(1, 3))
    hits = blocks(mask.astype(np.int32))
    seen = blocks(valid.astype(np.int32))
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(seen > 0, hits / np.maximum(seen, 1), 0.0)
    return frac, seen


def detect_zone(rect: np.ndarray, plan: RectifyPlan, cfg: VisionConfig | None = None) -> ZoneMask:
    """Find the gas in one RECTIFIED frame.

    Thresholding after rectification rather than before keeps every stage in one coordinate frame
    and lets `plan.valid` do the HUD and out-of-viewport masking that would otherwise be repeated
    here. The resampling `rectify` applies is a mild downsample, which blurs nothing that a colour
    band cares about.
    """
    cfg = cfg or VisionConfig()
    if rect.shape[1] != plan.size_px[0] or rect.shape[0] != plan.size_px[1]:
        raise ValueError(
            f"detect_zone expects the rectified patch ({plan.size_px[0]}x{plan.size_px[1]}), got "
            f"{rect.shape[1]}x{rect.shape[0]}. Run RectifyPlan.rectify first."
        )
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    lo, hi = np.array(cfg.zone_hsv_low, np.uint8), np.array(cfg.zone_hsv_high, np.uint8)
    if cfg.zone_hsv_low[0] <= cfg.zone_hsv_high[0]:
        mask = cv2.inRange(hsv, lo, hi)
    else:
        # A hue range spanning red wraps past 179 back through 0, so it is the union of two bands.
        # `config.validate` permits low > high on H for exactly this and forbids it on S and V.
        a = cv2.inRange(hsv, np.array([lo[0], lo[1], lo[2]], np.uint8),
                        np.array([179, hi[1], hi[2]], np.uint8))
        b = cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8),
                        np.array([hi[0], hi[1], hi[2]], np.uint8))
        mask = cv2.bitwise_or(a, b)

    # Opening removes speckle -- damage numbers, pickup glints, a nameplate's green -- without
    # touching the clouds, which are tens of pixels across. Measured: it costs ~0.4 percentage
    # points of true coverage and removes rather more false positives than that.
    if cfg.zone_open_px > 1:
        k = np.ones((cfg.zone_open_px, cfg.zone_open_px), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    pixels = (mask > 0) & plan.valid
    frac, seen = _per_cell_fraction(pixels, plan.valid, plan.size_tiles, plan.pixels_per_tile)
    observed = seen >= cfg.zone_min_cell_pixels * plan.pixels_per_tile ** 2
    cells = observed & (frac >= cfg.zone_min_cell_fraction)
    denom = int(plan.valid.sum())
    return ZoneMask(pixels=pixels, cells=cells, cell_fraction=frac, observed=observed,
                    coverage=float(pixels.sum() / denom) if denom else 0.0)
