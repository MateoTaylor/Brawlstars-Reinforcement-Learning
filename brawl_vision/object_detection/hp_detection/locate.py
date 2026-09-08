"""Finding the HP text and the HP bar inside a detection box. Geometry and colour only.

**Nothing here reads a number.** This module answers "where on these pixels is the health
readout", `glyphs.py` answers "which digit is this shape", and `read.py` is the only place that
turns those into an integer with a confidence. Keeping localization separate is what makes the
failure modes below testable one at a time.

#### The layout, measured off `day10_gameplay.mp4`

Brawl Stars stacks a brawler's readout in a fixed order, top to bottom: a power-level pip, the
player's NAME, the HP number, the HP bar, and -- for your own brawler only -- an ammo bar and a
super-charge bar. Two properties of that stack are what this module is built on.

**The HP number is white; everything else near it is not.** The nameplate is team-coloured (green
for you, salmon-pink for an enemy) and the bars are saturated. So `(S < 70) & (V > 175)` isolates
the digits from their own neighbourhood without knowing anything about the map. That threshold is
doing more work than it looks: it is also what rejects the nameplate, which is the one other run
of glyphs in the same font a few pixels away.

**A digit is 15-17 px tall, always.** Across 195 harvested glyphs the height histogram is
12:5 13:4 14:2 **15:47 16:103 17:16** 18:2 19:6 20:4 21:2 24:4 -- 85% inside 15-17, and the
stragglers are damage popups caught mid-animation rather than mis-sized HP digits. This holds
because the capture is aspect-normalized and the camera is near-orthographic (`project.py`
measures a 40% perspective swing top-to-bottom on a POSITION, but the HUD is drawn in screen
space and does not scale at all). Horizontal pitch is a median 13 px, p05 9 (a `1`), p95 16.

#### Why the search is anchored on the DIGITS and not on the bar

The obvious design is to find the coloured bar first and read the text off it. It does not
survive real footage, for two independent reasons found by trying it.

**A hue-agnostic "saturated horizontal run" locks onto the map.** On the first enemy tested the
best-scoring row ran the full width of the crop at HSV hue 93 -- that is the day10 skin's TEAL
BUSH, not a red health bar. Colour alone does not distinguish a bar from a background.

**A class-specific hue does not save it either, because the bar DEPLETES.** The coloured run is
the filled portion only, so at half health it is half as long and its centre has slid left. Any
test anchored on it inherits that drift; an early version rejected a third of low-health frames
as "off-centre" for exactly this reason. The depleted remainder is a pale lavender that is not
cleanly separable from white text or a pale sprite, so recovering the bar's TRUE extent is not
the cheap operation it appears to be.

The digit row has neither problem: it is the same size and the same colour at 1 HP as at full.

#### What the bar is still used for

`find_bar` remains, because the filled run is genuinely reliable *given* a hue and a place to
look, and it is a free cross-check on the digits -- see `read.py`. It is anchored UNDER the
already-found digit row (measured: the bar's first solid row sits a median 3-4 px below the
digits' bottom edge), which is what keeps it off the bushes. Player green is hue 46-66 (median
56) and enemy red is 166-180 (median 173); those bands come from 161 measured bars.

**The bar's total length is deliberately not reported.** See `read.py` for why a fill RATIO is
not shipped rather than shipped as a guess.
"""
from dataclasses import dataclass

import cv2
import numpy as np

# Popup glyphs are the damage and healing numbers the game floats over a brawler when it is hit.
# They are the SAME FONT and the SAME WHITE, drawn much larger, and they animate straight across
# the HP readout -- so they are the single most destructive thing in this module's way. Anything
# white and taller than this is one, and `read.py` treats an overlap as fatal rather than noisy.
# 21 sits above the 19 px tail of real digits and well below the ~60 px a popup reaches.
POPUP_MIN_H = 21
POPUP_MIN_AREA = 150

# A component smaller than this is anti-aliasing debris or a sprite highlight, never a glyph.
MIN_GLYPH_AREA = 20
# Width band for a glyph at this scale. 4 is a `1`, 16 is a wide `0` with its outline.
MIN_GLYPH_W, MAX_GLYPH_W = 4, 16

# Widest bare gap allowed between consecutive glyphs of one number, in pixels. MEASURED over 893
# consecutive-glyph pairs on intact rows: 1-4 px, median 2, max 4. The narrowest glyph measured 6
# px wide, so one missing digit leaves a hole of at least ~8 px. 6 is the midpoint of a gap with
# nothing in it, which is why this separates cleanly instead of trading errors.
MAX_GLYPH_SPACE = 6

# How far apart two components may sit and still be called the same row of text: 3 px in top edge
# and 2 px in height. Tight, because the whole point is to separate the HP number from the
# nameplate directly above it.
ROW_TOP_TOL = 3
ROW_HEIGHT_TOL = 2

# Where under the digits to look for the bar. Measured bar-top-minus-digit-bottom is a median 3-4
# px with a p95 of 16; the window starts slightly ABOVE the digit bottom because the number
# straddles the bar's top edge when HP is short enough to centre it low.
BAR_SEARCH_ABOVE = 4
BAR_SEARCH_BELOW = 24
BAR_MIN_LEN = 16


@dataclass(frozen=True)
class Blob:
    """One white connected component, with its own mask already cut out."""
    xyxy: tuple[int, int, int, int]
    mask: np.ndarray

    @property
    def x0(self) -> int:
        return self.xyxy[0]

    @property
    def width(self) -> int:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> int:
        return self.xyxy[3] - self.xyxy[1]


@dataclass(frozen=True)
class DigitRow:
    """The run of white glyphs that is the HP number, left to right.

    `popup_overlap` is reported rather than acted on: this module's job is to say what is on the
    pixels, and whether a covered readout is fatal or merely suspicious is a policy `read.py`
    owns. It is fatal, but that decision belongs one layer up.
    """
    blobs: tuple[Blob, ...]
    popup_overlap: bool

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (min(b.xyxy[0] for b in self.blobs), min(b.xyxy[1] for b in self.blobs),
                max(b.xyxy[2] for b in self.blobs), max(b.xyxy[3] for b in self.blobs))

    @property
    def centre_x(self) -> float:
        x0, _, x1, _ = self.xyxy
        return (x0 + x1) / 2

    @property
    def bottom(self) -> int:
        return self.xyxy[3]

    @property
    def pitch(self) -> float:
        """Median left-edge spacing. The unit the centring test is expressed in, because a
        MISSING digit displaces the row's centre by exactly half of one."""
        if len(self.blobs) < 2:
            return float(self.blobs[0].width) if self.blobs else 1.0
        gaps = [self.blobs[i + 1].x0 - self.blobs[i].x0 for i in range(len(self.blobs) - 1)]
        return float(np.median(gaps))

    @property
    def widest_space(self) -> int:
        """Widest bare gap between consecutive glyphs, in pixels -- right edge to next left edge.

        **An ABSOLUTE space, not a ratio against the row's own spacing, and that distinction was a
        bug.** The first version compared the widest left-edge pitch to the MEDIAN pitch, which
        fails exactly when it matters: on `10400` with the `0` and `4` both covered, the surviving
        gaps are 36 px and 15 px, and a median over two samples is their mean -- so the 36 drags
        the reference up to 25.5 and hides itself at a ratio of 1.41. The read came back as `100`
        at confidence 1.00. (`configs/vision.yaml` warns about this same trap for
        `odometry.min_windows`: a median over two samples is not a robust statistic, it is an
        average.)

        The absolute spacing has no such failure because it is a property of the FONT, and it is
        measured: over 893 consecutive-glyph pairs on intact rows the space runs 1-4 px, median 2,
        maximum 4. The narrowest glyph is 6 px wide, so a single missing digit cannot leave a hole
        smaller than about 6 + 2 = 8 px. 4 and 8 do not overlap, which is what makes this a test
        rather than a threshold.
        """
        if len(self.blobs) < 2:
            return 0
        return max(self.blobs[i + 1].xyxy[0] - self.blobs[i].xyxy[2]
                   for i in range(len(self.blobs) - 1))


@dataclass(frozen=True)
class Bar:
    """The FILLED portion of the health bar. See the module docstring: this is not the whole bar,
    and the difference is the entire reason no fill ratio is reported."""
    xyxy: tuple[int, int, int, int]
    hue: int

    @property
    def length(self) -> int:
        return self.xyxy[2] - self.xyxy[0]


def crop_for(det, frame_shape, height_frac: float, pad_frac: float):
    """The slice of frame a readout can be in, plus the box centre inside it.

    Returns `(x0, y0, x1, y1, box_centre_x)` in frame pixels, with the centre already expressed
    relative to `x0` so callers never mix the two coordinate systems.

    **Padded horizontally on purpose.** A health bar is drawn to a width set by MAX HP and
    routinely overhangs the detector's box -- measured up to 160 px of bar against a 135 px box.
    The padding costs the risk of pulling a NEIGHBOUR's readout into the crop, which is real in a
    fight; the centring test in `read.py` is what rejects those, and it is why that test is not
    optional.
    """
    h, w = frame_shape[:2]
    bx0, by0, bx1, by1 = det.xyxy
    pad = (bx1 - bx0) * pad_frac
    x0 = max(0, int(round(bx0 - pad)))
    x1 = min(w, int(round(bx1 + pad)))
    y0 = max(0, int(round(by0)))
    y1 = min(h, int(round(by0 + (by1 - by0) * height_frac)))
    return x0, y0, x1, y1, (bx0 + bx1) / 2 - x0


def white_mask(crop_hsv: np.ndarray, max_sat: int, min_val: int) -> np.ndarray:
    """Pixels that are the HP number's white. Not a general "bright" test -- the SATURATION
    ceiling is what drops the team-coloured nameplate a few pixels above, which is otherwise the
    same font at the same size and would merge into the same row."""
    return ((crop_hsv[:, :, 1] < max_sat) & (crop_hsv[:, :, 2] > min_val)).astype(np.uint8)


def find_digit_row(crop_hsv: np.ndarray, box_centre_x: float, *, max_sat: int, min_val: int,
                   min_h: int, max_h: int) -> DigitRow | None:
    """The most plausible run of HP digits in `crop_hsv`, or None if there is no such run.

    Candidate rows are groups of similarly-sized components sharing a top edge. When two groups
    tie on size the one nearer `box_centre_x` wins -- with a padded crop, a neighbouring
    brawler's number is a perfectly well-formed row and length alone cannot separate them.
    """
    mask = white_mask(crop_hsv, max_sat, min_val)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    candidates, popups = [], []
    for i in range(1, n):
        x, y, w, h, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                            stats[i, cv2.CC_STAT_AREA])
        if area < MIN_GLYPH_AREA:
            continue
        if min_h <= h <= max_h and MIN_GLYPH_W <= w <= MAX_GLYPH_W:
            candidates.append((x, y, w, h, i))
        elif h >= POPUP_MIN_H and area >= POPUP_MIN_AREA:
            popups.append((x, y, x + w, y + h))

    if not candidates:
        return None

    best: list = []
    best_key = None
    for anchor in candidates:
        group = sorted((c for c in candidates
                        if abs(c[1] - anchor[1]) <= ROW_TOP_TOL
                        and abs(c[3] - anchor[3]) <= ROW_HEIGHT_TOL),
                       key=lambda c: c[0])
        cx = (group[0][0] + group[-1][0] + group[-1][2]) / 2
        key = (len(group), -abs(cx - box_centre_x))
        if best_key is None or key > best_key:
            best, best_key = group, key

    blobs = []
    for x, y, w, h, i in best:
        blobs.append(Blob((x, y, x + w, y + h), (labels[y:y + h, x:x + w] == i)))
    row = DigitRow(tuple(blobs), popup_overlap=False)

    rx0, ry0, rx1, ry1 = row.xyxy
    overlapped = any(px0 < rx1 and px1 > rx0 and py0 < ry1 and py1 > ry0
                     for px0, py0, px1, py1 in popups)
    return DigitRow(tuple(blobs), popup_overlap=overlapped)


def _longest_run(row: np.ndarray) -> tuple[int, int]:
    """Length and start of the longest contiguous True run. Run LENGTH rather than a count of set
    pixels, because the thing being looked for is a solid bar and the thing being rejected is
    text -- which has the same pixel count spread over gaps."""
    best = best_start = run = start = 0
    for i, v in enumerate(row):
        if v:
            if run == 0:
                start = i
            run += 1
            if run > best:
                best, best_start = run, start
        else:
            run = 0
    return best, best_start


def find_bar(crop_hsv: np.ndarray, hue_range: tuple[int, int], below_y: int) -> Bar | None:
    """The filled health bar under a digit row, searched only in this class's hue.

    `hue_range` may wrap past 179 back through 0 -- red does, and `brawl_vision.config.validate`
    permits exactly that asymmetry for `zone_hsv_*` for the same reason.
    """
    h, s, v = crop_hsv[:, :, 0], crop_hsv[:, :, 1], crop_hsv[:, :, 2]
    lo, hi = hue_range
    hue_ok = (h >= lo) & (h <= hi) if lo <= hi else ((h >= lo) | (h <= hi))
    solid = hue_ok & (s > 110) & (v > 90)

    y_from = max(0, below_y - BAR_SEARCH_ABOVE)
    y_to = min(crop_hsv.shape[0], below_y + BAR_SEARCH_BELOW)
    best = (0, 0, 0)
    for y in range(y_from, y_to):
        length, start = _longest_run(solid[y])
        if length > best[0]:
            best = (length, start, y)
    length, start, y = best
    if length < BAR_MIN_LEN:
        return None
    return Bar((start, y, start + length, y + 1), int(np.median(h[y, start:start + length])))
