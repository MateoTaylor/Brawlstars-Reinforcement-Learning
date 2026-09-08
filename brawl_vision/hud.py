"""Screen-anchored HUD readouts. Right now that is one field: "Brawlers left: N".

**This module sits in the package root and not under `object_detection/`, because nothing here
needs a detection.** `hp_detection` reads a stack that is drawn *on a brawler* and therefore
cannot run without a box to anchor to; the match readout is drawn at a fixed place on the screen
and is there whether or not the detector saw anything. That is the same split
`data/hud_mask.json` already makes between screen-anchored UI and world-following UI, and this
module reads the one rect in that file it never had a reader for -- `brawlers_left`.

#### The layout, measured across six clips and two capture resolutions

Every constant below is in REFERENCE PIXELS at a 1126 px-tall viewport, scaled at read time by
`frame_height / 1126`. That works because `capture.normalize_viewport` trims every source to
exactly 16:9, so one scalar describes the whole frame and width is redundant with height.

**1126 because that is what deployment presents**, not because it is the commoner fixture:
`DeployCapture` resizes the normalized grab to 2002x1126, the viewport the whole vision stack is
calibrated at (BRAWL_DEPLOYMENT_DESIGN.md 9.4 item 4). The fixtures carry a second height for
free and it is the useful one -- `ClipReader` trims to 16:9 and never resizes, so
`bluestacks-example-new.mp4` arrives at its native **1920x1080** and is what tests the scaling
rather than letting it be assumed::

                       @1080        x1126/1080     @1126 measured
    colon left x        350           364.9          365 - 367
    colon dot           11x10         11.5x10.4      11x11
    colon dy (top-top)  18            18.8           19
    digit top y         37 - 38       38.6 - 39.6    38 - 40
    digit height        39 - 40       40.7 - 41.7    40 - 42
    digit width         19 - 32       19.8 - 33.4    19 - 33
    colon -> digit gap  13 - 14       13.6 - 14.6    14 - 16
    inter-digit gap     5             5.2            5 - 6

The scaled 1080 column lands inside the 1126 measurement on every row. **The HUD scales with
viewport height, exactly, and nothing here needs a per-recording calibration.**

**That is not true of the control buttons, which is why it is worth stating.** `gameplay.py`
found two button layouts in this same footage -- the game's own UI-scale setting -- and had to
calibrate the attack ring per recording: `zone_grows_from_east` and `day10_gameplay` put it at
(1490, 816) with r 65-78, while `day12_recording1` puts it at (1752, 782) with r 46. Both of
those clips report the colon at x = 366. **The UI-scale setting moves the controls and leaves the
top-left match readout alone**, so a fixed fractional position is safe here in a way it was not
there. If that ever stops being true the colon gate below fails closed rather than reading the
wrong pixels.

#### The colon is the landmark, and it is doing three jobs

"Brawlers left:" is white text in the same font as the HP digits, so a white-pixel search over
this band returns the LABEL's glyphs as readily as the count's, and cap-height letters (`B`, `l`,
`f`, `t`) measure the same 39-40 px as a digit. Height cannot separate them. The colon can: two
small squares at a fixed x, stacked 19 px apart, which nothing else in the band looks like.

It also decides whether the widget is on screen at all. The counter is absent during the
match-start fly-in and on the loading screen (measured: `bluestacks-example-new.mp4` has no colon
before frame 230 and does not lose one after it), and *nothing else* about those frames says so
-- the band still contains white pixels from the map. And it pins the count's x to within a few
pixels, which is what lets the `colon -> digit` gap gate reject the white world pixels that a
wide screen-space band inevitably admits.

**The third job is the one that needed no extra code: the colon's x IS the label's rendered
width.** The label is left-aligned at a fixed x (measured 44) and the colon closes it, so gating
the colon's position gates the string that precedes it. A different readout in the same corner --
`hud_mask.json` records the training cave putting "Damage per second" there -- is four characters
longer and lands its colon well outside a 26 px window. Checking the label glyph by glyph would
buy nothing over that.

#### What the count means, which is not what the observation wants

The game counts brawlers **including you**. `obs_schema`'s `meta.n_enemies_alive` is
`n_alive - hero_alive`, so the conversion is a subtraction whose operand this module cannot see:
call `BrawlersLeft.enemies_alive(hero_alive=...)` rather than doing it at the call site, so the
assumption is written down somewhere. The reading itself reports what is on the pixels.

#### The templates transferred, and that was the open question

`glyphs.py` classifies against ten mean templates harvested at **15-17 px** -- the HP font. These
digits are the same typeface at **40-42 px**, 2.6x larger, and whether `glyphs.normalize` is
genuinely scale-free was a thing to measure rather than to reason about. It is, and comfortably:
across the ten Showdown clips the accepted count glyphs score a per-clip **p05 of 0.893-0.930**
against templates they were never harvested from, against the HP population's own p05 of 0.698.
Downsampling a big clean glyph to the 18x14 canvas loses nothing that distinguishes a digit. No
second template bank, and `scripts/vision_hp_calibrate.py` does not need a HUD mode.

**Every failure in the first, ungated pass was segmentation and not classification** -- a 71x34
white blob from the map scoring 0.262 as a `4`, a 26x27 one scoring 0.546 as an `8`, dragging
that pass's score p05 down to 0.308. The structural gates removed all of it. Which is why the
gates below are structural and the score floor is a backstop rather than the defence.

#### What it reads, over 4684 sampled frames of ten clips

    clip                       samples   read        score p05 / min   margin p05 / min
    day10_gameplay                 918   100.0 %       0.893  0.890      0.185  0.160
    showdown_alternate_map         853   100.0 %       0.901  0.894      0.171  0.161
    9_5_brawlstars_eval            671   100.0 %       0.930  0.923      0.228  0.189
    day12_recording2               594    96.1 %       0.900  0.758      0.166  0.156
    showdown_alternate_map2        415   100.0 %       0.928  0.867      0.186  0.157
    day12_recording1               343    77.8 %       0.905  0.902      0.184  0.163
    zone_grows_from_east           244   100.0 %       0.906  0.906      0.160  0.152
    bluestacks-example-new (1080)  209    78.0 %       0.894  0.885      0.229  0.179
    showdown_has_gadget            166    99.4 %       0.894  0.890      0.179  0.161
    day12_recording3               271   100.0 %       0.900  0.896      0.223  0.186

**4538 reads. The two clips below 99% are not misses** -- `bluestacks-example-new` and
`day12_recording1` start before the match does, and in both, every non-read precedes the frame
the counter first appears on (230 and 380) with none after it. `day12_recording2` loses 22 frames
the same way and refuses exactly one mid-match frame as `too-long`, which is the reader failing
closed. Nothing in 4684 frames produced a wrong number.

**The free ground truth, and the check that would have caught a wrong one.** Brawlers left is
monotone non-increasing within a match -- a count that goes UP is a mis-segmentation, and no
labelling is needed to say so. Over 4528 consecutive pairs there are **zero increases**, and the
per-clip value runs walk cleanly down (`showdown_alternate_map2` reads 10, 9, 8, ... 1). That is
asserted in the footage test rather than left as a number in prose.

**Off-mode it stays quiet.** Three training-cave clips, 284 frames, zero reads -- but their band
is bare map, so this measures the reader not inventing a counter and NOT that it can tell one
labelled readout from another. `hud_mask.json` says the cave shows "Damage per second" in this
corner; these particular clips do not. The gap is covered by a test instead: a widget whose label
is four characters longer puts its colon outside the window and is refused.

#### Failure policy

A `BrawlersLeft` with `count = None` and a `status` saying which gate closed, rather than `None`
or a plausible number -- the taxonomy is rich enough that on the emulator "I could not find the
colon" and "I found three digits" want different fixes, which is the same argument
`read.py:HealthReading` makes for the HP readout. §6.4 of BRAWL_DEPLOYMENT_DESIGN.md is why it is
never a guess: §6.3's shadow state treats CV as the *check* on dead reckoning, so a missing read
costs one tick of confirmation and a fabricated one is silent.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .object_detection.hp_detection.glyphs import Glyph, GlyphBank, normalize

# The viewport height every constant in this module is expressed against. 1126 because that is
# what `DeployCapture` resizes to and what the vision stack is calibrated at -- not because it is
# the commoner fixture height. The 1080 clips exercise the scaling.
REF_H = 1126

# Rows of the frame the counter can occupy, in reference pixels. The digit row measures y 38-80,
# and the band is not widened further DOWNWARD for a reason: at y ~ 123 there is another white
# HUD element whose top 17 px leaked into an early measurement pass and produced a phantom 26x18
# "digit" in 40 of 163 frames.
BAND_Y = (24, 100)
# ...and how far right to search. The count ends by x 440; past there is world.
BAND_X = 560

# Where the colon's dots sit, and how far the search will chase them. The tolerance is ~7% of the
# position -- loose enough for a UI revision that nudges the widget, tight enough that a pair of
# white map pixels at the other end of the band is not a candidate.
COLON_X = 366
COLON_X_TOL = 26
# One dot, and the vertical top-to-top spacing of the two. Measured 11x11 and 19; a dot is square
# to within a pixel at every resolution measured.
COLON_DOT_PX = (7, 16)
COLON_DY = (14, 26)
COLON_DX_TOL = 4

# A count digit. Measured 40-42 px tall and 19-33 wide (a `1` is the narrow end, a `0` the wide).
DIGIT_H = (35, 47)
DIGIT_W = (14, 40)
# Colon's right edge to the first digit's left edge. Measured 14-16.
COLON_GAP = (8, 24)
# Right edge to next left edge WITHIN the count. Measured 5-6, and the narrowest glyph is 19 px
# wide, so a dropped digit cannot leave a hole under ~20. Same reasoning as
# `locate.MAX_GLYPH_SPACE`, at this scale.
DIGIT_SPACE_MAX = 12
# How much two glyphs of one number may disagree about where the row is.
ROW_TOP_TOL = 5
ROW_H_TOL = 6

# Below this a component is anti-aliasing debris. Scales with the SQUARE of the linear scale.
MIN_AREA = 40

# Solo Showdown fields ten. A read outside 1..this is a mis-segmentation by definition, and two
# digits is the most the field can need.
MAX_DIGITS = 2

OK = "ok"
NO_COLON = "no-colon"
NO_DIGITS = "no-digits"
TOO_LONG = "too-long"
LEADING_ZERO = "leading-zero"
OUT_OF_RANGE = "out-of-range"
LOW_GLYPH = "low-glyph"


@dataclass(frozen=True)
class BrawlersLeft:
    """What the "Brawlers left: N" readout says, or which gate stopped it being read.

    `count` includes the hero. See `enemies_alive`.
    """
    count: int | None
    status: str
    score: float = 0.0
    margin: float = 0.0
    digits: tuple[Glyph, ...] = ()
    xyxy: tuple[int, int, int, int] | None = None

    @property
    def ok(self) -> bool:
        return self.status == OK

    def enemies_alive(self, hero_alive: bool) -> int | None:
        """`obs_schema.meta.n_enemies_alive` -- non-hero entities alive.

        `hero_alive` is not optional and has no default, because the game stops counting you the
        moment you die and the off-by-one that produces is invisible in the output: a plausible
        integer either way. The caller knows whether the hero is alive; this module does not.
        """
        if self.count is None:
            return None
        return self.count - 1 if hero_alive else self.count


class HudReader:
    """Reads the screen-anchored match readout. Construct once, call per frame.

    Stateless between frames, matching `HealthReader` and `ObjectDetector` -- everything temporal
    belongs to §6.3's shadow state, which is also the only consumer that knows whether the hero is
    alive. Holds the `GlyphBank` because loading it is the one non-trivial cost here.
    """

    def __init__(self, bank: GlyphBank | None = None, *, max_brawlers: int = 10,
                 white_max_sat: int = 70, white_min_val: int = 175,
                 min_glyph_score: float = 0.70, min_glyph_margin: float = 0.05):
        self.bank = bank or GlyphBank.load()
        self.max_brawlers = int(max_brawlers)
        self.white_max_sat = int(white_max_sat)
        self.white_min_val = int(white_min_val)
        self.min_glyph_score = float(min_glyph_score)
        self.min_glyph_margin = float(min_glyph_margin)

    @classmethod
    def from_config(cls, cfg=None, **overrides) -> "HudReader":
        from .config import VisionConfig
        cfg = cfg or VisionConfig()
        kwargs = dict(
            max_brawlers=cfg.hud_max_brawlers,
            # The SAME white as the HP digits, deliberately sharing their keys rather than getting
            # a near-duplicate pair that can drift apart. It is one font in one colour: measured
            # here at saturation 2-5 and value 203-255, which is the middle of `(S<70, V>175)`.
            white_max_sat=cfg.hp_white_max_sat, white_min_val=cfg.hp_white_min_val,
            min_glyph_score=cfg.hud_min_glyph_score,
            min_glyph_margin=cfg.hud_min_glyph_margin,
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def brawlers_left(self, frame: np.ndarray) -> BrawlersLeft:
        """The match readout in one normalized-viewport BGR frame.

        `frame` must be a `capture.normalize_viewport` output (or a `ClipReader`/`ScreenCapture`
        frame, which are already that) -- every position here is a fraction of a 16:9 viewport,
        and a raw pillarboxed capture puts the widget somewhere else. A frame that is not one
        fails closed at the colon gate rather than reading whatever is at those coordinates.
        """
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"expected an (H, W, 3) uint8 BGR frame, got "
                             f"{frame.shape} {frame.dtype}")
        s = frame.shape[0] / REF_H

        def px(ref: float) -> int:
            return int(round(ref * s))

        y0, y1 = px(BAND_Y[0]), min(frame.shape[0], px(BAND_Y[1]))
        x1 = min(frame.shape[1], px(BAND_X))
        if y1 - y0 < px(DIGIT_H[0]) or x1 < px(COLON_X):
            return BrawlersLeft(None, NO_COLON)

        hsv = cv2.cvtColor(frame[y0:y1, 0:x1], cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 1] < self.white_max_sat)
                & (hsv[:, :, 2] > self.white_min_val)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        comps = []
        min_area = MIN_AREA * s * s
        for i in range(1, n):
            x, y, w, h, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                                stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                                stats[i, cv2.CC_STAT_AREA])
            if area >= min_area:
                comps.append((int(x), int(y), int(w), int(h), i))

        colon = self._find_colon(comps, px)
        if colon is None:
            return BrawlersLeft(None, NO_COLON)

        row = self._digit_row(comps, colon, px)
        if not row:
            return BrawlersLeft(None, NO_DIGITS)
        if len(row) > MAX_DIGITS:
            return BrawlersLeft(None, TOO_LONG, xyxy=_bbox(row, 0, y0))

        glyphs = []
        for x, y, w, h, i in row:
            digit, score, margin = self.bank.classify(normalize(labels[y:y + h, x:x + w] == i))
            glyphs.append(Glyph(digit, score, margin, (x, y + y0, x + w, y + h + y0)))
        glyphs = tuple(glyphs)
        box = _bbox(row, 0, y0)
        worst_score = min(g.score for g in glyphs)
        worst_margin = min(g.margin for g in glyphs)

        if len(glyphs) > 1 and glyphs[0].digit == 0:
            return BrawlersLeft(None, LEADING_ZERO, worst_score, worst_margin, glyphs, box)
        count = int("".join(str(g.digit) for g in glyphs))
        if not 1 <= count <= self.max_brawlers:
            return BrawlersLeft(None, OUT_OF_RANGE, worst_score, worst_margin, glyphs, box)
        if worst_score < self.min_glyph_score or worst_margin < self.min_glyph_margin:
            return BrawlersLeft(None, LOW_GLYPH, worst_score, worst_margin, glyphs, box)
        return BrawlersLeft(count, OK, worst_score, worst_margin, glyphs, box)

    def _find_colon(self, comps, px):
        """The two stacked dots of "left:", or None.

        Nearest to the measured x wins when several pairs qualify, rather than the first found:
        with a tolerance wide enough to survive a UI nudge, a bright pair of map pixels inside the
        window is possible, and "closest to where it has always been" is the tiebreak that does
        not depend on component ordering.
        """
        lo, hi = px(COLON_DOT_PX[0]), px(COLON_DOT_PX[1])
        dots = [c for c in comps if lo <= c[2] <= hi and lo <= c[3] <= hi]
        dy_lo, dy_hi = px(COLON_DY[0]), px(COLON_DY[1])
        dx_tol, want_x = px(COLON_DX_TOL), px(COLON_X)
        x_tol = px(COLON_X_TOL)
        best = None
        for top in dots:
            for bot in dots:
                if not dy_lo <= bot[1] - top[1] <= dy_hi or abs(top[0] - bot[0]) > dx_tol:
                    continue
                if abs(top[0] - want_x) > x_tol:
                    continue
                key = abs(top[0] - want_x)
                if best is None or key < best[0]:
                    best = (key, top, bot)
        return None if best is None else (best[1], best[2])

    def _digit_row(self, comps, colon, px):
        """Digit-sized components running rightward from the colon, left to right.

        Built by WALKING right from the colon rather than by collecting everything in a window,
        so a gap wide enough to be a dropped digit ends the row instead of being spanned. The row
        that comes back can be too long -- that is the caller's `TOO_LONG`, and it has to be
        reachable, because three digits in a field whose maximum is 10 means the segmentation is
        wrong and not that one component should be dropped.
        """
        top, bot = colon
        cx = max(top[0] + top[2], bot[0] + bot[2])
        h_lo, h_hi = px(DIGIT_H[0]), px(DIGIT_H[1])
        w_lo, w_hi = px(DIGIT_W[0]), px(DIGIT_W[1])
        gap_lo, gap_hi = px(COLON_GAP[0]), px(COLON_GAP[1])
        space_max = px(DIGIT_SPACE_MAX)
        top_tol, h_tol = px(ROW_TOP_TOL), px(ROW_H_TOL)

        sized = sorted((c for c in comps
                        if h_lo <= c[3] <= h_hi and w_lo <= c[2] <= w_hi and c[0] >= cx),
                       key=lambda c: c[0])
        first = next((c for c in sized if gap_lo <= c[0] - cx <= gap_hi), None)
        if first is None:
            return []
        row = [first]
        for c in sized:
            if c[0] <= row[-1][0]:
                continue
            prev = row[-1]
            if (c[0] - (prev[0] + prev[2]) <= space_max
                    and abs(c[1] - first[1]) <= top_tol and abs(c[3] - first[3]) <= h_tol):
                row.append(c)
        return row


def _bbox(comps, x_off: int, y_off: int) -> tuple[int, int, int, int]:
    return (min(c[0] for c in comps) + x_off, min(c[1] for c in comps) + y_off,
            max(c[0] + c[2] for c in comps) + x_off, max(c[1] + c[3] for c in comps) + y_off)
