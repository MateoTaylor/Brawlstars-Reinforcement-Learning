"""One detection box -> one HP number and a confidence in it. The seam of this chunk.

`locate.py` finds shapes, `glyphs.py` names them, and this is the only module that turns those
into an integer -- and the only one that decides how much to believe it.

#### Confidence is the product, not the number

The policy this feeds can absorb a noisy HP. What it cannot absorb is a CONFIDENT WRONG one, and
this pipeline's natural failure is exactly that. When the game floats a heal popup over a
brawler, the popup covers all but one digit of `8400`; the digit left standing is a clean,
perfectly-formed `8` that the classifier scores 0.928 with a 0.181 margin. Per-glyph confidence
is high, and the read is off by a factor of a thousand. Any design that scores only the glyphs
ships that error at full confidence.

So the gates below are mostly STRUCTURAL -- they ask whether the row of digits is intact, not
whether each digit is pretty. Three are fatal and the rest grade.

**Fatal, because the reading is not merely uncertain but meaningless:**

* `occluded` -- a popup glyph (white, same font, >= 21 px tall) overlaps the digit row. Measured
  at 26 of 1000 sampled boxes on `day10_gameplay.mp4`.
* `gap` -- two consecutive glyphs sit more than `locate.MAX_GLYPH_SPACE` apart. Digits of one
  number are 1-4 px apart (measured, n=893) and the narrowest glyph is 6 px wide, so a hole
  bigger than that is a digit that did not survive the white threshold. Note this catches TWO
  adjacent missing digits as readily as one, which the ratio test it replaced did not -- see
  `DigitRow.widest_space` for the `10400` -> `100` read at confidence 1.00 that found the bug.
* `leading-zero` -- a multi-digit read starting in `0`. The game never zero-pads, so this is
  proof a leading digit was lost rather than a number that starts with zero.

**Graded, via `_ramp`, into the confidence:**

* glyph score and glyph margin, both taken as the MINIMUM over the digits -- a number is only as
  trustworthy as its worst character.
* centring, in units of digit PITCH rather than pixels. The HP number is drawn centred on the
  brawler, so a row missing an end digit sits half a pitch off. This is a HINT and not a gate:
  measured, an intact row is already 0.14 pitches off the detector box's centre at the median and
  0.38 at p75, because the box itself breathes asymmetrically with auras -- so the honest spread
  overlaps the 0.5 that a truncation produces. `smooth._is_truncation` is what actually catches
  those, by recognising the shape of the error rather than trying to threshold through an
  overlap.

**Deliberately NOT fatal: a missing bar.** Early versions rejected a box whose health bar could
not be found, which sounds safe and is not -- the filled bar shrinks with health, so the boxes it
discards are precisely the nearly-dead enemies whose HP the policy most needs. A missing bar is a
mild confidence penalty here, nothing more.

#### Why no fill ratio

A fraction-of-max-health would be a genuinely useful second signal and it is not shipped, because
measuring it honestly costs more than it is worth right now. The coloured run is only the FILLED
part of the bar; the depleted remainder is a pale lavender that does not separate cleanly from
white text or from a pale sprite, and during the white damage-flash the whole bar washes out. So
the denominator is the hard part, and a ratio computed against the filled run alone would be the
constant 1.0 dressed up as a measurement. `Bar.length` is reported raw for anyone who wants to
build on it; `HealthReading` claims no ratio. This is the same discipline `configs/vision.yaml`
applies to its own unmeasured values -- an unmeasured number that looks like a finding is worse
than an absent one.

#### What a CNN would buy

The glyph classifier is a nearest-template NCC (see `glyphs.py`). Every failure catalogued above
is a SEGMENTATION or OCCLUSION failure, not a classification one -- the glyphs that reach the
classifier are read correctly almost every time, and the ones that are not are fragments no
classifier would recognise either. A trained model would replace the part that is already
working. The leverage is in this module and in `smooth.py`, which is where the effort went.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from . import locate
from .glyphs import Glyph, GlyphBank, normalize

# Confidence ramps: value at which a component scores 0, and value at which it scores 1.
#
# CALIBRATED against the measured distribution of accepted reads on `day10_gameplay.mp4`, not
# chosen for roundness, and the first cut of these was wrong in an instructive way. They were set
# from the glyph bank's own clustering statistics (median NCC 0.949), but that is a glyph scored
# against a cluster it helped define. Scored against the ten FINAL templates, and taking the
# minimum over a 4-5 digit number rather than a single glyph, the same reads run::
#
#     min glyph score   p05 0.698   p25 0.832   med 0.874   p75 0.902
#     min glyph margin  p05 0.106   p25 0.152   med 0.171   p75 0.189
#
# A ramp topping out at 0.90 therefore marked the MEDIAN honest read as uncertain, and 96 of 277
# accepted reads fell below 0.5 confidence for no better reason than being ordinary. A confidence
# that is low on the typical case is not a confidence, it is an offset -- so these ends sit at the
# good population's own median and its p05, which is what makes the number mean something.
SCORE_RAMP = (0.62, 0.85)
MARGIN_RAMP = (0.03, 0.13)
# Centring, in digit pitches, ramping DOWNWARD -- 1.0 at or below the first, 0.0 at or above the
# second.
#
# **These two populations overlap, and pretending otherwise was the other half of the same
# mistake.** An intact row's offset from the box centre runs med 0.14, p75 0.38 pitches -- not the
# ~0.1 a first pass suggested -- because the reference is the DETECTOR's box, and that box
# breathes asymmetrically as auras come and go (measured 264-380 px of height within one clip).
# Losing an end digit displaces the centre by 0.5 pitch, which sits inside the honest spread.
#
# So centring is a graded hint here and NOT the defence against a truncated read. The defence is
# `smooth._is_truncation`, which recognises the specific shape of the error instead of trying to
# separate two overlapping distributions. Tightening these numbers to make centring do that job
# is the thing not to do: it rejects sound reads at a rate far above the errors it catches.
CENTRE_RAMP = (0.35, 0.70)
# What a box with no findable health bar keeps. Not zero: see the module docstring.
NO_BAR_CONFIDENCE = 0.75

# The game's HP never reaches six figures, so a longer row is a mis-segmentation by definition.
MAX_DIGITS = 5

OK = "ok"
NO_DIGITS = "no-digits"
OCCLUDED = "occluded"
GAP = "gap"
LEADING_ZERO = "leading-zero"
TOO_LONG = "too-long"


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at or below `lo`, 1 at or above `hi`, linear between. `lo > hi` ramps downward."""
    if lo == hi:
        return 1.0 if x >= hi else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


@dataclass(frozen=True)
class HealthReading:
    """What one box's health readout says, and how much of it to believe.

    `hp` is None exactly when `status` is fatal. Otherwise it is the number as read, WITH its
    confidence, rather than being suppressed below some threshold -- the caller knows what it can
    tolerate and `trusted` is offered as the conventional answer, not enforced as the only one.
    """
    hp: int | None
    confidence: float
    status: str
    detection: object
    digits: tuple[Glyph, ...] = ()
    row_xyxy: tuple[int, int, int, int] | None = None
    bar_xyxy: tuple[int, int, int, int] | None = None
    bar_px: int | None = None
    centre_offset: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == OK

    def trusted(self, minimum: float) -> bool:
        return self.hp is not None and self.confidence >= minimum


class HealthReader:
    """Reads HP out of detection boxes. Construct once, call `read_all` per frame.

    Stateless between frames, matching `ObjectDetector`. Everything temporal -- and the
    consistency the policy actually wants -- lives in `smooth.HealthTracker`, which wraps this.
    """

    def __init__(self, bank: GlyphBank | None = None, *, bar_hues=None,
                 white_max_sat: int = 70, white_min_val: int = 175,
                 glyph_min_height: int = 13, glyph_max_height: int = 19,
                 crop_height_frac: float = 0.55, crop_pad_frac: float = 0.12,
                 min_glyph_score: float = 0.70, min_glyph_margin: float = 0.03):
        self.bank = bank or GlyphBank.load()
        # Enemy red is (166, 5) and WRAPS past 179. Writing it (166, 180) is wrong twice -- 180 is
        # not an OpenCV hue, and with low <= high the band does not wrap either, so it silently
        # matches nothing at the red end it exists for. Kept in step with `VisionConfig` by
        # `test_the_constructor_defaults_match_the_shipped_config`.
        self.bar_hues = dict(bar_hues or {"player": (46, 66), "enemy": (166, 5)})
        self.white_max_sat = int(white_max_sat)
        self.white_min_val = int(white_min_val)
        self.glyph_min_height = int(glyph_min_height)
        self.glyph_max_height = int(glyph_max_height)
        self.crop_height_frac = float(crop_height_frac)
        self.crop_pad_frac = float(crop_pad_frac)
        self.score_ramp = (float(min_glyph_score), SCORE_RAMP[1])
        self.margin_ramp = (float(min_glyph_margin), MARGIN_RAMP[1])

    @classmethod
    def from_config(cls, cfg=None, **overrides) -> "HealthReader":
        from ...config import VisionConfig
        cfg = cfg or VisionConfig()
        kwargs = dict(
            bar_hues={"player": tuple(cfg.hp_bar_hue_player),
                      "enemy": tuple(cfg.hp_bar_hue_enemy)},
            white_max_sat=cfg.hp_white_max_sat, white_min_val=cfg.hp_white_min_val,
            glyph_min_height=cfg.hp_glyph_min_height, glyph_max_height=cfg.hp_glyph_max_height,
            crop_height_frac=cfg.hp_crop_height_frac, crop_pad_frac=cfg.hp_crop_pad_frac,
            min_glyph_score=cfg.hp_min_glyph_score, min_glyph_margin=cfg.hp_min_glyph_margin,
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def read_all(self, image: np.ndarray, detections) -> list[HealthReading]:
        """Every detection's readout, one per input, in order.

        A class with no configured bar hue (`teammate`, which Solo Showdown has none of) is still
        READ -- its digits are white like anyone's -- it just gets no bar and so carries the
        `NO_BAR_CONFIDENCE` penalty. Skipping those outright would silently drop a brawler in any
        mode this is later pointed at.
        """
        return [self.read(image, det) for det in detections]

    def read(self, image: np.ndarray, det) -> HealthReading:
        """`det`'s HP readout, from the raw BGR frame it was detected in."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"expected an (H, W, 3) uint8 BGR frame, got "
                             f"{image.shape} {image.dtype}")
        x0, y0, x1, y1, box_cx = locate.crop_for(det, image.shape, self.crop_height_frac,
                                                 self.crop_pad_frac)
        if x1 - x0 < 16 or y1 - y0 < self.glyph_max_height:
            return HealthReading(None, 0.0, NO_DIGITS, det)

        hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        row = locate.find_digit_row(hsv, box_cx, max_sat=self.white_max_sat,
                                    min_val=self.white_min_val,
                                    min_h=self.glyph_min_height, max_h=self.glyph_max_height)
        if row is None:
            return HealthReading(None, 0.0, NO_DIGITS, det)

        offset_pitches = abs(row.centre_x - box_cx) / max(row.pitch, 1e-6)
        rx0, ry0, rx1, ry1 = row.xyxy
        frame_row = (rx0 + x0, ry0 + y0, rx1 + x0, ry1 + y0)

        if row.popup_overlap:
            return HealthReading(None, 0.0, OCCLUDED, det, row_xyxy=frame_row,
                                 centre_offset=offset_pitches)
        if row.widest_space > locate.MAX_GLYPH_SPACE:
            return HealthReading(None, 0.0, GAP, det, row_xyxy=frame_row,
                                 centre_offset=offset_pitches)
        if len(row.blobs) > MAX_DIGITS:
            return HealthReading(None, 0.0, TOO_LONG, det, row_xyxy=frame_row,
                                 centre_offset=offset_pitches)

        glyphs = []
        for blob in row.blobs:
            digit, score, margin = self.bank.classify(normalize(blob.mask))
            gx0, gy0, gx1, gy1 = blob.xyxy
            glyphs.append(Glyph(digit, score, margin,
                                (gx0 + x0, gy0 + y0, gx1 + x0, gy1 + y0)))
        glyphs = tuple(glyphs)

        if len(glyphs) > 1 and glyphs[0].digit == 0:
            return HealthReading(None, 0.0, LEADING_ZERO, det, digits=glyphs,
                                 row_xyxy=frame_row, centre_offset=offset_pitches)

        bar = locate.find_bar(hsv, self.bar_hues[det.label], row.bottom) \
            if det.label in self.bar_hues else None

        # The weakest link, not an average. Each component is a separate way the read can be
        # wrong, so a strong showing on two of them must not paper over a collapse in the third.
        confidence = min(
            _ramp(min(g.score for g in glyphs), *self.score_ramp),
            _ramp(min(g.margin for g in glyphs), *self.margin_ramp),
            _ramp(offset_pitches, CENTRE_RAMP[1], CENTRE_RAMP[0]),
            1.0 if bar is not None else NO_BAR_CONFIDENCE,
        )
        hp = int("".join(str(g.digit) for g in glyphs))
        frame_bar = None
        if bar is not None:
            bx0, by0, bx1, by1 = bar.xyxy
            frame_bar = (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0)
        return HealthReading(hp, confidence, OK, det, digits=glyphs, row_xyxy=frame_row,
                             bar_xyxy=frame_bar, bar_px=bar.length if bar else None,
                             centre_offset=offset_pitches)
