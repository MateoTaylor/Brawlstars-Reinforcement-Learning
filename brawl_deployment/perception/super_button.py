"""The Super's charge, read off the Super BUTTON. BRAWL_DEPLOYMENT_DESIGN.md 6.4.

**Why the button and not the hero's readout stack.** Until 2026-11-02 this was read from a third bar
under the hero's ammo pips (`hero_bars.read_super`, now gone). That bar is Mortis's LONG-DASH meter,
not his super: on bluestacks-example-new.mp4 it empties on every ammo drop, refills in ~4.5 s and
turns magenta when full, while the Super button sits uncharged for the whole clip. Bull has no long
dash and no such bar, so the reader returned None on every frame and the shadow held the super
illegal all match -- on bluestacks-bull-hypercharge.mp4 the button sat READY from t=22 s to t=79 s,
and the live telemetry of 2026-11-01 offered the super on 0 of 155 decisions.

The button is the game's own readout of the super, drawn the same way for every brawler, and it
needs no hero detection. Its centre is `control_calibration.json`'s `super`, the disc the controls
already press.

Three looks, MEASURED on the four BlueStacks clips (OpenCV HSV, hue 0-179):

  charging  a blue disc (hue 104-107, sat 144-170) inside a dark navy ring (hue 114-118, val
            52-58). A yellow arc (hue 20-33, val 200-250) grows in the ring, clockwise from
            6 o'clock, as the charge does.
  ready     the disc turns yellow-orange (hue 11-31, sat 181-220) and swells past the ring.
  flash     a hit that charges the super washes the whole button pale for a frame or two. Hue
            survives it -- a blue disc reads 91-107, the arc 29-43 -- but saturation drops to
            44-115 (disc) and 23-81 (arc). So the blue and arc tests are on HUE with a low
            saturation floor, and a flash over a charging button reads as the state under it.

The READY test is the one exception, because a flat salmon screen passes a hue-only test for it:
the match intro and a death screen paint the whole button area hue 7-11, sat 116-136, and read as
a ready super. The ready disc's own hue sits higher (median 21-23) and its saturation far higher
(p10 196-205), so it is tested on both. The price is a ready disc under a flash (sat 71-96), which
reads None for its 1-2 frames; the shadow holds the last reading through that.

Anything else -- the knob dragged off its centre while a super is aimed, a death or results screen
-- matches neither disc and reads None, which the shadow treats as "could not read".
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Where to look, in units of the calibrated button radius. The disc band skips the skull icon
# (out to ~0.55 r) and stops short of the disc's own edge, so a centre 1-3 px off (BRAWL_DEPLOYMENT_
# DESIGN.md 4.5: centres transfer to that accuracy) still sees disc. The ring band is the navy track
# the arc is painted in: MEASURED, the arc's coverage peaks between 1.10 and 1.20 r.
DISC_BAND = (0.6, 0.95)
RING_BAND = (1.08, 1.30)

BLUE_HUE = (85, 130)        # the charging disc, flash included (91-107)
YELLOW_HUE = (15, 45)       # the ready disc (median 21-23) and the arc (20-43); salmon is 7-11
MIN_SAT = 20                # below this hue is noise; the palest flash pixel measured is 23
READY_MIN_SAT = 150         # the ready disc's p10 is 196; a salmon screen's p90 is 136
ARC_MIN_VAL = 150           # lit arc 189-255, the navy track behind it 50-63

# A disc look needs this share of the disc band. Neither look reaches it while the knob is dragged.
DISC_SHARE = 0.6

# The arc is counted in angular bins, each lit when this share of its ring pixels is. A count
# rather than a run from 6 o'clock: the arc is contiguous, so the two agree, and a count does not
# lose the whole charge to one occluded bin.
ARC_BINS = 72
ARC_BIN_SHARE = 0.4


@dataclass(frozen=True)
class SuperReading:
    """One frame's super, in the units `brawl_sim`'s obs schema asks for."""
    charge: float                # 0..1 -- obs_schema hero.super_charge_frac
    ready: bool                  # obs_schema hero.super_ready -- the disc's colour, not charge >= 1
    state: str                   # "empty" | "charging" | "ready"


def _hue_in(h: np.ndarray, band: tuple[int, int]) -> np.ndarray:
    return (h >= band[0]) & (h <= band[1])


def read_super(frame: np.ndarray, button: tuple[float, float, float]) -> SuperReading | None:
    """The super's charge off the Super button, or None if the button does not show it.

    `button` is `(cx, cy, r)` in viewport pixels -- `Calibration.viewport_button("super")`.
    """
    cx, cy, r = button
    half = int(np.ceil(RING_BAND[1] * r)) + 1
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + 2 * half + 1, y0 + 2 * half + 1
    if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
        return None
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    dy, dx = np.mgrid[y0:y1, x0:x1]
    dx, dy = dx - cx, dy - cy
    rr = np.hypot(dx, dy) / r

    disc = (rr >= DISC_BAND[0]) & (rr <= DISC_BAND[1])
    yellow = _hue_in(h, YELLOW_HUE) & (s >= MIN_SAT)
    n_disc = int(disc.sum())
    if int((yellow & (s >= READY_MIN_SAT) & disc).sum()) >= DISC_SHARE * n_disc:
        return SuperReading(1.0, True, "ready")
    if int((_hue_in(h, BLUE_HUE) & (s >= MIN_SAT) & disc).sum()) < DISC_SHARE * n_disc:
        return None

    ring = (rr >= RING_BAND[0]) & (rr <= RING_BAND[1])
    lit = (yellow & (v >= ARC_MIN_VAL))[ring]
    # 0 at 6 o'clock, increasing clockwise on screen (y down): the direction the arc grows in.
    angle = np.degrees(np.arctan2(-dx[ring], dy[ring])) % 360.0
    bins = np.minimum((angle * ARC_BINS / 360.0).astype(int), ARC_BINS - 1)
    share = (np.bincount(bins, weights=lit, minlength=ARC_BINS)
             / np.maximum(np.bincount(bins, minlength=ARC_BINS), 1))
    charge = float((share >= ARC_BIN_SHARE).mean())
    return SuperReading(charge, False, "charging" if charge > 0 else "empty")
