"""The Super's charge off the Super button. See brawl_deployment/perception/super_button.py.

Synthetic buttons drawn from the HSV the game was MEASURED to paint, plus footage checks on the
BlueStacks clips (`@pytest.mark.vision`). Each synthetic test is one of the looks the module
docstring lists, or a way the reader was seen to go wrong while it was built:

  - the match intro and death screens are flat salmon and read as a READY super on hue alone
  - a hit flash washes the button pale, and a hue test with a saturation floor survives it
  - an uncharged super must read as zero, not as "could not read" -- opposite things to the shadow
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from brawl_deployment.perception.super_button import SuperReading, read_super

C, R = 100.0, 34.7                 # the button's centre and radius in the synthetic frame
BUTTON = (C, C, R)


def _bgr(h, s, v):
    """A BGR pixel from the HSV it was measured in, so the thresholds and the fixtures are written
    in the same units and a transcription error cannot hide between them."""
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


# MEASURED on the BlueStacks clips (module docstring), each taken inside its band, not on an edge.
BLUE = (105, 157, 215)             # charging disc: hue 104-107, sat 144-170
NAVY = (116, 150, 55)              # the ring the arc grows in: hue 114-118, val 52-58
ARC = (27, 200, 225)               # the arc: hue 20-33, val 200-250
READY = (22, 205, 245)             # ready disc: hue median 21-23, sat p10 196-205
SALMON = (9, 126, 230)             # intro / death screen: hue 7-11, sat 116-136
FLASH_BLUE = (100, 80, 245)        # a hit flash over the charging disc: sat 44-115
FLASH_ARC = (36, 50, 250)          # ... and over the arc: hue 29-43, sat 23-81
FLASH_READY = (26, 85, 250)        # ... and over a ready disc: hue 21-31, sat 71-96
MAP = (60, 120, 120)               # grass behind the button
SKULL = (0, 0, 240)                # the white icon in the disc's middle


def synth(*, disc=BLUE, disc_r=1.0, ring=NAVY, arc=0.0, arc_hsv=ARC, size=200):
    """A Super button: `disc` out to `disc_r`, the ring, and an arc covering `arc` of the turn."""
    frame = np.full((size, size, 3), _bgr(*MAP), np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    dx, dy = xx - C, yy - C
    rr = np.hypot(dx, dy) / R
    if ring is not None:
        frame[(rr >= 1.0) & (rr <= 1.35)] = _bgr(*ring)
    if disc is not None:
        frame[rr <= disc_r] = _bgr(*disc)
        frame[rr <= 0.5] = _bgr(*SKULL)
    if arc > 0:
        turn = np.degrees(np.arctan2(-dx, dy)) % 360.0
        frame[(rr >= 1.05) & (rr <= 1.33) & (turn < 360.0 * arc)] = _bgr(*arc_hsv)
    return frame


# ---------------------------------------------------------------------------------------------
# The three looks
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("want", [0.1, 0.25, 0.5, 0.75, 0.95])
def test_the_arc_reads_back_the_charge_it_was_drawn_with(want):
    got = read_super(synth(arc=want), BUTTON)
    assert got.state == "charging" and not got.ready
    assert got.charge == pytest.approx(want, abs=0.03)


def test_an_uncharged_super_is_zero_not_unreadable():
    """Opposite things to the shadow: None holds the last reading, zero says the super is gone."""
    assert read_super(synth(), BUTTON) == SuperReading(0.0, False, "empty")


def test_a_ready_disc_reads_ready_whatever_the_ring_shows():
    """The ready disc swells past the ring and hides the arc, so ready is the disc's colour and
    never `charge >= 1`."""
    got = read_super(synth(disc=READY, disc_r=1.15), BUTTON)
    assert got == SuperReading(1.0, True, "ready")


def test_ready_is_the_disc_colour_not_a_full_arc():
    got = read_super(synth(arc=1.0), BUTTON)
    assert not got.ready and got.state == "charging" and got.charge == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------------------------
# Screens that are not the button
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("salmon", [SALMON, (11, 136, 230), (7, 116, 230)])
def test_a_salmon_intro_or_death_screen_is_unreadable_not_ready(salmon):
    """MEASURED: bluestacks-example3.mp4's intro and death screen paint the whole button area
    hue 7-11, sat 116-136, and read as a ready super until the ready test asked for saturation."""
    frame = np.full((200, 200, 3), _bgr(*salmon), np.uint8)
    assert read_super(frame, BUTTON) is None


def test_the_knob_dragged_off_its_centre_is_unreadable():
    """While a super is aimed the disc follows the finger and the map shows through."""
    assert read_super(synth(disc=None), BUTTON) is None


def test_a_button_off_the_frame_is_unreadable():
    assert read_super(synth(arc=0.5), (20.0, C, R)) is None


# ---------------------------------------------------------------------------------------------
# The hit flash
# ---------------------------------------------------------------------------------------------

def test_a_flash_over_a_charging_button_reads_the_state_under_it():
    got = read_super(synth(disc=FLASH_BLUE, ring=(116, 60, 160), arc=0.4, arc_hsv=FLASH_ARC),
                     BUTTON)
    assert got.state == "charging" and got.charge == pytest.approx(0.4, abs=0.03)


def test_a_flash_over_a_ready_button_is_unreadable_not_charging():
    """The price of the salmon fix, documented in the module: 1-2 frames of None, which the shadow
    holds through. What must never happen is a ready super reading as not ready."""
    assert read_super(synth(disc=FLASH_READY, disc_r=1.15), BUTTON) is None


@pytest.mark.parametrize("off", [(2.0, -2.0), (-3.0, 0.0), (0.0, 3.0)])
def test_a_centre_a_few_pixels_off_still_reads(off):
    """BRAWL_DEPLOYMENT_DESIGN.md 4.5: calibrated centres transfer to 1-3 px."""
    button = (C + off[0], C + off[1], R)
    assert read_super(synth(arc=0.5), button).charge == pytest.approx(0.5, abs=0.06)
    assert read_super(synth(disc=READY, disc_r=1.15), button).ready


# ---------------------------------------------------------------------------------------------
# Footage
# ---------------------------------------------------------------------------------------------

FIXTURES = Path("tests/fixtures/vision")


def _trace(name):
    """`(t, reading)` for every frame of a fixture clip, at the calibrated Super button."""
    from brawl_deployment.capture import to_viewport
    from brawl_deployment.match_state import Calibration
    from brawl_vision.clips import ClipReader

    clip = FIXTURES / name
    if not clip.exists():
        pytest.skip(f"{clip} not available")
    button = Calibration.load().viewport_button("super")
    with ClipReader(clip, trim_tail=False) as frames:
        return [(f.t, read_super(to_viewport(f.image), button)) for f in frames]


@pytest.mark.vision
def test_bulls_super_reads_ready_for_as_long_as_the_button_shows_it():
    """bluestacks-bull-hypercharge.mp4, MEASURED frame by frame: empty to 15.8 s, charging to
    21.8 s, READY from 21.9 s until the super is spent at 79.8 s. The old hero-bar reader read
    None on every frame of this clip, so the live super was masked all match."""
    trace = _trace("bluestacks-bull-hypercharge.mp4")

    early = [r for t, r in trace if 7.0 <= t <= 15.5 and r is not None]
    assert early and all(r.state == "empty" for r in early)

    charging = [r for t, r in trace if 16.0 <= t <= 21.7 and r is not None]
    assert not any(r.ready for r in charging)
    assert max(r.charge for r in charging) >= 0.9

    held = [r for t, r in trace if 23.0 <= t <= 79.5]
    assert not any(r is not None and not r.ready for r in held), "a ready super read as not ready"
    assert sum(r is None for r in held) <= 0.05 * len(held), "too many flash frames unreadable"

    spent = [r for t, r in trace if 79.5 <= t <= 81.5 and r is not None]
    assert any(not r.ready for r in spent), "the spend at 79.8 s never read"


@pytest.mark.vision
@pytest.mark.parametrize("name", ["bluestacks-example-new.mp4", "bluestacks-example-zone.mp4",
                                  "bluestacks-example3.mp4"])
def test_the_mortis_clips_never_read_a_ready_super(name):
    """None of these matches charged the super. The long-dash meter the old reader mistook for it
    went magenta every ~4.5 s on example-new; example3's intro and death screen are salmon."""
    trace = _trace(name)
    reads = [r for _, r in trace if r is not None]
    assert len(reads) >= 0.5 * len(trace), f"only {len(reads)} of {len(trace)} frames read"
    assert not any(r.ready for r in reads)
