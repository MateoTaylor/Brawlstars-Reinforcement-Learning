"""The "Brawlers left: N" readout. See brawl_vision/hud.py.

**Synthetic apart from one marked test, but unlike `test_vision_hero_bars.py` these tests do need
`data/digits.npz`** -- this reader classifies glyphs, so a count drawn out of plain rectangles
would test nothing about it. The digits here are therefore drawn FROM THE SHIPPED TEMPLATES,
upscaled from the 18x14 canvas to the 41 px the HUD renders at, which is the honest synth: it is
the same font at the same size ratio the real widget presents, and it exercises the exact path
that was in question -- an 18 px template, blown up 2.3x, measured, and shrunk back to 18 px by
`glyphs.normalize`. The templates are committed, so this still runs on a fresh clone with no
footage and no GPU.

**Two frame sizes, and that is the point of half these tests.** Every constant in `hud.py` is a
reference pixel at a 1126 px viewport -- what `DeployCapture` resizes to -- scaled at read time.
The fixtures carry a second height for free: `ClipReader` never resizes, so
`bluestacks-example-new.mp4` arrives at its native 1920x1080. A reader that quietly hardcoded
either would pass on half the footage and be wrong on the other half, in a way no single-clip
test can see. `synth_hud(height=...)` is how that is pinned.

**What is asserted is the failure modes**, following `test_vision_hp.py` and
`test_vision_hero_bars.py`. Each of these is a thing that was real, or a gap the measurement
could not close:

  - the LABEL's own cap-height letters are 39-41 px, exactly digit-sized, and only the colon
    separates them from the count
  - a phantom 26x18 "digit" in 40 of 163 frames, from a different white HUD element 23 px below
    the band's original bottom edge
  - a screen-space band admits white map pixels by construction, and a 71x34 blob from the map
    classified as a `4` at 0.262 before the structural gates existed
  - the off-by-one between "brawlers left" (includes you) and `meta.n_enemies_alive` (does not)
  - a widget with a LONGER label puts its colon somewhere else. The three training-cave clips
    produce zero reads over 284 frames, but their band is bare map, so that measures the reader
    staying quiet off-mode and NOT that it can tell one labelled readout from another. The
    shifted-colon test below is what covers that, because footage of it does not exist here.
"""
import cv2
import numpy as np
import pytest

from brawl_vision.config import VisionConfig, load_vision_config, validate
from brawl_vision.hud import (
    BAND_Y, COLON_DY, COLON_X, COLON_X_TOL, DIGIT_H, LEADING_ZERO, LOW_GLYPH, NO_COLON,
    NO_DIGITS, OK, OUT_OF_RANGE, REF_H, TOO_LONG, BrawlersLeft, HudReader,
)
from brawl_vision.object_detection.hp_detection.glyphs import GlyphBank

CLIP = "tests/fixtures/vision/bluestacks-example-new.mp4"

WHITE_BGR = (250, 250, 250)
# The plate the readout is drawn on: measured saturation ~127, value ~65. Dark and saturated, so
# it is nowhere near the (S<70, V>175) white the glyphs are found by.
PLATE_BGR = (95, 55, 40)

# Measured on real footage at the 1126 px viewport, and the numbers `hud.py`'s constants come
# from. Kept here as literals rather than imported so a constant that drifts breaks a test
# instead of silently redefining what the tests mean.
COLON_DOT = 11
COLON_TOP_Y = 50
DIGIT_TOP_Y = 39
DIGIT_HEIGHT = 41
COLON_TO_DIGIT = 15
DIGIT_SPACE = 6

_BANK = GlyphBank.load()


def glyph_bitmap(digit: int, height: int) -> np.ndarray:
    """One digit as a bool bitmap `height` px tall, at the template's own aspect.

    Drawn from `data/digits.npz`, which is what makes this a test of the SCALE path rather than of
    a font nobody ships: the real HUD digits measure 19-33 x 40-42, and the natural widths these
    templates produce at height 41 are 18-32.
    """
    t = (_BANK.templates[digit] > 0.5).astype(np.uint8)
    rows, cols = np.where(t.any(1))[0], np.where(t.any(0))[0]
    tight = t[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    width = max(1, round(tight.shape[1] * height / tight.shape[0]))
    return cv2.resize(tight.astype(np.float32), (width, height),
                      interpolation=cv2.INTER_NEAREST) > 0.5


def synth_hud(count=9, *, height=REF_H, colon_x=COLON_X, shift=0, gap=COLON_TO_DIGIT,
              spacing=DIGIT_SPACE, digit_h=DIGIT_HEIGHT, label=True, colon=True,
              digits=None, blocks=()):
    """A normalized-viewport frame carrying the match readout, laid out as measured.

    Every argument is in REFERENCE pixels at 1126 and scaled by `height / 1126`, matching the
    module under test, so a test says "shift the colon 60 px right" once and it means the same
    thing at both resolutions.

    `shift` moves the WHOLE widget -- label, colon and count together -- which is what a UI
    revision does. `colon_x` moves only the colon and the count, which is what a LONGER LABEL
    does. Two different things, and the reader is meant to survive the first and refuse the
    second, so they are separate arguments.

    `digits` overrides the drawn glyphs (a sequence of ints), which is how a three-digit row or a
    leading zero gets drawn without `count` having to be able to express one. `blocks` are extra
    white `(x, y, w, h)` rectangles -- the map showing through, which is the thing a screen-space
    band cannot avoid admitting.
    """
    s = height / REF_H
    w = int(round(height * 16 / 9))

    def p(ref):
        return int(round(ref * s))

    frame = np.full((height, w, 3), 30, np.uint8)
    frame[p(20):p(104), p(30 + shift):p(470 + shift)] = PLATE_BGR
    colon_x += shift

    if label:
        # "Brawlers left" -- cap-height letters at the digits' own 41 px and x-height ones at 29,
        # spanning x 44-346 as measured. These are the decoys: nothing about their SIZE says they
        # are not the count.
        for x, glyph_h in ((44, 41), (78, 29), (108, 29), (140, 41), (172, 29), (204, 29),
                           (238, 41), (268, 41), (296, 29), (326, 29)):
            top = DIGIT_TOP_Y + (41 - glyph_h)
            frame[p(top):p(top + glyph_h), p(x + shift):p(x + shift + 22)] = WHITE_BGR

    if colon:
        for top in (COLON_TOP_Y, COLON_TOP_Y + COLON_DY[1] // 2 + 3):
            frame[p(top):p(top + COLON_DOT), p(colon_x):p(colon_x + COLON_DOT)] = WHITE_BGR

    if digits is None:
        digits = [int(c) for c in str(count)]
    x = colon_x + COLON_DOT + gap
    for d in digits:
        bitmap = glyph_bitmap(d, p(digit_h))
        gh, gw = bitmap.shape
        y0, x0 = p(DIGIT_TOP_Y), p(x)
        frame[y0:y0 + gh, x0:x0 + gw][bitmap] = WHITE_BGR
        x += round(gw / s) + spacing

    for bx, by, bw, bh in blocks:
        frame[p(by):p(by + bh), p(bx):p(bx + bw)] = WHITE_BGR
    return frame


@pytest.fixture(scope="module")
def reader():
    return HudReader.from_config()


# ---------------------------------------------------------------- the synth reads back

def test_the_synth_reads_back_every_count_the_field_can_take(reader):
    """If this drifts, every other test here is measuring the wrong thing."""
    for want in range(1, 11):
        got = reader.brawlers_left(synth_hud(want))
        assert got.ok, f"count {want} came back {got.status}"
        assert got.count == want


def test_the_templates_carry_across_the_scale_they_were_not_harvested_at(reader):
    """THE question this reader opened: `glyphs.py` harvested at 15-17 px, the HUD renders at
    40-42. Measured on footage the count digits score a median 0.904 against those templates --
    better than the HP population's own 0.874 -- and the synth agrees."""
    got = reader.brawlers_left(synth_hud(8))
    assert got.ok and got.count == 8
    assert got.score > 0.85 and got.margin > 0.10


def test_ten_is_read_as_one_number_and_not_as_a_one(reader):
    """The only two-digit value Solo Showdown produces, and it is the opening state of every
    match, so a reader that stops at the first glyph is wrong from the first frame."""
    got = reader.brawlers_left(synth_hud(10))
    assert got.ok and got.count == 10
    assert len(got.digits) == 2


# ---------------------------------------------------------------- localization

def test_the_labels_own_letters_are_not_read_as_the_count(reader):
    """"Brawlers left" is the same font in the same white a few pixels away, and its cap-height
    letters are 41 px -- exactly a digit. Size cannot separate them; the colon can."""
    got = reader.brawlers_left(synth_hud(7))
    assert got.ok and got.count == 7
    assert len(got.digits) == 1, "a label glyph joined the count"
    assert got.xyxy[0] > 0.17 * 2002, "the read is somewhere in the label, not after the colon"


def test_no_colon_means_no_counter_rather_than_a_guess(reader):
    """The counter is absent during the match-start fly-in and on the loading screen -- measured,
    frames 0-225 of the BlueStacks clip -- and nothing else about those frames says so, because
    the band still contains white map pixels."""
    got = reader.brawlers_left(synth_hud(9, colon=False))
    assert got.count is None and got.status == NO_COLON


def test_a_widget_whose_label_is_longer_is_not_this_widget(reader):
    """The gap the footage cannot close. A different labelled readout at the same corner would be
    read as a brawler count if position were not gated -- and the colon's x IS the label's
    rendered width, which is why gating it is enough. Shifted by 60 reference px, about four
    characters of this font."""
    got = reader.brawlers_left(synth_hud(9, colon_x=COLON_X + 60))
    assert got.count is None and got.status == NO_COLON


def test_the_colon_window_admits_the_measured_spread(reader):
    """Measured colon x: 366-367 at 1126, and 350 at 1080 which scales to 364.9. The window has
    to hold that with room for a UI nudge -- the whole widget sliding, which is `shift` -- and
    the previous test is what stops it holding a different widget instead."""
    for dx in (-COLON_X_TOL + 2, 0, COLON_X_TOL - 2):
        got = reader.brawlers_left(synth_hud(6, shift=dx))
        assert got.ok and got.count == 6, f"shift={dx} came back {got.status}"


def test_the_colon_nearest_its_measured_home_wins(reader):
    """With a window wide enough to survive a UI nudge, a pair of white map pixels inside it is
    possible. Nearest-to-measured is the tiebreak because it does not depend on the order
    `connectedComponentsWithStats` happens to return components in."""
    decoy = ((COLON_X - 20, COLON_TOP_Y, COLON_DOT, COLON_DOT),
             (COLON_X - 20, COLON_TOP_Y + COLON_DY[1] // 2 + 3, COLON_DOT, COLON_DOT))
    got = reader.brawlers_left(synth_hud(5, blocks=decoy))
    assert got.ok and got.count == 5


def test_white_map_pixels_further_right_do_not_join_the_count(reader):
    """A screen-space band admits the world by construction. Before the structural gates, a 71x34
    blob out of the map classified as a `4` at score 0.262 and a 26x27 one as an `8` at 0.546."""
    intruders = ((470, DIGIT_TOP_Y, 71, 34), (300, 12, 26, 27))
    got = reader.brawlers_left(synth_hud(4, blocks=intruders))
    assert got.ok and got.count == 4
    assert len(got.digits) == 1


def test_the_band_stops_above_the_hud_element_below_it(reader):
    """A different white element sits at y ~ 123 and its top edge produced a phantom 26x18
    component in 40 of 163 frames of an early pass. The band's lower edge is where it is on
    purpose, so a block drawn there must not be reachable."""
    got = reader.brawlers_left(synth_hud(3, blocks=((380, 118, 27, 30),)))
    assert got.ok and got.count == 3
    assert BAND_Y[1] < 118


# ---------------------------------------------------------------- resolution

@pytest.mark.parametrize("height", [1126, 1080])
def test_the_same_widget_reads_the_same_at_both_capture_resolutions(reader, height):
    """The fixtures carry both: 2002x1126, which is also what `DeployCapture` resizes to, and
    1920x1080, which is what the BlueStacks clip arrives at because `ClipReader` never resizes. A
    hardcoded pixel would pass on one and fail on the other."""
    for want in (1, 9, 10):
        got = reader.brawlers_left(synth_hud(want, height=height))
        assert got.ok, f"{want} at {height}px came back {got.status}"
        assert got.count == want


def test_the_scale_is_taken_from_height_and_the_viewport_is_sixteen_by_nine(reader):
    """`capture.normalize_viewport` trims every source to exactly 16:9, which is what makes one
    scalar enough. If that ever changes this reader needs a second one."""
    from brawl_vision.capture import SHOWDOWN_ASPECT
    assert SHOWDOWN_ASPECT == pytest.approx(16 / 9)
    frame = synth_hud(9, height=1080)
    assert frame.shape[1] / frame.shape[0] == pytest.approx(16 / 9, abs=1e-3)


# ---------------------------------------------------------------- refusal

def test_a_third_digit_is_fatal_rather_than_truncated(reader):
    """Three digits in a field whose maximum is 10 means the segmentation is wrong, not that one
    component should be dropped -- the same call `read.py` makes with `MAX_DIGITS`."""
    got = reader.brawlers_left(synth_hud(digits=[1, 2, 3]))
    assert got.count is None and got.status == TOO_LONG


def test_a_count_the_mode_cannot_produce_is_refused(reader):
    """Two well-formed digits reading 47 is a mis-segmentation with a plausible shape. Solo
    Showdown fields ten."""
    got = reader.brawlers_left(synth_hud(47))
    assert got.count is None and got.status == OUT_OF_RANGE


def test_zero_is_not_a_count(reader):
    """The match is over before the last brawler dies, so a 0 on screen is a misread -- and 0 is
    the value a caller would be most likely to act on."""
    got = reader.brawlers_left(synth_hud(digits=[0]))
    assert got.count is None and got.status == OUT_OF_RANGE


def test_a_leading_zero_is_a_lost_digit_and_not_a_number(reader):
    """The game never zero-pads. Same gate, and same reasoning, as `read.py:LEADING_ZERO`."""
    got = reader.brawlers_left(synth_hud(digits=[0, 9]))
    assert got.count is None and got.status == LEADING_ZERO


def test_a_gap_too_wide_after_the_colon_is_not_the_count(reader):
    """Measured colon-to-digit gap is 14-16 px. Something digit-shaped 40 px out is a map pixel
    that happened to land in the band, not the count drawn oddly."""
    got = reader.brawlers_left(synth_hud(9, gap=40))
    assert got.count is None and got.status == NO_DIGITS


def test_a_shape_that_is_not_a_digit_is_refused_by_the_glyph_floor(reader):
    """The backstop, and only that. A solid 28x41 white block classifies as an `8` at score
    0.626, under the 0.70 floor -- but a solid 19x41 one classifies as a `1` at 0.753 and would
    be read. The structural gates are the defence; this catches what is left."""
    got = reader.brawlers_left(synth_hud(digits=[], blocks=(
        (COLON_X + COLON_DOT + COLON_TO_DIGIT, DIGIT_TOP_Y, 28, 41),)))
    assert got.count is None and got.status == LOW_GLYPH


def test_a_dropped_trailing_digit_reads_low_and_this_reader_cannot_see_it(reader):
    """**The one failure with no defence, recorded rather than fixed.** If the `0` of a `10` fails
    the white threshold, the row simply ends after the `1` -- there is no gap to notice, because
    the count is left-aligned to the colon and a one-digit and a two-digit count start at the same
    x. It reads 1, which is legal.

    Not reachable in the measurement (1692 in-match reads, zero non-monotone steps, many of them
    two-digit), and structurally it needs the `0`'s white to fail while the `1`'s survives two
    pixels away. §6.3's shadow state is where it would be caught, because the count it produces
    is always TOO LOW and never too high."""
    got = reader.brawlers_left(synth_hud(10, spacing=30))
    assert got.ok and got.count == 1, "the wide gap should end the row, not span it"


def test_a_frame_that_is_not_bgr_raises_rather_than_reading_noise(reader):
    with pytest.raises(ValueError):
        reader.brawlers_left(np.zeros((1126, 2002), np.uint8))
    with pytest.raises(ValueError):
        reader.brawlers_left(np.zeros((1126, 2002, 3), np.float32))


def test_a_frame_too_small_to_hold_the_widget_fails_closed(reader):
    got = reader.brawlers_left(np.zeros((64, 114, 3), np.uint8))
    assert got.count is None and got.status == NO_COLON


# ---------------------------------------------------------------- the off-by-one

def test_brawlers_left_includes_the_hero_and_the_observation_does_not():
    """`obs_schema.meta.n_enemies_alive` is `n_alive - hero_alive`; the game counts you. The
    conversion takes `hero_alive` with no default so the assumption is written down at the call
    site rather than guessed here -- the error it prevents is invisible, being a plausible
    integer either way."""
    reading = BrawlersLeft(4, OK)
    assert reading.enemies_alive(hero_alive=True) == 3
    assert reading.enemies_alive(hero_alive=False) == 4


def test_an_unread_counter_converts_to_none_rather_than_to_minus_one():
    assert BrawlersLeft(None, NO_COLON).enemies_alive(hero_alive=True) is None


# ---------------------------------------------------------------- config

def test_the_constructor_defaults_match_the_shipped_config():
    """The same pin `read.py` carries: two places state these numbers and they must not drift."""
    cfg = load_vision_config("configs/vision.yaml")
    default = HudReader(bank=_BANK)
    from_cfg = HudReader.from_config(cfg, bank=_BANK)
    for name in ("max_brawlers", "min_glyph_score", "min_glyph_margin",
                 "white_max_sat", "white_min_val"):
        assert getattr(default, name) == getattr(from_cfg, name), name


def test_the_hud_reader_shares_the_hp_white_rather_than_owning_a_copy():
    """One font in one colour. A near-duplicate pair of keys would be two knobs that must agree."""
    cfg = VisionConfig()
    reader = HudReader.from_config(cfg, bank=_BANK)
    assert reader.white_max_sat == cfg.hp_white_max_sat
    assert reader.white_min_val == cfg.hp_white_min_val


def test_configs_vision_yaml_carries_the_hud_settings():
    cfg = load_vision_config("configs/vision.yaml")
    validate(cfg)
    assert cfg.hud_max_brawlers == 10
    assert 0.0 < cfg.hud_min_glyph_score < 1.0
    assert 0.0 <= cfg.hud_min_glyph_margin < 1.0


def test_a_three_digit_maximum_is_refused_because_the_reader_cannot_read_one():
    """`MAX_DIGITS` is 2. A mode with 100 brawlers would have every full count rejected as a
    mis-segmentation, silently, so the config refuses the combination instead."""
    cfg = VisionConfig(hud_max_brawlers=100)
    with pytest.raises(ValueError, match="three digits"):
        validate(cfg)


def test_the_digit_height_gate_holds_both_measured_resolutions():
    """Measured 40-42 at 1126 and 39-40 at 1080, which scale to 40.7-41.7. The gate is in
    reference pixels, so it must hold the 1126 numbers directly."""
    assert DIGIT_H[0] <= 40 and DIGIT_H[1] >= 42
    assert DIGIT_H[0] <= 39 * 1126 / 1080 and DIGIT_H[1] >= 40 * 1126 / 1080


# ---------------------------------------------------------------- footage

@pytest.mark.vision
def test_brawlers_left_across_the_bluestacks_clip():
    """The footage check. Two properties, neither an accuracy threshold on gitignored footage:

    every read is inside 1..10, and **the count never increases** -- brawlers left is monotone
    non-increasing within a match, which is the free ground truth this field happens to have and
    the check a mis-segmentation fails. Measured over the whole clip: 163 reads, values 10 -> 9 ->
    8 -> 7, zero increases, and every non-read is before frame 230 where the counter has not
    appeared yet.
    """
    from brawl_vision.clips import ClipReader

    reader = HudReader.from_config()
    reads = []
    misses_after_first = 0
    for i, f in enumerate(ClipReader(CLIP)):
        if i % 5:
            continue
        got = reader.brawlers_left(getattr(f, "image", f))
        if got.ok:
            reads.append((i, got))
        elif reads:
            misses_after_first += 1

    assert len(reads) >= 100, f"only {len(reads)} reads -- the clip or the locator changed"
    assert all(1 <= r.count <= 10 for _, r in reads)
    rises = [(i, a.count, j, b.count) for (i, a), (j, b) in zip(reads, reads[1:])
             if b.count > a.count]
    assert not rises, f"the count went UP, which a match cannot do: {rises}"
    assert misses_after_first == 0, (
        f"{misses_after_first} frames lost the counter after it first appeared")
