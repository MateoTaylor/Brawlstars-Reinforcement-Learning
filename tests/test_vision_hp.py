"""Reading HP out of a detection box. See brawl_vision/object_detection/hp_detection/.

**The same three tiers `tests/test_vision_detector.py` uses, for the same reasons.** Localization,
classification, the gates and the tracker are all ours and take synthetic inputs, so they are
tested unconditionally. Tests needing the digit templates skip without them. Tests running the
detector over real footage carry `@pytest.mark.vision` on top of that.

**What is asserted here is the FAILURE MODES, not an accuracy number.** The accuracy that matters
was measured by hand against real crops and is recorded in the package docstring; a threshold on
it in CI would be a regression test against gitignored footage. What these tests pin down is
every specific way this stage is known to go wrong -- a popup covering the readout, a digit lost
off one end, a neighbour's number pulled into a padded crop, a track carrying a stale value --
because each of those was a real bug, and each would otherwise come back silently.
"""
import numpy as np
import pytest

from brawl_vision.config import VisionConfig, load_vision_config, validate
from brawl_vision.object_detection import Detection
from brawl_vision.object_detection.hp_detection import glyphs as glyph_mod
from brawl_vision.object_detection.hp_detection import locate
from brawl_vision.object_detection.hp_detection.draw import confidence_color, draw_health
from brawl_vision.object_detection.hp_detection.read import (
    GAP, LEADING_ZERO, OCCLUDED, OK, HealthReader, HealthReading, _ramp,
)
from brawl_vision.object_detection.hp_detection.smooth import (
    HealthTracker, _is_truncation, associate,
)

CLIP = "tests/fixtures/vision/day10_gameplay.mp4"


def _has_templates() -> bool:
    return glyph_mod.TEMPLATES_PATH.exists()


needs_templates = pytest.mark.skipif(
    not _has_templates(),
    reason="digit templates are a measured artifact -- scripts/vision_hp_calibrate.py",
)


@pytest.fixture(scope="module")
def bank():
    if not _has_templates():
        pytest.skip("no digit templates")
    return glyph_mod.GlyphBank.load()


# ---------------------------------------------------------------------------
# synthetic frames: a readout drawn the way the game draws one
# ---------------------------------------------------------------------------

def _digit_mask(bank, digit, height=16):
    """A template rendered back out at a plausible on-screen size."""
    import cv2
    t = bank.templates[digit]
    w = max(3, int(round(t.shape[1] * height / t.shape[0])))
    big = cv2.resize(t, (w, height), interpolation=cv2.INTER_LINEAR)
    # Trim the blank columns the canonical canvas pads with, so the blob is the glyph itself.
    cols = np.nonzero((big > 0.4).any(axis=0))[0]
    return (big[:, cols.min():cols.max() + 1] > 0.4) if len(cols) else (big > 0.4)


def synth_frame(bank, number, *, label="player", pitch_gap=2, cover=None, erase=None,
                erase2=None, drop_first=False, shift=0, size=(300, 400)):
    """A BGR frame with one HP readout on it, plus the `Detection` that encloses it.

    Built from the shipped templates rather than from a captured crop, so the test states what it
    depends on. Layout mirrors the real one: white digits above a saturated bar, a team-coloured
    nameplate above those, all centred on the box.

    `cover=i` paints a popup glyph over digit `i` -- white, same colour, taller than a digit, the
    way the game floats a damage number. `erase=i` deletes digit `i` outright. Both take an INDEX
    rather than a flag because both failures depend on WHICH digit goes: one lost off an end is a
    truncation, one lost from the middle is a gap, and they are caught by different code.
    """
    h, w = size
    frame = np.full((h, w, 3), 40, np.uint8)
    digits = [int(c) for c in str(number)]
    if drop_first:
        digits = digits[1:]
    masks = [_digit_mask(bank, d) for d in digits]
    total = sum(m.shape[1] for m in masks) + pitch_gap * (len(masks) - 1)
    x = (w - total) // 2 + shift
    y = 60
    spans = []
    for i, m in enumerate(masks):
        if i != erase and i != erase2:
            frame[y:y + m.shape[0], x:x + m.shape[1]][m] = (250, 250, 250)
        spans.append((x, x + m.shape[1]))
        x += m.shape[1] + pitch_gap
    # bar: green (hue ~56) for player, red for enemy, under the digits
    color = (60, 230, 90) if label == "player" else (60, 60, 230)
    frame[y + 20:y + 30, (w - 120) // 2:(w + 120) // 2] = color
    # nameplate: same font-ish blobs but team-coloured, which must NOT join the digit row
    frame[y - 26:y - 10, w // 2 - 40:w // 2 + 40] = color
    if cover is not None:
        # Taller than a digit in BOTH directions, which is what makes it a popup rather than a
        # glyph, and what makes its bounding box intersect the surviving row's.
        cx0, cx1 = spans[cover]
        frame[y - 14:y + 30, cx0 - 1:cx1 + 1] = (250, 250, 250)
    det = Detection(label, 0.9, (float(w // 2 - 100), 20.0, float(w // 2 + 100), 260.0))
    return frame, det


# ---------------------------------------------------------------------------
# glyphs
# ---------------------------------------------------------------------------

@needs_templates
def test_the_shipped_templates_are_ten_digits_of_the_canonical_size(bank):
    assert bank.templates.shape == (10, glyph_mod.GLYPH_H, glyph_mod.GLYPH_W)
    assert bank.templates.min() >= 0.0 and bank.templates.max() <= 1.0


@needs_templates
def test_every_template_classifies_as_itself(bank):
    """The floor. A template set where a digit does not win against its own image is scrambled --
    which is exactly what a mislabelled cluster in the calibration script would produce."""
    for digit in range(10):
        got, score, _ = bank.classify(bank.templates[digit])
        assert got == digit
        assert score > 0.99


@needs_templates
def test_no_two_templates_are_near_duplicates(bank):
    """`6` and `8` are the closest pair at 0.802. Anything above ~0.9 would mean two clusters got
    the same label during calibration, which is invisible until numbers start coming out wrong."""
    w = glyph_mod._whiten(bank.templates)
    sim = w @ w.T
    np.fill_diagonal(sim, 0.0)
    assert sim.max() < 0.9, f"templates {np.unravel_index(sim.argmax(), sim.shape)} are too alike"


def test_normalize_preserves_aspect_so_a_one_stays_narrow():
    """Height is filled and width is centred, deliberately -- squashing every glyph into a square
    would throw away the cheapest cue there is for the narrow digits."""
    wide = glyph_mod.normalize(np.ones((16, 12), bool))
    narrow = glyph_mod.normalize(np.ones((16, 4), bool))
    assert wide.sum() > narrow.sum() * 2
    assert wide.shape == narrow.shape == (glyph_mod.GLYPH_H, glyph_mod.GLYPH_W)


def test_normalize_survives_a_degenerate_mask():
    assert glyph_mod.normalize(np.zeros((0, 0), bool)).shape == (glyph_mod.GLYPH_H,
                                                                 glyph_mod.GLYPH_W)


def test_a_missing_template_file_says_how_to_regenerate_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="vision_hp_calibrate"):
        glyph_mod.GlyphBank.load(tmp_path / "nope.npz")


def test_a_wrong_shaped_template_set_is_refused():
    with pytest.raises(ValueError, match="10 templates"):
        glyph_mod.GlyphBank(np.zeros((7, glyph_mod.GLYPH_H, glyph_mod.GLYPH_W), np.float32))


# ---------------------------------------------------------------------------
# locate
# ---------------------------------------------------------------------------

def test_the_crop_is_the_top_of_the_box_and_is_padded_sideways():
    """Padded because the health bar is drawn to a width set by MAX HP and overhangs the box --
    measured up to 160 px of bar against a 135 px box."""
    det = Detection("player", 0.9, (100.0, 200.0, 300.0, 500.0))
    x0, y0, x1, y1, cx = locate.crop_for(det, (1126, 2002, 3), 0.55, 0.12)
    assert (x0, x1) == (76, 324)                 # 200 px box, 24 px each side
    assert (y0, y1) == (200, 365)                # top 55% of 300 px
    assert cx == pytest.approx(124.0)            # box centre, relative to the crop


def test_the_crop_is_clamped_to_the_frame():
    """A brawler at the screen edge has a box that legitimately runs past it -- `Detection.xyxy`
    keeps the model's own coordinates, so the clamp has to live here."""
    det = Detection("enemy", 0.9, (-40.0, -30.0, 60.0, 200.0))
    x0, y0, x1, y1, _ = locate.crop_for(det, (300, 400, 3), 0.55, 0.12)
    assert x0 == 0 and y0 == 0 and x1 <= 400 and y1 <= 300


@needs_templates
def test_the_nameplate_does_not_join_the_digit_row(bank):
    """The nameplate sits a few pixels above the number in the same font at the same size, and is
    told apart ONLY by being team-coloured. If the saturation ceiling ever stops excluding it,
    the row grows extra 'digits' and the number silently gains leading characters."""
    import cv2
    frame, det = synth_frame(bank, 8400)
    x0, y0, x1, y1, cx = locate.crop_for(det, frame.shape, 0.55, 0.12)
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    row = locate.find_digit_row(hsv, cx, max_sat=70, min_val=175, min_h=13, max_h=19)
    assert row is not None and len(row.blobs) == 4


def test_longest_run_measures_a_run_and_not_a_count():
    """A solid bar and a row of text can have identical pixel COUNTS. Run length is what separates
    them, and getting this backwards is how a hue search locks onto a nameplate."""
    assert locate._longest_run(np.array([1, 1, 0, 1, 1, 1, 0], bool)) == (3, 3)
    assert locate._longest_run(np.zeros(5, bool)) == (0, 0)


def test_the_bar_hue_band_may_wrap_past_the_top_of_the_wheel():
    """Enemy red lives at BOTH ends of OpenCV's 0-179 hue wheel, so its band is spelled (166, 5)
    with the low edge above the high one.

    This is also the regression test for the way it was first written. `(166, 180)` looks like
    "red up to the top of the wheel" and is wrong twice over: 180 is not a hue, and because
    166 <= 180 the band does not wrap either, so it silently matched nothing at the red end it
    exists for. `validate` now rejects the 180; this pins the wrap itself.
    """
    hsv = np.zeros((6, 40, 3), np.uint8)
    hsv[:, :, 0] = 90                            # background: teal, well outside any red band
    hsv[:, :, 1:] = 255
    hsv[2, 5:35, 0] = 2                          # hue 2: red, on the far side of the wrap
    assert locate.find_bar(hsv, (166, 5), below_y=0).length == 30
    assert locate.find_bar(hsv, (166, 179), below_y=0) is None, "a non-wrapping band must not"


def test_a_bar_of_the_other_team_s_colour_is_not_found():
    hsv = np.zeros((6, 40, 3), np.uint8)
    hsv[:, :, 1:] = 255
    hsv[2, 5:35, 0] = 56                         # player green
    assert locate.find_bar(hsv, (166, 180), below_y=0) is None


# ---------------------------------------------------------------------------
# read: the gates
# ---------------------------------------------------------------------------

@needs_templates
def test_a_clean_readout_is_read_correctly_and_confidently(bank):
    reading = HealthReader(bank).read(*synth_frame(bank, 8400))
    assert reading.status == OK
    assert reading.hp == 8400
    assert reading.confidence > 0.8


@needs_templates
@pytest.mark.parametrize("number", [500, 2000, 8400, 10800, 11600])
def test_numbers_of_every_length_the_game_shows(bank, number):
    reading = HealthReader(bank).read(*synth_frame(bank, number))
    assert reading.hp == number


@needs_templates
def test_a_popup_over_the_readout_is_fatal_rather_than_noisy(bank):
    """THE failure this chunk exists for. A heal popup covers all but one digit of `8400`, and
    the survivor is a clean `8` the classifier scores 0.928 -- so per-glyph confidence is high and
    the read is off by a factor of a thousand. It must not come back as a low-confidence number."""
    reading = HealthReader(bank).read(*synth_frame(bank, 10800, cover=2))
    assert reading.hp is None
    assert reading.status == OCCLUDED
    assert reading.confidence == 0.0


@needs_templates
def test_a_digit_missing_from_the_middle_is_caught_by_the_gap(bank):
    """A digit lost out of the MIDDLE leaves a hole bigger than any real inter-glyph space.
    Checked separately from the popup case because nothing has to be drawn over the number for
    this to happen -- a digit sitting on a bright sprite fails the white threshold on its own."""
    reading = HealthReader(bank).read(*synth_frame(bank, 10800, erase=2))
    assert reading.hp is None and reading.status == GAP


@needs_templates
def test_two_adjacent_missing_digits_are_caught_too(bank):
    """The regression test for a confident wrong read found on real footage: `10400` with both the
    `0` and the `4` covered came back as `100` at confidence 1.00.

    It slipped through because the gap test used to compare the widest left-edge pitch against the
    MEDIAN pitch, and with only two gaps surviving (36 px and 15 px) the median is their mean --
    so the 36 pulled the reference up to 25.5 and hid itself. The test is now an absolute space
    against a measured font constant, which cannot be dragged around by the very gap it is
    looking for.
    """
    reading = HealthReader(bank).read(*synth_frame(bank, 10400, erase=1, erase2=2))
    assert reading.hp != 100
    assert reading.hp is None and reading.status == GAP


def test_the_gap_test_is_absolute_and_not_a_ratio_against_the_row():
    """The property that makes the above work, stated directly on `DigitRow`: one enormous gap
    must not be able to raise the bar it is measured against."""
    def row(edges):
        return locate.DigitRow(tuple(locate.Blob((x0, 0, x1, 16), np.ones((16, x1 - x0), bool))
                                     for x0, x1 in edges), popup_overlap=False)
    assert row([(0, 12), (14, 26), (28, 40)]).widest_space == 2
    # the real failure: two glyphs missing between the first pair, one normal space after
    assert row([(72, 79), (108, 121), (123, 135)]).widest_space == 29


@needs_templates
def test_a_leading_zero_is_proof_a_digit_was_lost(bank):
    """The game never zero-pads, so `0400` is not a number it can have drawn."""
    reading = HealthReader(bank).read(*synth_frame(bank, 10800, drop_first=True))
    assert reading.hp is None and reading.status == LEADING_ZERO


@needs_templates
def test_an_off_centre_row_is_graded_down_but_still_reported(bank):
    """Centring is a HINT, not a gate -- an intact row already sits 0.38 pitches off the box
    centre at p75, because the detector's box breathes with auras. Turning this into a rejection
    discards sound reads at a rate far above the errors it catches."""
    centred = HealthReader(bank).read(*synth_frame(bank, 8400))
    skewed = HealthReader(bank).read(*synth_frame(bank, 8400, shift=14))
    assert skewed.hp == 8400
    assert skewed.confidence < centred.confidence


@needs_templates
def test_a_missing_bar_is_a_penalty_and_not_a_rejection(bank):
    """The filled bar shrinks with health, so rejecting boxes whose bar cannot be found discards
    precisely the nearly-dead enemies whose HP the policy most needs."""
    frame, det = synth_frame(bank, 8400)
    frame[80:92, :] = 40                          # erase the bar, keep the digits
    reading = HealthReader(bank).read(frame, det)
    assert reading.hp == 8400
    assert reading.bar_px is None
    assert 0.0 < reading.confidence <= 0.75


@needs_templates
def test_reading_an_empty_patch_is_not_an_error(bank):
    frame = np.full((300, 400, 3), 40, np.uint8)
    det = Detection("player", 0.9, (100.0, 20.0, 300.0, 260.0))
    reading = HealthReader(bank).read(frame, det)
    assert reading.hp is None and reading.confidence == 0.0


@needs_templates
def test_read_rejects_anything_that_is_not_a_uint8_bgr_frame(bank):
    det = Detection("player", 0.9, (0.0, 0.0, 10.0, 10.0))
    with pytest.raises(ValueError, match="uint8 BGR"):
        HealthReader(bank).read(np.zeros((50, 50, 3), np.float32), det)


def test_the_confidence_ramp_saturates_at_both_ends_and_inverts():
    assert _ramp(0.5, 0.6, 0.9) == 0.0
    assert _ramp(1.0, 0.6, 0.9) == 1.0
    assert _ramp(0.75, 0.6, 0.9) == pytest.approx(0.5)
    # lo > hi ramps downward, which is how centring is scored
    assert _ramp(0.1, 0.7, 0.35) == 1.0 and _ramp(0.9, 0.7, 0.35) == 0.0


# ---------------------------------------------------------------------------
# smooth
# ---------------------------------------------------------------------------

def test_truncation_recognises_a_partly_covered_readout():
    """`8400` covered at either end reads as `400` or `840`, and both are plausible HP values that
    no range check would question. This is the one shape the tracker refuses to take on trust."""
    assert _is_truncation(400, 8400)
    assert _is_truncation(840, 8400)
    assert _is_truncation(8, 8400)
    assert _is_truncation(8400, 400)              # symmetric: either side can be the covered one
    assert not _is_truncation(8400, 7400)         # a real hit
    assert not _is_truncation(8400, 8400)
    assert not _is_truncation(1370, 8200)


def _reading(hp, confidence, xyxy=(100.0, 100.0, 200.0, 300.0), label="player", status=OK):
    return HealthReading(hp, confidence, status, Detection(label, 0.9, xyxy))


def test_a_one_frame_wrong_value_never_reaches_the_output():
    """The whole point. A transient must be outvoted by the frames either side of it without ever
    being committed, even once."""
    t = HealthTracker(reader=object(), confirm_frames=2)
    assert t.update_readings([_reading(8400, 0.95)])[0].hp == 8400
    assert t.update_readings([_reading(8400, 0.95)])[0].hp == 8400
    assert t.update_readings([_reading(400, 0.95)])[0].hp == 8400     # truncation, held pending
    assert t.update_readings([_reading(8400, 0.95)])[0].hp == 8400


def test_a_real_change_commits_after_confirmation():
    t = HealthTracker(reader=object(), confirm_frames=2)
    t.update_readings([_reading(8400, 0.95)])
    assert t.update_readings([_reading(7400, 0.6)])[0].hp == 8400     # not yet
    assert t.update_readings([_reading(7400, 0.6)])[0].hp == 7400     # confirmed


def test_a_very_confident_change_commits_immediately():
    """The latency `confirm_frames` costs is not paid when the evidence does not need it."""
    t = HealthTracker(reader=object(), confirm_frames=2, instant_confidence=0.9)
    t.update_readings([_reading(8400, 0.95)])
    assert t.update_readings([_reading(7400, 0.99)])[0].hp == 7400


def test_a_truncation_never_commits_on_confidence_alone():
    """The measured worst case scored 0.928 with a 0.181 margin -- a perfectly formed `8` that
    happened to be all that was left of `8400`. Confidence cannot be allowed to buy this one."""
    t = HealthTracker(reader=object(), confirm_frames=2, instant_confidence=0.9)
    t.update_readings([_reading(8400, 0.95)])
    assert t.update_readings([_reading(8, 0.99)])[0].hp == 8400
    # but a genuinely repeated one still gets through, because the alternative is a stuck value
    assert t.update_readings([_reading(8, 0.99)])[0].hp == 8


def test_a_track_coasts_through_an_unusable_frame_with_decaying_confidence():
    """A readout covered by its own damage popup belongs to a brawler whose HP is still about what
    it was. Reporting nothing there punches a hole in the observation while the agent is being
    shot at."""
    t = HealthTracker(reader=object(), max_misses=3, decay=0.75)
    t.update_readings([_reading(8400, 1.0)])
    first = t.update_readings([_reading(None, 0.0, status=OCCLUDED)])[0]
    assert first.hp == 8400 and first.confidence == pytest.approx(0.75)
    second = t.update_readings([_reading(None, 0.0, status=OCCLUDED)])[0]
    assert second.hp == 8400 and second.confidence == pytest.approx(0.5625)


def test_a_track_gives_up_after_max_misses():
    t = HealthTracker(reader=object(), max_misses=2)
    t.update_readings([_reading(8400, 1.0)])
    for _ in range(3):
        last = t.update_readings([_reading(None, 0.0, status=OCCLUDED)])[0]
    assert last.hp is None


def test_corroboration_raises_confidence_without_ever_reaching_certainty():
    t = HealthTracker(reader=object())
    conf = [t.update_readings([_reading(8400, 0.8)])[0].confidence for _ in range(6)]
    assert conf == sorted(conf)
    assert conf[-1] < 1.0


def test_association_never_crosses_classes():
    """In Solo Showdown a `player` box inheriting an `enemy` track is the one confusion that would
    be actively harmful."""
    tracker = HealthTracker(reader=object())
    tracker.update_readings([_reading(8400, 0.9, label="enemy")])
    out = tracker.update_readings([_reading(3000, 0.9, label="player")])
    assert out[0].hp == 3000
    assert len(tracker.tracks) == 2


def test_association_drops_a_box_that_moved_too_far():
    """A Mortis dash covers ~340 px between decisions against a ~180 px box. Breaking there is
    intended -- this is not a tracker."""
    near = _reading(1, 0.9, (100.0, 100.0, 200.0, 300.0))
    far = _reading(1, 0.9, (900.0, 100.0, 1000.0, 300.0))
    tracker = HealthTracker(reader=object(), match_distance_frac=1.5)
    tracker.update_readings([near])
    matched, fresh = associate([far], tracker.tracks, 1.5)
    assert not matched and len(fresh) == 1


def test_the_tracker_can_be_reset_between_clips():
    t = HealthTracker(reader=object())
    t.update_readings([_reading(8400, 0.9)])
    t.reset()
    assert t.tracks == []


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def test_drawing_does_not_mutate_the_frame_it_was_given():
    """The array handed here is the same one the terrain pipeline reads in that iteration of a
    render loop, so annotating in place would feed overlay ink into a perception stage."""
    frame = np.full((200, 400, 3), 30, np.uint8)
    before = frame.copy()
    out = draw_health(frame, [_reading(8400, 0.9, (10.0, 10.0, 90.0, 90.0))])
    assert np.array_equal(frame, before)
    assert not np.array_equal(out, frame)


def test_a_fatal_reading_draws_its_status_instead_of_a_number():
    frame = np.full((200, 400, 3), 30, np.uint8)
    assert draw_health(frame, [_reading(None, 0.0, status=OCCLUDED)]).any()


def test_confidence_colour_runs_low_to_high_without_a_jump():
    low, mid, high = (confidence_color(c) for c in (0.0, 0.5, 1.0))
    assert low != mid != high
    assert confidence_color(0.5) == mid          # the stops meet rather than stepping
    assert confidence_color(-5.0) == low and confidence_color(5.0) == high


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_the_shipped_config_validates():
    validate(load_vision_config())


@needs_templates
def test_the_constructor_defaults_match_the_shipped_config(bank):
    """`HealthReader` spells its defaults twice -- once in `__init__` for direct construction and
    once in `VisionConfig` -- and they drifted apart once already: the enemy hue was fixed in the
    config while `__init__` kept the broken non-wrapping `(166, 180)`, so every reader built
    without a config silently stopped finding enemy health bars. Tests construct readers that
    way, which is exactly why nothing caught it."""
    cfg = load_vision_config()
    direct, configured = HealthReader(bank), HealthReader.from_config(cfg, bank=bank)
    for field in ("bar_hues", "white_max_sat", "white_min_val", "glyph_min_height",
                  "glyph_max_height", "crop_height_frac", "crop_pad_frac",
                  "score_ramp", "margin_ramp"):
        assert getattr(direct, field) == getattr(configured, field), field


def test_the_shipped_config_reaches_the_reader():
    cfg = load_vision_config()
    assert cfg.hp_glyph_max_height < locate.POPUP_MIN_H
    # Enemy red wraps: low > high. `validate` permits exactly that and checks both edges are hues.
    assert cfg.hp_bar_hue_enemy[0] > cfg.hp_bar_hue_enemy[1]
    assert all(0 <= e <= 179 for e in cfg.hp_bar_hue_enemy + cfg.hp_bar_hue_player)


def test_a_glyph_band_reaching_into_popup_territory_is_refused():
    """The single most dangerous config edit available. Above `POPUP_MIN_H` a floating damage
    number is indistinguishable from an HP digit, and a covered readout stops being an error and
    becomes a confident wrong number."""
    with pytest.raises(ValueError, match="POPUP_MIN_H"):
        validate(VisionConfig(hp_glyph_max_height=locate.POPUP_MIN_H + 2))


def test_an_inverted_glyph_height_band_is_refused():
    with pytest.raises(ValueError, match="glyph_min_height"):
        validate(VisionConfig(hp_glyph_min_height=19, hp_glyph_max_height=13))


def test_confirm_frames_may_not_be_zero():
    with pytest.raises(ValueError, match="confirm_frames"):
        validate(VisionConfig(hp_confirm_frames=0))


def test_a_hue_written_on_the_wrong_wheel_is_refused():
    """OpenCV hue is 0-179, a HALF-degree wheel. Writing a 0-359 value is the easiest mistake here
    and produces an empty mask rather than an error -- the same trap `zone.hsv_low` documents."""
    with pytest.raises(ValueError, match=r"\[0, 179\]"):
        validate(VisionConfig(hp_bar_hue_player=(120, 240)))


@pytest.mark.parametrize("field", ["hp_min_glyph_score", "hp_decay", "hp_min_confidence",
                                   "hp_instant_confidence"])
def test_the_unit_interval_fields_are_checked(field):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate(VisionConfig(**{field: 1.5}))


# ---------------------------------------------------------------------------
# real footage
# ---------------------------------------------------------------------------

@pytest.mark.vision
@needs_templates
def test_it_reads_plausible_hp_off_real_footage():
    """Deliberately weak, and about the WIRING rather than the accuracy -- the real accuracy was
    hand-checked and is recorded in the package docstring. What this catches is the pipeline
    coming apart end to end: no boxes, no digits found, or numbers outside anything the game can
    display."""
    from pathlib import Path

    pytest.importorskip("onnxruntime")
    from brawl_vision.object_detection import ObjectDetector
    from brawl_vision.sources import open_source

    if not Path(CLIP).exists():
        pytest.skip(f"{CLIP} is gitignored game footage")
    cfg = load_vision_config()
    detector = ObjectDetector.from_config(cfg)
    tracker = HealthTracker.from_config(cfg)
    trusted = []
    with open_source(CLIP, cfg) as source:
        for frame in source:
            if frame.index > 900:
                break
            if frame.index % 60:
                continue
            for reading in tracker.update(frame.image, detector.predict(frame.image)):
                if reading.trusted(cfg.hp_min_confidence):
                    trusted.append(reading.hp)
    assert len(trusted) >= 8, f"only {len(trusted)} trusted reads in 15 frames"
    # Brawl Stars HP is a positive integer well under six figures; a number outside that is a
    # segmentation failure that got through, not an unusual brawler.
    assert all(1 <= hp < 100000 for hp in trusted), trusted


@pytest.mark.vision
@needs_templates
def test_the_tracker_is_steadier_than_the_bare_reader():
    """The claim the whole of `smooth.py` rests on. Counts how often the reported HP CHANGES over
    a stretch of footage: real HP moves, so neither is constant, but the per-frame reader also
    flickers through transients the tracker absorbs."""
    from pathlib import Path

    pytest.importorskip("onnxruntime")
    from brawl_vision.object_detection import ObjectDetector
    from brawl_vision.sources import open_source

    if not Path(CLIP).exists():
        pytest.skip(f"{CLIP} is gitignored game footage")
    cfg = load_vision_config()
    detector = ObjectDetector.from_config(cfg)
    reader = HealthReader.from_config(cfg)
    tracker = HealthTracker.from_config(cfg, reader=reader)
    raw_seq, smooth_seq = [], []
    with open_source(CLIP, cfg) as source:
        for frame in source:
            if frame.index > 1800:
                break
            if frame.index % 5:
                continue
            dets = [d for d in detector.predict(frame.image) if d.label == "player"]
            if not dets:
                continue
            raw = reader.read_all(frame.image, dets[:1])
            raw_seq.append(raw[0].hp)
            smooth_seq.append(tracker.update_readings(raw)[0].hp)

    def flips(seq):
        return sum(1 for a, b in zip(seq, seq[1:]) if a != b)

    assert len(raw_seq) > 30
    assert flips(smooth_seq) < flips(raw_seq), (
        f"tracker changed value {flips(smooth_seq)} times against the reader's {flips(raw_seq)}"
    )
