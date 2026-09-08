"""The ammo track under the hero's own HP bar. See object_detection/hp_detection/hero_bars.py.

**Synthetic throughout, and no digit templates needed.** `read_ammo` reaches the pips through
`locate.find_digit_row` and `locate.find_bar`, both of which are pure geometry and colour --
`locate.py` classifies nothing. So a readout can be drawn here out of white rectangles and a
green run without `data/digits.npz`, and these tests run on a fresh clone with no footage and no
GPU. The one test that touches real frames carries `@pytest.mark.vision`.

**What is asserted is the failure modes, following `tests/test_vision_hp.py`.** The accuracy
number belongs in the module docstring where it can carry its measurement; a threshold on it here
would be a regression test against gitignored footage. What is pinned down instead is every way
this reader is known to go wrong, because each of these was a real bug during its development:

  - the map's orange crate borders counted as pips, putting 16% of readings above a physical
    maximum of 3
  - a pip rendering 35 px against a calibrated 34 pushing a full magazine to 3.09
  - an empty magazine reported as "could not read" rather than as zero, which are opposite
    things to the shadow state that consumes this
  - the window keyed to the HP bar's WIDTH, which shrinks with damage, clipping a full pip to 29%

The near-1.0 readings are asserted with a tolerance of 0.1, deliberately. A full pip renders
33-35 px against the calibrated 34, so demanding 1.00 exactly would be a test of the tolerance
rather than of the reader -- that mistake cost a debugging pass and is recorded here so it is not
repeated.
"""
import cv2
import numpy as np
import pytest

from brawl_vision.config import VisionConfig, validate
from brawl_vision.object_detection import Detection
from brawl_vision.object_detection.hp_detection import AmmoReading, read_ammo
from brawl_vision.object_detection.hp_detection.hero_bars import (
    AMMO_BAND, MAX_AMMO_SLOTS, PIP_PITCH_PX, PIP_WIDTH_PX, SUPER_BAND, _band_mask,
    _bridged_runs, _runs, pip_mask, read_super, slot_fills,
)

CLIP = "tests/fixtures/vision/bluestacks-example-new.mp4"

# The colours the game actually draws, in BGR. Orange sampled at hue 10 sat ~190 val ~230, which
# is the middle of the configured band rather than its edge -- a test sitting on a threshold
# fails for the wrong reason when the threshold moves by one.
ORANGE_BGR = (40, 140, 240)
GREEN_BGR = (60, 230, 90)
WHITE_BGR = (250, 250, 250)


def synth_stack(fills=(1.0, 1.0, 1.0), *, pitch=PIP_PITCH_PX, pip_px=PIP_WIDTH_PX,
                intruders=(), bar_len=112, size=(320, 420), no_digits=False, no_bar=False):
    """A frame carrying one hero readout stack, plus the `Detection` that encloses it.

    Layout mirrors what was measured on real footage: white digits, then the HP bar 20 rows
    below their top, then the ammo track 18 rows below the bar. `intruders` are extra orange
    runs given as `(x_offset_from_first_pip, length)` -- the map showing through, which is the
    thing this reader has to reject.

    `bar_len` exists so a test can shrink the HP bar the way damage does, without moving the
    pips: that decoupling is the bug the anchor rule was written against.
    """
    h, w = size
    frame = np.full((h, w, 3), 40, np.uint8)

    # Digit row: three blobs inside locate's glyph band (13-19 px tall, 4-16 px wide), sharing a
    # top edge so they group into one row.
    digit_y, digit_h = 60, 16
    x = w // 2 - 21
    if not no_digits:
        for _ in range(3):
            frame[digit_y:digit_y + digit_h, x:x + 10] = WHITE_BGR
            x += 14

    bar_y = digit_y + 20
    if not no_bar:
        frame[bar_y:bar_y + 10, (w - bar_len) // 2:(w - bar_len) // 2 + bar_len] = GREEN_BGR

    # Ammo track, 18 rows below the bar's top row -- inside AMMO_BAND's 14-27.
    pip_y = bar_y + 18
    pip_x0 = (w - bar_len) // 2
    for i, f in enumerate(fills):
        painted = int(round(pip_px * f))
        if painted <= 0:
            continue
        sx = pip_x0 + int(round(i * pitch))
        frame[pip_y:pip_y + 8, sx:sx + painted] = ORANGE_BGR
    for dx, length in intruders:
        sx = pip_x0 + dx
        frame[pip_y:pip_y + 8, max(0, sx):max(0, sx) + length] = ORANGE_BGR

    det = Detection("player", 0.9, (float(w // 2 - 100), 20.0, float(w // 2 + 100), 300.0))
    return frame, det


# ---------------------------------------------------------------------------
# runs and the colour mask
# ---------------------------------------------------------------------------

def test_runs_reports_start_and_length_left_to_right():
    m = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0], bool)
    assert _runs(m) == [(1, 3), (6, 2)]


def test_runs_closes_a_run_that_reaches_the_right_edge():
    """Off-by-one at the end of the row would silently shorten the third pip, which is exactly
    where a partial reload lives."""
    assert _runs(np.array([0, 1, 1], bool)) == [(1, 2)]


def test_pip_mask_accepts_a_hue_range_that_wraps_past_179():
    """`locate.find_bar` and `config.validate` both permit a wrapped hue for enemy red. This mask
    matches that contract so the two cannot drift apart."""
    row = np.array([[175, 200, 200], [5, 200, 200], [90, 200, 200]], np.uint8)
    got = pip_mask(row, (170, 10), 110, 130)
    assert got.tolist() == [True, True, False]


def test_pip_mask_rejects_a_dim_or_washed_out_orange():
    """Saturation and value are what separate a lit pip from the dark track behind it; hue alone
    matches both."""
    row = np.array([[10, 200, 200], [10, 40, 200], [10, 200, 60]], np.uint8)
    assert pip_mask(row, (2, 18), 110, 130).tolist() == [True, False, False]


# ---------------------------------------------------------------------------
# slot_fills -- the structural constraint that does the real work
# ---------------------------------------------------------------------------

def test_three_pips_at_the_measured_pitch_read_one_each():
    runs = [(0, 34), (38, 34), (77, 34)]
    assert slot_fills(runs, 34) == pytest.approx((1.0, 1.0, 1.0), abs=0.02)


def test_a_partly_reloaded_pip_reports_its_fraction_rather_than_rounding():
    """`hero.ammo` is specified as fractional and the game draws the reloading pip part-filled.
    Rounding here would quantise the field to thirds and remove the reason it is read at all."""
    fills = slot_fills([(0, 34), (38, 34), (77, 14)], 34)
    assert fills[2] == pytest.approx(14 / 34, abs=0.01)


def test_an_off_lattice_run_is_discarded_as_map_not_added_as_ammo():
    """THE bug this constraint exists for: the map's orange crate borders pass the colour test,
    and counting every orange run put 16% of readings above the physical maximum of 3, peaking at
    3.71. A run that is not near a multiple of the pitch is not a pip."""
    with_map = slot_fills([(0, 34), (38, 34), (77, 34), (140, 20)], 34)
    assert with_map == pytest.approx((1.0, 1.0, 1.0), abs=0.02)
    assert sum(with_map) <= MAX_AMMO_SLOTS


def test_a_run_past_the_third_slot_is_discarded_even_when_it_is_on_the_lattice():
    """A magazine has three pips. A fourth lattice position is the map lining up by chance, and
    admitting it is how a reading exceeds a maximum that is a property of the brawler."""
    assert slot_fills([(0, 34), (38, 34), (77, 34), (115, 34)], 34) == \
        pytest.approx((1.0, 1.0, 1.0), abs=0.02)


def test_the_anchor_is_the_longest_run_so_debris_left_of_the_track_cannot_shift_every_pip():
    """Insurance, and the reason it is worth having. Anchoring on the LEFTMOST run assumes that
    run is a pip; a scrap of map to its left throws all three real pips off the lattice at once,
    which is silent and total rather than one bad slot. Every intruder observed in the fixture
    was to the right, where either anchor rejects it -- so this case is untested by footage and
    is pinned here instead."""
    fills = slot_fills([(0, 9), (25, 34), (63, 34), (102, 34)], 34)
    assert fills == pytest.approx((1.0, 1.0, 1.0), abs=0.02)


def test_a_pip_rendered_wider_than_the_calibration_is_capped_at_full_not_carried_into_the_total():
    """Pips render 33-35 px against a calibrated 34, which read 3.09 before this cap. "A pip
    cannot be more than full" is a fact about the widget, so the cap goes per slot; clamping the
    total to 3 afterwards gives the same number while hiding which pip was wrong."""
    fills = slot_fills([(0, 35), (38, 35), (77, 35)], 34)
    assert all(f <= 1.0 for f in fills)
    assert sum(fills) == pytest.approx(3.0, abs=0.001)


def test_no_runs_is_an_empty_magazine_not_a_shrug():
    assert slot_fills([], 34) == (0.0,) * MAX_AMMO_SLOTS


def test_slot_fills_always_returns_one_entry_per_pip():
    """The obs assembler indexes these positionally, so a short tuple would be an IndexError at
    4 Hz rather than a bad number."""
    for runs in ([], [(0, 34)], [(0, 34), (38, 34)], [(0, 5)], [(200, 34)]):
        assert len(slot_fills(runs, 34)) == MAX_AMMO_SLOTS


# ---------------------------------------------------------------------------
# read_ammo end to end, on a drawn readout
# ---------------------------------------------------------------------------

def test_a_full_magazine_reads_three():
    frame, det = synth_stack((1.0, 1.0, 1.0))
    r = read_ammo(frame, det)
    assert isinstance(r, AmmoReading)
    assert r.ammo == pytest.approx(3.0, abs=0.1)
    assert r.frac == pytest.approx(1.0, abs=0.04)
    assert r.whole == 3


def test_frac_and_whole_are_derived_from_ammo_and_stay_consistent():
    frame, det = synth_stack((1.0, 1.0, 0.5))
    r = read_ammo(frame, det)
    assert r.frac == pytest.approx(r.ammo / MAX_AMMO_SLOTS)
    assert r.whole == int(r.ammo)
    assert r.whole == 2


def test_two_pips_read_two_and_leave_the_third_slot_empty():
    frame, det = synth_stack((1.0, 1.0, 0.0))
    r = read_ammo(frame, det)
    assert r.ammo == pytest.approx(2.0, abs=0.1)
    assert r.slots[2] == 0.0


def test_an_empty_magazine_is_a_reading_of_zero_not_a_failure_to_read():
    """These are opposite signals to the shadow state: zero ammo means "do not fire, and expect a
    reload", while a failed read means "trust dead reckoning this tick". Mortis empties his
    magazine constantly, so conflating them would fire the fallback path several times a match."""
    frame, det = synth_stack(())
    r = read_ammo(frame, det)
    assert r is not None
    assert r.ammo == 0.0
    assert r.slots == (0.0,) * MAX_AMMO_SLOTS


def test_a_missing_digit_row_returns_none_rather_than_a_guess():
    """No anchor means no idea which rows the track is on. `agent_obs_lowinfo.yaml`'s rule --
    a fabricated value in a column the policy trusts fails silently -- is why this is None."""
    frame, det = synth_stack((1.0, 1.0, 1.0), no_digits=True)
    assert read_ammo(frame, det) is None


def test_a_missing_hp_bar_returns_none_rather_than_a_guess():
    frame, det = synth_stack((1.0, 1.0, 1.0), no_bar=True)
    assert read_ammo(frame, det) is None


def test_a_damaged_hero_reads_the_same_ammo_as_a_healthy_one():
    """THE anchor rule. `find_bar` returns the FILLED run, so the bar's width tracks health --
    measured 112 px at 8000 HP and 75 px at 5560 -- while the tracks below it stay full width. An
    early pass keyed its window to that width and reported a full third pip as 29% full, which is
    indistinguishable from a real partial reload."""
    healthy = read_ammo(*synth_stack((1.0, 1.0, 1.0), bar_len=112))
    hurt = read_ammo(*synth_stack((1.0, 1.0, 1.0), bar_len=75))
    assert healthy.ammo == pytest.approx(hurt.ammo, abs=0.05)
    assert hurt.ammo == pytest.approx(3.0, abs=0.1)


def test_orange_map_showing_through_beside_the_track_does_not_add_ammo():
    frame, det = synth_stack((1.0, 1.0, 1.0), intruders=((130, 22), (170, 12)))
    r = read_ammo(frame, det)
    assert r.ammo <= MAX_AMMO_SLOTS
    assert r.ammo == pytest.approx(3.0, abs=0.1)


def test_pip_px_reports_the_frames_own_widest_run_as_a_calibration_check():
    """A free cross-check that the viewport has not been rescaled under the calibrated constant:
    if this drifts away from `PIP_WIDTH_PX`, the denominator is stale. It is REPORTED, not acted
    on, because acting on it would need a full pip to be present to be meaningful."""
    frame, det = synth_stack((1.0, 1.0, 1.0), pip_px=34)
    assert read_ammo(frame, det).pip_px == pytest.approx(34, abs=1)


def test_the_reported_row_is_in_frame_coordinates_not_crop_coordinates():
    """Two coordinate spaces have already bitten this pipeline once. `row` is for drawing an
    overlay on the frame, so it must be offset by the crop's origin."""
    frame, det = synth_stack((1.0, 1.0, 1.0))
    r = read_ammo(frame, det)
    assert r.row > int(det.xyxy[1])
    assert r.row < frame.shape[0]


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_the_ammo_crop_must_reach_further_down_the_box_than_the_digit_crop():
    """The track is BELOW the HP bar, so a crop shorter than the digit search's cannot contain
    the thing it exists for -- and it would fail as an unreadable frame, not as a bad config."""
    import dataclasses
    cfg = dataclasses.replace(VisionConfig(), hp_ammo_crop_height_frac=0.4, hp_crop_height_frac=0.55)
    with pytest.raises(ValueError, match="ammo_crop_height_frac"):
        validate(cfg)


def test_the_shipped_ammo_settings_validate():
    validate(VisionConfig())


def test_configs_vision_yaml_carries_the_ammo_settings():
    """Settings live in the yaml, not only in the dataclass default -- the repo rule, and the
    only place the measurement behind `ammo_pip_px` is written down for whoever re-measures it
    at a new capture resolution."""
    from brawl_vision.config import load_vision_config
    cfg = load_vision_config("configs/vision.yaml")
    assert cfg.hp_ammo_pip_px == 34
    assert tuple(cfg.hp_ammo_hue) == (2, 18)
    assert cfg.hp_ammo_crop_height_frac == 0.75


# ---------------------------------------------------------------------------
# real footage
# ---------------------------------------------------------------------------

@pytest.mark.vision
def test_readings_on_real_footage_are_quantised_and_never_exceed_the_magazine():
    """The label-free properties a correct reader must have, which is all that can be asserted
    without hand-labelling frames: ammo clusters on whole numbers, never exceeds 3, and the slot
    shape is one a magazine can take. Measured over 175 reads of the fixture: 98.3% within 0.15
    of an integer, 0 above 3, 98.3% prefix-shaped. The bounds here are loose enough to pass on
    any correct reader and to fail on the specific regressions above."""
    pytest.importorskip("onnxruntime")
    from pathlib import Path
    if not Path(CLIP).exists():
        pytest.skip(f"{CLIP} is gitignored game footage")

    from brawl_deployment.capture import to_viewport
    from brawl_vision.clips import ClipReader
    from brawl_vision.object_detection import ObjectDetector

    cfg = VisionConfig()
    det = ObjectDetector.from_config(cfg)
    reads = []
    n = 0
    for i, f in enumerate(ClipReader(CLIP, cfg)):
        if i % 5 or n >= 40:
            if n >= 40:
                break
            continue
        img = to_viewport(f.image)
        players = [d for d in det.predict(img) if d.label == "player"]
        if not players:
            continue
        n += 1
        r = read_ammo(img, max(players, key=lambda p: p.confidence), cfg)
        if r is not None:
            reads.append(r)

    assert len(reads) >= 20, "too few reads to say anything"
    a = np.array([r.ammo for r in reads])
    assert a.max() <= MAX_AMMO_SLOTS, f"exceeded the magazine: {a.max()}"
    assert a.min() >= 0.0
    near_whole = np.min([np.abs(a - k) for k in range(4)], axis=0) < 0.15
    assert near_whole.mean() > 0.85, f"only {near_whole.mean():.0%} of reads are near an integer"
    # A full pip is 33-35 px; a median that has walked away from the calibrated width means the
    # viewport was rescaled and the denominator is stale.
    pips = np.array([r.pip_px for r in reads if r.pip_px])
    assert abs(np.median(pips) - PIP_WIDTH_PX) <= 2, f"pip width drifted to {np.median(pips)}"


# =============================================================================================
# Super charge. Same stack, one track lower, and a different shape of problem: this track's empty
# remainder is painted, so the denominator comes from the frame instead of from a constant.
#
# Every test below is a bug that was real during development, or the measurement that settled a
# constant:
#   - a full super reading 0.0 charge, because the track turns MAGENTA rather than filling yellow
#   - the empty-colour mask admitting the map behind the widget, inflating a 116 px track to 225
#     and deflating every charge reading in proportion
#   - an uncharged super reported as "could not read" rather than as zero -- opposite things to
#     the shadow state that consumes this
# =============================================================================================

def _bgr(h, s, v):
    """A BGR pixel from the HSV the game was MEASURED to draw. Written this way round because the
    measurement is in HSV and the config thresholds are in HSV; a hardcoded BGR triple here would
    be a second, silent, transcription of the same number."""
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


# Measured over 181 located frames: filled hue 26-29 / sat 114-197 / val 219-255; ready hue
# 162-166 / sat 255 / val 189-234; empty hue 124-133 / sat 102-143 / val 74-99. Each taken at the
# middle of its band, not its edge -- a test sitting on a threshold fails when the threshold moves.
SUPER_FILLED_BGR = _bgr(28, 160, 249)
SUPER_READY_BGR = _bgr(162, 255, 229)
SUPER_EMPTY_BGR = _bgr(126, 138, 96)

SUPER_TRACK_PX = VisionConfig().hp_super_track_px


def synth_super(charge=0.5, *, ready=False, track_px=SUPER_TRACK_PX, bar_len=112,
                size=(320, 420), seam_px=0, background=None, no_digits=False, no_bar=False):
    """A frame carrying a hero readout stack whose super track is `charge` full.

    `seam_px` leaves that many unpainted pixels between the filled part and the empty remainder,
    which is what the real widget's anti-aliased boundary looks like and what `SUPER_BRIDGE_PX`
    exists to cross. `background` paints an extra run of a given `(x, length, bgr)` in the same
    row -- the map showing through, which is the thing that has to not become part of the track.
    """
    h, w = size
    frame = np.full((h, w, 3), 40, np.uint8)

    digit_y, digit_h = 60, 16
    x = w // 2 - 21
    if not no_digits:
        for _ in range(3):
            frame[digit_y:digit_y + digit_h, x:x + 10] = WHITE_BGR
            x += 14

    bar_y = digit_y + 20
    if not no_bar:
        frame[bar_y:bar_y + 10, (w - bar_len) // 2:(w - bar_len) // 2 + bar_len] = GREEN_BGR

    # 36 rows below the bar's top row, inside SUPER_BAND's 31-42.
    y = bar_y + 36
    x0 = (w - track_px) // 2
    if ready:
        frame[y:y + 8, x0:x0 + track_px] = SUPER_READY_BGR
    else:
        filled = int(round(track_px * charge))
        if filled:
            frame[y:y + 8, x0:x0 + filled] = SUPER_FILLED_BGR
        if filled + seam_px < track_px:
            frame[y:y + 8, x0 + filled + seam_px:x0 + track_px] = SUPER_EMPTY_BGR
    if background is not None:
        bx, blen, bgr = background
        frame[y:y + 8, bx:bx + blen] = bgr

    det = Detection("player", 0.9, (float(w // 2 - 100), 20.0, float(w // 2 + 100), 300.0))
    return frame, det


def test_the_super_synth_reads_back_the_charge_it_was_drawn_with():
    """If this drifts every other test in this section is measuring the wrong thing."""
    for want in (0.0, 0.25, 0.5, 0.75, 1.0):
        frame, det = synth_super(want)
        got = read_super(frame, det)
        assert got is not None, f"could not read a {want} track"
        assert got.charge == pytest.approx(want, abs=0.03)


def test_a_ready_super_reads_full_and_not_empty():
    """THE failure this reader exists to avoid. At full charge the widget does not fill with
    yellow, it turns magenta -- so a reader that counts yellow and divides reports 0.0 charge at
    the exact moment the super became available, silently, into the field that gates the action
    mask."""
    frame, det = synth_super(ready=True)
    got = read_super(frame, det)
    assert got is not None
    assert got.ready is True
    assert got.state == "ready"
    assert got.charge == 1.0, "a magenta track is FULL, and counting yellow would call it empty"


def test_super_ready_is_the_colour_test_not_a_threshold_on_charge():
    """`hero.super_ready` gates action masking, so it must come from the state the game draws
    rather than from an arithmetic edge of the continuous field."""
    charging, det = synth_super(0.99)
    assert read_super(charging, det).ready is False
    ready, det = synth_super(ready=True)
    assert read_super(ready, det).ready is True


def test_an_uncharged_super_is_zero_not_unreadable():
    """Opposite things to the shadow state: zero is a reading it can check against dead reckoning,
    and None costs it a tick of confirmation."""
    frame, det = synth_super(0.0)
    got = read_super(frame, det)
    assert got is not None
    assert got.charge == 0.0 and got.state == "empty"


def test_the_denominator_is_the_frames_own_track_not_the_configured_width():
    """The structural difference from `read_ammo`, which must divide by a calibrated pip width
    because its empty remainder is not separable from the map. Here it is, so a track drawn a few
    pixels off nominal still reads the right fraction rather than a scaled one."""
    cfg = VisionConfig()
    for track in (cfg.hp_super_track_px - 10, cfg.hp_super_track_px + 10):
        frame, det = synth_super(0.5, track_px=track)
        got = read_super(frame, det)
        assert got.track_px == pytest.approx(track, abs=3)
        assert got.charge == pytest.approx(0.5, abs=0.04)


def test_the_anti_aliased_seam_is_bridged_rather_than_ending_the_track():
    """The filled/empty boundary is a few pixels of neither colour. Treating it as the end of the
    track would halve the denominator and double the reported charge."""
    frame, det = synth_super(0.5, seam_px=4)
    got = read_super(frame, det)
    assert got.track_px >= SUPER_TRACK_PX - 6
    assert got.charge == pytest.approx(0.5, abs=0.05)


def test_map_background_of_a_different_darkness_does_not_join_the_track():
    """The measured empty colour is val 74-99. The map behind the widget is the same family of
    dark blue-purple at other values, and a band loose enough to include it inflates the
    denominator -- which was exactly the first pass's failure, at 110-225 px for a 116 px widget."""
    frame, det = synth_super(0.5, background=(30, 90, _bgr(126, 138, 40)))
    got = read_super(frame, det)
    assert got.track_px == pytest.approx(SUPER_TRACK_PX, abs=6)
    assert got.charge == pytest.approx(0.5, abs=0.05)


def test_a_track_inflated_past_the_gate_is_refused_rather_than_reported_low():
    """When the background IS the widget's exact colour there is nothing left to distinguish them,
    and the length gate is the last line. Refusing is right: a deflated charge is a confident wrong
    number in a field the policy was trained to trust."""
    w = 420
    x0 = (w - SUPER_TRACK_PX) // 2
    frame, det = synth_super(0.5, background=(x0 + SUPER_TRACK_PX, 90, SUPER_EMPTY_BGR))
    assert read_super(frame, det) is None


def test_charge_cannot_exceed_one():
    """`filled` and `empty` can each claim a pixel of the seam, so a nearly-charged track can total
    slightly more than its own extent."""
    frame, det = synth_super(1.0)
    got = read_super(frame, det)
    assert got.charge <= 1.0


def test_the_reading_does_not_move_when_the_hp_bar_shrinks_with_damage():
    """The anchor rule `read_ammo` was written against, re-pinned here: the HP bar's ROW is stable
    and its WIDTH is not, and the tracks below it stay full width regardless of health."""
    full = read_super(*synth_super(0.5, bar_len=112))
    hurt = read_super(*synth_super(0.5, bar_len=75))
    assert hurt is not None
    assert hurt.charge == pytest.approx(full.charge, abs=0.02)
    assert hurt.track_px == full.track_px


def test_an_unlocatable_stack_reads_none_rather_than_guessing():
    assert read_super(*synth_super(0.5, no_digits=True)) is None
    assert read_super(*synth_super(0.5, no_bar=True)) is None


# ---------------------------------------------------------------- bridged runs, in isolation

def test_bridged_runs_merges_across_a_small_gap_and_not_a_large_one():
    mask = np.zeros(60, bool)
    mask[0:10] = True
    mask[14:30] = True        # 4-pixel gap -- the seam
    mask[50:60] = True        # 20-pixel gap -- a different thing
    assert _bridged_runs(mask, 6) == [(0, 30), (50, 10)]


def test_bridged_runs_with_no_bridge_is_plain_runs():
    mask = np.array([1, 1, 0, 1, 1, 1], bool)
    assert _bridged_runs(mask, 0) == _runs(mask)


def test_the_band_mask_is_closed_on_both_ends_unlike_pip_mask():
    """`pip_mask` takes minimums because a pip is only ever brighter than its threshold. The empty
    remainder is bounded ABOVE too -- that upper bound on value is the whole thing separating the
    widget from the map."""
    row = np.array([[126, 138, 96], [126, 138, 200], [126, 138, 30]], np.uint8)
    got = _band_mask(row, (117, 137), (90, 165), (62, 118))
    assert got.tolist() == [True, False, False]


# ---------------------------------------------------------------- constants

def test_the_super_band_sits_below_the_ammo_band():
    """Measured stack order: HP bar, gap, ammo track, gap, super track. If these ever overlap one
    reader is reading the other's widget."""
    assert SUPER_BAND[0] >= AMMO_BAND[1]


def test_the_ready_colour_and_the_filled_colour_do_not_overlap():
    """They are separate states, and `read_super` decides between them by which mask covers the
    track. Overlapping bands would make that decision depend on run order."""
    cfg = VisionConfig()
    (flo, fhi), (rlo, rhi) = cfg.hp_super_hue, cfg.hp_super_ready_hue
    assert fhi < rlo or rhi < flo


def test_the_track_gate_admits_the_measured_spread_and_refuses_double_it():
    """Measured extents over 171 reads: p10 114, p50 116, p90 118, min 104, max 128. The gate has
    to hold all of that and reject the 225 px run the first empty-mask attempt produced."""
    cfg = VisionConfig()
    lo = cfg.hp_super_track_px - cfg.hp_super_track_tol_px
    hi = cfg.hp_super_track_px + cfg.hp_super_track_tol_px
    assert lo <= 104 and hi >= 128
    assert hi < 225


@pytest.mark.vision
def test_super_reads_across_the_bluestacks_clip():
    """The footage check, mirroring the ammo one. Two properties, both of which a broken
    denominator violates and neither of which is an accuracy threshold on gitignored footage:
    charge stays inside [0, 1], and it never JUMPS UP -- a super fills gradually and empties
    instantly, so a large positive step means the extent moved, not the charge."""
    from brawl_vision.clips import ClipReader
    from brawl_vision.object_detection.detector import ObjectDetector

    cfg = VisionConfig()
    det = ObjectDetector.from_config(cfg)
    reads = []
    for i, f in enumerate(ClipReader(CLIP)):
        if i >= 1200:
            break
        if i % 8:
            continue
        img = getattr(f, "image", f)
        players = [d for d in det.predict(img) if d.label == "player"]
        if not players:
            continue
        s = read_super(img, players[0], cfg)
        if s is not None:
            reads.append((i, s))

    assert len(reads) >= 60, f"only {len(reads)} reads -- the clip or the locator changed"
    assert all(0.0 <= s.charge <= 1.0 for _, s in reads)

    widths = np.array([s.track_px for _, s in reads])
    assert abs(np.median(widths) - cfg.hp_super_track_px) <= 4, (
        f"track extent median {np.median(widths)} has walked from the calibrated "
        f"{cfg.hp_super_track_px}; something upstream rescaled")

    jumps = [b.charge - a.charge for (i, a), (j, b) in zip(reads, reads[1:])
             if j - i <= 8 and not a.ready and not b.ready]
    assert max(jumps, default=0.0) <= 0.25, "charge jumped up -- the denominator moved"
