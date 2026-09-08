"""The ammo and super bars under the HERO's own HP bar. See BRAWL_DEPLOYMENT_DESIGN.md 6.4.

**Why this lives in `hp_detection` and not somewhere named after it.** `locate.py` records the
layout it measured: *"Brawl Stars stacks a brawler's readout in a fixed order, top to bottom: a
power-level pip, the player's NAME, the HP number, the HP bar, and -- for your own brawler only --
an ammo bar and a super-charge bar."* These are the last two rows of that same stack. They are
box-relative, they cannot run without a detection, and they anchor on the digit row and HP bar
this package already finds. The package name is now a little narrow for what it holds; renaming it
would touch every importer for no behavioural gain, so this docstring says so instead.

#### The denominator problem, and why it does not apply here

`read.py` explains at length why it ships no HP fill ratio: the depleted remainder of the HP bar
is a pale wash that does not separate from white text or a pale sprite, so *"a ratio computed
against the filled run alone would be the constant 1.0 dressed up as a measurement."* That is
still true of HP. It is not true here, and the reason is different for each bar:

  ammo    the track is exactly THREE IDENTICAL SEGMENTS. Their common width is the denominator,
          and it is visible in any frame where a single pip is full.
  super   the empty portion is a strongly saturated dark purple, not a wash.

#### What measurement changed about the obvious design

The obvious plan -- find the track's full extent, divide by three -- was tried and **failed**.
Over 60 frames, a mask of "orange or dark slate" returned 35-236 px for a track that is ~115 px:
the map behind the widget is itself dark blue-purple, so the empty part of the track is not
separable from the background even though it is separable from the filled part. The extent is not
recoverable, and anything built on it inherits that.

What replaced it needs no extent at all. **The pips are identical, so painted pixels divided by
one pip's width is the ammo count directly.** Measured over 353 runs, the run-length histogram is
33:44 **34:154 35:120** with nothing else above 20 px -- one sharply defined width, exactly as a
widget drawn from a sprite should be.

**Colour alone still is not enough, and the fix is structure.** Counting every orange run put 16%
of readings above the physical maximum of 3, peaking at 3.71, because the map's orange crate
borders pass the same colour test. Pips sit at a fixed pitch and the bar fills from the left, so a
run whose offset from the leftmost is not near a multiple of that pitch is not a pip. Applying
that dropped the maximum to 3.09. This is `locate.py`'s own lesson -- *"colour alone does not
distinguish a bar from a background"* -- reached independently for a different bar.

**The last 3 px are capped per slot, not clamped at the end.** A frame whose pips render 35 px
against a calibrated 34 reads 3.09. Capping each slot at 1.0 removes it structurally, because "a
pip cannot be more than full" is a fact about the widget; clamping the total to 3 afterwards would
produce the same number while hiding which pip the error was in.

#### Precision, and why it is not worth chasing

**A full pip reads 0.94-1.03, not 1.00.** It renders 33-35 px against the calibrated 34, so ±0.06
per slot is the floor for any single denominator. Two things were checked before accepting it:

  - *Is 34 simply the wrong constant?* No. Across 144 frames showing all three pips the mean pip
    width is 34.33 with sd 0.66, and the mode of 353 individual runs is 34. An earlier pass that
    seemed to show 35 was reporting each frame's LONGEST run -- a maximum, compared against a mode.
  - *Does the widget scale, so that a per-frame denominator would be exact?* Barely. Pitch and
    width do correlate (+0.81), but pitch sd is 0.88 px on 38 (2.3%) and pitch barely tracks the
    digit height (+0.33), so this is shared anti-aliasing noise rather than camera zoom. Deriving
    the denominator from each frame's own pitch moves the spread from 0.88-1.03 to 0.96-1.07 -- no
    better, and it needs two visible pips to work at all.

So the residual is reported rather than hidden. **Do not snap a near-1.0 slot to 1.0**: the game
draws a reloading pip partially filled and `hero.ammo` is specified as *"fractional, reloads
continuously"*, so a snap threshold would quantise away the very signal the field exists for. On
`ammo_frac` the error is ±0.02, which is far below anything a policy trained on it can act on.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from . import locate

# Rows below the HP bar's own row to search for the ammo track. MEASURED over 40 frames: the HP
# bar occupies dy 0-9, a gap runs to dy 14, and the ammo track is dy 15-25. The window is widened
# by a row at each end for the sub-pixel drift of `find_bar`'s chosen row.
AMMO_BAND = (14, 27)

# The pips' centre-to-centre spacing, and how far a run may sit from its slot and still be one.
# MEASURED: pip starts at ~0, ~38, ~77 px. The tolerance is generous because the anchor is the
# leftmost run's own left edge, so error does not accumulate across slots.
PIP_PITCH_PX = 38.5
SLOT_TOL_PX = 8

# Ignore runs shorter than this. Below it, a "pip" is anti-aliasing on a map edge; a genuinely
# reloading pip passes through these widths but spends almost no time there at 4 Hz.
MIN_RUN_PX = 4

# The full width of one pip, in viewport pixels. MEASURED as the mode of 353 run lengths (34, with
# 33 and 35 as its shoulders). It is a property of the widget at a fixed viewport, so it belongs
# here rather than being re-derived per frame -- but `AmmoReading.pip_px` reports the frame's own
# longest run, which is the free cross-check that it has not drifted.
#
# **WHICH viewport, because an earlier version of this comment said the wrong one.** The 353 runs
# came from `bluestacks-example-new.mp4`, which `ClipReader` normalizes to **1920x1080** -- not to
# the 2002x1126 the rest of the vision stack is calibrated at, because `normalize_viewport` trims
# to 16:9 and never resizes, and that clip is natively 16:9 already. This readout stack scales
# with viewport height exactly as `hud.py` measured the top-left HUD to: the super track runs 116
# px at 1080 and a measured 120 at 1126, and the pip runs a median 33 at 1080 and 34 at 1126.
#
# So this constant is right for DEPLOYMENT, which resizes to 2002x1126 (BRAWL_DEPLOYMENT_DESIGN.md
# 9.4 item 4) -- but by luck rather than by design, the 33-35 spread straddling both. Nothing here
# scales, so pointing these readers at a third resolution needs the `REF_H` treatment `hud.py`
# gives its own constants.
PIP_WIDTH_PX = 34

MAX_AMMO_SLOTS = 3


@dataclass(frozen=True)
class AmmoReading:
    """One frame's ammo, in the units `brawl_sim`'s obs schema asks for.

    `slots` is per-pip fill in 0..1, left to right, and is what makes a bad read legible: a
    plausible one is some prefix of full pips, at most one partial, then empties.
    """
    ammo: float                      # 0..3, fractional -- obs_schema hero.ammo
    frac: float                      # ammo / 3        -- obs_schema hero.ammo_frac
    whole: int                       # floor(ammo)     -- obs_schema hero.ammo_whole
    slots: tuple[float, ...]
    row: int                         # the frame row the pips were read from
    pip_px: int                      # longest run this frame; compare against PIP_WIDTH_PX


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """`[(start, length)]` of True runs, longest-first order not imposed -- callers want left to
    right, because the leftmost pip is the anchor."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def pip_mask(crop_hsv_row: np.ndarray, hue_range: tuple[int, int], sat_min: int,
             val_min: int) -> np.ndarray:
    """Pixels that are a filled pip's orange, in one row.

    `hue_range` may wrap past 179 back through 0, matching `locate.find_bar`'s contract and
    `brawl_vision.config.validate`'s allowance for the same asymmetry at the red end.
    """
    h = crop_hsv_row[:, 0].astype(int)
    s = crop_hsv_row[:, 1].astype(int)
    v = crop_hsv_row[:, 2].astype(int)
    lo, hi = hue_range
    hue_ok = (h >= lo) & (h <= hi) if lo <= hi else ((h >= lo) | (h <= hi))
    return hue_ok & (s >= sat_min) & (v >= val_min)


def slot_fills(runs: list[tuple[int, int]], pip_px: int,
               pitch: float = PIP_PITCH_PX, tol: float = SLOT_TOL_PX) -> tuple[float, ...]:
    """Per-pip fill in 0..1, from the painted runs in one row.

    **Anchored on the LONGEST run, not the leftmost.** The leftmost is the tempting choice, since
    the bar fills from the left and slot 0 is therefore painted whenever any ammo is left. It is
    also the one that fails worst: it assumes the leftmost run IS a pip, so a scrap of orange map
    to its left throws every real pip off the lattice at once. Every intruder actually observed in
    the fixture was to the RIGHT of the track (x 189-226 against pips at 59-135), where both
    anchors reject it, so this choice changed no reading here -- it is free insurance against the
    one case that would be silent and total rather than a case already measured.

    Everything is then fitted to a lattice through that reference; runs off the lattice are
    discarded as map, and the surviving runs are shifted so the leftmost sits in slot 0.
    """
    fills = [0.0] * MAX_AMMO_SLOTS
    if not runs:
        return tuple(fills)

    ref = max(runs, key=lambda r: r[1])[0]
    on_lattice = []
    for start, length in runs:
        offset = start - ref
        k = int(round(offset / pitch))
        if abs(offset - k * pitch) <= tol:
            on_lattice.append((k, length))
    if not on_lattice:
        return tuple(fills)

    shift = min(k for k, _ in on_lattice)
    for k, length in on_lattice:
        slot = k - shift
        if slot < MAX_AMMO_SLOTS:
            # min(): a pip cannot be more than full. See the docstring -- this is where the
            # calibrated pip width's last pixel of slack goes, in the slot it belongs to.
            fills[slot] = min(1.0, fills[slot] + length / pip_px)
    return tuple(fills)


def read_ammo(frame: np.ndarray, det, cfg=None) -> AmmoReading | None:
    """Ammo from the hero's own readout stack, or None if the stack could not be located.

    `det` must be the `player` detection -- the ammo bar is drawn for your own brawler only, so
    running this on an enemy box reads whatever the map put there. `None` means "could not read",
    never a guess: BRAWL_DEPLOYMENT_DESIGN.md 6.4 and `configs/agent_obs_lowinfo.yaml` both turn
    on a fabricated value being worse than an absent one.
    """
    from ...config import VisionConfig
    cfg = cfg or VisionConfig()

    x0, y0, x1, y1, box_cx = locate.crop_for(det, frame.shape, cfg.hp_ammo_crop_height_frac,
                                             cfg.hp_crop_pad_frac)
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)

    # Anchor on the digit row and the HP bar's ROW -- never on the HP bar's WIDTH, which shrinks
    # with health (measured 112 px at 8000 HP and 75 px at 5560) while the tracks below it stay
    # full width. An early measurement keyed its window to that width and clipped a full pip to
    # 29%, which looked exactly like a real partial reload.
    row = locate.find_digit_row(hsv, box_cx, max_sat=cfg.hp_white_max_sat,
                                min_val=cfg.hp_white_min_val, min_h=cfg.hp_glyph_min_height,
                                max_h=cfg.hp_glyph_max_height)
    if row is None:
        return None
    bar = locate.find_bar(hsv, cfg.hp_bar_hue_player, row.bottom)
    if bar is None:
        return None

    best_runs: list[tuple[int, int]] = []
    best_row = -1
    best_painted = 0
    for dy in range(*AMMO_BAND):
        yy = bar.xyxy[1] + dy
        if yy >= hsv.shape[0]:
            break
        runs = [r for r in _runs(pip_mask(hsv[yy], cfg.hp_ammo_hue, cfg.hp_ammo_sat_min,
                                          cfg.hp_ammo_val_min)) if r[1] >= MIN_RUN_PX]
        painted = sum(length for _, length in runs)
        if painted > best_painted:
            best_painted, best_runs, best_row = painted, runs, yy

    if best_row < 0:
        # No orange anywhere in the band. That is a real reading, not a failure: an empty magazine
        # looks exactly like this, and Mortis empties his often.
        return AmmoReading(0.0, 0.0, 0, (0.0,) * MAX_AMMO_SLOTS, bar.xyxy[1] + AMMO_BAND[0] + y0, 0)

    fills = slot_fills(best_runs, cfg.hp_ammo_pip_px)
    ammo = float(sum(fills))
    return AmmoReading(ammo=ammo, frac=ammo / MAX_AMMO_SLOTS, whole=int(ammo),
                       slots=fills, row=best_row + y0,
                       pip_px=max((length for _, length in best_runs), default=0))


# ---------------------------------------------------------------------------------------------
# Super charge. Same stack, one track below the ammo pips, and a structurally different problem.
# ---------------------------------------------------------------------------------------------

# Rows below the HP bar's own row to search for the super track. MEASURED over 181 located frames
# of bluestacks-example-new.mp4: the row carrying the most filled paint was dy 33-39 (mode 36) and
# the row carrying the most ready-magenta was dy 33-38 (mode 34). Widened by a row at each end for
# the same sub-pixel drift AMMO_BAND allows for.
SUPER_BAND = (31, 42)

# Runs separated by at most this many pixels are one run. The boundary between the filled part and
# the empty remainder is anti-aliased across a few pixels belonging to neither mask, and treating
# that seam as the end of the track would halve the denominator. MEASURED gaps: 0-4 px.
SUPER_BRIDGE_PX = 6

# Below this, a "run" is a map edge that happens to pass the colour test.
SUPER_MIN_RUN_PX = 6

# Magenta covering at least this share of the track means READY. Not 1.0: the widget's ends are
# rounded and anti-aliased, so a fully magenta track still reports a few percent of neither colour.
SUPER_READY_COVERAGE = 0.6


@dataclass(frozen=True)
class SuperReading:
    """One frame's super charge, in the units `brawl_sim`'s obs schema asks for.

    `track_px` is the extent this frame actually measured, and it is the DENOMINATOR rather than a
    diagnostic -- unlike `AmmoReading.pip_px`, which reports a cross-check against a constant. The
    difference between the two is that this track's empty remainder is painted and ammo's is not.
    """
    charge: float                # 0..1 -- obs_schema hero.super_charge_frac
    ready: bool                  # obs_schema hero.super_ready -- the MAGENTA test, not charge >= 1
    state: str                   # "charging" | "ready" | "empty"
    track_px: int                # this frame's own track extent, the denominator that was used
    row: int                     # the frame row it was read from


def _band_mask(row_hsv: np.ndarray, hue: tuple[int, int], sat: tuple[int, int],
               val: tuple[int, int]) -> np.ndarray:
    """Pixels inside a closed HSV box, with `hue` allowed to wrap past 179 as `pip_mask` does."""
    h = row_hsv[:, 0].astype(int)
    s = row_hsv[:, 1].astype(int)
    v = row_hsv[:, 2].astype(int)
    lo, hi = hue
    hue_ok = (h >= lo) & (h <= hi) if lo <= hi else ((h >= lo) | (h <= hi))
    return hue_ok & (s >= sat[0]) & (s <= sat[1]) & (v >= val[0]) & (v <= val[1])


def _bridged_runs(mask: np.ndarray, bridge: int) -> list[tuple[int, int]]:
    """`[(start, length)]` of True runs, merging any two separated by at most `bridge` False."""
    out: list[tuple[int, int]] = []
    for start, length in _runs(mask):
        if out and start - (out[-1][0] + out[-1][1]) <= bridge:
            out[-1] = (out[-1][0], start + length - out[-1][0])
        else:
            out.append((start, length))
    return out


def read_super(frame: np.ndarray, det, cfg=None) -> SuperReading | None:
    """Super charge from the hero's own readout stack, or None if it could not be read.

    `det` must be the `player` detection: like the ammo pips, this track is drawn for your own
    brawler only. `None` means "could not read" and never a guess, for BRAWL_DEPLOYMENT_DESIGN.md
    6.4's reason -- 6.3's shadow state dead-reckons this field and treats CV as the check, so a
    missing read costs one tick of confirmation while a fabricated one is silent.

    **The magenta test comes first, and a reader without it would fail in the worst way this field
    allows.** At full charge the track does not fill with yellow: the whole widget turns magenta,
    MEASURED at hue 162-166 with saturation exactly 255 at every percentile from p5 to p95. So a
    reader that counts yellow and divides reports **0.0 charge at the exact moment the super became
    available** -- silently, and into the field that gates whether the action mask offers bin 2 to
    the policy at all.

    **The denominator is this frame's own track, not a calibrated constant, and that is the one
    structural difference from `read_ammo`.** Ammo could not work this way: its empty remainder is
    dark slate against a dark blue-purple map, a mask for it returned 35-236 px for a ~115 px
    track, and so ammo counts painted pixels against a fixed pip width instead. The super track's
    empty remainder is a distinct colour -- MEASURED hue 124-133, sat 102-143, val 74-99 -- so
    filled and empty together recover the extent and the reading normalizes itself.

    That separation is narrow, and the narrowness is load-bearing rather than incidental. A first
    pass used sat >= 80 with val 60-140, which reads as a reasonable band and is in fact wide
    enough to admit the map behind the widget: merged tracks came out 110-225 px for a 108 px
    widget, and every charge reading was deflated in proportion. `hp.super_track_px` exists to
    catch precisely that -- it decides whether the extent found is credible, and it is never
    divided by.
    """
    from ...config import VisionConfig
    cfg = cfg or VisionConfig()

    x0, y0, x1, y1, box_cx = locate.crop_for(det, frame.shape, cfg.hp_ammo_crop_height_frac,
                                             cfg.hp_crop_pad_frac)
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)

    # The HP bar's ROW, never its width -- see read_ammo. The tracks below it stay full width while
    # the bar itself shrinks with damage, and keying to that width is what once clipped a full pip.
    row = locate.find_digit_row(hsv, box_cx, max_sat=cfg.hp_white_max_sat,
                                min_val=cfg.hp_white_min_val, min_h=cfg.hp_glyph_min_height,
                                max_h=cfg.hp_glyph_max_height)
    if row is None:
        return None
    bar = locate.find_bar(hsv, cfg.hp_bar_hue_player, row.bottom)
    if bar is None:
        return None

    lo = cfg.hp_super_track_px - cfg.hp_super_track_tol_px
    hi = cfg.hp_super_track_px + cfg.hp_super_track_tol_px
    best: SuperReading | None = None

    for dy in range(*SUPER_BAND):
        yy = bar.xyxy[1] + dy
        if yy >= hsv.shape[0]:
            break
        r = hsv[yy]
        filled = pip_mask(r, cfg.hp_super_hue, cfg.hp_super_sat_min, cfg.hp_super_val_min)
        ready = pip_mask(r, cfg.hp_super_ready_hue, cfg.hp_super_ready_sat_min,
                         cfg.hp_super_ready_val_min)
        empty = _band_mask(r, cfg.hp_super_empty_hue, cfg.hp_super_empty_sat,
                           cfg.hp_super_empty_val)

        for start, length in _bridged_runs(filled | ready | empty, SUPER_BRIDGE_PX):
            if length < SUPER_MIN_RUN_PX or not lo <= length <= hi:
                continue
            seg = slice(start, start + length)
            n_ready = int(ready[seg].sum())
            n_filled = int(filled[seg].sum())
            if n_ready >= SUPER_READY_COVERAGE * length:
                cand = SuperReading(1.0, True, "ready", length, yy + y0)
            else:
                # Capped rather than trusted: `filled` and `empty` can each claim a pixel of the
                # anti-aliased seam, so a nearly-charged track can total slightly more than its own
                # extent. A charge above 1 is not a state this field has.
                charge = min(1.0, n_filled / length)
                cand = SuperReading(charge, False, "charging" if n_filled else "empty",
                                    length, yy + y0)
            # Longest credible track wins. A partly occluded row yields a shorter one that can
            # still clear the gate, and it is a worse view of the same widget.
            if best is None or cand.track_px > best.track_px:
                best = cand

    return best
