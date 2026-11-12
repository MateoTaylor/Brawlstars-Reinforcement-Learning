"""The ammo bar under the HERO's own HP bar. See BRAWL_DEPLOYMENT_DESIGN.md 6.4.

**Why this lives in `hp_detection` and not somewhere named after it.** `locate.py` records the
layout it measured: *"Brawl Stars stacks a brawler's readout in a fixed order, top to bottom: a
power-level pip, the player's NAME, the HP number, the HP bar, and -- for your own brawler only --
an ammo bar and a super-charge bar."* The ammo bar is the next row of that same stack. It is
box-relative, it cannot run without a detection, and it anchors on the digit row and HP bar this
package already finds. The package name is now a little narrow for what it holds; renaming it
would touch every importer for no behavioural gain, so this docstring says so instead.

**There is no super reader here any more** (2026-11-02). The bar under the pips that one was built
on is Mortis's LONG-DASH meter, not his super, and Bull has no such bar; the super is read off the
Super button instead, in `brawl_deployment/perception/super_button.py`.

#### The denominator problem, and why it does not apply here

`read.py` explains at length why it ships no HP fill ratio: the depleted remainder of the HP bar
is a pale wash that does not separate from white text or a pale sprite, so *"a ratio computed
against the filled run alone would be the constant 1.0 dressed up as a measurement."* That is
still true of HP. It is not true of ammo: the track is exactly THREE IDENTICAL SEGMENTS, their
common width is the denominator, and it is visible in any frame where a single pip is full.

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

#### The three ways this reader was silently wrong, and how a deployed run found them

The deployed shadow state compares a dead-reckoned ammo count against this reader once per
decision (BRAWL_DEPLOYMENT_DESIGN.md 6.3). Two live runs kept tripping that comparison, which put
a measurement on the reader that no assertion in `tests/test_vision_hero_bars.py` could: over
**4685 frames of six fixture clips**, scored with NO ground truth on two physical impossibilities
-- a slot fuller than the one to its left, and a whole-pip change reversed inside 0.3 s, which is
faster than any reload or any two shots.

    reader                            non-monotonic frames    impossible reversals   unreadable
    as shipped before this            83  (1.77%)             99 of 182  (54.4%)      0
    + constants scaled to viewport    87  (1.86%)             92 of 179  (51.4%)      0
    + merged runs split               83  (1.77%)             93 of 185  (50.3%)      0
    + both gates (what ships now)      0  (0.00%)             48 of 129  (37.2%)    213  (4.55%)

**More than half the ammo changes this reader reported were physically impossible**, and the test
suite could not see it because every synthetic frame here is drawn at the calibrated scale with
nothing behind it. The three causes, and note that only two of them are causes:

  1. **The constants did not scale with the viewport.** See `REF_VIEWPORT_H`. The lattice error
     accumulates with slot index, so the THIRD pip is the one that vanishes and a full magazine
     reads 2.
  2. **A merged pair lost a pip.** See `split_merged`. 116 of 4108 painted frames carry a run
     wider than one pip.
  3. **A misread returned a number.** See `is_plausible` and the two length gates in `read_ammo`.
     This removed no cause at all -- it makes the reader ADMIT it could not read, which is the
     contract the function's own docstring already claimed. It is also the single largest
     improvement in the table, which is worth sitting with.

**4.55% unreadable is the price, and for this consumer it is the right one.** The shadow state
treats a missing read as `NO_READ` and skips the comparison; it treats a wrong read as evidence
and resyncs onto it. Two live runs were stopped by false resyncs. None was ever harmed by a
skipped one.

Buying more was tried and refused: rejecting frames with no orange at all -- rather than reading
them as the empty magazine they usually are -- reaches 33.8% reversals, but at **16.9%
unreadable**, and it blinds the reader exactly when Mortis is out of ammo, which is the state he
spends the most time in and the one the reload measurement is made of.
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
# with viewport height exactly as `hud.py` measured the top-left HUD to: the pip runs a median 33
# at 1080 and 34 at 1126.
#
# So this constant is quoted at 1126, which is what DEPLOYMENT resizes to
# (BRAWL_DEPLOYMENT_DESIGN.md 9.4 item 4) -- and the fixture clips are 1080, which is why the
# 33-35 spread straddles both and why nothing looked wrong for a long time. `read_ammo` now
# applies the `REF_H` treatment `hud.py` gives its own constants; see `REF_VIEWPORT_H` for what
# it cost not to.
PIP_WIDTH_PX = 34

# The viewport height `PIP_PITCH_PX` and `PIP_WIDTH_PX` are quoted at. The widget scales with
# viewport height, and until this constant existed the reader simply did not: MEASURED over 334
# adjacent-run gaps, the pitch is **36 px on 1080-normalized footage and 38 px on 1126**, and the
# width 32 against 34 -- both within a percent of the 1126/1080 ratio.
#
# **That is not cosmetic: the lattice error ACCUMULATES.** `slot_fills` tests each run against
# `k * pitch` from an anchor, so at slot 2 a pitch that is 2.5 px small lands 5 px off, and at
# pitch 34 it lands 9 px off -- past `SLOT_TOL_PX`, so the THIRD PIP IS SILENTLY DISCARDED. A full
# clip then reads 2.00. Frame 29 of `bluestacks-example-new.mp4` shows it exactly: three clean
# 31 px runs at 56/90/124 and a reading of (0.91, 0.91, 0.00). Whether it happened depended on
# which run was longest that frame, so the same full clip flickered between 2 and 3 -- which is
# what the deployed shadow's ammo canary kept tripping on.
REF_VIEWPORT_H = 1126

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


def split_merged(runs: list[tuple[int, int]], pip_px: int,
                 pitch: float, tol: float = SLOT_TOL_PX) -> list[tuple[int, int]]:
    """Cut runs that span more than one pip into per-pip pieces.

    Adjacent pips are separated by a few pixels of unpainted track, and when anti-aliasing paints
    that seam the two merge into a single run. MEASURED: runs of 68-70 px against a 34 px pip,
    5% of the impossible reversals in the fixture set.

    The old code handed such a run to one slot and then applied `min(1.0, ...)`, so the merge cost
    a whole pip and the reading dropped by one for a single frame. Cutting on the pitch keeps both.
    """
    out: list[tuple[int, int]] = []
    for start, length in runs:
        if length <= pip_px + tol:
            out.append((start, length))
            continue
        k, off = 0, 0
        while off < length - 1:
            out.append((start + off, min(pip_px, length - off)))
            k += 1
            off = round(k * pitch)
    return out


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
    runs = split_merged(runs, pip_px, pitch, tol)

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


def is_plausible(fills: tuple[float, ...], slack: float = 0.15) -> bool:
    """Does this reading look like an ammo bar at all?

    The bar fills from the LEFT, so the only shapes the widget can draw are some prefix of full
    pips, at most one partial, then empties. A slot fuller than the one to its left is therefore
    not a low reading, it is *not a reading* -- the row that was measured was not the ammo track,
    or an intruder anchored the lattice and shifted every real pip one slot right.

    MEASURED over 4685 fixture frames: 1.77% of readings violate this, and rejecting them removes
    a third of the physically impossible frame-to-frame changes on its own. `slack` admits the
    anti-aliasing spread the module docstring already prices at +/-0.06 per slot, twice over.

    This is the package's own rule applied where it had been skipped: `read_ammo` promises that
    None means "could not read, never a guess", and an impossible slot pattern is a guess.
    """
    return all(b <= a + slack for a, b in zip(fills, fills[1:]))


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

    # Scale the widget constants to THIS frame's viewport. See `REF_VIEWPORT_H`: the pips are
    # 36 px apart at 1080 and 38 at 1126, and a fixed pitch drops the third pip on the smaller one.
    scale = frame.shape[0] / REF_VIEWPORT_H
    pip_px = max(1, round(cfg.hp_ammo_pip_px * scale))
    pitch = PIP_PITCH_PX * scale

    longest = max((length for _, length in best_runs), default=0)
    if longest < pip_px - SLOT_TOL_PX:
        # Nothing here is pip-sized. A 4 px scrap is not a nearly-empty magazine, it is a frame
        # where the track was not found -- MEASURED as 21% of the impossible reversals, every one
        # of them reading a confident 0.00 between two neighbours that agreed on 2.
        return None
    if longest > (MAX_AMMO_SLOTS - 1) * pitch + pip_px + SLOT_TOL_PX:
        # Wider than the whole three-pip track, so it is not the track. This is INSURANCE, not a
        # measured fix -- 0 of 4108 fixture frames trip it -- and it is here because
        # `split_merged` opened the hole it closes: before the split, a 300 px orange band read
        # `min(1.0, 300/34)` in one slot and came out as 1.0; after it, that band would be cut
        # into eight lattice pieces and come out as a confident, plausible-looking 3.0.
        return None

    fills = slot_fills(best_runs, pip_px, pitch=pitch)
    if not is_plausible(fills):
        return None
    ammo = float(sum(fills))
    return AmmoReading(ammo=ammo, frac=ammo / MAX_AMMO_SLOTS, whole=int(ammo),
                       slots=fills, row=best_row + y0, pip_px=longest)
