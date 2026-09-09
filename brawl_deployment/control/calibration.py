"""The measurements behind `data/control_calibration.json`, and the live checks that prove them.

See BRAWL_DEPLOYMENT_DESIGN.md 4.5. Two kinds of thing live here and they are not the same kind:

- **Fits.** Where the buttons are, from pixels. Deterministic, offline-testable, re-runnable.
- **Verifications.** Whether the touches we send have the effect we think. These need a live
  emulator, they emit input, and they are the only reason `scripts/deploy_calibrate.py` exists as
  something separate from `deploy_run.py --dry-run`.

**Screen coordinates are not evidence that a tap landed, and this is the lesson of 6.8.** The
Training Grounds run tapped the calibrated attack centre and 2 of 3 taps consumed no ammo -- not
because the aim was wrong but because a venue-only green button sat there and swallowed the touch.
A check that compares "where we tapped" against "where the button is" would have passed that run
with full marks. So every verification here is defined on a **game-state consequence**: ammo goes
down, the super discharges, the camera moves. Those are venue-independent; geometry is not.

**Nothing here writes the whole calibration file.** `update_calibration` replaces one block at a
time, because only some blocks have a script behind them: the buttons do, and the joystick's
saturation radius was measured by hand in 4.3 with no reader that can reproduce it. Regenerating
the file wholesale would silently replace hand-measured provenance with a default.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .buttons import ATTACK_FIRE

# --- fitting the button table ---------------------------------------------------------------

# How far a re-fit centre may move from the stored one before it is treated as a different button
# rather than the same one, better located. Generous relative to the 1-3 px that centres actually
# disagree by across frame sources (4.5), tight relative to the ~100 px between neighbouring
# buttons -- so the assignment cannot silently swap gadget for attack.
MAX_CENTRE_SHIFT_PX = 40.0

# Frames folded into the temporal median before fitting. The median is what makes this a fit of the
# HUD rather than of whatever sprite happened to be under it: at 20 Hz this is ~1.5 s of play, over
# which the world moves and the buttons do not.
MEDIAN_FRAMES = 31


@dataclass
class ButtonFit:
    """One re-fitted button, in the space of the frames it was fitted on."""
    name: str
    x: float
    y: float
    r: float
    score: float
    shift_px: float            # distance from the stored centre this was matched to
    dr_px: float               # change in radius from the stored one

    def as_json(self, scale: tuple[float, float]) -> dict[str, float]:
        """The stored form: DEVICE pixels, which is what `Calibration.button` hands to a tap.

        `scale` is `Calibration.device_to_viewport`, so this divides. The radius uses the mean of
        the two axes for the same reason `viewport_button` does -- it is one number describing a
        circle, and the two axis ratios differ in the fourth decimal.
        """
        sx, sy = scale
        return {"x": round(self.x / sx, 1), "y": round(self.y / sy, 1),
                "r": round(self.r / ((sx + sy) / 2.0), 1)}


def temporal_median(frames) -> np.ndarray:
    """Median of an iterable of same-shaped BGR frames.

    `gameplay.median_frame` does this for a clip on disk by seeking; a live capture cannot seek,
    so this takes the frames the caller already has. Same statistic, same purpose.
    """
    stack = np.stack([np.asarray(f) for f in frames])
    return np.median(stack, axis=0).astype(np.uint8)


def fit_buttons(median: np.ndarray, expected: dict[str, tuple[float, float, float]],
                *, max_shift: float = MAX_CENTRE_SHIFT_PX) -> dict[str, ButtonFit]:
    """Re-fit the stored buttons on this frame source -> `{name: ButtonFit}`.

    `expected` is `{name: (cx, cy, r)}` in the median's own space -- i.e.
    `Calibration.viewport_button(name)` for every button, when the median is a viewport frame.

    **This is a re-fit, not a discovery, and the difference is what `expected` is for.** Hough
    returns unnamed circles and the map is only unambiguous if you already know roughly where each
    button is; guessing names from geometry ("the leftmost is the super") is how a HUD-layout
    change turns into an agent that taps the gadget. Candidates that land more than `max_shift`
    from every stored centre are dropped rather than assigned to their nearest, so a missing button
    is a missing key and never a wrong one.

    A name absent from the result means "not found here", which is a real answer: the gadget button
    greys out on cooldown and the fit may legitimately lose it on a given median.
    """
    from brawl_vision.gameplay import DEPLOY_RADIUS_PX, calibrate_buttons

    # min_score 0.0: the caller sees every candidate's score and the assignment below is what
    # rejects junk. `MIN_CALIBRATION_SCORE` exists to answer "is this recording a match at all",
    # which is not the question here -- we already know it is.
    candidates = calibrate_buttons(median, max_anchors=12, radius_px=DEPLOY_RADIUS_PX,
                                   min_score=0.0)
    out: dict[str, ButtonFit] = {}
    for c in candidates:
        best, best_d = None, max_shift
        for name, (ex, ey, er) in expected.items():
            d = math.hypot(c["cx"] - ex, c["cy"] - ey)
            if d < best_d:
                best, best_d = name, d
        if best is None or best in out:
            continue        # unmatched, or a weaker candidate for a button already taken
        er = expected[best][2]
        out[best] = ButtonFit(name=best, x=c["cx"], y=c["cy"], r=c["r"], score=c["score"],
                              shift_px=round(best_d, 1), dr_px=round(c["r"] - er, 1))
    return out


# --- verifying that a tap lands -------------------------------------------------------------

# Ammo at or above this counts as "loaded enough to spend one". Near-full rather than >= 1.0 so the
# trough below is unambiguous: starting at 1.2 and reading 0.4 could be a shot or a misread pip,
# starting at 3.0 and reading 2.0 could only be a shot.
AMMO_ARMED = 2.8
# Longest wait for the bar to refill between trials. Mortis reloads one pip in ~1.7 s, so a full
# bar from empty is ~5 s; this is that plus slack, and running out of it is reported rather than
# raised -- "the bar never refilled" is a finding about the venue, not a crash.
ARM_TIMEOUT_S = 8.0
# How long to watch for the drop after the tap. The swing has an animation and the reload starts
# refilling almost immediately afterwards, so the signal is a TROUGH, not a level.
TROUGH_WINDOW_S = 0.9
# Ammo the trough must fall by. **One shot does not present as 1.0**, which is the whole reason
# there is slack here: the bar starts reloading the instant the swing spends it, so the trough a
# sampler can actually see is already partly refilled -- ~0.88 at Mortis's 1.7 s per pip against
# the window below. 0.75 covers that plus the pip reader's own quantisation, and is still three
# times the largest drift a non-firing bar could show over the same window (a reload only goes up).
MIN_DROP = 0.75
# Gap between the down and the up, mirroring what the loop does: `Buttons.tap` sends the down and
# leaves the release to `settle()` on the next perception tick, ~50 ms later.
SETTLE_S = 0.06
# Gap between ammo samples inside the trough window.
SAMPLE_S = 0.05
# Sampling period for `verify_move`, in seconds. The deployed 20 Hz tick, deliberately -- see that
# function on why a per-frame bound is only meaningful at the rate the loop runs.
SAMPLE_PERIOD_S = 0.05


@dataclass
class TapTrial:
    """One press, and what the game did about it."""
    before: float | None = None
    trough: float | None = None
    samples: int = 0
    note: str = ""

    @property
    def drop(self) -> float | None:
        if self.before is None or self.trough is None:
            return None
        return self.before - self.trough

    @property
    def valid(self) -> bool:
        """Whether this trial is evidence either way. An unreadable bar is neither."""
        return self.drop is not None

    @property
    def ok(self) -> bool:
        return self.valid and self.drop >= MIN_DROP


@dataclass
class TapReport:
    point: tuple[float, float]
    action: int
    trials: list[TapTrial] = field(default_factory=list)

    @property
    def valid(self) -> list[TapTrial]:
        return [t for t in self.trials if t.valid]

    @property
    def ok(self) -> bool:
        """**Every** readable trial must have consumed ammo, and there must be at least two.

        Not a majority. A tap either reaches the game or it does not; there is no mechanism that
        makes it land two times in three, so a mixed result is exactly the 6.8 failure -- a button
        that swallows the touch some of the time -- and is worth failing on. Two is the floor
        because one trial cannot distinguish a landed tap from a coincidence.
        """
        v = self.valid
        return len(v) >= 2 and all(t.ok for t in v)

    def summary(self) -> str:
        parts = []
        for t in self.trials:
            parts.append("unread" if not t.valid else f"{t.before:.1f}->{t.trough:.1f}")
        verdict = "PASS" if self.ok else "FAIL"
        return (f"{verdict}  tap ({self.point[0]:.0f}, {self.point[1]:.0f}) action={self.action}  "
                + "  ".join(parts))


def verify_tap(read_ammo, buttons, *, action: int = ATTACK_FIRE, trials: int = 3,
               sleep=time.sleep, now=time.perf_counter) -> TapReport:
    """Press `action` `trials` times and check the ammo bar goes down. The venue-independent check.

    `read_ammo` is a zero-argument callable returning ammo in 0..3, or `None` when the bar could
    not be read. It is a callable rather than a frame because the caller owns capture, detection
    and the hero box -- and because a test can then drive this whole state machine in microseconds.

    Each trial: wait for the bar to be near-full, press, then sample for `TROUGH_WINDOW_S` and keep
    the **minimum**. The minimum rather than a single delayed reading, because the swing animation
    delays the spend and the reload begins undoing it -- a fixed delay can sample before the drop
    or after it has been partly refilled, and both look like a failed tap.

    The tap's release is sent by this function rather than left to the caller: `Buttons.tap` only
    presses, and a calibration run has no perception tick to call `settle()` on.

    `sleep` and `now` are injectable as a pair. Not for mocking's sake -- the windows here are
    tuned to a reload rate and a swing animation, and "is the trough window long enough" is a
    property worth a test rather than a comment. A fake clock lets one run in microseconds; a
    `sleep` seam alone would leave the tests spinning for the real 0.9 s a trial.
    """
    report = TapReport(point=buttons.attack if action == ATTACK_FIRE else buttons.super_,
                       action=action)
    for _ in range(trials):
        trial = TapTrial()
        report.trials.append(trial)

        deadline = now() + ARM_TIMEOUT_S
        while True:
            value = read_ammo()
            if value is not None and value >= AMMO_ARMED:
                trial.before = float(value)
                break
            if now() >= deadline:
                trial.note = ("ammo bar never reached full" if value is not None
                              else "ammo bar unreadable")
                break
            sleep(SAMPLE_S)
        if trial.before is None:
            continue

        buttons.tap(action)
        sleep(SETTLE_S)
        buttons.settle()

        end = now() + TROUGH_WINDOW_S
        while now() < end:
            value = read_ammo()
            if value is not None:
                trial.samples += 1
                trial.trough = float(value) if trial.trough is None else min(trial.trough,
                                                                            float(value))
            sleep(SAMPLE_S)
        if trial.trough is None:
            trial.note = "ammo bar unreadable after the tap"
    return report


# --- verifying that movement lands, and that odometry sees it -------------------------------

# Cosine between the commanded direction and the odometry displacement for a trial to pass. 0.85 is
# ~32 degrees, which admits terrain deflection and a bot bumping the hero, and rejects a sign
# error -- the failure this exists to catch -- which lands at -1.0 on one axis.
MIN_COS = 0.85
# Tiles the HERO must actually travel for a trial to mean anything. Below this the direction is
# noise: a hero walled in, or a venue that does not scroll (Training Grounds reads a correct ~0,
# which is exactly why it could not answer this question -- 10.11).
MIN_TRAVEL_TILES = 1.0
# World tiles of displacement from the baseline that count as "the hero has started moving", for
# `MoveTrial.onset_seconds`. 0.25 is comfortably above the per-sample jitter the first live run
# showed on a standing hero (max single-frame camera shift 0.00-0.20 t) and well under one sample
# of real walking (2.73 tiles/s * 0.05 s = 0.14 t/sample, so ~2 samples of genuine motion).
ONSET_TILES = 0.25


@dataclass
class MoveTrial:
    """One bin held for `hold_seconds`, measured in the WORLD frame.

    **`observed` is the hero's world displacement, not the camera's**, and the distinction is the
    whole reason this dataclass carries three vectors instead of one. The world frame is
    `camera_relative + odometry.position_tiles` -- the project's own rule -- and a check that
    watches only the second term cannot tell "the hero did not move" from "the hero moved while
    the camera stayed put".

    That is not hypothetical. The first live run (2026-09-09) failed both horizontal bins with
    x displacement of exactly +0.00 and -0.00 tiles while both vertical bins were perfect, which
    is the signature of a camera that is not panning in x -- and camera panning is a property of
    the map and where you are standing, not of whether the joystick works. Splitting the two terms
    turns that from a mystery into a reading.
    """

    bin: int
    commanded: tuple[float, float]
    observed: tuple[float, float]        # hero, WORLD frame = camera + camera-relative
    camera: tuple[float, float]          # the odometry term alone
    relative: tuple[float, float]        # the hero-on-screen term alone
    travel: float
    cos: float
    max_shift_tiles: float      # largest single-FRAME shift seen, against the 2.0 bound
    frames: int
    lost: int                   # frames where odometry.status was not "ok"
    seen: int = 0               # samples the hero detector actually found the hero in
    # (seconds_since_apply, world_x, world_y) per sample, including the baseline at t=0. Kept so
    # ACTUATION LATENCY is measurable: total travel alone cannot separate "moved slower than the
    # kit constant says" from "started late", and those have different causes and fixes.
    series: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.travel >= MIN_TRAVEL_TILES and self.cos >= MIN_COS

    @property
    def onset_seconds(self) -> float | None:
        """Seconds from `joystick.apply` to the hero first moving `ONSET_TILES`, or None.

        **This is end-to-end actuation latency**, and it is not free: `adb sendevent` has to cross
        the ADB socket, the emulator has to inject the touch, and the game has to accept it and
        accelerate the hero. The sim has none of it -- `movement.py` applies a velocity the same
        tick the action arrives -- so anything here is a deploy-only discrepancy that the shadow's
        dead-reckoned position does not model.

        The first live run implied ~0.25-0.28 s of it from travel distance alone (2.52-2.60 tiles
        against the 3.28 that `move_speed: 2.73` predicts for a 1.2 s hold). This measures it
        rather than inferring it from a total that acceleration and a wall can equally explain.
        """
        if not self.series:
            return None
        _, ox, oy = self.series[0]
        for t, x, y in self.series[1:]:
            if math.hypot(x - ox, y - oy) >= ONSET_TILES:
                return t
        return None

    @property
    def camera_only(self) -> bool:
        """Did the camera do all the work? Then the hero-relative term added nothing, which is
        normal: the camera follows the hero, so a free-walking hero barely moves on screen."""
        return math.hypot(*self.camera) > 0.5 * self.travel

    def diagnosis(self) -> str:
        """One line naming WHY a trial failed, in terms the operator can act on."""
        if self.ok:
            return "ok"
        if self.seen == 0:
            return "FAIL (the hero was never detected -- this measured nothing)"
        if self.travel < MIN_TRAVEL_TILES:
            cam, rel = math.hypot(*self.camera), math.hypot(*self.relative)
            if cam < 0.2 and rel < 0.2:
                return ("FAIL (hero did not move: walled in, or the input did not land -- "
                        "retry from open ground before suspecting the joystick)")
            return f"FAIL (moved only {self.travel:.2f} t; camera {cam:.2f}, on-screen {rel:.2f})"
        return f"FAIL (moved {self.travel:.2f} t but {math.degrees(math.acos(max(-1.0, min(1.0, self.cos)))):.0f} deg off)"


def verify_move(sample, joystick, *, bins, hold_seconds: float = 1.2,
                period: float = SAMPLE_PERIOD_S, sleep=time.sleep,
                now=time.perf_counter) -> list[MoveTrial]:
    """Drive each bin in `bins` and check the HERO moves the way it was told to.

    `sample` is a zero-argument callable returning `(odometry_result, hero_tile_or_None)` for a
    freshly grabbed frame, where `hero_tile` is the hero's CAMERA-RELATIVE tile position from
    `project.to_tiles`. `joystick` is the real `Joystick`, and the commanded direction is read back
    out of it via `contact_point` rather than recomputed here -- so a sign error in the joystick is
    a **failure** of this check rather than something it reproduces and agrees with.

    **Why the hero and not the camera.** This measured `odometry.position_tiles` alone until a live
    run showed why that is not enough: both horizontal bins failed with x displacement of exactly
    +0.00 and -0.00 tiles while both vertical bins were perfect. Odometry tracks the CAMERA, the
    camera pans only when the map is bigger than the screen in that axis, and whether it can pan in
    x depends on the map and where the hero is standing. So a correct ~0 from odometry is
    indistinguishable from a joystick that does nothing -- the same trap Training Grounds set for
    the tap check (6.8), in a venue that looked valid.

    The world frame is `camera_relative + odometry.position_tiles`, which is the project's rule
    everywhere else, and it is right here too: it is correct whether the camera pans or not. Both
    terms are kept on the trial so a failure names itself instead of needing another run.

    `max_shift_tiles` is the largest single-frame CAMERA displacement, checked against
    `configs/vision.yaml`'s `odometry.max_shift_tiles: 2.0` -- the per-frame bound the whole
    perception cadence exists to satisfy (6.13). A live run is the only place that number can be
    observed against a real capture rate rather than argued about.

    **`period` is why this paces itself instead of sampling flat out, and it is not politeness.**
    A per-FRAME shift bound is only meaningful at the rate the loop actually runs: sampling as fast
    as the capture allows would understate it and sampling slower would overstate it, and either
    way the number would not be about the deployed cadence.
    """
    trials: list[MoveTrial] = []
    joystick.acquire()
    for move in bins:
        cx, cy = joystick.contact_point(move)
        dx, dy = cx - joystick.anchor[0], cy - joystick.anchor[1]
        norm = math.hypot(dx, dy) or 1.0
        commanded = (dx / norm, dy / norm)

        t_apply = now()
        joystick.apply(move)
        odo, hero = sample()
        cam_origin = cam_last = odo.position_tiles
        # The hero-relative baseline waits for the first frame the hero is actually detected, and
        # `seen` counts them: a trial where the detector never found the hero measured nothing at
        # all, which must not read as "it did not move".
        rel_origin = rel_last = hero
        seen = 1 if hero is not None else 0
        max_shift, frames, lost = 0.0, 0, 0
        rel0 = hero or (0.0, 0.0)
        series = [(0.0, cam_origin[0] + rel0[0], cam_origin[1] + rel0[1])]
        end = now() + hold_seconds
        while now() < end:
            t0 = now()
            odo, hero = sample()
            frames += 1
            if odo.status != "ok":
                lost += 1
            max_shift = max(max_shift, math.hypot(*odo.delta_tiles))
            cam_last = odo.position_tiles
            if hero is not None:
                seen += 1
                rel_last = hero
                if rel_origin is None:
                    rel_origin = hero
            r = rel_last if rel_last is not None else (0.0, 0.0)
            series.append((t0 - t_apply,
                           odo.position_tiles[0] + r[0], odo.position_tiles[1] + r[1]))
            sleep(max(0.0, period - (now() - t0)))
        joystick.apply(0)
        sleep(0.3)          # let the hero coast to a stop before the next bin's baseline

        camera = (cam_last[0] - cam_origin[0], cam_last[1] - cam_origin[1])
        if rel_origin is None or rel_last is None:
            relative = (0.0, 0.0)
        else:
            relative = (rel_last[0] - rel_origin[0], rel_last[1] - rel_origin[1])
        ox, oy = camera[0] + relative[0], camera[1] + relative[1]
        travel = math.hypot(ox, oy)
        cos = (commanded[0] * ox + commanded[1] * oy) / travel if travel > 1e-6 else 0.0
        trials.append(MoveTrial(bin=move, commanded=commanded, observed=(ox, oy),
                                camera=camera, relative=relative,
                                travel=travel, cos=cos, max_shift_tiles=max_shift,
                                frames=frames, lost=lost, seen=seen, series=series))
    return trials


# --- writing the file -----------------------------------------------------------------------

def update_calibration(path: Path | str, *, blocks: dict, provenance: dict[str, list[str]] | None
                       = None) -> dict:
    """Replace whole top-level blocks of the calibration JSON, keeping the rest verbatim.

    `blocks` is `{key: value}` for the keys this run actually measured. `provenance` is
    `{key: [comment lines]}`, written into that block's `_comment` -- because the comment describes
    how the numbers below it were obtained, and leaving a stale one beside fresh numbers is worse
    than having none.

    Returns the written document, so a caller can diff it against what it read.
    """
    path = Path(path)
    doc = json.loads(path.read_text())
    for key, value in blocks.items():
        if isinstance(value, dict) and provenance and key in provenance:
            value = {"_comment": provenance[key], **value}
        doc[key] = value
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return doc
