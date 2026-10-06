"""`brawl_deployment/control/calibration.py` -- the fits and the live checks, offline.

Every verification in that module exists to be run against a live emulator, which is exactly why
it needs tests that are not: the whole point of `verify_tap` is that it says FAIL on the §6.8
Training Grounds result, and the only way to know it does is to hand it that result.

The clock is faked. `verify_tap`'s windows are tuned to a reload rate and a swing animation, so
"the trough window is long enough" and "the arm timeout is respected" are properties worth
asserting rather than comments -- and a real clock would spin for 0.9 s a trial to assert them.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from brawl_deployment.control import ATTACK_FIRE, ATTACK_SUPER, SLOT_TAP, Buttons, NullBackend
from brawl_deployment.control import calibration as cal
from brawl_deployment.window import WindowFault


# --- a fake clock ---------------------------------------------------------------------------

class Clock:
    """`now`/`sleep` pair. `sleep` is the only thing that advances it, so a loop that forgets to
    sleep spins forever here -- which is the correct outcome for a test, not a hazard."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class _Buttons:
    """`Buttons` reduced to what `verify_tap` touches, recording the press/settle order. For the
    trials that must never press; the ones that do use the real class (`_game_buttons`)."""

    def __init__(self, attack=(1690.0, 594.0), super_=(1462.5, 1000.5)):
        self.attack, self.super_ = attack, super_
        self.events = []

    def press(self, action, bearing):
        self.events.append(("press", action))
        return action != 0

    def settle(self):
        self.events.append(("settle", None))


class _Game:
    """A pretend Mortis: ammo reloads against the fake clock, and a press spends one after the
    swing animation -- counted from the LIFT, which is the touch that fires.

    A scripted list of readings would not exercise the thing worth exercising. `verify_tap` waits
    for the bar to refill between trials and hunts a trough that the reload immediately starts
    filling back in -- both of those are interactions with a reloading bar, and a fixed script
    cannot get them wrong in the way a real bar can.
    """

    RELOAD_S = 1.7            # Mortis, one pip
    SWING_S = 0.15            # the tap lands, then the swing spends the ammo

    def __init__(self, clock, *, fires=None, readable=True, ammo=3.0):
        self.clock, self.readable = clock, readable
        # `fires[i]` says whether the i-th tap reaches the game. All-True is a working setup;
        # a False is the 6.8 failure -- a button sat over the tap point and swallowed it.
        self.fires = list(fires) if fires is not None else None
        self._ammo, self._t, self._pending, self.taps = ammo, clock.t, None, 0

    def _advance(self):
        now = self.clock.t
        dt, self._t = now - self._t, now
        if self._pending is not None and now >= self._pending:
            self._ammo, self._pending = max(0.0, self._ammo - 1.0), None
        self._ammo = min(3.0, self._ammo + dt / self.RELOAD_S)

    def read(self):
        self._advance()
        return round(self._ammo, 4) if self.readable else None

    def fire(self):
        self._advance()
        lands = True if self.fires is None else (self.taps < len(self.fires)
                                                 and self.fires[self.taps])
        self.taps += 1
        if lands and self._ammo >= 1.0:
            self._pending = self.clock.t + self.SWING_S


class _GameBackend(NullBackend):
    """A touchscreen wired to a `_Game`. Lifting SLOT_TAP is what fires, as in the real game: a
    double that fired on the down would let a `verify_tap` that never lifts pass."""

    def __init__(self, game):
        super().__init__()
        self.game = game

    def up(self, slot):
        if slot == SLOT_TAP and slot in self.contacts:
            self.game.fire()
        super().up(slot)


def _game_buttons(game) -> Buttons:
    """The REAL `Buttons` on a backend wired to the game, so the touches under test are the ones
    the loop sends."""
    return Buttons(_GameBackend(game), attack=(1690.0, 594.0), super_=(1462.5, 1000.5),
                   gadget=(1559.9, 901.1))


# --- temporal_median ------------------------------------------------------------------------

def test_temporal_median_is_the_median_not_the_mean():
    # Three frames, one an outlier. The mean would be dragged; the median is not -- which is the
    # whole reason the HUD survives and the moving world does not.
    frames = [np.full((4, 4, 3), 10, np.uint8),
              np.full((4, 4, 3), 12, np.uint8),
              np.full((4, 4, 3), 200, np.uint8)]
    out = cal.temporal_median(frames)
    assert out.dtype == np.uint8
    assert (out == 12).all()


def test_temporal_median_keeps_a_constant_hud_through_a_moving_world():
    base = np.zeros((60, 60, 3), np.uint8)
    cv2.circle(base, (30, 30), 10, (255, 255, 255), 2)
    frames = []
    for i in range(9):
        f = base.copy()
        cv2.rectangle(f, (i * 6, 0), (i * 6 + 5, 59), (200, 0, 0), -1)   # a "world" sweeping past
        frames.append(f)
    med = cal.temporal_median(frames)
    assert med[30, 20, 0] == base[30, 20, 0]          # the ring is still there
    assert med[0, 20, 0] == 0                          # the sweep passed here and left nothing
    assert not (med == 200).any()                      # ...and nowhere else either


# --- fit_buttons ----------------------------------------------------------------------------

def _candidates(monkeypatch, rows):
    import brawl_vision.gameplay as gameplay
    monkeypatch.setattr(gameplay, "calibrate_buttons", lambda *a, **k: list(rows))


def test_fit_buttons_assigns_by_nearest_expected_centre(monkeypatch):
    _candidates(monkeypatch, [
        {"score": 0.9, "cx": 1748.0, "cy": 1042.0, "r": 36.0},     # near attack
        {"score": 0.8, "cx": 1524.0, "cy": 1043.0, "r": 48.0},     # near super
    ])
    expected = {"attack": (1747.0, 1042.0, 36.6), "super": (1524.0, 1043.0, 48.5),
                "gadget": (1626.0, 939.0, 34.6)}
    fits = cal.fit_buttons(np.zeros((10, 10, 3), np.uint8), expected)
    assert set(fits) == {"attack", "super"}
    assert fits["attack"].shift_px == pytest.approx(1.0, abs=0.05)
    assert fits["super"].dr_px == pytest.approx(-0.5, abs=0.05)


def test_fit_buttons_drops_a_candidate_that_matches_nothing(monkeypatch):
    # A map decoration, or a spell effect. Assigning it to its nearest button is how a HUD change
    # becomes an agent that taps the gadget; a missing key is the safe answer.
    _candidates(monkeypatch, [{"score": 0.95, "cx": 300.0, "cy": 1000.0, "r": 40.0}])
    fits = cal.fit_buttons(np.zeros((10, 10, 3), np.uint8),
                           {"attack": (1747.0, 1042.0, 36.6)})
    assert fits == {}


def test_fit_buttons_keeps_the_best_candidate_for_a_contested_button(monkeypatch):
    # `calibrate_buttons` returns best-score-first, so the first match wins and the second is
    # dropped rather than overwriting it.
    _candidates(monkeypatch, [
        {"score": 0.90, "cx": 1747.0, "cy": 1042.0, "r": 36.0},
        {"score": 0.40, "cx": 1752.0, "cy": 1046.0, "r": 30.0},
    ])
    fits = cal.fit_buttons(np.zeros((10, 10, 3), np.uint8),
                           {"attack": (1747.0, 1042.0, 36.6)})
    assert list(fits) == ["attack"]
    assert fits["attack"].score == 0.90


def test_fit_buttons_respects_the_shift_gate(monkeypatch):
    _candidates(monkeypatch, [{"score": 0.9, "cx": 1747.0 + 39.0, "cy": 1042.0, "r": 36.0}])
    expected = {"attack": (1747.0, 1042.0, 36.6)}
    assert "attack" in cal.fit_buttons(np.zeros((10, 10, 3), np.uint8), expected)
    assert cal.fit_buttons(np.zeros((10, 10, 3), np.uint8), expected, max_shift=20.0) == {}


def test_deploy_radius_band_finds_a_button_the_default_band_cannot():
    """§5's finding, as a test: the search range is what blocked auto-calibration, not scoring.

    The shipped `RADIUS_PX = (40, 115)` never proposes a 33 px gadget to HoughCircles, so
    `calibrate_buttons` returns nothing at all on this HUD. `DEPLOY_RADIUS_PX` is that fix, and
    parameterising the band rather than copying the Hough call is what keeps the two in one place.
    """
    from brawl_vision.gameplay import DEPLOY_RADIUS_PX, RADIUS_PX, calibrate_buttons

    frame = np.full((1126, 2002, 3), 40, np.uint8)
    cv2.circle(frame, (1620, 940), 33, (230, 230, 230), 5)      # inside BUTTON_ROI
    assert calibrate_buttons(frame, radius_px=RADIUS_PX, min_score=0.0) == []
    found = calibrate_buttons(frame, radius_px=DEPLOY_RADIUS_PX, min_score=0.0)
    assert found, "the lowered band should propose a 33 px ring"
    assert abs(found[0]["cx"] - 1620) <= 8 and abs(found[0]["cy"] - 940) <= 8


def test_button_fit_round_trips_through_the_device_conversion():
    """`as_json` must invert `Calibration.viewport_button`, or a re-fit walks the tap point."""
    from brawl_deployment.match_state import Calibration

    c = Calibration.load()
    vx, vy, vr = c.viewport_button("super")
    fit = cal.ButtonFit(name="super", x=vx, y=vy, r=vr, score=1.0, shift_px=0.0, dr_px=0.0)
    stored = fit.as_json(c.device_to_viewport)
    assert stored["x"] == pytest.approx(c.buttons["super"]["x"], abs=0.1)
    assert stored["y"] == pytest.approx(c.buttons["super"]["y"], abs=0.1)
    assert stored["r"] == pytest.approx(c.buttons["super"]["r"], abs=0.1)


# --- verify_tap -----------------------------------------------------------------------------

def test_verify_tap_passes_when_ammo_drops():
    clock = Clock()
    game = _Game(clock)
    report = cal.verify_tap(game.read, _game_buttons(game), trials=3,
                            sleep=clock.sleep, now=clock.now)
    assert report.ok
    assert game.taps == 3
    # A single shot does NOT present as a clean 1.0. The bar starts reloading the instant the
    # swing spends it, so by the time the sampler sees the trough some of it is already back --
    # ~0.88 against this reload rate. That gap is what MIN_DROP's slack is for, and asserting
    # 1.0 here would be asserting a bar that does not reload.
    assert all(cal.MIN_DROP <= t.drop <= 1.0 for t in report.trials)
    assert "PASS" in report.summary()


def test_verify_tap_fails_the_training_grounds_result():
    """§6.8: the tap landed on a real button that swallowed it, and ammo never moved. This is the
    single result the whole function exists to say FAIL to."""
    clock = Clock()
    game = _Game(clock, fires=[False, False, False])
    report = cal.verify_tap(game.read, _game_buttons(game), trials=3,
                            sleep=clock.sleep, now=clock.now)
    assert not report.ok
    assert all(t.drop == pytest.approx(0.0, abs=1e-6) for t in report.trials)
    assert "FAIL" in report.summary()


def test_verify_tap_rejects_a_mixed_result_rather_than_taking_the_majority():
    # 2 of 3 is exactly what Training Grounds produced. A tap has no mechanism for landing
    # two times in three, so a majority rule would have called that setup good.
    clock = Clock()
    game = _Game(clock, fires=[True, False, False])
    report = cal.verify_tap(game.read, _game_buttons(game), trials=3,
                            sleep=clock.sleep, now=clock.now)
    assert [t.ok for t in report.valid] == [True, False, False]
    assert not report.ok


def test_verify_tap_takes_the_trough_not_the_last_sample():
    """The reload starts refilling almost immediately, so a fixed delayed sample can land after
    the bar has partly recovered and read that as a failed tap."""
    clock = Clock()
    game = _Game(clock)
    trial = cal.verify_tap(game.read, _game_buttons(game), trials=1,
                           sleep=clock.sleep, now=clock.now).trials[0]
    assert trial.trough == pytest.approx(2.0, abs=0.05)
    # By the end of the window the bar has already refilled well past the trough -- a single
    # delayed sample would have read this as a partial drop and failed a working setup.
    assert game.read() > trial.trough + 0.3
    assert trial.ok


def test_verify_tap_waits_for_the_bar_to_refill_before_pressing():
    clock = Clock()
    game = _Game(clock, ammo=0.4)
    report = cal.verify_tap(game.read, _game_buttons(game), trials=1,
                            sleep=clock.sleep, now=clock.now)
    assert game.taps == 1
    assert report.trials[0].before >= cal.AMMO_ARMED    # armed, not the 0.4 it started at
    assert clock.t >= (cal.AMMO_ARMED - 0.4) * _Game.RELOAD_S
    assert report.trials[0].ok


def test_verify_tap_reports_a_bar_that_never_refills_without_pressing():
    clock = Clock()
    buttons = _Buttons()
    report = cal.verify_tap(lambda: 0.0, buttons, trials=1, sleep=clock.sleep, now=clock.now)
    assert buttons.events == []                # nothing was tapped at 0 ammo
    assert not report.valid
    assert not report.ok
    assert "never reached full" in report.trials[0].note


def test_verify_tap_distinguishes_an_unreadable_bar_from_a_failed_tap():
    clock = Clock()
    report = cal.verify_tap(lambda: None, _Buttons(), trials=2, sleep=clock.sleep, now=clock.now)
    assert report.valid == []                  # neither trial is evidence either way
    assert not report.ok
    assert all("unreadable" in t.note for t in report.trials)
    assert "unread" in report.summary()


def test_verify_tap_needs_two_readable_trials():
    clock = Clock()
    game = _Game(clock)
    report = cal.verify_tap(game.read, _game_buttons(game), trials=1,
                            sleep=clock.sleep, now=clock.now)
    assert report.trials[0].ok                 # the one trial passed
    assert not report.ok                       # ...and one is not enough


def test_verify_tap_walks_every_press_to_its_lift():
    """Each trial is the loop's own touch sequence -- down, drag, lift -- and nothing is left
    down at the end. A trial whose press never lifts never fires, and would read as the 6.8
    failure on a setup that works."""
    clock = Clock()
    game = _Game(clock)
    buttons = _game_buttons(game)
    cal.verify_tap(game.read, buttons, trials=3, sleep=clock.sleep, now=clock.now)
    kinds = [e[0] for e in buttons.backend.log]
    assert kinds == ["down", "move", "up"] * 3
    assert not buttons.is_held and SLOT_TAP not in buttons.backend.contacts


def test_verify_tap_drags_along_the_bearing_it_was_given():
    """Never a bare tap: the game auto-aims those, and the loop never sends one, so a tap passing
    here would say nothing about the path the loop actually uses."""
    clock = Clock()
    game = _Game(clock)
    buttons = _game_buttons(game)
    cal.verify_tap(game.read, buttons, bearing=math.pi / 2, trials=1,
                   sleep=clock.sleep, now=clock.now)
    (drag,) = [e for e in buttons.backend.log if e[0] == "move"]
    ax, ay = buttons.attack
    assert drag[2:] == pytest.approx((ax, ay + buttons.aim_radius_px), abs=1e-6)


def test_verify_tap_reports_the_point_for_the_action_it_was_given():
    clock = Clock()
    game = _Game(clock)
    buttons = _game_buttons(game)
    fire = cal.verify_tap(game.read, buttons, action=ATTACK_FIRE, trials=1,
                          sleep=clock.sleep, now=clock.now)
    sup = cal.verify_tap(game.read, buttons, action=ATTACK_SUPER, trials=1,
                         sleep=clock.sleep, now=clock.now)
    assert fire.point == buttons.attack
    assert sup.point == buttons.super_
    downs = [e[2:] for e in buttons.backend.log if e[0] == "down"]
    assert downs == [buttons.attack, buttons.super_]


# --- verify_move ----------------------------------------------------------------------------

class _Odo:
    """`OdometryResult` reduced to the three fields `verify_move` reads."""

    def __init__(self, position_tiles, delta_tiles=(0.0, 0.0), status="ok"):
        self.position_tiles = position_tiles
        self.delta_tiles = delta_tiles
        self.status = status


class _Joystick:
    """The real `Joystick`'s geometry contract, with a switchable y sign so the check can be
    shown to catch a flip rather than share one."""

    def __init__(self, y_sign=1.0, n_bins=16):
        import math as _m
        self._m, self.anchor, self.n_bins = _m, (345.6, 669.6), n_bins
        self.y_sign, self.radius_px = y_sign, 110.0
        self.calls = []

    def contact_point(self, move):
        if move <= 0:
            return self.anchor
        theta = (move - 1) * (2.0 * self._m.pi / self.n_bins)
        return (self.anchor[0] + self.radius_px * self._m.cos(theta),
                self.anchor[1] + self.y_sign * self.radius_px * self._m.sin(theta))

    def acquire(self):
        self.calls.append(("acquire", None))

    def apply(self, move):
        self.calls.append(("apply", move))

    def release(self):
        self.calls.append(("release", None))


def _walker(steps, per_frame, status="ok", hero_per_frame=(0.0, 0.0), hero=(0.0, 0.0)):
    """A `sample` returning `(odometry, hero_camera_relative_tile)`.

    `per_frame` moves the CAMERA and `hero_per_frame` moves the hero ACROSS the screen. The default
    -- camera moves, hero stays put on screen -- is the normal case, because the camera follows the
    hero. `hero=None` models a frame where the detector found nothing.
    """
    state = {"p": (0.0, 0.0), "h": hero, "n": 0}

    def sample():
        moving = state["n"] < steps
        if moving:
            state["p"] = (state["p"][0] + per_frame[0], state["p"][1] + per_frame[1])
            if state["h"] is not None:
                state["h"] = (state["h"][0] + hero_per_frame[0],
                              state["h"][1] + hero_per_frame[1])
            state["n"] += 1
        delta = per_frame if moving else (0.0, 0.0)
        return _Odo(state["p"], delta, status), state["h"]
    return sample


def test_verify_move_passes_when_the_camera_follows_the_command():
    clock = Clock()
    joy = _Joystick()
    trials = cal.verify_move(_walker(50, (0.2, 0.0)), joy, bins=(1,), hold_seconds=0.5,
                             sleep=clock.sleep, now=clock.now)
    t = trials[0]
    assert t.cos == pytest.approx(1.0, abs=1e-6)
    assert t.travel > cal.MIN_TRAVEL_TILES
    assert t.ok


def test_verify_move_catches_a_y_flip():
    """The failure this whole check exists for. The joystick is told to go DOWN (+y) and the
    world moves UP; cos lands at -1 and the trial fails."""
    clock = Clock()
    joy = _Joystick()
    trials = cal.verify_move(_walker(50, (0.0, -0.2)), joy, bins=(5,), hold_seconds=0.5,
                             sleep=clock.sleep, now=clock.now)
    assert trials[0].commanded[1] == pytest.approx(1.0, abs=1e-6)     # bin 5 is +y, down
    assert trials[0].cos == pytest.approx(-1.0, abs=1e-6)
    assert not trials[0].ok


def test_verify_move_reads_the_commanded_direction_out_of_the_joystick():
    """A joystick with a sign error must FAIL this check, not be agreed with. Recomputing the
    direction here instead of asking the joystick is what would make that impossible to catch."""
    clock = Clock()
    flipped = _Joystick(y_sign=-1.0)
    trials = cal.verify_move(_walker(50, (0.0, 0.2)), flipped, bins=(5,), hold_seconds=0.5,
                             sleep=clock.sleep, now=clock.now)
    assert trials[0].commanded[1] == pytest.approx(-1.0, abs=1e-6)
    assert trials[0].cos == pytest.approx(-1.0, abs=1e-6)
    assert not trials[0].ok


def test_verify_move_rejects_a_venue_that_does_not_scroll():
    """Training Grounds, precisely: odometry is correct and reads ~0, and a direction fitted to
    ~0 travel is noise. §10.11's reason for needing a real match."""
    clock = Clock()
    trials = cal.verify_move(_walker(50, (0.001, 0.0)), _Joystick(), bins=(1,), hold_seconds=0.5,
                             sleep=clock.sleep, now=clock.now)
    assert trials[0].travel < cal.MIN_TRAVEL_TILES
    assert not trials[0].ok


def test_a_camera_that_cannot_pan_sideways_is_not_a_movement_failure():
    """The 2026-09-09 live run. Both horizontal bins came back at exactly 0.00 tiles of camera
    displacement while both vertical bins were perfect -- which is what a map whose width fits the
    screen looks like, not a broken joystick. Odometry tracks the CAMERA; the hero still walked.

    Measuring `camera_relative + odometry.position_tiles` -- the world frame the rest of the
    project already uses -- is right whether the camera pans or not."""
    clock = Clock()
    # Camera pinned, hero crossing the screen: exactly the clamped-axis case.
    sample = _walker(50, (0.0, 0.0), hero_per_frame=(0.2, 0.0))
    t = cal.verify_move(sample, _Joystick(), bins=(1,), hold_seconds=0.5,
                        sleep=clock.sleep, now=clock.now)[0]
    assert t.camera == pytest.approx((0.0, 0.0))
    assert t.relative[0] > cal.MIN_TRAVEL_TILES
    assert t.ok and t.cos == pytest.approx(1.0, abs=1e-6)
    assert not t.camera_only


def test_a_hero_that_never_moves_still_fails_and_says_which_kind_of_nothing():
    """The other half: if BOTH terms are ~0 the hero genuinely did not move, and the operator
    needs to be told it could be a wall rather than sent to debug the joystick."""
    clock = Clock()
    t = cal.verify_move(_walker(50, (0.0, 0.0)), _Joystick(), bins=(1,), hold_seconds=0.5,
                        sleep=clock.sleep, now=clock.now)[0]
    assert not t.ok
    assert "walled in" in t.diagnosis()


def test_a_trial_that_never_saw_the_hero_reports_that_rather_than_no_movement():
    """A detector miss for the whole trial measured nothing at all. Reporting it as "did not move"
    would send the operator after an input bug that the run has no evidence for."""
    clock = Clock()
    t = cal.verify_move(_walker(50, (0.0, 0.0), hero=None), _Joystick(), bins=(1,),
                        hold_seconds=0.5, sleep=clock.sleep, now=clock.now)[0]
    assert t.seen == 0 and not t.ok
    assert "never detected" in t.diagnosis()


def test_the_onset_measures_when_motion_STARTED_not_how_far_it_got():
    """Travel alone cannot separate "started late" from "moved slower than the kit constant" --
    a wall explains a short trial just as well. The first live run showed 2.52-2.60 tiles against
    the 3.28 that `move_speed: 2.73` predicts for a 1.2 s hold, which is ~0.26 s unaccounted for;
    this is what turns that inference into a measurement."""
    clock = Clock()
    # Still for 6 samples (0.3 s), then walking at a full 2.73 tiles/s.
    state = {"n": 0, "p": (0.0, 0.0)}

    def sample():
        state["n"] += 1
        if state["n"] > 6:
            state["p"] = (state["p"][0] + 2.73 * 0.05, state["p"][1])
        return _Odo(state["p"], (0.0, 0.0)), (0.0, 0.0)

    t = cal.verify_move(sample, _Joystick(), bins=(1,), hold_seconds=1.0,
                        sleep=clock.sleep, now=clock.now)[0]
    assert t.onset_seconds == pytest.approx(0.30, abs=0.06)
    assert t.ok                                  # it did move, just late


def test_a_hero_that_never_starts_has_no_onset_rather_than_a_late_one():
    """`None`, not the end of the window. A number there would be averaged into a latency median
    and quietly report a hero that never moved as a slow one."""
    clock = Clock()
    t = cal.verify_move(_walker(50, (0.0, 0.0)), _Joystick(), bins=(1,), hold_seconds=0.5,
                        sleep=clock.sleep, now=clock.now)[0]
    assert t.onset_seconds is None


def test_verify_move_reports_the_worst_single_frame_shift():
    """The per-frame bound the whole 20 Hz cadence exists to satisfy, observed rather than argued.
    A live run is the only place it can be measured against a real capture rate."""
    clock = Clock()
    seq = [(0.2, 0.0)] * 5 + [(1.4, 0.0)] + [(0.2, 0.0)] * 5
    state = {"i": 0, "p": (0.0, 0.0)}

    def sample():
        d = seq[min(state["i"], len(seq) - 1)]
        state["i"] += 1
        state["p"] = (state["p"][0] + d[0], state["p"][1] + d[1])
        return _Odo(state["p"], d), (0.0, 0.0)
    t = cal.verify_move(sample, _Joystick(), bins=(1,), hold_seconds=0.5,
                        sleep=clock.sleep, now=clock.now)[0]
    assert t.max_shift_tiles == pytest.approx(1.4, abs=1e-6)


def test_verify_move_counts_frames_odometry_could_not_trust():
    clock = Clock()
    trials = cal.verify_move(_walker(50, (0.2, 0.0), status="uncertain"), _Joystick(), bins=(1,),
                             hold_seconds=0.5, sleep=clock.sleep, now=clock.now)
    assert trials[0].lost == trials[0].frames > 0


def test_verify_move_acquires_once_and_returns_to_idle_between_bins():
    clock = Clock()
    joy = _Joystick()
    cal.verify_move(_walker(80, (0.2, 0.0)), joy, bins=(1, 5), hold_seconds=0.5,
                    sleep=clock.sleep, now=clock.now)
    assert joy.calls[0] == ("acquire", None)
    assert [c for c in joy.calls if c[0] == "acquire"] == [("acquire", None)]
    assert [c[1] for c in joy.calls if c[0] == "apply"] == [1, 0, 5, 0]
    # Idle is expressed as a POSITION. `verify_move` must never release mid-run, or the next bin
    # re-acquires from the stick's home corner and the trial measures the wrong thing.
    assert ("release", None) not in joy.calls


# --- update_calibration ---------------------------------------------------------------------

def test_update_calibration_replaces_one_block_and_keeps_the_rest(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({
        "_comment": ["top"], "screen": [1920, 1080],
        "buttons": {"_comment": ["old"], "attack": {"x": 1.0, "y": 2.0, "r": 3.0}},
        "joystick": {"_comment": ["hand-measured"], "anchor": {"x": 9.0, "y": 8.0}},
    }))
    doc = cal.update_calibration(
        path, blocks={"buttons": {"attack": {"x": 10.0, "y": 20.0, "r": 30.0}}},
        provenance={"buttons": ["fresh"]})
    assert doc["buttons"]["attack"]["x"] == 10.0
    assert doc["buttons"]["_comment"] == ["fresh"]        # the stale one is gone, not kept
    assert doc["joystick"]["_comment"] == ["hand-measured"]
    assert doc["screen"] == [1920, 1080]
    assert json.loads(path.read_text()) == doc


def test_update_calibration_leaves_untouched_blocks_byte_identical(tmp_path):
    """The joystick block's provenance was measured by hand in §4.3 and no script can reproduce
    it. Regenerating the file wholesale would replace that with a default."""
    src = json.loads((Path(__file__).resolve().parents[1]
                      / "brawl_deployment" / "data" / "control_calibration.json").read_text())
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(src))
    doc = cal.update_calibration(path, blocks={"screen": [2560, 1440]})
    assert doc["joystick"] == src["joystick"]
    assert doc["match_gate"] == src["match_gate"]
    assert doc["buttons"] == src["buttons"]
    assert doc["screen"] == [2560, 1440]


# --- the deployed HUD mask, against the controls it has to hide ------------------------------

def test_the_deployed_hud_mask_covers_every_calibrated_control_and_leaves_the_hero_clear():
    """Two measurements of the same screen that must agree, taken independently:
    `control_calibration.json`'s button centres are ring-score fits and its joystick anchor was
    measured live, while `hud_mask.json` came from which pixels stay static as the world scrolls.
    Any control outside the mask gets deposited into the terrain map as a wall on every tick."""
    from brawl_deployment.loop import HUD_MASK_PATH
    from brawl_deployment.match_state import Calibration
    from brawl_vision.camera import load_hud_mask

    w, h = 2002, 1126
    c = Calibration.load(viewport=(w, h))
    hud = load_hud_mask(HUD_MASK_PATH)
    mask = hud.bool_at((h, w))
    for name in c.buttons:
        x, y, _ = c.viewport_button(name)
        assert mask[int(y), int(x)], f"the {name} button is not masked"
    # The held stick, anywhere the loop can put the contact: the anchor plus the commanded radius,
    # which is past the knob's 92 px saturation.
    sx, sy = c.device_to_viewport
    ax, ay = c.joystick_anchor
    for a in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
        r = c.joystick_radius_px
        assert mask[int((ay + r * np.sin(a)) * sy), int((ax + r * np.cos(a)) * sx)]
    # The camera centres the hero, and the cells around it are the ones the policy reads hardest.
    assert not mask[h // 2 - 150:h // 2 + 150, w // 2 - 250:w // 2 + 250].any()
    assert hud.coverage((h, w)) < 0.25


# ---------------------------------------------------------- the operator's first ten seconds

def _load_calibrate_script():
    """`scripts/deploy_calibrate.py` is a script, not a package module. Import it by path so the
    one piece of it with real branching -- the wait loop -- is testable without a live emulator."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "deploy_calibrate.py"
    spec = importlib.util.spec_from_file_location("deploy_calibrate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _WaitRig:
    """Enough of `Rig` for `wait_for_gate`, which is all this exercises."""

    def __init__(self, reasons, gate_after=0):
        self.reasons = list(reasons)
        self.gate_after = gate_after
        self.grabs = 0
        self.guard = self
        self.match = self
        self.vision = self
        self.frame = _Frame()

    def check(self):                          # WindowGuard.check
        return self.reasons.pop(0) if self.reasons else None

    def grab(self):
        self.grabs += 1
        return self.frame

    def update(self, _image):                 # MatchState.update
        return self.grabs > self.gate_after

    def refine(self, _image):
        return 0.87

    def warm(self, _image):
        pass


class _Frame:
    image = np.zeros((4, 4, 3), np.uint8)


OCCLUDED = WindowFault("occluded", "1 window(s) are drawn over the emulator (hwnd [42]). "
                                   "The capture is showing them, not the game.")
MOVED = WindowFault("moved", "the emulator window moved or resized: client area was A, is now B")


def test_the_operators_own_terminal_does_not_kill_the_calibration_run():
    """The script is started FROM a terminal, so that terminal is drawn over the fullscreen
    emulator at the instant it launches. Raising on the first window check would fail every run
    before the human could alt-tab back -- which is the exact workflow the script exists for."""
    mod = _load_calibrate_script()
    rig = _WaitRig([OCCLUDED, OCCLUDED, OCCLUDED])
    assert mod.Rig.wait_for_gate(rig, timeout=5.0) is True


def test_a_moved_window_still_raises_rather_than_waiting():
    """Waiting fixes an occlusion and fixes nothing else. A moved window means the cached content
    box is cropping the wrong pixels, and every further second measures garbage."""
    mod = _load_calibrate_script()
    rig = _WaitRig([MOVED])
    with pytest.raises(RuntimeError, match="moved or resized"):
        mod.Rig.wait_for_gate(rig, timeout=5.0)


def test_an_emulator_that_stays_covered_times_out_rather_than_hanging():
    """No grace-period constant is needed because the gate cannot open through an occlusion, so
    `--wait` already bounds it. This is that property."""
    mod = _load_calibrate_script()
    rig = _WaitRig([OCCLUDED] * 500)
    assert mod.Rig.wait_for_gate(rig, timeout=0.25) is False
    assert rig.grabs == 0                      # nothing was measured through the covering window



# ---------------------------------------------------------- the gadget anchor probe

def _probe_cal():
    from brawl_deployment.match_state import Calibration
    return Calibration.load(viewport=(2002, 1126))


def _button_frame(cal, discs):
    """A viewport frame with a filled disc at each named button's calibrated centre, `discs`
    mapping name to grey level. A filled disc scores 0.997 at every anchor, a grey-90 one
    0.87-0.92, a blank frame 0.000 (measured 2026-09-21)."""
    frame = np.zeros((1126, 2002, 3), np.uint8)
    for name, level in discs.items():
        cx, cy, r = cal.viewport_button(name)
        cv2.circle(frame, (round(cx), round(cy)), round(r), (level, level, level), -1)
    return frame


def _probe_gates(cal, frame):
    """The gates `probe_gadget` builds: the production gate as `wait_for_gate` leaves it (in
    match and refined), and one gate per other button, refined with a floor of 0.

    Keyed off `cal.gate_anchor` rather than a literal, because which disc gates is now a property
    of the shipped file: it moved off the Super on 2026-09-22 and these fixtures must move with
    it, or they assert the probe against a gate the probe does not use.
    """
    from dataclasses import replace

    from brawl_deployment.match_state import MatchState
    anchor = MatchState(cal)
    for _ in range(cal.enter_samples):
        anchor.update(frame)
    anchor.refine(frame)
    gates = {cal.gate_anchor: anchor}
    for name in cal.buttons:
        if name == cal.gate_anchor:
            continue
        gates[name] = MatchState(replace(cal, gate_anchor=name))
        gates[name].refine(frame, min_score=0.0)
    return gates


class _FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0.0
        self.now += seconds


def _probe_buttons(backend):
    """The shipped geometry, pinned rather than loaded: attack is `control.attack_tap`'s
    clearance point (0.88, 0.55 of 1920x1080), and super/gadget are the discs the calibration
    names after the 2026-09-22 rotation. Literals on purpose -- a double built from the same
    source as the code cannot catch the code moving."""
    return Buttons(backend, attack=(1689.6, 594.0), super_=(1559.9, 901.1),
                   gadget=(1675.9, 997.0), aim_radius_px=77.0)


def _record(mod, grab, gates, buttons, check=lambda: None, pre=8, post=12,
            gate_name="hypercharge"):
    clock = _FakeClock()
    return mod.record_gadget_trace(grab, check, gates, buttons, gate_name=gate_name,
                                   tick_seconds=1 / 12, pre_roll_ticks=pre, record_ticks=post,
                                   check_every=4, clock=clock, sleep=clock.sleep)


def test_the_probe_taps_the_gadget_once_after_the_pre_roll_and_lifts_it_on_the_next_tick():
    """The one input the probe sends, sent the policy's way: a bare down on the gadget after
    that tick's gate read, lifted by the next tick's settle, nothing else. Grabs are logged into
    the same list, so the order of touches against ticks is asserted, not just their count."""
    mod = _load_calibrate_script()
    cal = _probe_cal()
    lit = _button_frame(cal, {"gadget": 200, "super": 200, "hypercharge": 200})
    null = NullBackend()

    def grab():
        null.log.append(("grab",))
        return lit

    trace = _record(mod, grab, _probe_gates(cal, lit), _probe_buttons(null))
    assert null.log[:9] == [("grab",)] * 9
    assert null.log[9:12] == [("down", SLOT_TAP, 1675.9, 997.0), ("grab",), ("up", SLOT_TAP)]
    assert null.log[12:] == [("grab",)] * 10
    assert trace["tap_tick"] == 8 and trace["aborted"] is None
    assert [k["i"] for k in trace["ticks"] if k["tapped"]] == [8]
    assert len(trace["ticks"]) == 20
    assert trace["ticks"][0]["t"] == pytest.approx(-8 / 12, abs=1e-4)     # stored to 4 places
    assert trace["ticks"][-1]["t"] == pytest.approx(11 / 12, abs=1e-4)
    assert trace["tick_hz"] == 12.0 and trace["recharge_s"] == 18.0
    assert (trace["threshold"], trace["enter_samples"], trace["exit_samples"]) == (0.45, 6, 4)
    assert list(trace["anchors"]) == ["hypercharge", "gadget", "super"]


def test_the_probe_records_each_anchor_through_the_real_gate_class():
    """The gadget disc vanishes for six ticks after the tap and comes back, which is what a
    recharge looks like. A gate anchored on the gadget would exit on the fourth dark tick and could
    not re-enter for five more (enter_samples is 6) -- and that shadow gate is recorded.
    The PRODUCTION gate is on `hypercharge`, which the tap does not touch, so it holds
    straight through: after 2026-09-22 that separation is the point of the whole probe."""
    mod = _load_calibrate_script()
    cal = _probe_cal()
    lit = _button_frame(cal, {"gadget": 200, "super": 200, "hypercharge": 200})
    dark = _button_frame(cal, {"super": 200, "hypercharge": 200})
    frames = iter([lit] * 9 + [dark] * 6 + [lit] * 5)
    trace = _record(mod, lambda: next(frames), _probe_gates(cal, lit),
                    _probe_buttons(NullBackend()))
    post = trace["ticks"][9:]
    assert [k["in_match"]["gadget"] for k in post] == [True] * 3 + [False] * 8
    assert all(k["in_match"]["super"] and k["in_match"]["hypercharge"] for k in post)
    # The alternatives start out of match and enter on their sixth lit tick, inside the pre-roll.
    assert [k["in_match"]["super"] for k in trace["ticks"][:6]] == [False] * 5 + [True]
    assert all(k["score"]["gadget"] < 0.45 for k in post[:6])
    assert all(k["score"]["gadget"] > 0.9 for k in trace["ticks"][:9] + post[6:])
    assert all(k["score"]["super"] > 0.9 and k["score"]["hypercharge"] > 0.9 for k in post)
    assert trace["ticks"][0]["colour"]["gadget"] == [200.0, 200.0, 200.0]
    assert post[0]["colour"]["gadget"] == [0.0, 0.0, 0.0]
    assert post[0]["colour"]["super"] == [200.0, 200.0, 200.0]


def test_the_probe_never_taps_through_a_closed_gate():
    """The interlock every press obeys. The `hypercharge` disc, which is what the gate reads, goes
    dark on tick 2, so the production gate exits on tick 5 and reads out of match at the tap tick:
    nothing goes down, and the trace ends there saying why. The gadget stays lit throughout, so
    this asserts the interlock rather than the tapped button's own state."""
    mod = _load_calibrate_script()
    cal = _probe_cal()
    lit = _button_frame(cal, {"gadget": 200, "super": 200, "hypercharge": 200})
    dark = _button_frame(cal, {"gadget": 200, "super": 200})
    frames = iter([lit] * 2 + [dark] * 20)
    null = NullBackend()
    trace = _record(mod, lambda: next(frames), _probe_gates(cal, lit), _probe_buttons(null))
    assert [e for e in null.log if e[0] == "down"] == []
    assert trace["tap_tick"] is None
    assert "out of match at the tap tick" in trace["aborted"]
    assert len(trace["ticks"]) == 9


def test_a_covered_emulator_stops_the_probe_and_a_moved_one_raises():
    """A covered window's frames show the terminal, not the buttons, so the recording stops there
    rather than scoring it. A moved window makes every calibrated coordinate wrong, so it
    raises. Neither leaves a contact down."""
    mod = _load_calibrate_script()
    cal = _probe_cal()
    lit = _button_frame(cal, {"gadget": 200, "super": 200, "hypercharge": 200})
    # Checks run on ticks 0, 4, 8 (the tap) and 12. Covered at the fourth: after the tap.
    checks = iter([None, None, None, OCCLUDED])
    null = NullBackend()
    trace = _record(mod, lambda: lit, _probe_gates(cal, lit), _probe_buttons(null),
                    check=lambda: next(checks))
    assert trace["tap_tick"] == 8
    assert "covered at tick 12" in trace["aborted"]
    assert trace["window_faults"][0]["tick"] == 12
    assert len(trace["ticks"]) == 12
    assert null.log[-1] == ("up", SLOT_TAP) and not null.contacts

    checks = iter([None, None, None, MOVED])
    null = NullBackend()
    buttons = _probe_buttons(null)
    with pytest.raises(RuntimeError, match="moved or resized"):
        _record(mod, lambda: lit, _probe_gates(cal, lit), buttons, check=lambda: next(checks))
    assert not null.contacts


def test_a_capture_failure_right_after_the_tap_still_lifts_it():
    """The tick after the tap lifts it in its `settle()`, which runs after that tick's grab. A
    grab that raises there would leave the gadget held down, so the probe lifts on the way out."""
    mod = _load_calibrate_script()
    cal = _probe_cal()
    lit = _button_frame(cal, {"gadget": 200, "super": 200, "hypercharge": 200})
    frames = [lit] * 9

    def grab():
        if not frames:
            raise OSError("capture lost")
        return frames.pop()

    null = NullBackend()
    with pytest.raises(OSError, match="capture lost"):
        _record(mod, grab, _probe_gates(cal, lit), _probe_buttons(null))
    assert null.log == [("down", SLOT_TAP, 1675.9, 997.0), ("up", SLOT_TAP)]
    assert not null.contacts


def test_the_probe_holds_its_tick_rate_without_bursting():
    """`DeployLoop._pace`'s contract, which the probe copies: an on-time tick sleeps to the next
    deadline, and an overrun restarts from now rather than firing the missed ticks back to back,
    since the hysteresis counts samples and a burst would squeeze an exit into less time."""
    mod = _load_calibrate_script()
    clock = _FakeClock()
    assert mod._pace(1000.0, 0.25, clock, clock.sleep) == 1000.25
    assert clock.now == 1000.25
    clock.now = 1001.0                          # the next tick ran 0.75 s long
    assert mod._pace(1000.25, 0.25, clock, clock.sleep) == 1001.0
    assert clock.now == 1001.0                  # and nothing slept


def _probe_live(mod, monkeypatch, tmp_path, frame, null):
    """`probe_gadget` on a fake rig holding one frame, with the recording shortened and the tick
    rate raised to 1 kHz so it runs in a moment: 8 ticks, the tap, 11 more."""
    from types import SimpleNamespace

    from brawl_deployment.match_state import MatchState
    monkeypatch.setattr(mod, "PROBE_PRE_ROLL_S", 0.008)
    monkeypatch.setattr(mod, "PROBE_RECORD_S", 0.012)
    cal = _probe_cal()
    gate = MatchState(cal)                      # as `wait_for_gate` leaves it
    for _ in range(cal.enter_samples):
        gate.update(frame)
    rig = SimpleNamespace(cal=cal, match=gate, gate_refine_score=gate.refine(frame),
                          grab=lambda: SimpleNamespace(image=frame),
                          guard=SimpleNamespace(check=lambda: None),
                          controls=SimpleNamespace(buttons=_probe_buttons(null)))
    cfg = SimpleNamespace(loop_tick_hz=1000.0, window_check_every_n_ticks=4)
    out = tmp_path / "audit" / "trace.json"
    return mod.probe_gadget(rig, cfg, SimpleNamespace(trace_out=str(out))), out


def test_probe_gadget_writes_a_trace_that_loads_back_whole(tmp_path, monkeypatch):
    """The live entry point end to end: every button gets a gate, a refine score and its
    re-fitted radius, and the file on disk parses back to exactly the trace returned."""
    mod = _load_calibrate_script()
    cal = _probe_cal()
    null = NullBackend()
    trace, out = _probe_live(mod, monkeypatch, tmp_path,
                             _button_frame(cal, {"gadget": 200, "super": 200, "hypercharge": 200}),
                             null)
    text = out.read_text(encoding="utf-8")
    assert json.loads(text) == trace
    assert len([l for l in text.splitlines() if l.startswith('  {"i": ')]) == len(trace["ticks"])
    assert trace["tap_tick"] == 8 and len(trace["ticks"]) == 20
    assert set(trace["anchors"]) == {"gadget", "super", "hypercharge"}
    assert all(a["refine_score"] > 0.9 for a in trace["anchors"].values())
    # Re-fitted radii: the stored 34.72 and 31.49 moved one and three 0.5 px refine steps. Both
    # numbers are untouched by the 2026-09-22 rotation, which moved only which disc owns them:
    # 34.72 was called the gadget's and is the Super's, 31.49 was the attack's and is the gadget's.
    assert (trace["anchors"]["super"]["r"], trace["anchors"]["gadget"]["r"]) == (35.22, 29.99)
    assert all(trace["ticks"][8]["in_match"].values())
    assert [e[0] for e in null.log] == ["down", "up"]


def test_a_button_that_cannot_anchor_the_gate_is_recorded_rather_than_fatal(tmp_path,
                                                                            monkeypatch):
    """The probe refines every non-gate button with a floor of 0. One that does not read as a
    ring on this source is a finding, not a reason to lose the trace.

    The dark disc here is the GADGET, which is the strongest form of the case: the probe still
    taps it, still records, and still writes the trace, because the interlock is the gate anchor's
    reading and the gate anchor is lit. Before 2026-09-22 this could not have been expressed --
    the gadget was believed to be the gate, so a dark gadget was a closed gate."""
    mod = _load_calibrate_script()
    cal = _probe_cal()
    trace, _ = _probe_live(mod, monkeypatch, tmp_path,
                           _button_frame(cal, {"hypercharge": 200, "super": 200}), NullBackend())
    assert trace["anchors"]["gadget"]["refine_score"] < 0.45
    assert trace["tap_tick"] == 8
    warnings = mod.summarize_trace(trace)["warnings"]
    assert any(w.startswith("gadget never read in match before the tap") for w in warnings)


def test_the_probe_runs_alone(capsys):
    """Every job presses attack or Super, each an anchor the trace measures, and `--write` stores
    buttons the probe never fits. Refused before anything touches the emulator."""
    mod = _load_calibrate_script()
    for extra in (["--jobs", "tap"], ["--write"]):
        with pytest.raises(SystemExit) as exc:
            mod.main(["--probe-gadget", *extra])
        assert exc.value.code == 2
        assert "--probe-gadget runs alone" in capsys.readouterr().err


def test_the_probe_waits_out_the_sims_own_gadget_recharge():
    """A trace must cover the whole recharge to say KEEP, and the recharge it covers is the one
    the sim trains on: Mortis's `gadget_cooldown` in configs/brawlers.yaml."""
    import yaml
    mod = _load_calibrate_script()
    brawlers = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" /
                               "brawlers.yaml").read_text(encoding="utf-8"))
    assert brawlers["hero_mortis"]["gadget_cooldown"] == 18.0
    assert mod.GADGET_RECHARGE_S == 18.0
    assert mod.PROBE_RECORD_S > mod.GADGET_RECHARGE_S


# Summaries of hand-built traces: tap on tick 2 at 12 Hz, a 0.25 s recharge so seven post-tap
# ticks cover it. Every number below is a literal, so each expected value can be checked by eye.
LIT, DIM = [200.0, 200.0, 200.0], [90.0, 90.0, 90.0]


def _summary_trace(score, in_match, colour, *, recharge_s=0.25, aborted=None):
    names = list(score)
    return {"tick_hz": 12.0, "tap_tick": 2, "aborted": aborted, "recharge_s": recharge_s,
            "gate_anchor": "gadget", "threshold": 0.45, "enter_samples": 6, "exit_samples": 4,
            "anchors": {a: {"cx": 0.0, "cy": 0.0, "r": 30.0, "refine_score": 0.99}
                        for a in names},
            "window_faults": [],
            "ticks": [{"i": i, "t": (i - 2) / 12, "window": None, "tapped": i == 2,
                       "score": {a: score[a][i] for a in names},
                       "in_match": {a: in_match[a][i] for a in names},
                       "colour": {a: colour[a][i] for a in names}, "tick_ms": 1.0}
                      for i in range(len(score[names[0]]))]}


T, F = True, False
STEADY = {"super": [0.70, 0.71, 0.70, 0.69, 0.66, 0.68, 0.70, 0.71, 0.70, 0.70],
          "attack": [0.60, 0.58, 0.59, 0.50, 0.52, 0.55, 0.57, 0.58, 0.59, 0.60]}
ALL_IN = [T] * 10


def test_a_gadget_that_dims_but_never_dips_keeps_the_anchor():
    mod = _load_calibrate_script()
    score = {"gadget": [0.80, 0.81, 0.80, 0.60, 0.55, 0.52, 0.58, 0.62, 0.79, 0.80], **STEADY}
    colour = {"gadget": [LIT] * 3 + [DIM] * 5 + [LIT] * 2, "super": [LIT] * 10,
              "attack": [LIT] * 10}
    s = mod.summarize_trace(_summary_trace(score, dict.fromkeys(score, ALL_IN), colour))
    assert s["verdict"] == "KEEP"
    assert s["anchors"]["gadget"]["ticks_under"] == 0
    assert s["anchors"]["gadget"]["colour_moved"] == 110.0
    assert s["anchors"]["gadget"]["post_min"] == 0.52
    assert s["covered_s"] == pytest.approx(7 / 12)

    # The same scores with a button that never changed: nothing shows the gadget fired, so a
    # steady score proves nothing and the verdict must not be KEEP.
    colour["gadget"] = [LIT] * 10
    s = mod.summarize_trace(_summary_trace(score, dict.fromkeys(score, ALL_IN), colour))
    assert s["verdict"] == "INCONCLUSIVE"
    assert "looks the same after the tap" in s["reason"]


def test_a_gadget_whose_gate_exits_mid_match_is_a_change_and_names_the_wider_margin():
    """Dark for five ticks while super and attack stay up: a false exit on the fourth, the
    alternatives ranked by floor, and the 2-of-3 vote holding, since two gates stay in."""
    mod = _load_calibrate_script()
    score = {"gadget": [0.80, 0.81, 0.80, 0.30, 0.20, 0.25, 0.28, 0.35, 0.79, 0.80], **STEADY}
    in_match = {"gadget": [T] * 6 + [F] * 4, "super": ALL_IN, "attack": ALL_IN}
    colour = {"gadget": [LIT] * 3 + [DIM] * 5 + [LIT] * 2, "super": [LIT] * 10,
              "attack": [LIT] * 10}
    s = mod.summarize_trace(_summary_trace(score, in_match, colour))
    g = s["anchors"]["gadget"]
    assert s["verdict"] == "CHANGE"
    assert (g["ticks_under"], g["longest_under"]) == (5, 5)
    assert g["false_exits"] == [pytest.approx(4 / 12)]
    assert (g["post_min"], g["post_min_t"]) == (0.20, pytest.approx(2 / 12))
    assert s["anchors"]["super"]["margin"] == pytest.approx(0.21)
    assert s["anchors"]["attack"]["margin"] == pytest.approx(0.05)
    assert s["vote"] == {"exits": [], "false_exits": []}
    assert "widest alternative anchor is super" in s["reason"]
    assert "EXITED at t=+0.33 s" in s["reason"] and "vote held throughout" in s["reason"]
    lines = mod.format_summary(s)
    assert lines[-1].startswith("CHANGE: ")
    assert sum(" FALSE" in line for line in lines) == 1


def test_two_buttons_down_at_once_close_the_vote_while_the_match_goes_on():
    """The vote's worst case: the gadget recharging while something else hides the attack
    button. Both gates exit on tick 6 with super still up, so one gate is left, the vote exits
    too, and since super never dipped it is a false exit rather than the controls vanishing."""
    mod = _load_calibrate_script()
    score = {"gadget": [0.80, 0.81, 0.80, 0.30, 0.20, 0.25, 0.28, 0.35, 0.30, 0.30],
             "super": STEADY["super"],
             "attack": [0.60, 0.58, 0.59, 0.30, 0.30, 0.30, 0.30, 0.30, 0.59, 0.60]}
    in_match = {"gadget": [T] * 6 + [F] * 4, "super": ALL_IN, "attack": [T] * 6 + [F] * 4}
    colour = {"gadget": [LIT] * 3 + [DIM] * 7, "super": [LIT] * 10, "attack": [LIT] * 10}
    s = mod.summarize_trace(_summary_trace(score, in_match, colour))
    assert s["common_mode_ticks"] == 0
    assert s["vote"] == {"exits": [pytest.approx(4 / 12)], "false_exits": [pytest.approx(4 / 12)]}
    assert s["verdict"] == "CHANGE"
    assert "widest alternative anchor is super" in s["reason"]
    assert "vote would have exited at t=+0.33 s" in s["reason"]


def test_a_dip_too_short_to_exit_is_still_a_change():
    """The verdict is about the score, not the exit: any post-tap tick under the threshold means
    the margin is gone. Two dark ticks, one lit, three dark: five under, longest three, and no
    run reaches exit_samples."""
    mod = _load_calibrate_script()
    score = {"gadget": [0.80, 0.81, 0.80, 0.30, 0.20, 0.50, 0.28, 0.35, 0.40, 0.80], **STEADY}
    colour = {"gadget": [LIT] * 3 + [DIM] * 6 + [LIT], "super": [LIT] * 10,
              "attack": [LIT] * 10}
    s = mod.summarize_trace(_summary_trace(score, dict.fromkeys(score, ALL_IN), colour))
    g = s["anchors"]["gadget"]
    assert s["verdict"] == "CHANGE"
    assert (g["ticks_under"], g["longest_under"], g["exits"]) == (5, 3, [])
    assert "its gate held" in s["reason"]


def test_every_button_going_dark_together_is_not_the_gadgets_doing():
    """The hero died on tick 5: all three scores fall together, every gate exits on tick 8, and
    none of it counts as a gadget dip or a false exit. The controls were up only to t=+0.17 s,
    short of the 0.5 s recharge, so the trace cannot answer."""
    mod = _load_calibrate_script()
    gone = [0.05] * 5
    score = {"gadget": [0.80, 0.81, 0.80, 0.78, 0.79] + gone,
             "super": [0.70, 0.71, 0.70, 0.69, 0.66] + gone,
             "attack": [0.60, 0.58, 0.59, 0.50, 0.52] + gone}
    in_match = dict.fromkeys(score, [T] * 8 + [F] * 2)
    colour = {"gadget": [LIT] * 3 + [DIM] * 7, "super": [LIT] * 10, "attack": [LIT] * 10}
    s = mod.summarize_trace(_summary_trace(score, in_match, colour, recharge_s=0.5))
    assert s["common_mode_ticks"] == 5
    assert s["covered_s"] == pytest.approx(2 / 12)
    assert all(st["ticks_under"] == 0 for st in s["anchors"].values())
    assert all(st["exits"] == [pytest.approx(6 / 12)] and st["false_exits"] == []
               for st in s["anchors"].values())
    assert s["vote"]["false_exits"] == []
    assert s["verdict"] == "INCONCLUSIVE" and "the controls vanished" in s["reason"]


def test_an_aborted_probe_is_never_a_verdict():
    mod = _load_calibrate_script()
    score = {"gadget": [0.80] * 10, **STEADY}
    colour = {"gadget": [LIT] * 3 + [DIM] * 7, "super": [LIT] * 10, "attack": [LIT] * 10}
    s = mod.summarize_trace(_summary_trace(score, dict.fromkeys(score, ALL_IN), colour,
                                           aborted="the emulator was covered at tick 4"))
    assert s["verdict"] == "INCONCLUSIVE"
    assert "covered at tick 4" in s["reason"]
