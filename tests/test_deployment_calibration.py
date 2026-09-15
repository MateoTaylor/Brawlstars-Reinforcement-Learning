"""`brawl_deployment/control/calibration.py` -- the fits and the live checks, offline.

Every verification in that module exists to be run against a live emulator, which is exactly why
it needs tests that are not: the whole point of `verify_tap` is that it says FAIL on the §6.8
Training Grounds result, and the only way to know it does is to hand it that result.

The clock is faked. `verify_tap`'s windows are tuned to a reload rate and a swing animation, so
"the trough window is long enough" and "the arm timeout is respected" are properties worth
asserting rather than comments -- and a real clock would spin for 0.9 s a trial to assert them.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from brawl_deployment.control import ATTACK_FIRE, ATTACK_SUPER
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
    """`Buttons` reduced to what `verify_tap` touches, recording the down/up order."""

    def __init__(self, attack=(1690.0, 594.0), super_=(1462.5, 1000.5)):
        self.attack, self.super_ = attack, super_
        self.events = []

    def tap(self, action):
        self.events.append(("tap", action))
        return action != 0

    def settle(self):
        self.events.append(("settle", None))


class _Game:
    """A pretend Mortis: ammo reloads against the fake clock, and a tap spends one after the
    swing animation.

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


class _GameButtons(_Buttons):
    """`_Buttons` wired to a `_Game`, so a tap has a consequence to observe."""

    def __init__(self, game, **kw):
        super().__init__(**kw)
        self.game = game

    def tap(self, action):
        self.game.fire()
        return super().tap(action)


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
    report = cal.verify_tap(game.read, _GameButtons(game), trials=3,
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
    report = cal.verify_tap(game.read, _GameButtons(game), trials=3,
                            sleep=clock.sleep, now=clock.now)
    assert not report.ok
    assert all(t.drop == pytest.approx(0.0, abs=1e-6) for t in report.trials)
    assert "FAIL" in report.summary()


def test_verify_tap_rejects_a_mixed_result_rather_than_taking_the_majority():
    # 2 of 3 is exactly what Training Grounds produced. A tap has no mechanism for landing
    # two times in three, so a majority rule would have called that setup good.
    clock = Clock()
    game = _Game(clock, fires=[True, False, False])
    report = cal.verify_tap(game.read, _GameButtons(game), trials=3,
                            sleep=clock.sleep, now=clock.now)
    assert [t.ok for t in report.valid] == [True, False, False]
    assert not report.ok


def test_verify_tap_takes_the_trough_not_the_last_sample():
    """The reload starts refilling almost immediately, so a fixed delayed sample can land after
    the bar has partly recovered and read that as a failed tap."""
    clock = Clock()
    game = _Game(clock)
    trial = cal.verify_tap(game.read, _GameButtons(game), trials=1,
                           sleep=clock.sleep, now=clock.now).trials[0]
    assert trial.trough == pytest.approx(2.0, abs=0.05)
    # By the end of the window the bar has already refilled well past the trough -- a single
    # delayed sample would have read this as a partial drop and failed a working setup.
    assert game.read() > trial.trough + 0.3
    assert trial.ok


def test_verify_tap_waits_for_the_bar_to_refill_before_pressing():
    clock = Clock()
    game = _Game(clock, ammo=0.4)
    report = cal.verify_tap(game.read, _GameButtons(game), trials=1,
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
    report = cal.verify_tap(game.read, _GameButtons(game), trials=1,
                            sleep=clock.sleep, now=clock.now)
    assert report.trials[0].ok                 # the one trial passed
    assert not report.ok                       # ...and one is not enough


def test_verify_tap_releases_every_press_it_makes():
    clock = Clock()
    game = _Game(clock)
    buttons = _GameButtons(game)
    cal.verify_tap(game.read, buttons, trials=3, sleep=clock.sleep, now=clock.now)
    taps = [e for e in buttons.events if e[0] == "tap"]
    settles = [e for e in buttons.events if e[0] == "settle"]
    assert len(taps) == len(settles) == 3
    assert buttons.events[0][0] == "tap" and buttons.events[1][0] == "settle"


def test_verify_tap_reports_the_point_for_the_action_it_was_given():
    clock = Clock()
    game = _Game(clock)
    buttons = _GameButtons(game)
    fire = cal.verify_tap(game.read, buttons, action=ATTACK_FIRE, trials=1,
                          sleep=clock.sleep, now=clock.now)
    sup = cal.verify_tap(game.read, buttons, action=ATTACK_SUPER, trials=1,
                         sleep=clock.sleep, now=clock.now)
    assert fire.point == buttons.attack
    assert sup.point == buttons.super_
    assert [e[1] for e in buttons.events if e[0] == "tap"] == [ATTACK_FIRE, ATTACK_SUPER]


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
