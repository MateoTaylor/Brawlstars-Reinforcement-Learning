"""Measure and verify the control calibration against a live match. Run it during one battle.

    python scripts/deploy_calibrate.py                  # every job, report only
    python scripts/deploy_calibrate.py --write          # ...and store the re-fitted buttons
    python scripts/deploy_calibrate.py --jobs tap       # just the one you care about
    python scripts/deploy_calibrate.py --probe-gadget   # G6.1: does the gate survive a gadget tap?

**This is the script the friendly battle is for** (BRAWL_DEPLOYMENT_DESIGN.md 10.11). Three
questions cannot be answered anywhere else, and Training Grounds gave a wrong answer to all three:

  tap      does `control_attack_tap` actually fire?   A venue-only button sat over the tap point
                                                      there and swallowed 2 of 3 touches (6.8).
  move     does the camera go where we steer?         Training Grounds does not scroll, so
                                                      odometry correctly reads ~0 and proves
                                                      nothing.
  buttons  where are the buttons on THIS source?      Radii do not transfer between frame
                                                      sources; a 6.7 px error flips the gate.

**What the operator actually does, in order:**

  1. BlueStacks fullscreen on the 1440p monitor, Nulls Brawl open, Mortis selected.
  2. Start a friendly battle against bots. Do not wait for it to load.
  3. Alt-tab to the terminal, run this, and alt-tab straight back to BlueStacks.
  4. Leave the emulator on top and do not touch the controls for ~25 seconds. Losing is fine.
  5. Alt-tab back and read the report. Nothing was written unless `--write` was passed.

Step 3 is safe in either order because **an occluded emulator is a WAIT, not an error** -- your
terminal is drawn over the game at the instant this launches, and `wait_for_gate` treats that as
"not ready" rather than a fault (a moved or minimized window still raises). The match gate also
cannot open through your terminal, so `--wait` bounds the whole thing on its own.

Nothing needs timing: the script waits for the gate itself, and emits input only while the gate is
true, same interlock as `loop.py`.

**It plays badly on purpose.** It stands still, dashes right, and walks in four straight lines. Losing the
match is fine and expected; the point is that every action is one whose consequence is checkable.

Nothing is written unless `--write` is passed, and then only the blocks this run measured -- the
joystick's saturation radius was measured by hand in 4.3 and no script here can reproduce it.

**`--probe-gadget` is a separate run with its own protocol** (SIM_OVERHAUL_STEPS.md Step G6.1).
The probe waits for the gate like every job, scores every calibrated disc on every perception
tick for 2 s, taps the gadget ONCE (the policy's own bare tap, interlocked on the gate), and
keeps scoring for 20 s: the 18 s recharge and 2 s of the charged button coming back. It writes
runs/audit/gadget_anchor_trace.json (`--trace-out`) and prints G6.2's verdict: KEEP, CHANGE or
INCONCLUSIVE. It runs alone: no `--jobs`, no `--write`.

It asks TWO questions about TWO different discs, which is the correction of 2026-09-22. Did the tap
land? That is the GADGET's colour, because the gadget is what was pressed. Did the gate survive?
That is the ANCHOR's score, and the anchor is `hypercharge`, which nothing presses. The original
probe read both off the gate anchor, which was sound only while the file believed the gadget and
the anchor were one disc. They never were: it was the Super.

  1. A real match: a friendly battle against bots, never Training Grounds. Mortis, with his
     gadget equipped.
  2. Start it and alt-tab straight back, as above. It starts itself once the gate opens.
  3. For the whole ~25 s, do NOT press attack, Super or the gadget: each is an anchor being
     measured, and a Super detonating over the buttons drops all three. Moving to stay alive is
     fine. Dying is not: the buttons vanish, and the rest of the trace shows no controls.
  4. A few seconds in, Mortis throws his gadget. If he never does, the summary says INCONCLUSIVE
     rather than guessing.
  5. Keep the emulator on top. A covered window stops the probe rather than scoring your terminal.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from brawl_deployment.config import load_deployment_config, validate            # noqa: E402
from brawl_deployment.control import ATTACK_FIRE, ATTACK_GADGET, ATTACK_SUPER  # noqa: E402
from brawl_deployment.control import calibration as cal_lib                     # noqa: E402
from brawl_deployment.loop import Controls, VisionStack, pin_thread_pools       # noqa: E402
from brawl_deployment.match_state import CALIBRATION_PATH, Calibration, MatchState   # noqa: E402

JOBS = ("buttons", "tap", "super", "move")

# Bins driven by the `move` job: right, down, left, up at 16 bins. Cardinals rather than diagonals
# so a sign error on either axis is unmistakable in the printed vector rather than shared between
# both components. `+1` is `hero.decode_action`'s offset -- action 0 is idle.
CARDINAL_BINS = (1, 5, 9, 13)

# Where the `tap` and `super` jobs drag their presses: 0 rad, screen-right (no y flip, so this is
# also the sim's +x). One fixed direction so the operator can check the dash by eye.
AIM_BEARING = 0.0


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Rig:
    """Capture, vision and controls, wired the way `DeployLoop.from_config` wires them.

    Built here rather than by borrowing a `DeployLoop`, because this script's whole job is to do
    things the loop must never do -- fire on demand, walk in a fixed pattern, ignore the policy.
    Sharing the constructor would mean the loop grew a mode, and a mode is a code path that can be
    entered by accident. What IS shared is `VisionStack` and `Controls`, which is where the
    geometry lives and where a divergence would actually matter.
    """

    def __init__(self, cfg, n_move_bins: int = 16):
        import mss

        from brawl_deployment.capture import DeployCapture
        from brawl_deployment.window import WindowGuard, find_emulator_window, set_dpi_aware

        pin_thread_pools(cfg)      # the emulator is on these cores too (loop.py, design 7.2)
        set_dpi_aware()
        window = find_emulator_window(cfg.window_exe)
        with mss.mss() as sct:
            monitors = [dict(m) for m in sct.monitors]
        self.guard = WindowGuard(window, monitors,
                                 require_fullscreen=cfg.window_require_fullscreen,
                                 check_occlusion=cfg.window_check_occlusion)
        self.capture = DeployCapture.from_window(window, monitors)
        self.cal = Calibration.load()
        self.match = MatchState(self.cal)
        self.vision = VisionStack.build(detector_threads=cfg.compute_detector_threads,
                                        detector_spin=cfg.compute_detector_spin)
        # n_move_bins is 16 here rather than read from the checkpoint, and the checkpoint is not
        # loaded at all: this script issues no policy actions, so pulling in a 300M-step run to
        # read one integer would be cost with no signal. `CARDINAL_BINS` assumes the same 16.
        self.controls = Controls.build(self.cal, cfg, n_move_bins)
        self.frame = None
        self.gate_refine_score = None        # what `wait_for_gate`'s refine achieved

    def grab(self):
        self.frame = self.capture.grab()
        return self.frame

    def hero(self, image=None):
        """The `player` detection in `image`, or None. Same selector `loop._read_own_bars` uses."""
        image = self.frame.image if image is None else image
        return next((d for d in self.vision.entities.predict(image) if d.label == "player"), None)

    def ammo(self):
        """Ammo right now, or None. The callable `verify_tap` drives."""
        from brawl_vision.object_detection.hp_detection.hero_bars import read_ammo

        det = self.hero(self.grab().image)
        if det is None:
            return None
        reading = read_ammo(self.frame.image, det, self.vision.cfg)
        return None if reading is None else reading.ammo

    def super_charge(self):
        from brawl_vision.object_detection.hp_detection.hero_bars import read_super

        det = self.hero(self.grab().image)
        if det is None:
            return None
        return read_super(self.frame.image, det, self.vision.cfg)

    def odometry(self):
        """One fresh frame through odometry. The camera pose alone."""
        return self.vision.odometry.update(self.vision.plan.rectify(self.grab().image))

    def pose(self):
        """`(odometry, hero_camera_relative_tile_or_None)` for one fresh frame.

        The callable `verify_move` drives, and it returns BOTH terms of the world frame on purpose
        -- odometry tracks the camera, and the camera only pans when the map is larger than the
        screen in that axis. See `verify_move` for the live run that made this necessary.
        """
        from brawl_vision.object_detection.project import to_tiles

        image = self.grab().image
        odo = self.vision.odometry.update(self.vision.plan.rectify(image))
        det = self.hero(image)
        if det is None:
            return odo, None
        placed = to_tiles([det], self.vision.plan)
        return odo, placed[0][1]

    def wait_for_gate(self, timeout: float) -> bool:
        """Block until the match gate opens, refining the ring radius on the first gameplay frame.

        The same order `loop.py` uses and for the same reason: `refine` needs a frame that is known
        to be gameplay, and the gate going true is what makes it known.

        **Occlusion is a WAIT here, not an error.** The operator starts this from a terminal, and
        that terminal is by definition drawn over the fullscreen emulator at the moment the script
        launches -- so raising on the first check would fail every run before the human could
        alt-tab back. It needs no grace-period constant either: the gate cannot open while the
        game is covered, so `timeout` already bounds the wait. Every OTHER window failure still
        raises immediately, because a moved, minimized or closed window is not something waiting
        fixes. This is the one place `loop.py`'s occlusion/fault distinction is worth repeating.
        """
        deadline = time.perf_counter() + timeout
        warned = False
        while time.perf_counter() < deadline:
            reason = self.guard.check()
            if reason is not None:
                if not reason.recoverable:
                    raise RuntimeError(f"window check failed before anything was measured: "
                                       f"{reason}")
                if not warned:
                    _log("the emulator is covered (your terminal?) -- switch to BlueStacks and "
                         "leave it on top; still waiting")
                    warned = True
                time.sleep(0.1)
                continue
            if warned:
                _log("emulator is visible again")
                warned = False
            if self.match.update(self.grab().image):
                score = self.gate_refine_score = self.match.refine(self.frame.image)
                _log(f"gate open (ring {score:.3f}); warming the detectors")
                self.vision.warm(self.frame.image)
                return True
            time.sleep(0.1)
        return False


# --- the jobs -------------------------------------------------------------------------------

def job_buttons(rig, args) -> dict:
    """Re-fit the button table on the live capture source, and report how far it moved."""
    _log(f"collecting {cal_lib.MEDIAN_FRAMES} frames for the temporal median")
    frames = []
    for _ in range(cal_lib.MEDIAN_FRAMES):
        frames.append(rig.grab().image)
        time.sleep(0.05)
    median = cal_lib.temporal_median(frames)

    expected = {name: rig.cal.viewport_button(name) for name in rig.cal.buttons}
    fits = cal_lib.fit_buttons(median, expected)
    for name in sorted(expected):
        fit = fits.get(name)
        if fit is None:
            _log(f"  {name:7s} NOT FOUND on this source (stored centre kept)")
            continue
        _log(f"  {name:7s} score {fit.score:.3f}  centre moved {fit.shift_px:5.1f} px  "
             f"r {expected[name][2]:.1f} -> {fit.r:.1f} ({fit.dr_px:+.1f})")
    return {"fits": fits, "median_shape": median.shape}


def job_tap(rig, args) -> dict:
    """The headline check: does `control_attack_tap` consume ammo?

    Pressed the way the loop presses -- down, dragged along `AIM_BEARING`, lifted -- so it proves
    the aimed path, not a bare tap. Ammo is what it asserts; the direction is for the operator's
    eyes, which is why every press goes the same way.
    """
    _log(f"pressing attack at {rig.controls.buttons.attack} x{args.trials}, dragged "
         f"{rig.controls.buttons.aim_radius_px:g} px screen-right: each dash should go RIGHT")
    report = cal_lib.verify_tap(rig.ammo, rig.controls.buttons,
                                action=ATTACK_FIRE, bearing=AIM_BEARING, trials=args.trials)
    _log("  " + report.summary())
    for t in report.trials:
        if t.note:
            _log(f"    note: {t.note}")
    if not report.ok:
        _log("  -> control_attack_tap is NOT firing. Something is on that point, or the touch is "
             "not reaching the game. Check `adb shell getevent` while re-running.")
    return {"report": report}


# How long to watch the super track for the spend, and how often to look. This used to be one
# reading at a flat 0.8 s, which FAILED a Super the operator watched fire correctly (2026-09-22).
# Two measured reasons, neither a guess. Deployment observables lag the input by roughly a second
# -- the ammo bar shows a spend that late (design 6.8) -- so 0.8 s sat under the floor. And
# Mortis's super is a DASH: the hero detector can lose him mid-flight, and a lone sample that
# lands in that window reads "unreadable" and is scored as a failure. A poll skips those frames
# and keeps looking. 3.0 s is 3x the known lag; widen it only against a trace, never a hunch.
SUPER_DRAIN_WINDOW_S = 3.0
SUPER_SAMPLE_S = 0.1


def job_super(rig, args) -> dict:
    """Same check for the Super, which taps the button table rather than a clearance point.

    Best-effort by design: the Super may simply not be charged during the window this runs in, and
    "not charged" is not a failure of the tap. It is reported as skipped, never as a pass.

    **The verdict is a POLL, not one reading at a guessed offset** (2026-09-22). PASS is the first
    frame whose track reads not-ready, and the latency it arrived at is printed, so the job
    measures the lag instead of assuming it. Frames where the hero cannot be found decide nothing;
    they are counted and skipped, because a dashing Mortis is exactly when the detector is least
    sure and exactly when this job looks.
    """
    reading = rig.super_charge()
    if reading is None or not reading.ready:
        state = "unreadable" if reading is None else reading.state
        _log(f"  super not ready ({state}) -- skipped, not failed")
        return {"skipped": state}
    _log(f"pressing super at {rig.controls.buttons.super_} (charge {reading.charge:.2f}, ready), "
         f"dragged screen-right")
    cal_lib.press_and_lift(rig.controls.buttons, ATTACK_SUPER, AIM_BEARING)

    t0 = time.perf_counter()
    samples: list[tuple[float, str, float]] = []
    after, unreadable = None, 0
    while time.perf_counter() - t0 < SUPER_DRAIN_WINDOW_S:
        time.sleep(SUPER_SAMPLE_S)
        elapsed = time.perf_counter() - t0
        now = rig.super_charge()
        if now is None:
            unreadable += 1
            continue
        samples.append((round(elapsed, 2), now.state, round(now.charge, 2)))
        if not now.ready:
            after = now
            break

    if after is not None:
        _log(f"  PASS  super drained to {after.state} ({after.charge:.2f}) "
             f"{samples[-1][0]:.2f}s after the tap")
    else:
        trail = ", ".join(f"{t:.2f}s {s} {c:.2f}" for t, s, c in samples) or "nothing readable"
        _log(f"  FAIL  super still ready through {SUPER_DRAIN_WINDOW_S:.1f}s "
             f"({unreadable} frame(s) unreadable): {trail}")
        _log("  -> if you SAW it fire, the tap is fine and this is the reader or the window; if "
             "you did not, the tap missed the button.")
    return {"ok": after is not None, "before": reading, "after": after,
            "samples": samples, "unreadable": unreadable}


def job_move(rig, args) -> dict:
    """Four cardinals, checked against the hero's WORLD displacement.

    World = `camera_relative + odometry.position_tiles`, both terms. This measured only the camera
    until the 2026-09-09 run, where a camera that does not pan horizontally on that map failed both
    x bins at exactly 0.00 tiles while both y bins were perfect -- a venue artifact that looks
    exactly like a broken joystick. `verify_move`'s docstring has the reasoning; the two terms are
    printed separately below so the next such failure reads itself.
    """
    _log(f"driving bins {CARDINAL_BINS}, {args.hold:.1f}s each")
    rig.pose()          # seed the odometry segment before the first trial's baseline
    trials = cal_lib.verify_move(rig.pose, rig.controls.joystick,
                                 bins=CARDINAL_BINS, hold_seconds=args.hold)
    rig.controls.joystick.release()
    worst = 0.0
    for t in trials:
        worst = max(worst, t.max_shift_tiles)
        _log(f"  bin {t.bin:2d} -> ({t.commanded[0]:+.2f}, {t.commanded[1]:+.2f})  "
             f"hero ({t.observed[0]:+6.2f}, {t.observed[1]:+6.2f}) t  "
             f"cos {t.cos:+.3f}  {t.diagnosis()}")
        onset = "never" if t.onset_seconds is None else f"{t.onset_seconds * 1e3:.0f} ms"
        _log(f"          camera ({t.camera[0]:+6.2f}, {t.camera[1]:+6.2f})  "
             f"on-screen ({t.relative[0]:+6.2f}, {t.relative[1]:+6.2f})  "
             f"onset {onset}  "
             f"[{t.frames + 1} samples, {t.seen} with hero, {t.lost} odo not ok, "
             f"max shift {t.max_shift_tiles:.2f} t]")
    onsets = [t.onset_seconds for t in trials if t.onset_seconds is not None]
    if onsets:
        # Latency the sim does not have: `movement.py` applies velocity the tick the action
        # arrives. Reported as a median because one bin ending against a wall skews a mean.
        med = sorted(onsets)[len(onsets) // 2]
        _log(f"  actuation onset: median {med * 1e3:.0f} ms over {len(onsets)} bins "
             f"({', '.join(f'{o * 1e3:.0f}' for o in onsets)} ms) -- "
             f"{med / 0.25:.2f} decision periods")
    bound = rig.vision.cfg.odometry_max_shift_tiles
    _log(f"  worst single-frame shift {worst:.2f} tiles against the {bound:.1f} bound "
         f"({'inside' if worst < bound else 'VIOLATED'})")
    return {"trials": trials, "worst_shift": worst}


JOB_FUNCS = {"buttons": job_buttons, "tap": job_tap, "super": job_super, "move": job_move}


# --- the gadget anchor probe (SIM_OVERHAUL_STEPS.md Step G6.1) ------------------------------

# Seconds scored before the tap, as the baseline every drop is read against, and after it.
PROBE_PRE_ROLL_S = 2.0
PROBE_RECORD_S = 20.0
PROBE_TRACE_PATH = REPO / "runs" / "audit" / "gadget_anchor_trace.json"
# Mortis's `gadget_cooldown` in configs/brawlers.yaml, the game's recharge as the sim models it.
# A trace whose controls vanish before this much of it has passed cannot say KEEP.
GADGET_RECHARGE_S = 18.0
# Which calibrated disc the tap goes to. `Buttons.press(ATTACK_GADGET)` presses `buttons.gadget`,
# which `Controls.build` takes from `cal.button("gadget")`, so this name and the gate anchor are
# independent and must stay so -- conflating them is what made the 2026-09-22 trace inconclusive.
TAPPED_ANCHOR = "gadget"
# The gadget disc's mean colour must move by more than this many levels, in some channel, some
# time after the tap, or the probe cannot tell that the gadget fired. Per channel, so a button the
# game greys out without dimming still counts. A first guess, not a measurement: revise it
# against the first real trace.
UNCHANGED_COLOUR = 8.0


def _pace(deadline: float, tick_seconds: float, clock=time.perf_counter, sleep=time.sleep):
    """`DeployLoop._pace` for a loop that is not a DeployLoop: hold the rate without drift, and
    after an overrun restart from now rather than bursting to catch up."""
    deadline += tick_seconds
    now = clock()
    if deadline <= now:
        return now
    sleep(deadline - now)
    return deadline


def _disc_colour(image, cx: float, cy: float, r: float, frac: float = 0.8) -> list[float]:
    """Mean BGR inside `frac * r` of a button's centre: what the button LOOKS like, which the
    ring score deliberately ignores."""
    rr = r * frac
    h, w = image.shape[:2]
    x0, x1 = max(0, math.floor(cx - rr)), min(w, math.ceil(cx + rr) + 1)
    y0, y1 = max(0, math.floor(cy - rr)), min(h, math.ceil(cy + rr) + 1)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= rr * rr
    return [round(float(c), 1) for c in image[y0:y1, x0:x1][inside].mean(axis=0)]


def _roi_digest(image, cx: float, cy: float, r: float) -> str:
    """A cheap fingerprint of the pixels the ring score reads at this anchor.

    **This exists because the 2026-09-22 trace could not tell "static" from "stale".** Its two
    untouched anchors returned exactly one distinct score and one distinct mean colour across all
    264 ticks, which reads either as a lossless capture of a pixel-for-pixel static disc or as a
    stale ROI, and a score alone cannot separate them. A digest can: identical digests mean
    identical pixels, and a digest that moves under a frozen score is a bug in the scorer.
    """
    pad = math.ceil(r) + 2
    h, w = image.shape[:2]
    x0, x1 = max(0, math.floor(cx) - pad), min(w, math.ceil(cx) + pad + 1)
    y0, y1 = max(0, math.floor(cy) - pad), min(h, math.ceil(cy) + pad + 1)
    roi = np.ascontiguousarray(image[y0:y1, x0:x1])
    return hashlib.blake2b(roi.tobytes(), digest_size=6).hexdigest()


def record_gadget_trace(grab, check_window, gates, buttons, *, gate_name: str,
                        tick_seconds: float, pre_roll_ticks: int, record_ticks: int,
                        check_every: int, recharge_s: float = GADGET_RECHARGE_S,
                        clock=time.perf_counter, sleep=time.sleep) -> dict:
    """Score every gate on every tick, tap the gadget once at tick `pre_roll_ticks`, keep scoring
    to `pre_roll_ticks + record_ticks`. Returns the trace; writing it is the caller's.

    `gates` maps each button to a `MatchState` anchored on it; `gates[gate_name]` is the
    production gate as `wait_for_gate` left it. Each tick runs in `loop.tick`'s order: grab,
    `settle()`, the window check on its cadence, every gate's `update`, and on the tap tick
    alone the press. So the tap reaches the game exactly as the policy's does, down after that
    tick's gate read and lifted by the next tick's settle.

    **The tap is interlocked like every press `loop.py` sends:** when the production gate reads
    out of match at the tap tick, nothing is pressed and the trace says why. A covered window
    stops the recording, since its frames show whatever covers the game, and any other window
    fault raises. After the lift the probe sends nothing; it only watches.
    """
    gate = gates[gate_name]
    ticks, faults = [], []
    tap_tick = aborted = None
    deadline = clock()
    try:
        for i in range(pre_roll_ticks + record_ticks):
            t_start = clock()
            image = grab()
            buttons.settle()
            window = None
            if i % check_every == 0:
                reason = check_window()
                if reason is not None and not reason.recoverable:
                    raise RuntimeError(f"window check failed during the probe: {reason}")
                window = "ok" if reason is None else reason.kind
                if reason is not None:
                    faults.append({"tick": i, "reason": str(reason)})
                    aborted = (f"the emulator was covered at tick {i}, so the frames showed "
                               f"whatever covered it, not the buttons")
                    break
            row = {"i": i, "t": t_start, "window": window, "tapped": False,
                   "score": {}, "in_match": {}, "colour": {}, "roi": {}}
            for name, g in gates.items():
                row["in_match"][name] = g.update(image)
                row["score"][name] = round(float(g.last_score), 4)
                row["colour"][name] = _disc_colour(image, *g.anchor)
                row["roi"][name] = _roi_digest(image, *g.anchor)
            if i == pre_roll_ticks:
                if not gate.in_match:
                    aborted = (f"the {gate_name} gate read out of match at the tap tick, so "
                               f"nothing was pressed")
                elif not buttons.press(ATTACK_GADGET, 0.0):
                    aborted = "the buttons have no gadget centre, so nothing was pressed"
                else:
                    row["tapped"] = True
                    tap_tick = i
            row["tick_ms"] = round((clock() - t_start) * 1e3, 2)
            ticks.append(row)
            if aborted:
                break
            deadline = _pace(deadline, tick_seconds, clock, sleep)
    finally:
        buttons.release()        # a no-op after the normal lift; never leave the tap down

    ref = ticks[tap_tick]["t"] if tap_tick is not None else (ticks[0]["t"] if ticks else 0.0)
    for row in ticks:
        row["t"] = round(row["t"] - ref, 4)
    cal = gate.cal
    return {
        "what": "SIM_OVERHAUL_STEPS.md Step G6.1: every calibrated button's ring score on "
                "every perception tick, before and after one gadget tap",
        "tick_hz": round(1.0 / tick_seconds, 6),
        "pre_roll_ticks": pre_roll_ticks,
        "record_ticks": record_ticks,
        "tap_tick": tap_tick,
        "aborted": aborted,
        "recharge_s": recharge_s,
        "gate_anchor": gate_name,
        "threshold": cal.threshold,
        "enter_samples": cal.enter_samples,
        "exit_samples": cal.exit_samples,
        "anchors": {name: dict(zip(("cx", "cy", "r"), (round(float(v), 2) for v in g.anchor)))
                    for name, g in gates.items()},
        "window_faults": faults,
        "ticks": ticks,
    }


def trace_json(trace: dict) -> str:
    """The trace as JSON with one line per tick: readable, greppable, and still one `json.load`."""
    head = [f" {json.dumps(k)}: {json.dumps(v)}" for k, v in trace.items() if k != "ticks"]
    rows = ",\n".join(f"  {json.dumps(row)}" for row in trace["ticks"])
    return "{\n" + ",\n".join(head + [f' "ticks": [\n{rows}\n ]']) + "\n}\n"


def _runs(flags) -> list[int]:
    """Lengths of the runs of True in `flags`."""
    runs, n = [], 0
    for f in list(flags) + [False]:
        if f:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    return runs


def summarize_trace(trace: dict) -> dict:
    """What G6.2 needs from a trace: numbers per anchor, the 2-of-3 vote, and a verdict. Pure, so
    a stored trace can be summarized again after the match.

    **A tick where EVERY anchor reads under threshold is common-mode:** the controls are gone
    (the hero died, the match ended) or something covered all three at once, and neither is the
    gadget's doing. Common-mode ticks count toward no anchor's dips or floor, and a gate exit on
    one is a true exit. A FALSE exit is a gate going out of match while some other button still
    reads above threshold, which is the failure G6 exists to catch. `covered_s` is the last
    post-tap time the controls were still up.

    The tap tick counts as pre-tap: its frame was grabbed before the press went down.
    """
    thr, hz, gate = trace["threshold"], trace["tick_hz"], trace["gate_anchor"]
    tap, ticks, names = trace["tap_tick"], trace["ticks"], list(trace["anchors"])
    out = {"gate_anchor": gate, "threshold": thr, "tick_hz": hz, "n_ticks": len(ticks),
           "tap_tick": tap, "exit_samples": trace["exit_samples"], "anchors": {},
           "warnings": []}
    if tap is not None:
        pre = [k for k in ticks if k["i"] <= tap]
        post = [k for k in ticks if k["i"] > tap]
        common = {k["i"]: all(k["score"][a] < thr for a in names) for k in post}
        alive = [k for k in post if not common[k["i"]]]
        out["covered_s"] = alive[-1]["t"] if alive else 0.0
        out["common_mode_ticks"] = sum(common.values())

        def exits(in_match):
            """(all exits, false exits) as times, for a per-tick in-match reading."""
            found, prev = ([], []), in_match(ticks[tap])
            for k in post:
                now = in_match(k)
                if prev and not now:
                    found[0].append(k["t"])
                    if not common[k["i"]]:
                        found[1].append(k["t"])
                prev = now
            return found

        for a in names:
            under = [k["score"][a] < thr and not common[k["i"]] for k in post]
            runs = _runs(under)
            live = pre + alive
            low = min(alive, key=lambda k: k["score"][a], default=None)
            colour_pre = np.median([k["colour"][a] for k in pre], axis=0)
            moved = max((float(np.max(np.abs(np.asarray(k["colour"][a]) - colour_pre)))
                         for k in alive), default=0.0)
            all_exits, false_exits = exits(lambda k, a=a: k["in_match"][a])
            floor = min(k["score"][a] for k in live)
            out["anchors"][a] = {
                "refine_score": trace["anchors"][a].get("refine_score"),
                "pre_median": float(np.median([k["score"][a] for k in pre])),
                "post_min": None if low is None else low["score"][a],
                "post_min_t": None if low is None else low["t"],
                "ticks_under": sum(under),
                "longest_under": max(runs, default=0),
                "floor": floor,
                "margin": floor - thr,
                "in_match_at_tap": ticks[tap]["in_match"][a],
                "exits": all_exits,
                "false_exits": false_exits,
                "colour_moved": moved,
            }
        vote_exits, vote_false = exits(lambda k: sum(k["in_match"][a] for a in names) >= 2)
        out["vote"] = {"exits": vote_exits, "false_exits": vote_false}
        for a, st in out["anchors"].items():
            if a != gate and not st["in_match_at_tap"]:
                out["warnings"].append(
                    f"{a} never read in match before the tap (pre-roll median "
                    f"{st['pre_median']:.3f}), so it cannot anchor the gate on this source")
        worst = max(k["tick_ms"] for k in ticks)
        if worst > 1e3 / hz:
            out["warnings"].append(
                f"a tick took {worst:.0f} ms against the {1e3 / hz:.0f} ms period, so some "
                f"samples were further apart than the loop's")
    out["verdict"], out["reason"] = _verdict(trace, out)
    return out


def _verdict(trace: dict, s: dict) -> tuple[str, str]:
    """G6.2's rule, applied: KEEP when the gate anchor never dips, CHANGE when it does, and
    INCONCLUSIVE whenever the trace cannot tell.

    The colour check and the dip check read DIFFERENT anchors on purpose. `TAPPED_ANCHOR` is the
    disc the press went to, so its colour is the only evidence the gadget actually fired; the gate
    anchor is the disc the match gate reads, so its score is the only evidence the gate survived.
    """
    gate, thr, hz = s["gate_anchor"], s["threshold"], s["tick_hz"]
    if trace.get("aborted"):
        return "INCONCLUSIVE", f"the probe stopped early: {trace['aborted']}. Re-run."
    if s["tap_tick"] is None:
        return "INCONCLUSIVE", "no tap was sent. Re-run."
    if s["covered_s"] < trace["recharge_s"]:
        return "INCONCLUSIVE", (
            f"the controls vanished at t=+{s['covered_s']:.1f} s, before the "
            f"{trace['recharge_s']:g} s recharge was over: the hero died or the match ended. "
            f"Re-run and stay alive.")
    g = s["anchors"][gate]
    tapped = s["anchors"].get(TAPPED_ANCHOR)
    if tapped is None:
        return "INCONCLUSIVE", (
            f"no {TAPPED_ANCHOR} anchor was scored, so nothing in this trace can say whether the "
            f"tap landed. Check the calibration's button names.")
    if tapped["colour_moved"] <= UNCHANGED_COLOUR:
        return "INCONCLUSIVE", (
            f"the {TAPPED_ANCHOR} button looks the same after the tap (its colour moved "
            f"{tapped['colour_moved']:.1f} levels at most), so the tap may not have landed or no "
            f"gadget is equipped. Watch for Mortis's throw and re-run.")
    if g["ticks_under"] == 0:
        return "KEEP", (
            f"the {gate} never read under {thr:g} while the other buttons were up, through "
            f"t=+{s['covered_s']:.1f} s. G6.2: keep the anchor and cite this trace in "
            f"match_gate._comment.")
    if g["false_exits"]:
        gate_text = f"its gate EXITED at t=+{g['false_exits'][0]:.2f} s while the match went on"
    elif g["exits"]:
        gate_text = f"its gate exited at t=+{g['exits'][0]:.2f} s"
    else:
        gate_text = f"its gate held, since no run reached exit_samples={s['exit_samples']}"
    held = [a for a, st in s["anchors"].items()
            if a != gate and st["in_match_at_tap"] and st["ticks_under"] == 0]
    best = max(held, key=lambda a: s["anchors"][a]["margin"], default=None)
    alt = (f"The widest alternative anchor is {best} (floor {s['anchors'][best]['floor']:.3f}, "
           f"margin {s['anchors'][best]['margin']:+.3f})." if best else
           "No other button held above the threshold either.")
    vote = s["vote"]
    vote_text = (f"would have exited at t=+{vote['false_exits'][0]:.2f} s" if vote["false_exits"]
                 else "held throughout")
    return "CHANGE", (
        f"the {gate} read under {thr:g} on {g['ticks_under']} post-tap tick(s), longest run "
        f"{g['longest_under']} ({g['longest_under'] / hz:.2f} s), and {gate_text}. G6.2: "
        f"re-anchor or vote. {alt} The 2-of-3 vote {vote_text}.")


def format_summary(s: dict) -> list[str]:
    """`summarize_trace`'s result as report lines."""
    lines = [f"trace: {s['n_ticks']} ticks at {s['tick_hz']:g} Hz, tap at tick {s['tap_tick']}"]
    if s["anchors"]:
        lines.append(f"controls up through t=+{s['covered_s']:.1f} s; every button under "
                     f"{s['threshold']:g} on {s['common_mode_ticks']} tick(s)")
        lines.append("  anchor   refine  pre-med  post-min      under  longest  floor  "
                     "exits          colour moved")
        for a, st in s["anchors"].items():
            refine = "  --  " if st["refine_score"] is None else f"{st['refine_score']:.3f}"
            low = ("   --" if st["post_min"] is None
                   else f"{st['post_min']:.3f} @{st['post_min_t']:+5.1f}s")
            exits = ", ".join(f"{t:+.2f}s" for t in st["exits"]) or "none"
            if st["false_exits"]:
                exits += " FALSE"
            lines.append(f"  {a:7s}  {refine}  {st['pre_median']:.3f}    {low:>13s}  "
                         f"{st['ticks_under']:5d}  {st['longest_under']:7d}  "
                         f"{st['floor']:.3f}  {exits:13s}  {st['colour_moved']:5.1f}")
        vote = s["vote"]
        lines.append("  2-of-3 vote: " + (
            "held throughout" if not vote["false_exits"] else
            "exited at " + ", ".join(f"{t:+.2f}s" for t in vote["false_exits"])))
    lines += [f"WARNING: {w}" for w in s["warnings"]]
    lines.append(f"{s['verdict']}: {s['reason']}")
    return lines


def probe_gadget(rig, cfg, args) -> dict:
    """G6.1 on the live rig: a gate per button, one tap, the trace written. Returns the trace."""
    gate_name = rig.cal.gate_anchor
    image = rig.grab().image
    gates, refined = {gate_name: rig.match}, {gate_name: rig.gate_refine_score}
    for name in rig.cal.buttons:
        if name == gate_name:
            continue
        # The production gate's own class and hysteresis, anchored on another button. A floor
        # of 0 so a button that cannot anchor the gate is recorded as such, not fatal.
        gates[name] = MatchState(replace(rig.cal, gate_anchor=name))
        refined[name] = gates[name].refine(image, min_score=0.0)
    hz = cfg.loop_tick_hz
    _log(f"scoring {', '.join(gates)} every tick at {hz:g} Hz: {PROBE_PRE_ROLL_S:g} s, one "
         f"{TAPPED_ANCHOR} tap, then {PROBE_RECORD_S:g} s (gate on {gate_name}). Hands off "
         f"attack, Super and gadget.")
    trace = record_gadget_trace(
        lambda: rig.grab().image, rig.guard.check, gates, rig.controls.buttons,
        gate_name=gate_name, tick_seconds=1.0 / hz,
        pre_roll_ticks=round(PROBE_PRE_ROLL_S * hz), record_ticks=round(PROBE_RECORD_S * hz),
        check_every=cfg.window_check_every_n_ticks)
    for name, score in refined.items():
        trace["anchors"][name]["refine_score"] = None if score is None else round(float(score), 4)
    trace["recorded"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = Path(args.trace_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(trace_json(trace), encoding="utf-8")
    _log(f"wrote {len(trace['ticks'])} ticks -> {out}")
    return trace


# --- writing --------------------------------------------------------------------------------

def write_buttons(rig, fits) -> None:
    """Store the re-fitted button table, converted back to device pixels."""
    if not fits:
        _log("nothing to write: no button was re-fitted")
        return
    buttons = dict(rig.cal.buttons)
    scale = rig.cal.device_to_viewport
    for name, fit in fits.items():
        buttons[name] = fit.as_json(scale)
    provenance = [
        "MEASURED by scripts/deploy_calibrate.py against the LIVE mss capture, from a temporal",
        f"median of {cal_lib.MEDIAN_FRAMES} gameplay frames at the deployment viewport, then",
        "converted back to device pixels. This is the source the match gate actually reads, which",
        "the previous fit (an OBS recording's median) was not -- see BRAWL_DEPLOYMENT_DESIGN.md",
        "4.5 on radii not transferring between frame sources.",
        "",
        "CENTRES are authoritative. RADII are still advisory: MatchState.refine() re-fits the",
        "gate anchor's radius on the first gameplay frame of every run, and that stays true even",
        "when the stored radius came from the same kind of source.",
        "Refitted: " + ", ".join(f"{n} score {f.score:.3f} shift {f.shift_px:.1f}px"
                                 for n, f in sorted(fits.items())),
    ]
    cal_lib.update_calibration(CALIBRATION_PATH, blocks={"buttons": buttons},
                               provenance={"buttons": provenance})
    _log(f"wrote buttons -> {CALIBRATION_PATH}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs" / "deployment.yaml"))
    ap.add_argument("--jobs", default=None,
                    help=f"comma-separated subset of {JOBS}, run in this order (default: all)")
    ap.add_argument("--trials", type=int, default=3, help="attack taps to verify")
    ap.add_argument("--hold", type=float, default=1.2, help="seconds to hold each movement bin")
    ap.add_argument("--wait", type=float, default=180.0,
                    help="seconds to wait for the match gate before giving up")
    ap.add_argument("--write", action="store_true",
                    help="store the measured blocks into data/control_calibration.json")
    ap.add_argument("--probe-gadget", action="store_true",
                    help="SIM_OVERHAUL_STEPS.md G6.1: tap the gadget once and record every "
                         "button's ring score for 20 s. Runs alone")
    ap.add_argument("--trace-out", default=str(PROBE_TRACE_PATH),
                    help="where --probe-gadget writes its trace")
    args = ap.parse_args(argv)

    if args.probe_gadget and (args.jobs is not None or args.write):
        # A job presses attack or Super, each an anchor the trace measures, and --write stores
        # buttons the probe never fits.
        ap.error("--probe-gadget runs alone: drop --jobs and --write")
    jobs = [] if args.probe_gadget else [
        j.strip() for j in (args.jobs or ",".join(JOBS)).split(",") if j.strip()]
    unknown = [j for j in jobs if j not in JOB_FUNCS]
    if unknown:
        ap.error(f"unknown job(s) {unknown}; choose from {JOBS}")

    cfg = load_deployment_config(args.config)
    validate(cfg)
    rig = Rig(cfg)
    _log(f"capture {rig.capture.viewport[0]}x{rig.capture.viewport[1]}; "
         f"waiting up to {args.wait:.0f}s for the match gate")

    results, failures, probe = {}, [], None
    try:
        with rig.capture:
            if not rig.wait_for_gate(args.wait):
                _log("no match started -- nothing was measured and no input was sent")
                return 2
            if args.probe_gadget:
                _log("--- probe-gadget ---")
                try:
                    probe = probe_gadget(rig, cfg, args)
                except Exception as exc:
                    _log(f"  probe-gadget raised: {type(exc).__name__}: {exc}")
                    failures.append("probe-gadget")
            for name in jobs:
                _log(f"--- {name} ---")
                if not rig.match.in_match:
                    _log(f"the match ended before `{name}` ran; stopping here")
                    break
                try:
                    results[name] = JOB_FUNCS[name](rig, args)
                except Exception as exc:            # one bad job must not cost the others
                    _log(f"  {name} raised: {type(exc).__name__}: {exc}")
                    failures.append(name)
    finally:
        # Same guarantee `loop.run` gives: whatever happened, no contact is left down.
        rig.controls.release_all()
        _log("released every contact")

    if "tap" in results and not results["tap"]["report"].ok:
        failures.append("tap")
    if "move" in results and not all(t.ok for t in results["move"]["trials"]):
        failures.append("move")
    if "super" in results and results["super"].get("ok") is False:
        failures.append("super")

    if probe is not None:
        summary = summarize_trace(probe)
        for line in format_summary(summary):
            _log(line)
        if summary["verdict"] == "INCONCLUSIVE":
            failures.append("probe-gadget")

    if args.write and "buttons" in results:
        write_buttons(rig, results["buttons"]["fits"])
    elif args.write:
        _log("--write given but the buttons job did not run; nothing written")

    _log(f"done: {len(results) + (probe is not None)} job(s) run"
         + (f", FAILED: {', '.join(sorted(set(failures)))}" if failures else ", all passed"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
