"""Measure and verify the control calibration against a live match. Run it during one battle.

    python scripts/deploy_calibrate.py                  # every job, report only
    python scripts/deploy_calibrate.py --write          # ...and store the re-fitted buttons
    python scripts/deploy_calibrate.py --jobs tap       # just the one you care about

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

**It plays badly on purpose.** It stands still, taps, and walks in four straight lines. Losing the
match is fine and expected; the point is that every action is one whose consequence is checkable.

Nothing is written unless `--write` is passed, and then only the blocks this run measured -- the
joystick's saturation radius was measured by hand in 4.3 and no script here can reproduce it.
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from brawl_deployment.config import load_deployment_config, validate            # noqa: E402
from brawl_deployment.control import ATTACK_FIRE, ATTACK_SUPER                  # noqa: E402
from brawl_deployment.control import calibration as cal_lib                     # noqa: E402
from brawl_deployment.loop import Controls, VisionStack                         # noqa: E402
from brawl_deployment.match_state import CALIBRATION_PATH, Calibration, MatchState   # noqa: E402

JOBS = ("buttons", "tap", "super", "move")

# Bins driven by the `move` job: right, down, left, up at 16 bins. Cardinals rather than diagonals
# so a sign error on either axis is unmistakable in the printed vector rather than shared between
# both components. `+1` is `hero.decode_action`'s offset -- action 0 is idle.
CARDINAL_BINS = (1, 5, 9, 13)


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
        self.vision = VisionStack.build()
        # n_move_bins is 16 here rather than read from the checkpoint, and the checkpoint is not
        # loaded at all: this script issues no policy actions, so pulling in a 300M-step run to
        # read one integer would be cost with no signal. `CARDINAL_BINS` assumes the same 16.
        self.controls = Controls.build(self.cal, cfg, n_move_bins)
        self.frame = None

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
                score = self.match.refine(self.frame.image)
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
    """The headline check: does `control_attack_tap` consume ammo?"""
    _log(f"tapping attack at {rig.controls.buttons.attack} x{args.trials}")
    report = cal_lib.verify_tap(rig.ammo, rig.controls.buttons,
                                action=ATTACK_FIRE, trials=args.trials)
    _log("  " + report.summary())
    for t in report.trials:
        if t.note:
            _log(f"    note: {t.note}")
    if not report.ok:
        _log("  -> control_attack_tap is NOT firing. Something is on that point, or the touch is "
             "not reaching the game. Check `adb shell getevent` while re-running.")
    return {"report": report}


def job_super(rig, args) -> dict:
    """Same check for the Super, which taps the button table rather than a clearance point.

    Best-effort by design: the Super may simply not be charged during the window this runs in, and
    "not charged" is not a failure of the tap. It is reported as skipped, never as a pass.
    """
    reading = rig.super_charge()
    if reading is None or not reading.ready:
        state = "unreadable" if reading is None else reading.state
        _log(f"  super not ready ({state}) -- skipped, not failed")
        return {"skipped": state}
    _log(f"tapping super at {rig.controls.buttons.super_} (charge {reading.charge:.2f}, ready)")
    rig.controls.buttons.tap(ATTACK_SUPER)
    time.sleep(cal_lib.SETTLE_S)
    rig.controls.buttons.settle()
    time.sleep(0.8)
    after = rig.super_charge()
    ok = after is not None and not after.ready
    _log(f"  {'PASS' if ok else 'FAIL'}  super after the tap: "
         f"{'unreadable' if after is None else after.state}")
    return {"ok": ok, "before": reading, "after": after}


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
    ap.add_argument("--jobs", default=",".join(JOBS),
                    help=f"comma-separated subset of {JOBS}, run in this order")
    ap.add_argument("--trials", type=int, default=3, help="attack taps to verify")
    ap.add_argument("--hold", type=float, default=1.2, help="seconds to hold each movement bin")
    ap.add_argument("--wait", type=float, default=180.0,
                    help="seconds to wait for the match gate before giving up")
    ap.add_argument("--write", action="store_true",
                    help="store the measured blocks into data/control_calibration.json")
    args = ap.parse_args(argv)

    jobs = [j.strip() for j in args.jobs.split(",") if j.strip()]
    unknown = [j for j in jobs if j not in JOB_FUNCS]
    if unknown:
        ap.error(f"unknown job(s) {unknown}; choose from {JOBS}")

    cfg = load_deployment_config(args.config)
    validate(cfg)
    rig = Rig(cfg)
    _log(f"capture {rig.capture.viewport[0]}x{rig.capture.viewport[1]}; "
         f"waiting up to {args.wait:.0f}s for the match gate")

    results, failures = {}, []
    try:
        with rig.capture:
            if not rig.wait_for_gate(args.wait):
                _log("no match started -- nothing was measured and no input was sent")
                return 2
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

    if args.write and "buttons" in results:
        write_buttons(rig, results["buttons"]["fits"])
    elif args.write:
        _log("--write given but the buttons job did not run; nothing written")

    _log(f"done: {len(results)} job(s) run"
         + (f", FAILED: {', '.join(sorted(set(failures)))}" if failures else ", all passed"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
