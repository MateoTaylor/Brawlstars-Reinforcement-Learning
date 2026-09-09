"""Run the trained policy against a live Nulls Brawl match in BlueStacks.

    python scripts/deploy_run.py --dry-run            # everything except touching the device
    python scripts/deploy_run.py                      # for real
    python scripts/deploy_run.py --matches 0          # keep playing until Ctrl-C

**Start it before you queue the battle.** It waits on the match gate, emits nothing in the menus,
and takes the movement contact the moment the gate opens -- so there is nothing to time. The gate's
radius is refined on the first gameplay frame it sees, which is why it must not be started on one.

**Then alt-tab back to BlueStacks and leave it on top.** A window drawn over the fullscreen
emulator -- your terminal, a notification toast -- is captured instead of the game, so the loop
PAUSES on it: contacts released, still capturing, and it resumes by itself when the emulator is
visible again. That is recoverable and costs only the covered seconds. Minimizing BlueStacks, or
moving/resizing it, is NOT: those stop the run for good, because the content box was computed once
at startup and would then be cropping the wrong pixels. Turning on focus assist / do-not-disturb
for the session is worth it.

**`--dry-run` is the first thing to run on a new setup.** It builds the whole stack -- window,
capture, both detectors, the checkpoint, the calibration -- and drives every decision through a
`NullBackend`, so a geometry or weights problem surfaces without the agent flailing. The console
line is the same either way.

Ctrl-C is a clean stop: the loop's `finally` releases every contact on the way out, so the joystick
never stays stuck down.
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from brawl_deployment.config import load_deployment_config, validate           # noqa: E402
from brawl_deployment.loop import Controls, DeployLoop, Phase                  # noqa: E402


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _summarise(loop: DeployLoop) -> None:
    """Per-tick timing and what the loop actually did, from the telemetry ring.

    The tail is what matters, not the mean: the 50 ms budget exists to keep
    `odometry.max_shift_tiles` satisfied, and that bound is per frame (§7.1).
    """
    rows = list(loop.telemetry)
    if not rows:
        _log("no ticks recorded")
        return
    # The warm-up tick is excluded from the tail, not hidden: it pays ~1 s of PTX JIT before the
    # gate opens, precisely so no gameplay tick does, and leaving it in reports a 758 ms max
    # against an 83 ms budget -- which buries the p95 that the rate choice actually rests on.
    warm = [r for r in rows if r.warmup]
    # PLAYING ticks only. A paused or waiting tick grabs and returns without perception, so it is
    # a few milliseconds -- mixing them in halves the median and makes the budget look roomier
    # than it is. The first live run spent 183 of 555 ticks paused and reported a 30.7 ms median
    # against a real in-match p95 of 65.9.
    timed = [r for r in rows if not r.warmup and r.phase == "playing"]
    idle = len(rows) - len(timed) - len(warm)
    if not timed:
        _log(f"{len(rows)} ticks, none in a match -- no timing to report")
        return
    ticks = sorted(r.tick_ms for r in timed)
    grabs = sorted(r.grab_ms for r in timed)
    decisions = [r for r in rows if r.decision]
    skipped = [r for r in rows if r.note and not r.decision]

    def p(values, q):
        return values[min(len(values) - 1, int(q * len(values)))]

    _log(f"{len(rows)} ticks, {len(decisions)} decisions, {len(skipped)} held")
    _log(f"tick ms  median {p(ticks, 0.5):5.1f}  p95 {p(ticks, 0.95):5.1f}  max {ticks[-1]:5.1f}"
         f"   (budget {loop.tick_seconds * 1e3:.0f}; {len(timed)} in-match ticks"
         f"{f', {idle} idle excluded' if idle else ''})")
    _log(f"grab ms  median {p(grabs, 0.5):5.1f}  p95 {p(grabs, 0.95):5.1f}  max {grabs[-1]:5.1f}"
         f"   (of the tick above)")
    for r in warm:
        _log(f"warm-up tick {r.tick_ms:.0f} ms (detector JIT, once, before the gate) -- excluded")
    for note in sorted({r.note for r in skipped}):
        _log(f"  held {sum(1 for r in skipped if r.note == note):4d}x  {note}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs" / "deployment.yaml"))
    ap.add_argument("--dry-run", action="store_true",
                    help="build everything and decide for real, but send no touches")
    ap.add_argument("--matches", type=int, default=1,
                    help="stop after N matches; 0 waits for the next one indefinitely")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="wall-clock ceiling on the whole run, including time spent waiting")
    ap.add_argument("--telemetry", default=None, help="write the tick ring to this CSV on exit")
    args = ap.parse_args(argv)

    cfg = load_deployment_config(args.config)
    validate(cfg)
    _log(f"config {args.config}")
    _log(f"run    {cfg.run_dir} ({cfg.run_checkpoint}) on {cfg.policy_device}")

    loop = DeployLoop.from_config(cfg, log=_log)
    if args.dry_run:
        # Swap the backend rather than adding a flag the loop has to remember to check. There is
        # then no code path in which a dry run can touch the device, which a boolean cannot promise.
        from brawl_deployment.control.backend import NullBackend

        null = NullBackend()
        loop.controls = Controls(backend=null,
                                 joystick=type(loop.controls.joystick)(
                                     null, loop.controls.joystick.anchor,
                                     loop.controls.joystick.radius_px,
                                     n_bins=loop.controls.joystick.n_bins),
                                 buttons=type(loop.controls.buttons)(
                                     null, loop.controls.buttons.attack,
                                     loop.controls.buttons.super_))
        _log("DRY RUN -- decisions are real, touches go nowhere")

    _log(f"capture {loop.capture.viewport[0]}x{loop.capture.viewport[1]} from monitor "
         f"{loop.guard.monitor_index}; waiting for the match gate")
    _log(f"rates: {1.0 / loop.tick_seconds:.0f} Hz perception, "
         f"{1.0 / (loop.tick_seconds * loop.decision_every):.0f} Hz decisions")

    try:
        reason = loop.run(max_seconds=args.max_seconds,
                          max_matches=None if args.matches == 0 else args.matches)
    except KeyboardInterrupt:
        # `run`'s own `finally` has already released every contact by the time this lands.
        reason = "interrupted"
    _log(f"stopped: {reason}")

    _summarise(loop)
    if args.telemetry:
        loop.write_csv(args.telemetry)
        _log(f"telemetry -> {args.telemetry}")
    return 0 if loop.phase is Phase.STOPPED else 1


if __name__ == "__main__":
    raise SystemExit(main())
