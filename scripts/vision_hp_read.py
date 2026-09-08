"""Read HP out of a clip and write an annotated video, or just report what happened.

    python scripts/vision_hp_read.py tests/fixtures/vision/day10_gameplay.mp4 -o hp.mp4
    python scripts/vision_hp_read.py clip.mp4 --stop 3000            # no video, just the tally
    python scripts/vision_hp_read.py clip.mp4 --raw -o hp.mp4        # skip the tracker
    python scripts/vision_hp_read.py clip.mp4 --trace player         # print the HP sequence

**The counterpart to `scripts/vision_detect.py`, one stage further on.** That one runs the
detector alone; this adds the HP reader on top and nothing else -- no homography, no odometry, no
terrain. If the numbers are wrong here, nothing that consumes them is worth debugging.

`--trace` is the one to reach for first. A printed HP sequence is far more diagnostic than a
video, because the failure this stage is built against is a value that is wrong while looking
entirely reasonable, and it is obvious in a column and invisible in a frame.
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brawl_vision.config import load_vision_config
from brawl_vision.object_detection import ObjectDetector, draw_detections
from brawl_vision.object_detection.hp_detection import HealthReader, HealthTracker
from brawl_vision.object_detection.hp_detection.draw import draw_health, draw_summary
from brawl_vision.sources import is_live, open_source
from brawl_vision.video import VideoSink


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", help="a gameplay .mp4, or 'screen' / 'screen:N' for live")
    p.add_argument("-o", "--out", default=None, help="output .mp4; omit for the tally alone")
    p.add_argument("--raw", action="store_true",
                   help="report the per-frame reader without the tracker -- what one frame can "
                        "see on its own, which is what the tracker's value is measured against")
    p.add_argument("--trace", default=None, choices=("player", "enemy", "all"),
                   help="print the HP sequence for this class instead of drawing")
    p.add_argument("--fps", type=float, default=4.0,
                   help="OUTPUT rate. Default 4, the DECISION rate (sim.action_repeat 5 at 20 Hz) "
                        "the tracker's frame counts assume. Raise it for something watchable and "
                        "set --cv-fps back down to keep the perception honest")
    p.add_argument("--cv-fps", type=float, default=None,
                   help="how often the detector and HP reader actually RUN, held and redrawn in "
                        "between (default: --fps). Pair a high --fps for smooth playback with a "
                        "low --cv-fps for the pipeline's real budget, e.g. --fps 24 --cv-fps 6. "
                        "Held boxes lag the sprite by design -- that is what a held detection is")
    p.add_argument("--width", type=int, default=1280, help="output width (default: 1280)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    cfg = load_vision_config(args.config) if args.config else load_vision_config()
    try:
        detector = ObjectDetector.from_config(cfg)
        reader = HealthReader.from_config(cfg)
    except (FileNotFoundError, ImportError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 2
    tracker = None if args.raw else HealthTracker.from_config(cfg, reader=reader)

    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))
    say(f"{detector.path.name} on {detector.provider}; templates from "
        f"{reader.bank.meta.get('sources', ['?'])}, {reader.bank.meta.get('n_glyphs', '?')} "
        f"glyphs")
    if args.trace:
        print(f"{'frame':>7} {'track':>5} {'label':>7} {'hp':>7} {'conf':>5} {'raw':>7} status")

    cv_fps = args.fps if args.cv_fps is None else args.cv_fps
    if cv_fps <= 0 or cv_fps > args.fps:
        # Above --fps the extra results land between written frames and are thrown away, which
        # reads as a performance knob and is really just waste.
        print(f"--cv-fps must be in (0, --fps]; got {cv_fps:g} against --fps {args.fps:g}",
              file=sys.stderr)
        return 2
    statuses: Counter = Counter()
    trusted = total = n_in = n_cv = n_out = 0
    sink = None
    next_t = next_cv_t = None
    step = 1.0 / max(args.fps, 1e-6)
    cv_step = 1.0 / max(cv_fps, 1e-6)
    # Held between CV frames and redrawn onto each written frame. No staleness correction is
    # needed or wanted here: these are SCREEN-space boxes on a picture of the screen, so a held
    # box belongs at its own screen position, lagging the sprite. The map panels in
    # terrain/evaluate.py DO need one -- see `_held_origin` there -- because they are world space.
    dets: list = []
    readings: list = []
    t0 = time.perf_counter()
    hp_seconds = 0.0
    try:
        with open_source(args.clip, cfg) as source:
            if args.stop is None and is_live(source) and args.out:
                say("   live capture with no --stop: ctrl-C to finish the file")
            for frame in source:
                if frame.index < args.start:
                    continue
                if args.stop is not None and frame.index > args.stop:
                    break
                if next_t is None:
                    next_t = next_cv_t = frame.t
                if frame.t < next_t:
                    continue
                next_t += step
                n_in += 1

                cv_due = frame.t >= next_cv_t
                if cv_due:
                    next_cv_t += cv_step
                    n_cv += 1
                    dets = detector.predict(frame.image)
                    t_hp = time.perf_counter()
                    if tracker is None:
                        readings = reader.read_all(frame.image, dets)
                    else:
                        readings = tracker.update(frame.image, dets)
                    hp_seconds += time.perf_counter() - t_hp

                # Tallied per CV frame, not per written frame: a held frame carries the same
                # readings, and counting those again would inflate every rate in the summary.
                for r in (readings if cv_due else ()):
                    raw = getattr(r, "raw", r)
                    statuses[raw.status] += 1
                    total += 1
                    if r.trusted(cfg.hp_min_confidence):
                        trusted += 1
                    if args.trace and (args.trace == "all"
                                       or raw.detection.label == args.trace):
                        print(f"{frame.index:7d} {getattr(r, 'track_id', -1):5d} "
                              f"{raw.detection.label:>7} "
                              f"{('--' if r.hp is None else r.hp):>7} {r.confidence:5.2f} "
                              f"{('--' if raw.hp is None else raw.hp):>7} {raw.status}")

                if args.out is None:
                    continue
                canvas = draw_health(draw_detections(frame.image, dets, show_confidence=False),
                                     readings)
                draw_summary(canvas, readings)
                h = max(2, int(round(canvas.shape[0] * args.width / canvas.shape[1])))
                canvas = cv2.resize(canvas, (args.width, h), interpolation=cv2.INTER_AREA)
                canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)     # VideoSink takes RGB
                if sink is None:
                    sink = VideoSink(args.out, (canvas.shape[1], canvas.shape[0]),
                                     fps=args.fps).__enter__()
                sink.write(canvas)
                n_out += 1
                if not args.quiet and not args.trace and n_in % 50 == 0:
                    print(f"   {n_in:5d} frames  t={frame.t:7.2f}s", flush=True)
    except KeyboardInterrupt:
        say("\ninterrupted -- finishing the file with what was written")
    finally:
        if sink is not None:
            sink.__exit__(None, None, None)

    if n_in == 0:
        print("the source yielded no frames in that range", file=sys.stderr)
        return 2
    if args.out:
        say(f"wrote {args.out}  {n_out} frames")
    say(f"\n{n_in} written, {n_cv} CV frames, {time.perf_counter() - t0:.1f}s total, "
        f"{hp_seconds / max(n_cv, 1) * 1e3:.2f} ms per CV frame in the HP stage")
    say(f"   {total} boxes over {n_cv} CV frames, {trusted} trusted at confidence >= "
        f"{cfg.hp_min_confidence} "
        f"({trusted / max(total, 1):.1%})")
    for status, n in statuses.most_common():
        say(f"      {status:<14} {n:5d}  {n / max(total, 1):6.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
