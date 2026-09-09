"""Run the entity detector alone over a clip and write an annotated video.

    python scripts/vision_detect.py tests/fixtures/vision/showdown_alternate_map.mp4 -o boxes.mp4
    python scripts/vision_detect.py match.mp4 -o boxes.mp4 --conf 0.35 --ignore ''
    python scripts/vision_detect.py match.mp4 --stop 300          # no video, just the tally
    python scripts/vision_detect.py --list-classes

**Deliberately the whole pipeline for this chunk: frames in, boxes out, nothing else.** No
homography, no odometry, no terrain classifier, no trained checkpoint of ours. That is what makes
it the right first thing to run -- if the boxes look wrong here, nothing downstream is worth
debugging, and none of the other stages can be to blame.

`scripts/vision_evaluate.py --detect` is the other one: the same detector, drawn beside the
reconstructed map, for looking at both at once.

`--projectiles` adds OUR trained projectile model on top, in magenta; `--no-entities` drops the
third-party one so you get the projectile model ALONE over the raw footage. That combination is
the manual-evaluation tool for a fresh training run -- boxes on a clip the model never saw, and
the tally at the end for sweeping `--projectile-conf`:

    # what a new checkpoint does on new footage
    python scripts/vision_detect.py match.mp4 -o shots.mp4 --projectiles --no-entities

    # sweep the threshold with no video written -- boxes/frame is the number to watch
    python scripts/vision_detect.py match.mp4 --projectiles --no-entities --projectile-conf 0.15
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

from brawl_vision.config import load_vision_config
from brawl_vision.object_detection import ObjectDetector, draw_detections
from brawl_vision.object_detection.draw import draw_summary
from brawl_vision.object_detection.projectile_detection import (ProjectileDetector,
                                                                draw_projectiles)
from brawl_vision.sources import is_live, open_source
from brawl_vision.video import VideoSink


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", nargs="?", help="a gameplay .mp4, or 'screen' / 'screen:N' for live")
    p.add_argument("-o", "--out", default=None,
                   help="output .mp4 (or .gif). Omit to run the detector and only report the "
                        "tally -- which is the fast way to sweep --conf")
    p.add_argument("--conf", type=float, default=None, help="override detector.conf")
    p.add_argument("--ignore", default=None,
                   help="comma-separated classes to drop, overriding detector.ignore. Pass '' to "
                        "keep all of them")
    p.add_argument("--projectiles", action="store_true",
                   help="also run OUR trained projectile detector and draw its boxes in magenta")
    p.add_argument("--no-entities", action="store_true",
                   help="skip the third-party entity detector entirely. Only useful with "
                        "--projectiles, and it is the honest way to look at the projectile model "
                        "on its own -- brawler boxes are large and busy and will hide a 30 px "
                        "shot box behind them")
    p.add_argument("--projectile-conf", type=float, default=None,
                   help="override projectile.conf (default: 0.30). This model is NMS-free, so "
                        "there is no objectness gate ahead of this threshold and it is the only "
                        "thing thinning 300 rows per frame")
    p.add_argument("--projectile-model", default=None,
                   help="path to a specific .onnx, bypassing projectile.model -- point it at a "
                        "run's weights/best.onnx to compare runs without promoting either")
    p.add_argument("--fps", type=float, default=12.0,
                   help="OUTPUT rate, sampled on the clip's own timestamps (default: 12)")
    p.add_argument("--width", type=int, default=1280,
                   help="output width; the source is ~2002 px wide and a 1:1 copy is a big file "
                        "for no extra legibility (default: 1280)")
    p.add_argument("--start", type=int, default=0, help="first frame index")
    p.add_argument("--stop", type=int, default=None, help="last frame index, inclusive")
    p.add_argument("--no-trim", action="store_true",
                   help="read the whole file instead of stopping where ClipReader thinks "
                        "gameplay ends. That detector is tuned to drop outros and full-screen "
                        "character art, and on an EDITED clip it will take real footage with "
                        "them -- edited_day14_broll loses its last 195 frames. Recordings only")
    p.add_argument("--config", default=None, help="a vision.yaml to use instead of the default")
    p.add_argument("--list-classes", action="store_true",
                   help="print the model's own class list, read from the ONNX metadata, and exit")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.no_entities and not args.projectiles:
        p.error("--no-entities with no --projectiles leaves nothing to run")
    if args.no_trim and args.clip is not None and str(args.clip).startswith("screen"):
        p.error("--no-trim is a recording option -- a live source has no tail to trim")

    cfg = load_vision_config(args.config) if args.config else load_vision_config()
    detector = projectiles = None
    try:
        if not args.no_entities:
            overrides = {}
            if args.conf is not None:
                overrides["conf"] = args.conf
            if args.ignore is not None:
                overrides["ignore"] = tuple(c for c in args.ignore.split(",") if c)
            detector = ObjectDetector.from_config(cfg, **overrides)
        if args.projectiles:
            # Second, so that a missing checkpoint -- the common case until a run finishes --
            # reports itself after the entity model has already proved the ONNX path works.
            overrides = {}
            if args.projectile_conf is not None:
                overrides["conf"] = args.projectile_conf
            if args.projectile_model is not None:
                overrides["path"] = args.projectile_model
            projectiles = ProjectileDetector.from_config(cfg, **overrides)
    except (FileNotFoundError, ImportError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.list_classes:
        # From the file, not from a constant -- V1 and V2 order these differently, and that is
        # the whole reason this flag exists.
        for model in (m for m in (detector, projectiles) if m is not None):
            ignore = getattr(model, "ignore", ())
            for index, name in sorted(model.names.items()):
                print(f"{index}: {name}" + ("   [ignored]" if name in ignore else ""))
        return 0

    if args.clip is None:
        p.error("a clip (or 'screen') is required unless --list-classes is given")
    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))
    if detector is not None:
        say(f"{detector.path.name} on {detector.provider}, classes {detector.classes}, "
            f"conf {detector.conf}" + (f", ignoring {sorted(detector.ignore)}"
                                       if detector.ignore else ""))
    if projectiles is not None:
        say(f"{Path(projectiles.path).name} on {projectiles.provider}, classes "
            f"{projectiles.classes}, conf {projectiles.conf}   [projectiles, magenta]")

    counts: dict[str, int] = {}
    n_in = n_out = 0
    sink = None
    next_t = None
    step = 1.0 / max(args.fps, 1e-6)
    t0 = time.perf_counter()
    infer_seconds = 0.0
    try:
        # Only for a file: `open_source` hands kwargs straight to the source it built, and
        # `ScreenCapture` has no `trim_tail` to receive.
        opts = {"trim_tail": False} if args.no_trim else {}
        with open_source(args.clip, cfg, **opts) as source:
            if args.stop is None and is_live(source) and args.out:
                say("   live capture with no --stop: ctrl-C to finish the file")
            for frame in source:
                if frame.index < args.start:
                    continue
                if args.stop is not None and frame.index > args.stop:
                    break
                if next_t is None:
                    next_t = frame.t
                if frame.t < next_t:
                    continue
                next_t += step
                n_in += 1
                dets: list = []
                pdets: list = []
                t_infer = time.perf_counter()
                if detector is not None:
                    dets = detector.predict(frame.image)
                if projectiles is not None:
                    pdets = projectiles.predict(frame.image)
                infer_seconds += time.perf_counter() - t_infer
                for det in (*dets, *pdets):
                    counts[det.label] = counts.get(det.label, 0) + 1
                if args.out is None:
                    continue
                # Projectiles UNDER the entity boxes: a shot leaving a brawler's gun overlaps that
                # brawler's box, and the small marker is the one that has to survive the overlap.
                canvas = frame.image
                if pdets:
                    canvas = draw_projectiles(canvas, pdets, show_confidence=True)
                if detector is not None:
                    canvas = draw_detections(canvas, dets)
                elif not pdets:
                    canvas = canvas.copy()      # draw_summary writes in place
                canvas = draw_summary(canvas, [*dets, *pdets])
                h = max(2, int(round(canvas.shape[0] * args.width / canvas.shape[1])))
                canvas = cv2.resize(canvas, (args.width, h), interpolation=cv2.INTER_AREA)
                canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)   # VideoSink takes RGB
                if sink is None:
                    sink = VideoSink(args.out, (canvas.shape[1], canvas.shape[0]),
                                     fps=args.fps).__enter__()
                sink.write(canvas)
                n_out += 1
                if not args.quiet and n_in % 50 == 0:
                    print(f"   {n_in:5d} frames  t={frame.t:7.2f}s", flush=True)
    except KeyboardInterrupt:
        say("\ninterrupted -- finishing the file with what was written")
    finally:
        if sink is not None:
            sink.__exit__(None, None, None)

    if n_in == 0:
        print("the source yielded no frames in that range", file=sys.stderr)
        return 2
    elapsed = time.perf_counter() - t0
    if args.out:
        say(f"wrote {args.out}  {n_out} frames")
    say(f"   {n_in} frames, {elapsed:.1f}s total, {infer_seconds / n_in * 1e3:.1f} ms/frame in "
        f"the detector")
    total = sum(counts.values())
    say(f"   {total} boxes ({total / n_in:.2f}/frame): "
        + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
