"""Run a recorded clip (or the live screen) through the terrain pipeline and watch every stage.

    python scripts/vision_watch.py tests/fixtures/vision/counted_walking.mp4
    python scripts/vision_watch.py tests/fixtures/vision/standstill.mp4 --start 100 --stop 400
    python scripts/vision_watch.py clip.mp4 --save out.mp4 --fps 15   # headless render
    python scripts/vision_watch.py --live                             # grab this screen instead
    python scripts/vision_watch.py screen:2 --stop 600                # a specific monitor

The source is opened through `brawl_vision.sources.open_source`, so a recording and the live
screen are the same object to everything downstream -- nothing below this script knows or cares
which one it got.

This is the debugging tool the rest of the plan is built with, not a demo: Phases C, D, F, G, H
and I are all far faster to check by looking than by reading numbers. Space pauses.

**Every stage is now built.** The occupancy panel needs a trained terrain classifier -- pass
`--terrain <checkpoint>` -- and stays empty without one, since there is nothing to accumulate until
Phase H has labels. Capture, rectification, odometry and zone detection need nothing.

The panel worth watching first is `capture`, which carries two projected tile grids. The cyan
dashed one is the WALL-TOP plane and should sit on the block-top seams you can see; the solid
yellow one is the GROUND plane and sits ~0.88 tiles off it, because that is how tall a wall block
is in projected terms. Yellow not lining up with block tops is correct. Either grid *shearing*
away from the block edges as it crosses the frame is not, and means the calibration has drifted.
"""
import argparse
import sys
import time

import matplotlib

from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
from brawl_vision.config import load_vision_config
from brawl_vision.sources import open_source
from brawl_vision.terrain.odometry import Odometry
from brawl_vision.terrain.occupancy import UNKNOWN, OccupancyMap
from brawl_vision.terrain.zone import detect_zone
from brawl_vision.terrain.overlay import StageFrame, TerrainOverlay


def _stages(frames, plan, odometry, cfg, occupancy, classifier=None):
    """Turn `Frame`s into `StageFrame`s, timing each stage as it goes.

    Timings are reported per stage rather than as a total because the question this tool exists to
    answer about performance is "which stage got slow", and a total cannot answer it.
    """
    for frame in frames:
        t0 = time.perf_counter()
        rect = plan.rectify(frame.image)
        t1 = time.perf_counter()
        odo = odometry.update(rect)
        t2 = time.perf_counter()
        zone = detect_zone(rect, plan, cfg)
        t3 = time.perf_counter()
        grid = None
        if classifier is not None:
            cells, _ = classifier.predict(rect, plan)
            occupancy.update(cells, odo, plan, zone=zone.at_least(0.05), cfg=cfg)
            grid = _occupancy_view(occupancy)
        t4 = time.perf_counter()
        yield StageFrame(
            index=frame.index, t=frame.t, raw=frame.image, rect=rect,
            # A cut leaves the position untouched and opens a new segment; plotting the track
            # across it would draw a straight line through a discontinuity that never happened.
            camera_tile=odo.position_tiles if odo.status != "lost" else None,
            zone=zone.pixels,
            occupancy=grid,
            timings={"rectify": (t1 - t0) * 1e3, "odometry": (t2 - t1) * 1e3,
                     "zone": (t3 - t2) * 1e3, "classify+map": (t4 - t3) * 1e3},
        )


def _occupancy_view(occupancy):
    """The observed region of the map, as Tile ids with UNKNOWN preserved.

    Cropped to what has actually been seen: a 128x128 grid is mostly UNKNOWN all match, and
    rendering it whole would show a magenta field with a postage stamp of map in the middle.
    """
    import numpy as np

    from brawl_vision.terrain.labeling import CLASSES
    bounds = occupancy.observed_bounds()
    if bounds is None:
        return None
    r0, r1, c0, c1 = bounds
    best = occupancy.best()[r0:r1, c0:c1]
    out = np.full(best.shape, UNKNOWN, np.int16)
    for i, tile in enumerate(CLASSES):
        out[best == i] = int(tile)
    return out


def _window(source, start, stop, step=1):
    """Clip the source to [start, stop] by frame index. One implementation for both live capture
    and recordings, because `sources.open_source` makes them the same shape -- live capture just
    never reaches `stop` unless one is given."""
    for frame in source:
        if frame.index < start:
            continue
        if stop is not None and frame.index > stop:
            break
        if (frame.index - start) % step:
            continue
        yield frame


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="?",
                   help="a recorded .mp4, or 'screen'/'screen:<monitor>' for live capture")
    p.add_argument("--live", action="store_true", help="shorthand for source='screen'")
    p.add_argument("--start", type=int, default=0, help="first frame index")
    p.add_argument("--stop", type=int, default=None, help="last frame index, inclusive")
    p.add_argument("--step", type=int, default=1,
                   help="render every Nth frame. Odometry still sees only the rendered frames, so "
                        "a large step means larger inter-frame motion for it to correlate")
    p.add_argument("--fps", type=int, default=20, help="playback rate")
    p.add_argument("--save", default=None, help="render headless to this .mp4 or .gif")
    p.add_argument("--config", default=None, help="a vision.yaml to use instead of the default")
    p.add_argument("--terrain", default=None,
                   help="a trained classifier checkpoint; without it the occupancy panel stays "
                        "empty, since there is nothing to accumulate")
    args = p.parse_args(argv)

    # `open_source(None)` would happily grab the screen, but bare `vision_watch.py` typed by
    # someone expecting usage text should not silently start capturing their desktop.
    if args.live == bool(args.source):
        p.error("give either a source ('clip.mp4', 'screen', 'screen:2') or --live, "
                "not both and not neither")
    if args.save:
        matplotlib.use("Agg")     # must precede pyplot's first figure; no window is wanted here

    cfg = load_vision_config(args.config) if args.config else load_vision_config()
    hud = load_hud_mask()
    model = load_camera_model()
    plan = build_rectify_plan(model, hud)

    classifier = None
    if args.terrain:
        from brawl_vision.terrain.classifier import TerrainClassifier
        classifier = TerrainClassifier.load(args.terrain)

    overlay = TerrainOverlay(plan, hud, model)
    occupancy = OccupancyMap.from_config(cfg)
    with open_source(None if args.live else args.source, cfg) as source:
        stages = _stages(_window(source, args.start, args.stop, args.step), plan, Odometry(plan, cfg), cfg,
                         occupancy, classifier)
        if args.save:
            overlay.save(args.save, stages, fps=args.fps)
            print(f"wrote {args.save}")
        else:
            overlay.play(stages, fps=args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
