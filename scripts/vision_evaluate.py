"""Gameplay mp4 in, reconstructed-map mp4 out. See Terrain_Perception_Build_Plan.md Phase L.

    python scripts/vision_evaluate.py tests/fixtures/vision/showdown_alternate_map.mp4 -o map.mp4
    python scripts/vision_evaluate.py match.mp4 -o side.mp4 --layout side-by-side
    python scripts/vision_evaluate.py match.mp4 -o mosaic.mp4 --mode mosaic

`--mode map` (default) needs a trained classifier and renders the Phase I occupancy grid: the world
as the perception stack believes it to be. `--mode mosaic` needs only odometry and pastes the
rectified frames themselves into the world canvas -- not the deliverable, but the fastest way to
see whether odometry is holding, because drift shows up as ghosting and double walls.

`--layout view` is the one for checking the classifier: the rectified frame on the left, the map
cropped to exactly the same tiles on the right, both at the same scale, so a row in one panel is
that row in the other and you can read a disagreement off cell by cell. The yellow box is the
21x13 window the policy actually receives.

`--layout side-by-side` puts the raw frame next to the WHOLE accumulated map, with the current
frame's footprint outlined -- useful for watching coverage grow. `--layout map` is the artifact.

This takes a recording, not live capture: it reads the clip twice (once for odometry, once to
render), and a live source cannot be rewound.
"""
import argparse
import sys
from pathlib import Path

from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
from brawl_vision.config import load_vision_config
from brawl_vision.sources import open_source
from brawl_vision.terrain.classifier import TerrainClassifier
from brawl_vision.terrain.evaluate import render, scan_track

DEFAULT_TERRAIN = Path(__file__).resolve().parent.parent / "brawl_vision" / "data" / "terrain.pt"


def _window(source, start, stop):
    for frame in source:
        if frame.index < start:
            continue
        if stop is not None and frame.index > stop:
            break
        yield frame


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", help="a gameplay .mp4")
    p.add_argument("-o", "--out", required=True, help="output .mp4 (or .gif, for short clips)")
    p.add_argument("--mode", choices=("map", "mosaic"), default="map")
    p.add_argument("--layout", choices=("map", "side-by-side", "view"), default="map",
                   help="map: the reconstruction alone, the artifact. side-by-side: the raw frame "
                        "next to the whole map, with the current frame's tiles outlined. view: the "
                        "RECTIFIED frame next to the map cropped to exactly those tiles, both at "
                        "the same scale -- the only layout where a tile in one panel is the same "
                        "tile in the other")
    p.add_argument("--fps", type=float, default=8.0,
                   help="OUTPUT rate. Frames are sampled on the clip's own timestamps, so this is "
                        "the real rate rather than a nominal one (default: 8)")
    p.add_argument("--scale", type=int, default=None, help="output pixels per tile")
    p.add_argument("--terrain", default=str(DEFAULT_TERRAIN),
                   help="classifier checkpoint for --mode map")
    p.add_argument("--device", default="cpu")
    p.add_argument("--step", type=int, default=2,
                   help="pass 2 reads every Nth frame. Odometry is a chain, so a larger step means "
                        "larger inter-frame motion to correlate -- step 4 was measured within 0.4 "
                        "tiles of step 1 over a 22-tile walk (default: 2)")
    p.add_argument("--scan-step", type=int, default=4,
                   help="pass 1 reads every Nth frame. It only sizes the canvas, and skipped "
                        "frames cost ~4 ms instead of ~12. Measured on the 4802-frame fixture, "
                        "steps 1-6 all report the same bounding box to within 0.5 tiles and never "
                        "cut; step 8 cuts twice, which is why the default is not higher "
                        "(default: 4)")
    p.add_argument("--start", type=int, default=0, help="first frame index")
    p.add_argument("--stop", type=int, default=None, help="last frame index, inclusive")
    p.add_argument("--config", default=None, help="a vision.yaml to use instead of the default")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    clip = Path(args.clip)
    if not clip.exists():
        print(f"no such clip: {clip}", file=sys.stderr)
        return 2
    cfg = load_vision_config(args.config) if args.config else load_vision_config()
    plan = build_rectify_plan(load_camera_model(), load_hud_mask())

    classifier = None
    if args.mode == "map":
        ckpt = Path(args.terrain)
        if not ckpt.exists():
            print(f"no classifier at {ckpt}. Train one with scripts/vision_train_terrain.py, or "
                  f"run --mode mosaic, which needs only odometry.", file=sys.stderr)
            return 2
        classifier = TerrainClassifier.load(ckpt, device=args.device)

    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))

    say(f"pass 1/2  odometry over {clip.name} (every {args.scan_step} frames, canvas sizing only)")
    with open_source(clip, cfg, step=args.scan_step) as src:
        track = scan_track(_window(src, args.start, args.stop), plan, cfg,
                           progress=None if args.quiet else
                           (lambda n, t: print(f"   {n:5d} frames  t={t:7.2f}s", flush=True)))
    lo_x, hi_x, lo_y, hi_y = track.bounds_tiles()
    say(f"   {track.n_frames} frames, {track.seconds:.1f}s of clip, {track.scan_seconds:.1f}s to "
        f"scan")
    say(f"   camera track {hi_x - lo_x:.1f} x {hi_y - lo_y:.1f} tiles, "
        f"{track.n_segments} segment(s), {track.n_lost} lost frame(s)")
    if track.n_segments > 1:
        say("   WARNING: odometry cut during the SCAN. After a cut the position keeps accumulating "
            "with an unknown world offset, so the canvas above is sized from a track pass 1 cannot "
            "vouch for. Re-run with a smaller --scan-step.")

    say(f"pass 2/2  rendering {args.mode} at {args.fps:g} fps")
    with open_source(clip, cfg, step=args.step) as src:
        report = render(_window(src, args.start, args.stop), plan, track, args.out,
                        classifier=classifier, cfg=cfg, layout=args.layout, out_fps=args.fps,
                        scale=args.scale,
                        progress=None if args.quiet else
                        (lambda n, t: print(f"   {n:5d} frames  t={t:7.2f}s", flush=True)))

    say(f"wrote {args.out}  {report.size[0]}x{report.size[1]}  {report.out_frames} frames")
    speed = "faster" if report.faster_than_realtime else "SLOWER"
    say(f"   {report.render_seconds:.1f}s to render {report.clip_seconds:.1f}s of clip "
        f"({report.realtime_factor:.2f}x, {speed} than real time), "
        f"{report.classified} classifications")
    if report.outside_window:
        say(f"   WARNING: {report.outside_window} observed cells fell outside the canvas pass 1 "
            f"sized. Re-run with a smaller --scan-step.")
    if report.unknown_curve:
        first, last = report.unknown_curve[0][1], report.unknown_curve[-1][1]
        mono = "monotonic" if report.unknown_is_monotonic() else "NOT MONOTONIC (odometry lost lock)"
        say(f"   UNKNOWN cells {first} -> {last}, {mono}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
