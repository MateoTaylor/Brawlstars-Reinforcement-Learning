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

`--detect` adds the third-party entity detector (`brawl_vision.object_detection`) to that layout:
its boxes go on the raw frame, the reconstructed map stays exactly as it was. The two run in the
same loop but not through each other -- no detection touches odometry, the classifier, or the
occupancy grid. This is the "look at both before wiring them together" tool.

`--detect-on-map` goes one step further and puts each detection ON the map, by projecting its box
bottom through the same homography the terrain pipeline uses. With `--layout view` you get the
agent's own window twice -- rectified frame left, predicted map right, same tiles, same scale,
markers on both -- which is the layout where you can actually check whether a detected brawler
lands on the cell the map says it should.

    python scripts/vision_evaluate.py match.mp4 -o both.mp4 --detect
    python scripts/vision_evaluate.py match.mp4 -o view.mp4 --detect-on-map --layout view

`--hp` reads each brawler's HP out of the readout inside its own box and writes it, with a
confidence, beside that box on the raw panel. It implies `--detect` (it only ever looks at pixels
the detector pointed it at) and forces `--layout side-by-side` (it annotates the RAW frame).
Read the number WITH its confidence -- a covered readout is the one failure that looks like a
clean read; brawl_vision/object_detection/hp_detection/ documents why.

`--projectiles` adds OUR trained projectile model (`object_detection/projectile_detection`) to
the same layout, beside whatever else is on. It finds power cubes as well, one colour per class:
magenta projectiles, blue crates, green dropped cubes. It is independent of `--detect`: the two
are different models over the same raw frame, and either can run alone. With both, the corner
tally counts both, and it doubles as the colour legend.

    python scripts/vision_evaluate.py match.mp4 -o all.mp4 --hp --projectiles
    python scripts/vision_evaluate.py match.mp4 -o shots.mp4 --projectiles --projectile-conf 0.15

`--projectiles-on-map` projects them onto the map too, and it is the shakiest thing here: the
homography inverts perspective for the GROUND plane and a projectile is in the air, so every
marker lands further from the camera than the shot really is. Off by default, and for looking
rather than for reading a tile off. See `object_detection/project.PROJECTILE_ANCHOR_FRAC`.

`--cv-fps` separates how often the CV runs from how often a frame is written. The output rate is
a playback question; the CV rate is the pipeline's real budget (4-8 Hz against the 250 ms decision
tick). Running a detector 24 times a second to make a video is three times the work for no extra
information, so pair a high `--fps` with a low `--cv-fps` and the results are held in between:

    # 24 fps for playback, perception at 6 Hz
    python scripts/vision_evaluate.py match.mp4 -o yt.mp4 --hp --detect-on-map         --map-extent view --fps 24 --cv-fps 6 --step 1

`--map-extent` decides how much accumulated map the right panel shows, and only applies to
side-by-side. `full` is the whole explored map with this frame's footprint outlined -- good for
watching coverage grow, useless for comparing one frame against the terrain under it, because the
panels cover different world. `view` crops to exactly the tiles this frame covers, so the panels
are the same extent tile-for-tile. `policy` crops further to the 21x13 the agent is handed:

    # boxes + HP on the video, beside just the player's field of view
    python scripts/vision_evaluate.py match.mp4 -o fov.mp4 --hp --detect-on-map         --map-extent view

The projection has a MEASURED bias -- markers land ~2 tiles further from the camera than the
sprite they mark -- and it is left uncorrected by default. `--detect-ground-offset` is the knob.
See brawl_vision/object_detection/project.py for the numbers and why no constant is baked in.

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
    p.add_argument("--cv-fps", type=float, default=None,
                   help="how often the classifier, detector and HP reader actually RUN, held and "
                        "redrawn in between (default: --fps, i.e. every written frame). This is "
                        "the pipeline's real budget -- 4-8 Hz against the 250 ms decision tick -- "
                        "so pair a high --fps for smooth playback with a low --cv-fps for honest "
                        "perception. Odometry is exempt and still runs every source frame")
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
    p.add_argument("--detect", action="store_true",
                   help="also run the third-party entity detector over the RAW frame and draw its "
                        "boxes on the left panel. Implies --layout side-by-side, which is the only "
                        "layout showing a raw frame. Needs the weights: run "
                        "scripts/vision_fetch_detector.py")
    p.add_argument("--detect-on-map", action="store_true",
                   help="ALSO project each detection's ground point through the homography and "
                        "mark it on the predicted map. Implies --detect. Read "
                        "brawl_vision/object_detection/project.py first: the projection carries a "
                        "measured ~2-tile bias away from the camera, uncorrected by default")
    p.add_argument("--detect-anchor", type=float, default=None,
                   help="where in a box the feet are, as a fraction of box height up from the "
                        "BOTTOM edge (default: config's 0.30, measured). 0 is the raw box bottom, "
                        "which places a brawler barely better than chance -- these boxes enclose "
                        "the nameplate and any aura, not just the sprite")
    p.add_argument("--detect-ground-offset", type=float, default=0.0,
                   help="tiles to pull each projected marker back toward the camera, "
                        "compensating for that bias. Default 0 (uncorrected). ~1.2 lines the "
                        "markers up on the four fixtures measured, but is a constant fitted to "
                        "one hero at one zoom -- it is a knob to explore, not a calibration")
    p.add_argument("--detect-conf", type=float, default=None,
                   help="override detector.conf from the config (default: 0.5, measured)")
    p.add_argument("--detect-ignore", default=None,
                   help="comma-separated classes to drop, overriding detector.ignore. Pass an "
                        "empty string to keep all of them -- 'teammate' is impossible in Solo "
                        "Showdown but is the canary for a mis-fed detector")
    p.add_argument("--projectiles", action="store_true",
                   help="also run OUR trained projectile/power-cube detector over the RAW frame "
                        "and draw its boxes on the left panel (magenta projectile, blue crate, "
                        "green dropped cube). Independent of --detect -- either "
                        "model can run alone. Needs a trained model: run "
                        "brawl_vision/object_detection/projectile_detection/train.py, which "
                        "exports the ONNX this loads")
    p.add_argument("--projectiles-on-map", action="store_true",
                   help="ALSO mark each projectile on the predicted map. Implies --projectiles. "
                        "READ THIS FIRST: the homography inverts perspective for the ground "
                        "plane, and a projectile is above it, so every marker lands further from "
                        "the camera than the shot actually is by an unmeasured amount. Useful for "
                        "'roughly over there', not for reading a tile index")
    p.add_argument("--projectile-conf", type=float, default=None,
                   help="override projectile.conf from the config (default: 0.30, a starting "
                        "point rather than a measurement). This model is NMS-free, so its 300 "
                        "output rows have no objectness gate ahead of them and this threshold is "
                        "the only thing thinning them -- it bites harder here than --detect-conf "
                        "does on the entity model")
    p.add_argument("--projectile-model", default=None,
                   help="path to a specific .onnx, bypassing projectile.model. Point it straight "
                        "at a run's weights/best.onnx to compare two training runs without "
                        "promoting either")
    p.add_argument("--hp", action="store_true",
                   help="also read each brawler's HP out of its own box and write the number and "
                        "a confidence beside it on the raw panel. Implies --detect (it only reads "
                        "inside detection boxes) and needs --layout side-by-side (it annotates the "
                        "RAW frame). Read the number WITH its confidence: see "
                        "brawl_vision/object_detection/hp_detection/")
    p.add_argument("--map-extent", choices=("full", "view", "policy"), default="full",
                   help="how much accumulated map the right panel of --layout side-by-side shows. "
                        "'full' (default) is the whole explored map with this frame's footprint "
                        "outlined; 'view' crops to just the tiles this frame covers, so both "
                        "panels are the same extent tile-for-tile; 'policy' crops further to the "
                        "21x13 window the agent is actually handed. Cropping needs a classifier")
    p.add_argument("--config", default=None, help="a vision.yaml to use instead of the default")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    clip = Path(args.clip)
    if not clip.exists():
        print(f"no such clip: {clip}", file=sys.stderr)
        return 2
    cfg = load_vision_config(args.config) if args.config else load_vision_config()
    plan = build_rectify_plan(load_camera_model(), load_hud_mask())

    detector = None
    projectiles = None
    if args.detect_on_map or args.hp:
        args.detect = True
    if args.projectiles_on_map:
        args.projectiles = True
    if args.hp and args.layout in ("map", "view"):
        # Same courtesy --detect gets, and said out loud for the same reason: HP is read inside a
        # box and written next to it, so it needs the one layout with a raw frame. 'view' has
        # none, so this is a genuine override rather than filling in a default.
        print(f"--hp annotates the raw frame, so --layout side-by-side (not {args.layout!r})",
              file=sys.stderr)
        args.layout = "side-by-side"
    if (args.detect or args.projectiles) and args.layout == "map":
        # Chosen for the user rather than refused, because `map` is the default and every
        # --detect run would otherwise be a two-command dance. Said out loud, because a layout
        # silently different from the one you asked for is the other way to be annoying.
        # `view` is left alone: it has no raw frame, but with the -on-map flags it draws markers
        # on both of its panels, which is the layout worth comparing in.
        #
        # 'view' only if EVERY model that is running is going on the map. With one of them drawing
        # boxes on the raw frame, 'view' would have to silently drop that model, and dropping the
        # thing you asked to see is worse than picking the other layout.
        on_map = ((args.detect_on_map or not args.detect)
                  and (args.projectiles_on_map or not args.projectiles))
        chosen = "view" if on_map else "side-by-side"
        asked = " / ".join(f for f, on in (("--detect", args.detect),
                                           ("--projectiles", args.projectiles)) if on)
        print(f"{asked} needs somewhere to draw, so --layout {chosen} (not 'map')",
              file=sys.stderr)
        args.layout = chosen
    if args.layout == "view":
        # Checked per model rather than once: 'view' has no raw frame, so each enabled detector
        # independently has nowhere to put boxes and markers are all it can contribute.
        if args.detect and not args.detect_on_map:
            print("--layout view has no raw frame to draw boxes on; adding --detect-on-map",
                  file=sys.stderr)
            args.detect_on_map = True
        if args.projectiles and not args.projectiles_on_map:
            print("--layout view has no raw frame to draw boxes on; adding --projectiles-on-map",
                  file=sys.stderr)
            args.projectiles_on_map = True
    if args.detect:
        from brawl_vision.object_detection import ObjectDetector
        overrides = {}
        if args.detect_conf is not None:
            overrides["conf"] = args.detect_conf
        if args.detect_ignore is not None:
            overrides["ignore"] = tuple(c for c in args.detect_ignore.split(",") if c)
        try:
            detector = ObjectDetector.from_config(cfg, **overrides)
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            print(exc, file=sys.stderr)
            return 2

    if args.projectiles:
        from brawl_vision.object_detection.projectile_detection import ProjectileDetector
        overrides = {}
        if args.projectile_conf is not None:
            overrides["conf"] = args.projectile_conf
        if args.projectile_model is not None:
            overrides["path"] = args.projectile_model
        try:
            projectiles = ProjectileDetector.from_config(cfg, **overrides)
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            # FileNotFoundError here is the common one and it is not a failure of this script:
            # weights.require() raises it with the list of training runs that could satisfy it.
            print(exc, file=sys.stderr)
            return 2

    health = None
    if args.hp:
        from brawl_vision.object_detection.hp_detection import HealthTracker
        try:
            # A FRESH tracker for this render. It is stateful, and its thresholds count RENDERED
            # frames -- so --fps is what hp.confirm_frames is denominated in here, not the 4 Hz
            # decision rate the defaults were chosen for.
            health = HealthTracker.from_config(cfg)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 2

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

    if detector is not None:
        say(f"   detector {detector.path.name} on {detector.provider}, classes "
            f"{detector.classes}, conf {detector.conf}"
            + (f", ignoring {sorted(detector.ignore)}" if detector.ignore else ""))
    if projectiles is not None:
        say(f"   projectiles {projectiles.path.name} on {projectiles.provider}, classes "
            f"{projectiles.classes}, conf {projectiles.conf}"
            + ("  (NMS-free: conf is the only thing thinning 300 rows/frame)"
               if projectiles.end2end else ""))
    cv_fps = args.fps if args.cv_fps is None else args.cv_fps
    if health is not None:
        say(f"   HP from {health.reader.bank.meta.get('n_glyphs', '?')} calibrated glyphs; "
            f"confirm_frames {health.confirm_frames} = "
            f"{health.confirm_frames / max(cv_fps, 1e-6) * 1000:.0f} ms at the {cv_fps:g} Hz CV "
            f"rate")
    if args.map_extent != "full":
        say(f"   right panel cropped to the {args.map_extent} extent"
            + (" (21x13, what the agent receives)" if args.map_extent == "policy"
               else " (this frame's footprint)"))
    if args.detect_on_map:
        frac = cfg.detector_anchor_frac if args.detect_anchor is None else args.detect_anchor
        say(f"   projecting onto the map from {frac:.0%} up each box"
            + (f", offset {args.detect_ground_offset:+.2f} tiles"
               if args.detect_ground_offset else "")
            + (".  NOTE: 0% is the raw box bottom, which is barely better than chance -- "
               "see Detection.anchor" if frac == 0 else ""))
    if args.projectiles_on_map:
        say("   projecting projectiles onto the map from each box CENTRE.")
        say("     NOTE: the homography inverts perspective for the GROUND plane, and a "
            "projectile is above it,")
        say("     so every marker lands further from the camera than the shot really is. "
            "Look at it; do not read a tile off it.")
    say(f"pass 2/2  rendering {args.mode} at {args.fps:g} fps"
        + (f", CV at {cv_fps:g} Hz (held in between)" if cv_fps != args.fps else ""))
    with open_source(clip, cfg, step=args.step) as src:
        report = render(_window(src, args.start, args.stop), plan, track, args.out,
                        classifier=classifier, detector=detector, health=health,
                        projectiles=projectiles,
                        projectiles_on_map=args.projectiles_on_map,
                        map_extent=args.map_extent,
                        detect_on_map=args.detect_on_map,
                        ground_offset_tiles=args.detect_ground_offset,
                        anchor_frac=args.detect_anchor,
                        cfg=cfg, layout=args.layout, out_fps=args.fps, cv_fps=args.cv_fps,
                        scale=args.scale,
                        progress=None if args.quiet else
                        (lambda n, t: print(f"   {n:5d} frames  t={t:7.2f}s", flush=True)))

    say(f"wrote {args.out}  {report.size[0]}x{report.size[1]}  {report.out_frames} frames")
    speed = "faster" if report.faster_than_realtime else "SLOWER"
    say(f"   {report.render_seconds:.1f}s to render {report.clip_seconds:.1f}s of clip "
        f"({report.realtime_factor:.2f}x, {speed} than real time), "
        f"{report.classified} classifications")
    if report.detected_frames:
        say(f"   {report.detections} boxes over {report.detected_frames} detected frames "
            f"({report.detections / report.detected_frames:.2f}/frame)")
    if report.projectile_frames:
        # Reported separately from the entity boxes rather than summed. The two models find
        # different kinds of thing at different rates, and one boxes-per-frame number over both
        # would move for reasons you could not attribute.
        say(f"   {report.projectile_boxes} projectile-model boxes over "
            f"{report.projectile_frames} frames "
            f"({report.projectile_boxes / report.projectile_frames:.2f}/frame)")
        # And per class, for the same reason one level down: a crate sits on screen for the whole
        # clip and a shot for a few frames, so the combined rate is dominated by whichever of them
        # this clip happened to have more of.
        for label, n in sorted(report.projectile_labels.items()):
            say(f"     {label:<20s} {n:6d}  ({n / report.projectile_frames:.2f}/frame)")
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
