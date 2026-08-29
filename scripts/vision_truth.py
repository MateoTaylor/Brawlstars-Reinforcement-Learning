"""Make (and then score) the hand-verified `<clip>.grid.csv`. See
Terrain_Perception_Build_Plan.md Phases I and L.

    # 1. generate a reference picture and a blank grid to fill in
    python scripts/vision_truth.py tests/fixtures/vision/showdown_alternate_map.mp4

    # 2. ...fill in the CSV against the PNG, then see where you are
    python scripts/vision_truth.py tests/fixtures/vision/showdown_alternate_map.mp4 --progress

    # 3. score the perception stack against it
    python scripts/vision_truth.py tests/fixtures/vision/showdown_alternate_map.mp4 --check

Step 1 writes three files next to the clip:

    <clip>.grid.png    the map as a photograph, with the tile grid and coordinates drawn on
    <clip>.grid.csv    a blank grid of the same size, every cell `?`
    <clip>.grid.json   where that region sits, so scoring does not have to guess

**Every square in the PNG is one character in the CSV.** The numbers down the left edge are the
CSV's line numbers and the numbers along the top are its field positions, both counting from 0.

Legend: `.` floor  `#` wall  `b` bush  `~` water  `f` fence  `?` don't know.

**Two spellings of a row both work, and the compact one is much easier to type.** The file is
written comma-separated (`.,.,#,#`), but in that form column N sits at character 2N, so the text
does not line up with the picture and counting columns is where a long template goes wrong. Delete
the commas and it is one character per tile (`..##`), which lines up in any monospace editor
exactly as the reference image does. Either loads.

**`?` is free.** Unknown cells are skipped on both sides of the comparison, so anything ambiguous
-- under a loot box, under a brawler, too streaky to call -- stays `?` and costs you nothing. Do
not guess to fill space; a guessed cell is worse than a blank one, because it looks like evidence.
"""
import argparse
import sys
from pathlib import Path

from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
from brawl_vision.config import load_vision_config
from brawl_vision.sources import open_source
from brawl_vision.terrain import truth as T
from brawl_vision.terrain.evaluate import Mosaic, scan_track, world_window
from brawl_vision.terrain.occupancy import OccupancyMap

DEFAULT_TERRAIN = Path(__file__).resolve().parent.parent / "brawl_vision" / "data" / "terrain.pt"


def _paths(clip: Path):
    stem = clip.with_suffix("")
    return (Path(f"{stem}.grid.png"), Path(f"{stem}.grid.csv"), Path(f"{stem}.grid.json"))


def _build_mosaic(clip, cfg, plan, step, scan_step, stop, say):
    say(f"pass 1/2  odometry over {clip.name} (every {scan_step} frames)")
    with open_source(clip, cfg, step=scan_step) as src:
        track = scan_track((f for f in src if stop is None or f.index <= stop), plan, cfg)
    lo_x, hi_x, lo_y, hi_y = track.bounds_tiles()
    say(f"   camera track {hi_x - lo_x:.1f} x {hi_y - lo_y:.1f} tiles, "
        f"{track.n_segments} segment(s)")
    if track.n_segments > 1:
        say("   WARNING: odometry cut. The mosaic will contain two unrelated worlds overlaid; "
            "use --stop to take a stretch before the cut.")
    occ = OccupancyMap.from_config(cfg)
    window = world_window(track, plan, occ.height, occ.width)
    mosaic = Mosaic(window, plan, occ.origin, scale=T._REF_SCALE)
    plate = T.Plate(mosaic)
    say("pass 2/2  building the median plate (transients are outvoted, terrain is not)")
    with open_source(clip, cfg, step=step) as src:
        from brawl_vision.terrain.odometry import Odometry
        odo = Odometry(plan, cfg)
        for frame in src:
            if stop is not None and frame.index > stop:
                break
            rect = plan.rectify(frame.image)
            r = odo.update(rect)
            if r.tracking:
                plate.add(rect, r.position_tiles)
    return track, occ, plate


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", help="a gameplay .mp4")
    p.add_argument("--check", action="store_true",
                   help="score the perception stack against the filled-in CSV")
    p.add_argument("--progress", action="store_true", help="report how much of the CSV is filled")
    p.add_argument("--refresh-image", action="store_true",
                   help="rebuild the reference PNG for the region the sidecar already records, "
                        "leaving the CSV and the region alone. Use this after a change to how the "
                        "plate is built, when a template is already partly filled in")
    p.add_argument("--max-cols", type=int, default=24, help="widest template to generate")
    p.add_argument("--max-rows", type=int, default=18, help="tallest template to generate")
    p.add_argument("--min-coverage", type=float, default=T._MIN_COVERAGE,
                   help="fraction of a tile's pixels the mosaic must have painted for it to be "
                        "included -- lower it to get a bigger but streakier region")
    p.add_argument("--step", type=int, default=2, help="read every Nth frame when building")
    p.add_argument("--scan-step", type=int, default=4, help="...and when sizing the canvas")
    p.add_argument("--stop", type=int, default=None, help="last frame index, inclusive")
    p.add_argument("--terrain", default=str(DEFAULT_TERRAIN), help="classifier for --check")
    p.add_argument("--device", default="cpu")
    p.add_argument("--config", default=None, help="a vision.yaml to use instead of the default")
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate a blank CSV over one that already exists. This throws away "
                        "hand-written work; the PNG and JSON are rewritten either way")
    args = p.parse_args(argv)

    clip = Path(args.clip)
    if not clip.exists():
        print(f"no such clip: {clip}", file=sys.stderr)
        return 2
    png, csv_path, sidecar = _paths(clip)
    say = lambda *a: print(*a, flush=True)          # noqa: E731

    # -- progress: needs nothing but the CSV ---------------------------------
    if args.progress:
        if not csv_path.exists():
            print(f"no {csv_path.name} yet -- run without --progress to generate one",
                  file=sys.stderr)
            return 2
        grid = T.load_grid_csv(csv_path)
        pr = T.progress(grid)
        say(f"{csv_path.name}: {pr['filled']}/{pr['cells']} cells filled "
            f"({pr['fraction'] * 100:.0f}%)")
        if pr["counts"]:
            say("   " + "  ".join(f"{k}:{v}" for k, v in pr["counts"].items()))
        if pr["filled"] < pr["cells"]:
            say(f"   {pr['cells'] - pr['filled']} still `?`. That is fine -- `?` cells are simply "
                f"skipped when scoring. Fill in what you are sure of.")
        return 0

    cfg = load_vision_config(args.config) if args.config else load_vision_config()
    plan = build_rectify_plan(load_camera_model(), load_hud_mask())

    # -- check: score the pipeline against a filled-in CSV --------------------
    if args.check:
        if not csv_path.exists() or not sidecar.exists():
            print(f"need both {csv_path.name} and {sidecar.name}; generate them first",
                  file=sys.stderr)
            return 2
        import json
        meta = json.loads(sidecar.read_text())
        grid = T.load_grid_csv(csv_path)
        pr = T.progress(grid)
        if pr["filled"] == 0:
            print(f"{csv_path.name} is still entirely `?` -- nothing to score against.",
                  file=sys.stderr)
            return 2
        say(f"scoring against {pr['filled']}/{pr['cells']} filled cells")

        ckpt = Path(args.terrain)
        if not ckpt.exists():
            print(f"no classifier at {ckpt}", file=sys.stderr)
            return 2
        from brawl_vision.terrain.classifier import TerrainClassifier
        from brawl_vision.terrain.evaluate import Window
        from brawl_vision.terrain.odometry import Odometry
        from brawl_vision.terrain.zone import detect_zone

        classifier = TerrainClassifier.load(ckpt, device=args.device)
        occ = OccupancyMap.from_config(cfg)
        odo = Odometry(plan, cfg)
        say("running the pipeline over the clip")
        with open_source(clip, cfg, step=args.step) as src:
            for frame in src:
                if args.stop is not None and frame.index > args.stop:
                    break
                rect = plan.rectify(frame.image)
                r = odo.update(rect)
                if r.segment != occ.segment:
                    occ.reset(r.segment)
                zone = detect_zone(rect, plan, cfg).at_least(0.05)
                cells, _ = classifier.predict(rect, plan)
                occ.update(cells, r, plan, zone=zone, cfg=cfg)

        r0, r1, c0, c1 = meta["grid_window"]
        got = T.score(occ.to_chars(), grid, Window(row0=r0, row1=r1, col0=c0, col1=c1))
        say("")
        say(f"  cells compared            {got['cells_compared']}")
        say(f"  accuracy at zero offset   {got['accuracy_at_zero_offset']:.3f}")
        say(f"  accuracy at best offset   {got['accuracy_at_best_offset']:.3f}")
        say(f"  best offset (rows, cols)  {got['best_offset']}  "
            f"= {got['position_error_tiles']:.2f} tiles")
        if got["wall_vs_fence"] is not None:
            say(f"  WALL vs FENCE             {got['wall_vs_fence']:.3f}")
        say("")
        say("  A large offset with high accuracy is odometry drift (Phase F).")
        say("  A zero offset with low accuracy is the classifier (Phase H).")
        return 0

    # -- refresh the picture only, for the region the CSV already describes ----
    if args.refresh_image:
        if not sidecar.exists():
            print(f"no {sidecar.name} to take the region from", file=sys.stderr)
            return 2
        import json
        meta = json.loads(sidecar.read_text())
        track, occ, plate = _build_mosaic(clip, cfg, plan, args.step, args.scan_step, args.stop, say)
        # The region comes from the sidecar, NOT from re-running the chooser. A filled-in CSV is
        # addressed by row and column; re-picking the region could shift it by a tile and silently
        # invalidate every cell of somebody's afternoon.
        r0, r1, c0, c1 = meta["grid_window"]
        m = plate.mosaic.window
        local = T.Window(row0=r0 - m.row0, row1=r1 - m.row0, col0=c0 - m.col0, col1=c1 - m.col0)
        if local.row0 < 0 or local.col0 < 0 or local.row1 > m.rows or local.col1 > m.cols:
            print("the recorded region is not inside this run's canvas -- did --stop change?",
                  file=sys.stderr)
            return 2
        import cv2
        cv2.imwrite(str(png), T.reference_image(plate, local))
        say(f"\nrewrote {png.name} for the SAME {local.cols} x {local.rows} region "
            f"({csv_path.name} untouched)")
        return 0

    # -- generate ------------------------------------------------------------
    track, occ, plate = _build_mosaic(clip, cfg, plan, args.step, args.scan_step, args.stop, say)
    template, local = T.build_template(clip.stem, plate, occ.origin, args.max_cols, args.max_rows,
                                       args.min_coverage)
    import cv2
    cv2.imwrite(str(png), T.reference_image(plate, local))
    T.write_sidecar(template, sidecar)
    existed = csv_path.exists()
    try:
        T.write_template(template, csv_path, overwrite=args.overwrite)
    except FileExistsError:
        say(f"\nkept {csv_path.name} (it already has work in it); "
            f"rewrote {png.name} and {sidecar.name}")
    else:
        say(f"\nwrote {csv_path.name}" + (" (overwritten)" if existed else ""))
    say(f"wrote {png.name}  {template.cols} x {template.rows} tiles, "
        f"{template.coverage * 100:.0f}% mosaic coverage")
    say(f"wrote {sidecar.name}  world origin {template.origin_tile}")
    say("")
    say(f"Open {png.name} next to {csv_path.name}. Every square in the picture is one character")
    say("in the file; the numbers on the edges are the CSV's own row and column positions.")
    say("Legend: . floor   # wall   b bush   ~ water   f fence   ? don't know")
    say("`?` costs nothing -- leave anything you are unsure about, and do not guess to fill space.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
