"""Cut recordings into still frames for annotation in CVAT, keeping only the actual match.

    python scripts/vision_export_frames.py                      # every recording, 2 fps
    python scripts/vision_export_frames.py --dry-run            # report the cuts, write nothing
    python scripts/vision_export_frames.py clip.mp4 --fps 4
    python scripts/vision_export_frames.py --review cuts.png    # contact sheet of every boundary

**The trimming is the point, not the sampling.** Every one of these recordings brackets the match
with something that is not gameplay -- a loading screen, the lobby, the "BRAWL" intro swoop, the
"Defeated" overlay, the results screen -- and a projectile detector trained on those learns that
menu art is a scene it might be asked about. `brawl_vision.gameplay` finds the span; this script
decodes it. See that module for why the attack button is the signal.

**Frames are written NORMALIZED (2002x1126), not as captured (2436x1126).** This is a train/deploy
matching decision and it is the one thing here worth getting right: `capture.normalize_viewport` is
what every frame passes through before any model in this repo sees it, so annotating raw captures
would train boxes in a coordinate space that never occurs at inference, and would spend annotation
effort on the pillarbox bars. `--raw` overrides it for the case where the frames are wanted for
something other than this pipeline.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brawl_vision.config import DATA_DIR
from brawl_vision.gameplay import (GAMEPLAY_THRESHOLD, load_gameplay_span, span_frames,
                                    walk_frames)

VIDEO_DIR = DATA_DIR / "training_videos"
DEFAULT_OUT = VIDEO_DIR / "frames"


def slug(stem: str) -> str:
    """A filename safe to hand to CVAT and to a shell. The recordings arrive named
    `ScreenRecording_08-17-2026 15-11-17_1`, and the space in the middle of that is a nuisance in
    every tool downstream."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)


def export(path: Path, out_dir: Path, scan: dict, margin: int, quality: int,
           raw: bool, flat: bool) -> int:
    """Decode the span's frames and write them as JPEGs. Returns how many were written."""
    keep = span_frames(scan, margin=margin)
    if not keep:
        return 0
    box = tuple(scan["content_box"])
    stem = slug(path.stem)
    dest = out_dir if flat else out_dir / stem
    dest.mkdir(parents=True, exist_ok=True)

    # `walk_frames` is shared with the scan, which is what guarantees that the frame written under
    # index i is the frame that was scored as gameplay at index i. See its docstring.
    written = 0
    for index, image in walk_frames(path, box, keep, normalize=not raw):
        cv2.imwrite(str(dest / f"{stem}_f{index:06d}.jpg"), image,
                    [cv2.IMWRITE_JPEG_QUALITY, quality])
        written += 1
    return written


def review_sheet(paths, scans, out_path: Path, margin: int, pad: int = 2) -> None:
    """One image showing the frames either side of every cut -- the check that the trim is right.

    A boundary is only correct if the frame just outside it is NOT gameplay and the frame just
    inside it IS, so both are drawn, labelled and colour-coded. This is the acceptance test for
    the whole script and it takes one look.

    Frames come from `walk_frames` rather than a seek per frame, for the same reason the export
    does: `CAP_PROP_POS_FRAMES` does not land where it is asked on these files. Measured on
    day12_recording1 f1682, a seek differs from the sequentially decoded frame by a mean of 17.9
    grey levels while the exported JPEG differs by 1.6 -- so a seeking review sheet would be
    showing frames adjacent to the ones under review, which for a tool whose entire job is to
    display one specific frame is the wrong kind of approximately right.
    """
    rows = []
    tw, th = 300, 139
    for path, scan in zip(paths, scans):
        keep = span_frames(scan, margin=margin)
        if not keep:
            continue
        box = tuple(scan["content_box"])
        step = scan["step"]
        first, last = keep[0], keep[-1]
        marks = ([(first - k * step, False) for k in range(pad, 0, -1)]
                 + [(first, True), (last, True)]
                 + [(last + k * step, False) for k in range(1, pad + 1)])
        marks = [(fi, inside) for fi, inside in marks if 0 <= fi < scan["n_frames"]]
        images = dict(walk_frames(path, box, [fi for fi, _ in marks]))
        cells = []
        for fi, inside in marks:
            cell = np.zeros((th + 16, tw, 3), np.uint8)
            if fi in images:
                cell[:th] = cv2.resize(images[fi], (tw, th))
                colour = (80, 230, 80) if inside else (60, 60, 235)
                cv2.rectangle(cell, (0, 0), (tw - 1, th - 1), colour, 3)
                cv2.putText(cell, f"f{fi} {'KEEP' if inside else 'cut'}", (4, th + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1)
            cells.append(cell)
        strip = np.hstack(cells)
        label = np.zeros((18, strip.shape[1], 3), np.uint8)
        cv2.putText(label, f"{path.name}   span f{first}-f{last}  ({len(keep)} frames)",
                    (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        rows.append(np.vstack([label, strip]))
    if not rows:
        return
    width = max(r.shape[1] for r in rows)
    canvas = np.vstack([
        np.hstack([r, np.zeros((r.shape[0], width - r.shape[1], 3), np.uint8)]) for r in rows
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clips", nargs="*", type=Path,
                   help=f"recordings to cut (default: every .mp4 in {VIDEO_DIR})")
    p.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT,
                   help=f"output directory (default: {DEFAULT_OUT})")
    p.add_argument("--fps", type=float, default=2.0,
                   help="frames per second to sample, and the grid the span is scored on "
                        "(default: 2)")
    p.add_argument("--threshold", type=float, default=GAMEPLAY_THRESHOLD,
                   help=f"button ring score above which a frame is gameplay "
                        f"(default: {GAMEPLAY_THRESHOLD})")
    p.add_argument("--margin", type=int, default=0,
                   help="extra SAMPLES to trim from each end of the span, for when the detector's "
                        "boundary is right but you want to be sure (default: 0)")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality (default: 95)")
    p.add_argument("--raw", action="store_true",
                   help="write frames as captured instead of normalized -- see the module "
                        "docstring before using this")
    p.add_argument("--flat", action="store_true",
                   help="write every frame into --out directly, instead of one subdirectory per "
                        "recording. Filenames carry the recording either way")
    p.add_argument("--rescan", action="store_true", help="ignore cached .gameplay.json sidecars")
    p.add_argument("--dry-run", action="store_true", help="scan and report, write no frames")
    p.add_argument("--review", type=Path, default=None,
                   help="write a contact sheet of the frames either side of every cut, and exit "
                        "without writing frames")
    args = p.parse_args(argv)

    clips = args.clips or sorted(VIDEO_DIR.glob("*.mp4"))
    if not clips:
        print(f"no recordings found in {VIDEO_DIR}", file=sys.stderr)
        return 1

    scans, total, t0 = [], 0, time.perf_counter()
    print(f"{'recording':46s} {'span (frames)':>17s} {'kept':>6s} {'cut':>13s}")
    print("-" * 90)
    for path in clips:
        scan = load_gameplay_span(path, scan_fps=args.fps, threshold=args.threshold,
                                  refresh=args.rescan)
        scans.append(scan)
        keep = span_frames(scan, margin=args.margin)
        fps = scan["source_fps"]
        if not keep:
            print(f"{path.name[:46]:46s} {'NO GAMEPLAY FOUND':>17s} {0:>6d}")
            continue
        head = keep[0] / fps
        tail = (scan["n_frames"] - 1 - keep[-1]) / fps
        n_written = 0 if (args.dry_run or args.review) else export(
            path, args.out, scan, args.margin, args.quality, args.raw, args.flat)
        total += len(keep)
        print(f"{path.name[:46]:46s} {f'{keep[0]}-{keep[-1]}':>17s} {len(keep):>6d} "
              f"{f'{head:.1f}s / {tail:.1f}s':>13s}"
              + ("" if args.dry_run or args.review else f"  -> {n_written} written"))

    print("-" * 90)
    print(f"{len(clips)} recordings, {total} frames at {args.fps} fps "
          f"({time.perf_counter() - t0:.0f}s)")
    if args.review:
        review_sheet(clips, scans, args.review, args.margin)
        print(f"review sheet -> {args.review}")
    elif not args.dry_run:
        print(f"frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
