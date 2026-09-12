"""Fine-tune YOLO26n to detect Brawl Stars projectiles and power cubes.

    python brawl_vision/object_detection/projectile_detection/train.py
    python brawl_vision/object_detection/projectile_detection/train.py --audit-only
    python brawl_vision/object_detection/projectile_detection/train.py --val 08-31,9-10_new3

Drop a CVAT "Ultralytics YOLO Detection" export (.zip) beside this file and run it; the newest
zip is used unless `--archive` names one. `prepare.py` checks the export, unpacks it to
`dataset/<zip stem>/`, holds out whole recordings for validation, and prints an audit before
anything trains. That module explains why each step exists. `--audit-only` stops there without
importing ultralytics.

The class names and their ids come from the export, not from this file. The three-class export
numbers them 0 Power Cube Box, 1 Power Cube Dropped, 2 Projectile, and the ONNX carries that
mapping in its metadata, which is where `ProjectileDetector` reads it.

Weights land in `projectile_detection/runs/<name><N>/weights/` as best.pt, last.pt and
best.onnx. Re-running never overwrites: Ultralytics increments the run name.

Notes on the values below, so they are not mistaken for defaults worth copying elsewhere:

  IMGSZ=960    Must be a multiple of 32 or Ultralytics rounds it up (900 -> 928, 1000 -> 1024).
               sqrt(area) in the source frame, min / p10 / median / p90 / max, three-class
               export: crate 67 / 115 / 143 / 159 / 192 px, dropped cube 46 / 57 / 65 / 72 / 153,
               projectile 22 / 47 / 88 / 239 / 903. Letterboxed to 960 the dropped cube, now the
               smallest class, runs 22 / 27 / 31 / 36 / 73 and the projectile floor is 11. 640
               would put the cube's median at 21 px and the projectile floor at 7, below the ~10 px
               where detections start dropping. 1280 was better for the smallest boxes but pinned
               the GPU. `--audit-only` prints these numbers for whatever export you have.
  batch=16     Measured 10.4 GB of the 5070 Ti's 16 GB at imgsz 1280; 960 is 56% of that pixel
               count, so expect roughly 6-7 GB. Raise the batch if you want the headroom back.
  workers=0    Windows has no fork, so every DataLoader worker is a spawned process holding its own
               copy of the dataset. Leave this at 0.
  scale=0.25   Showdown holds a fixed camera zoom, so an object's size in pixels is a real cue.
               The 0.5 default teaches the model to ignore the one measurement it can trust.
  degrees=0    The camera never rolls and the world has a fixed up (HUD, health bars, shadows), so
  flipud=0     rotated or flipped frames are images the game will never produce.
  hsv_h=0.015  Kept small on purpose. Every brawler's shots are colour-coded, and hue is much of
               what separates a green cube from a green bush, so it has to stay a cue, not noise.
  close_mosaic Mosaic manufactures composition variety out of ~1400 labelled images, but stitches
    =25       four frames with HUD through the middle. The last 25 epochs run on real frames.

The frames with no label file (660 of 2052 in the three-class export) are background images,
not gaps: they are an explicit "nothing here". Keep them. This model runs on every frame of a
match, and the game is full of bright circular VFX it will otherwise fire on.

Validation holds out whole recordings, because frames from one recording are near-duplicates and
a frame-level split would score the model on what it trained on. With a held-out set `patience`
means something: training stops when unseen recordings stop improving, and best.pt is the epoch
that scored best on them. `--no-val` goes back to validating on the training set. Use it only for
a final fit on everything, once a held-out run has told you how many epochs to ask for.
"""
import argparse
from pathlib import Path

from brawl_vision.object_detection.projectile_detection.prepare import (
    DEFAULT_VAL_FRAC, audit, find_archive, prepare,
)

HERE = Path(__file__).resolve().parent


def _parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", help="the CVAT export: a path, or a file name beside train.py "
                                      "(default: the newest *.zip there)")
    split = ap.add_mutually_exclusive_group()
    split.add_argument("--val", help="recordings to hold out, comma-separated; a substring that "
                                     "matches one name will do (default: chosen so every class "
                                     "has about --val-frac of its boxes held out)")
    split.add_argument("--no-val", action="store_true",
                       help="hold nothing out and validate on the training set")
    ap.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                    help=f"target share of each class's boxes to hold out (default "
                         f"{DEFAULT_VAL_FRAC})")
    ap.add_argument("--audit-only", action="store_true",
                    help="check, unpack, split and print the audit, then stop without training")
    ap.add_argument("--name", default="projectiles", help="run name under runs/")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--weights", default=str(HERE / "yolo26n.pt"),
                    help="checkpoint to fine-tune from (default: the COCO yolo26n.pt)")
    args = ap.parse_args(argv)
    if args.imgsz % 32:
        ap.error(f"--imgsz {args.imgsz} is not a multiple of 32; Ultralytics would quietly round "
                 f"it up to {-(-args.imgsz // 32) * 32}")
    return args


def main(argv=None) -> None:
    args = _parse(argv)
    archive = find_archive(args.archive)
    print(f"archive: {archive}" + ("" if args.archive else "  (the newest *.zip here)"))
    ready = prepare(archive, val=args.val, val_frac=args.val_frac, no_val=args.no_val)
    print()
    print(audit(ready, args.imgsz))
    print()
    if args.audit_only:
        print(f"--audit-only: stopping before training. To train on exactly this split, drop "
              f"--audit-only; the data file is {ready.data_yaml}")
        return

    from ultralytics import YOLO    # here, not at the top, so --audit-only needs no torch

    model = YOLO(args.weights)

    print("Starting training...")
    print("-" * 60)

    results = model.train(
        data=str(ready.data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device='cuda',
        patience=20,
        save=True,
        project=str(HERE / "runs"),
        name=args.name,
        verbose=True,
        plots=True,
        workers=0,
        lr0=5e-4,
        lrf=5e-5,
        optimizer="AdamW",

        fliplr=0.5,
        translate=0.1,
        scale=0.25,
        degrees=0.0,
        flipud=0.0,
        hsv_h=0.015,
        close_mosaic=25,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Export to ONNX, the format brawl_vision/object_detection/detector.py already runs.
    best = Path(results.save_dir) / "weights" / "best.pt"
    trained = YOLO(str(best))
    print(f"\nweights -> {best}")
    print(f"classes -> {trained.names}")
    print(f"onnx    -> {trained.export(format='onnx', imgsz=args.imgsz, opset=12)}")


if __name__ == '__main__':
    main()
