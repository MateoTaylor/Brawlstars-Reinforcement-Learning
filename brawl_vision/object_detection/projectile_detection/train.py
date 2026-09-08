"""Fine-tune YOLO26n to detect Brawl Stars projectiles.

    python brawl_vision/object_detection/projectile_detection/train.py

Weights land in `projectile_detection/runs/projectiles<N>/weights/` as best.pt, last.pt and
best.onnx. Re-running never overwrites: Ultralytics increments the run name.

Notes on the values below, so they are not mistaken for defaults worth copying elsewhere:

  IMGSZ=960    Must be a multiple of 32 or Ultralytics rounds it up (900 -> 928, 1000 -> 1024).
               Measured over all 559 boxes, sqrt(area) in the 2002x1126 source runs
               31 / 55 / 97 / 206 / 514 px at min / p10 / median / p90 / max. Letterboxed to 960
               those become 15 / 26 / 47 / 99 / 246. The median sits well clear of the stride-8 P3
               head and only the bottom decile is inside COCO's 32 px "small" bucket. 1280 was
               better for the smallest boxes but pinned the GPU; 640 puts the median at 31 px and
               the floor at 10 px, which is where detections start dropping.
  batch=16     Measured 10.4 GB of the 5070 Ti's 16 GB at imgsz 1280; 960 is 56% of that pixel
               count, so expect roughly 6-7 GB. Raise the batch if you want the headroom back.
  workers=0    Windows has no fork, so every DataLoader worker is a spawned process holding its own
               copy of the dataset. Leave this at 0.
  scale=0.25   Showdown holds a fixed camera zoom, so a projectile's size in pixels is a real cue.
               The 0.5 default teaches the model to ignore the one measurement it can trust.
  degrees=0    The camera never rolls and the world has a fixed up (HUD, health bars, shadows), so
  flipud=0     rotated or flipped frames are images the game will never produce.
  hsv_h=0.015  Doing real work here: every brawler's shots are colour-coded, and with one
               `Projectile` class the model is a step away from learning "pink blob" instead.
  close_mosaic Mosaic manufactures composition variety out of 337 labelled images, but stitches
    =25       four frames with HUD through the middle. The last 25 epochs run on real frames.

The 1380 frames with no label file are background images, not gaps -- an explicit "nothing here".
Keep them: this model runs on every frame of a match, ~80% of which have no projectile, and the
game is full of bright circular VFX it will otherwise fire on.

Validation runs on the training set, because every frame is trained on. mAP will climb toward 1.0
and says nothing about a new match -- judge this model on fresh footage.
"""
import zipfile
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "labeled_projectiles_brawlstars.zip"
DATASET = HERE / "dataset"
DATA_YAML = DATASET / "yolo_data.yaml"
IMGSZ = 960

if __name__ == '__main__':

    # Unpack the CVAT export once. Neither the 906 MB archive nor the 869 MB tree is committed.
    if not (DATASET / "images" / "train").exists():
        print(f"extracting {ARCHIVE.name} ...")
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extractall(DATASET)

    # CVAT's own data.yaml points at a train.txt whose paths do not resolve from here, so write our
    # own. Ultralytics recurses into the frames/<recording>/ subdirectories and finds each label by
    # swapping the last /images/ in the path for /labels/.
    DATA_YAML.write_text(
        f"path: {DATASET.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/train\n"
        f"names:\n"
        f"  0: Projectile\n"
    )

    model = YOLO(str(HERE / "yolo26n.pt"))

    # Train the model
    print("Starting training...")
    print("-" * 60)

    results = model.train(
        data=str(DATA_YAML),
        epochs=300,
        imgsz=IMGSZ,
        batch=16,
        device='cuda',
        patience=20,
        save=True,
        project=str(HERE / "runs"),
        name='projectiles',
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
    print(f"\nweights -> {best}")
    print(f"onnx    -> {YOLO(str(best)).export(format='onnx', imgsz=IMGSZ, opset=12)}")
