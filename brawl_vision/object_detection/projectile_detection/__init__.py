"""A projectile detector of our own -- the one model in this repo that we train.

`object_detection/` runs somebody else's YOLOv11n and says plainly that nothing there trains
anything. This package is the exception, and it is split so the two halves never meet:

    train.py     A standalone script, not a library. Run it directly; it imports ultralytics and
                 torch and is the only thing here that does. Nothing at inference time imports it.
    prepare.py   What train.py does to a CVAT export before training: check, unpack, split by
                 recording, audit. No torch, so `train.py --audit-only` and its tests need none.
    classes.py   The label strings -- `Projectile`, `Power Cube Box`, `Power Cube Dropped`.
    detect.py    `ProjectileDetector`, a thin subclass of `object_detection.ObjectDetector` that
                 runs the exported ONNX. onnxruntime only, no torch.
    weights.py   Which of your `runs/` is the current model, and what to say when there is none.
    draw.py      Boxes onto a frame, sized for objects a tenth the size of a brawler.

**`train.py` is deliberately not importable from here.** Importing this package to draw a box
would otherwise pull in ultralytics and torch, which the deploy-time path has no business
requiring -- the same reasoning `terrain/evaluate.py` gives for importing `object_detection`
lazily inside the branch that needs it.
"""
from .detect import Detection, ProjectileDetector
from .draw import draw_projectiles

__all__ = ["Detection", "ProjectileDetector", "draw_projectiles"]
