"""Running the trained projectile model over a raw frame. Inference only -- `train.py` is separate
and nothing here imports ultralytics or torch.

**The arithmetic is `ObjectDetector`'s and this subclass adds none of it.** Letterboxing, the
BGR->RGB conversion at the boundary, reading `names` out of the ONNX metadata, and the NMS-free
decode are all one implementation over there, because every one of them is a property of "a YOLO
ONNX export" rather than of which objects it was trained to find. Duplicating the letterbox to own
a second copy is how the two would end up disagreeing about the padding colour.

What this class actually changes is three things, all of them about *provenance*:

  * **Where the file comes from.** The entity weights are fetched from a URL and digest-checked;
    ours come out of a training run on this machine. `weights.py` here explains the split.
  * **Which config section applies.** `projectile.*`, not `detector.*` -- and the thresholds
    genuinely differ. See `from_config`.
  * **No `ignore` in config.** The model has three classes now (`classes.py`). Each consumer
    decides for itself which ones it wants: the render scripts draw all three, and the deploy
    tracker keeps only `Projectile`. A config-wide suppression list would hide the power cubes from
    the renders too, and there is no consumer that needs them gone everywhere. The parent's
    `ignore=` is still reachable through `from_config(..., ignore=...)` for a one-off.

**This model is NMS-free and that changes what `conf` does.** YOLO26 emits a fixed 300 rows per
frame, sorted, one box per object, with no objectness gate ahead of them -- so the padding rows
arrive at ~0.003 and the confidence threshold is the *only* thing standing between you and 300
boxes per frame. With the anchor-grid entity model, an objectness score has already thinned the
field before `conf` sees it. Same knob, more load.

**A projectile is not a brawler and the two failure modes are opposite.** A brawler is large,
persistent and roughly one per identity, so a missed frame is recoverable from the next one. A
projectile is small, on screen for a handful of frames, and there may be six at once from
different sources -- so recall is what matters and a duplicate box costs almost nothing. That
argues for a LOWER threshold here than the entity detector's measured 0.5, and the default says
so, but the number is a starting point rather than a measurement: nobody has scored this model
against held-out footage yet.
"""
from pathlib import Path

from ..detector import Detection, ObjectDetector, providers_for
from . import weights as _weights

__all__ = ["Detection", "ProjectileDetector"]


class ProjectileDetector(ObjectDetector):
    """The projectile model, constructed the same way every other detector in this package is.

    `Detection.label` comes back as whatever the export says. For our training set that is one of
    `Projectile`, `Power Cube Box` or `Power Cube Dropped`, spelled as CVAT spells them, because
    `prepare.py` carries the export's names into `yolo_data.yaml` unchanged. `classes.py` holds
    those strings. `object_detection/draw.py` keys its colours on them and the deploy tracker
    filters on them, so a class renamed in CVAT silently loses its colour and, for `Projectile`,
    its track too. That is why `prepare.py` refuses labels it does not know, and why deployment
    refuses a model with no `Projectile` class (`perception.projectiles.require_projectile_class`).

    The name is historical. It is the model that finds projectiles AND power cubes, and callers
    that want only one kind filter by label.

    Stateless between frames, like its parent. **That is a real limitation here and worth naming**:
    projectiles are the one thing in this game where a track across frames carries information the
    single frame does not -- direction, speed, and therefore where it will be in 250 ms. This class
    deliberately does not attempt that. It reports what one frame contains; turning a sequence of
    boxes into a velocity is a layer above that does not exist yet.
    """

    @classmethod
    def from_config(cls, cfg=None, **overrides) -> "ProjectileDetector":
        """Build from a `VisionConfig`, reading the `projectile.*` section.

        `iou` is not passed, and its absence is the point: the parent stores one because the
        anchor-grid path runs class-wise NMS, and this path runs none at all. Handing it a value
        would put a number in `self.iou` that no code reads -- the kind of setting somebody later
        "tunes" for an afternoon.
        """
        from ...config import VisionConfig
        cfg = cfg or VisionConfig()
        device = overrides.pop("device", cfg.projectile_device)
        providers = overrides.pop("providers", None)
        if providers is None:
            providers = providers_for(device, "projectile.device")
        # `require` rather than a bare path, so a missing model raises with the list of runs that
        # could satisfy it instead of onnxruntime's "No such file or directory".
        path = overrides.pop("path", None)
        kwargs = dict(path=_weights.require(path if path is not None else cfg.projectile_model),
                      conf=cfg.projectile_conf, providers=providers)
        kwargs.update(overrides)
        return cls(**kwargs)

    def __repr__(self) -> str:
        return (f"ProjectileDetector({Path(self.path).name}, {self.provider}, "
                f"conf={self.conf}, classes={self.classes})")
