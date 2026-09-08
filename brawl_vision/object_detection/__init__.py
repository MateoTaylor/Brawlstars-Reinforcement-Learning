"""brawl_vision.object_detection -- third-party YOLO detectors run over the raw frame.

**This is a chunk beside `terrain/`, not a stage inside it.** The package docstring has always
said entity detection lands here and reuses the root modules unchanged; this is that. The two
chunks share `sources.Frame` and nothing else. Terrain answers "what is the world made of" from
the *rectified* patch and accumulates across frames; this answers "who is on screen right now"
from the *raw* frame and remembers nothing. Neither reads the other's output, and the render
scripts compose them rather than either one calling the other.

**The weights are not ours and are not in the repo.** `PylaEntityDetectorV2.onnx` is a
YOLOv11n trained by the PylaAI community on real Brawl Stars footage and published in
AngelFireLA/BrawlStarsBotMaking. `weights.py` fetches it on demand; `weights/NOTICE.md` records
where it came from and under what licence. Nothing here trains anything -- if a detector needs
retraining, that is a different module.

**Why the raw frame and not the rectified one.** Every one of these models was trained on
straight screenshots of the game. The rectified patch is an inverse-perspective warp of that
same pixel data: correct geometry, but a projection no training image ever contained, and a
detector handed one is out of distribution in a way that shows up as silently worse boxes rather
than an error. Detections come back in raw-frame pixels, and turning one into a tile is
`RectifyPlan`'s job at the point where somebody actually needs a tile.

**`hp_detection/` is a stage INSIDE this package, not a chunk beside it.** It reads a brawler's
HP number out of the readout drawn at the top of its own box, so it cannot run without these
detections and never looks at a pixel they did not point it at. Unlike this package's
relationship to `terrain/`, that dependency is real, and the import direction says so. It is not
re-exported here: `from brawl_vision.object_detection.hp_detection import HealthTracker` is the
way in, because a caller should have to know it is asking for a second stage.
"""
from .detector import Detection, ObjectDetector
from .draw import draw_detections

__all__ = ["Detection", "ObjectDetector", "draw_detections"]
