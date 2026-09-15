"""Per-cell terrain labels: the format, the store, and the cluster proposal that makes labelling
cheap. See Terrain_Perception_Build_Plan.md Phase H.

**Labels live on the RECTIFIED patch, not the raw frame.** Cells there are uniform squares on a
fixed grid, so labelling is clicking a checkerboard instead of mentally undoing perspective while
aiming at a trapezoid. It also means a label is addressed by `(clip, frame, row, col)` and can be
regenerated exactly, provided the rectification geometry has not changed -- which is why every file
records that geometry and `LabelGrid.load` refuses a mismatch rather than silently shifting every
label by half a tile.

**The file format is JSON carrying a list of strings, one per grid row, in the map-CSV legend**
(`. # b ~ f`, `brawl_sim.constants.CHAR_TO_TILE`). JSON rather than a bare CSV because the geometry
has to travel with the grid; a list of equal-length strings rather than nested arrays because that
is what makes a diff readable -- a changed cell shows up as one character on one line.

`?` means **unlabelled**, and it is not a class. Cells under gas, under a loot box, or simply not
looked at yet are all `?`: the plan excludes gassed and box-occluded cells from both training and
inference, and a label file has no reason to distinguish "skipped" from "not reached".
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from brawl_sim.constants import CHAR_TO_TILE, TILE_TO_CHAR, Tile

from ..camera import RectifyPlan
from ..clips import load_bounds
from ..config import DATA_DIR, REPO_ROOT
from ..gameplay import walk_frames

UNLABELLED = "?"

# Where a label's `clip` name is looked up, in order. The fixtures come first so every existing
# label resolves exactly where it always has; training_videos holds the recordings that have no
# fixture copy (the 1080p emulator captures, the 09-04 set).
CLIP_DIRS: tuple[Path, ...] = (REPO_ROOT / "tests" / "fixtures" / "vision",
                               DATA_DIR / "training_videos")

# The five physical terrain classes. SPAWN and BOX are placement markers, never classifier outputs
# (plan Section 2), so they are absent here even though the legend can spell them.
CLASSES: tuple[Tile, ...] = (Tile.FLOOR, Tile.WALL, Tile.BUSH, Tile.WATER, Tile.FENCE)
CLASS_CHARS: tuple[str, ...] = tuple(TILE_TO_CHAR[t] for t in CLASSES)
CLASS_INDEX = {t: i for i, t in enumerate(CLASSES)}


@dataclass
class LabelGrid:
    """One frame's labels, plus the geometry they were drawn against."""
    clip: str
    frame: int
    origin_tile: tuple[int, int]
    size_tiles: tuple[int, int]
    pixels_per_tile: int
    chars: np.ndarray = field(default=None)     # (rows, cols) of '<legend char>' or '?'
    notes: str = ""

    def __post_init__(self):
        cols, rows = self.size_tiles
        if self.chars is None:
            self.chars = np.full((rows, cols), UNLABELLED, dtype="<U1")
        if self.chars.shape != (rows, cols):
            raise ValueError(
                f"label grid is {self.chars.shape} but size_tiles says {(rows, cols)}"
            )

    # -- conversions ---------------------------------------------------------

    @property
    def labelled(self) -> np.ndarray:
        return self.chars != UNLABELLED

    def as_class_index(self) -> np.ndarray:
        """(rows, cols) int8 of indices into `CLASSES`, with -1 where unlabelled. -1 rather than a
        sentinel class so an accidental `argmax` over it cannot silently produce FLOOR."""
        out = np.full(self.chars.shape, -1, np.int8)
        for i, ch in enumerate(CLASS_CHARS):
            out[self.chars == ch] = i
        return out

    def count(self) -> dict[str, int]:
        return {t.name: int((self.chars == TILE_TO_CHAR[t]).sum()) for t in CLASSES}

    def set_cell(self, row: int, col: int, char: str) -> None:
        if char != UNLABELLED and char not in CLASS_CHARS:
            raise ValueError(
                f"{char!r} is not a terrain class. Use one of {CLASS_CHARS} or {UNLABELLED!r}; "
                f"SPAWN and BOX are placement markers, not classifier outputs."
            )
        self.chars[row, col] = char

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "clip": self.clip, "frame": self.frame,
            "origin_tile": list(self.origin_tile), "size_tiles": list(self.size_tiles),
            "pixels_per_tile": self.pixels_per_tile, "notes": self.notes,
            "legend": {c: CHAR_TO_TILE[c].name for c in CLASS_CHARS} | {UNLABELLED: "UNLABELLED"},
            "grid": ["".join(row) for row in self.chars],
        }

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n")

    @classmethod
    def load(cls, path, plan: RectifyPlan | None = None) -> "LabelGrid":
        raw = json.loads(Path(path).read_text())
        grid = np.array([list(r) for r in raw["grid"]], dtype="<U1")
        obj = cls(clip=raw["clip"], frame=int(raw["frame"]),
                  origin_tile=tuple(raw["origin_tile"]), size_tiles=tuple(raw["size_tiles"]),
                  pixels_per_tile=int(raw["pixels_per_tile"]), chars=grid,
                  notes=raw.get("notes", ""))
        if plan is not None:
            obj.check_plan(plan, source=str(path))
        return obj

    def check_plan(self, plan: RectifyPlan, source: str = "") -> None:
        """Refuse geometry the labels were not drawn against.

        A recalibration that moves the window would leave every stored label pointing at a
        different piece of world. Nothing about the file would look wrong, and the classifier would
        train on a systematically shifted target -- so this is an error, not a warning.
        """
        mine = (tuple(self.origin_tile), tuple(self.size_tiles), self.pixels_per_tile)
        theirs = (tuple(plan.origin_tile), tuple(plan.size_tiles), plan.pixels_per_tile)
        if mine != theirs:
            raise ValueError(
                f"{source or 'label grid'} was drawn against origin/size/ppt {mine}, but the "
                f"current rectification is {theirs}. These labels point at different cells now; "
                f"re-label or restore the old calibration."
            )


def load_label_dir(directory, plan: RectifyPlan | None = None) -> list[LabelGrid]:
    """Every `*.json` in `directory`, sorted, each checked against `plan`."""
    return [LabelGrid.load(p, plan) for p in sorted(Path(directory).glob("*.json"))]


# ---------------------------------------------------------------------------
# the frame a label points at
# ---------------------------------------------------------------------------

def find_clip(name: str) -> Path:
    """The recording a label's `clip` names: the first `CLIP_DIRS` entry holding `<name>.mp4`."""
    for directory in CLIP_DIRS:
        path = directory / f"{name}.mp4"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no {name}.mp4 in any of {[str(d) for d in CLIP_DIRS]}"
    )


def read_label_frames(clip: str, indices, viewport: tuple[int, int] | None = None
                      ) -> dict[int, np.ndarray]:
    """`{index: normalized frame}` for each of `indices` that `clip` has, in one decode pass.

    **Sequential, deliberately.** A label is addressed by frame index, and seeking returns a frame
    NEAR the requested one rather than that one (tests/fixtures/vision/README.md). `walk_frames`
    counts every decoded frame, so its index is the one `ClipReader` yields, and it is pixel-for-
    pixel the same image. It `grab()`s the frames in between rather than converting them, which
    makes a late frame in a long clip ~5x faster to reach.

    Indices past the clip's usable range are left out, as `ClipReader` would never yield them.

    `viewport` resizes frames of another size to it. That is for 1080p recordings (the BlueStacks
    and emulator captures), which normalize to 1920x1080 against a camera model calibrated at
    2002x1126. The deployed capture resizes the same way (`brawl_deployment.capture.to_viewport`),
    so a label drawn here sits on the geometry the bot actually sees.
    """
    path = find_clip(clip)
    bounds = load_bounds(path)
    wanted = sorted({int(i) for i in indices if 0 <= int(i) <= bounds["usable"][1]})
    out = {}
    for index, image in walk_frames(path, tuple(bounds["content_box"]), wanted):
        if viewport is not None and image.shape[1::-1] != tuple(viewport):
            image = cv2.resize(image, tuple(viewport), interpolation=cv2.INTER_AREA)
        out[index] = np.ascontiguousarray(image)
    return out


# ---------------------------------------------------------------------------
# cell features and the cluster proposal
# ---------------------------------------------------------------------------

def cell_features(rect: np.ndarray, plan: RectifyPlan) -> np.ndarray:
    """(rows, cols, 6): mean H, S, V plus the standard deviation of each, per cell.

    The spread matters as much as the mean and is what a mean-only feature throws away: a bush and
    a floor tile can share an average colour while one is textured and the other flat, and that is
    exactly the WALL/FENCE-style distinction this has to support.
    """
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV).astype(np.float32)
    cols, rows = plan.size_tiles
    ppt = plan.pixels_per_tile
    blocks = hsv.reshape(rows, ppt, cols, ppt, 3)
    mean = blocks.mean(axis=(1, 3))
    std = blocks.std(axis=(1, 3))
    return np.concatenate([mean, std], axis=2)


def propose_clusters(rect: np.ndarray, plan: RectifyPlan, k: int = 8, seed: int = 0):
    """Group cells by appearance so a whole group can be labelled at once.

    This is the bootstrap the plan asks for, done the cheap way: on one frame there are ~570 cells
    but only a handful of distinct materials, so assigning a class to each of `k` clusters and then
    fixing the stragglers is a very different amount of work from clicking 570 times. It knows
    nothing about terrain and is not a classifier -- it only proposes groups.

    Returns `(labels, valid)`, both (rows, cols); `labels` is -1 where the cell is not usable.
    """
    feats = cell_features(rect, plan)
    cols, rows = plan.size_tiles
    ppt = plan.pixels_per_tile
    seen = plan.valid.reshape(rows, ppt, cols, ppt).mean(axis=(1, 3))
    valid = seen > 0.9
    out = np.full((rows, cols), -1, np.int32)
    x = feats[valid].astype(np.float32)
    if len(x) < k:
        return out, valid
    # Scale each channel to unit variance so V (0-255) does not dominate H (0-179) by range alone.
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-6)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, lab, _ = cv2.kmeans(x, k, None, crit, 5, cv2.KMEANS_PP_CENTERS, )
    out[valid] = lab.ravel()
    return out, valid
