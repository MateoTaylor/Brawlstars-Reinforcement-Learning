"""Click-to-label terrain cells on a rectified frame. See Terrain_Perception_Build_Plan.md Phase H.

    python scripts/vision_label.py zone_grows_from_east 700
    python scripts/vision_label.py showdown_alternate_map 1200 --clusters 10
    python scripts/vision_label.py zone_grows_from_east 700 --review      # look, do not edit

Labels are drawn on the **rectified** patch, where every cell is the same square and the grid is
fixed, rather than on the raw frame where you would be aiming at a trapezoid and correcting for
perspective in your head.

Controls
--------
    1 . floor     2 # wall     3 b bush     4 ~ water     5 f fence     0 ? clear
    click / drag  paint the held class onto cells
    c             assign the held class to the WHOLE CLUSTER under the cursor
    g             cycle the cluster overlay on/off
    s             save          u  undo last stroke          q  save and quit

Why the cluster key matters
---------------------------
A frame has ~360 labellable cells and maybe six distinct materials. `c` assigns a class to every
cell that looks alike, so the job is "name the six things, then fix the stragglers" instead of
clicking 360 times. The clustering knows nothing about terrain -- it only groups by appearance, so
check what it selected before trusting it. It is a labour-saving proposal, never a label.

**Gassed cells are refused, not painted.** Gas tints the terrain underneath it, so a gassed cell is
not a clean example of anything (plan Section 2 and Phase G). They are drawn hatched and clicking
them does nothing.

Output goes to `tests/fixtures/vision/labels/<clip>_f<frame>.json` -- tracked text in the map-CSV
legend, one line per grid row, so a change to one cell is one character in a diff.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from brawl_sim.constants import TILE_TO_CHAR
from brawl_sim.render.viewer import TILE_COLORS
from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
from brawl_vision.config import load_vision_config
from brawl_vision.sources import open_source
from brawl_vision.terrain.labeling import (
    CLASS_CHARS, CLASSES, UNLABELLED, LabelGrid, propose_clusters,
)
from brawl_vision.terrain.zone import detect_zone

REPO = Path(__file__).resolve().parent.parent
CLIPS = REPO / "tests" / "fixtures" / "vision"
LABELS = CLIPS / "labels"
KEYS = {"1": CLASS_CHARS[0], "2": CLASS_CHARS[1], "3": CLASS_CHARS[2],
        "4": CLASS_CHARS[3], "5": CLASS_CHARS[4], "0": UNLABELLED}
_OURS = set(KEYS) | {"c", "g", "s", "u", "q"}


def _release_matplotlib_keys():
    """Take back the keys matplotlib binds by default.

    Four of this tool's keys are claimed out of the box: `s` opens a save-figure dialog, `q` closes
    the window, `g` toggles the axes grid and `c` navigates the view stack. Our handler still runs,
    but matplotlib's runs too, so the tool appears to do something random alongside what you asked
    for. Nothing about that reads as a keybinding conflict when you hit it.
    """
    for name, value in list(matplotlib.rcParams.items()):
        if name.startswith("keymap.") and isinstance(value, list):
            keep = [k for k in value if k not in _OURS]
            if keep != value:
                matplotlib.rcParams[name] = keep


def _frame_at(clip, index, cfg):
    """Sequential read, deliberately: a label is addressed by frame index, and seeking returns a
    frame NEAR the requested one rather than that one (see tests/fixtures/vision/README.md). A
    label file that points at an approximately-right frame is worse than useless."""
    with open_source(CLIPS / f"{clip}.mp4", cfg) as src:
        for frame in src:
            if frame.index == index:
                return frame.image
    raise SystemExit(f"{clip}.mp4 has no frame {index}")


class Labeller:
    def __init__(self, clip, index, plan, rect, grid, zone, clusters, path, review=False):
        self.plan, self.rect, self.grid, self.path = plan, rect, grid, path
        self.zone, self.clusters, self.review = zone, clusters, review
        self.held = CLASS_CHARS[0]
        self.show_clusters = False
        self.undo: list[tuple[int, int, str]] = []
        self.painting = False
        self.fig, self.ax = plt.subplots(figsize=(15, 9))
        self.fig.canvas.manager.set_window_title(f"label {clip} f{index}")
        self._draw_base(clip, index)
        for event, handler in (("key_press_event", self.on_key),
                               ("button_press_event", self.on_press),
                               ("button_release_event", self.on_release),
                               ("motion_notify_event", self.on_move)):
            self.fig.canvas.mpl_connect(event, handler)

    # -- drawing -------------------------------------------------------------

    def _draw_base(self, clip, index):
        import cv2
        ppt = self.plan.pixels_per_tile
        self.ax.imshow(cv2.cvtColor(self.rect, cv2.COLOR_BGR2RGB))
        cols, rows = self.plan.size_tiles
        for x in range(0, self.plan.size_px[0] + 1, ppt):
            self.ax.axvline(x, color="w", lw=0.3, alpha=0.35)
        for y in range(0, self.plan.size_px[1] + 1, ppt):
            self.ax.axhline(y, color="w", lw=0.3, alpha=0.35)
        self.overlay = self.ax.imshow(self._rgba(), interpolation="nearest",
                                      extent=(0, self.plan.size_px[0], self.plan.size_px[1], 0))
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.title = self.ax.set_title("", fontsize=10)
        self._retitle(clip, index)

    def _rgba(self):
        from matplotlib.colors import to_rgba
        cols, rows = self.plan.size_tiles
        img = np.zeros((rows, cols, 4), np.float32)
        if self.show_clusters and self.clusters is not None:
            cmap = plt.get_cmap("tab10")
            ok = self.clusters >= 0
            img[ok] = cmap((self.clusters[ok] % 10) / 9.0)
            img[..., 3] = np.where(ok, 0.55, 0.0)
            return img
        for tile in CLASSES:
            sel = self.grid.chars == TILE_TO_CHAR[tile]
            img[sel] = to_rgba(TILE_COLORS[tile])
        img[..., 3] = np.where(self.grid.labelled, 0.55, 0.0)
        img[self.zone] = (1.0, 0.0, 0.7, 0.35)          # gassed: refused, shown tinted
        return img

    def _retitle(self, clip, index):
        counts = self.grid.count()
        done = int(self.grid.labelled.sum())
        total = int((~self.zone).sum())
        held = [t.name for t in CLASSES if TILE_TO_CHAR[t] == self.held]
        self.title.set_text(
            f"{clip} f{index}   holding: {held[0] if held else 'CLEAR'} ({self.held})   "
            f"{done}/{total} cells   " + "  ".join(f"{k}:{v}" for k, v in counts.items())
            + ("   [REVIEW ONLY]" if self.review else "")
        )
        self.overlay.set_data(self._rgba())
        self.fig.canvas.draw_idle()
        self._clip, self._index = clip, index

    # -- interaction ---------------------------------------------------------

    def _cell(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return None
        ppt = self.plan.pixels_per_tile
        col, row = int(event.xdata // ppt), int(event.ydata // ppt)
        cols, rows = self.plan.size_tiles
        if not (0 <= col < cols and 0 <= row < rows):
            return None
        return row, col

    def _paint(self, cell, char, record=True):
        row, col = cell
        if self.review or self.zone[row, col]:
            return
        if record:
            self.undo.append((row, col, self.grid.chars[row, col]))
        self.grid.set_cell(row, col, char)

    def on_press(self, event):
        cell = self._cell(event)
        if cell is None:
            return
        self.painting = True
        self._paint(cell, self.held)
        self._retitle(self._clip, self._index)

    def on_release(self, _event):
        self.painting = False

    def on_move(self, event):
        if not self.painting:
            return
        cell = self._cell(event)
        if cell is not None:
            self._paint(cell, self.held)
            self._retitle(self._clip, self._index)

    def on_key(self, event):
        if event.key in KEYS:
            self.held = KEYS[event.key]
            name = next((t.name for t in CLASSES if TILE_TO_CHAR[t] == self.held), "CLEAR")
            print(f"holding {name} ({self.held})")
        elif event.key == "g":
            self.show_clusters = not self.show_clusters
        elif event.key == "c":
            cell = self._cell(event)
            if cell is None:
                print("hover the cursor over a cell before pressing c")
            elif self.clusters is not None:
                target = self.clusters[cell]
                if target >= 0:
                    before = int(self.grid.labelled.sum())
                    for r, c in np.argwhere(self.clusters == target):
                        self._paint((r, c), self.held)
                    print(f"cluster fill: {int(self.grid.labelled.sum()) - before} cells")
        elif event.key == "u":
            if self.undo:
                row, col, prev = self.undo.pop()
                self.grid.chars[row, col] = prev
        elif event.key == "s":
            self.save()
        elif event.key == "q":
            self.save()
            plt.close(self.fig)
            return
        self._retitle(self._clip, self._index)

    def save(self):
        if self.review:
            print("review mode: not saving")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.grid.save(self.path)
        print(f"wrote {self.path}  ({int(self.grid.labelled.sum())} cells)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", help="clip name without .mp4, e.g. zone_grows_from_east")
    p.add_argument("frame", type=int, help="frame index, as ClipReader counts them")
    p.add_argument("--clusters", type=int, default=8, help="k for the appearance grouping")
    p.add_argument("--review", action="store_true", help="open read-only; never writes")
    p.add_argument("--out", default=None, help="override the label file path")
    args = p.parse_args(argv)

    cfg = load_vision_config()
    plan = build_rectify_plan(load_camera_model(), load_hud_mask())
    rect = plan.rectify(_frame_at(args.clip, args.frame, cfg))

    path = Path(args.out) if args.out else LABELS / f"{args.clip}_f{args.frame}.json"
    if path.exists():
        grid = LabelGrid.load(path, plan)
        print(f"resuming {path} ({int(grid.labelled.sum())} cells already labelled)")
    else:
        grid = LabelGrid(clip=args.clip, frame=args.frame, origin_tile=plan.origin_tile,
                         size_tiles=plan.size_tiles, pixels_per_tile=plan.pixels_per_tile)

    zone = detect_zone(rect, plan, cfg).at_least(0.05)
    # `at_least(0.05)`, not `cells`: for training data any tinting at all disqualifies a cell, and
    # over-excluding costs a few examples where under-excluding poisons the label set.
    unusable = zone | ~grid_observed(plan)
    clusters, _ = propose_clusters(rect, plan, k=args.clusters)

    _release_matplotlib_keys()
    labeller = Labeller(args.clip, args.frame, plan, rect, grid, unusable, clusters, path,
                        review=args.review)
    # Hold a reference for the lifetime of the window. `mpl_connect` stores BOUND METHODS BEHIND
    # WEAK REFERENCES, so an unreferenced Labeller is garbage-collected immediately and every
    # handler silently stops firing -- the window still opens and draws correctly, it just ignores
    # every key and click. Attaching it to the figure ties its lifetime to the thing that outlives
    # this function.
    labeller.fig._brawl_labeller = labeller
    plt.show()
    return 0


def grid_observed(plan):
    """Cells with enough real world in them to be labelled at all."""
    cols, rows = plan.size_tiles
    ppt = plan.pixels_per_tile
    return plan.valid.reshape(rows, ppt, cols, ppt).mean(axis=(1, 3)) > 0.9


if __name__ == "__main__":
    if matplotlib.get_backend().lower() == "agg":
        print("matplotlib is headless (Agg); this tool needs a GUI backend.", file=sys.stderr)
    sys.exit(main())
