"""Label a whole game map cell by cell, over a top-down image of it.
archive/KNOWN_MAP_LOCALIZATION_PLAN.md sections 4 and 5.

    .venv/Scripts/python.exe scripts/map_label.py dark_passage --image <path to the png>
    .venv/Scripts/python.exe scripts/map_label.py dark_passage                 # resume
    .venv/Scripts/python.exe scripts/map_label.py dark_passage --review        # look, do not edit
    .venv/Scripts/python.exe scripts/map_label.py dark_passage --mark 20,13 6,39   # ring cells

Files live in `brawl_deployment/data/maps/`: `<name>.png` (the image; `--image` copies it in the
first time), `<name>.csv` (the label, `perception/known_map.py` has the format) and `<name>.json`
(the lattice the label was drawn against, so a reopened label sits where it was drawn).

The window
----------
The image is on the left and the label on the right. The label panel paints every cell opaque in
its class colour, with each colour's key and count in the legend on top, so what is labelled never
depends on how a map's own colours mix with a tint. Both panels share one view and either takes
paint. Under the cursor, both panels box the cell and outline in yellow its cluster, the cells `c`
would fill; the line under the panels gives the cell's (col, row) and class and the cluster's
make-up.

    1 . floor   2 # wall   3 b bush   4 ~ water   5 f fence   6 S spawn   0 ? clear
    click / drag  paint the held class onto cells
    c             give the held class to the WHOLE CLUSTER under the cursor, the outlined one
                  (spawns keep theirs; paint over one by hand to change it)
    u             undo the last stroke or cluster fill
    g             cycle what the image panel draws over the image: nothing, clusters, label tint
    scroll        zoom both panels about the cursor      right-drag  pan      h  the whole map
    n             centre the next --mark cell (ringed in cyan)
    s             save          q  save and quit

Closing the window saves unsaved edits too (`--review` never writes). The toolbar's zoom and pan
also work, but while one is on, clicks zoom or pan instead of painting; the line under the panels
says so.

The first pass
--------------
With no CSV yet, the tool fits the lattice, stamps every spawn (the white ring the image draws on
one), and leaves the terrain `?`, magenta on the label panel. Hover a cell, press its class's key,
then `c`: the outlined cluster turns that class's colour on the label panel. Then fix the
stragglers. Save runs `known_map.check` and prints what it says, the count of each class, and the
spawn cells; a label that fails is still written.

There are no crates to label: power cube boxes spawn semi-randomly each match, so a map image shows
only the fixed terrain. The blue-and-green sprites on Dark Passage are candles, which are walls.

Reading the image: label a raised sprite at its BASE
----------------------------------------------------
A wall, stump, tombstone or candle is drawn standing on its cell's bottom edge and rises about half
a tile into the cell above. The next row's sprite covers the bottom ~40-50% of a cell, and a
sprite's shadow darkens the top ~0.2 of the cell below it. So a cell's own content is its
upper-middle band, rows 0.18-0.44 and columns 0.2-0.8, and that band is all the clustering reads
(`band_colours`). Measured on Dark Passage at 8x zoom, 2026-09-27.
"""
import argparse
import datetime
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Patch, Rectangle
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from brawl_deployment.perception.known_map import (  # noqa: E402
    LEGEND, MAP_TILES, MAPS_DIR, UNLABELLED, check, read_chars, write_chars,
)
from brawl_sim.constants import TILE_TO_CHAR, Tile                   # noqa: E402
from brawl_sim.render.viewer import TILE_COLORS                      # noqa: E402

KEYS = {"1": ".", "2": "#", "3": "b", "4": "~", "5": "f", "6": "S", "0": UNLABELLED}
KEY_OF = {char: key for key, char in KEYS.items()}
_OURS = set(KEYS) | {"c", "g", "s", "u", "q", "n", "h"}
NAMES = {TILE_TO_CHAR[t]: t.name for t in Tile} | {UNLABELLED: "CLEAR"}
SHOWN = {char: name.lower() for char, name in NAMES.items()} | {UNLABELLED: "unlabelled"}
ORDER = tuple(LEGEND) + (UNLABELLED,)      # how counts are listed
MARKERS = ("S",)                           # cells a cluster fill leaves alone

# The label panel paints each cell opaque in the sim palette, so a map's own colours never blend
# into a class: on Crystal Eye Castle's purple floor the old 15% floor tint and 25% magenta `?`
# looked alike (2026-11-03). SPAWN gets its own colour; the sim draws it as floor.
PANEL = {TILE_TO_CHAR[t]: TILE_COLORS[t] for t in Tile} | {"S": "#e8342a", UNLABELLED: "#ff00ff"}
TINT_ALPHA = {".": 0.0, UNLABELLED: 0.35}  # the image panel's label tint; other classes 0.5
HOVER, CURSOR, MARK = "#ffe600", "#ffffff", "#00e5ff"   # cluster outline, cell box, --mark rings
ZOOM_STEP = 1.25                           # per scroll-wheel notch
MARK_VIEW = 8                              # tiles each way of a cell `n` centres

BAND_ROWS = (0.18, 0.44)
BAND_COLS = (0.2, 0.8)

# Dark Passage, 2026-09-28: the 10 spawn rings score .56-.62 and the best other spot .12.
SPAWN_SCORE = 0.45


def _release_matplotlib_keys():
    """Take back the keys matplotlib binds by default (`s` save dialog, `q` close, `g` grid, `c`
    back). Both handlers would run, so the tool would seem to do something random alongside what was
    asked. The same fix as `vision_label.py`'s, for this tool's keys."""
    for name, value in list(matplotlib.rcParams.items()):
        if name.startswith("keymap.") and isinstance(value, list):
            keep = [k for k in value if k not in _OURS]
            if keep != value:
                matplotlib.rcParams[name] = keep


# ---------------------------------------------------------------------------
# the lattice
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Lattice:
    """Where the tiles are in the image. Image coordinates put pixel i on [i, i+1)."""
    pitch: tuple[float, float]      # pixels per tile, (x, y)
    origin: tuple[float, float]     # tile (0, 0)'s top-left corner

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        """`(col, row)` of the tile holding image point (x, y)."""
        return (int(np.floor((x - self.origin[0]) / self.pitch[0])),
                int(np.floor((y - self.origin[1]) / self.pitch[1])))

    def span(self, index: int, frac: tuple[float, float], axis: int) -> slice:
        """The pixels whose centres fall in fractions `frac` of tile `index` along `axis`."""
        start = self.origin[axis] + index * self.pitch[axis]
        lo = int(np.ceil(start + frac[0] * self.pitch[axis] - 0.5))
        hi = int(np.floor(start + frac[1] * self.pitch[axis] - 0.5))
        return slice(max(lo, 0), max(hi + 1, 0))

    @property
    def extent(self) -> tuple[float, float, float, float]:
        cols, rows = MAP_TILES
        x0, y0 = self.origin
        return (x0, x0 + cols * self.pitch[0], y0 + rows * self.pitch[1], y0)

    def inside(self, shape) -> np.ndarray:
        """`(h, w)` bool: pixels on the map, which leaves out any frame around it."""
        x0, x1, y1, y0 = self.extent
        ys, xs = np.mgrid[:shape[0], :shape[1]] + 0.5
        return (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)

    def to_dict(self) -> dict:
        return {"pitch_px": [round(v, 4) for v in self.pitch],
                "origin_px": [round(v, 3) for v in self.origin]}

    @classmethod
    def from_dict(cls, raw: dict) -> "Lattice":
        return cls(tuple(raw["pitch_px"]), tuple(raw["origin_px"]))


def _floor(img: np.ndarray) -> np.ndarray:
    """`(h, w)` bool: pixels in the commonest colours that together cover half the image. That is
    floor on any map, which is most of one, and it takes both checker shades however far apart."""
    q = (img // 8).astype(np.int32)
    key = (q[..., 0] * 32 + q[..., 1]) * 32 + q[..., 2]
    counts = np.bincount(key.ravel())
    order = np.argsort(-counts)
    return np.isin(key, order[:np.searchsorted(np.cumsum(counts[order]), key.size / 2) + 1])


def _edge_profile(lum: np.ndarray, floor: np.ndarray, axis: int) -> np.ndarray:
    """Summed luminance steps between neighbouring FLOOR pixels along `axis`, one value per boundary
    position. Only the floor checkerboard steps at every tile boundary; sprite edges would blur it,
    worst across rows, where raised sprites stand half a tile off their cells. Steps are clipped at
    12 levels so texture cannot outvote the checkerboard."""
    step = np.minimum(np.abs(np.diff(lum, axis=axis)), 12.0)
    both = np.delete(floor, 0, axis) & np.delete(floor, -1, axis)
    return (step * both).sum(axis=1 - axis)


def _comb(profile: np.ndarray, pitches: np.ndarray) -> np.ndarray:
    """`(len(pitches), 2)`: for each pitch, the highest mean of `profile` over an evenly spaced
    comb of that pitch, and that comb's offset."""
    xs = np.arange(len(profile), dtype=np.float64)
    out = np.zeros((len(pitches), 2))
    for i, p in enumerate(pitches):
        offsets = np.arange(0.0, p, 0.05)
        pos = offsets[:, None] + np.arange(int(len(profile) / p) + 1)[None, :] * p
        ok = pos <= len(profile) - 1
        score = (np.interp(np.where(ok, pos, 0.0), xs, profile) * ok).sum(1) / ok.sum(1)
        out[i] = score.max(), offsets[score.argmax()]
    return out


def fit_lattice(img: np.ndarray) -> tuple[Lattice, float]:
    """The lattice, from the floor checkerboard, and how clearly it showed: the winning comb's score
    over the best comb more than 0.15 px of pitch away, on the worse axis. Dark Passage gives 2.8;
    under 1.5, check the grid lines by eye.

    Per axis: the pitch is searched between 0.6 and 1.02 of the image size over 60, which admits a
    frame up to 40% of the image and never admits twice the pitch. The comb only finds boundaries
    modulo the pitch, so tile 0 is the boundary from which 60 tiles fit inside the image, the most
    centred one if several do.
    """
    lum = img.astype(np.float32).mean(axis=2)
    floor = _floor(img)
    pitch, origin, clarity = [], [], []
    for axis, n in ((1, MAP_TILES[0]), (0, MAP_TILES[1])):
        profile = _edge_profile(lum, floor, axis)
        size = img.shape[axis]
        coarse = np.arange(0.6 * size / n, 1.02 * size / n, 0.02)
        scores = _comb(profile, coarse)
        p = coarse[scores[:, 0].argmax()]
        runner_up = scores[np.abs(coarse - p) > 0.15, 0].max()
        fine = np.arange(p - 0.02, p + 0.02, 0.001)
        scores = _comb(profile, fine)
        best = scores[:, 0].argmax()
        p, offset = float(fine[best]), float(scores[best, 1])
        # The step between pixels i and i+1 is the boundary at image coordinate i + 1.
        first, span = offset + 1.0, n * p
        starts = [first + k * p for k in range(-n, n + 1)
                  if first + k * p >= -0.5 and first + k * p + span <= size + 0.5]
        if not starts:
            raise ValueError(f"no {n}-tile span of pitch {p:.3f} px fits the {size} px image; "
                             f"pass --pitch and --origin, or --corners")
        pitch.append(p)
        origin.append(min(starts, key=lambda s: abs(s - (size - span - s))))
        clarity.append(scores[best, 0] / runner_up)
    return Lattice(tuple(pitch), tuple(origin)), min(clarity)


# ---------------------------------------------------------------------------
# the first pass
# ---------------------------------------------------------------------------

def _channels(img):
    return (img[..., i].astype(np.int32) for i in range(3))


def find_spawns(img: np.ndarray, lat: Lattice) -> list[tuple[int, int]]:
    """`(col, row)` of every spawn, from the white ring the image draws on each, raster order."""
    b, g, r = _channels(img)
    white = ((r > 150) & (g > 140) & (b > 140) & (np.abs(r - g) < 45) & (np.abs(g - b) < 45)
             & lat.inside(img.shape))
    p = float(np.mean(lat.pitch))
    radius, thick = 0.64 * p, max(1, int(round(0.16 * p)))
    half = int(np.ceil(radius + 0.16 * p))
    ring = np.zeros((2 * half + 1, 2 * half + 1), np.float32)
    cv2.circle(ring, (half, half), int(round(radius)), 1.0, thick)
    score = cv2.matchTemplate(white.astype(np.float32), ring, cv2.TM_CCORR) / ring.sum()
    keep = int(round(1.2 * p))
    cells = []
    while True:
        _, top, _, (x, y) = cv2.minMaxLoc(score)
        if top < SPAWN_SCORE:
            break
        cells.append(lat.cell_of(x + half + 0.5, y + half + 0.5))
        score[max(0, y - keep):y + keep + 1, max(0, x - keep):x + keep + 1] = 0.0
    return sorted(set(cells), key=lambda cr: (cr[1], cr[0]))


def band_colours(img: np.ndarray, lat: Lattice) -> np.ndarray:
    """`(rows, cols, 3)` float32: the median BGR of each cell's upper-middle band (see the module
    docstring for why that band)."""
    cols, rows = MAP_TILES
    out = np.zeros((rows, cols, 3), np.float32)
    for r in range(rows):
        ys = lat.span(r, BAND_ROWS, axis=1)
        for c in range(cols):
            out[r, c] = np.median(img[ys, lat.span(c, BAND_COLS, axis=0)].reshape(-1, 3), axis=0)
    return out


def propose_clusters(img: np.ndarray, lat: Lattice, k: int = 12, seed: int = 0) -> np.ndarray:
    """`(rows, cols)` int32 cluster of each cell by its band colour, in Lab so that distance means
    what it looks like. A proposal to name with `c`, never a label."""
    colours = band_colours(img, lat).astype(np.uint8)
    lab = cv2.cvtColor(colours.reshape(1, -1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, _ = cv2.kmeans(lab.astype(np.float32), k, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(colours.shape[:2]).astype(np.int32)


def first_pass(img: np.ndarray, lat: Lattice) -> np.ndarray:
    """A label with every spawn stamped and the terrain left `?`."""
    cols, rows = MAP_TILES
    chars = np.full((rows, cols), UNLABELLED, dtype="<U1")
    for c, r in find_spawns(img, lat):
        if 0 <= c < cols and 0 <= r < rows:
            chars[r, c] = "S"
    return chars


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------

class MapLabeller:
    """The window: the image on the left, the label on the right (the module docstring has the
    controls). Both panels share one view, so zooming or panning either moves both."""
    MODES = ("none", "clusters", "label")      # what the image panel draws over the image

    def __init__(self, name, img, lat, chars, clusters, paths, review=False, mode="none",
                 marks=()):
        self.name, self.img, self.lat, self.chars = name, img, lat, chars
        self.clusters, self.paths, self.review, self.mode = clusters, paths, review, mode
        self.marks = list(marks)               # (col, row), in the order `n` visits them
        self.held = "."
        self.strokes: list[list[tuple[int, int, str]]] = []
        self.painting = False
        self.pan_from = None                   # where a right-button drag started
        self.hover = None                      # (row, col) under the cursor
        self.outlined = None                   # (clusters array, cluster id) the outline shows
        self.visits = 0                        # `n` presses so far
        self.dirty = False                     # edits since the last save
        self.message = ""                      # what the last key did
        h, w = img.shape[:2]
        self.full = ((0.0, float(w)), (float(h), 0.0))
        self.fig, (self.ax, self.ax_label) = plt.subplots(1, 2, sharex=True, sharey=True,
                                                          figsize=(18, 9.6))
        self.axes = (self.ax, self.ax_label)
        self.fig.subplots_adjust(left=0.005, right=0.995, top=0.935, bottom=0.075, wspace=0.01)
        # RGBA uint8 draws ~15% faster than RGB; the image is the costliest thing in a redraw.
        rgba = np.dstack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB), np.full((h, w), 255, np.uint8)])
        self.ax.imshow(rgba, extent=(0, w, h, 0), interpolation="nearest")
        # The cell layers are meshes: a 60 x 60 image is resampled to the panel's full size on
        # every draw (~33 ms each at 2560 x 1340), a mesh fills 3600 quads (~2 ms).
        cols, rows = MAP_TILES
        self.edges = (lat.origin[0] + np.arange(cols + 1) * lat.pitch[0],
                      lat.origin[1] + np.arange(rows + 1) * lat.pitch[1])
        self.overlay = self.ax.pcolormesh(*self.edges, self._tint(), antialiased=False,
                                          edgecolors="none")
        self.ax_label.set_facecolor("#9a9a9a")
        self.label_view = self.ax_label.pcolormesh(*self.edges, self._panel(),
                                                   antialiased=False, edgecolors="none")
        self.ax_label.set_aspect("equal")      # imshow sets it on the image panel; a mesh does not
        self._furnish()
        self.ax.set_xlim(*self.full[0])
        self.ax.set_ylim(*self.full[1])
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None:
            toolbar.push_current()   # the toolbar's Home button: this whole-map view
        self.legend = self.fig.legend(
            handles=[Patch(facecolor=PANEL[c], edgecolor="black", linewidth=0.6)
                     for c in KEYS.values()],
            labels=[SHOWN[c] for c in KEYS.values()], loc="upper center", ncol=len(KEYS),
            frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.995), columnspacing=2.5)
        self.status = self.fig.text(0.006, 0.047, "", fontsize=10, family="monospace")
        self.note = self.fig.text(0.006, 0.015, "", fontsize=10, family="monospace")
        # The hover layer is blitted over the last full draw, so following the mouse never
        # redraws the map image (_blit). Listed in drawing order: the outline under the box.
        self.hover_layer = [*self.outlines, *self.cursors, self.status, self.note]
        for artist in self.hover_layer:
            artist.set_animated(True)
        self.background = None
        self._refresh()
        for event, handler in (("draw_event", self._on_draw),
                               ("key_press_event", self.on_key),
                               ("button_press_event", self.on_press),
                               ("button_release_event", self.on_release),
                               ("motion_notify_event", self.on_move),
                               ("scroll_event", self.on_scroll),
                               ("close_event", self.on_close)):
            self.fig.canvas.mpl_connect(event, handler)
        self._maximise()

    # -- drawing ----------------------------------------------------------------

    def _furnish(self):
        """Grid lines, the cluster outline, the cursor box and the --mark rings, on both panels.
        The outline is two plain line collections, black under yellow: a path-effect halo would
        draw its few thousand edges one at a time on every hover."""
        lat = self.lat
        x0, x1, y1, y0 = lat.extent
        xs, ys = self.edges
        halo = [patheffects.withStroke(linewidth=3.5, foreground="black")]
        self.outlines, self.cursors = [], []
        for ax, grid, alpha in ((self.ax, "white", 0.3), (self.ax_label, "black", 0.15)):
            ax.vlines(xs, y0, y1, colors=grid, lw=0.3, alpha=alpha)
            ax.hlines(ys, x0, x1, colors=grid, lw=0.3, alpha=alpha)
            for colour, width in (("black", 3.2), (HOVER, 1.4)):
                edges = LineCollection([], colors=colour, linewidths=width, zorder=5)
                ax.add_collection(edges, autolim=False)
                self.outlines.append(edges)
            box = Rectangle((x0, y0), *lat.pitch, fill=False, edgecolor=CURSOR, linewidth=1.8,
                            zorder=6, visible=False, path_effects=halo)
            ax.add_patch(box)
            self.cursors.append(box)
            if self.marks:
                rings = PatchCollection(
                    [Circle(self._centre(c, r), 0.8 * max(lat.pitch)) for c, r in self.marks],
                    facecolors="none", edgecolors=MARK, linewidths=1.8, zorder=7,
                    path_effects=halo)
                ax.add_collection(rings, autolim=False)
            ax.set_xticks([])
            ax.set_yticks([])

    def _maximise(self):
        """Fill the screen on Tk, this repo's backend on Windows; elsewhere keep the figsize.
        Deferred: the window is still withdrawn here, and `plt.show` deiconifies it."""
        window = getattr(self.fig.canvas.manager, "window", None)
        if hasattr(window, "after") and hasattr(window, "state"):
            window.after(200, _zoom_window, window)

    def _centre(self, col, row) -> tuple[float, float]:
        (px, py), (ox, oy) = self.lat.pitch, self.lat.origin
        return ox + (col + 0.5) * px, oy + (row + 0.5) * py

    def _colours(self, alpha) -> np.ndarray:
        """`(rows, cols, 4)`: each cell in its class colour at opacity `alpha(char)`."""
        out = np.zeros(self.chars.shape + (4,), np.float32)
        for char in np.unique(self.chars):
            out[self.chars == char] = to_rgba(PANEL.get(char, PANEL[UNLABELLED]), alpha(char))
        return out

    def _panel(self) -> np.ndarray:
        """The label panel: opaque, so what it shows never depends on the map's colours."""
        return self._colours(lambda char: 1.0)

    def _tint(self) -> np.ndarray:
        """What the image panel draws over the image, by `mode`."""
        if self.mode == "clusters":
            out = matplotlib.colormaps["tab20"](self.clusters % 20).astype(np.float32)
            out[..., 3] = 0.5
            return out
        if self.mode == "label":
            return self._colours(lambda char: TINT_ALPHA.get(char, 0.5))
        return np.zeros(self.chars.shape + (4,), np.float32)

    def _outline(self, mask: np.ndarray) -> np.ndarray:
        """`(n, 2, 2)`: the cell edges between `mask` and the rest, in image pixels."""
        (px, py), (ox, oy) = self.lat.pitch, self.lat.origin
        m = np.pad(mask, 1)
        r, c = np.nonzero(m[1:-1, 1:] != m[1:-1, :-1])        # the left edge of cell (r, c)
        x = ox + c * px
        down = np.stack([x, oy + r * py, x, oy + (r + 1) * py], 1)
        r, c = np.nonzero(m[1:, 1:-1] != m[:-1, 1:-1])        # the top edge of cell (r, c)
        y = oy + r * py
        across = np.stack([ox + c * px, y, ox + (c + 1) * px, y], 1)
        return np.concatenate([down, across]).reshape(-1, 2, 2)

    def _refresh(self):
        """Redraw everything the label decides. Every edit calls it, so both panels always show
        the label as it is now."""
        self.label_view.set_array(self._panel())
        self.overlay.set_array(self._tint())
        self.overlay.set_visible(self.mode != "none")
        for text, char in zip(self.legend.get_texts(), KEYS.values()):
            text.set_text(f"{KEY_OF[char]}  {SHOWN[char]}  {int((self.chars == char).sum())}")
            text.set_fontweight("bold" if char == self.held else "normal")
        self.ax.set_title(f"image   overlay: {self.mode}   (g: none, clusters, label)",
                          fontsize=10)
        self.ax_label.set_title("label: what is saved"
                                + ("   (REVIEW ONLY: nothing is saved)" if self.review else ""),
                                fontsize=10)
        self.fig.canvas.manager.set_window_title(
            f"map label {self.name}" + ("  (unsaved edits)" if self.dirty else ""))
        self._status()
        self.fig.canvas.draw_idle()

    def _status(self):
        """The two lines under the panels: the hovered cell and its cluster; then what is held,
        anything blocking a click, whether edits are unsaved, and what the last key did."""
        if self.hover is None:
            line = "hover a cell: its (col, row), its class and its cluster show here"
        else:
            row, col = self.hover
            cid = int(self.clusters[row, col])
            chars, counts = np.unique(self.chars[self.clusters == cid], return_counts=True)
            pairs = sorted(zip(counts.tolist(), chars.tolist()), reverse=True)
            mix = ", ".join(f"{SHOWN[c]} {n}" for n, c in pairs)
            marked = "  [marked]" if (col, row) in self.marks else ""
            line = (f"(col {col}, row {row}) {SHOWN[self.chars[row, col]].upper()}{marked}   "
                    f"cluster {cid}, outlined: {int(counts.sum())} cells ({mix})")
        self.status.set_text(line)
        bits = [f"holding {SHOWN[self.held].upper()} (key {KEY_OF[self.held]})"]
        tool = self._toolbar_mode()
        if tool:
            bits.append(f"TOOLBAR {tool.upper()} IS ON: clicks do not paint until you click "
                        f"its button again")
        if self.review:
            bits.append("review only")
        elif self.dirty:
            bits.append("unsaved edits: s saves, and so does closing the window")
        if self.message:
            bits.append(self.message)
        self.note.set_text("   |   ".join(bits))

    def _on_draw(self, _event):
        """After every full draw, keep the picture without the hover layer, then add the layer.
        Saving a figure draws the layer too, but must not replace the window's background."""
        if not self.fig.canvas.is_saving():
            self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for artist in self.hover_layer:
            self.fig.draw_artist(artist)

    def _blit(self):
        """Redraw only the hover layer over the last full draw. A full draw resamples the map
        image, ~60 ms at 2560 x 1340: too slow to follow the mouse."""
        canvas = self.fig.canvas
        if self.background is None:
            canvas.draw_idle()
            return
        canvas.restore_region(self.background)
        for artist in self.hover_layer:
            self.fig.draw_artist(artist)
        canvas.blit(self.fig.bbox)

    # -- the cursor -------------------------------------------------------------

    def _cell(self, event):
        """`(row, col)` under the cursor on either panel, or None."""
        if event.inaxes not in self.axes or event.xdata is None:
            return None
        col, row = self.lat.cell_of(event.xdata, event.ydata)
        cols, rows = MAP_TILES
        return (row, col) if 0 <= col < cols and 0 <= row < rows else None

    def _toolbar_mode(self) -> str:
        """The toolbar tool that owns clicks ('zoom rect', 'pan/zoom'), or '' if none does."""
        return str(getattr(getattr(self.fig.canvas, "toolbar", None), "mode", "") or "")

    def _toolbar_busy(self) -> bool:
        return bool(self._toolbar_mode())

    def _set_hover(self, cell) -> bool:
        """Box `cell` on both panels and outline its cluster; True if the hover moved."""
        if cell == self.hover:
            return False
        self.hover = cell
        for box in self.cursors:
            box.set_visible(cell is not None)
            if cell is not None:
                box.set_xy((self.lat.origin[0] + cell[1] * self.lat.pitch[0],
                            self.lat.origin[1] + cell[0] * self.lat.pitch[1]))
        cid = None if cell is None else int(self.clusters[cell])
        shown = self.outlined
        if shown is None or shown[0] is not self.clusters or shown[1] != cid:
            self.outlined = (self.clusters, cid)
            edges = np.zeros((0, 2, 2)) if cid is None else self._outline(self.clusters == cid)
            for lines in self.outlines:
                lines.set_segments(edges)
        self._status()
        return True

    # -- mouse ------------------------------------------------------------------

    def on_press(self, event):
        if self._toolbar_busy() or event.inaxes not in self.axes:
            return
        if event.button == 3:                  # right-drag pans
            self.pan_from = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())
            return
        cell = self._cell(event)
        if event.button != 1 or cell is None:
            return
        self.painting = True
        self.strokes.append([])
        self._paint(cell, self.held, self.strokes[-1])
        self.message = ""
        self._refresh()

    def on_release(self, _event):
        self.painting = False
        self.pan_from = None
        if self.strokes and not self.strokes[-1]:
            self.strokes.pop()

    def on_move(self, event):
        if self.pan_from is not None:
            self._pan(event)
            return
        cell = self._cell(event)
        moved = self._set_hover(cell)
        if self.painting and cell is not None and self._paint(cell, self.held, self.strokes[-1]):
            self._refresh()
        elif moved:
            self._blit()

    def _pan(self, event):
        """Drag the view with the cursor: the point grabbed stays under it."""
        x0, y0, (xa, xb), (ya, yb) = self.pan_from
        box = self.ax.bbox
        dx = (event.x - x0) * (xb - xa) / box.width
        dy = (event.y - y0) * (yb - ya) / box.height
        self.ax.set_xlim(xa - dx, xb - dx)
        self.ax.set_ylim(ya - dy, yb - dy)
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        """Zoom both panels about the cursor, a notch at a time, out to the whole image."""
        if event.inaxes not in self.axes or event.xdata is None:
            return
        f = ZOOM_STEP ** -(event.step or (1 if event.button == "up" else -1))
        (xa, xb), (ya, yb) = self.ax.get_xlim(), self.ax.get_ylim()
        if (xb - xa) * f >= self.full[0][1] - self.full[0][0]:
            self._home()
            return
        if (xb - xa) * f < 4 * self.lat.pitch[0]:
            return
        x, y = event.xdata, event.ydata
        self.ax.set_xlim(x - (x - xa) * f, x + (xb - x) * f)
        self.ax.set_ylim(y - (y - ya) * f, y + (yb - y) * f)
        self.fig.canvas.draw_idle()

    def _home(self):
        self.ax.set_xlim(*self.full[0])
        self.ax.set_ylim(*self.full[1])
        self.fig.canvas.draw_idle()

    # -- keys -------------------------------------------------------------------

    def on_key(self, event):
        key = event.key
        if key in KEYS:
            self.held = KEYS[key]
            print(f"holding {NAMES[self.held]} ({self.held})")
        elif key == "g":
            self.mode = self.MODES[(self.MODES.index(self.mode) + 1) % len(self.MODES)]
        elif key == "c":
            self._fill(self._cell(event))
        elif key == "u":
            self._undo()
        elif key == "n":
            self._next_mark()
        elif key == "h":
            self._home()
        elif key == "s":
            self.save()
        elif key == "q":
            self.save()
            plt.close(self.fig)
            return
        else:
            return
        self._refresh()

    def _paint(self, cell, char, stroke) -> bool:
        row, col = cell
        if self.review or self.chars[row, col] == char:
            return False
        stroke.append((row, col, str(self.chars[row, col])))
        self.chars[row, col] = char
        self.dirty = True
        return True

    def _fill(self, cell):
        """`c`: the held class onto every cell of the cluster under the cursor but the spawns."""
        if cell is None:
            print("hover the cursor over a cell before pressing c")
            self.message = "c needs the cursor on a cell"
            return
        cid = int(self.clusters[cell])
        stroke = []
        for r, c in np.argwhere(self.clusters == cid):
            if self.chars[r, c] not in MARKERS:
                self._paint((r, c), self.held, stroke)
        if stroke:
            self.strokes.append(stroke)
        print(f"cluster fill: {len(stroke)} cells to {NAMES[self.held]}")
        self.message = (f"cluster {cid}: {len(stroke)} cells -> {NAMES[self.held]}"
                        + ("   (u undoes)" if stroke else "   (nothing to change)"))

    def _undo(self):
        if not self.strokes:
            self.message = "nothing to undo"
            return
        stroke = self.strokes.pop()
        for row, col, prev in reversed(stroke):
            self.chars[row, col] = prev
        self.dirty = True
        self.message = f"undid {len(stroke)} cells"

    def _next_mark(self):
        """`n`: centre the next --mark cell, MARK_VIEW tiles each way."""
        if not self.marks:
            self.message = "no cells to visit: start the tool with --mark C,R ..."
            return
        i = self.visits % len(self.marks)
        self.visits += 1
        col, row = self.marks[i]
        x, y = self._centre(col, row)
        px, py = self.lat.pitch
        self.ax.set_xlim(x - MARK_VIEW * px, x + MARK_VIEW * px)
        self.ax.set_ylim(y + MARK_VIEW * py, y - MARK_VIEW * py)
        self.message = (f"mark {i + 1} of {len(self.marks)}: (col {col}, row {row}), now "
                        f"{SHOWN[self.chars[row, col]]}")

    # -- saving -----------------------------------------------------------------

    def save(self):
        if self.review:
            print("review mode: not saving")
            self.message = "review only: nothing saved"
            return
        csv, meta = self.paths
        write_chars(csv, self.chars)
        write_sidecar(meta, self.img, self.lat)
        self.dirty = False
        print(f"wrote {csv} and {meta.name}")
        self.message = f"saved {csv.name}: checks {report(self.chars)}"

    def on_close(self, _event):
        """Closing the window keeps the work: the user's first two sessions on Crystal Eye Castle
        (2026-11-03) ended at the window's X, and nothing was written."""
        if self.dirty and not self.review:
            print("window closed with unsaved edits: saving them")
            self.save()


def _zoom_window(window) -> None:
    try:
        window.state("zoomed")
    except Exception:           # tkinter.TclError: X11's Tk has no "zoomed" state
        pass


def write_sidecar(meta: Path, img: np.ndarray, lat: Lattice) -> None:
    h, w = img.shape[:2]
    meta.write_text(json.dumps({
        "image": meta.with_suffix(".png").name, "image_size": [w, h], **lat.to_dict(),
        "origin_convention": "tile (0, 0)'s top-left corner; pixel i spans [i, i+1)",
        "date": datetime.date.today().isoformat(),
    }, indent=1) + "\n")


def report(chars: np.ndarray) -> str:
    """What the save prints: the checks' verdict, the class counts, the spawn cells. Returns the
    verdict."""
    try:
        check(chars)
        verdict = "PASS"
    except ValueError as e:
        verdict = f"FAIL, {e}"
    print(f"checks: {verdict}")
    print("  " + "  ".join(f"{NAMES[c]} {int((chars == c).sum())}" for c in ORDER))
    for marker in MARKERS:
        cells = [(int(c), int(r)) for r, c in np.argwhere(chars == marker)]
        print(f"  {NAMES[marker]} {len(cells)}: " + " ".join(f"({c},{r})" for c, r in cells))
    return verdict


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _adopt_image(src, png: Path) -> None:
    if src is None:
        if not png.exists():
            raise SystemExit(f"no {png}; pass --image the first time")
        return
    src = Path(src)
    if not src.exists():
        raise SystemExit(f"no such image {src}")
    if png.exists():
        if png.read_bytes() != src.read_bytes():
            raise SystemExit(f"{png} already exists and differs from {src}. The label was drawn on "
                             f"it; delete {png.stem}.png/.csv/.json to start this map again.")
        return
    png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, png)
    print(f"copied {src} -> {png}")


def _lattice(args, meta: Path, img: np.ndarray) -> Lattice:
    cols, rows = MAP_TILES
    if args.corners:
        x0, y0, x1, y1 = args.corners
        lat = Lattice(((x1 - x0) / cols, (y1 - y0) / rows), (x0, y0))
        print(f"lattice from --corners: {lat.to_dict()}")
    elif args.pitch or args.origin:
        if not (args.pitch and args.origin):
            raise SystemExit("--pitch and --origin go together")
        pitch = tuple(args.pitch) * (2 if len(args.pitch) == 1 else 1)
        lat = Lattice(pitch[:2], tuple(args.origin))
        print(f"lattice from --pitch/--origin: {lat.to_dict()}")
    elif meta.exists():
        lat = Lattice.from_dict(json.loads(meta.read_text()))
        print(f"lattice from {meta.name}: {lat.to_dict()}")
    else:
        lat, clarity = fit_lattice(img)
        print(f"lattice fitted: {lat.to_dict()}, clarity {clarity:.2f}")
        if clarity < 1.5:
            print("  weak fit: check the grid lines, and pass --pitch/--origin or --corners if off")
    return lat


def _mark(text: str) -> tuple[int, int]:
    """`--mark 20,13` (or `(20,13)`) as `(col, row)`, a cell on the map."""
    try:
        col, row = (int(v) for v in text.strip("() ").split(","))
    except ValueError:
        raise SystemExit(f"--mark takes col,row pairs like 20,13, not {text!r}") from None
    cols, rows = MAP_TILES
    if not (0 <= col < cols and 0 <= row < rows):
        raise SystemExit(f"--mark {text} is off the {cols} x {rows} map")
    return col, row


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="map name, e.g. dark_passage")
    p.add_argument("--image", default=None, help="the top-down map image (first run only)")
    p.add_argument("--pitch", type=float, nargs="+", default=None,
                   help="pixels per tile: one value, or x and y (with --origin)")
    p.add_argument("--origin", type=float, nargs=2, default=None, metavar=("X", "Y"),
                   help="tile (0, 0)'s top-left corner in image pixels (with --pitch)")
    p.add_argument("--corners", type=float, nargs=4, default=None,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="the map's top-left and bottom-right corners in image pixels, for a map "
                        "whose floor has no checkerboard to fit")
    p.add_argument("--clusters", type=int, default=12, help="k for the colour grouping")
    p.add_argument("--review", action="store_true", help="open read-only; never writes")
    p.add_argument("--mark", nargs="+", default=[], metavar="C,R",
                   help="cells to ring on both panels, col,row each (20,13 6,39); n visits them")
    args = p.parse_args(argv)
    marks = [_mark(text) for text in args.mark]

    png, csv, meta = (MAPS_DIR / f"{args.name}{ext}" for ext in (".png", ".csv", ".json"))
    _adopt_image(args.image, png)
    img = cv2.imread(str(png), cv2.IMREAD_COLOR)
    lat = _lattice(args, meta, img)
    if csv.exists():
        chars = read_chars(csv)
        if chars.shape != MAP_TILES[::-1]:
            raise SystemExit(f"{csv} is {chars.shape[1]} x {chars.shape[0]}, not "
                             f"{MAP_TILES[0]} x {MAP_TILES[1]}")
        print(f"resuming {csv}")
    else:
        chars = first_pass(img, lat)
        print(f"first pass: {int((chars == 'S').sum())} spawns; terrain is yours: hover a cell, "
              f"press its class's key, then c to name its whole cluster")
    report(chars)

    _release_matplotlib_keys()
    labeller = MapLabeller(args.name, img, lat, chars, propose_clusters(img, lat, args.clusters),
                           (csv, meta), review=args.review, marks=marks)
    # `mpl_connect` holds bound methods behind WEAK references: an unreferenced labeller is
    # collected at once and every key and click is silently ignored (vision_label.py hit this).
    labeller.fig._brawl_labeller = labeller
    plt.show()
    return 0


if __name__ == "__main__":
    if matplotlib.get_backend().lower() == "agg":
        print("matplotlib is headless (Agg); this tool needs a GUI backend.", file=sys.stderr)
    sys.exit(main())
