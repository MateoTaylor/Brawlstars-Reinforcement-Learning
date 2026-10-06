"""Label a whole game map cell by cell, over a top-down image of it. KNOWN_MAP_LOCALIZATION_PLAN.md
sections 4 and 5.

    .venv/Scripts/python.exe scripts/map_label.py dark_passage --image <path to the png>
    .venv/Scripts/python.exe scripts/map_label.py dark_passage                 # resume
    .venv/Scripts/python.exe scripts/map_label.py dark_passage --review        # look, do not edit

Files live in `brawl_deployment/data/maps/`: `<name>.png` (the image; `--image` copies it in the
first time), `<name>.csv` (the label, `perception/known_map.py` has the format) and `<name>.json`
(the lattice the label was drawn against, so a reopened label sits where it was drawn).

Controls
--------
    1 . floor   2 # wall   3 b bush   4 ~ water   5 f fence   6 S spawn   0 ? clear
    click / drag  paint the held class onto cells
    c             give the held class to the WHOLE CLUSTER under the cursor (spawns keep theirs;
                  paint over one by hand to change it)
    g             cycle the overlay: label, clusters, bare image
    u             undo the last stroke or cluster fill
    s             save          q  save and quit

Zoom and pan with the toolbar. Clicks paint only while no toolbar tool is active.

The first pass
--------------
With no CSV yet, the tool fits the lattice, stamps every spawn (the white ring the image draws on
one), and leaves the terrain `?` with the cluster overlay up. Name each cluster with `c`, then fix
the stragglers. Save runs `known_map.check` and prints what it says, the count of each class, and
the spawn cells; a label that fails is still written.

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
_OURS = set(KEYS) | {"c", "g", "s", "u", "q"}
NAMES = {TILE_TO_CHAR[t]: t.name for t in Tile} | {UNLABELLED: "CLEAR"}
ORDER = tuple(LEGEND) + (UNLABELLED,)      # how counts are listed
MARKERS = ("S",)                           # cells a cluster fill leaves alone

# The sim palette, except that it draws SPAWN as floor. Floor is faint, so the image shows through
# the 70% of the map it covers.
COLOURS = {TILE_TO_CHAR[t]: TILE_COLORS[t] for t in Tile} | {"S": "#ffffff"}
ALPHA = {".": 0.15, UNLABELLED: 0.25}
UNLABELLED_COLOUR = "#ff00ff"

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
    MODES = ("label", "clusters", "image")

    def __init__(self, name, img, lat, chars, clusters, paths, review=False, mode="label"):
        self.name, self.img, self.lat, self.chars = name, img, lat, chars
        self.clusters, self.paths, self.review, self.mode = clusters, paths, review, mode
        self.held = "."
        self.strokes: list[list[tuple[int, int, str]]] = []
        self.painting = False
        self.fig, self.ax = plt.subplots(figsize=(11, 11))
        self.fig.canvas.manager.set_window_title(f"map label {name}")
        h, w = img.shape[:2]
        self.ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent=(0, w, h, 0),
                       interpolation="nearest")
        x0, x1, y1, y0 = lat.extent
        cols, rows = MAP_TILES
        self.ax.vlines(x0 + np.arange(cols + 1) * lat.pitch[0], y0, y1, colors="w", lw=0.3,
                       alpha=0.3)
        self.ax.hlines(y0 + np.arange(rows + 1) * lat.pitch[1], x0, x1, colors="w", lw=0.3,
                       alpha=0.3)
        self.overlay = self.ax.imshow(self._rgba(), interpolation="nearest", extent=lat.extent)
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.title = self.ax.set_title("", fontsize=9)
        self._retitle()
        for event, handler in (("key_press_event", self.on_key),
                               ("button_press_event", self.on_press),
                               ("button_release_event", self.on_release),
                               ("motion_notify_event", self.on_move)):
            self.fig.canvas.mpl_connect(event, handler)

    def _rgba(self) -> np.ndarray:
        from matplotlib.colors import to_rgba
        img = np.zeros(self.chars.shape + (4,), np.float32)
        if self.mode == "clusters":
            img[:] = plt.get_cmap("tab20")((self.clusters % 20) / 19.0)
            img[..., 3] = 0.5
        elif self.mode == "label":
            for char in np.unique(self.chars):
                colour = COLOURS.get(char, UNLABELLED_COLOUR)
                img[self.chars == char] = to_rgba(colour, ALPHA.get(char, 0.55))
        return img

    def _retitle(self):
        counts = "  ".join(f"{c}:{int((self.chars == c).sum())}" for c in ORDER)
        self.title.set_text(
            f"{self.name}   holding {NAMES[self.held]} ({self.held})   overlay: {self.mode}   "
            f"{counts}" + ("   [REVIEW ONLY]" if self.review else ""))
        self.overlay.set_data(self._rgba())
        self.fig.canvas.draw_idle()

    def _cell(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return None
        col, row = self.lat.cell_of(event.xdata, event.ydata)
        cols, rows = MAP_TILES
        return (row, col) if 0 <= col < cols and 0 <= row < rows else None

    def _toolbar_busy(self) -> bool:
        return bool(getattr(getattr(self.fig.canvas, "toolbar", None), "mode", ""))

    def _paint(self, cell, char, stroke):
        row, col = cell
        if self.review or self.chars[row, col] == char:
            return
        stroke.append((row, col, str(self.chars[row, col])))
        self.chars[row, col] = char

    def on_press(self, event):
        if event.button != 1 or self._toolbar_busy():
            return
        cell = self._cell(event)
        if cell is None:
            return
        self.painting = True
        self.strokes.append([])
        self._paint(cell, self.held, self.strokes[-1])
        self._retitle()

    def on_release(self, _event):
        self.painting = False
        if self.strokes and not self.strokes[-1]:
            self.strokes.pop()

    def on_move(self, event):
        if self.painting:
            cell = self._cell(event)
            if cell is not None:
                self._paint(cell, self.held, self.strokes[-1])
                self._retitle()

    def on_key(self, event):
        if event.key in KEYS:
            self.held = KEYS[event.key]
            print(f"holding {NAMES[self.held]} ({self.held})")
        elif event.key == "g":
            self.mode = self.MODES[(self.MODES.index(self.mode) + 1) % len(self.MODES)]
        elif event.key == "c":
            cell = self._cell(event)
            if cell is None:
                print("hover the cursor over a cell before pressing c")
            else:
                stroke = []
                for r, c in np.argwhere(self.clusters == self.clusters[cell]):
                    if self.chars[r, c] not in MARKERS:
                        self._paint((r, c), self.held, stroke)
                if stroke:
                    self.strokes.append(stroke)
                print(f"cluster fill: {len(stroke)} cells to {NAMES[self.held]}")
        elif event.key == "u":
            if self.strokes:
                for row, col, prev in reversed(self.strokes.pop()):
                    self.chars[row, col] = prev
        elif event.key == "s":
            self.save()
        elif event.key == "q":
            self.save()
            plt.close(self.fig)
            return
        self._retitle()

    def save(self):
        if self.review:
            print("review mode: not saving")
            return
        csv, meta = self.paths
        write_chars(csv, self.chars)
        write_sidecar(meta, self.img, self.lat)
        print(f"wrote {csv} and {meta.name}")
        report(self.chars)


def write_sidecar(meta: Path, img: np.ndarray, lat: Lattice) -> None:
    h, w = img.shape[:2]
    meta.write_text(json.dumps({
        "image": meta.with_suffix(".png").name, "image_size": [w, h], **lat.to_dict(),
        "origin_convention": "tile (0, 0)'s top-left corner; pixel i spans [i, i+1)",
        "date": datetime.date.today().isoformat(),
    }, indent=1) + "\n")


def report(chars: np.ndarray) -> None:
    """What the save prints: the checks' verdict, the class counts, the spawn cells."""
    try:
        check(chars)
        print("checks: PASS")
    except ValueError as e:
        print(f"checks: FAIL, {e}")
    print("  " + "  ".join(f"{NAMES[c]} {int((chars == c).sum())}" for c in ORDER))
    for marker in MARKERS:
        cells = [(int(c), int(r)) for r, c in np.argwhere(chars == marker)]
        print(f"  {NAMES[marker]} {len(cells)}: " + " ".join(f"({c},{r})" for c, r in cells))


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
    args = p.parse_args(argv)

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
        mode = "label"
    else:
        chars = first_pass(img, lat)
        print(f"first pass: {int((chars == 'S').sum())} spawns; terrain is yours, name the "
              f"clusters with c")
        mode = "clusters"
    report(chars)

    _release_matplotlib_keys()
    labeller = MapLabeller(args.name, img, lat, chars, propose_clusters(img, lat, args.clusters),
                           (csv, meta), review=args.review, mode=mode)
    # `mpl_connect` holds bound methods behind WEAK references: an unreferenced labeller is
    # collected at once and every key and click is silently ignored (vision_label.py hit this).
    labeller.fig._brawl_labeller = labeller
    plt.show()
    return 0


if __name__ == "__main__":
    if matplotlib.get_backend().lower() == "agg":
        print("matplotlib is headless (Agg); this tool needs a GUI backend.", file=sys.stderr)
    sys.exit(main())
