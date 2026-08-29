"""Debug overlay: every terrain stage side by side, driven by a clip or a live capture. See
Terrain_Perception_Build_Plan.md Phase E.

**Built fourth, not last, because Phases C through I are all debugged by looking.** "Does the
rectification line up", "is the camera drifting", "did that cell lock to the wrong class" are
questions a side-by-side answers in seconds and a column of numbers answers in an afternoon. It
is a working tool for the rest of the plan, not a wrap-up nicety.

**Stages that do not exist yet render as labelled placeholders rather than being omitted.** The
panel layout is therefore stable from the first run to the last, and wiring up a new stage is
filling in one `StageFrame` field -- no layout surgery, and no temptation to skip the overlay
because plumbing it is work.

**The palette comes from `terrain.palette`, not the simulator.** It used to import
`brawl_sim.render.viewer.TILE_COLORS` and
`TILE_CMAP` were promoted from private for exactly this. A perceived occupancy grid drawn in the
same colours as the simulator's replay viewer is directly comparable against it by eye, which is
most of the value of building this at all -- and a second, hand-copied palette would drift.

**This module renders; it does not compute.** It takes whatever the pipeline already produced and
draws it. That keeps the overlay off the inference path entirely (`scripts/vision_watch.py` is
what actually runs stages) and means a rendering bug can never change a measurement.
"""
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, Rectangle

from brawl_sim.constants import Tile

from ..camera import CameraModel, HudMask, RectifyPlan
from ..video import require_ffmpeg
from .occupancy import UNKNOWN
from .palette import TILE_COLORS, UNKNOWN_COLOR

# UNKNOWN must be unmistakably not one of the five terrain classes at a glance, because the whole
# point of Phase I's UNKNOWN state is that "not yet seen" is visibly different from "seen and it
# is floor". The five real colours are sand, dark grey, green, blue and brown -- all muted, all
# natural -- so a saturated magenta is the one hue that cannot be confused with any of them.

# Tile ids are 0..N_TILES-1 and UNKNOWN is -1, so the occupancy panel displays `grid - UNKNOWN`
# against a colormap whose first entry is UNKNOWN's. That keeps the shift in one place instead of
# scattering `+1`s through the drawing code.
OCCUPANCY_CMAP = ListedColormap([UNKNOWN_COLOR] + [TILE_COLORS[t] for t in Tile])

_GRID_EVERY_TILES = 5      # rectified-panel guide lines; every tile would be an opaque mesh
_TRACK_TAIL = 400          # camera-track points retained, at ~30 fps a little over ten seconds
_TRACK_MIN_SPAN_TILES = 4.0   # floor on each track axis; a straight leg is otherwise all noise


@dataclass
class StageFrame:
    """One frame's worth of pipeline output. Every stage field is optional: `None` means "that
    stage does not exist yet or produced nothing for this frame", and the panel says so rather
    than drawing something misleading.

    `raw` and `rect` are BGR, as everything from OpenCV is; conversion happens at draw time so
    callers never have to remember which convention a field is in.
    """
    index: int
    t: float = 0.0
    raw: np.ndarray | None = None                    # normalized viewport, BGR
    rect: np.ndarray | None = None                   # rectified patch, BGR
    camera_tile: tuple[float, float] | None = None   # Phase F, in the plan's tile frame
    zone: np.ndarray | None = None                   # Phase G, bool in rectified space
    occupancy: np.ndarray | None = None              # Phase I, tile ids with UNKNOWN for unseen
    timings: dict[str, float] = field(default_factory=dict)   # stage -> milliseconds


def _bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


class TerrainOverlay:
    """Six panels over one figure, artists created once and updated in place.

    Deliberately a streaming viewer rather than a scrubbable one, unlike the simulator's
    `ReplayViewer`. That viewer holds a whole match in memory as small arrays; one clip here is
    ~1350 frames of 2002x1126x3, which is several gigabytes before any stage output. Playback
    scrubs nothing and buffers nothing; `save()` re-runs the source instead of replaying a cache.
    """

    def __init__(self, plan: RectifyPlan, hud: HudMask | None = None,
                 model: CameraModel | None = None, figsize=(16.0, 9.0)):
        self.plan = plan
        self.hud = hud
        self.model = model
        self.paused = False
        self._track: list[tuple[float, float]] = []
        self.fig, axes = plt.subplots(2, 3, figsize=figsize)
        self.fig.canvas.manager.set_window_title("brawl_vision — terrain")
        (self.ax_capture, self.ax_rect, self.ax_zone), \
            (self.ax_occ, self.ax_cam, self.ax_legend) = axes
        self._build()
        self.fig.tight_layout(rect=(0, 0, 1, 0.96))
        self._status = self.fig.text(0.5, 0.975, "", ha="center", va="top", fontsize=11,
                                     family="monospace")

    # -- construction --------------------------------------------------------

    def _build(self) -> None:
        vw, vh = self.plan.viewport
        rw, rh = self.plan.size_px

        self.ax_capture.set_title("capture — HUD mask, projected tile grid", fontsize=10)
        self.im_capture = self.ax_capture.imshow(np.zeros((vh, vw, 3), np.uint8))
        self.ax_capture.set_xlim(0, vw)
        self.ax_capture.set_ylim(vh, 0)
        if self.hud is not None:
            for rect in self.hud.rects:
                x0, y0, x1, y1 = rect.pixels(vw, vh)
                self.ax_capture.add_patch(Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, fill=False, ec="#ff3b30", lw=1.2, ls="--"))
        # The tile grid projected back onto the capture -- the single most useful thing on screen.
        # A homography maps lines to lines, so two mapped endpoints per line are exact.
        #
        # BOTH planes are drawn when the camera model is available, and that is not decoration.
        # Wall blocks are ~0.88 tiles tall in projected terms, so the GROUND grid lands close to,
        # but deliberately not on, the block-top seams you actually see. Shown alone it reads as
        # "the calibration is slightly off" when it is exactly right; shown next to the wall-top
        # grid -- which does sit on those seams -- the gap between them IS the parallax, and the
        # pair together says at a glance whether the fit still holds across the whole frame.
        self.ax_capture.add_collection(LineCollection(
            self._grid_segments(self.plan.M_inv), colors="#ffd60a", linewidths=0.6, alpha=0.7))
        if self.model is not None:
            # The plan's tile->rectified scaling is `M . H` (since `M` is that scaling composed
            # with `H_inv`); reusing it rather than rebuilding it keeps the two grids anchored to
            # the same tile origin by construction, which is the whole basis of the comparison.
            tile_to_rect = self.plan.M @ self.model.H
            self.ax_capture.add_collection(LineCollection(
                self._grid_segments(self.model.H_top @ np.linalg.inv(tile_to_rect)),
                colors="#00e5ff", linewidths=0.6, alpha=0.7, linestyles="--"))

        self.ax_rect.set_title(f"rectified — {self.plan.size_tiles[0]}x{self.plan.size_tiles[1]} "
                               f"tiles @ {self.plan.pixels_per_tile}px", fontsize=10)
        self.im_rect = self.ax_rect.imshow(np.zeros((rh, rw, 3), np.uint8))
        # Everything outside `valid` is either beyond the viewport or under the HUD. Shown as a
        # translucent wash rather than solid fill so what is underneath stays legible.
        wash = np.zeros((rh, rw, 4), np.float32)
        wash[~self.plan.valid] = (0.0, 0.0, 0.0, 0.55)
        self.ax_rect.imshow(wash)
        ppt = self.plan.pixels_per_tile
        for x in range(0, rw + 1, ppt * _GRID_EVERY_TILES):
            self.ax_rect.axvline(x, color="#ffd60a", lw=0.4, alpha=0.35)
        for y in range(0, rh + 1, ppt * _GRID_EVERY_TILES):
            self.ax_rect.axhline(y, color="#ffd60a", lw=0.4, alpha=0.35)

        self.ax_zone.set_title("zone — Phase G", fontsize=10)
        self.im_zone = self.ax_zone.imshow(np.zeros((rh, rw), np.uint8), cmap="viridis",
                                           vmin=0, vmax=1)
        self.im_zone.set_visible(False)

        self.ax_occ.set_title("occupancy — Phase I", fontsize=10)
        self.im_occ = self.ax_occ.imshow(
            np.zeros(self.plan.size_tiles[::-1], np.int16) - UNKNOWN,
            cmap=OCCUPANCY_CMAP, vmin=0, vmax=len(Tile), interpolation="nearest")
        self.im_occ.set_visible(False)

        self.ax_cam.set_title("camera track — Phase F", fontsize=10)
        (self.line_cam,) = self.ax_cam.plot([], [], "-", color="#0a84ff", lw=1.2)
        (self.dot_cam,) = self.ax_cam.plot([], [], "o", color="#ff3b30", ms=5)
        self.ax_cam.set_aspect("equal")
        self.ax_cam.invert_yaxis()          # tile y runs downward, as in the simulator
        self.ax_cam.grid(alpha=0.25)
        self.ax_cam.set_xlabel("tile x", fontsize=8)
        self.ax_cam.set_ylabel("tile y", fontsize=8)

        self._legend()
        for ax in (self.ax_capture, self.ax_rect, self.ax_zone, self.ax_occ):
            ax.set_xticks([])
            ax.set_yticks([])

        self._pending = {}
        for ax, text in ((self.ax_zone, "Phase G — zone detection\nnot built yet"),
                         (self.ax_occ, "Phase I — occupancy\nnot built yet"),
                         (self.ax_cam, "Phase F — odometry\nnot built yet")):
            self._pending[ax] = ax.text(
                0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#8e8e93", family="monospace")

    def _grid_segments(self, to_capture: np.ndarray) -> np.ndarray:
        """Tile-grid lines in rectified space, mapped to capture pixels by `to_capture`."""
        rw, rh = self.plan.size_px
        step = self.plan.pixels_per_tile * _GRID_EVERY_TILES
        segs = [[[x, 0], [x, rh]] for x in range(0, rw + 1, step)]
        segs += [[[0, y], [rw, y]] for y in range(0, rh + 1, step)]
        flat = np.array(segs, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(flat, to_capture).reshape(-1, 2, 2)

    def _legend(self) -> None:
        self.ax_legend.axis("off")
        self.ax_legend.set_title("legend", fontsize=10)
        # SPAWN and BOX share FLOOR's colour and are placement markers, never classifier outputs
        # (see the plan, Section 2), so listing them here would advertise classes this pipeline
        # cannot emit.
        classes = (Tile.FLOOR, Tile.WALL, Tile.BUSH, Tile.WATER, Tile.FENCE)
        handles = [Patch(facecolor=TILE_COLORS[t], edgecolor="#3a3a3c", label=t.name)
                   for t in classes]
        handles.append(Patch(facecolor=UNKNOWN_COLOR, edgecolor="#3a3a3c", label="UNKNOWN"))
        handles += [
            Patch(facecolor="none", edgecolor="#ff3b30", ls="--", label="HUD mask"),
            Patch(facecolor="none", edgecolor="#ffd60a",
                  label=f"ground grid (every {_GRID_EVERY_TILES})"),
        ]
        if self.model is not None:
            handles.append(Patch(facecolor="none", edgecolor="#00e5ff", ls="--",
                                 label="wall-top grid (+parallax)"))
        self.ax_legend.legend(handles=handles, loc="center", frameon=False, fontsize=10)

    # -- per frame -----------------------------------------------------------

    def update(self, frame: StageFrame) -> list:
        """Draw one `StageFrame`. Returns the artists that changed."""
        touched = []
        if frame.raw is not None:
            self.im_capture.set_data(_bgr(frame.raw))
            touched.append(self.im_capture)
        if frame.rect is not None:
            self.im_rect.set_data(_bgr(frame.rect))
            touched.append(self.im_rect)

        if frame.zone is not None:
            self._reveal(self.ax_zone, self.im_zone)
            self.im_zone.set_data(frame.zone.astype(np.uint8))
            touched.append(self.im_zone)
        if frame.occupancy is not None:
            self._reveal(self.ax_occ, self.im_occ)
            self.im_occ.set_data(np.asarray(frame.occupancy, np.int16) - UNKNOWN)
            touched.append(self.im_occ)
        if frame.camera_tile is not None:
            self._reveal(self.ax_cam)
            self._track.append(tuple(frame.camera_tile))
            del self._track[:-_TRACK_TAIL]
            xs, ys = zip(*self._track)
            self.line_cam.set_data(xs, ys)
            self.dot_cam.set_data([xs[-1]], [ys[-1]])
            # Limits computed from the data rather than left to autoscale. `set_xlim` DISABLES
            # autoscaling on that axis, so the first frame -- one point, zero span -- would floor
            # the limits to the minimum span and freeze them there for the rest of the run, no
            # matter how far the camera walked. That is not a cosmetic bug: the track panel showed
            # +-2 tiles through a 22-tile walk.
            lo_x, hi_x = self._track_bounds([p[0] for p in self._track])
            lo_y, hi_y = self._track_bounds([p[1] for p in self._track])
            self.ax_cam.set_xlim(lo_x, hi_x)
            self.ax_cam.set_ylim(hi_y, lo_y)          # tile y runs downward
            touched += [self.line_cam, self.dot_cam]

        self._status.set_text(self._status_text(frame))
        touched.append(self._status)
        return touched

    @staticmethod
    def _track_bounds(values):
        """Data range with a floor on the span, so a dead-straight leg still reads as a track
        instead of as noise magnified into six-decimal ticks."""
        lo, hi = min(values), max(values)
        mid, span = (lo + hi) / 2, max(hi - lo, _TRACK_MIN_SPAN_TILES)
        half = span / 2 * 1.1
        return mid - half, mid + half

    def _reveal(self, ax, *artists) -> None:
        placeholder = self._pending.pop(ax, None)
        if placeholder is not None:
            placeholder.set_visible(False)
            ax.set_title(ax.get_title().split(" — ")[0], fontsize=10)
        for artist in artists:
            artist.set_visible(True)

    def _status_text(self, frame: StageFrame) -> str:
        bits = [f"frame {frame.index:5d}", f"t={frame.t:7.2f}s"]
        if frame.timings:
            bits.append(" ".join(f"{k} {v:5.1f}ms" for k, v in frame.timings.items()))
            bits.append(f"total {sum(frame.timings.values()):5.1f}ms")
        if self.paused:
            bits.append("PAUSED")
        return "   ".join(bits)

    # -- driving -------------------------------------------------------------

    def play(self, source, fps: int = 20) -> None:
        """Stream `source` (an iterable of `StageFrame`) into a live window. Space pauses."""
        self.fig.canvas.mpl_connect(
            "key_press_event",
            lambda e: setattr(self, "paused", not self.paused) if e.key == " " else None)
        plt.show(block=False)
        for frame in source:
            self.update(frame)
            self.fig.canvas.draw_idle()
            plt.pause(1.0 / max(fps, 1))
            while self.paused and plt.fignum_exists(self.fig.number):
                plt.pause(0.05)
            if not plt.fignum_exists(self.fig.number):
                break

    def save(self, path, source, fps: int = 20, dpi: int = 110) -> None:
        """Render `source` to `path` with no window. `.mp4` needs ffmpeg (on PATH or via the
        optional `imageio-ffmpeg` package); `.gif` always works, at a much larger file size.

        `source` is consumed once and streamed straight to the writer -- a clip's worth of frames
        does not fit in memory, so this cannot buffer and re-play them the way `ReplayViewer` does.
        """
        suffix = Path(path).suffix.lower()
        if suffix == ".mp4":
            # `require_ffmpeg` rather than `locate_ffmpeg` + a message: Phase L needs the same
            # binary and the same sentence, and two copies of an instruction drift apart.
            mpl.rcParams["animation.ffmpeg_path"] = require_ffmpeg()
            writer = FFMpegWriter(fps=fps, bitrate=-1)
        elif suffix == ".gif":
            writer = PillowWriter(fps=fps)
        else:
            raise ValueError(f"unsupported video extension {suffix!r} -- use .mp4 or .gif")

        # The writer is driven directly rather than through FuncAnimation. FuncAnimation wants to
        # know its frame count up front and caches frames to get it, which is exactly what a
        # streamed source cannot provide -- and passing save_count to suppress that silently
        # truncates the output instead of erroring.
        # The first frame is pulled BEFORE the writer opens: an empty source otherwise reaches
        # PillowWriter.finish() and dies on an index error after already creating the file.
        stream = iter(source)
        try:
            first = next(stream)
        except StopIteration:
            raise ValueError(
                f"the frame source was empty, so {path} would contain nothing"
            ) from None
        with writer.saving(self.fig, str(path), dpi):
            for frame in itertools.chain([first], stream):
                self.update(frame)
                writer.grab_frame()
