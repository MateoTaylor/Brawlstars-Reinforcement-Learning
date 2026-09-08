"""Gameplay mp4 in, reconstructed-map mp4 out. See Terrain_Perception_Build_Plan.md Phase L.

**This is an artifact, not a diagnostic.** Phase E's `TerrainOverlay` shows six panels of
intermediates to a person asking "which stage is wrong". This shows one canvas to a person asking
"over a whole match, did perception reconstruct the right world?" -- a question six panels actively
obscure, because the thing being judged is buried among five distractions. They are separate tools
on purpose.

The canvas is also a different thing. The overlay renders the **camera-relative** rectified patch,
where tile (0,0) is wherever the camera is. This renders a **world-anchored** map that accumulates:
the canvas holds still and the observed region grows into it.

Two modes:

* `mosaic` needs only Phase F. Each rectified patch is averaged into the world canvas at its
  odometry offset, producing a photographic mosaic. It is the best test odometry will ever get --
  drift appears as *ghosting and double walls*, which no scalar drift number conveys as quickly.
* `map` is the deliverable: Phase I's occupancy grid in the simulator's own palette, `UNKNOWN`
  magenta, filling in as the camera explores.

Three decisions the plan settles in advance, all of them load-bearing:

**Two passes, because a video's frame size is fixed at stream open.** A canvas that grows as the
camera explores is not expressible, and guessing a large one up front wastes most of the frame on
a map 40 tiles across. Pass one runs odometry alone and records the camera track's bounding box;
pass two allocates exactly that and renders into it. Pass one also produces the progress total a
4802-frame clip needs.

**No matplotlib.** Phase E measured the figure path at ~0.49 s/frame, which is 39 minutes for the
longest fixture. The canvas here is composed as a numpy array -- tile ids through a colour lookup
table, `cv2.resize` with `INTER_NEAREST` to output scale -- and handed to `video.VideoSink`. That
is microseconds and leaves the whole thing bounded by decode.

**Timestamps, not frame indices.** These recordings are variable-rate, so `index / fps` drifts
against reality. Output frames are emitted when the clip's own `Frame.t` crosses the next output
instant, and the stated output rate is the real one.

#### Decode is the budget, and the plan's cost model for pass one was wrong

The plan calls pass one "cheap (no classifier, no rendering)". Measured, the stages it skips are
not where the time goes: **decode is 9.6-11.4 ms/frame** on these 2436x1126 60 fps recordings,
against rectify at 1.2 and odometry at 6.9. Two full decodes of the 4802-frame fixture cost ~92 s
on their own, and the clip is 80 s long -- so a naive two-pass render is slower than real time
before a single stage runs, and no amount of avoiding matplotlib fixes it.

Two changes make the criterion reachable:

* **Pass one steps.** It only needs the camera track's *bounding box*, so it reads every Nth frame
  (`scan_step`, default 8) through `ClipReader(step=)`, which advances the decoder with `grab()`
  and skips the conversion-and-copy that is most of the cost -- 2.5 ms per source frame instead of
  12. Sampling the track cannot miss its extent by much at 60 fps, and `_MARGIN_TILES` absorbs both
  that and the extra drift a coarse odometry chain accumulates. Pass two reports if the map ever
  deposited outside the window it sized, so this stays checkable rather than assumed.

* **Classification runs only on rendered frames.** Odometry is a *chain* -- each frame correlates
  against the previous one -- so it wants every frame it can get. Classification is not: terrain is
  static, each frame is classified independently, and the occupancy map accumulates votes across
  frames anyway. At 12 output fps that is ~960 classifications for the long fixture instead of
  4802, and it costs nothing in map quality.

Pass two recomputes odometry rather than replaying pass one's, because the two passes read
different frames. `--step` on pass two trades odometry fidelity for time if a clip needs it;
measured, step 4 lands within 0.4 tiles of step 1 over 22 tiles of travel.
"""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from matplotlib.colors import to_rgb

from ..camera import RectifyPlan
from ..config import VisionConfig
from ..video import VideoSink
from .labeling import CLASSES
from .occupancy import UNKNOWN, OccupancyMap
from .palette import TILE_COLORS, UNKNOWN_COLOR
from .odometry import Odometry
from .zone import detect_zone

_DEFAULT_SCALE = 14          # output pixels per tile in map mode
_MOSAIC_SCALE = 12           # ...and in mosaic mode, where the source is 48 px/tile
# Breathing room around the region pass one predicts. It absorbs two different errors: pass one
# SAMPLES the camera track, and its coarser odometry chain drifts slightly differently from pass
# two's. Three tiles covers both with room -- at 60 fps a step-8 sample is 0.13 s apart, which is
# well under a tile of camera motion, and step-4 odometry was measured within 0.4 tiles of step-1
# over a 22-tile walk.
_MARGIN_TILES = 3


def tile_lut() -> np.ndarray:
    """(len(CLASSES) + 1, 3) uint8 RGB, indexed by `best() + 1` so UNKNOWN (-1) lands at row 0.

    The +1 shift is what lets the whole render be one fancy-index with no masking: `best()` already
    uses -1 for never-observed, and numpy would happily read that as the LAST row instead.
    """
    rows = [to_rgb(UNKNOWN_COLOR)] + [to_rgb(TILE_COLORS[t]) for t in CLASSES]
    return (np.array(rows, np.float32) * 255).round().astype(np.uint8)


@dataclass
class Track:
    """Pass one's output: where the camera was on every frame, and how big a canvas that needs."""
    positions: dict = field(default_factory=dict)      # frame index -> (x, y) tiles
    segments: dict = field(default_factory=dict)       # frame index -> segment id
    statuses: dict = field(default_factory=dict)       # frame index -> odometry status
    times: dict = field(default_factory=dict)          # frame index -> seconds
    n_frames: int = 0
    n_lost: int = 0
    seconds: float = 0.0
    scan_seconds: float = 0.0

    @property
    def n_segments(self) -> int:
        return len(set(self.segments.values())) if self.segments else 0

    def bounds_tiles(self) -> tuple[float, float, float, float]:
        """(lo_x, hi_x, lo_y, hi_y) of the camera positions, or zeros if there are none."""
        if not self.positions:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p[0] for p in self.positions.values()]
        ys = [p[1] for p in self.positions.values()]
        return (min(xs), max(xs), min(ys), max(ys))


def scan_track(source, plan: RectifyPlan, cfg: VisionConfig | None = None,
               progress=None) -> Track:
    """Pass one. Rectify and odometry only -- no classifier, no zone, no rendering."""
    cfg = cfg or VisionConfig()
    odo = Odometry(plan, cfg)
    track = Track()
    t0 = time.perf_counter()
    for frame in source:
        r = odo.update(plan.rectify(frame.image))
        track.positions[frame.index] = r.position_tiles
        track.segments[frame.index] = r.segment
        track.statuses[frame.index] = r.status
        track.times[frame.index] = frame.t
        track.n_frames += 1
        track.n_lost += r.status == "lost"
        track.seconds = frame.t
        if progress is not None and track.n_frames % 200 == 0:
            progress(track.n_frames, frame.t)
    track.scan_seconds = time.perf_counter() - t0
    return track


@dataclass
class Window:
    """The rectangle of the occupancy grid the output covers, in grid indices."""
    row0: int
    row1: int
    col0: int
    col1: int

    @property
    def rows(self) -> int:
        return self.row1 - self.row0

    @property
    def cols(self) -> int:
        return self.col1 - self.col0


def world_window(track: Track, plan: RectifyPlan, grid_h: int, grid_w: int,
                 margin: int = _MARGIN_TILES) -> Window:
    """Grid rectangle covering every cell the camera could deposit into, from the track alone.

    A frame at camera position `p` deposits `plan.origin_tile + p` through `+ size_tiles`, so the
    union over the track is the track's bounding box grown by one frame footprint. Computing this
    from the track rather than from `observed_bounds()` is the whole point of pass one:
    `observed_bounds()` is only final once pass two has finished, and by then the canvas is
    already open.
    """
    lo_x, hi_x, lo_y, hi_y = track.bounds_tiles()
    ox, oy = plan.origin_tile
    cols_t, rows_t = plan.size_tiles
    grid_ox, grid_oy = -(grid_w // 2), -(grid_h // 2)
    col0 = int(np.floor(lo_x + ox)) - grid_ox - margin
    col1 = int(np.ceil(hi_x + ox)) + cols_t - grid_ox + margin
    row0 = int(np.floor(lo_y + oy)) - grid_oy - margin
    row1 = int(np.ceil(hi_y + oy)) + rows_t - grid_oy + margin
    return Window(row0=max(0, row0), row1=min(grid_h, row1),
                  col0=max(0, col0), col1=min(grid_w, col1))


def render_map(occupancy: OccupancyMap, window: Window, lut: np.ndarray,
               scale: int = _DEFAULT_SCALE) -> np.ndarray:
    """The occupancy grid's `window` as an RGB image at `scale` pixels per tile.

    One fancy-index and one nearest-neighbour resize. `INTER_NEAREST` is not a performance choice:
    any interpolation invents colours between a wall and the floor beside it, and a blurred tile
    boundary is exactly the artifact a viewer would read as classifier uncertainty.
    """
    best = occupancy.best()[window.row0:window.row1, window.col0:window.col1]
    rgb = lut[best + 1]
    return cv2.resize(rgb, (window.cols * scale, window.rows * scale),
                      interpolation=cv2.INTER_NEAREST)


class Mosaic:
    """Photographic accumulation of rectified patches in the world frame.

    A running MEAN, not an overwrite. Overwriting shows the newest patch and hides disagreement;
    averaging makes a wall seen at two different odometry offsets appear twice at half weight,
    which is what "drift" looks like when it is real.
    """

    def __init__(self, window: Window, plan: RectifyPlan, grid_origin: tuple[int, int],
                 scale: int = _MOSAIC_SCALE):
        self.window, self.plan, self.scale = window, plan, scale
        self.grid_origin = grid_origin
        self.h, self.w = window.rows * scale, window.cols * scale
        self.total = np.zeros((self.h, self.w, 3), np.float32)
        self.count = np.zeros((self.h, self.w, 1), np.float32)
        ppt = plan.pixels_per_tile
        cols, rows = plan.size_tiles
        self._patch_size = (cols * scale, rows * scale)
        self._valid = cv2.resize(plan.valid.astype(np.uint8), self._patch_size,
                                 interpolation=cv2.INTER_NEAREST).astype(bool)[..., None]

    def place(self, rect: np.ndarray, position: tuple[float, float]):
        """`(patch, valid, dst_slice, src_slice)` for one frame, or None if it misses the canvas.

        Split out of `add` so that `terrain.truth`'s median plate can reuse the placement maths
        without a second copy of the clipping. The two composites differ only in what they do with
        the pixels once they are located, and that difference is the whole reason both exist.
        """
        px, py = position
        ox, oy = self.plan.origin_tile
        gx, gy = self.grid_origin
        # SUB-TILE placement. Rounding the position to whole tiles here throws away the fractional
        # part Phase F went to trouble to measure -- and it is not small: over 800 frames of
        # showdown_alternate_map2 the discarded remainder averages 0.27 tiles, and 80% of frames
        # land more than a quarter tile out. At the reference plate's 44 px/tile that is ~12 px of
        # misregistration per frame, in a random direction, before any averaging runs. Composite
        # a dozen of those and the result is a ~half-tile smear that reads as "the camera is
        # blurry" rather than as an integer-rounding bug.
        fx = (px + ox - gx - self.window.col0) * self.scale
        fy = (py + oy - gy - self.window.row0) * self.scale
        x0, y0 = int(np.floor(fx)), int(np.floor(fy))
        patch = cv2.resize(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB), self._patch_size,
                           interpolation=cv2.INTER_AREA)
        valid = self._valid
        sub_x, sub_y = fx - x0, fy - y0
        if sub_x or sub_y:
            shift = np.float32([[1, 0, sub_x], [0, 1, sub_y]])
            patch = cv2.warpAffine(patch, shift, self._patch_size, flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)
            # The mask moves with the pixels, and is re-thresholded rather than warped loosely:
            # a bilinear edge pixel is part border fill, and letting those through would paint the
            # patch's own boundary into the canvas as if it were world.
            vm = cv2.warpAffine(self._valid[..., 0].astype(np.uint8) * 255, shift,
                                self._patch_size, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
            valid = (vm >= 250)[..., None]
        ph, pw = patch.shape[:2]
        # Clip against the canvas: a frame at the edge of the track can hang off by a tile, and
        # numpy's negative indices would wrap it round to the opposite side rather than drop it.
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(self.w, x0 + pw), min(self.h, y0 + ph)
        if dx1 <= dx0 or dy1 <= dy0:
            return None
        sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
        return (patch, valid,
                (slice(dy0, dy1), slice(dx0, dx1)), (slice(sy0, sy1), slice(sx0, sx1)))

    def add(self, rect: np.ndarray, position: tuple[float, float]) -> None:
        placed = self.place(rect, position)
        if placed is None:
            return
        patch, valid, dst, src = placed
        m = valid[src]
        self.total[dst] += patch[src] * m
        self.count[dst] += m

    def image(self) -> np.ndarray:
        out = np.where(self.count > 0, self.total / np.maximum(self.count, 1), 0.0)
        return out.clip(0, 255).astype(np.uint8)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _side_by_side(source_bgr: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """Source frame left, reconstruction right, matched in height.

    This is what makes a *disagreement* legible: a wall in the map with no wall on screen is
    obvious side by side and invisible alone.
    """
    h = canvas.shape[0]
    src = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    w = max(1, int(round(src.shape[1] * h / src.shape[0])))
    return np.hstack([cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA), canvas])


def crop_to_view(occupancy: OccupancyMap, plan: RectifyPlan, position: tuple[float, float],
                 lut: np.ndarray, scale: int) -> np.ndarray:
    """The accumulated map, cropped to exactly the tiles this frame covers, at `scale` px/tile.

    The `side-by-side` layout puts one frame next to the WHOLE explored map, and at that point the
    two panels are different sizes covering different world -- you can see that the map has walls
    somewhere, not whether it has them where the frame does. This crops the map to the frame's own
    footprint so the panels are the same extent, tile for tile, and a disagreement is a thing you
    can point at.

    Cells outside the grid come back UNKNOWN rather than raising: near the edge of the occupancy
    grid part of the footprint genuinely has no storage behind it, and that is the same "nothing is
    known here" the rest of the map already expresses.
    """
    cols, rows = plan.size_tiles
    ox, oy = plan.origin_tile
    gx, gy = occupancy.origin
    c0 = int(round(position[0] + ox)) - gx
    r0 = int(round(position[1] + oy)) - gy
    best = occupancy.best()
    out = np.full((rows, cols), UNKNOWN, best.dtype)
    sr0, sc0 = max(0, r0), max(0, c0)
    sr1, sc1 = min(best.shape[0], r0 + rows), min(best.shape[1], c0 + cols)
    if sr1 > sr0 and sc1 > sc0:
        out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = best[sr0:sr1, sc0:sc1]
    return cv2.resize(lut[out + 1], (cols * scale, rows * scale),
                      interpolation=cv2.INTER_NEAREST)


def draw_policy_window(canvas: np.ndarray, plan: RectifyPlan, scale: int,
                       view_w: int, view_h: int) -> None:
    """Outline the tiles the POLICY actually receives, inside the camera's footprint.

    The footprint is what the camera covers; the policy's window is the largest hero-centred
    rectangle inside it (Phase K), which is 21x13 against a 30x19 patch. Drawing it is the
    difference between "the map looks right" and "the map looks right where it matters".

    Centred on the footprint rather than on the hero, and that is an approximation worth naming:
    the hero sits +0.09 tiles horizontally and +0.80 vertically from the camera centre, so this
    box is within a tile of the truth. Doing it properly needs the hero detector to be a module
    rather than a measurement script.
    """
    cols, rows = plan.size_tiles
    x0 = int(round((cols - view_w) / 2)) * scale
    y0 = int(round((rows - view_h) / 2)) * scale
    cv2.rectangle(canvas, (x0, y0), (x0 + view_w * scale - 1, y0 + view_h * scale - 1),
                  (255, 210, 40), 2)


def draw_footprint(canvas: np.ndarray, window: Window, plan: RectifyPlan,
                   grid_origin: tuple[int, int], position: tuple[float, float],
                   scale: int) -> None:
    """Outline the tiles the CURRENT frame covers, on the accumulated map. Mutates `canvas`.

    Side by side without this does not actually work, which only became obvious on looking at the
    output: the left panel is one frame and the right panel is the whole explored map, so a viewer
    has no way to tell which part of the map to compare against. Every apparent disagreement is
    then unfalsifiable -- you cannot tell "the map is wrong here" from "you are looking at a
    different part of the map".

    Deliberately NOT drawn in the `map` layout. That one is the artifact, and an annotation over it
    is exactly the diagnostic clutter Phase E already owns.
    """
    px, py = position
    ox, oy = plan.origin_tile
    gx, gy = grid_origin
    cols, rows = plan.size_tiles
    c0 = int(round(px + ox)) - gx - window.col0
    r0 = int(round(py + oy)) - gy - window.row0
    x0, y0 = c0 * scale, r0 * scale
    cv2.rectangle(canvas, (x0, y0), (x0 + cols * scale - 1, y0 + rows * scale - 1),
                  (255, 255, 255), 2)


def _held_origin(plan: RectifyPlan, position, cv_position):
    """Camera-relative tile origin for markers computed at `cv_position` but drawn on a panel
    cropped at `position`. Returns `plan.origin_tile` when the two coincide.

    **Needed because a detection's tile coordinates are relative to the camera that saw it.**
    `to_tiles` returns "tile (0, 0) is wherever the calibration's origin corner sits in THIS
    frame", so with `cv_fps` below `out_fps` a held detection is expressed against a camera that
    has since moved. Drawing it against the current crop makes every marker slide backwards as
    the camera pans, then snap when the detector next runs -- which looks exactly like a
    projection bug and is really just a stale frame of reference.

    A brawler's own world position is `cv_position + t`, so the tile it should occupy in the
    current frame is `t - (position - cv_position)`; shifting the ORIGIN by the camera's motion
    is the same thing with nothing per-marker to get wrong. The correction is sub-tile at walking
    speed (2.73 tiles/s over an eighth of a second is 0.34 tiles) and a couple of tiles during a
    Mortis dash, which is the case that makes it visible.
    """
    ox, oy = plan.origin_tile
    if cv_position is None:
        return (ox, oy)
    return (ox + position[0] - cv_position[0], oy + position[1] - cv_position[1])


@dataclass
class RenderReport:
    """What pass two did, for the caller to print and for the acceptance criteria to be read off."""
    out_frames: int = 0
    in_frames: int = 0
    classified: int = 0
    size: tuple[int, int] = (0, 0)
    render_seconds: float = 0.0
    clip_seconds: float = 0.0
    unknown_curve: list = field(default_factory=list)     # (t, unknown cell count) per output frame
    segments_seen: int = 1
    outside_window: int = 0        # cells deposited beyond what pass one predicted
    detections: int = 0            # boxes drawn, total, when a detector was passed
    detected_frames: int = 0       # frames the detector ran on -- the denominator for the above
    projectile_boxes: int = 0      # ditto for our projectile model, counted separately because
    projectile_frames: int = 0     # a boxes-per-frame rate mixing the two means nothing

    @property
    def realtime_factor(self) -> float:
        """Clip seconds rendered per wall second. >1 is faster than real time, <1 is slower.

        Named and oriented deliberately: "0.6x real time" is ambiguous about which way it runs, and
        the acceptance criterion is a direction, not a magnitude.
        """
        return self.clip_seconds / self.render_seconds if self.render_seconds else float("inf")

    @property
    def faster_than_realtime(self) -> bool:
        return self.render_seconds < self.clip_seconds

    def unknown_is_monotonic(self) -> bool:
        """Coverage must only grow. A rise means the world frame moved under the grid -- odometry
        lost lock -- and it is the acceptance criterion a per-frame drift number cannot see."""
        vals = [u for _, u in self.unknown_curve]
        return all(b <= a for a, b in zip(vals, vals[1:]))


def render(source, plan: RectifyPlan, track: Track, path, *, classifier=None, detector=None,
           health=None, projectiles=None, projectiles_on_map: bool = False,
           detect_on_map: bool = False, ground_offset_tiles: float = 0.0,
           anchor_frac: float | None = None, map_extent: str = "full",
           cfg: VisionConfig | None = None, layout: str = "map", out_fps: float = 12.0,
           cv_fps: float | None = None, scale: int | None = None, progress=None,
           view_w: int = 21, view_h: int = 13) -> RenderReport:
    """Pass two. Consume `source` again, render `layout` at `out_fps`, write to `path`.

    `track` is used only to size the canvas. Odometry is recomputed here rather than replayed: the
    two passes read different frames (pass one steps), so pass one has no answer for most of the
    frames this one sees, and interpolating between the ones it does have would invent positions
    that were never measured.

    **`detector` is an optional `object_detection.ObjectDetector`, and it is a passenger.** Its
    boxes are drawn onto the raw-frame panel of the `side-by-side` layout, and with
    `detect_on_map` its projected ground points are marked on the map panel too. Neither goes any
    further -- not into odometry, not into the occupancy grid, not into any threshold. That is the
    whole integration, deliberately: the two systems answer different questions from different
    pixel spaces, and the point of putting them in one frame is to *look at them together* before
    deciding whether they should ever talk. It runs only on rendered frames, the same rule the
    classifier follows and for the same reason -- nothing accumulates, so a skipped frame costs
    nothing.

    **`detect_on_map` crosses from screen space into tile space, and the anchor inside the box is
    what makes that hop right or wrong.** `anchor_frac` defaults to the config's measured 0.30;
    the raw box bottom (0.0) places a brawler barely better than chance. See `Detection.anchor`
    and `object_detection/project.py` before reading a tile index off a marker.

    In the `view` layout there is no raw panel, so a detector there means markers on BOTH the
    rectified frame and the map crop -- which is the layout where a marker in one panel is the
    same tile in the other, and therefore the one worth comparing in.

    **`health` is an `object_detection.hp_detection.HealthTracker`, and it is a passenger too.**
    It reads each brawler's HP out of the readout inside its own detection box and writes the
    number and a confidence beside that box on the raw panel. Like the detector it feeds nothing:
    not odometry, not the occupancy grid, not a threshold. It needs `detector`, because it only
    ever looks at pixels the detector pointed it at, and it needs a raw frame to annotate --
    so `side-by-side`, not `view`.

    Note the tracker is STATEFUL and its thresholds count rendered frames, so `out_fps` is what
    `hp.confirm_frames` and `hp.max_misses` are denominated in here. At the default 12 fps a
    change takes ~170 ms to confirm rather than the 250 ms it would at the 4 Hz decision rate.
    Pass a fresh tracker per render; `HealthTracker.reset()` is the alternative.

    **`projectiles` is a `projectile_detection.ProjectileDetector`, and it is a THIRD passenger.**
    Ours, not third-party -- the one model in this repo that gets trained here. Its boxes go on
    the raw panel in magenta beside the brawler boxes, and it feeds nothing either: not odometry,
    not the occupancy grid, not the entity detector. It is independent of `detector`, so
    `--projectiles` alone is a valid way to look at just this model, and the per-class tally in
    the corner counts whichever of the two are running.

    It runs on the `cv_fps` clock like everything else here, and **that is a real distortion for
    this particular model.** A brawler held for three frames is a brawler that has walked a few
    pixels. A projectile crosses the screen in well under a second, so at 4 Hz a held box can be
    most of a tile behind the shot, and a shot fired and landed between two CV frames is simply
    absent. Judge recall at `--cv-fps` equal to `--fps`; use a lower one to see what the agent
    would actually get at its decision rate. Those are two different questions and this knob is
    the difference between them.

    **`projectiles_on_map` projects them onto the map, and it is the weakest thing in this
    function.** `PROJECTILE_ANCHOR_FRAC` in `object_detection/project.py` spells out why: the
    homography inverts perspective for the ground plane, and a projectile is not on the ground, so
    every marker lands further from the camera than the shot really is by an unmeasured amount
    that grows with flight height. It is off by default. Turn it on to see roughly where the
    model is firing; do not read a tile index off it.

    **`map_extent` decides how much accumulated map the `side-by-side` right panel shows.**
    `full` (the default, and the old behaviour) is the whole explored map with the current
    frame's footprint outlined. `view` crops it to exactly the tiles this frame covers, so the
    two panels are the same extent tile-for-tile -- the same crop the `view` layout uses, but
    beside the RAW frame instead of the rectified one, which is what lets you compare boxes
    against terrain. `policy` crops further, to the 21x13 window the agent is actually handed.

    Cropping needs somewhere to crop FROM, so `view` and `policy` require a classifier; in mosaic
    mode there is no occupancy grid to take a window out of.

    **`cv_fps` decouples how often the expensive stages RUN from how often a frame is WRITTEN.**
    `out_fps` is the output video's rate; `cv_fps` (default: the same, which is the old behaviour)
    is how often the classifier, the detector and the health reader actually execute. Between
    those, their last results are held and redrawn over the current frame.

    That split exists because the two rates answer to different things. The output rate is a
    playback question -- 24 fps for something a person watches. The CV rate is the pipeline's real
    budget, 4-8 Hz against the 250 ms decision tick, and running a detector 24 times a second to
    produce a video is three times the work for no extra information. What you see is the video
    playing smoothly with boxes that update in steps, which is an honest picture of what the agent
    gets rather than an artificially fluid one.

    Odometry is exempt and still runs on every source frame, because it is a cumulative
    measurement: skipping frames changes the answer rather than just sampling it, which is exactly
    what `Odometry` is documented to require.

    Held detections are compensated for camera motion when they are drawn on a MAP panel -- see
    `_held_origin`. Without that the markers slide as the camera pans between CV frames.
    """
    cfg = cfg or VisionConfig()
    mode = "mosaic" if classifier is None else "map"
    if layout not in ("map", "side-by-side", "view"):
        raise ValueError(f"unknown layout {layout!r} -- use 'map', 'side-by-side' or 'view'")
    if layout == "view" and classifier is None:
        raise ValueError("the 'view' layout renders the classified map, so it needs a classifier")
    if detector is not None and layout not in ("side-by-side", "view"):
        # 'map' is the artifact -- the reconstruction alone, no annotation. Boxes need the raw
        # frame ('side-by-side') and markers need a panel in tile space ('view', or the map panel
        # beside the raw frame). Refusing beats silently dropping them.
        raise ValueError(
            f"a detector needs a layout with somewhere to draw: 'side-by-side' (boxes on the raw "
            f"frame, markers on the map with detect_on_map) or 'view' (markers on both panels). "
            f"Got {layout!r}"
        )
    if detector is not None and layout == "view" and not detect_on_map:
        # There is no raw panel in 'view', so boxes have nowhere to go and markers are the only
        # thing a detector can contribute. Silently rendering nothing would look like a broken
        # detector rather than a contradictory request.
        raise ValueError(
            "the 'view' layout has no raw frame, so a detector there can only draw projected "
            "markers -- pass detect_on_map=True (or use --layout side-by-side for boxes)"
        )
    if detect_on_map and classifier is None:
        raise ValueError("detect_on_map marks the classified map, so it needs a classifier")
    # The same three rules the entity detector gets, stated separately rather than folded in with
    # `detector or projectiles`, because the two are independent: --projectiles with no --detect
    # is a normal way to run this, and a message naming the wrong model sends you to fix the
    # wrong flag.
    if projectiles is not None and layout not in ("side-by-side", "view"):
        raise ValueError(
            f"a projectile detector needs a layout with somewhere to draw: 'side-by-side' (boxes "
            f"on the raw frame, markers on the map with projectiles_on_map) or 'view' (markers "
            f"only). Got {layout!r}"
        )
    if projectiles is not None and layout == "view" and not projectiles_on_map:
        raise ValueError(
            "the 'view' layout has no raw frame, so a projectile detector there can only draw "
            "projected markers -- pass projectiles_on_map=True (or use --layout side-by-side)"
        )
    if projectiles_on_map and classifier is None:
        raise ValueError(
            "projectiles_on_map marks the classified map, so it needs a classifier")
    if projectiles_on_map and projectiles is None:
        # Silently doing nothing would read as "the model found no projectiles in the whole clip",
        # which is a plausible-looking answer and the wrong one.
        raise ValueError(
            "projectiles_on_map has nothing to project without a projectile detector")
    if map_extent not in ("full", "view", "policy"):
        raise ValueError(
            f"unknown map_extent {map_extent!r} -- use 'full', 'view' or 'policy'")
    if map_extent != "full" and layout != "side-by-side":
        # 'map' is the artifact and 'view' is ALREADY the footprint crop, so cropping there is
        # either meaningless or a second way to spell what the layout already does. Refusing beats
        # accepting an argument that quietly does nothing.
        raise ValueError(
            f"map_extent {map_extent!r} only applies to --layout side-by-side ('view' is already "
            f"a footprint crop, and 'map' is the whole artifact). Got layout {layout!r}"
        )
    if map_extent != "full" and classifier is None:
        raise ValueError(
            f"map_extent {map_extent!r} crops the accumulated map, which mosaic mode does not "
            f"build -- it needs a classifier"
        )
    if health is not None and detector is None:
        # The reader only ever looks inside a detection box. Without one it has nothing to read
        # and would silently annotate nothing at all.
        raise ValueError("health reading needs a detector -- it reads HP inside detection boxes")
    if health is not None and layout != "side-by-side":
        raise ValueError(
            f"health reading annotates the RAW frame, which only 'side-by-side' shows. Got "
            f"layout {layout!r}"
        )
    cv_fps = out_fps if cv_fps is None else cv_fps
    if cv_fps <= 0:
        raise ValueError(f"cv_fps must be > 0, got {cv_fps}")
    if cv_fps > out_fps:
        # Results computed between two written frames are discarded, so this is pure waste --
        # and it reads as a performance setting, which makes it the kind of mistake nobody
        # notices from the output.
        raise ValueError(
            f"cv_fps ({cv_fps:g}) above out_fps ({out_fps:g}) computes results that are never "
            f"written. Lower cv_fps, or raise out_fps."
        )

    occupancy = OccupancyMap.from_config(cfg)
    window = world_window(track, plan, occupancy.height, occupancy.width)
    if window.rows < 1 or window.cols < 1:
        raise ValueError("the camera track covers no cells -- odometry never locked on")
    scale = scale or (_DEFAULT_SCALE if mode == "map" else _MOSAIC_SCALE)
    lut = tile_lut()
    mosaic = Mosaic(window, plan, occupancy.origin, scale) if mode == "mosaic" else None
    odo = Odometry(plan, cfg)

    report = RenderReport(clip_seconds=track.seconds)
    step = 1.0 / max(out_fps, 1e-6)
    cv_step = 1.0 / max(cv_fps, 1e-6)
    next_t = next_cv_t = None
    # Held between CV frames, redrawn over every written frame. `cv_position` is the camera the
    # held detections were measured against -- see `_held_origin`.
    map_base = None
    dets: list = []
    placed = None
    readings = None
    pdets: list = []
    pplaced = None
    cv_position = None
    t_start = time.perf_counter()
    sink = None
    try:
        for frame in source:
            report.in_frames += 1
            rect = plan.rectify(frame.image)
            r = odo.update(rect)
            if next_t is None:
                next_t = next_cv_t = frame.t
            due = frame.t >= next_t
            if mosaic is not None and r.tracking:
                mosaic.add(rect, r.position_tiles)   # every frame: the mosaic IS the odometry test
            if not due:
                continue
            next_t += step
            # `or map_base is None` so the first written frame always classifies. Both clocks
            # start together so it is already true, but a first frame rendered from an empty
            # occupancy grid would be a blank panel nobody could explain.
            cv_due = frame.t >= next_cv_t or (mosaic is None and map_base is None)
            if cv_due:
                next_cv_t += cv_step
            if mosaic is None:
                if cv_due:
                    if r.segment != occupancy.segment:
                        occupancy.reset(r.segment)
                        report.segments_seen = occupancy.segments_seen
                    zone = detect_zone(rect, plan, cfg).at_least(0.05)
                    cells, _ = classifier.predict(rect, plan)
                    occupancy.update(cells, r, plan, zone=zone, cfg=cfg)
                    report.classified += 1
                    map_base = render_map(occupancy, window, lut, scale)
                    best = occupancy.best()
                    # Sampled per CLASSIFICATION, not per written frame: this curve is the
                    # coverage-growth acceptance criterion, and padding it with held frames would
                    # only flatten it.
                    report.unknown_curve.append(
                        (frame.t, int((best[window.row0:window.row1,
                                            window.col0:window.col1] == UNKNOWN).sum())))
                    report.outside_window = _outside(occupancy, window)
                canvas = map_base.copy()
            else:
                canvas = mosaic.image()
            if cv_due and (detector is not None or projectiles is not None):
                # Once per rendered frame, before the layout branch: 'view' draws the result on
                # two panels and 'side-by-side' on up to two, and running the model per panel
                # would double the cost to produce identical boxes.
                from ..object_detection.project import PROJECTILE_ANCHOR_FRAC, to_tiles

                # ONE camera reference for both models, set before either runs. They see the same
                # frame on the same tick, so a held box from either is stale against the same
                # amount of camera motion and `_held_origin` corrects both with one number.
                cv_position = r.position_tiles
                if detector is not None:
                    dets = detector.predict(frame.image)
                    report.detections += len(dets)
                    report.detected_frames += 1
                    placed = to_tiles(
                        dets, plan, ground_offset_tiles,
                        anchor_frac=(cfg.detector_anchor_frac if anchor_frac is None
                                     else anchor_frac),
                    ) if detect_on_map else []
                    if health is not None:
                        # After `predict`, on the SAME boxes -- the reader is defined on
                        # detections, so running it on anything else would annotate boxes nobody
                        # drew.
                        readings = health.update(frame.image, dets)
                if projectiles is not None:
                    pdets = projectiles.predict(frame.image)
                    report.projectile_boxes += len(pdets)
                    report.projectile_frames += 1
                    # `PROJECTILE_ANCHOR_FRAC`, not the brawler anchor, and no ground offset: the
                    # offset is a correction fitted to where a brawler's feet sit inside its box,
                    # which is not a quantity a projectile has. See that constant for why this
                    # projection is approximate no matter what is passed here.
                    pplaced = to_tiles(pdets, plan,
                                       anchor_frac=PROJECTILE_ANCHOR_FRAC) if projectiles_on_map                         else []
            if layout == "view":
                # Both panels are the same 30x19 tiles at the same scale, so a row in one is the
                # same row in the other. The whole point of this layout is that you can put a
                # finger on a tile and read off what the classifier called it.
                vscale = scale * 2
                cols, rows = plan.size_tiles
                left = cv2.resize(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB),
                                  (cols * vscale, rows * vscale), interpolation=cv2.INTER_AREA)
                right = crop_to_view(occupancy, plan, r.position_tiles, lut, vscale)
                for panel in (left, right):
                    draw_policy_window(panel, plan, vscale, view_w, view_h)
                if placed:
                    # DIFFERENT annotations on the two panels, deliberately, because the panels
                    # are different kinds of thing. Left is a picture of the screen, so it gets
                    # the detector's real box, warped by the same homography that warped the
                    # pixels under it -- what the model actually found, around the character.
                    # Right is a tile grid, where a box enclosing a nameplate would be nonsense,
                    # so it gets the footprint the projection resolves to.
                    #
                    # Both still use one origin and one scale, so a footprint on the right sits at
                    # the same spot as the base of its box on the left. That equivalence is the
                    # entire reason to compare in this layout.
                    from ..object_detection.project import draw_boxes, draw_markers, to_tile_quads
                    # The LEFT panel is a picture of the current screen and the boxes are in
                    # screen space, so a held box belongs at its own screen position -- lagging
                    # the sprite, which is the truth about a held detection. No correction.
                    draw_boxes(left, to_tile_quads([d for d, _ in placed], plan),
                               plan.origin_tile, vscale)
                    # The RIGHT panel is world space and was cropped at the CURRENT camera, so a
                    # held marker does need the camera's motion taken out of it.
                    draw_markers(right, placed,
                                 _held_origin(plan, r.position_tiles, cv_position), vscale)
                if pplaced:
                    # The same two treatments, for the same reason: the real box warped onto the
                    # rectified picture, the footprint onto the tile grid. `label=False` because
                    # one class means the text can only say "Projectile" -- the colour already
                    # does, and there can be six of them in the space two brawlers occupy.
                    from ..object_detection.project import (PROJECTILE_FOOTPRINT_TILES, draw_boxes,
                                                            draw_markers, to_tile_quads)
                    draw_boxes(left, to_tile_quads([d for d, _ in pplaced], plan),
                               plan.origin_tile, vscale, label=False)
                    draw_markers(right, pplaced,
                                 _held_origin(plan, r.position_tiles, cv_position), vscale,
                                 footprint_tiles=PROJECTILE_FOOTPRINT_TILES)
                canvas = np.hstack([left, np.zeros((left.shape[0], 6, 3), np.uint8), right])
            elif layout == "side-by-side":
                # `mscale`, not `scale`, once the panel is cropped. A 21x13 policy window at 14
                # px/tile is a 294 px-wide panel, and `_side_by_side` matches the RAW frame to the
                # canvas height -- so a small right panel silently shrinks the left one to match
                # and the boxes this layout exists to show become unreadable. The `view` layout
                # doubles its scale for the same reason.
                mscale = scale if map_extent == "full" else scale * 2
                if map_extent == "full":
                    if r.tracking:
                        draw_footprint(canvas, window, plan, occupancy.origin,
                                       r.position_tiles, scale)
                    gx, gy = occupancy.origin
                    # `cv_position`, not the current one: these markers are camera-relative to
                    # the frame the detector actually saw, so converting them to world tiles has
                    # to use that camera. The footprint outline above is a different thing and
                    # correctly tracks the current frame.
                    mpos = cv_position or r.position_tiles
                    marker_origin = (gx + window.col0 - mpos[0], gy + window.row0 - mpos[1])
                else:
                    # Same crop the 'view' layout takes, so a marker here means the same tile it
                    # would there -- and `plan.origin_tile` is the matching origin, not the
                    # window arithmetic the full map needs.
                    canvas = crop_to_view(occupancy, plan, r.position_tiles, lut, mscale)
                    marker_origin = _held_origin(plan, r.position_tiles, cv_position)
                    if map_extent == "policy":
                        cols, rows = plan.size_tiles
                        dx = int(round((cols - view_w) / 2))
                        dy = int(round((rows - view_h) / 2))
                        canvas = canvas[dy * mscale:(dy + view_h) * mscale,
                                        dx * mscale:(dx + view_w) * mscale].copy()
                        # The crop moves the panel's top-left tile, so the marker origin moves
                        # with it. Getting this wrong offsets every marker by ~4 tiles, which
                        # looks like a projection bug rather than an arithmetic one.
                        marker_origin = (marker_origin[0] + dx, marker_origin[1] + dy)
                    else:
                        draw_policy_window(canvas, plan, mscale, view_w, view_h)
                source_frame = frame.image
                if projectiles is not None:
                    # Imported here, not at module scope. `object_detection` is a sibling chunk
                    # this one does not otherwise depend on, and a top-level import would make
                    # every terrain render pull it in to satisfy a branch that usually does not
                    # run. Same reasoning `sources.open_source` gives for importing ScreenCapture
                    # lazily.
                    from ..object_detection.projectile_detection.draw import draw_projectiles

                    # FIRST, so brawler boxes land on top. A projectile is usually leaving or
                    # arriving at a brawler and the two boxes overlap constantly; the brawler box
                    # is the bigger, more stable thing and reads better as the upper layer.
                    #
                    # draw_projectiles copies. It must: `frame.image` is the same array odometry
                    # and the classifier read from this iteration, and painting boxes into it
                    # would feed box outlines to the terrain pipeline.
                    source_frame = draw_projectiles(source_frame, pdets)
                if detector is not None:
                    from ..object_detection.draw import draw_detections

                    source_frame = draw_detections(source_frame, dets)
                if detector is not None or projectiles is not None:
                    from ..object_detection.draw import draw_summary

                    # ONE tally over both models. `draw_summary` counts by label and colours each
                    # line with the same `color_for` the boxes used, so a combined list reads as
                    # "enemy 2 / player 1 / Projectile 3" and cannot disagree with the picture.
                    # In place, on the copy the draws above already returned.
                    source_frame = draw_summary(source_frame, [*dets, *pdets])
                if readings:
                    # After the boxes, so the HP text sits on top of the box outline rather than
                    # under it. `draw_health` copies as well, for the reason above.
                    from ..object_detection.hp_detection.draw import draw_health
                    source_frame = draw_health(source_frame, readings)
                if placed and r.tracking:
                    # Camera-relative tiles -> whichever world canvas `map_extent` selected. Left
                    # unrounded so a marker keeps the sub-tile position the projection produced.
                    # Only while tracking: after a lost frame the position is fiction, and a
                    # marker placed from it would be a confident lie.
                    from ..object_detection.project import draw_markers
                    draw_markers(canvas, placed, marker_origin, mscale)
                if pplaced and r.tracking:
                    # Same origin and scale as the brawler markers above, so the two are in one
                    # coordinate system and a projectile drawn near a brawler really is near it
                    # -- to the accuracy the ground-plane projection allows, which for a flying
                    # object is not much. See PROJECTILE_ANCHOR_FRAC.
                    from ..object_detection.project import (PROJECTILE_FOOTPRINT_TILES,
                                                            draw_markers)
                    draw_markers(canvas, pplaced, marker_origin, mscale,
                                 footprint_tiles=PROJECTILE_FOOTPRINT_TILES)
                canvas = _side_by_side(source_frame, canvas)
            _label(canvas, f"{mode}  t={frame.t:6.2f}s  frame {frame.index}")
            if sink is None:
                report.size = (canvas.shape[1], canvas.shape[0])
                sink = VideoSink(path, report.size, fps=out_fps).__enter__()
            sink.write(canvas)
            report.out_frames += 1
            if progress is not None and report.out_frames % 50 == 0:
                progress(report.in_frames, frame.t)
    finally:
        if sink is not None:
            sink.__exit__(None, None, None)
    report.render_seconds = time.perf_counter() - t_start
    if sink is None:
        raise ValueError("no frames were rendered -- the source yielded nothing")
    return report


def _outside(occupancy: OccupancyMap, window: Window) -> int:
    """Observed cells that fell outside the canvas pass one sized.

    Pass one samples the track, so its window is a prediction. Anything past it is invisible in the
    output and would otherwise be indistinguishable from terrain the camera never saw -- which is
    the one thing this whole render is supposed to make legible.
    """
    seen = occupancy.observed
    inside = np.zeros_like(seen)
    inside[window.row0:window.row1, window.col0:window.col1] = True
    return int((seen & ~inside).sum())
