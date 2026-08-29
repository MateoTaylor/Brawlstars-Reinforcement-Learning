"""Live screen capture, plus the frame-geometry primitives every frame source shares.
See Terrain_Perception_Build_Plan.md Phase A and Section 3.

**Color order is BGR everywhere in `brawl_vision`, and it is not negotiable.** Every OpenCV
call in Phases C/D/F/G takes BGR, `mss` hands back BGRA (so BGR is a free slice), and the one
source that natively produces RGB (`cv2.VideoCapture` does not; `imageio` would) is not used.
Getting this wrong is invisible in a grayscale stage and silently wrong in a color one.

**The size of that error depends on the color, which is worth knowing before writing a test for
it.** Swapping R and B reflects hue about the green axis, so it barely moves a GREEN-DOMINANT
color: Phase G's gas sits at hue ~55 and lands on ~65 swapped, still inside any sane gas band.
Measured on a real frame, the gas mask is 6.10% of the viewport read correctly and 6.11% read
backwards -- gas is nearly useless as a channel-order canary. Colors where green is not the
maximum move hard: this repo's purple map terrain goes from hue ~119 to ~4. So the guard in
tests/test_vision_clips.py is built on whole-frame blue-vs-red dominance, which cleanly inverts,
rather than on anything green.

**Two geometry primitives live here rather than in `camera.py`, deliberately.** `detect_content_box`
and `normalize_viewport` are properties of a FRAME, not of a calibration, and both frame sources
(this module and `clips.py`) need them before any camera model exists. Phase B's `camera.py` will
import them rather than re-implement them; it is the layer above.

**What normalization is for** (the full argument is Section 3 of the plan): Solo Showdown
deliberately constricts the visible width relative to other modes, at the same zoom, to keep the
match fair. Measured on real footage: the training cave renders 2436 px wide, Solo Showdown 2002 px
(exactly 16:9), both at 1126 px tall and both at ~77 px per tile on screen. So every source is
cropped to the NARROWEST framing before anything downstream touches it, which is what lets a
single homography serve every mode.
"""
import time
from dataclasses import dataclass

import numpy as np

from .config import VisionConfig

# The Solo Showdown viewport's aspect: exactly 16:9. Expressed as an ASPECT rather than a pixel
# width so it survives a resolution change -- a calibration is tied to a resolution, but "how much
# of the world Showdown lets you see" is not.
#
# This started out as a measured 2006/1126 = 1.7815, "suspiciously close to 16:9 without being
# equal to it". The discrepancy was the measurement, not the game: a max-based content detector
# was latching onto ~11 px of HUD glow outside the world viewport. With the fraction-based
# detector in `detect_content_box`, the recovered viewport on real footage is x=[217, 2218] --
# 2002x1126, exactly 16:9, exactly centered in the 2436 px frame.
SHOWDOWN_ASPECT = 16 / 9


@dataclass(frozen=True)
class Frame:
    """One frame from any source. `image` is (H, W, 3) uint8 BGR.

    `t` is SECONDS and monotonic within a single source, but its zero point is source-specific:
    wall time since capture started for `ScreenCapture`, position within the file for
    `clips.ClipReader`. Nothing downstream should compare `t` across sources, and nothing should
    assume a constant interval between consecutive frames -- the recorded clips are screen
    captures with dropped frames, carrying real gaps of up to 166 ms (10 frames at 60 Hz) in the
    middle of an otherwise 16.7 ms stream. Phase F converts pixel shift to tile velocity and will
    need the true dt, which is exactly why this is a measured field and not `index / fps`.
    """
    image: np.ndarray
    t: float
    index: int

    @property
    def size(self) -> tuple[int, int]:
        """(w, h), matching OpenCV's argument order rather than numpy's shape order."""
        return self.image.shape[1], self.image.shape[0]


def detect_content_box(images, black_level: int = 25,
                       min_fraction: float = 0.5) -> tuple[int, int, int, int]:
    """Bounding box of actual rendered content within a frame, as INCLUSIVE `(x0, x1, y0, y1)`.

    Strips the letterbox/pillarbox bars a recording carries when the game does not fill the
    capture rectangle. Solo Showdown pillarboxes to 16:9 inside a 2.16:1 phone screen; the
    training cave does not.

    **A column counts as content when `min_fraction` of its pixels exceed `black_level` -- not
    when its BRIGHTEST pixel does, which is the obvious implementation and is wrong here.** The
    boundary is not a step from black to content; there is a ~10 px band of dim glow outside the
    world viewport, from HUD elements laid out to the full screen. Measured across that band on
    `zone_grows_from_east.mp4`: columns 206-216 have a column-mean of 6-10 and only 2-23% of their
    pixels lit, then column 218 jumps to a mean of 122 with 99.9% lit. A max-based test latches
    onto the glow and reports the viewport ~11 px too wide and apparently off-center; the fraction
    test lands on the real edge, and the recovered viewport then comes out at exactly 16:9 and
    exactly frame-centered -- which is what a pillarbox should be, and is the check that this
    detector is measuring the right thing.

    Takes an ITERABLE of images and unions their boxes: a single dark frame (a night map, a death
    fade, the match-start transition) can have genuine content below threshold at its edges and
    would report a box tighter than the truth, and cropping to a too-small box silently discards
    real world pixels. For a whole clip, prefer the median across many frames -- see
    `clips.detect_usable_range`.
    """
    x0 = y0 = 1 << 30
    x1 = y1 = -1
    seen = False
    for img in images:
        seen = True
        lit = img.max(axis=2) > black_level

        # Pass 1: rough bounds from "any lit pixel at all". Needed because the fraction test in
        # pass 2 has to be relative to the content's extent on the OTHER axis, not to the whole
        # frame -- a region small in both axes lights only a fraction of each full column and
        # would otherwise be rejected as glow.
        any_col = np.nonzero(lit.any(axis=0))[0]
        any_row = np.nonzero(lit.any(axis=1))[0]
        if not len(any_col) or not len(any_row):
            continue  # a frame with no content contributes nothing rather than collapsing the box
        ry0, ry1 = int(any_row[0]), int(any_row[-1])
        rx0, rx1 = int(any_col[0]), int(any_col[-1])

        # Pass 2: within that span, keep only columns/rows that are MOSTLY lit.
        cols = np.nonzero(lit[ry0:ry1 + 1, :].mean(axis=0) >= min_fraction)[0]
        rows = np.nonzero(lit[:, rx0:rx1 + 1].mean(axis=1) >= min_fraction)[0]
        if not len(cols) or not len(rows):
            continue
        x0, x1 = min(x0, int(cols[0])), max(x1, int(cols[-1]))
        y0, y1 = min(y0, int(rows[0])), max(y1, int(rows[-1]))
    if not seen:
        raise ValueError("detect_content_box got no images")
    if x1 < 0:
        raise ValueError("every sampled frame was entirely below black_level; nothing to crop to")
    return x0, x1, y0, y1


# How many pixels the detected viewport may fall SHORT of the exact target width before that
# counts as a real framing mismatch rather than rounding at the content boundary. Well separated
# from anything meaningful: boundary rounding costs 1-3 px, while genuinely different framing
# (the training cave against Solo Showdown) differs by 434.
_WIDTH_SLACK_PX = 4


def normalize_viewport(image: np.ndarray, box: tuple[int, int, int, int],
                       aspect: float = SHOWDOWN_ASPECT) -> np.ndarray:
    """Crop `image` to `box`, then symmetrically trim its width to `aspect`. Returns a VIEW, not
    a copy -- callers that keep frames must copy, and `ScreenCapture`/`ClipReader` already do.

    The symmetric trim is what makes a training-cave frame geometrically identical to a Showdown
    one: at 1126 px tall the target is 2002 px wide, so both a 2436 px-wide cave frame and a
    pillarboxed Showdown frame land on the same `x = [217, 2218]`. One operation, both sources,
    same output geometry.

    **The crop is centered on `box`, NOT on the frame, and that is a measured correction rather
    than an obvious choice.** The intuitive assumption -- a pillarbox is centered by construction,
    so use frame center -- is false on this hardware. Measured on `zone_grows_from_east.mp4` at
    frame 200, the left bar runs to x=210 and the right bar starts at x=2218, so the game's
    viewport sits ~6 px LEFT of the capture rectangle's center: an iPhone safe-area inset around
    the notch, which is not symmetric in landscape. Centering on the frame put ~3 px of black bar
    inside the normalized right edge and clipped ~5 px of real world off the left. Centering on
    the detected viewport aligns the WORLD across sources, which is the thing that actually has to
    match.

    Raises rather than upscaling when the content is MEANINGFULLY narrower than the target
    (`_WIDTH_SLACK_PX`). Stretching would invent world that the game did not render and hand
    Phase C a homography fit against fabricated pixels; there is no correct recovery, only a
    louder or quieter failure. A shortfall of a pixel or two is not that -- it is rounding at the
    anti-aliased content boundary, and the pixels just outside the detected box are real world,
    not bar.
    """
    x0, x1, y0, y1 = box
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    target_w = int(round(h * aspect))
    if w < target_w - _WIDTH_SLACK_PX:
        raise ValueError(
            f"content box is {w}x{h}, narrower than the {target_w}x{h} normalized viewport "
            f"(aspect {aspect:.4f}) by more than {_WIDTH_SLACK_PX} px. Refusing to upscale -- "
            f"see this function's docstring."
        )
    nx0 = int(round((x0 + x1 + 1) / 2.0 - target_w / 2.0))
    nx0 = max(0, min(nx0, image.shape[1] - target_w))   # stay in the frame on odd geometries
    return image[y0:y1 + 1, nx0:nx0 + target_w]


class ScreenCapture:
    """Live frames off the screen via `mss`, as `Frame`s.

    `mss` rather than the Windows DXGI Desktop Duplication API, per the plan: pure Python,
    trivial setup, and almost certainly fast enough to get every other phase working. DXGI is
    lower latency and more setup; do not reach for it before Phase I exists and the full pipeline
    has been timed against the 250 ms decision budget.

    Use as a context manager -- `mss` holds an OS handle, and a capture loop that leaks one per
    restart is exactly the silent failure Phase A's acceptance criterion is checking for.
    """

    def __init__(self, cfg: VisionConfig | None = None, monitor: int | None = None,
                 normalize: bool = True, box_sample: int = 5):
        self.cfg = cfg or VisionConfig()
        self.monitor = self.cfg.capture_monitor if monitor is None else monitor
        self.normalize = normalize
        self._box_sample = box_sample
        self._box: tuple[int, int, int, int] | None = None
        self._sct = None
        self._index = 0
        self._t0: float | None = None
        self.grab_seconds = 0.0   # cumulative, for the plan's "report your own timing from day one"

    def __enter__(self):
        import mss  # imported lazily so `import brawl_vision` works without the [vision] extra
        # mss 10 renamed the factory to MSS and deprecated the lowercase alias; fall back so this
        # keeps working against whichever version the [vision] extra happens to resolve.
        self._sct = getattr(mss, "MSS", None) or mss.mss
        self._sct = self._sct()
        return self

    def __exit__(self, *exc):
        if self._sct is not None:
            self._sct.close()
            self._sct = None
        return False

    @property
    def n_frames(self) -> None:
        """Always `None`: live capture is unbounded. See `sources.FrameSource` -- returning a
        number here would make a bounded loop over the screen look correct."""
        return None

    @property
    def read_seconds(self) -> float:
        """`sources.FrameSource`'s name for `grab_seconds`."""
        return self.grab_seconds

    def _raw(self) -> np.ndarray:
        if self._sct is None:
            raise RuntimeError("ScreenCapture must be used as a context manager (`with ...`)")
        shot = self._sct.grab(self._sct.monitors[self.monitor])
        # mss gives BGRA in a buffer it reuses between grabs; the [..., :3] slice drops alpha and
        # np.array copies, so a returned Frame never aliases the next grab's contents.
        return np.array(shot, dtype=np.uint8)[..., :3]

    def grab(self) -> Frame:
        t_start = time.perf_counter()
        img = self._raw()
        if self._t0 is None:
            self._t0 = t_start
        if self.normalize:
            if self._box is None:
                # Sample a few frames rather than trusting one -- see detect_content_box.
                self._box = detect_content_box(
                    [img] + [self._raw() for _ in range(self._box_sample - 1)],
                    self.cfg.capture_black_level,
                )
            img = np.ascontiguousarray(
                normalize_viewport(img, self._box, self.cfg.capture_normalized_aspect)
            )
        frame = Frame(image=img, t=t_start - self._t0, index=self._index)
        self._index += 1
        self.grab_seconds += time.perf_counter() - t_start
        return frame

    def __iter__(self):
        while True:
            yield self.grab()
