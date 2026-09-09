"""Screen -> the viewport the vision stack was calibrated for. See BRAWL_DEPLOYMENT_DESIGN.md 9.3.

**The whole module exists because of one number: 2002x1126.** Every calibrated constant in
`brawl_vision` was fit at that normalized viewport -- `data/homography.json`'s `viewport`,
`hud_mask.json`'s, `gameplay.RADIUS_PX`, the apparent size of a brawler in the detector's training
frames. The monitor here is 2560x1440. So there is a resize, and the only question is what to
resize *to*.

**Resize to the calibration's viewport, not to the emulator's 1920x1080.** Both are downscales
from a 1440p grab, so neither invents pixels, but they are not equally cheap:

    target        what it costs
    1920x1080     a rescaled CameraModel carried forever, and every brawler 4% smaller than
                  the frames the detector was trained on
    2002x1126     nothing -- the shipped homography, HUD mask and thresholds apply unchanged

MEASURED, and this is the result that made the choice safe rather than merely tidy: the iOS
calibration transfers to BlueStacks **unmodified**. Rectifying a BlueStacks frame with the shipped
homography puts wall blocks exactly on the 48 px tile grid across the entire frame -- correct
perspective *and* correct absolute scale, which is not obvious, since it means the game's virtual
camera does not depend on the device. Odometry over 60 frames of that footage: inlier ratio 1.000
median and 1st percentile, response 0.92, 59/59 frames `ok`.

**Do not "simplify" this to a fixed 0.75 scale factor.** The resize target is a viewport, not a
ratio, and that difference is load-bearing: `detect_content_box` trims whatever bars the capture
actually has (measured: it shaved a row off a fullscreen grab, giving 2558x1439 rather than
2560x1440), and a windowed emulator would give something else again. Resizing the *normalized*
frame to a fixed viewport absorbs all of that; multiplying by a constant does not.

**Two coordinate spaces meet in this package and they are not the same space.** This module
produces VIEWPORT pixels, 2002x1126, for perception. Touch injection addresses DEVICE pixels, the
Android screen's own 1920x1080, which `adb` scales to its 0..32767 axis range and which has
nothing to do with the monitor. `match_state.Calibration` holds the device numbers and converts;
see its `viewport_button`. They happened to coincide while the fixture was 1080p, which is exactly
the sort of coincidence that survives every test and then breaks in the field.
"""
import time

import cv2
import numpy as np

from brawl_vision.camera import load_camera_model
from brawl_vision.capture import Frame, ScreenCapture

# Above this, resizing would be an UPSCALE, and the same objection `normalize_viewport` raises
# applies: the content is genuinely lower-resolution than the pipeline expects and stretching it
# hands every downstream stage invented pixels. 1.02 is slack for boundary rounding, not for a
# real shortfall -- a fullscreen 1440p grab lands at 0.78, and even a 1080p one lands at 1.04,
# which is why this is a warning threshold rather than the 1.0 it looks like it should be.
MAX_UPSCALE = 1.10

# How much of the raw grab's HEIGHT the detected content box must span. Below this, something dark
# is touching the top or bottom edge of the screen and rows of real game have been cropped away.
#
# **This exists because vertical shrinkage is the one occlusion that fails SILENTLY.** MEASURED, by
# pasting sub-`black_level` rectangles into a real 2560x1440 grab and re-running the detector:
#
#     occlusion                     content box    normalize_viewport
#     anywhere in the interior      unchanged      fine -- a bounding box cannot be dented
#     touching the left/right edge  narrower       RAISES ("refusing to upscale")
#     touching the top/bottom edge  shorter        SUCCEEDS, at the wrong scale
#
# The width test in `normalize_viewport` is *relative* -- it compares width against `height *
# aspect` -- so a 120 px dark strip along the top gives a 2560x1319 box, the target width drops to
# 2345 to match, the aspect test passes, and what comes out is a valid-looking 16:9 frame that is
# really a zoomed crop. `DeployCapture` then resizes it to 2002x1126 and no stage downstream can
# tell: every projected tile coordinate is off by the height ratio. Hence a check here, against the
# grab, since `normalize_viewport` sees only the box and has no idea what it was cut from.
#
# 0.99 is far outside observed rounding (a fullscreen 1440p grab spans 1439/1440 = 0.9993) and far
# inside the smallest occlusion that matters (a 15 px strip is already a 1% scale error).
#
# Width needs no equivalent: it is already guarded, loudly, by the aspect test above.
MIN_HEIGHT_COVERAGE = 0.99


def calibrated_viewport() -> tuple[int, int]:
    """`(w, h)` the vision stack was fit at, read from the shipped camera model rather than
    written down here. A constant would be a second source of truth for a number that already has
    one, and the failure if they drift apart is `RectifyPlan.rectify` raising on every frame."""
    return tuple(load_camera_model().viewport)


class DeployCapture:
    """`ScreenCapture` plus the resize to the calibrated viewport. Context manager, like the thing
    it wraps -- `mss` holds an OS handle.

    Wraps rather than reimplements: `ScreenCapture` already handles the `mss` version split, the
    content-box sampling and caching, the BGRA slice and the copy that stops a returned frame
    aliasing the next grab. All that is left is one `cv2.resize`.

    **Pass `window` whenever there is one.** `from_window` is the deployed constructor and it
    takes the crop rectangle from the window manager instead of inferring it from pixels. The
    inference is not merely redundant here, it is wrong: see `window.client_box_in_monitor` for
    the measurement (the Nulls Brawl lobby's own artwork has a black 21 px band at the top, so a
    loop started at the lobby caches a box 21 rows short and crops every gameplay frame with it).
    Detection stays the default only for sources that have no window to ask.
    """

    def __init__(self, viewport: tuple[int, int] | None = None, monitor: int | None = None,
                 interpolation: int = cv2.INTER_AREA,
                 box: tuple[int, int, int, int] | None = None):
        self.viewport = tuple(viewport) if viewport else calibrated_viewport()
        # INTER_AREA, not INTER_LINEAR. This is minification by ~1.28x, where AREA is a genuine
        # box filter and LINEAR samples 2x2 and aliases. Measured: 2.6 ms vs 0.8 ms at 1440p, and
        # 1.8 ms of anti-aliasing is worth having in a 50 ms per-frame budget when the things
        # being detected are small bright sprites.
        self.interpolation = interpolation
        self._cap = ScreenCapture(monitor=monitor, normalize=True, box=box)
        self.resize_seconds = 0.0
        self.source_size: tuple[int, int] | None = None   # (w, h) BEFORE the resize

    @classmethod
    def from_window(cls, window, monitors: list[dict], **kw) -> "DeployCapture":
        """The deployed constructor: capture the monitor the emulator is on, cropped to its
        client area. `window` is a `brawl_deployment.window.WindowInfo`, `monitors` is
        `mss.monitors`.

        Both facts come from the same place, which is the point -- resolving the monitor index and
        the crop box from one authoritative source removes the possibility of the two disagreeing.
        """
        from .window import client_box_in_monitor, monitor_index_for
        index = monitor_index_for(window.client, monitors)
        return cls(monitor=index, box=client_box_in_monitor(window, monitors[index]), **kw)

    def __enter__(self):
        self._cap.__enter__()
        return self

    def __exit__(self, *exc):
        return self._cap.__exit__(*exc)

    @property
    def grab_seconds(self) -> float:
        return self._cap.grab_seconds

    def grab(self) -> Frame:
        """One frame at the calibrated viewport. `t` and `index` pass through from the source."""
        f = self._cap.grab()
        h, w = f.image.shape[:2]
        if self.source_size is None:
            self.source_size = (w, h)
            self._check_content_box()
            self._check_scale(w, h)
        if (w, h) == self.viewport:
            return f
        t0 = time.perf_counter()
        img = cv2.resize(f.image, self.viewport, interpolation=self.interpolation)
        self.resize_seconds += time.perf_counter() - t0
        return Frame(image=img, t=f.t, index=f.index)

    def _check_scale(self, w: int, h: int) -> None:
        up = max(self.viewport[0] / w, self.viewport[1] / h)
        if up > MAX_UPSCALE:
            raise ValueError(
                f"the normalized capture is {w}x{h}, which would have to be upscaled {up:.2f}x to "
                f"reach the calibrated {self.viewport[0]}x{self.viewport[1]} viewport. Refusing: "
                f"the game is rendering below the resolution the vision pipeline was fit at, and "
                f"stretching would hand every downstream stage invented detail. Raise the "
                f"emulator's or the monitor's resolution."
            )

    def _check_content_box(self) -> None:
        """Fail closed when a *detected* content box has lost rows off the top or bottom.

        Runs once, on the first frame, because the box is computed once and cached -- so the
        exposure is the first few frames of a session and nothing after.

        **Skipped entirely when the box was supplied** (`from_window`), because there is then
        nothing to be uncertain about: the crop came from the window manager, not from pixels, and
        a client area that does not fill its monitor is `WindowGuard`'s business and is caught
        earlier with a better message. This check exists for the inference path.

        MEASURED, and it is why the inference path needs a guard at all: at the Nulls Brawl lobby
        `detect_content_box` returns 2560x1418 of a 2560x1440 grab -- 98.5% -- because the lobby's
        own artwork has a black band across the top. Nothing is covering the screen; the app is
        simply not drawing there. A loop started at the lobby would cache that box and crop 21
        real rows off every gameplay frame for the rest of the match.
        """
        if self._cap.box_supplied:
            return
        box, raw = self._cap.content_box, self._cap.raw_size
        if box is None or raw is None:
            return   # not normalizing, or nothing grabbed yet -- nothing to check against
        box_h = box[3] - box[2] + 1
        coverage = box_h / raw[1]
        if coverage < MIN_HEIGHT_COVERAGE:
            raise ValueError(
                f"the detected content box is {box_h} px tall out of a {raw[1]} px grab "
                f"({coverage:.1%}), so rows of the screen read as black. Either something dark is "
                f"touching the top or bottom edge of the capture (a window, a taskbar, a "
                f"non-fullscreen emulator), or the app is drawing a dark band there -- the Nulls "
                f"Brawl lobby does exactly that, costing 21 rows. Cropping to this would rescale "
                f"every projected tile coordinate by {coverage:.3f} and nothing downstream would "
                f"notice. Prefer DeployCapture.from_window, which does not infer the box at all."
            )

    def __iter__(self):
        while True:
            yield self.grab()


def to_viewport(image: np.ndarray, viewport: tuple[int, int] | None = None,
                interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize one already-normalized frame to the calibrated viewport.

    For offline sources -- a fixture clip read through `brawl_vision.clips` -- so the recorded and
    the live paths land on identical geometry and a test on the fixture means something about the
    deployed loop.
    """
    viewport = tuple(viewport) if viewport else calibrated_viewport()
    if image.shape[1::-1] == viewport:
        return image
    return cv2.resize(image, viewport, interpolation=interpolation)
