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
    """

    def __init__(self, viewport: tuple[int, int] | None = None, monitor: int | None = None,
                 interpolation: int = cv2.INTER_AREA):
        self.viewport = tuple(viewport) if viewport else calibrated_viewport()
        # INTER_AREA, not INTER_LINEAR. This is minification by ~1.28x, where AREA is a genuine
        # box filter and LINEAR samples 2x2 and aliases. Measured: 2.6 ms vs 0.8 ms at 1440p, and
        # 1.8 ms of anti-aliasing is worth having in a 50 ms per-frame budget when the things
        # being detected are small bright sprites.
        self.interpolation = interpolation
        self._cap = ScreenCapture(monitor=monitor, normalize=True)
        self.resize_seconds = 0.0
        self.source_size: tuple[int, int] | None = None   # (w, h) BEFORE the resize

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
