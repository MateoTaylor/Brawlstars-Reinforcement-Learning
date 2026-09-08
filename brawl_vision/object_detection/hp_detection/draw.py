"""Drawing HP readings onto a frame. Rendering only -- nothing here computes anything.

The same split `object_detection/draw.py` and `terrain/overlay.py` both state for themselves, and
for the same reason: a drawing bug must never be able to change a measurement.

**Confidence is drawn as well as the number, because the number alone is the misleading half.**
This chunk's whole design is about telling a good read from a confident-looking bad one, and an
overlay that renders `8` and `8400` identically hides exactly what you opened it to look at. The
value is tinted by confidence and a fatal read prints its status instead of a number.
"""
import cv2
import numpy as np

from ..draw import _FONT, _sizes, color_for
from .read import OK

# Confidence -> BGR, low to high. Red for a reading you should not use, amber for a marginal one,
# white for a good one. Not the class colours: this axis is orthogonal to player-vs-enemy, and
# reusing those would make a low-confidence player box look like an enemy.
_LOW = (60, 60, 235)
_MID = (60, 200, 245)
_HIGH = (255, 255, 255)


def confidence_color(confidence: float) -> tuple[int, int, int]:
    """Piecewise-linear over the two stops, so the eye reads a gradient rather than three bins."""
    c = float(np.clip(confidence, 0.0, 1.0))
    a, b, t = ((_LOW, _MID, c / 0.5) if c < 0.5 else (_MID, _HIGH, (c - 0.5) / 0.5))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def draw_health(image: np.ndarray, readings, *, font_scale: float | None = None,
                show_boxes: bool = True) -> np.ndarray:
    """HP and confidence onto a COPY of `image` (BGR uint8), returned.

    A copy for the reason `object_detection.draw.draw_detections` gives at length: the array
    handed here is the same one the terrain pipeline reads in that iteration of a render loop,
    and annotating it in place would feed overlay ink into a perception stage.

    Takes either `HealthReading`s or `SmoothedReading`s -- both carry `hp`, `confidence` and
    `detection`, which is all this needs.
    """
    out = image.copy()
    h, w = out.shape[:2]
    thickness, scale = _sizes(w, None, font_scale)
    scale *= 0.8
    for reading in readings:
        det = reading.detection
        color = confidence_color(reading.confidence)
        if reading.hp is None:
            text = getattr(reading, "status", None) or getattr(reading.raw, "status", "?")
        else:
            text = f"{reading.hp} {reading.confidence:.2f}"

        raw = getattr(reading, "raw", reading)
        if show_boxes and raw.row_xyxy is not None:
            x0, y0, x1, y1 = raw.row_xyxy
            cv2.rectangle(out, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), color, 1, cv2.LINE_AA)
        if show_boxes and raw.bar_xyxy is not None:
            x0, y0, x1, y1 = raw.bar_xyxy
            cv2.line(out, (x0, y0), (x1, y0), color_for(det.label), 2, cv2.LINE_AA)

        bx0, by0, bx1, by1 = (int(round(v)) for v in det.xyxy)
        (tw, th), base = cv2.getTextSize(text, _FONT, scale, max(1, thickness // 2))
        tx = max(0, min(int((bx0 + bx1) / 2 - tw / 2), w - tw - 2))
        # Under the box, not over it: the top of a brawler's box is where the readout being
        # annotated actually is, and a label there covers the evidence.
        ty = min(by1 + th + base + 4, h - 2)
        cv2.putText(out, text, (tx, ty), _FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(out, text, (tx, ty), _FONT, scale, color, max(1, thickness // 2), cv2.LINE_AA)
    return out


def draw_summary(image: np.ndarray, readings, origin=(8, 22)) -> np.ndarray:
    """A status tally in the corner, in place. Counts what was READ, so a run of `occluded`
    frames is visible as a number rather than as an absence you have to notice."""
    counts: dict[str, int] = {}
    for r in readings:
        status = getattr(r, "status", None) or getattr(getattr(r, "raw", r), "status", OK)
        counts[status] = counts.get(status, 0) + 1
    thickness, scale = _sizes(image.shape[1])
    scale *= 0.7
    x, y = origin
    for status in sorted(counts, key=lambda k: (k != OK, k)):
        text = f"{status}: {counts[status]}"
        cv2.putText(image, text, (x, y), _FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), _FONT, scale,
                    (255, 255, 255) if status == OK else (60, 200, 245),
                    max(1, thickness // 2), cv2.LINE_AA)
        y += int(round(scale * 34))
    return image
