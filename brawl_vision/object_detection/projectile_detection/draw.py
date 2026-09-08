"""Drawing projectile boxes onto a frame. Rendering only, same split as `object_detection/draw.py`.

**Why this is not just `draw_detections` with a different colour.** It is the same picture of the
same kind of box, but the boxes are an order of magnitude smaller and there are more of them, and
both of those break the sibling function's sizing in ways you only see in the output.

  * **Line weight.** `draw_detections` sizes its stroke at width/500 -- 4 px on a 2002 px frame,
    which is right for a brawler box 300 px tall. The measured projectile boxes run 31-514 px
    diagonal with a MEDIAN of 97, so a 4 px outline on a median box is ~8% of it and the smallest
    boxes become solid magenta lozenges. Halved to width/1000, and floored at 1 so it survives the
    2.4x downscale `_side_by_side` applies.
  * **Labels off by default.** One class means the text can only ever say "Projectile", so the
    label carries no information the colour does not, and there can be six of them at once in a
    space where the brawler boxes have two. The confidence IS worth seeing while judging a model,
    so `show_confidence=True` gets it back as a bare number with no filled chip behind it.
  * **A centre dot.** A projectile is a blob, and the useful question about it is *where*, not
    *how big*. The box answers a question nobody asked; the dot answers the one that matters and
    stays visible when the box shrinks below a few pixels after downscaling.

Colour comes from `object_detection/draw.color_for`, not from a constant here -- one table for
every box that can land in the same frame, so the class-vs-colour mapping cannot fork.
"""
import cv2
import numpy as np

from ..draw import color_for

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Half the entity detector's 1/500, for the reason in the module docstring. Both are fractions of
# frame WIDTH rather than fixed pixel counts, so they hold up on the raw 2002 px frame and on the
# ~830 px panel a side-by-side render scales it down to.
_THICKNESS_FRAC = 1 / 1000
_FONT_FRAC = 1 / 2400        # ~0.83 at 2002 px, ~0.35 after the side-by-side downscale


def draw_projectiles(image: np.ndarray, detections, *, thickness: int | None = None,
                     font_scale: float | None = None, show_confidence: bool = False,
                     centre_dot: bool = True) -> np.ndarray:
    """Boxes onto a COPY of `image` (BGR uint8), returned.

    A copy for the same non-negotiable reason `draw_detections` documents: in the render loop this
    frame is also what odometry and the terrain classifier read, and painting outlines into their
    input would feed the terrain pipeline pictures of boxes.
    """
    out = image.copy()
    h, w = out.shape[:2]
    thickness = max(1, round(w * _THICKNESS_FRAC)) if thickness is None else thickness
    font_scale = max(0.3, w * _FONT_FRAC) if font_scale is None else font_scale
    for det in detections:
        color = color_for(det.label)
        x0, y0, x1, y1 = (int(round(v)) for v in det.xyxy)
        # Clamped for drawing only, exactly as `draw_detections` does -- `Detection.xyxy` keeps
        # the model's own coordinates, and a shot leaving the frame legitimately runs past it.
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0, x1 = max(0, min(x0, w - 1)), max(0, min(x1, w - 1))
        y0, y1 = max(0, min(y0, h - 1)), max(0, min(y1, h - 1))
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
        if centre_dot and 0 <= cx < w and 0 <= cy < h:
            # Black rim under the dot, the same trick `project.draw_markers` uses: this frame is
            # gameplay, not a flat palette, and a projectile sits on top of whatever it is flying
            # over -- frequently something bright.
            cv2.circle(out, (cx, cy), thickness + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), thickness, color, -1, cv2.LINE_AA)
        if show_confidence:
            # No filled chip behind it. At this box size the chip would be larger than the box,
            # and it is the box you are trying to look at.
            text = f"{det.confidence:.2f}"
            (tw, th), base = cv2.getTextSize(text, _FONT, font_scale, 1)
            ty = y0 - base - 1 if y0 - th - base - 1 >= 0 else min(y1 + th + 1, h - 1)
            tx = max(0, min(x0, w - tw - 1))
            cv2.putText(out, text, (tx, ty), _FONT, font_scale, (0, 0, 0), thickness + 2,
                        cv2.LINE_AA)
            cv2.putText(out, text, (tx, ty), _FONT, font_scale, color, max(1, thickness),
                        cv2.LINE_AA)
    return out
