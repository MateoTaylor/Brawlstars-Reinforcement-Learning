"""Drawing detections onto a frame. Rendering only -- nothing here computes anything.

The same split `terrain/overlay.py` states for itself, and for the same reason: a drawing bug
must never be able to change a measurement. `predict` returns boxes; this puts them on pixels.

**Colours are per class and fixed, and they were checked against the terrain palette rather than
guessed.** These boxes end up in the same output frame as the reconstructed map, so the actual
values in `terrain/palette.py` matter -- currently floor #4e2b62 and wall #965ec6 (purples), bush
#1d9737 forest green, water #90f07c neon green, fence #e8d44a yellow.

The test is the largest per-channel gap, as a fraction of 255, between a marker and everything it
can be drawn over -- every terrain colour and every other marker. It is a per-channel max rather
than a distance because that is what "can I still see the outline" depends on: one channel far
apart is enough, three channels slightly apart is not.

**The palette was retargeted to the `9_5_brawlstars_eval` skin, and that cost the player its
green.** Worth reading before picking a colour here, because it is the concrete case the warning
at the bottom of this docstring was written for. Under day10 the player was a LIME, pushed
yellow-green specifically to clear day10's water (#17c338, a saturated neon) at 0.459. This skin's
water is #90f07c -- a *light yellow-green*, which is very nearly where that lime was pushed to --
and the same marker measured 0.25 against it. Both green slots are now terrain: forest green is
bush, light green is water, and there is no third green that clears them both.

So the player is WHITE, at 0.51 against its worst neighbour (water again). Pure blue scored higher
(0.59) and was rejected on meaning rather than on arithmetic: teammate is already a light blue, and
"which of these two blues is me" is the one question the marker exists to answer instantly. White
is the only strong option no terrain class and no other marker competes for.

    marker                floor  wall  bush  water  fence | enemy  player  teammate | worst
    enemy      (40,40,255) 0.69  0.62  0.89  0.78   0.67 |   -     0.84     0.76    | 0.62 wall
    player  (255,255,255)  0.83  0.63  0.89  0.51   0.71 |  0.84    -       0.76    | 0.51 water
    teammate (235,200,60)  0.62  0.42  0.71  0.44   0.67 |  0.76   0.76      -      | 0.42 wall
    Projectile (255,0,255) 0.69  0.41  0.89  0.94   0.83 |  0.84   1.00     0.78    | 0.41 wall

Every marker is ALSO drawn over a black rim (`project.draw_markers`), and that is not redundancy
with the table above -- it is what makes the table's worst row survivable. The palette is
retargeted per map, this retarget already broke one marker, and the next one will break another.
Re-derive these numbers when it happens; `terrain/palette.py` holds them, and
`scripts/vision_sample_palette.py` is how that file's values were measured in the first place.
"""
import cv2
import numpy as np

# BGR, because that is what `Frame.image` is and converting to RGB is the caller's last step.
CLASS_COLORS = {
    "enemy": (40, 40, 255),        # red -- the thing the policy is being trained to care about
    "player": (255, 255, 255),     # white -- you, and there should be exactly one. Green is the
                                   # obvious choice and this skin's terrain has taken both of them
    "teammate": (235, 200, 60),    # cyan-ish; meaningless in Solo Showdown, see `--ignore`
    "Projectile": (255, 0, 255),   # magenta -- our own model's one class. Capitalised because
                                   # `names` comes out of the ONNX and CVAT's label was
                                   # `Projectile`; this table is keyed by what the file says.
}
_FALLBACK = (0, 235, 235)          # yellow, for a model whose classes we have no opinion about

# **Why PURE magenta for projectiles rather than the pink that first looks right.** Same test as
# above, and WALL is the binding constraint both times: #965ec6 is itself a light purple, so a
# pink box over a wall cell reads as the same colour. Measured against the current palette:
#
#                floor  wall  bush  water  fence | enemy  player  teammate
#   (255,0,255)   0.69  0.41  0.89  0.94   0.83  |  0.84   1.00     0.78
#   (255,60,224)  0.69  0.29  0.89  0.91   0.83  |  0.72   0.76     0.76
#   (0,140,255)   0.62  0.42  0.78  0.56   0.29  |  0.84   1.00     0.76   <- orange, fails FENCE
#
# Dropping green to 0 is what opens the wall gap, because wall's green is 94 and that is the
# channel with room. Magenta is also the one hue nothing else here claims -- the palette is
# purple/green/yellow and the entities are red/white/cyan.
#
# 0.41 is the thinnest margin in this file and it is worth saying so plainly: it got thinner in
# the retarget (0.55 under day10's paler wall), and a projectile box crossing a wall is a little
# harder to follow than it was. It survives because a projectile box is small, moving, and drawn
# with a centre dot -- motion carries it where contrast alone would be marginal. If a future skin
# pushes it lower, that is the point to pick a new hue rather than to keep shaving.

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Line width and text height as a fraction of frame WIDTH. Both are sized to the frame rather
# than fixed, and that is not cosmetic tidying -- a normalized frame is 2002 px wide and
# `evaluate._side_by_side` scales it to a 476 px-tall panel, a 2.4x reduction. Drawing at a fixed
# 2 px and 0.5 font (fine on the frame itself) lands as a sub-pixel line and unreadable text in
# the very output the boxes exist to be looked at in. These fractions hold up on both.
_THICKNESS_FRAC = 1 / 500
_FONT_FRAC = 1 / 1670        # ~1.2 font scale at 2002 px, ~0.5 after a side-by-side downscale


def color_for(label: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(label, _FALLBACK)


def _sizes(width: int, thickness=None, font_scale=None) -> tuple[int, float]:
    """Line thickness and font scale for a frame this wide, unless the caller pinned them."""
    return (max(1, round(width * _THICKNESS_FRAC)) if thickness is None else thickness,
            max(0.35, width * _FONT_FRAC) if font_scale is None else font_scale)


def draw_detections(image: np.ndarray, detections, *, thickness: int | None = None,
                    font_scale: float | None = None, show_confidence: bool = True) -> np.ndarray:
    """Boxes and labels onto a COPY of `image` (BGR uint8), returned.

    A copy, not in place: the caller's frame is also what odometry and the classifier see in the
    same iteration of the render loop, and annotating the array they are handed would put box
    outlines into the terrain pipeline's input. That failure is subtle and permanent-looking --
    a locked cell that disagrees with the footage -- so the copy is not optional.
    """
    out = image.copy()
    h, w = out.shape[:2]
    thickness, font_scale = _sizes(w, thickness, font_scale)
    text_thickness = max(1, thickness // 2)
    for det in detections:
        color = color_for(det.label)
        x0, y0, x1, y1 = (int(round(v)) for v in det.xyxy)
        # Clamped only for drawing. `Detection.xyxy` keeps the model's own coordinates, which for
        # a brawler walking off the edge legitimately run past the frame.
        x0, x1 = max(0, min(x0, w - 1)), max(0, min(x1, w - 1))
        y0, y1 = max(0, min(y0, h - 1)), max(0, min(y1, h - 1))
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)

        text = f"{det.label} {det.confidence:.2f}" if show_confidence else det.label
        (tw, th), base = cv2.getTextSize(text, _FONT, font_scale, text_thickness)
        # Above the box, or inside its top when there is no room -- a label drawn off the top of
        # the frame is the one case where the box you most want to read is the one you cannot.
        ty = y0 - base - 2 if y0 - th - base - 2 >= 0 else min(y0 + th + base + 2, h - 1)
        cv2.rectangle(out, (x0, ty - th - base), (x0 + tw + 4, ty), color, -1)
        cv2.putText(out, text, (x0 + 2, ty - base), _FONT, font_scale, (0, 0, 0),
                    text_thickness, cv2.LINE_AA)
    return out


def draw_summary(image: np.ndarray, detections, origin=None) -> np.ndarray:
    """A per-class tally in the corner, in place. Counts what is DRAWN, so a class being ignored
    is absent here too -- the number and the picture never disagree.

    Top-RIGHT, and right-aligned, which took looking at the output to get right. The top-left is
    already occupied twice over: the game draws "Brawlers left: N" there, and
    `evaluate._label` writes the timestamp into the same corner of the composed canvas. Three
    overlapping texts is what you get by defaulting to (8, 22) without checking.
    """
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    w = image.shape[1]
    thickness, font_scale = _sizes(w)
    line = int(round(font_scale * 34))
    right = origin is None
    x, y = origin if origin is not None else (w - int(w * 0.01), int(line * 1.25))
    for label in sorted(counts):
        text = f"{label}: {counts[label]}"
        tx = x - cv2.getTextSize(text, _FONT, font_scale, thickness)[0][0] if right else x
        cv2.putText(image, text, (tx, y), _FONT, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(image, text, (tx, y), _FONT, font_scale, color_for(label),
                    max(1, thickness // 2), cv2.LINE_AA)
        y += line
    return image
