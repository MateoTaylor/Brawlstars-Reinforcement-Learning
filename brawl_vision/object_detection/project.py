"""Putting a screen-space box onto the tile map. The one place the two chunks actually meet.

`detector.py` returns boxes in raw frame pixels and knows nothing about geometry; `terrain/` owns
a calibrated ground plane and knows nothing about detections. This module is the seam, and it is
its own file so that the seam is a thing you can read rather than a few lines buried in a render
loop.

**The transform is: box bottom-centre -> viewport px -> rectified px -> tile.** Both hops already
exist -- `RectifyPlan.M` is the viewport-to-rectified homography and `rect_to_tile` is a scale and
an offset -- so nothing here re-derives geometry. What it does is pick the right POINT to push
through them, and that choice is the whole content of this module.

#### The anchor is wrong, and this is the measurement that says why

A brawler is drawn as a sprite with height, standing on the ground plane the homography was fit
to. Its contact point is the bottom of the sprite, so `Detection.ground_point` (the box
bottom-centre) *ought* to be "where it is standing". Projected markers land consistently further
from the camera than the thing they mark, and three measurements together say what is going on.

**1. The precision is fine; only the anchor is off.** Projecting the PLAYER every 30th frame and
comparing against the centre of the camera's window, four fixtures:

    standstill               n=14   dx +1.82 +-0.01   dy +2.27 +-0.03
    counted_walking          n=17   dx +5.63 +-2.17   dy +2.36 +-0.18
    showdown_alternate_map   n=41   dx -0.46 +-1.04   dy +2.02 +-0.22
    showdown_alternate_map2  n=41   dx -0.90 +-0.23   dy +1.93 +-0.20

Within a clip the y scatter is ~0.2 tiles. That is a *displacement*, not noise -- which is the
good case, because a displacement is a calibration problem and noise would not be fixable at all.
(dx is not a bias: it swings -0.9 to +5.6 across clips and scatters 2.2 tiles inside the walking
one, because the camera leads and lags the hero rather than centring it. Beware that this also
makes the dy column above an OVERSTATEMENT -- see point 3.)

**2. It is not perspective foreshortening.** That was the obvious suspect and it is mostly
innocent. Pushing a point down the frame by a fixed 40 screen px and reading off the tile
displacement, at frame centre:

    screen y   200    400    600    800   1000   1120
    tiles     +0.72  +0.67  +0.62  +0.57  +0.53  +0.51

A 40% swing top to bottom, on an error the same measurement puts at ~2 tiles. This camera is
nearly orthographic-oblique, so a fixed pixel error is a nearly fixed tile error. Across the frame
WIDTH it is flatter still (+0.59 to +0.61, with |dx| <= 0.08).

**3. The box does not enclose the brawler.** Look at one, drawn on a raw frame: the box spans the
NAMEPLATE, the health bar, the ammo pips, the sprite, and any active aura. On `standstill` f300 the
bottom edge sits on the rim of Mortis's green healing aura, ~68 px -- about one tile -- below his
feet. The rest of the discrepancy in point 1 is a bad baseline: the hero does not sit at the
window centre vertically, so part of that "+2.0" was never projection error.

And because the box tracks transient effects, the overshoot is not even constant. Player box
HEIGHT, sampled every 20th frame:

    standstill               n= 21   min 262  med 267  max 271   (spread  9 px)
    showdown_alternate_map   n= 76   min 264  med 308  max 380   (spread 116 px = 1.5 tiles)
    showdown_alternate_map2  n= 76   min 254  med 305  max 319   (spread  66 px)

Enemies swing wider still. A brawler with a super charged, a gadget aura or a shield is a taller
box than the same brawler standing plain, and the bottom edge moves with it.

#### Where that left the anchor

The fix is the ANCHOR, not a constant: `Detection.anchor(frac)` takes the point `frac` of the box
height up from the bottom edge, and `detector.anchor_frac` defaults to a measured **0.30**. The
justification is in that method's docstring -- a brawler cannot stand in a WALL or in WATER, the
occupancy map knows where those are, and that gives an accuracy test needing no hand labels. The
raw box bottom scores 23.7% impossible against a 26.5% base rate; 0.30 roughly halves it on both
clips measured.

`ground_offset_tiles` survives as a residual knob in TILES, on top of the anchor, for whatever a
per-box fraction cannot express. It defaults to 0 and should mostly stay there now.

**This is still not a calibration.** The optimum is a band (0.20 on the wall-heavy clip, 0.40-0.50
on the water-heavy one), the two disagree partly because the terrain classifier's own weak spot is
walls, and the box breathes by 1.5 tiles within a clip as auras come and go. Read a marker as
"a brawler is about here", not as a tile index. Settling it properly needs frames labelled with a
brawler's true tile, or a shadow-ellipse detector inside the box that does not care how tall the
box is.

Two more things worth stating plainly, because they bound what this can ever be used for. The
hero is **not** rigidly pinned to the camera -- near a map boundary the camera stops and the hero
walks toward the screen edge, which the dx column above already shows (+1.8 to +5.6 across clips,
scattering 2.2 tiles inside the walking one). So "odometry already knows where the hero is" is
only true away from boundaries, and a common-mode offset between hero and enemy does not reliably
cancel in relative terms either. Getting the anchor right is load-bearing, not cosmetic.
"""
import cv2
import numpy as np

from .draw import color_for
from .projectile_detection.classes import CUBE_BOX, CUBE_DROPPED


DEFAULT_ANCHOR_FRAC = 0.30

# A brawler's collision footprint, in tiles, taken from the simulator's own `entities.unit_radius:
# 0.4` (configs/default.yaml) -- diameter 0.8. Deliberately the SIM's number rather than a drawing
# constant invented here: the point of putting a detection on this map is to compare it against a
# world the sim also describes, and a marker sized to anything else would make that comparison
# read wrong by construction. Same reasoning `terrain/overlay.py` gives for importing the
# simulator's TILE_COLORS instead of picking its own.
FOOTPRINT_TILES = 0.8

# A projectile's collision footprint, same source: `proj_radius` in configs/brawlers.yaml runs
# 0.25-0.30 across the roster, so 0.30 doubled is the modal diameter. Drawn smaller than a
# brawler because it IS smaller, and because the two end up on the same panel.
PROJECTILE_FOOTPRINT_TILES = 0.6

# Where in a projectile's box to take its position: the CENTRE, not the 0.30-up-from-the-bottom
# that `DEFAULT_ANCHOR_FRAC` uses for a brawler.
#
# **The two constants answer different questions and only one of them has a defensible answer.**
# For a brawler the question is "which cell is it standing on", and the box bottom is nearly the
# ground, so an anchor low in the box is a real estimate of a real contact point. A projectile is
# not standing on anything. It is a blob in flight whose position is its middle, and there is no
# contact point to find -- so the centre is the only honest reading of the box.
#
# **And then projecting it through the ground homography is still wrong, unavoidably.** `plan.M`
# inverts the camera perspective for the plane z=0. A projectile at head height is above that
# plane, so it lands further from the camera than it really is -- the same failure mode as
# anchoring a brawler at its nameplate, which measured several tiles out. The error grows with
# flight height and with distance from the camera, and no constant fixes it: recovering a real
# tile from a flying object needs its height, which one box does not contain.
#
# So `projectiles_on_map` is a LOOK-AT-IT tool, not a measurement. It answers "is the detector
# firing on the thing I think it is, roughly over there" and nothing finer. Read a tile index off
# one of these markers and you will be reading the projectile's shadow position, displaced away
# from the camera by an amount nobody has measured.
PROJECTILE_ANCHOR_FRAC = 0.5

# The projectile model's other two classes DO sit on the ground, so unlike a projectile they have a
# contact point to find, and both were measured. The numbers are in brawl_deployment/perception/
# loot.py, under "Where a box is on the ground: the two anchors":
#
#   crate   0.30   the brawler default, and right for a crate too: sd 0.03 tiles on the
#                  calibration clip, the one clip with true tile positions
#   cube   -0.15   NEGATIVE, so below the box. A dropped cube floats, and its ground point is the
#                  shadow under it, a median 10 px below a median 65 px box
#
# They live here rather than in loot.py so the render projects with them too. A demo and the
# deployed bot should not disagree about where a crate is.
CRATE_ANCHOR_FRAC = 0.30
CUBE_ANCHOR_FRAC = -0.15
LOOT_ANCHOR_FRAC = {CUBE_BOX: CRATE_ANCHOR_FRAC, CUBE_DROPPED: CUBE_ANCHOR_FRAC}


def projectile_model_anchor(label: str) -> float:
    """The anchor for one of the projectile model's classes: measured for a crate or a cube, the
    centre (`PROJECTILE_ANCHOR_FRAC`) for a projectile and for any label not listed above."""
    return LOOT_ANCHOR_FRAC.get(label, PROJECTILE_ANCHOR_FRAC)


def to_tiles(detections, plan, ground_offset_tiles: float = 0.0,
             anchor_frac: float = DEFAULT_ANCHOR_FRAC):
    """`[(Detection, (tile_x, tile_y))]` in the plan's CAMERA-RELATIVE tile frame.

    Camera-relative, not world: tile (0, 0) is wherever the calibration's origin corner sits in
    this frame, exactly as `RectifyPlan` documents. Anchoring to the accumulated map means adding
    the odometry position, which is the caller's business because only the caller knows whether
    odometry is currently tracking.

    `anchor_frac` picks the point inside each box -- see `Detection.anchor` for how 0.30 was
    measured. `ground_offset_tiles` is the residual knob on top of it, in TILES rather than box
    fractions, for the part a per-box fraction cannot express. With a sane anchor it should
    rarely be needed; it defaults to 0.
    """
    detections = list(detections)
    if not detections:
        return []
    points = np.array([d.anchor(anchor_frac) for d in detections], np.float64).reshape(-1, 1, 2)
    rect = cv2.perspectiveTransform(points, plan.M).reshape(-1, 2)
    tiles = plan.rect_to_tile(rect)
    if ground_offset_tiles:
        tiles = tiles - np.array([0.0, ground_offset_tiles])
    return [(d, (float(t[0]), float(t[1]))) for d, t in zip(detections, tiles)]


def to_tile_quads(detections, plan):
    """`[(Detection, (4, 2) tile corners)]` -- each box as a QUAD in camera-relative tile space.

    **This is the DETECTOR's box, not a footprint, and it is for drawing on a rectified frame.**
    `to_tiles` answers "which cell is it standing on" and deliberately throws the box away;
    this keeps the whole box, because on a rectified panel the box is still the honest
    representation of what the model actually found.

    It lands correctly for a reason worth stating: the rectified frame is the raw frame put
    through `plan.M`, so pushing the box corners through the SAME matrix warps them exactly as the
    pixels under them were warped. The quad therefore encloses the same sprite it enclosed on
    screen -- nameplate, aura and all.

    A quad, not a rectangle. The homography shears -- mildly, this camera is near
    orthographic-oblique, but visibly -- and squaring it up to an axis-aligned bounding box would
    draw a shape that is not the model's output and not the sprite's outline either.

    Note the corners are NOT on the ground plane, which is the whole reason `to_tiles` exists
    separately. The box top is a nameplate floating above the world, so it projects to a tile
    several away and the quad is much taller than the brawler. That is honest for a picture of a
    screen-space box; it is exactly what you must not read a position off.
    """
    out = []
    for det in detections:
        x0, y0, x1, y1 = det.xyxy
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                           np.float64).reshape(-1, 1, 2)
        rect = cv2.perspectiveTransform(corners, plan.M).reshape(-1, 2)
        out.append((det, plan.rect_to_tile(rect)))
    return out


def draw_boxes(canvas: np.ndarray, quads, origin_tiles, scale: float, *,
               rgb: bool = True, label: bool = True) -> np.ndarray:
    """Draw full detection quads on `canvas`, IN PLACE. Same `(origin_tiles, scale)` contract as
    `draw_markers`, so the two can annotate the same panel without a second mapping to keep right.

    For the rectified panel. The map panel gets `draw_markers` instead -- a box that encloses a
    nameplate means something on a picture of the screen and nothing at all on a tile grid.
    """
    cx0, cy0 = origin_tiles
    thickness = max(1, int(round(scale / 14)))
    h, w = canvas.shape[:2]
    for det, quad in quads:
        pts = (quad - np.asarray([cx0, cy0], np.float64)) * scale
        if not np.isfinite(pts).all() or np.abs(pts).max() > 1e5:
            continue      # a corner projected near the horizon; nothing sane to draw
        pts = np.clip(pts, -1e4, 1e4).round().astype(np.int32).reshape(-1, 1, 2)
        color = color_for(det.label)
        if rgb:
            color = color[::-1]
        cv2.polylines(canvas, [pts], True, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.polylines(canvas, [pts], True, color, thickness, cv2.LINE_AA)
        if label:
            text = f"{det.label} {det.confidence:.2f}"
            fs = max(0.34, scale / 60)
            tx, ty = int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min())
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ty = ty - base - 2 if ty - th - base - 2 >= 0 else min(ty + th + base + 2, h - 1)
            tx = max(0, min(tx, w - tw - 4))
            cv2.rectangle(canvas, (tx, ty - th - base), (tx + tw + 4, ty), color, -1)
            cv2.putText(canvas, text, (tx + 2, ty - base), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def draw_markers(canvas: np.ndarray, placed, origin_tiles, scale: float, *,
                 footprint_tiles: float = FOOTPRINT_TILES, label: bool = False,
                 rgb: bool = True) -> np.ndarray:
    """Box each projected detection on `canvas`, IN PLACE. Returns it for chaining.

    `canvas` pixel (0, 0) is tile `origin_tiles`, and one tile is `scale` pixels -- the one
    formula both layouts reduce to, so this function does not need to know which one called it.

    **A box of the brawler's FOOTPRINT, not a dot and not the detector's own box.** The detector's
    box is in screen pixels and encloses a nameplate and an aura (see `Detection.anchor`), so
    warping it here would draw a skewed quad around mostly-not-brawler. What is drawn instead is
    the collision footprint the simulator uses -- `FOOTPRINT_TILES`, straight from the sim's
    `entities.unit_radius` -- centred on the projected contact point. It therefore means the same
    thing in both panels of the `view` layout and the same thing the sim means by "a brawler is
    here", which is the comparison this whole render exists to support.

    **`rgb=True`, opposite to `draw.draw_detections`, and it is not an oversight.** That one
    annotates `Frame.image`, which is BGR like everything out of OpenCV. This one annotates
    RENDER canvases -- `render_map` builds its lookup table with matplotlib's `to_rgb`, and
    `_side_by_side` converts the frame before compositing -- because `VideoSink` takes RGB. Left
    unswapped, `CLASS_COLORS["enemy"]` renders every enemy marker BLUE, which is a bug that looks
    like a design choice.
    """
    cx0, cy0 = origin_tiles
    half = max(2, int(round(footprint_tiles * scale / 2)))
    thickness = max(1, int(round(scale / 12)))
    for det, (tx, ty) in placed:
        x = int(round((tx - cx0) * scale))
        y = int(round((ty - cy0) * scale))
        if not (-half <= x < canvas.shape[1] + half and -half <= y < canvas.shape[0] + half):
            continue          # projected outside this panel; a clipped box is worse than none
        color = color_for(det.label)
        if rgb:
            color = color[::-1]
        p0, p1 = (x - half, y - half), (x + half, y + half)
        # Black rim first, colour on top. Enemy red is nearly the terrain palette's BUSH red, so
        # a bare box vanishes over a bush -- which is precisely where an enemy tends to be.
        cv2.rectangle(canvas, p0, p1, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.rectangle(canvas, p0, p1, color, thickness, cv2.LINE_AA)
        # A dot at the exact contact point: the box is a fixed footprint, so without this the
        # sub-tile position the projection actually produced is rounded away by the box itself.
        cv2.circle(canvas, (x, y), max(1, thickness), (0, 0, 0), -1, cv2.LINE_AA)
        if label:
            cv2.putText(canvas, det.label, (x + half + 3, y + half // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.35, scale / 40), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, det.label, (x + half + 3, y + half // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.35, scale / 40), color, 1, cv2.LINE_AA)
    return canvas
