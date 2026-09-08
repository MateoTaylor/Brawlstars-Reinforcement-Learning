"""Where in a recording the player was actually playing -- the span between the lobby and the
results screen.

**Why this is not `clips.detect_usable_range`.** That function answers a different question with a
different signal: it finds where the *recording* stops showing the game, by looking for the iOS
Control Center sliding in as a spike in whole-frame difference. It says nothing about the lobby,
the "BRAWL" intro swoop, or the "Defeated" overlay -- all of which are Brawl Stars rendering the
match's own map, so whole-frame difference sees nothing unusual and every one of them survives the
cut. This module answers "was the player in control of a brawler", which is the question that
matters when the frames are going out for annotation.

**The signal is the on-screen controls, because they exist if and only if you can act.** The
attack and Super buttons are absent on the loading screen, absent in the lobby, absent during the
intro animation, absent the instant "Defeated" appears, and absent on the results screen -- all
measured. Nothing else on screen has that property: the map, the HUD readouts and the brawler
sprites all appear in the menus too.

**Detected as a RING, not as a picture of a button, and that is what makes it survive a brawler
swap.** The disc under the ring changes colour with ammo state (blue loaded, orange mid-reload)
and the icon inside it is the brawler's own, so template-matching a reference crop would key on
exactly the parts that vary. What does not vary is the chrome: a dark annulus at a fixed screen
position. `ring_score_at` measures how much of the gradient around that circle points *radially*,
which a ring does and the map underneath, however busy, does not.

**The button positions are calibrated PER RECORDING rather than read from `data/hud_mask.json`,
and that is not redundancy -- this footage contains two different layouts.** The mask records one
of them (attack at (1487, 813) in the normalized viewport, radius ~65), measured on the 2026-08
clips. Four of the sixteen recordings here instead put the attack button at (1752, 781) with
radius ~47 -- smaller and further right, the game's own UI-scale setting. Scored against the
mask's fixed rects those four peak at 0.36 and read as containing no gameplay at all, which is
how the discrepancy surfaced. A layout is a property of a recording, so it is measured from one.

**Calibration runs on a TEMPORAL MEDIAN, which is what makes it easy.** Over frames spread across
the whole recording the world scrolls and dissolves to mush while the screen-anchored buttons stay
pin-sharp, so the only crisp circles left in the lower right are the ones being looked for. The
median is also what makes the calibration robust to the menus: they are a minority of any
recording that contains a match, and a median discards minorities. `cv2.HoughCircles` proposes
candidates on that image and `ring_score_at` ranks them -- Hough alone is not enough, because it
also finds the large faint aim halos, and its radius estimate is loose enough to cost 0.3 of score
(day12_recording1 scored 0.56 at Hough's radius and 0.90 after refinement).

Measured, per-frame, at the calibrated anchors:

    gameplay      0.649 - 0.790     (hand-labelled, two clips, three brawler/ammo states)
    lobby         0.280             <- the busiest false positive: portrait art and UI circles
    intro swoop   0.254
    defeated      0.114
    loading       0.120
    results       0.002 - 0.121

`GAMEPLAY_THRESHOLD` sits in the middle of that 2.3x gap rather than snug against either side,
because the sample is a handful of clips of one game mode and what matters is how it behaves on a
mode nobody has looked at yet.

**The span is the longest run, not the first.** A recording may start mid-match (several here do),
and a menu may flicker a frame over the threshold. Taking the longest gap-filled run is robust to
both, and to the case a plain threshold gets wrong: a Super detonating over the button drops the
score for a few frames in the *middle* of a match, and those frames are gameplay -- they are
bracketed by it on both sides. Interior gaps are filled; only the outer boundary is cut.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from .capture import normalize_viewport
from .clips import load_bounds

# Sidecar written next to each recording, the same idiom as `clips.BOUNDS_SUFFIX`: the scan is a
# decode pass whose answer never changes for a given file, and re-running the export at a different
# frame rate should not pay for it twice.
SPAN_SUFFIX = ".gameplay.json"

# Midpoint of the measured gap (0.280 non-gameplay / 0.649 gameplay). See the module docstring.
GAMEPLAY_THRESHOLD = 0.45

# Where a control button may be found, as fractions of the normalized viewport. Both observed
# layouts put attack and Super inside this; the floor at y=0.60 is what keeps the calibration off
# the menu chrome, and it is the difference between the one recording here that contains no match
# being rejected and it being exported whole -- its brawler-select screen has a crisp circle at
# y=0.50 that scores 0.81 and is not a button.
BUTTON_ROI = (0.55, 0.60, 1.00, 1.00)

# Radius band to search, in pixels of the 2002x1126 normalized viewport. The two observed layouts
# sit at ~47 and ~65; the band is wide enough to admit a third without being wide enough to start
# matching map decoration.
RADIUS_PX = (40, 115)

# A calibration whose best candidate scores below this is not a control cluster. The weakest
# genuine recording here calibrates at 0.77.
MIN_CALIBRATION_SCORE = 0.60

# How many calibrated anchors to score each frame against, best first. More than one so that a
# Super detonating over a button does not drop the frame -- the effect is large but never covers
# the whole cluster.
MAX_ANCHORS = 3

# Frames sampled for the temporal median. Enough that the world averages away; few enough that
# calibration stays a second or two of seeking.
MEDIAN_SAMPLES = 24

# Interior dropouts up to this many SAMPLES are filled rather than treated as the end of the
# match. At the 2 Hz default that is 3 s of unreadable buttons -- longer than any detonation
# observed, far shorter than the shortest non-gameplay stretch (the ~10 s of loading + lobby +
# intro on ScreenRecording_08-27).
MAX_GAP_SAMPLES = 6

# A run shorter than this is not a match. Guards against a few menu frames scoring high together
# in a recording that contains no gameplay at all.
MIN_RUN_SAMPLES = 10


def _gradients(image: np.ndarray):
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 0)
    return (cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))


def ring_score_at(gx, gy, cx: float, cy: float, radius: float, n_theta: int = 180) -> float:
    """How strongly a ring of `radius` sits at `(cx, cy)`, in [0, 1]. `gx`/`gy` come from
    `_gradients` and are in the same pixel space as the coordinates.

    Two factors, both needed. The first is the mean of |gradient . radial_direction| divided by the
    mean gradient magnitude around the circle: "does the edge here curve about this centre". The
    second is that mean magnitude itself: "is there an edge here at all". The ratio alone fires on
    flat regions where a few compression-noise gradients happen to align; the magnitude alone fires
    on any busy terrain.
    """
    h, w = gx.shape
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ct, st = np.cos(theta), np.sin(theta)
    xs = np.clip((cx + radius * ct).astype(int), 0, w - 1)
    ys = np.clip((cy + radius * st).astype(int), 0, h - 1)
    vx, vy = gx[ys, xs], gy[ys, xs]
    magnitude = np.hypot(vx, vy)
    radial = np.abs(vx * ct + vy * st)
    # The 200 cap keeps one blown-out specular edge from carrying the whole circle.
    return float(np.mean(radial) / (np.mean(magnitude) + 1e-6)
                 * np.mean(np.minimum(magnitude, 200.0)) / 200.0)


def refine_circle(image: np.ndarray, cx: float, cy: float, radius: float):
    """Local search around a proposed circle -> `(score, cx, cy, radius)` at the best fit.

    Hough's radius estimate is loose -- it locates the accumulator peak, not the annulus this
    scores against -- and the cost of accepting it is real: day12_recording1's attack button
    scores 0.56 at the proposed radius and 0.90 two pixels and four radius steps away. Since the
    result is reused for every frame of the recording, the search is worth paying for once.
    """
    gx, gy = _gradients(image)
    best = (-1.0, cx, cy, radius)
    for dx in range(-8, 9, 2):
        for dy in range(-8, 9, 2):
            for r in np.arange(0.72 * radius, 1.20 * radius, 0.03 * radius):
                s = ring_score_at(gx, gy, cx + dx, cy + dy, r)
                if s > best[0]:
                    best = (s, float(cx + dx), float(cy + dy), float(r))
    return best


def walk_frames(path, box, indices, normalize: bool = True):
    """Decode `path` once, yielding `(index, image)` for exactly `indices`, normalized to the
    viewport unless `normalize` is False.

    **The scan and the export MUST agree about which frame an index names, and sharing this walker
    is what guarantees it.** The obvious stepped read -- `grab()` step-1 times, then `read()` --
    does not: 29 grabs followed by a read returns frame 29, not frame 0, so labelling it with the
    loop counter puts every index 29 frames (half a second at 2 Hz) ahead of the pixels that were
    actually scored. Written that way the scan scored one frame and the export wrote a different
    one, and the frame index in the filename was a claim about neither. Here the position counter
    is advanced by the decode itself, so an index always names the frame that was decoded at it.

    `grab()` rather than `read()` for the skipped frames, as `clips.ClipReader` does: both advance
    the decoder, but `retrieve()` is where the 2436x1126 frame is converted and copied, and that
    is most of the cost.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open clip: {path}")
    try:
        pending = iter(sorted(indices))
        nxt = next(pending, None)
        pos = 0
        while nxt is not None:
            if pos == nxt:
                ok, raw = cap.read()
                if not ok:
                    break
                yield pos, (np.ascontiguousarray(normalize_viewport(raw, box))
                            if normalize else raw)
                nxt = next(pending, None)
            elif not cap.grab():
                break
            pos += 1
    finally:
        cap.release()


def median_frame(path, box, n: int = MEDIAN_SAMPLES) -> np.ndarray:
    """Per-pixel median of `n` normalized frames spread evenly across the recording.

    The world moves and the HUD does not, so this is a picture of the HUD. See the module
    docstring on why that is the right image to calibrate against.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open clip: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        for fi in np.linspace(0, max(0, total - 1), n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, raw = cap.read()
            if ok:
                frames.append(np.ascontiguousarray(normalize_viewport(raw, box)))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"{path} decoded zero frames")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def calibrate_buttons(median: np.ndarray, max_anchors: int = MAX_ANCHORS) -> list[dict]:
    """Find the control-button rings in a recording's median frame, best first.

    Returns `[{score, cx, cy, r}, ...]`, empty if this recording shows no control cluster -- which
    is a real answer, not a failure: one of the recordings here is nine seconds of menus.
    """
    h, w = median.shape[:2]
    rx0, ry0, rx1, ry1 = BUTTON_ROI
    x0, y0 = int(rx0 * w), int(ry0 * h)
    sub = median[y0:int(ry1 * h), x0:int(rx1 * w)]
    gray = cv2.GaussianBlur(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY), (5, 5), 1.5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=60,
                               param1=120, param2=45,
                               minRadius=RADIUS_PX[0], maxRadius=RADIUS_PX[1])
    if circles is None:
        return []
    found = []
    for cx, cy, r in circles[0]:
        score, cx, cy, r = refine_circle(median, float(cx) + x0, float(cy) + y0, float(r))
        # Circles clipped by the frame edge are rejected rather than scored. Sampling one means
        # clamping a stretch of the annulus onto the edge row, which invents a long straight
        # "radial" agreement out of a single pixel and scores whatever happens to be there.
        if cx - r < 0 or cy - r < 0 or cx + r >= w or cy + r >= h:
            continue
        found.append({"score": round(score, 4), "cx": round(cx, 1),
                      "cy": round(cy, 1), "r": round(r, 1)})
    found.sort(key=lambda c: -c["score"])
    if not found or found[0]["score"] < MIN_CALIBRATION_SCORE:
        return []
    return found[:max_anchors]


def longest_span(flags, max_gap: int = MAX_GAP_SAMPLES, min_run: int = MIN_RUN_SAMPLES):
    """Longest run of True in `flags`, tolerating interior gaps of up to `max_gap`. Returns
    `(first, last)` inclusive, or `None` if nothing qualifies.

    Gaps are filled BEFORE the longest run is chosen, not after, so a match interrupted by two
    brief dropouts is one span rather than three competing ones.
    """
    hits = [i for i, f in enumerate(flags) if f]
    if not hits:
        return None
    runs = []
    start = prev = hits[0]
    for i in hits[1:]:
        if i - prev - 1 > max_gap:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    first, last = max(runs, key=lambda r: r[1] - r[0])
    return (first, last) if last - first + 1 >= min_run else None


def scan_gameplay(path, scan_fps: float = 2.0, threshold: float = GAMEPLAY_THRESHOLD) -> dict:
    """One stepped decode pass -> the frames that are gameplay, and their scores.

    Samples on the same grid the export will later write, via the same `walk_frames`, so every
    exported frame is one that was individually scored rather than merely bracketed by scored
    ones. `frames` holds file-relative indices; `span` indexes into `frames`/`scores`, not into
    the file.
    """
    path = Path(path)
    bounds = load_bounds(path)
    box = tuple(bounds["content_box"])
    anchors = calibrate_buttons(median_frame(path, box))

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open clip: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.release()
    step = max(1, int(round(fps / scan_fps)))
    # The WHOLE file is scanned, and `bounds["usable"]` is deliberately not used as an outer bound
    # even though it is right there. Its tail heuristic is a frame-difference spike, which a
    # violent enough moment of play also produces: on ScreenRecording_08-29 it cuts at f4585 while
    # f4747 is still unambiguous gameplay -- a Super detonating, not the Control Center. That cut
    # is the right conservative default for its own caller, where a few stray frames would poison
    # an integrated odometry track; here the cost is reversed, because every frame it removes is a
    # frame that would have been annotated. The button signal is the more specific instrument for
    # this question, so it is left to answer it alone.
    last = bounds["n_frames"] - 1

    # Gradients are computed over a crop around the anchors rather than the whole frame: the
    # anchors span a few hundred pixels of one corner, and Sobel over all 2.3 megapixels for each
    # of them is most of the per-frame cost for none of the information.
    frames, scores = [], []
    if anchors:
        pad = 12
        ax0 = max(0, int(min(a["cx"] - a["r"] for a in anchors) - pad))
        ay0 = max(0, int(min(a["cy"] - a["r"] for a in anchors) - pad))
        ax1 = int(max(a["cx"] + a["r"] for a in anchors) + pad)
        ay1 = int(max(a["cy"] + a["r"] for a in anchors) + pad)
        for index, image in walk_frames(path, box, range(0, last + 1, step)):
            h, w = image.shape[:2]
            crop = image[ay0:min(h, ay1), ax0:min(w, ax1)]
            gx, gy = _gradients(crop)
            frames.append(index)
            scores.append(max(ring_score_at(gx, gy, a["cx"] - ax0, a["cy"] - ay0, a["r"])
                              for a in anchors))

    span = longest_span([s >= threshold for s in scores])
    return {
        "source_fps": float(fps),
        "scan_fps": float(scan_fps),
        "step": step,
        "threshold": float(threshold),
        "n_frames": bounds["n_frames"],
        "content_box": list(box),
        "anchors": anchors,
        "frames": frames,
        "scores": [round(float(s), 4) for s in scores],
        "span": list(span) if span else None,
    }


def load_gameplay_span(path, scan_fps: float = 2.0, threshold: float = GAMEPLAY_THRESHOLD,
                       refresh: bool = False) -> dict:
    """`scan_gameplay` for `path`, cached in a `.gameplay.json` sidecar beside the recording.

    Keyed on the scan parameters as well as the file: a sidecar written at a different `scan_fps`
    or `threshold` describes a different grid and is rescanned rather than silently reused.
    """
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + SPAN_SUFFIX)
    if sidecar.exists() and not refresh:
        cached = json.loads(sidecar.read_text())
        if cached.get("scan_fps") == scan_fps and cached.get("threshold") == threshold:
            return cached
    scan = scan_gameplay(path, scan_fps=scan_fps, threshold=threshold)
    sidecar.write_text(json.dumps(scan, indent=1) + "\n")
    return scan


def span_frames(scan: dict, margin: int = 0) -> list[int]:
    """File-relative indices of the gameplay frames in `scan`, trimmed by `margin` SAMPLES at each
    end. Includes frames inside the span that individually scored below threshold -- see the
    module docstring on why interior dropouts are kept.
    """
    if not scan["span"]:
        return []
    first, last = scan["span"]
    first, last = first + margin, last - margin
    if first > last:
        return []
    return scan["frames"][first:last + 1]
