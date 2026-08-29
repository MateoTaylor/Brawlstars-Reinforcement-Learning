"""Recorded-clip playback, behind the same `Frame` interface as live capture.
See Terrain_Perception_Build_Plan.md Phase A.

**This module is the test harness for the entire chunk, not optional scaffolding.** Every
acceptance criterion from Phase C onward is a replay against a clip with known ground truth --
live-game iteration is too slow and too uncontrollable to debug a homography against.

**`cv2.VideoCapture` rather than `imageio-ffmpeg`, reversing the plan's original call.** The plan
picked imageio to avoid adding a dependency, but OpenCV became a hard dependency of this package
the moment Phase C existed, so that argument is moot. Two things settled it:

  - **Real per-frame timestamps.** `CAP_PROP_POS_MSEC` reports the container's actual
    presentation times; imageio exposes only a nominal fps. These clips are phone screen
    recordings at a nominal 56.9-59.4 fps that genuinely drop frames -- `standstill.mp4` carries a
    166.67 ms gap (10 frames at 60 Hz) mid-clip. Synthesizing `index / fps` would hand Phase F a
    uniform dt that is wrong exactly where the camera moved furthest between samples.
  - **Frame count agreement.** `cv2` and `imageio_ffmpeg.count_frames_and_secs` agree on all three
    fixtures (478 / 547 / 1351), while imageio's streaming `read_frames` over-yields (503 / 565 /
    1364). Two readers disagreeing about which frame is "frame 400" would silently invalidate
    every frame-indexed ground-truth annotation.

**Orientation is asserted, not trusted.** These files are stored portrait (1126x2436) with a
rotation flag that makes them display landscape (2436x1126). OpenCV 5.0 and imageio both apply it
here, but whether a given reader honors container rotation has changed across OpenCV versions --
so a source that comes back taller than it is wide raises rather than quietly feeding every later
stage a sideways world.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .capture import Frame, detect_content_box, normalize_viewport
from .config import VisionConfig

# Sidecar written next to each clip. TRACKED in git despite living beside gitignored footage:
# it is a derived ANNOTATION (a few integers), not Supercell's pixels, and re-deriving it costs a
# full decode pass. See tests/fixtures/vision/README.md.
BOUNDS_SUFFIX = ".bounds.json"

_DIFF_SAMPLE_DIV = 8      # frame-difference is computed on a 1/8-scale gray copy; ample for this
_TAIL_FRACTION = 0.85     # only the last 15% of a clip is searched for the contamination onset
_TAIL_SIGMA = 6.0         # ...and it must exceed body mean + this many sd to count
_TAIL_TRIM = 0.02         # top 2% of body diffs dropped before that mean/sd is computed


def _open(path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open clip: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if h > w:
        cap.release()
        raise ValueError(
            f"{path} decodes as {w}x{h} (portrait). These clips are stored portrait with a "
            f"rotation flag and must decode landscape; this reader is not applying it. See this "
            f"module's docstring."
        )
    return cap


def detect_usable_range(path, tail_fraction: float = _TAIL_FRACTION,
                        sigma: float = _TAIL_SIGMA, trim: float = _TAIL_TRIM) -> dict:
    """One full decode pass -> `{n_frames, usable: [first, last], content_box}`.

    **Finds where the recording stops being gameplay.** Every fixture clip ends with the iOS
    Control Center swiping in over the game -- the recording was stopped from the phone, so the
    last second or so is phone UI, not Brawl Stars. Feeding those frames to odometry would
    integrate a large bogus shift, and to the classifier would train it on a translucent settings
    panel.

    Detected as a sustained spike in frame-to-frame difference: the overlay slides in, which moves
    far more of the frame than gameplay ever does (measured at ~20x the body mean). Only the last
    `tail_fraction` of the clip is searched, so a genuinely violent moment of gameplay in the
    middle -- a death, a Super, the camera snapping -- cannot be mistaken for the end.

    **The threshold is computed from a TRIMMED body sample, and that is a bug fix.** Contamination
    that begins just before the `tail_fraction` boundary lands inside the sample that sets the
    threshold, and one 100-unit spike in a body whose mean is ~1 raises the 6-sigma bar far above
    anything the contamination itself produces afterwards. `training_gadget.mp4` cuts to a brawler
    splash screen at frame 508 of 599 -- 0.848 of the clip, ONE frame outside the search window --
    and that single excluded spike inflated the body sd from ~0.9 to 4.67 and the threshold from
    ~6 to 29. The detector then found nothing until frame 558, where the splash screen ends, and
    declared 50 frames of full-screen character art usable. Nothing failed: the pipeline happily
    classified the splash art as terrain and deposited it into the occupancy map.

    Dropping the top `trim` of body diffs before taking mean and sd removes exactly that
    self-concealment. It was preferred to a median/MAD threshold, which is more robust in the
    abstract but measured far too sensitive here -- gameplay diffs are smooth enough that
    median + 6*MAD trips on ordinary camera snaps and cut 56 frames of real play off two clips.
    With the trim, five of the seven fixtures keep the cut they already had.

    The content box is unioned over the same pass -- but **only over frames inside the usable
    range**, which is not a detail. The iOS overlay that contaminates every clip's tail spans the
    FULL capture width, so unioning it in reports the whole frame as content and defeats the
    pillarbox detection entirely: `zone_grows_from_east` came back as x=[0, 2435] instead of its
    true x=[217, 2217], which would have skipped normalization and silently handed Phase C a
    homography fit against black bars. Per-frame boxes are kept (four ints each, not the frames
    themselves) and unioned after the cut is known.
    """
    cap = _open(path)
    prev = None
    diffs = []
    sampled: list[tuple[int, tuple[int, int, int, int]]] = []
    n = 0
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if n % 25 == 0:
                # Keep the BOX, not the frame -- a full-resolution frame is ~8 MB and a long clip
                # would sample dozens of them.
                sampled.append((n, detect_content_box([fr])))
            small = cv2.resize(fr, (fr.shape[1] // _DIFF_SAMPLE_DIV, fr.shape[0] // _DIFF_SAMPLE_DIV))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            diffs.append(0.0 if prev is None else float(np.abs(gray - prev).mean()))
            prev = gray
            n += 1
    finally:
        cap.release()

    if n == 0:
        raise ValueError(f"{path} decoded zero frames")

    d = np.array(diffs)
    head = 10                      # skip the opening frames; the first diff is 0 by construction
    split = max(head + 1, int(n * tail_fraction))
    body = d[head:split]
    cut = n
    if len(body) > 1:
        keep = body[body <= np.quantile(body, 1.0 - trim)] if trim else body
        threshold = keep.mean() + sigma * keep.std()
        over = np.nonzero(d[split:] > threshold)[0]
        if len(over):
            cut = split + int(over[0])

    in_body = [box for idx, box in sampled if idx < cut]
    if not in_body:                                  # pathologically short clip
        in_body = [box for _, box in sampled]

    # MEDIAN of the per-frame bounds, not their union. The viewport is a fixed rectangle, so this
    # is estimating a constant corrupted by noise in BOTH directions, and a union only defends
    # against one of them:
    #   - a frame whose content happens to be dark near an edge reports the bound too far IN,
    #   - compression ringing and HUD glow at the bar boundary report it too far OUT.
    # Measured on `zone_grows_from_east`, the union put x0 at 196 -- outside the true viewport, so
    # the normalized crop carried a strip of black bar down its left edge. The median puts it at
    # 206, where the bulk of frames agree, and the crop lands entirely inside real world pixels.
    per_axis = np.array(in_body)
    x0, x1, y0, y1 = (int(round(float(np.median(per_axis[:, k])))) for k in range(4))

    return {
        "n_frames": n,
        "usable": [0, cut - 1],
        "content_box": [x0, x1, y0, y1],
    }


def load_bounds(path, cfg: VisionConfig | None = None, refresh: bool = False) -> dict:
    """Bounds for `path`, from its `.bounds.json` sidecar, computing and writing one if absent.

    Cached because `detect_usable_range` is a full decode pass (~2-7 s per fixture) and the answer
    never changes for a given file. The sidecar is small tracked text, so a fresh clone that has
    the annotations but not the footage still knows what the trim points were.
    """
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + BOUNDS_SUFFIX)
    if sidecar.exists() and not refresh:
        bounds = json.loads(sidecar.read_text())
    else:
        bounds = detect_usable_range(path)
        sidecar.write_text(json.dumps(bounds, indent=2) + "\n")
    return bounds


class ClipReader:
    """Frames from a recorded clip, as `Frame`s, with the same contract as `capture.ScreenCapture`.

    Iterating yields normalized, tail-trimmed gameplay frames with real timestamps. `index` is the
    frame's position in the file, which at the default `step=1` is also its position in the yielded
    sequence -- so a ground-truth annotation written against a reader with `trim_tail=True` stays
    valid, since trimming only ever removes from the end. Set `trim_tail=False` to see the raw
    file, which is what the tail-detection test needs.

    `step=N` yields every Nth frame and uses `grab()` to advance past the rest, which skips the
    conversion-and-copy that dominates decode. Frame indices stay file-relative, so a stepped read
    and a full read agree about which frame is which -- Phase L relies on that to line up a cheap
    scanning pass with an expensive rendering one.
    """

    def __init__(self, path, cfg: VisionConfig | None = None, normalize: bool = True,
                 trim_tail: bool = True, step: int = 1):
        self.path = Path(path)
        self.cfg = cfg or VisionConfig()
        self.normalize = normalize
        self.trim_tail = trim_tail
        self.step = max(1, int(step))
        self.bounds = load_bounds(self.path, self.cfg)
        self.decode_seconds = 0.0

    @property
    def n_frames(self) -> int:
        """How many frames iteration will yield, after any trimming and stepping."""
        last = self.bounds["usable"][1] if self.trim_tail else self.bounds["n_frames"] - 1
        return last // self.step + 1

    def __len__(self) -> int:
        return self.n_frames

    @property
    def content_box(self) -> tuple[int, int, int, int]:
        return tuple(self.bounds["content_box"])

    @property
    def read_seconds(self) -> float:
        """`sources.FrameSource`'s name for `decode_seconds`. The concrete name stays because it
        says which cost it is; the protocol name exists so a stage can report read time without
        knowing whether it is reading a file or a screen."""
        return self.decode_seconds

    # A clip holds its VideoCapture only for the duration of one `__iter__`, so there is nothing
    # to acquire or release here. The methods exist so that `ClipReader` and `ScreenCapture` --
    # which genuinely does hold an OS handle -- are used identically by callers.
    def __enter__(self) -> "ClipReader":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def __iter__(self):
        last = self.bounds["usable"][1] if self.trim_tail else self.bounds["n_frames"] - 1
        cap = _open(self.path)
        try:
            i = 0
            while i <= last:
                t_start = time.perf_counter()
                # `grab()` for the skipped frames, not `read()` then discard. Both advance the
                # decoder, but `retrieve()` is where the 2436x1126 frame is converted and copied
                # into a numpy array, and that is most of the cost: measured 9.6 ms/frame for
                # read() against 1.8 ms/source-frame at step 6. Phase L reads a clip twice and
                # cannot afford to pay full decode for frames it will not look at.
                skipped = False
                for _ in range(self.step - 1):
                    if not cap.grab():
                        skipped = True
                        break
                if skipped:
                    break
                ok, img = cap.read()
                if not ok:
                    break
                # POS_MSEC is read AFTER the frame, so it reports that frame's presentation time.
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if self.normalize:
                    img = np.ascontiguousarray(
                        normalize_viewport(img, self.content_box,
                                           self.cfg.capture_normalized_aspect)
                    )
                self.decode_seconds += time.perf_counter() - t_start
                yield Frame(image=img, t=t, index=i)
                i += self.step
        finally:
            cap.release()
