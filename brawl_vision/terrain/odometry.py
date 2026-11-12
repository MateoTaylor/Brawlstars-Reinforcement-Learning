"""Frame-to-frame camera translation, in tiles. See Terrain_Perception_Build_Plan.md Phase F.

This is what tells Phase I *where* a newly classified tile belongs, without ever identifying which
map is being played. The camera angle never changes, so consecutive rectified frames are related
by a pure translation and `cv2.phaseCorrelate` is exactly the right tool -- sub-pixel, FFT-cheap,
no rotation to solve.

**One correlation over the whole patch does not work, and the failure is not subtle.** Measured on
`counted_walking.mp4`: a single 823x581 window reports 2.8 tiles of camera travel across a stretch
where the background is provably identical frame to frame -- the camera never moved at all. Phase
correlation returns the single strongest peak, and in a frame containing large moving sprites
(bots, the local player, damage numbers) that peak can be the *foreground*. The response value
does not catch it: those estimates came back at r = 0.92-0.98, indistinguishable from a good one.

So the estimator correlates a GRID of windows independently and takes the **median**. Terrain
covers most of the patch, so most windows see the true camera motion and outvote the ones sitting
on a sprite. On the same stretch this returns 0.01 tiles, and every window geometry tried agrees
on the real leg to within 0.2 tiles (see the plan's Phase F notes for the table).

**Window agreement is the quality signal worth trusting, not the correlation response.** Windows
that disagree mean the scene is not undergoing one rigid translation -- a camera cut, or a large
occluder. The gate is the **inlier ratio**: the share of windows landing within
`max_disagreement_tiles` of the median. Median *deviation* was tried first and is too weak,
because a coherent split -- half the patch sliding one way, half the other -- puts the median
inside one group and drives the median deviation to zero. A ratio sees a bare majority for what
it is.

**A frame that merely looks doubtful is NOT a cut, and conflating the two is expensive.**
Measured across every fixture, the inlier ratio is 1.0 at the median and >=0.90 at the 1st
percentile, but one or two frames per thousand dip to ~0.64 when a large effect (an explosion, a
super, the gas edge) covers part of the patch. Those frames are still tracking correctly -- the
median over the surviving windows is the real camera motion. Calling them cuts would reset the
world frame once or twice per match, which is far worse for Phase I than a slightly noisy step. So
there are three outcomes: `ok` (trust it, deposit votes), `uncertain` (keep integrating, do not
deposit votes), and `lost` (a genuine discontinuity: stop, and open a new segment).

**Position is camera-relative and resets on a cut.** Integrating across a discontinuity (death,
respawn, the match-start fly-in) corrupts the whole accumulated map in one frame, so a cut instead
bumps `segment`. Phase I must treat votes from different segments as belonging to different world
frames -- there is no defined offset between them until loop closure exists.

**Screen-static windows can take the median, and the inlier gate cannot see it** (2026-11-05,
MAP_ONBOARDING.md's Truth section). The camera follows the hero, so whatever is drawn in screen
space reads exactly zero shift: the hero, its name tag, the "SHOWDOWN!" text at two brawlers
left, a "Defeated" popup, a recording's mouse cursor. A slow walk moves under 0.25 tiles per
12 Hz tick, which keeps zero inside `max_disagreement_tiles` of the true motion. So when half the
windows or more land on such content, the median is their zero, or the average of the two groups
when they tie. With three of eight, the per-axis median can still land halfway. Every window
still "agrees", and the frame comes back `ok` with the camera frozen or slowed. On
bluestacks-truth-west-upper's banner, 5 of 8 windows read zero for 3 s, and the odometry stood
still and fell 7.5 tiles behind the camera. On the same clip, the halfway median drifted it
1.6 tiles once and 1.0 tile once more. Window statistics alone cannot tell this from the moving
sprite above. It is the same pattern, a still majority and a coherent moving minority, with the
opposite answer.

The same windows cost trust when they are a minority. A brisk walk passes
`max_disagreement_tiles` per tick, so the hero and its name tag count against a correct median.
On the Truth replay, 17% of in-match ticks came back `uncertain` for that reason and held their
action.

So when the windows split into still ones and a coherent moving group, and the still ones change
the answer, the PIXELS decide. Every pixel of the core is compared under both candidate shifts. A
pixel whose two residuals differ by more than `overlay_grey` votes for the shift that explains it
better. The world fills most of the view, and an overlay or a sprite covers only part of it. So
the moving windows win only when they explain a share above `overlay_share` of the deciding
pixels. They then carry the estimate, and the still windows are set aside from the inlier ratio
as screen furniture rather than counted against it. Measured against image registration at 12 Hz,
this cut the Truth clip's error from 20.7 to 4.9 tiles, summed over the registered 100-frame spans
that hold no cut. It made no span worse. Replayed through the deploy loop, the clip's in-match
`uncertain` ticks fell from 160 to 33 of 932, and the localizer's three corrections (the two
drifts and the banner refix) did not happen. A frame whose still windows win the vote, or never
come into question, is estimated and gated exactly as before.
`scripts/probes/odometry_overlay/` runs the shipped rule over recorded frame pairs.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from ..camera import RectifyPlan
from ..config import VisionConfig

# Erosion applied to the plan's `valid` mask before windows are placed. The boundary between world
# and border fill is a hard step that correlates far more strongly than anything in the scene, so a
# window straddling it locks to zero motion.
_CORE_ERODE_PX = 9

# The overlay vote (module docstring). It needs at least this many moving windows agreeing with
# each other: one window alone is a sprite as often as not, and the vote would only overrule it.
_MIN_MOVERS = 2
# A moving window belongs to the moving group within `odometry_still_tiles` plus this share of the
# group's own speed. A fast dash spreads the windows' estimates more than a walk does.
_GROUP_SLACK_FRAC = 0.1
# A vote decided by fewer pixels than this share of the core is no vote. The frame is then
# estimated as if the windows had not split. Real splits are decided by 80k-400k pixels, 10-45%
# of the core.
_MIN_DECISIVE_FRAC = 0.01


@dataclass(frozen=True)
class OdometryResult:
    """One frame's estimate. `delta_tiles` is CAMERA motion, already sign-flipped from the image
    shift the correlator returns: content sliding up means the camera moved down."""
    delta_tiles: tuple[float, float]
    position_tiles: tuple[float, float]
    response: float                # median per-window correlation response
    agreement_tiles: float         # median distance from the estimate, for diagnostics
    inlier_ratio: float            # share of windows within max_disagreement_tiles of the estimate
    n_windows: int                 # windows that cleared the response floor
    status: str                    # "init" | "ok" | "uncertain" | "lost"
    segment: int
    # Set when the windows split into still ones and a moving group, and the still ones would
    # change the answer (`Odometry._split`). It is the share of the deciding pixels the moving
    # group explained better. Above `odometry_overlay_share` the moving group carried the
    # estimate, and the still windows were left out of `agreement_tiles` and `inlier_ratio`.
    # None when the windows did not split, or too few pixels decided.
    overlay_vote: float | None = None

    @property
    def ok(self) -> bool:
        """Confident enough to deposit terrain votes against. `uncertain` frames still advance
        the position -- they are tracked, just not trusted as evidence."""
        return self.status == "ok"

    @property
    def tracking(self) -> bool:
        """Position is still meaningful in the current segment's frame."""
        return self.status in ("ok", "uncertain")


class Odometry:
    """Integrates camera translation across a clip or a live session.

    Stateful by necessity -- it holds the previous frame -- so one instance per game.
    """

    def __init__(self, plan: RectifyPlan, cfg: VisionConfig | None = None):
        self.plan = plan
        self.cfg = cfg or VisionConfig()
        self.position = np.zeros(2, np.float64)
        self.segment = 0
        self._prev: np.ndarray | None = None

        win = int(self.cfg.odometry_window_px)
        stride = int(self.cfg.odometry_stride_px)
        core = cv2.erode(plan.valid.astype(np.uint8),
                         np.ones((_CORE_ERODE_PX, _CORE_ERODE_PX), np.uint8)).astype(bool)
        w, h = plan.size_px
        self.spots = [(x, y)
                      for y in range(0, h - win + 1, stride)
                      for x in range(0, w - win + 1, stride)
                      if core[y:y + win, x:x + win].all()]
        if len(self.spots) < self.cfg.odometry_min_windows:
            raise ValueError(
                f"only {len(self.spots)} correlation windows of {win}px fit inside the rectified "
                f"patch's valid region, below odometry.min_windows="
                f"{self.cfg.odometry_min_windows}. Lower odometry.window_px or stride_px."
            )
        self._win = win
        self._hann = cv2.createHanningWindow((win, win), cv2.CV_32F)
        self._core = core
        self._core_u8 = core.astype(np.uint8)
        self._core_px = int(core.sum())

    # -- the estimator -------------------------------------------------------

    def _shifts(self, a: np.ndarray, b: np.ndarray):
        """Per-window image shifts that cleared the response floor, plus their responses."""
        win, floor = self._win, self.cfg.odometry_min_response
        out, resp = [], []
        for x, y in self.spots:
            pa = np.ascontiguousarray(a[y:y + win, x:x + win])
            pb = np.ascontiguousarray(b[y:y + win, x:x + win])
            (dx, dy), r = cv2.phaseCorrelate(pa, pb, self._hann)
            if r >= floor:
                out.append((dx, dy))
                resp.append(r)
        return np.array(out, np.float64).reshape(-1, 2), np.array(resp, np.float64)

    def _split(self, shifts: np.ndarray, median: np.ndarray):
        """`(still, c0, c1)` when the windows split into still ones and a coherent moving group,
        and the still ones change the answer, else None.

        They change it in two ways. The median may not sit with the moving group: the banner
        freeze. Or the median does sit there, but the still windows alone make the frame
        `uncertain`. At 12 Hz a walk can pass `max_disagreement_tiles` per tick, and the hero and
        its name tag are then outliers on every tick. `still` masks the windows reading at most
        `odometry_still_tiles`; `c0` is their median shift and `c1` the moving group's."""
        ppt = self.plan.pixels_per_tile
        eps = self.cfg.odometry_still_tiles
        still = np.linalg.norm(shifts, axis=1) / ppt <= eps
        moving = ~still
        if not still.any() or moving.sum() < _MIN_MOVERS:
            return None
        v = np.median(shifts[moving], axis=0)
        slack = eps + _GROUP_SLACK_FRAC * np.linalg.norm(v) / ppt
        group = moving & (np.linalg.norm(shifts - v, axis=1) / ppt <= slack)
        if group.sum() < _MIN_MOVERS:
            return None
        if np.linalg.norm(median - v) / ppt <= slack:
            dev = np.linalg.norm(shifts - median, axis=1) / ppt
            agree = dev <= self.cfg.odometry_max_disagreement_tiles
            gate = self.cfg.odometry_min_agreement_ratio
            if agree.mean() >= gate or agree[moving].mean() < gate:
                return None            # trusted already, or doubtful whatever the still ones are
        return still, np.median(shifts[still], axis=0), np.median(shifts[group], axis=0)

    def _vote(self, a: np.ndarray, b: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> float | None:
        """Share of the core's deciding pixels that image shift `c1` explains better than `c0`.

        A candidate d explains pixel x when b(x + d) matches a(x), the convention
        `cv2.phaseCorrelate(a, b)` reports d in. A pixel decides only where the two residuals
        differ by more than `odometry_overlay_grey`, i.e. where it has texture along c1 - c0, and
        only where both shifted positions fall back inside the core. None below
        `_MIN_DECISIVE_FRAC` of the core."""
        h, w = a.shape
        valid = self._core.copy()
        residual = []
        for d in (c0, c1):
            m = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
            back = cv2.warpAffine(b, m, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE)
            residual.append(np.abs(a - back))
            valid &= cv2.warpAffine(self._core_u8, m, (w, h),
                                    flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        r0, r1 = residual
        deciding = valid & (np.abs(r0 - r1) > self.cfg.odometry_overlay_grey)
        n = int(np.count_nonzero(deciding))
        if n < _MIN_DECISIVE_FRAC * self._core_px:
            return None
        return float(np.count_nonzero(deciding & (r1 < r0))) / n

    def update(self, frame: np.ndarray) -> OdometryResult:
        """Feed one RECTIFIED frame (BGR or grayscale) and get the camera's motion since the last.

        Three outcomes. `ok` and `uncertain` both advance `position`; they differ only in whether
        the estimate is trustworthy enough for Phase I to vote on. `lost` is a genuine
        discontinuity -- too few windows correlated at all, or the implied motion is physically
        impossible -- and leaves `position` untouched while opening a new `segment`.

        A lost frame still becomes the new reference. Keeping the pre-cut frame would leave every
        later frame comparing against a stale world, i.e. permanently lost after one cut.
        """
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32, copy=False)
        if gray.shape[:2] != (self.plan.size_px[1], self.plan.size_px[0]):
            raise ValueError(
                f"odometry expects the rectified patch "
                f"({self.plan.size_px[0]}x{self.plan.size_px[1]}), got "
                f"{gray.shape[1]}x{gray.shape[0]}. Run RectifyPlan.rectify first."
            )

        if self._prev is None:
            self._prev = gray
            return self._result((0.0, 0.0), 0.0, 0.0, 0.0, 0, "init")

        prev, self._prev = self._prev, gray
        shifts, resp = self._shifts(prev, gray)

        if len(shifts) < self.cfg.odometry_min_windows:
            return self._cut(float(np.median(resp)) if len(resp) else 0.0, 0.0, 0.0,
                             len(shifts))

        ppt = self.plan.pixels_per_tile
        median = np.median(shifts, axis=0)
        estimate, used, vote = median, shifts, None
        split = self._split(shifts, median) if self.cfg.odometry_overlay_share < 1.0 else None
        if split is not None:
            still, c0, c1 = split
            vote = self._vote(prev, gray, c0, c1)
            if vote is not None and vote > self.cfg.odometry_overlay_share:
                # The still windows sit on screen furniture: they say nothing about the camera.
                estimate, used = c1, shifts[~still]
        dev = np.linalg.norm(used - estimate, axis=1) / ppt
        agreement = float(np.median(dev))
        inliers = float(np.mean(dev <= self.cfg.odometry_max_disagreement_tiles))
        delta = -estimate / ppt                          # image shift -> camera motion
        r = float(np.median(resp))

        # Order matters: an implausible shift is a discontinuity whatever the windows agreed on,
        # while mere disagreement is a confidence statement about an otherwise usable estimate.
        if np.hypot(*delta) > self.cfg.odometry_max_shift_tiles:
            return self._cut(r, agreement, inliers, len(shifts), vote)

        self.position = self.position + delta
        status = "ok" if inliers >= self.cfg.odometry_min_agreement_ratio else "uncertain"
        return self._result(tuple(delta), r, agreement, inliers, len(shifts), status, vote)

    # -- bookkeeping ---------------------------------------------------------

    def _cut(self, response, agreement, inliers, n, vote=None) -> OdometryResult:
        self.segment += 1
        return self._result((0.0, 0.0), response, agreement, inliers, n, "lost", vote)

    def _result(self, delta, response, agreement, inliers, n, status, vote=None) -> OdometryResult:
        return OdometryResult(
            delta_tiles=(float(delta[0]), float(delta[1])),
            position_tiles=(float(self.position[0]), float(self.position[1])),
            response=response, agreement_tiles=agreement, inlier_ratio=inliers,
            n_windows=n, status=status, segment=self.segment, overlay_vote=vote,
        )

    def reset(self) -> None:
        """Forget the reference frame and the accumulated position, keeping the segment counter
        moving so downstream votes are never merged across the gap."""
        self.position = np.zeros(2, np.float64)
        self.segment += 1
        self._prev = None
