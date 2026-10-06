"""Are we in a match right now? The safety interlock everything else hangs off.
See BRAWL_DEPLOYMENT_DESIGN.md 5.

**The signal is `brawl_vision.gameplay`, reused rather than reinvented.** Its insight is that the
on-screen controls exist if and only if you can act -- absent on the loading screen, in the lobby,
during the intro swoop, the instant "Defeated" appears, and on the results screen. Nothing else on
screen has that property: the map, the HUD readouts and the brawler sprites all appear in menus
too. `ring_score_at` measures how much of the gradient around a circle points radially, which the
button chrome does and the map underneath, however busy, does not.

MEASURED on tests/fixtures/vision/bluestacks-example-new.mp4 (1920x1080, 30 fps, BlueStacks
fullscreen, Mortis), every 5th frame:

                          the GADGET disc    the SUPER disc
    pre-match / loading   0.000 - 0.198      0.001 - 0.078
    gameplay              0.582 - 0.593      0.776 - 0.823
    post-match            0.000 - 0.198      0.001 - 0.078

Those columns were headed `attack anchor` and `gadget anchor` until 2026-09-22. **Both names were
wrong.** Every button name in `control_calibration.json` sat one disc off: stored `attack` is the
gadget, stored `gadget` is the Super, and the third disc, which nothing pressed, is now stored as
`hypercharge` after its icon. No number above moved; only the names did. Design 5.1 is the full
account, including why `fit_buttons`, `job_tap` and `job_super` could not have caught it. The old
docstring read the left column's 0.585 as the attack button scoring lower than in the iOS footage
(0.649-0.790) because Nulls Brawl draws a bright filled disc where iOS drew a dark annulus. That
disc-versus-annulus effect is real and still governs every score here; the comparison is not,
because the left column is not an attack button.

**The two tables here disagree, and only the second one reproduces.** Scoring the same clip today
at the 2026-09-08 radii: the Super's row comes back exactly, 0.082 menu max and 0.677 play min
against today's 0.083 and 0.675, because its stored radius barely moved (33.2 to 33.3). The
gadget's row does not come back at all -- at r=35.1 it never crosses 0.45 on any frame of the
clip, so it has no gameplay range to report; its stored radius is 30.2. Whatever produced
`0.582 - 0.593` was not this code path at this calibration. Read the first table as history and
the numbers below as the claim. A ring-score table is worth only the code path it was measured
through.

**The gate watches `hypercharge`, the one disc in the cluster the policy never touches**
(2026-09-22, BRAWL_DEPLOYMENT_DESIGN.md §5). That is the property a gate needs, and it is the
property this docstring spent two weeks claiming for whichever disc was called `gadget`. Re-measured
on the same clip through this class's own path, stored centres and radii, 238 samples, scores
partitioned by the threshold the way `test_gate_separates_gameplay_from_menus_on_real_footage`
partitions them:

    disc          menu max   play min   play median
    hypercharge     0.212      0.968       0.985
    super           0.083      0.675       0.770
    gadget          0.336      0.454       0.978

`hypercharge` clears the 0.45 threshold by 2.1x below and 2.2x above. `super` has the cleanest
menus but the policy fires it, and its face changes as it charges: that 0.675 is the exact number
the real-footage gate test failed on for two weeks while the gate sat here. The gadget's in-match
floor is 0.454 against a 0.45 threshold with nobody pressing it, so it is unusable before the press
hazard is even considered.

**Confirmed live 2026-09-22**, `runs/audit/gadget_anchor_trace.json`: a real match, 264 ticks at 12
Hz, one gadget throw. `hypercharge` held 0.991 with a floor of 0.991, zero ticks under threshold
and zero colour movement. The same trace shows why the disjointness is load-bearing rather than
tidy: the real gadget FALSE-EXITED 0.58 s after the throw and stayed under 0.45 for 60 consecutive
ticks, 5.0 s, while the recharge sweep ran. An agent gated on its own gadget stops dead for five
seconds after every throw.

**The guarantee is inertness, and it is conditional on the roster.** Mortis has no hypercharge, so
this button is drawn and never fills, sweeps or animates. That is stronger than "the policy never
presses it", which is all the gate was originally chosen on. It is also contingent: give Mortis a
hypercharge and the disc gains a charge meter that fills during a match, which is exactly what
disqualifies the Super here. Re-measure this gate if the brawler or its unlocks change.

**Hysteresis is asymmetric, because the two errors are not.** Entering gameplay wrongly means
spraying inputs at a menu; exiting wrongly means standing still for a beat. Exit still cannot be
instant -- `gameplay.py` documents that a Super detonating over the button drops the score for a
few frames in the MIDDLE of a match, and those frames are gameplay. So exit needs a short run of
sub-threshold samples, just a shorter one than entry.

**Two coordinate spaces, and this module is where they meet.** The stored anchors are DEVICE
pixels -- the Android screen's own 1920x1080, which is what the touch device addresses and what a
tap target must be in. The frames arriving here are VIEWPORT pixels, 2002x1126, because that is
what `brawl_vision` was calibrated at (`capture.py` explains why the resize targets that and not
the device size). So a button has two positions, and `Calibration` exposes both: `button()` for
where to press, `viewport_button()` for where to look. They were the same numbers while the
fixture happened to be 1080p, which is the sort of coincidence that passes every test and then
breaks the first time the display changes.

**This module never decides what to do about the answer.** It reports; `loop.py` interlocks. That
split is deliberate: the fail-closed path has to be one obvious place.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brawl_vision import gameplay as G

from .capture import calibrated_viewport

CALIBRATION_PATH = Path(__file__).parent / "data" / "control_calibration.json"


@dataclass(frozen=True)
class Calibration:
    """What `control_calibration.json` holds, resolved. See that file for provenance of each
    number and for which ones are still placeholders."""
    screen: tuple[int, int]
    buttons: dict[str, dict[str, float]]
    gate_anchor: str
    threshold: float
    enter_samples: int
    exit_samples: int
    joystick_anchor: tuple[float, float]
    joystick_radius_px: float
    joystick_measured: bool
    viewport: tuple[int, int]

    @classmethod
    def load(cls, path: Path | str = CALIBRATION_PATH,
             viewport: tuple[int, int] | None = None) -> "Calibration":
        raw = json.loads(Path(path).read_text())
        gate, joy = raw["match_gate"], raw["joystick"]
        return cls(
            viewport=tuple(viewport) if viewport else calibrated_viewport(),
            screen=tuple(raw["screen"]),
            buttons={k: v for k, v in raw["buttons"].items() if not k.startswith("_")},
            gate_anchor=gate["anchor"],
            threshold=gate["threshold"],
            enter_samples=gate["enter_samples"],
            exit_samples=gate["exit_samples"],
            joystick_anchor=(joy["anchor"]["x"], joy["anchor"]["y"]),
            joystick_radius_px=joy["radius_px"],
            joystick_measured=joy.get("measured", False),
        )

    def button(self, name: str) -> tuple[float, float]:
        """Where to PRESS, in device pixels. This is what `control.Buttons` wants."""
        b = self.buttons[name]
        return (b["x"], b["y"])

    @property
    def device_to_viewport(self) -> tuple[float, float]:
        """Per-axis scale from device pixels to viewport pixels.

        Per-axis rather than one number on purpose. The two are the same 16:9 game area, so a
        single factor is right to within rounding -- but 2002/1920 = 1.042708 and 1126/1080 =
        1.042593 differ in the fourth decimal, because `normalize_viewport` derives its width from
        its height. Scaling each axis by its own ratio is exact by construction and costs nothing.
        """
        return (self.viewport[0] / self.screen[0], self.viewport[1] / self.screen[1])

    def viewport_button(self, name: str) -> tuple[float, float, float]:
        """Where to LOOK, as `(cx, cy, r)` in viewport pixels. This is what the ring score wants.

        The radius is scaled too, but treat it as a starting point only -- `refine_radius` exists
        because a radius does not transfer across frame sources even at a fixed resolution.
        """
        b = self.buttons[name]
        sx, sy = self.device_to_viewport
        return (b["x"] * sx, b["y"] * sy, b["r"] * (sx + sy) / 2.0)

    def check_frame(self, frame_shape: tuple[int, ...]) -> None:
        """Raise if a frame is not at the calibrated viewport.

        Loud rather than quiet on purpose. `normalize_viewport` crops and trims but never
        rescales, so a raw 1440p grab does not degrade gracefully into these anchors -- it misses
        them by 1.28x, and every downstream reading is confidently wrong while looking fine. The
        check costs nothing and turns a whole class of silent misbehaviour into one message.

        `capture.DeployCapture` already produces frames at this size; this catches the case where
        something upstream hands over a raw grab instead.
        """
        h, w = frame_shape[0], frame_shape[1]
        if (w, h) != self.viewport:
            raise ValueError(
                f"frame is {w}x{h} but the vision stack is calibrated for "
                f"{self.viewport[0]}x{self.viewport[1]}. Feed frames through "
                f"brawl_deployment.capture.DeployCapture (or capture.to_viewport for an offline "
                f"clip). See BRAWL_DEPLOYMENT_DESIGN.md 9.3."
            )


# Pixels of slack around a scored ring before its gradient ROI is cut. `_gradients` blurs 5x5 and
# then Sobels 3x3, so a sample is influenced by at most 3 px around it; 24 covers that many times
# over and leaves room for `refine_radius` to scan outward. MEASURED: gradients inside the window
# match the full-frame ones to 0.00e+00 at every margin from 4 px up.
_ROI_MARGIN_PX = 24


def _gradient_roi(frame, cx: float, cy: float, r: float, margin: float = _ROI_MARGIN_PX):
    """Gradients of a small window around one ring -> `(gx, gy, cx_local, cy_local)`.

    **The gate scores one ~35 px circle, and computing `_gradients` over all 2002x1126 pixels to
    do it costs 5.50 ms of a 50 ms frame budget.** The same score off a 116x116 window costs
    0.034 ms -- 160x less. Every pixel the ring samples is interior to the window, so the blur and
    Sobel kernels never reach the cut and the gradients there are bit-identical.

    **The SCORE, however, is not always bit-identical, and the reason is worth knowing before
    someone "fixes" it.** `ring_score_at` truncates its sample coordinates with `.astype(int)`.
    Translating the centre by the ROI offset perturbs `cx + r*cos(theta)` in its last bits, so at
    an angle where that value lands on an exact integer -- theta = 3*pi/2 on a whole-pixel centre,
    say -- truncation can fall one pixel differently. Measured worst case: 4 of 180 samples move,
    changing the score by 6.7e-04 on uniform noise and 8.8e-05 on a real frame, and by exactly
    0.00e+00 at both real button anchors. Against a 0.45 gate with a 10x margin that is nothing.

    `gameplay.py` computes full-frame gradients because it is scanning for buttons whose location
    it does not know. Here the location is calibrated, so the search is already over.
    """
    h, w = frame.shape[:2]
    x0 = int(max(0, math.floor(cx - r - margin)))
    x1 = int(min(w, math.ceil(cx + r + margin) + 1))
    y0 = int(max(0, math.floor(cy - r - margin)))
    y1 = int(min(h, math.ceil(cy + r + margin) + 1))
    roi = np.ascontiguousarray(frame[y0:y1, x0:x1])
    gx, gy = G._gradients(roi)
    return gx, gy, cx - x0, cy - y0


def refine_radius(frame, cx: float, cy: float, r0: float,
                  span: float = 12.0, step: float = 0.5) -> tuple[float, float]:
    """Re-fit one button's radius on THIS frame by scanning for the peak `ring_score_at`.
    Returns `(radius, score)`.

    **Why this exists: `ring_score_at` is sharply radius-sensitive, and a radius measured on one
    frame source does not transfer to another.** Measured, same buttons, same screen resolution,
    same fixed HUD:

        source                          the SUPER button   score at the OTHER source's radius
        OBS recording, temporal median  r = 33.2                       --
        raw ADB framebuffer grab        r = 39.9           0.083  (vs 0.984 at its own radius)

    A 6.7 px radius error costs 0.90 of score and drops the gate from "clearly in a match" to
    "clearly not". The centres, by contrast, agree to 1-3 px across both sources -- so a stored
    calibration should be trusted for WHERE a button is and re-fit for HOW BIG it looks.

    `gameplay.py` already flagged the sensitivity in passing ("Hough's radius estimate is loose
    enough to cost 0.3 of score"); this is the same effect, larger, and it bites across sources
    rather than across Hough candidates. The encoded video's temporal median softens the button
    edge and pulls the best-fit annulus inward; the raw framebuffer keeps it crisp.
    """
    best_r, best_score = r0, -1.0
    # Sized for the WIDEST radius the scan will try, so one ROI serves every step of it.
    gx, gy, lx, ly = _gradient_roi(frame, cx, cy, r0 + span)
    n = int(2 * span / step) + 1
    for i in range(n):
        r = r0 - span + i * step
        if r < 8.0:
            continue
        score = G.ring_score_at(gx, gy, lx, ly, r)
        if score > best_score:
            best_r, best_score = r, score
    return best_r, best_score


class MatchState:
    """Rolling in-match detector. Feed it every perception tick; read `in_match`.

    Starts OUT of a match, which is the safe initial state: it takes `enter_samples` consecutive
    above-threshold frames to begin emitting input, so a loop started on a menu stays quiet.

    **Call `refine(frame)` once at startup, on a frame known to be in a match.** Without it the
    gate uses the stored radius, which is only correct for the source the calibration was measured
    on -- see `refine_radius`. `loop.py` does this during its warm-up.
    """

    def __init__(self, cal: Calibration):
        self.cal = cal
        self._anchor = cal.viewport_button(cal.gate_anchor)
        self.refined = False
        self._in_match = False
        self._above = 0
        self._below = 0
        self.last_score = 0.0

    def update(self, frame) -> bool:
        """One frame in, current in-match verdict out. `frame` is BGR, at the calibration's
        resolution."""
        cx, cy, r = self._anchor
        gx, gy, lx, ly = _gradient_roi(frame, cx, cy, r)
        self.last_score = score = G.ring_score_at(gx, gy, lx, ly, r)

        if score >= self.cal.threshold:
            self._above += 1
            self._below = 0
        else:
            self._below += 1
            self._above = 0

        if not self._in_match and self._above >= self.cal.enter_samples:
            self._in_match = True
        elif self._in_match and self._below >= self.cal.exit_samples:
            self._in_match = False
        return self._in_match

    def refine(self, frame, min_score: float | None = None) -> float:
        """Re-fit the gate anchor's radius on this frame. Returns the achieved score.

        `frame` must be one the caller knows shows gameplay -- refining on a menu would fit the
        radius to whatever the background happens to make ring-shaped. Raises if the best
        achievable score is still below threshold, because that means the assumption was wrong (a
        menu frame, the wrong resolution, a changed HUD) and silently continuing would leave the
        gate mis-tuned in a way nothing downstream can detect.
        """
        cx, cy, r0 = self._anchor
        r, score = refine_radius(frame, cx, cy, r0)
        floor = self.cal.threshold if min_score is None else min_score
        if score < floor:
            raise ValueError(
                f"refining the {self.cal.gate_anchor} anchor at ({cx:.1f}, {cy:.1f}) peaked at "
                f"{score:.3f} (r={r:.1f}), below the {floor:.2f} gate. This frame probably is not "
                f"gameplay, or the HUD/resolution has changed."
            )
        self._anchor = (cx, cy, r)
        self.refined = True
        return score

    def force_exit(self) -> None:
        """Drop out of the match without waiting for the hysteresis to run down. For the loop's
        other fail-closed triggers -- focus lost, capture stall, policy exception -- which know
        something the ring score does not."""
        self._in_match = False
        self._above = 0

    @property
    def in_match(self) -> bool:
        return self._in_match

    @property
    def anchor(self) -> tuple[float, float, float]:
        """`(cx, cy, r)` in viewport pixels, as `update` scores it: the stored centre, and after
        `refine` the re-fitted radius."""
        return self._anchor
