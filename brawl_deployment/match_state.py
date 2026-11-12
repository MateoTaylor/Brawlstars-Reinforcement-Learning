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

**The disc is not inert, and the gate holds anyway (measured 2026-11-01).** This paragraph used to
promise inertness. Both heroes have a hypercharge that cannot be unequipped. In a match, a magenta
meter fills around the disc, then the disc turns bright magenta with a bolt. Nothing the agent does
presses it, so it stays lit until the results screen. On a real Bull match
(`test_gate_holds_while_the_hypercharge_fills_and_lights_on_real_footage`, every frame, refined
r 35.9):

    look                          score
    dull, meter filling           0.98 falling to 0.80
    lit                           0.68 median
    the frame it lights           0.24, for 3 frames (100 ms)
    results screen                0.000

The 100 ms dip is the only one inside the match. It is one sample short of `exit_samples`, so
that margin is the thing not to trim. One more trap: the lit disc fits best at r 43, where the
dull one scores under 0.45. A gate first refined on a lit disc, which only happens if the loop is
started mid-match after the hypercharge lit, would never open on the next match. That is one more
reason `scripts/deploy_run.py` must be started before the battle is queued.

**The gate accepts several LOOKS, because the real game's account draws no hypercharge**
(2026-11-04, BRAWL_DEPLOYMENT_DESIGN.md §5.2). Everything above was measured in Nulls Brawl, where
every brawler owns everything. The live target is now the real app on the user's own account,
whose brawler (Shelly, power 3) has no gadget and no hypercharge, so neither disc is drawn. A gate
on the hypercharge alone scored at most 0.31 for the whole of the user's real match
(`bluestacks-starfruit-supernova-west-upper.mp4`) and never opened, and nothing got pressed.

So `match_gate.looks` lists every face of every disc that means "the controls are up", and the
score is the best of them:

    look                    radius (viewport)   when it is drawn
    hypercharge             35.9 (refined)      only when the brawler owns one; listed FIRST
    super, uncharged        34.7                every brawler, empty or recharging
    super, charged          41.7                every brawler, ready to fire

The Super's two faces are two radii, and neither alone holds a match: the charged face scores
0.02-0.10 at the uncharged radius, on both clips that charge it. The policy presses the Super,
which the hypercharge choice existed to avoid, so the cost was measured rather than assumed.
Dragging the Super to aim moves its disc off its centre. In the real match the best look dipped
under threshold five times, for 1, 3, 1, 2 and 3 ticks at 12 Hz, against `exit_samples` 4; the
three longer dips are the player holding a drag, and the two single ticks are shallow (0.36, 0.42). The agent drags for one tick and lifts on the next
(`control.buttons`), so its own dips are shorter. A spent Super does NOT grey out of the gate: it
turns into a dark disc with a recharge arc, which scores like the uncharged face (median 0.761 at
that radius). The history in `control_calibration.json` has a fired Super greying its button "for
the whole recharge", read as the gate going down with it; on these three clips it does not.

The gate opens at 15.7 s on that match, from a lobby maximum of 0.208. On both Nulls clips the
three looks open and close exactly where the hypercharge alone did.

The hypercharge stays first. When it is drawn, as for the Nulls Bull, the gate rests on a disc the
policy never presses, and the Super's looks only add margin. `MatchState.anchor` is that first look.

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
class GateLook:
    """One face of one HUD disc that the match gate accepts as "the controls are up".

    `r` is in DEVICE pixels, like every radius in `control_calibration.json`. None means the
    button's own calibrated radius, so the look follows that button when `deploy_calibrate.py`
    refits it. `face` only names the look in logs.
    """
    button: str
    r: float | None = None
    face: str = ""

    @property
    def label(self) -> str:
        return f"{self.button} ({self.face})" if self.face else self.button


@dataclass(frozen=True)
class Calibration:
    """What `control_calibration.json` holds, resolved. See that file for provenance of each
    number and for which ones are still placeholders."""
    screen: tuple[int, int]
    buttons: dict[str, dict[str, float]]
    gate_looks: tuple[GateLook, ...]
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
        buttons = {k: v for k, v in raw["buttons"].items() if not k.startswith("_")}
        # `anchor` is the single-disc form the file had until 2026-11-04: one look, its radius.
        looks = gate.get("looks") or [{"button": gate["anchor"]}]
        gate_looks = tuple(GateLook(lk["button"], lk.get("r"), lk.get("face", "")) for lk in looks)
        unknown = sorted({lk.button for lk in gate_looks} - set(buttons))
        if unknown:
            raise ValueError(f"match_gate looks name {unknown}, which `buttons` does not have")
        return cls(
            viewport=tuple(viewport) if viewport else calibrated_viewport(),
            screen=tuple(raw["screen"]),
            buttons=buttons,
            gate_looks=gate_looks,
            threshold=gate["threshold"],
            enter_samples=gate["enter_samples"],
            exit_samples=gate["exit_samples"],
            joystick_anchor=(joy["anchor"]["x"], joy["anchor"]["y"]),
            joystick_radius_px=joy["radius_px"],
            joystick_measured=joy.get("measured", False),
        )

    @property
    def gate_anchor(self) -> str:
        """The gate's FIRST look's button: the disc the policy never presses, which the gate
        rests on whenever the brawler draws it. The other looks back it up (module docstring)."""
        return self.gate_looks[0].button

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

    def viewport_look(self, look: GateLook) -> tuple[float, float, float]:
        """`(cx, cy, r)` in viewport pixels for one gate look: its button's centre, at the look's
        own radius when it has one."""
        cx, cy, r = self.viewport_button(look.button)
        if look.r is None:
            return (cx, cy, r)
        sx, sy = self.device_to_viewport
        return (cx, cy, look.r * (sx + sy) / 2.0)

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
    # A whole number of steps each side, so the scan is symmetric about r0 and lands on it.
    # `MatchState._span` hands out spans like 3.49 px, and `r0 - 3.49 + i * 0.5` never does. For a
    # span that is already whole steps (12, 4) this is the arithmetic the scan always had.
    half = math.floor(span / step + 1e-9) * step
    best_r, best_score = r0, -1.0
    # Sized for the WIDEST radius the scan will try, so one ROI serves every step of it.
    gx, gy, lx, ly = _gradient_roi(frame, cx, cy, r0 + half)
    n = int(round(2 * half / step)) + 1
    for i in range(n):
        r = r0 - half + i * step
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

    **The score is the best of `cal.gate_looks`** (module docstring). `last_scores` keeps each
    look's, in the same order, for logs.
    """

    def __init__(self, cal: Calibration, looks: tuple[GateLook, ...] | None = None):
        self.cal = cal
        self.gate_looks = tuple(cal.gate_looks if looks is None else looks)
        if not self.gate_looks:
            raise ValueError("a match gate needs at least one look")
        self._looks = [cal.viewport_look(look) for look in self.gate_looks]
        self.refined = False
        # Labels of the looks the last `refine` found drawn, for the loop's "match started" line.
        self.drawn: tuple[str, ...] = ()
        self._in_match = False
        self._above = 0
        self._below = 0
        self.last_score = 0.0
        self.last_scores: tuple[float, ...] = ()

    @classmethod
    def on(cls, cal: Calibration, button: str) -> "MatchState":
        """A gate on one button alone, at its calibrated radius: the per-disc gates
        `deploy_calibrate.py --probe-gadget` scores beside the real one."""
        return cls(cal, (GateLook(button),))

    def _scores(self, frame) -> tuple[float, ...]:
        """Each look's ring score on `frame`. Looks on one button share one gradient window,
        sized for the widest of them. A single look's window is exactly the one-anchor gate's."""
        widest: dict[tuple[float, float], float] = {}
        for cx, cy, r in self._looks:
            widest[(cx, cy)] = max(r, widest.get((cx, cy), 0.0))
        rois = {c: _gradient_roi(frame, c[0], c[1], r) for c, r in widest.items()}
        out = []
        for cx, cy, r in self._looks:
            gx, gy, lx, ly = rois[(cx, cy)]
            out.append(G.ring_score_at(gx, gy, lx, ly, r))
        return tuple(out)

    def update(self, frame) -> bool:
        """One frame in, current in-match verdict out. `frame` is BGR, at the calibration's
        resolution."""
        self.last_scores = self._scores(frame)
        self.last_score = score = max(self.last_scores)

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
        """Re-fit each look's radius on this frame. Returns the best score any look achieved.

        `frame` must be one the caller knows shows gameplay -- refining on a menu would fit the
        radius to whatever the background happens to make ring-shaped. Raises if no look reaches
        the threshold, because that means the assumption was wrong (a menu frame, the wrong
        resolution, a changed HUD) and silently continuing would leave the gate mis-tuned in a way
        nothing downstream can detect.

        **A look that is not drawn on this frame keeps its stored radius.** The charged Super is
        never drawn at a match's start, and a brawler without a hypercharge never draws one.
        Fitting either to whatever the background holds would move it off the face it exists for.

        **Each look starts from its STORED radius and stays nearer it than its sibling's** (the
        Super's other face, 7 px away). Refining from the last match's fit would let a look
        drift a little more every match. The ±12 px scan of a one-look gate could also jump a look
        onto its sibling's face: this is the "lit disc" trap in the module docstring, one face
        over.
        """
        floor = self.cal.threshold if min_score is None else min_score
        fitted, drawn, best = [], [], (-1.0, None)
        for i, look in enumerate(self.gate_looks):
            cx, cy, r0 = self.cal.viewport_look(look)
            r, score = refine_radius(frame, cx, cy, r0, span=self._span(i))
            if score > best[0]:
                best = (score, (look.label, cx, cy, r))
            if score >= floor:
                fitted.append((cx, cy, r))
                drawn.append(look.label)
            else:
                fitted.append((cx, cy, r0))
        if not drawn:
            label, cx, cy, r = best[1]
            raise ValueError(
                f"refining the gate peaked at {best[0]:.3f} on {label} at ({cx:.1f}, {cy:.1f}) "
                f"r={r:.1f}, below the {floor:.2f} gate. This frame probably is not gameplay, or "
                f"the HUD/resolution has changed."
            )
        self._looks = fitted
        self.drawn = tuple(drawn)
        self.refined = True
        return best[0]

    def _span(self, i: int) -> float:
        """How far look `i`'s radius may move in a refine: 12 px, or half the gap to another look
        on the same button, whichever is less. From the stored radii, so it never shrinks."""
        cx, cy, r = self.cal.viewport_look(self.gate_looks[i])
        span = 12.0
        for j, other in enumerate(self.gate_looks):
            ox, oy, o_r = self.cal.viewport_look(other)
            if j != i and (ox, oy) == (cx, cy) and o_r != r:
                span = min(span, abs(o_r - r) / 2.0)
        return span

    def button_drawn(self, frame, button: str) -> tuple[bool, float]:
        """Whether `button`'s disc is on screen in `frame`, and its best ring score.

        For loadout checks at a match's start: a brawler under power 7 draws no gadget, and
        `loop.py` keeps the policy off a button that is not there. The score is refined within
        4 px of the calibrated radius, not 12, so the check cannot go hunting through the map for
        a ring. One frame is not a verdict: the gadget flashes white just after it comes up, and
        reads 0.19-0.27 through the flash, so the loop looks over a window
        (`loop.GADGET_LOOK_SECONDS`).
        """
        cx, cy, r0 = self.cal.viewport_button(button)
        _, score = refine_radius(frame, cx, cy, r0, span=4.0)
        return score >= self.cal.threshold, score

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
        """`(cx, cy, r)` in viewport pixels of the FIRST look, as `update` scores it: the stored
        centre, and after `refine` the re-fitted radius. For a one-look gate that is the whole
        gate; for the shipped one it is the hypercharge, the disc nothing presses."""
        return tuple(self._looks[0])

    @property
    def looks(self) -> tuple[tuple[str, float, float, float], ...]:
        """`(label, cx, cy, r)` per look, in viewport pixels, as `update` scores them."""
        return tuple((look.label, *xyr) for look, xyr in zip(self.gate_looks, self._looks))
