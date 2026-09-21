"""Attack and Super as aimed drags. See BRAWL_DEPLOYMENT_DESIGN.md 4.4.

**Attack is an AREA, not a button, and this is the fact most likely to be un-learned later.**
Per the operator: any tap on the right side of the screen fires, as long as it does not overlap
the Super or gadget buttons. The attack stick FLOATS to wherever the finger lands, exactly like
the movement stick -- so `self.attack` is a point with good clearance, not an attempt to hit a
sprite, and re-centring it on the button graphic would be a downgrade rather than a fix.

That distinction is what the Training Grounds run actually measured. The calibrated centre sat
under a green chevron button that exists only in that venue; 2 of 3 taps consumed no ammo. The
anchor was not mis-aimed -- it was aimed at a real button that swallowed the touch. Choosing the
touch point by CLEARANCE is immune to that whole class of failure, including HUD-layout drift.

**Aimed along the move direction, never a bare tap.** A bare tap AUTO-AIMS in the real game: it
sends Mortis at the nearest enemy, which is not what the policy trained on. The sim's
`hero.start_dash` sends him along the decision's `move_dir`, or along `facing` when the move bin is
idle (`action.dash_on_idle: facing`), and the super goes along `move_dir` too. So a press here is
the attack stick worked by hand, the way the movement stick is: a contact goes down on the origin,
drags `aim_radius_px` along the bearing, and lifts. **The lift is what fires**, as it is for a
thumb. The bearing comes from `ShadowHero.attack_bearing`, which reads the same state the shadow's
`_start_dash` does, so the device and the shadow agree on the direction by construction.

This is not an action-space change. The policy's action is still `(move_bin, attack)`, and what
deployment emits is `attack in {0, 1, 2}`: `brawl_sim/config.py`'s `action_nvec` is
`(n_move_bins + 1, 4)` since SIM_OVERHAUL Step G3, and the fourth value, the gadget, stays masked
until Step G5. The aim is not a new choice, it is the one the policy already made with its move bin.

**One step per perception tick: down, drag, lift.** Never two in one call. The game samples touch
state on its own frame clock, and two events inside one device write can land in one sample.
A down and a drag in the same sample put the floating stick's centre at the dragged point, which
is a zero-length drag -- a tap, auto-aimed. So `press()` sends only the down, and `settle()`, which
the loop calls at the top of every tick, sends the drag on the next tick and the lift on the one
after. That is `PRESS_TICKS` = 3, and at the shipped 12 Hz a decision is exactly 3 ticks: the lift
lands on the last tick of the window, ~167 ms after the decision, and the next press has a clean
tick behind it. `config.validate` refuses a tick rate with fewer ticks per decision than that.

**One decision means at most one attack attempt**, mirroring `env._held`, which zeroes the fire
column on sub-ticks 2..K of a decision. The press goes in on the first perception tick of the
window and does not repeat. A held fire bit is a bug: the sim never trained under one, and
`hero.action_mask` promises "you may fire NOW", once.

**The gadget button is never touched.** The policy has no gadget action. It is also, conveniently,
the anchor `match_state.py` watches -- so leaving it alone keeps the agent from perturbing its own
in-match signal.
"""
import math

from .backend import SLOT_TAP

# Action column 1 values, from `hero.decode_action`.
ATTACK_NONE, ATTACK_FIRE, ATTACK_SUPER = 0, 1, 2

# Perception ticks one press occupies: the down, the drag, the lift. See the module docstring.
PRESS_TICKS = 3

# The default for `control.aim_radius_px` (`configs/deployment.yaml`, which carries the reasoning).
# The loop always passes the configured value; this default is for callers that construct a
# `Buttons` by hand.
AIM_RADIUS_PX = 75.0

_IDLE, _DOWN, _AIMED = "idle", "down", "aimed"


class Buttons:
    """Owns SLOT_TAP.

    A press is a `down` on the origin, a `move` to the aim point and an `up`, one per perception
    tick. The origin is `attack` for a fire and `super_` for a super -- the super is a FIXED button,
    unlike the attack stick, so its drag starts on the button centre. Direction only: Mortis's
    dash and his super both travel a fixed distance, so how far the drag goes past the game's
    aim deadzone changes nothing, and one radius serves both.
    """

    def __init__(self, backend, attack: tuple[float, float], super_: tuple[float, float],
                 aim_radius_px: float = AIM_RADIUS_PX):
        self.backend = backend
        self.attack = attack
        self.super_ = super_
        self.aim_radius_px = float(aim_radius_px)
        self._stage = _IDLE
        self._aim: tuple[float, float] | None = None

    def origin(self, action: int) -> tuple[float, float] | None:
        """Where a press for this action goes down. None for `ATTACK_NONE`."""
        if action == ATTACK_FIRE:
            return self.attack
        if action == ATTACK_SUPER:
            return self.super_
        return None

    def aim_point(self, action: int, bearing: float) -> tuple[float, float]:
        """Where the drag for this action ends. `bearing` is radians, in the sim's frame.

        **No y negation**, for the reason `Joystick.contact_point` gives: `brawl_sim`'s world y and
        the screen's y both increase downward, so `(cos, sin)` maps to a screen offset unchanged.
        A minus sign here dashes every attack mirrored across the horizontal.
        """
        origin = self.origin(action)
        if origin is None:
            raise ValueError(f"action {action} presses nothing, so it has no aim point")
        return (origin[0] + self.aim_radius_px * math.cos(bearing),
                origin[1] + self.aim_radius_px * math.sin(bearing))

    def require_on_screen(self, screen: tuple[int, int]) -> None:
        """Raise unless every bearing's aim point, for both buttons, lands on the screen.

        The ADB backend CLAMPS an off-screen point to the edge rather than rejecting it, which for
        a drag is worse than a failure: the lift lands somewhere real, in the wrong direction. The
        super button sits low (81 px of room below it at 1920x1080), so a radius that works for
        the attack stick can still fail here, and only for attacks aimed downward.
        """
        w, h = screen
        r = self.aim_radius_px
        for name, (x, y) in (("attack", self.attack), ("super", self.super_)):
            room = min(x, y, (w - 1) - x, (h - 1) - y)
            if room < r:
                raise ValueError(
                    f"control.aim_radius_px={r:g} drags the {name} press off-screen: its origin "
                    f"({x:.0f}, {y:.0f}) is {room:.0f} px from the edge of a {w}x{h} screen. The "
                    f"backend clamps, so the lift would land on the edge, aimed the wrong way.")

    def press(self, action: int, bearing: float) -> bool:
        """Start one aimed press. Returns whether anything was pressed.

        Only the down goes out here; `settle()` sends the drag and then the lift. A press that
        arrives while an earlier one is still in flight finishes that one first -- dragged and
        lifted, so its shot still goes where it was aimed -- rather than swallowing it. The loop
        never gets there at a legal tick rate; `verify_tap` and a caller pressing by hand can.

        Action masking is the caller's job, not this class's -- `policy.py` applies the same mask
        `MaskablePPO` saw in training, and the loop presses what `ShadowHero.act` modelled, so an
        uncharged Super never reaches here as `ATTACK_SUPER`. Doing it here too would hide a
        masking bug rather than surface it.
        """
        origin = self.origin(action)
        if origin is None:
            return False
        if self._stage is not _IDLE:
            self._finish()
        aim = self.aim_point(action, bearing)
        self.backend.down(SLOT_TAP, *origin)
        self._aim = aim
        self._stage = _DOWN
        return True

    def settle(self) -> None:
        """Advance a press by one step. Called once per perception tick by the loop: the tick
        after the down drags to the aim point, and the tick after that lifts, which fires."""
        if self._stage is _DOWN:
            self.backend.move(SLOT_TAP, *self._aim)
            self._stage = _AIMED
        elif self._stage is _AIMED:
            self._lift()

    def release(self) -> None:
        """Lift the contact now, wherever the press has got to. **Failure and match-end paths**
        (`Controls.release_all`), not the normal end of a press -- that is `settle()`'s.

        A lift is a release as far as the game knows, so a press cut short here may still fire:
        along the aim if it had been dragged, auto-aimed if it had not. That is the price of never
        leaving a contact down, and a stray swing at a match's end is not worth a second touch
        path to avoid.
        """
        if self._stage is not _IDLE:
            self._lift()

    def _finish(self) -> None:
        if self._stage is _DOWN:
            self.backend.move(SLOT_TAP, *self._aim)
        self._lift()

    def _lift(self) -> None:
        self.backend.up(SLOT_TAP)
        self._stage = _IDLE
        self._aim = None

    @property
    def is_held(self) -> bool:
        """Whether a press is in flight: down, or dragged and not yet lifted."""
        return self._stage is not _IDLE
