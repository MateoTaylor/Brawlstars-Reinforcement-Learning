"""Attack and Super as binary taps. See BRAWL_DEPLOYMENT_DESIGN.md 4.4.

**No aiming.** A bare tap fires in the game's default direction, which is what the trained policy
assumes: its action space is `(move_bin, attack)` with `attack in {0, 1, 2}` and carries no aim
component at all (`brawl_sim/config.py`'s `action_nvec = (n_move_bins + 1, 3)`). Dragging from the
button to aim is a real mechanic in the game and deliberately not modelled here -- adding it would
be an action-space change and a retrain, not a control-layer change.

**One decision means at most one attack attempt**, mirroring `env._held`, which zeroes the fire
column on sub-ticks 2..K of a decision. The tap goes in on the first perception tick of the window
and does not repeat for the other four. A held fire bit is a bug: the sim never trained under one,
and `hero.action_mask` promises "you may fire NOW", once.

**The gadget button is never touched.** The policy has no gadget action. It is also, conveniently,
the anchor `match_state.py` watches -- so leaving it alone keeps the agent from perturbing its own
in-match signal.
"""
from .backend import SLOT_TAP

# Action column 1 values, from `hero.decode_action`.
ATTACK_NONE, ATTACK_FIRE, ATTACK_SUPER = 0, 1, 2


class Buttons:
    """Owns SLOT_TAP.

    A tap is a `down` and an `up` on the same slot. They are deliberately NOT sent in one call:
    the game samples touch state on its own frame clock, and a down/up pair inside a single
    `sendevent` batch can land in one sample and be missed entirely. `tap()` sends the down and
    leaves the release to `settle()`, which the loop calls on the next perception tick -- ~50 ms
    later, comfortably longer than one game frame at any sane refresh rate.
    """

    def __init__(self, backend, attack: tuple[float, float], super_: tuple[float, float]):
        self.backend = backend
        self.attack = attack
        self.super_ = super_
        self._held = False

    def tap(self, action: int) -> bool:
        """Press the button for one action value. Returns whether anything was pressed.

        Action masking is the caller's job, not this class's -- `policy.py` applies the same mask
        `MaskablePPO` saw in training, so an uncharged Super never reaches here as
        `ATTACK_SUPER`. Doing it here too would hide a masking bug rather than surface it.
        """
        if action == ATTACK_FIRE:
            point = self.attack
        elif action == ATTACK_SUPER:
            point = self.super_
        else:
            return False
        if self._held:
            # A tap is still down from the previous tick. Release before re-pressing, or the
            # second press is swallowed and the agent silently loses a shot.
            self.settle()
        self.backend.down(SLOT_TAP, *point)
        self._held = True
        return True

    def settle(self) -> None:
        """Release a pending tap. Called once per perception tick by the loop."""
        if not self._held:
            return
        self.backend.up(SLOT_TAP)
        self._held = False

    @property
    def is_held(self) -> bool:
        return self._held
