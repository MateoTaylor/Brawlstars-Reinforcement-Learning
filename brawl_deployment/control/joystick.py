"""The floating movement stick: action bin -> screen contact point, and the held-contact policy
that drives it. See BRAWL_DEPLOYMENT_DESIGN.md 4.2-4.3.

**The stick is not fixed.** It teleports to wherever a contact lands on the left of the screen,
tracks the drag from there, and homes to the bottom-left on release. That is usually described as
the hard part; it is actually what makes this tractable, because it means WE CHOOSE THE ANCHOR.
The geometry is ours as long as we keep one contact down.

**One contact, held for the whole match. Idle is a POSITION, not a release.** The alternative --
tap, drag, release per decision -- pays a touch-down every 250 ms and, worse, homes the stick on
every release so the next decision re-acquires from scratch. `move_bin == 0` therefore moves the
contact back to the anchor and sits in the deadzone. Releasing to express idle is the single most
tempting wrong simplification in this package.

**One radius is enough.** The policy has no partial-speed action -- every non-idle bin is full
speed (`hero.decode_action` normalises `dir_from_bin` output) -- so the stick only ever needs to
sit past saturation. That collapses calibration from a response curve to one scalar.
"""
import math

from .backend import SLOT_MOVE


class Joystick:
    """Owns SLOT_MOVE and the anchor geometry.

    `n_bins` must match `cfg.n_move_bins` in the sim the policy was trained under (16 by default,
    `brawl_sim/config.py`). A mismatch does not raise -- it silently rotates every direction by a
    fraction of a turn, which presents as an agent that walks consistently off-target.
    """

    def __init__(self, backend, anchor: tuple[float, float], radius_px: float,
                 n_bins: int = 16):
        self.backend = backend
        self.anchor = anchor
        self.radius_px = radius_px
        self.n_bins = n_bins
        self._down = False
        self._last_bin: int | None = None

    def contact_point(self, move: int) -> tuple[float, float]:
        """Screen pixel for one action's move component.

        `move` is the raw action value: 0 = idle, 1..n_bins = `dir_from_bin(move - 1)`. That +1
        offset is `hero.decode_action`'s, not ours -- it clamps `move_bin - 1` and zeroes the
        direction where `move_bin == 0`.

        **No y negation.** `brawl_sim`'s world y and the screen's y both increase downward, so
        `dir_from_bin`'s `(cos, sin)` maps to a screen offset unchanged. A minus sign here is a
        silent 100% failure that presents as "the agent walks into walls".
        """
        if move <= 0:
            return self.anchor
        theta = (move - 1) * (2.0 * math.pi / self.n_bins)
        return (self.anchor[0] + self.radius_px * math.cos(theta),
                self.anchor[1] + self.radius_px * math.sin(theta))

    def acquire(self) -> None:
        """Put the contact down at the anchor. Called once when a match starts, and again by the
        loop's recovery path if the contact is ever detected as lost."""
        if self._down:
            return
        self.backend.down(SLOT_MOVE, *self.anchor)
        self._down = True
        self._last_bin = 0

    def apply(self, move: int) -> None:
        """Drive one decision's movement. Acquires the contact if it is not already down.

        Skips the write when the bin is unchanged, which is the common case at 4 Hz -- an agent
        holding a direction produces one `down` and then nothing until it turns. Saves ~5.8 ms of
        device-side process spawns on those ticks and, more usefully, keeps a getevent trace
        readable.
        """
        if not self._down:
            self.acquire()
        if move == self._last_bin:
            return
        self.backend.move(SLOT_MOVE, *self.contact_point(move))
        self._last_bin = move

    def release(self) -> None:
        """Lift the contact. **Match end and failure paths only** -- not idle. See the module
        docstring."""
        if not self._down:
            return
        self.backend.up(SLOT_MOVE)
        self._down = False
        self._last_bin = None

    @property
    def is_down(self) -> bool:
        return self._down
