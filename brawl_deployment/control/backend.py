"""What every input backend must do, and a no-op one for dry runs.

The protocol is deliberately about CONTACTS, not about taps and swipes. `input tap` / `input swipe`
shaped APIs cannot express the thing this package is built around -- a contact held across many
decisions while other contacts come and go beside it (BRAWL_DEPLOYMENT_DESIGN.md 4.2) -- so the
vocabulary here is `down` / `move` / `up` on an explicit slot.

Slots are the caller's to assign and are stable for the life of a contact. `control/joystick.py`
owns SLOT_MOVE, `control/buttons.py` owns SLOT_TAP; nothing else should invent one without saying
so here.
"""
from typing import Protocol


# Touch slots. The device advertises 16 (ABS_MT_SLOT max 15), so this is not a scarce resource --
# they are split by ROLE rather than pooled, which makes a leaked contact obvious in a getevent
# trace rather than anonymous.
SLOT_MOVE = 0       # the held movement contact, down for the whole match
SLOT_TAP = 1        # attack / super taps, down for a few ms at a time


class InputBackend(Protocol):
    """A touch injector.

    Implementations must be safe to `close()` twice and must release every contact they hold when
    closed -- the fail-closed path in `loop.py` calls `release_all()` and then `close()`, and both
    run in exception handlers where the backend may already be half dead.
    """

    def down(self, slot: int, x: float, y: float) -> None:
        """Place a new contact at screen pixel (x, y). Idempotent per slot is NOT assumed: calling
        `down` on a slot that already holds a contact is a caller bug, not a re-position. Use
        `move`."""

    def move(self, slot: int, x: float, y: float) -> None:
        """Move an existing contact. The steady-state operation -- one of these per decision drives
        the joystick (design 4.2)."""

    def up(self, slot: int) -> None:
        """Release one contact. Safe to call on a slot that holds nothing."""

    def release_all(self) -> None:
        """Release every contact this backend believes it holds. **This is the fail-closed
        primitive** -- design 8 routes every failure here, so it must not raise."""

    def close(self) -> None:
        """Tear down the transport. Implies `release_all()`."""

    @property
    def is_alive(self) -> bool:
        """False once the transport has died. `loop.py` checks this rather than assuming a write
        that did not raise actually landed."""

    @property
    def injects(self) -> bool:
        """Does this backend actually reach the device?

        False only for `NullBackend`. It exists because the shadow's ammo canary compares what the
        game did against what our actions should have caused -- and when nothing was injected, the
        canary is guaranteed to trip, correctly, on a link we broke on purpose. Without this the
        first live dry run burned 4 of its 5 resyncs in 22 seconds and would have stopped itself.

        Deliberately a property OF THE BACKEND, not a dry-run flag on the loop: the same reasoning
        that swaps the backend instead of adding a boolean (`deploy_run.py`) applies here."""


class NullBackend:
    """Accepts everything, injects nothing, remembers what it was told.

    For dry runs and for tests that exercise the decision path without an emulator attached --
    which is most of them, since the test suite must not require BlueStacks to be running.
    `contacts` is the assertion surface.
    """

    def __init__(self) -> None:
        self.contacts: dict[int, tuple[float, float]] = {}
        self.log: list[tuple] = []
        self._closed = False

    def down(self, slot: int, x: float, y: float) -> None:
        self.contacts[slot] = (x, y)
        self.log.append(("down", slot, x, y))

    def move(self, slot: int, x: float, y: float) -> None:
        self.contacts[slot] = (x, y)
        self.log.append(("move", slot, x, y))

    def up(self, slot: int) -> None:
        self.contacts.pop(slot, None)
        self.log.append(("up", slot))

    def release_all(self) -> None:
        for slot in list(self.contacts):
            self.up(slot)

    def close(self) -> None:
        self.release_all()
        self._closed = True

    @property
    def is_alive(self) -> bool:
        return not self._closed

    @property
    def injects(self) -> bool:
        """Nothing reaches a device from here; that is the entire point of the class."""
        return False
