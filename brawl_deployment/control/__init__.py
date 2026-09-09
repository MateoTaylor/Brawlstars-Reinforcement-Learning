"""The input side: touch injection and the geometry that decides where to touch.

    backend.py      the InputBackend protocol, slot assignments, and a NullBackend for dry runs
    adb.py          the real injector -- persistent `adb shell` + `sendevent`
    joystick.py     move_bin -> screen point, and the held-contact policy
    buttons.py      attack / super taps
    calibration.py  where those points come from, and the live checks that prove them

**The first four know nothing about the screen**, which is what keeps them testable with no
emulator attached: they are handed a calibration and they emit touches. `calibration.py` is the
one that closes the loop -- it fits the button table from pixels and verifies a tap by watching
the ammo bar -- and it is deliberately the last file, imported by the calibration script rather
than by the control path. Nothing the loop runs depends on it.
"""
from .backend import SLOT_MOVE, SLOT_TAP, InputBackend, NullBackend
from .buttons import ATTACK_FIRE, ATTACK_NONE, ATTACK_SUPER, Buttons
from .joystick import Joystick

__all__ = [
    "SLOT_MOVE", "SLOT_TAP", "InputBackend", "NullBackend",
    "Buttons", "ATTACK_NONE", "ATTACK_FIRE", "ATTACK_SUPER",
    "Joystick",
]
