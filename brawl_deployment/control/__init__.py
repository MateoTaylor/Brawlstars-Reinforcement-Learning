"""The input side: touch injection and the geometry that decides where to touch.

    backend.py    the InputBackend protocol, slot assignments, and a NullBackend for dry runs
    adb.py        the real injector -- persistent `adb shell` + `sendevent`
    joystick.py   move_bin -> screen point, and the held-contact policy
    buttons.py    attack / super taps

Nothing in here reads the screen. The only thing it knows about the game is the calibration it is
handed, which keeps the whole package testable without an emulator attached.
"""
from .backend import SLOT_MOVE, SLOT_TAP, InputBackend, NullBackend
from .buttons import ATTACK_FIRE, ATTACK_NONE, ATTACK_SUPER, Buttons
from .joystick import Joystick

__all__ = [
    "SLOT_MOVE", "SLOT_TAP", "InputBackend", "NullBackend",
    "Buttons", "ATTACK_NONE", "ATTACK_FIRE", "ATTACK_SUPER",
    "Joystick",
]
