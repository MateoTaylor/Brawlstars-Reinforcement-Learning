"""brawl_deployment's control layer and in-match gate.

Everything here runs WITHOUT BlueStacks attached -- the touch backend is exercised through
`NullBackend`, and only the tests that read real footage carry `@pytest.mark.vision`. That is
deliberate: a test suite that needs an emulator running is a test suite nobody runs.
"""
import io
import json
import math
import re
import struct

import numpy as np
import pytest

from brawl_deployment.control import (ATTACK_AUTO, ATTACK_FIRE, ATTACK_GADGET, ATTACK_NONE,
                                      ATTACK_SUPER, Buttons, Joystick, NullBackend, SLOT_MOVE,
                                      SLOT_TAP)
from brawl_deployment.control.adb import (DEVICE_FD, EVENT_STRUCTS, AdbTouchBackend,
                                          find_adb_serial)
from brawl_deployment.capture import to_viewport
from brawl_deployment.match_state import (CALIBRATION_PATH, Calibration, GateLook, MatchState,
                                          refine_radius)
from brawl_vision import gameplay as _G

CLIP = "tests/fixtures/vision/bluestacks-example-new.mp4"
HYPERCHARGE_CLIP = "tests/fixtures/vision/bluestacks-bull-hypercharge.mp4"
# The user's own account in the real app, not Nulls Brawl (design §5.2). Gitignored, like every
# clip under training_videos/, so the test that reads it skips without it.
REAL_APP_CLIP = "brawl_vision/data/training_videos/bluestacks-starfruit-supernova-west-upper.mp4"


# ---------------------------------------------------------------- joystick geometry

def test_idle_is_the_anchor_not_a_release():
    """`move == 0` parks the contact at the anchor. It must NOT lift it -- releasing homes the
    floating stick to the bottom-left and forces a re-acquire on the next non-idle decision."""
    b = NullBackend()
    j = Joystick(b, anchor=(300.0, 700.0), radius_px=140.0)
    j.acquire()
    j.apply(5)
    j.apply(0)
    assert j.contact_point(0) == (300.0, 700.0)
    assert j.is_down
    assert SLOT_MOVE in b.contacts
    assert not any(entry[0] == "up" for entry in b.log)


def test_bin_directions_have_no_y_flip():
    """The world's y and the screen's y both increase downward, so bin 4 of 16 (a quarter turn
    from +x) must go DOWN the screen. A sign error here is a silent 100% failure."""
    j = Joystick(NullBackend(), anchor=(0.0, 0.0), radius_px=100.0)
    east = j.contact_point(1)            # bin 0 -> +x
    south = j.contact_point(5)           # bin 4 -> +y, which is DOWN in screen space
    west = j.contact_point(9)            # bin 8 -> -x
    north = j.contact_point(13)          # bin 12 -> -y, UP the screen
    assert east == pytest.approx((100.0, 0.0), abs=1e-9)
    assert south == pytest.approx((0.0, 100.0), abs=1e-9)
    assert west == pytest.approx((-100.0, 0.0), abs=1e-9)
    assert north == pytest.approx((0.0, -100.0), abs=1e-9)


def test_bins_match_the_sims_own_decode():
    """`contact_point` must agree with `geometry.dir_from_bin`, which is what the policy's action
    means. Checked against the sim rather than against a hand-derived formula."""
    torch = pytest.importorskip("torch")
    from brawl_sim.core import geometry as geo

    n = 16
    j = Joystick(NullBackend(), anchor=(0.0, 0.0), radius_px=1.0, n_bins=n)
    dirs = geo.dir_from_bin(torch.arange(n), n).numpy()
    for bin_idx in range(n):
        x, y = j.contact_point(bin_idx + 1)     # action value is bin + 1; 0 is idle
        assert (x, y) == pytest.approx(tuple(dirs[bin_idx]), abs=1e-6)


def test_unchanged_bin_does_not_rewrite():
    """Holding a direction should cost nothing after the first write."""
    b = NullBackend()
    j = Joystick(b, anchor=(300.0, 700.0), radius_px=140.0)
    j.apply(3)
    n_after_first = len(b.log)
    j.apply(3)
    j.apply(3)
    assert len(b.log) == n_after_first


def test_apply_acquires_if_not_down():
    b = NullBackend()
    j = Joystick(b, anchor=(300.0, 700.0), radius_px=140.0)
    assert not j.is_down
    j.apply(2)
    assert j.is_down
    assert b.log[0][0] == "down"


# ---------------------------------------------------------------- buttons

def test_a_press_goes_down_on_the_right_origin():
    """Fire goes down on the attack clearance point, super on the super button's centre."""
    b = NullBackend()
    btn = Buttons(b, attack=(1690.0, 594.0), super_=(1462.5, 1000.5), gadget=(1559.9, 901.1))
    assert btn.press(ATTACK_FIRE, 0.0)
    assert b.log[-1] == ("down", SLOT_TAP, 1690.0, 594.0)
    btn.release()
    assert btn.press(ATTACK_SUPER, 0.0)
    assert b.log[-1] == ("down", SLOT_TAP, 1462.5, 1000.5)


def test_none_presses_nothing():
    b = NullBackend()
    btn = Buttons(b, attack=(1.0, 2.0), super_=(3.0, 4.0), gadget=(5.0, 6.0))
    assert not btn.press(ATTACK_NONE, 0.0)
    assert b.log == []
    assert not btn.is_held
    with pytest.raises(ValueError, match="no aim point"):
        btn.aim_point(ATTACK_NONE, 0.0)


def test_a_press_is_down_then_drag_then_lift_one_step_per_settle():
    """Never two steps in one call. A down and a drag landing in the same game frame would put
    the floating stick's centre at the dragged point -- a zero-length drag, which the game reads
    as a tap and auto-aims. The lift is what fires, and it comes on the second settle."""
    b = NullBackend()
    btn = Buttons(b, attack=(1000.0, 500.0), super_=(3.0, 4.0), gadget=(5.0, 6.0),
                  aim_radius_px=80.0)
    btn.press(ATTACK_FIRE, 0.0)
    assert b.log == [("down", SLOT_TAP, 1000.0, 500.0)]
    assert btn.is_held
    btn.settle()
    assert b.log[1] == ("move", SLOT_TAP, 1080.0, 500.0)
    assert btn.is_held
    btn.settle()
    assert b.log[2] == ("up", SLOT_TAP)
    assert not btn.is_held and SLOT_TAP not in b.contacts
    btn.settle()                                  # nothing in flight: nothing sent
    assert len(b.log) == 3


def test_the_drag_follows_the_bearing_with_no_y_flip():
    """World y and screen y both increase downward, so a quarter turn from +x drags DOWN the
    screen. A minus sign here would mirror every dash across the horizontal -- the attack twin of
    `test_bin_directions_have_no_y_flip`."""
    btn = Buttons(NullBackend(), attack=(0.0, 0.0), super_=(500.0, 500.0),
                  gadget=(300.0, 300.0), aim_radius_px=100.0)
    assert btn.aim_point(ATTACK_FIRE, 0.0) == pytest.approx((100.0, 0.0), abs=1e-9)
    assert btn.aim_point(ATTACK_FIRE, math.pi / 2) == pytest.approx((0.0, 100.0), abs=1e-9)
    assert btn.aim_point(ATTACK_FIRE, math.pi) == pytest.approx((-100.0, 0.0), abs=1e-9)
    assert btn.aim_point(ATTACK_FIRE, -math.pi / 2) == pytest.approx((0.0, -100.0), abs=1e-9)
    # The super drags from ITS origin, not the attack point's.
    assert btn.aim_point(ATTACK_SUPER, math.pi / 2) == pytest.approx((500.0, 600.0), abs=1e-9)


def test_a_press_over_a_press_in_flight_finishes_it_first():
    """A down on a slot that already holds a contact is not a fresh press, so an in-flight press
    is completed before the next goes down -- dragged to ITS aim and lifted, so that shot still
    goes where it was aimed rather than being swallowed or fired as a tap."""
    b = NullBackend()
    btn = Buttons(b, attack=(1000.0, 500.0), super_=(3.0, 4.0), gadget=(5.0, 6.0),
                  aim_radius_px=80.0)
    btn.press(ATTACK_FIRE, 0.0)                   # not yet dragged
    btn.press(ATTACK_FIRE, math.pi)
    assert b.log == [("down", SLOT_TAP, 1000.0, 500.0), ("move", SLOT_TAP, 1080.0, 500.0),
                     ("up", SLOT_TAP), ("down", SLOT_TAP, 1000.0, 500.0)]
    btn.settle()                                  # dragged; the next press only has to lift
    btn.press(ATTACK_FIRE, 0.0)
    assert [e[0] for e in b.log[4:]] == ["move", "up", "down"]
    assert b.log[4][2:] == pytest.approx((920.0, 500.0))


def test_release_lifts_at_any_stage_and_is_idempotent():
    """The fail-closed path. `release` lifts NOW -- it does not walk the press on -- and a second
    call sends nothing, because `Controls.release_all` runs in exception handlers."""
    for settles, kinds in ((0, ["down", "up"]), (1, ["down", "move", "up"])):
        b = NullBackend()
        btn = Buttons(b, attack=(1000.0, 500.0), super_=(3.0, 4.0), gadget=(5.0, 6.0))
        btn.press(ATTACK_FIRE, 0.0)
        for _ in range(settles):
            btn.settle()
        btn.release()
        btn.release()
        assert [e[0] for e in b.log] == kinds
        assert not btn.is_held


def test_a_press_does_not_disturb_the_movement_contact():
    """The whole point of multitouch here: firing must not lift or move the held joystick contact,
    at any step of the press."""
    b = NullBackend()
    j = Joystick(b, anchor=(300.0, 700.0), radius_px=140.0)
    btn = Buttons(b, attack=(1676.5, 999.5), super_=(1462.5, 1000.5), gadget=(1559.9, 901.1))
    j.apply(7)
    held = b.contacts[SLOT_MOVE]
    before = len(b.log)
    btn.press(ATTACK_FIRE, 1.0)
    btn.settle()
    btn.settle()
    assert b.contacts[SLOT_MOVE] == held
    assert [e[1] for e in b.log[before:]] == [SLOT_TAP] * 3
    assert j.is_down


def test_a_gadget_press_is_a_bare_tap_on_the_gadget_button():
    """The one press with no drag: the game aims the gadget at the nearest
    enemy by itself, as the sim does, so there is no bearing to drag along. Down on the button
    centre, up on the next settle, never a move, whatever bearing is passed."""
    b = NullBackend()
    btn = Buttons(b, attack=(1000.0, 500.0), super_=(3.0, 4.0), gadget=(1559.9, 901.1))
    assert btn.press(ATTACK_GADGET, math.pi / 2)
    assert b.log == [("down", SLOT_TAP, 1559.9, 901.1)]
    assert btn.is_held
    btn.settle()
    assert b.log[1:] == [("up", SLOT_TAP)]
    assert not btn.is_held and SLOT_TAP not in b.contacts
    btn.settle()                                  # nothing in flight: nothing sent
    assert len(b.log) == 2
    with pytest.raises(ValueError, match="no aim point"):
        btn.aim_point(ATTACK_GADGET, 0.0)


def test_a_tap_and_a_press_in_flight_each_finish_the_other_first():
    """The rule for two attacks holds across buttons. A dash in flight is dragged to ITS aim and
    lifted before the gadget goes down, so it still fires where it was aimed; and a tap in
    flight is lifted before the next press goes down."""
    b = NullBackend()
    btn = Buttons(b, attack=(1000.0, 500.0), super_=(3.0, 4.0), gadget=(1559.9, 901.1),
                  aim_radius_px=80.0)
    btn.press(ATTACK_FIRE, 0.0)
    btn.press(ATTACK_GADGET, 0.0)
    assert b.log == [("down", SLOT_TAP, 1000.0, 500.0), ("move", SLOT_TAP, 1080.0, 500.0),
                     ("up", SLOT_TAP), ("down", SLOT_TAP, 1559.9, 901.1)]
    btn.press(ATTACK_FIRE, 0.0)
    assert b.log[4:] == [("up", SLOT_TAP), ("down", SLOT_TAP, 1000.0, 500.0)]


def test_release_lifts_a_gadget_tap_and_is_idempotent():
    b = NullBackend()
    btn = Buttons(b, attack=(1000.0, 500.0), super_=(3.0, 4.0), gadget=(1559.9, 901.1))
    btn.press(ATTACK_GADGET, 0.0)
    btn.release()
    btn.release()
    assert [e[0] for e in b.log] == ["down", "up"]
    assert not btn.is_held


# ---------------------------------------------------------------- adb coordinate mapping

def test_pixels_scale_into_device_axis_units():
    """The device reports ABS_MT_POSITION over 0..32767, not pixels. Unscaled coordinates land in
    the top-left 6% of the screen and look like "injection is broken"."""
    be = AdbTouchBackend.__new__(AdbTouchBackend)      # no emulator, no shell
    be.screen_w, be.screen_h, be.abs_max = 1920, 1080, 32767
    assert be._to_raw(0, 0) == (0, 0)
    assert be._to_raw(1919, 1079) == (32767, 32767)
    mid = be._to_raw(959.5, 539.5)
    assert mid[0] == pytest.approx(16383, abs=2)
    assert mid[1] == pytest.approx(16383, abs=2)


def test_offscreen_coordinates_clamp():
    """An anchor near an edge plus a saturation radius can legitimately compute an off-screen
    point; a clamped touch is a better failure than a rejected one."""
    be = AdbTouchBackend.__new__(AdbTouchBackend)
    be.screen_w, be.screen_h, be.abs_max = 1920, 1080, 32767
    assert be._to_raw(-50, -50) == (0, 0)
    assert be._to_raw(5000, 5000) == (32767, 32767)


# ---------------------------------------------------------------- adb transport

_OCTAL = re.compile(r"\\0([0-7]{3})")


class _Shell:
    """The device-side shell as the backend sees it: a stdin that keeps every line, a stdout
    scripted by the test."""

    def __init__(self, stdout=b""):
        self.lines = []
        self.stdout = io.BytesIO(stdout)
        self.stdin = self
        self.terminated = False

    def write(self, data):
        self.lines.append(data)

    def flush(self):
        pass

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True


def _backend(stdout=b"", event_size=24):
    be = AdbTouchBackend.__new__(AdbTouchBackend)      # no emulator
    be.serial, be.device = "127.0.0.1:5555", "/dev/input/event4"
    be.screen_w, be.screen_h, be.abs_max = 1920, 1080, 32767
    be._contacts, be._next_tracking_id = {}, 1
    be.event_struct = EVENT_STRUCTS[event_size]
    be._proc = _Shell(stdout)
    return be


def _decode(line, event_size=24):
    """A shell line back into `(type, code, value)` events -- by the test's own regex and struct,
    not the backend's encoder, so a wrong escape or a wrong layout cannot agree with itself."""
    assert line.startswith(f"print -n '") and line.endswith(f"' >&{DEVICE_FD}\n"), line
    payload = line[len("print -n '"):-len(f"' >&{DEVICE_FD}\n")]
    raw = bytes(int(m, 8) for m in _OCTAL.findall(payload))
    assert len(raw) == len(payload) // 5, "every byte is exactly one five-character escape"
    assert len(raw) % event_size == 0
    fmt = "<qqHHi" if event_size == 24 else "<iiHHi"
    out = []
    for i in range(0, len(raw), event_size):
        sec, usec, t, c, v = struct.unpack(fmt, raw[i:i + event_size])
        assert (sec, usec) == (0, 0), "timestamps are ignored on the write path and go out zero"
        out.append((t, c, v))
    return out


def test_a_touch_step_is_one_builtin_write_and_spawns_nothing():
    """The lag of 2026-09-16: `sendevent` is a process spawn per event, ~45 ms each with the game
    loaded, and an aimed press is 13 of them. One `print` builtin writing the packed structs to a
    descriptor the shell holds open costs 0.6 ms for the same 13 -- so a touch step must be one
    line, one write, and never the name of a program."""
    be = _backend()
    be.down(SLOT_TAP, 1690.0, 594.0)
    assert len(be._proc.lines) == 1
    line = be._proc.lines[0].decode("ascii")
    assert "sendevent" not in line
    rx, ry = be._to_raw(1690.0, 594.0)
    assert _decode(line) == [(3, 47, SLOT_TAP), (3, 57, 1), (3, 53, rx), (3, 54, ry), (3, 48, 8),
                             (1, 330, 1), (0, 0, 0)]


def test_the_line_stays_ascii_for_negative_values():
    """adb 1.0.36 cannot forward binary on Windows, so the bytes travel as octal escapes. -1 (the
    Type-B "slot empty" tracking id) is the case with every byte set."""
    be = _backend()
    be.down(SLOT_TAP, 100.0, 100.0)
    be.up(SLOT_TAP)
    line = be._proc.lines[-1]
    line.decode("ascii")                       # raises if anything but ASCII went on the pipe
    assert all(32 <= b < 127 or b == 10 for b in line)
    assert _decode(line.decode()) == [(3, 47, SLOT_TAP), (3, 57, -1), (1, 330, 0), (0, 0, 0)]


def test_a_32_bit_shell_gets_16_byte_events():
    """The kernel sizes `input_event` by the WRITER's ABI and refuses the wrong length, so the
    struct follows the shell's ELF class, which the handshake reads."""
    be = _backend(event_size=16)
    be.move(SLOT_MOVE, 10.0, 10.0)             # not down: nothing sent
    assert be._proc.lines == []
    be.down(SLOT_MOVE, 10.0, 10.0)
    events = _decode(be._proc.lines[0].decode(), event_size=16)
    assert [e[:2] for e in events] == [(3, 47), (3, 57), (3, 53), (3, 54), (3, 48), (1, 330), (0, 0)]


def test_the_handshake_reads_the_shell_class_and_proves_one_write():
    be = _backend(stdout=b" 02\nOPEN\nOK\n", event_size=16)   # wrong size on purpose
    be._handshake()
    assert be.event_struct is EVENT_STRUCTS[24]
    sent = [l.decode() for l in be._proc.lines]
    assert sent[0].startswith("od -An -tx1 -j4 -N1 /proc/$$/exe; [ -w /dev/input/event4 ] && exec 3>/dev/input/event4")
    assert _decode(sent[1].replace(" && echo OK || echo FAIL", "")) == [(0, 0, 0)]
    assert sent[1].rstrip("\n").endswith(" && echo OK || echo FAIL")


@pytest.mark.parametrize("stdout,match", [
    (b" 02\nNOOPEN\n", "cannot open"),
    (b" 02\nOPEN\nFAIL\n", "refused"),
    (b"", "did not answer"),
])
def test_the_handshake_refuses_a_session_that_cannot_write(stdout, match, monkeypatch):
    """A session whose writes go nowhere would look exactly like a working one from the loop --
    `_send` never reads a reply -- so the one exchange at startup has to be the check."""
    monkeypatch.setattr("brawl_deployment.control.adb.HANDSHAKE_TIMEOUT_S", 0.2)
    be = _backend(stdout=stdout)
    shell = be._proc
    with pytest.raises(RuntimeError, match=match):
        be._handshake()
    assert shell.terminated and not be.is_alive


# ---------------------------------------------------------------- calibration

def test_calibration_loads_and_is_self_consistent():
    cal = Calibration.load()
    assert cal.screen == (1920, 1080)
    assert cal.gate_anchor in cal.buttons
    assert cal.gate_looks and cal.gate_anchor == cal.gate_looks[0].button
    assert {look.button for look in cal.gate_looks} <= set(cal.buttons)
    # No `attack`, on purpose: attack is an AREA, and `control.attack_tap` picks a clearance
    # point for it rather than a button centre. See the file's own notes.
    assert set(cal.buttons) == {"gadget", "super", "hypercharge"}
    for name in cal.buttons:
        x, y = cal.button(name)
        assert 0 < x < cal.screen[0] and 0 < y < cal.screen[1]


def test_the_gate_rests_on_a_button_the_policy_never_presses():
    """The invariant the 2026-09-22 re-labelling was missing, narrowed on 2026-11-04 (§5.2).

    The gate reads ring scores to decide whether we are in a match, and a control the agent
    presses changes its own face. The FIRST look is the one the gate rests on whenever the
    brawler draws it, so it must be a disc nothing presses. The file shipped with the gate on the
    Super until that was caught live, because its anchor was named `gadget`.

    The Super has backed the gate up since the real app's account turned out to draw no
    hypercharge. A pressed button may do that only with every face it shows: one Super face alone
    reads 0.02-0.10 while the Super shows the other. The gadget never: one throw held its ring
    under threshold for 5.0 s in the live trace (`control_calibration.json`).

    Stated against `Controls.build`'s own reads rather than a name list, so a fourth pressable
    control cannot be added without this failing.
    """
    cal = Calibration.load()
    pressed = {"super", "gadget"}           # Controls.build: super_=button("super"), gadget=...
    assert cal.gate_anchor not in pressed, (
        f"gate rests on {cal.gate_anchor!r}, which the policy presses")
    assert pressed <= set(cal.buttons), "Controls.build would KeyError on this calibration"
    looked_at = {look.button for look in cal.gate_looks}
    assert "gadget" not in looked_at
    for button in looked_at & pressed:
        radii = {cal.viewport_look(look)[2] for look in cal.gate_looks if look.button == button}
        assert len(radii) >= 2, f"{button} backs the gate with one face only"


def test_calibration_rejects_a_frame_that_is_not_at_the_viewport():
    """The raw-grab trap: a 1440p monitor grab misses the calibrated anchors by 1.28x and every
    downstream reading is confidently wrong while looking fine. Must be loud."""
    cal = Calibration.load()
    for bad in ((1440, 2560, 3), (1080, 1920, 3)):
        with pytest.raises(ValueError, match="calibrated for"):
            cal.check_frame(bad)
    cal.check_frame((cal.viewport[1], cal.viewport[0], 3))   # the matching case must not raise


def test_device_and_viewport_spaces_are_distinct_and_convert():
    """The two spaces coincided while the fixture was 1080p. They must not be conflated now:
    a tap goes to device pixels, the ring score reads viewport pixels."""
    cal = Calibration.load()
    assert cal.viewport != cal.screen, "the test below is vacuous if these are equal"
    for name in cal.buttons:
        dx, dy = cal.button(name)
        vx, vy, vr = cal.viewport_button(name)
        assert (vx, vy) != (dx, dy)
        assert vx / dx == pytest.approx(cal.viewport[0] / cal.screen[0], rel=1e-9)
        assert vy / dy == pytest.approx(cal.viewport[1] / cal.screen[1], rel=1e-9)
        assert vr > cal.buttons[name]["r"]


def test_joystick_calibration_is_measured_and_has_clearance():
    """The anchor must have room for a full swing on the left of the screen, and the commanded
    radius must clear the measured saturation distance -- below it the agent walks at less than
    full speed, which the policy never trained for."""
    raw = json.loads(CALIBRATION_PATH.read_text())
    if not raw["joystick"].get("measured", False):
        pytest.skip("joystick calibration still a placeholder, as declared")
    cal = Calibration.load()
    ax, ay = cal.joystick_anchor
    r = cal.joystick_radius_px
    saturation = raw["joystick"]["saturation_px"]
    assert r > saturation, f"radius {r} is inside the {saturation}px saturation distance"
    assert r < ax < cal.screen[0] / 2
    assert r < ay < cal.screen[1] - r


def test_every_bin_stays_on_screen():
    """A commanded contact point that lands off-screen gets clamped by the backend, which
    silently distorts that one direction. Every bin must fit without clamping."""
    cal = Calibration.load()
    j = Joystick(NullBackend(), cal.joystick_anchor, cal.joystick_radius_px)
    w, h = cal.screen
    for move in range(0, j.n_bins + 1):
        x, y = j.contact_point(move)
        assert 0 <= x < w and 0 <= y < h, f"bin {move} lands off-screen at ({x:.1f}, {y:.1f})"


def _shipped_buttons() -> tuple[Buttons, tuple[int, int]]:
    """Built the way `Controls.build` builds them, from the shipped calibration and config."""
    from brawl_deployment.config import load_deployment_config
    cal = Calibration.load()
    dcfg = load_deployment_config()
    w, h = cal.screen
    btn = Buttons(NullBackend(),
                  attack=(dcfg.control_attack_tap[0] * w, dcfg.control_attack_tap[1] * h),
                  super_=cal.button("super"), gadget=cal.button("gadget"),
                  aim_radius_px=dcfg.control_aim_radius_px)
    return btn, cal.screen


def test_every_aim_stays_on_screen():
    """The attack twin of the test above, and it bites harder: the backend clamps an off-screen
    point to the edge, so a clamped DRAG lifts somewhere real, aimed the wrong way. The super
    button sits 179 px above the bottom edge, so downward supers are still the case that fails
    first, but by a far wider margin since 2026-09-22: the disc this used to call the super sat
    81 px up, and was really a third control nothing drags.
    Swept over bearings rather than trusting `require_on_screen`'s own arithmetic."""
    btn, (w, h) = _shipped_buttons()
    btn.require_on_screen((w, h))
    for action in (ATTACK_FIRE, ATTACK_SUPER):
        for k in range(64):
            x, y = btn.aim_point(action, k * 2.0 * math.pi / 64)
            assert 0 <= x <= w - 1 and 0 <= y <= h - 1, (
                f"action {action} at bearing {k}/64 of a turn drags off-screen to ({x:.1f}, {y:.1f})")


def test_a_radius_that_drags_the_super_off_screen_is_refused():
    """What `Controls.build` runs before the first match. 185 px still clears the attack point
    (229 px from the right edge) and drags a downward super 7 px past the bottom.

    It was 90 px until 2026-09-22. The number moved because the SUPER moved: the disc the
    calibration called the super sits at y=998, 81 px up, and turned out to be a control nothing
    presses. The real Super is at y=901, so the ceiling roughly doubled."""
    btn, screen = _shipped_buttons()
    btn.aim_radius_px = 185.0
    with pytest.raises(ValueError, match="super"):
        btn.require_on_screen(screen)


def test_a_gadget_button_off_the_screen_is_refused():
    """No drag, so no room is needed around it: only the tap point has to be on the screen. The
    backend clamps, so an off-screen tap would land on the edge, on whatever is drawn there."""
    btn, screen = _shipped_buttons()
    btn.require_on_screen(screen)                 # the shipped calibration's centre is fine
    w, h = screen
    btn.gadget = (w - 1.0, 500.0)                 # the last column: on screen, zero room
    btn.require_on_screen(screen)
    btn.gadget = (w + 10.0, 500.0)
    with pytest.raises(ValueError, match="gadget"):
        btn.require_on_screen(screen)


def test_controls_build_taps_the_calibrated_gadget(monkeypatch):
    """`Controls.build` is the one place the loop's buttons come from, and `_shipped_buttons`
    above only copies it. This runs the real build from the shipped calibration and config, with
    just the ADB connection faked, and taps the gadget through what it returns. The down has to
    land on the calibration's gadget centre, which is neither the super's nor the attack point."""
    from brawl_deployment.config import load_deployment_config
    from brawl_deployment.control import adb
    from brawl_deployment.loop import Controls

    backend = NullBackend()
    monkeypatch.setattr(adb, "find_adb_serial", lambda conf=None, instance=None: "127.0.0.1:5555")
    monkeypatch.setattr(adb, "AdbTouchBackend", lambda serial, screen: backend)
    cal = Calibration.load()
    controls = Controls.build(cal, load_deployment_config(), 16)

    assert controls.backend is backend
    gadget = cal.button("gadget")
    assert gadget not in (cal.button("super"), controls.buttons.attack)
    assert controls.buttons.press(ATTACK_GADGET, 0.0)
    assert backend.log == [("down", SLOT_TAP, *gadget)]


def test_a_dry_run_keeps_every_control_and_touches_only_the_null_backend():
    """`scripts/deploy_run.py --dry-run` rebuilds the controls on a `NullBackend` through
    `Controls.with_backend`. The script's own positional rebuild would have crashed every dry
    run once the gadget became a required button, with nothing offline
    to see it. Every field is a number nothing else uses, the defaults included, so a field
    dropped or swapped in the copy shows."""
    from brawl_deployment.loop import Controls

    live = NullBackend()
    controls = Controls(backend=live,
                        joystick=Joystick(live, (310.0, 820.0), 121.0, n_bins=12),
                        buttons=Buttons(live, attack=(1690.0, 594.0), super_=(1462.5, 1000.5),
                                        gadget=(1559.9, 901.1), aim_radius_px=77.0,
                                        screen=(1919, 1079)))
    null = NullBackend()
    dry = controls.with_backend(null)

    assert dry.backend is null and dry.joystick.backend is null and dry.buttons.backend is null
    j, b = dry.joystick, dry.buttons
    assert (j.anchor, j.radius_px, j.n_bins) == ((310.0, 820.0), 121.0, 12)
    assert (b.attack, b.super_, b.gadget, b.aim_radius_px, b.screen) == (
        (1690.0, 594.0), (1462.5, 1000.5), (1559.9, 901.1), 77.0, (1919, 1079))
    assert b.press(ATTACK_GADGET, 0.0)
    assert null.log == [("down", SLOT_TAP, 1559.9, 901.1)]
    assert live.log == []


# ---------------------------------------------------------------- the in-match gate

def test_match_state_starts_out_of_match():
    """The safe initial state: a loop started on a menu must stay quiet."""
    ms = MatchState(Calibration.load())
    assert not ms.in_match


def test_match_state_needs_a_sustained_run_to_enter(monkeypatch):
    cal = Calibration.load()
    ms = MatchState(cal)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    monkeypatch.setattr(ms, "_looks", [(100.0, 100.0, 30.0)])
    monkeypatch.setattr("brawl_deployment.match_state.G.ring_score_at", lambda *a, **k: 0.9)
    for _ in range(cal.enter_samples - 1):
        assert not ms.update(frame)
    assert ms.update(frame)


def test_match_state_exit_is_not_instant(monkeypatch):
    """A Super detonating over the button drops the score for a few MID-MATCH frames. Those frames
    are gameplay -- `gameplay.py` documents exactly this failure."""
    cal = Calibration.load()
    ms = MatchState(cal)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    monkeypatch.setattr("brawl_deployment.match_state.G.ring_score_at", lambda *a, **k: 0.9)
    for _ in range(cal.enter_samples):
        ms.update(frame)
    assert ms.in_match
    monkeypatch.setattr("brawl_deployment.match_state.G.ring_score_at", lambda *a, **k: 0.01)
    for _ in range(cal.exit_samples - 1):
        assert ms.update(frame)
    assert not ms.update(frame)


def test_force_exit_is_immediate():
    ms = MatchState(Calibration.load())
    ms._in_match = True
    ms.force_exit()
    assert not ms.in_match


def test_refine_radius_finds_a_synthetic_ring():
    """A ring drawn at a known radius must be recovered even when the starting guess is well off
    -- the exact failure that made the stored calibration read 0.083 on a live frame."""
    import cv2
    frame = np.zeros((300, 300, 3), np.uint8)
    cv2.circle(frame, (150, 150), 40, (200, 200, 200), 2)
    r, score = refine_radius(frame, 150.0, 150.0, r0=33.0)
    # Biased slightly OUTWARD of the drawn radius, and that is correct rather than sloppy: the
    # stroke has width and `_gradients` blurs 5x5 before differencing, so the strongest radial
    # gradient sits at the ring's outer edge, not its centreline. The same bias applies to the
    # real buttons, which is why the fit is what gets stored rather than any nominal radius.
    assert r == pytest.approx(40.0, abs=4.0)
    assert score > 0.8
    # The wrong radius scores far worse -- this is the sensitivity that motivates refinement.
    gx, gy = _G._gradients(frame)
    assert _G.ring_score_at(gx, gy, 150.0, 150.0, 33.0) < score / 2


def test_refine_rejects_a_frame_with_no_ring():
    """Refining on a menu would fit the radius to whatever the background makes ring-shaped.
    Must raise rather than silently mis-tune the gate."""
    ms = MatchState(Calibration.load())
    with pytest.raises(ValueError, match="probably is not gameplay"):
        ms.refine(np.zeros((1080, 1920, 3), np.uint8))
    assert not ms.refined


# -- the gate's looks (2026-11-04, design §5.2) ------------------------------------------------

def _disc_frame(cal, discs):
    """A black viewport frame with one filled grey disc per `(button, viewport radius)`, on the
    button's calibrated centre: a HUD disc with nothing around it."""
    import cv2
    frame = np.zeros((cal.viewport[1], cal.viewport[0], 3), np.uint8)
    for button, r in discs:
        cx, cy, _ = cal.viewport_button(button)
        cv2.circle(frame, (round(cx), round(cy)), round(r), (200, 200, 200), -1)
    return frame


def _look_radius(cal, face):
    look = next(lk for lk in cal.gate_looks if lk.button == "super" and lk.face == face)
    return cal.viewport_look(look)[2]


@pytest.mark.parametrize("face", ["uncharged", "charged"])
def test_either_face_of_the_super_opens_the_gate_with_no_hypercharge_drawn(face):
    """The real app's account draws no hypercharge (Shelly, power 3), so the gate must open on
    the Super, whichever face it shows. The score is the best look's; `last_scores` keeps each,
    in `cal.gate_looks` order."""
    cal = Calibration.load()
    ms = MatchState(cal)
    frame = _disc_frame(cal, [("super", _look_radius(cal, face))])
    for _ in range(cal.enter_samples):
        ms.update(frame)
    assert ms.in_match
    labels = [label for label, *_ in ms.looks]
    scores = dict(zip(labels, ms.last_scores))
    assert scores[f"super ({face})"] > 0.9
    other = "charged" if face == "uncharged" else "uncharged"
    assert scores[f"super ({other})"] < 0.1 and scores["hypercharge"] < 0.1
    assert ms.last_score == max(ms.last_scores)


def test_refine_fits_only_the_looks_that_are_drawn():
    """A look not drawn on the refine frame keeps its STORED radius. The charged Super is never
    drawn at a match's start, and the real app's account draws no hypercharge; fitting either to
    the background would move it off the face it exists for."""
    cal = Calibration.load()
    ms = MatchState(cal)
    ms.refine(_disc_frame(cal, [("super", _look_radius(cal, "uncharged"))]))
    assert ms.drawn == ("super (uncharged)",)
    stored = [cal.viewport_look(look)[2] for look in cal.gate_looks]
    fitted = [r for *_, r in ms.looks]
    for look, want, got in zip(cal.gate_looks, stored, fitted):
        if look.label in ms.drawn:
            assert got == pytest.approx(want, abs=1.0), look.label
        else:
            assert got == want, f"{look.label} moved without being drawn"


def test_refine_cannot_move_one_super_face_onto_the_other():
    """The two faces are 7 px apart, and a one-look gate's +/-12 px scan would let the uncharged
    look jump onto a charged disc. That look would then never match again once the Super is
    spent, which is the "lit disc" trap in match_state.py, one face over. `_span` holds each
    look nearer its own stored radius than its sibling's."""
    cal = Calibration.load()
    ms = MatchState(cal)
    ms.refine(_disc_frame(cal, [("super", _look_radius(cal, "charged"))]))
    assert ms.drawn == ("super (charged)",)
    assert dict((label, r) for label, *_, r in ms.looks)["super (uncharged)"] == \
        _look_radius(cal, "uncharged")


def test_refine_radius_scans_whole_steps_either_side_of_its_start(monkeypatch):
    """`MatchState._span` hands out spans like 3.49 px. A scan from `r0 - span` in 0.5 px steps
    then never lands on r0, and reaches 3.0 px above it against 3.49 below."""
    tried = []
    monkeypatch.setattr("brawl_deployment.match_state.G.ring_score_at",
                        lambda gx, gy, cx, cy, r: tried.append(r) or 0.5)
    refine_radius(np.zeros((200, 200, 3), np.uint8), 100.0, 100.0, 34.72, span=3.49)
    assert tried == pytest.approx([34.72 + 0.5 * k for k in range(-6, 7)])


def test_button_drawn_tells_a_disc_from_an_empty_spot():
    """The loop's loadout check: a brawler under power 7 draws no gadget (design §5.2)."""
    cal = Calibration.load()
    ms = MatchState(cal)
    drawn, score = ms.button_drawn(
        _disc_frame(cal, [("gadget", cal.viewport_button("gadget")[2])]), "gadget")
    assert drawn and score > 0.9
    drawn, score = ms.button_drawn(_disc_frame(cal, []), "gadget")
    assert not drawn and score < 0.1


def _calibration_file(tmp_path, gate):
    """The shipped calibration with its `match_gate` replaced by `gate`, as a loadable path."""
    raw = json.loads(CALIBRATION_PATH.read_text())
    raw["match_gate"] = {k: v for k, v in raw["match_gate"].items()
                         if k not in ("looks", "anchor")} | gate
    path = tmp_path / "control_calibration.json"
    path.write_text(json.dumps(raw))
    return path


def test_the_single_anchor_form_of_the_gate_still_loads(tmp_path):
    """`anchor` is the form the file had until 2026-11-04: one button, at its own radius."""
    cal = Calibration.load(_calibration_file(tmp_path, {"anchor": "hypercharge"}))
    assert cal.gate_looks == (GateLook("hypercharge"),)
    assert MatchState(cal).looks == MatchState.on(Calibration.load(), "hypercharge").looks


def test_a_look_at_a_button_the_file_does_not_have_refuses_to_load(tmp_path):
    with pytest.raises(ValueError, match="hyperchrage"):
        Calibration.load(_calibration_file(tmp_path, {"looks": [{"button": "hyperchrage"}]}))


@pytest.mark.vision
def test_roi_gradients_match_full_frame_gradients():
    """The gate scores a small ring off a small window instead of Sobelling 2.25 M pixels. Two
    separate claims, asserted separately because only one of them is exact.

    The GRADIENTS in the window are bit-identical -- the ring never comes near the cut, so the
    blur and Sobel kernels never reach it. The SCORE is not, quite: `ring_score_at` truncates its
    sample coordinates, and translating the centre by the ROI offset can move a sample one pixel
    at an angle where the coordinate lands on an exact integer. Worst case measured is 4 samples
    of 180. Random noise is used deliberately -- it is the adversarial case for both claims."""
    cv2 = pytest.importorskip("cv2")
    from brawl_deployment.match_state import _ROI_MARGIN_PX, _gradient_roi
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, (1126, 2002, 3), dtype=np.uint8)
    gx, gy = _G._gradients(frame)
    for cx, cy, r in ((1627.1, 939.9, 34.6), (1747.9, 1042.1, 31.6), (300.0, 300.0, 50.0)):
        rgx, rgy, lx, ly = _gradient_roi(frame, cx, cy, r)
        x0, y0 = int(cx - lx), int(cy - ly)
        sub = gx[y0:y0 + rgx.shape[0], x0:x0 + rgx.shape[1]]
        keep = slice(_ROI_MARGIN_PX // 4, -(_ROI_MARGIN_PX // 4))
        assert np.array_equal(sub[keep, keep], rgx[keep, keep]), "interior gradients must be exact"

        want = _G.ring_score_at(gx, gy, cx, cy, r)
        got = _G.ring_score_at(rgx, rgy, lx, ly, r)
        assert got == pytest.approx(want, abs=2e-3), f"at ({cx}, {cy}, r={r})"

    # The anchors that actually matter are exact, and that is worth pinning separately: the
    # tolerance above exists for the pathological case, not as licence for the gate to drift.
    cal = Calibration.load()
    for name in cal.buttons:
        cx, cy, r = cal.viewport_button(name)
        rgx, rgy, lx, ly = _gradient_roi(frame, cx, cy, r)
        assert _G.ring_score_at(rgx, rgy, lx, ly, r) == _G.ring_score_at(gx, gy, cx, cy, r)


def test_refine_recovers_the_radius_on_the_recorded_source():
    """Refinement must improve, not degrade, the anchor on the source the centres came from.

    It is NOT a no-op here even though the centres were measured on this very clip: the stored
    radius is in device pixels and gets scaled by 1.0426 into viewport pixels, while the resize to
    2002x1126 resamples the button edge. Both move the best-fit annulus. That is precisely the
    effect `refine_radius` exists for -- see its docstring."""
    cv2 = pytest.importorskip("cv2")
    from brawl_vision import gameplay as G
    cal = Calibration.load()
    med = to_viewport(G.median_frame(CLIP, (0, 1919, 0, 1079), n=24), cal.viewport)
    ms = MatchState(cal)
    ms.update(med)
    before = ms.last_score
    after = ms.refine(med)
    assert after >= before
    assert after > 0.7, f"refined score {after:.3f} is too close to the {cal.threshold} gate"
    # The refined radius must stay in the neighbourhood of the scaled measurement -- a peak far
    # from it would mean the fit had latched onto something else entirely.
    assert ms.anchor[2] == pytest.approx(cal.viewport_button(cal.gate_anchor)[2], abs=6.0)


@pytest.mark.vision
def test_gate_separates_gameplay_from_menus_on_real_footage():
    """The end-to-end claim, against the real BlueStacks recording: the gate is out of a match at
    the start, in one through the middle, and out again after "Defeated".

    Measured margins on this clip at the 2002x1126 viewport, `hypercharge` anchor since 2026-09-22,
    radius refined, 238 samples: menus 0.000-0.212, gameplay 0.968-0.994, median 0.985. That is
    2.1x below the 0.45 threshold and 2.2x above it.

    **This test failed for two weeks on `gameplay floor 0.675 too close to the threshold`, and
    the anchor move fixed it at the root rather than by loosening a bound.** 0.675 is the SUPER
    button's own in-match floor on this clip: its face changes as it charges, so its ring score
    dips mid-match. The gate was watching it because all three button names in the calibration
    sat one disc off (design 5.1). The bounds below describe `hypercharge` instead, and they are
    looser on the menu side and far tighter on the gameplay side, because that is exactly where the
    two discs differ: `hypercharge` reads 0.212 in menus where the Super read 0.083, and 0.968 in
    play where the Super fell to 0.675. What matters is the 0.45 threshold, and `hypercharge`
    clears it both ways.

    The third disc is disqualified on this same clip without anyone pressing it: the real gadget's
    in-match floor is 0.454, against a 0.45 threshold.
    """
    cv2 = pytest.importorskip("cv2")
    cal = Calibration.load()
    ms = MatchState(cal)

    cap = cv2.VideoCapture(CLIP)
    if not cap.isOpened():
        pytest.skip(f"{CLIP} not available")
    verdicts, scores, idx = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % 5 == 0:
            # Through the same resize the live loop uses, so this test is about the deployed
            # geometry rather than about the recording's own 1080p.
            frame = to_viewport(frame, cal.viewport)
            cal.check_frame(frame.shape)
            verdicts.append(ms.update(frame))
            scores.append(ms.last_score)
        idx += 1
    cap.release()

    assert len(verdicts) > 100
    assert not verdicts[0], "must start out of a match"
    assert any(verdicts), "must find the match"
    assert not verdicts[-1], "must be out of the match after the results screen"

    # One contiguous run, not a flickering mess -- the whole point of the hysteresis.
    transitions = sum(1 for a, b in zip(verdicts, verdicts[1:]) if a != b)
    assert transitions == 2, f"expected one clean enter/exit pair, saw {transitions} transitions"

    # Score separation is a claim about the SIGNAL, so partition by threshold rather than by
    # verdict. Partitioning by verdict would fail by construction: during the enter window the
    # score is already high while the verdict is still False, and during the exit window the score
    # has collapsed while the verdict is still True. That lag is the hysteresis working, not a
    # signal problem -- an earlier version of this test asserted it away and caught nothing.
    hi = [s for s in scores if s >= cal.threshold]
    lo = [s for s in scores if s < cal.threshold]
    assert hi and lo
    # Bounds are stated against the threshold, because the threshold is the thing they protect.
    # Measured 0.968 and 0.212, so each keeps roughly a third of its own margin in reserve.
    assert min(hi) > 1.7 * cal.threshold, f"gameplay floor {min(hi):.3f} too close to threshold"
    assert max(lo) < 0.70 * cal.threshold, f"menu ceiling {max(lo):.3f} too close to threshold"
    # The gap is what makes the 0.45 threshold safe to leave alone across HUD-neutral changes.
    assert min(hi) / max(lo) > 4.0


@pytest.mark.vision
def test_gate_holds_while_the_hypercharge_fills_and_lights_on_real_footage():
    """BULL_HERO_PLAN.md §5.7d, on a real match: the user's Bull fills his hypercharge and finishes
    1st with it lit (`Videos/2026-11-01 10-10-07.mp4`, BlueStacks 1440p, 30 fps). The disc the gate
    watches is NOT inert. A magenta meter fills around it, then it turns bright magenta with a bolt
    and stays lit until the results screen, because nothing the agent does presses it.

    Measured through this path, every frame, refined at r 35.9 as the loop refines:
    - the filling meter erodes the dull disc from 0.98 to 0.80;
    - the lit disc holds a median of 0.68;
    - the moment it lights dips to 0.24 for three frames. That is the only sub-threshold run inside
      the match, one short of `exit_samples`. Live samples come every 50 ms, not every 33, so the
      dip reaches the live gate at most three times too.
    - The results screen scores 0.000.

    The 10-28 alarm (lit 0.17) came from Training Grounds clips, where the disc sits on bright
    grass with no dark track around it. That venue is invalid for the gate.

    Since 2026-11-04 the gate is the best of its looks (§5.2), and the Super's cover that dip: the
    gate itself never dips in this match. The numbers above are the hypercharge look's own,
    `last_scores[0]`, and are asserted as such.

    Every frame, not every 5th as above: denser sampling can only lengthen a sub-threshold run, so
    it replays the live 20 Hz strictly."""
    cv2 = pytest.importorskip("cv2")
    from brawl_deployment.loop import GADGET_LOOK_SECONDS
    cal = Calibration.load()
    ms = MatchState(cal)
    cap = cv2.VideoCapture(HYPERCHARGE_CLIP)
    if not cap.isOpened():
        pytest.skip(f"{HYPERCHARGE_CLIP} not available")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cx, cy, _ = ms.anchor
    yy, xx = np.mgrid[-26:27, -26:27]
    disc = (xx ** 2 + yy ** 2) <= 26 ** 2
    verdicts, scores, hyper, lit, gadget = [], [], [], [], []
    look_until = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = to_viewport(frame, cal.viewport)
        was_in = ms.in_match
        verdicts.append(ms.update(frame))
        if verdicts[-1] and not was_in:
            ms.refine(frame)                      # as loop._begin_match does, on the dull disc
            look_until = len(verdicts) - 1 + round(GADGET_LOOK_SECONDS * fps)
        if len(verdicts) - 1 <= look_until:       # the loop's gadget window, every frame
            gadget.append(ms.button_drawn(frame, "gadget")[0])
        scores.append(ms.last_score)
        hyper.append(ms.last_scores[0])           # the hypercharge look's own score
        patch = frame[int(cy) - 26:int(cy) + 27, int(cx) - 26:int(cx) + 27]
        lit.append(float(patch[disc][:, 2].mean()) > 150)     # mean red: dull ~117, lit ~178
    cap.release()

    transitions = sum(1 for a, b in zip(verdicts, verdicts[1:]) if a != b)
    assert not verdicts[0] and not verdicts[-1]
    assert transitions == 2, f"the gate dropped out mid-match: {transitions} transitions"
    assert ms.anchor[2] == pytest.approx(cal.viewport_button(cal.gate_anchor)[2], abs=1.0)

    first = verdicts.index(True)
    last = len(verdicts) - 1 - verdicts[::-1].index(True)
    # The clip really holds the look under test: lit for 15.2 s (measured), the match's end.
    run = longest_lit = lit_end = 0
    for i in range(first, last + 1):
        run = run + 1 if lit[i] else 0
        if run > longest_lit:
            longest_lit, lit_end = run, i
    assert longest_lit > 10 * 30, f"lit for only {longest_lit} frames"
    lit_start = lit_end - longest_lit + 1

    # Sub-threshold runs inside the match. The one ending on `last` is the exit's own tail on the
    # results screen, the hysteresis working.
    def dips(xs):
        out, run = [], 0
        for i in range(first, last + 1):
            if xs[i] < cal.threshold:
                run += 1
            elif run:
                out.append(run)
                run = 0
        return out
    assert dips(hyper) and max(dips(hyper)) < cal.exit_samples, f"in-match dips {dips(hyper)}"
    assert not dips(scores), f"the gate dipped in-match: {dips(scores)}"

    lit_scores = hyper[lit_start:lit_end + 1]
    dull = [s for s, on in zip(hyper[first:lit_start], lit[first:lit_start]) if not on]
    # Measured 0.676 and 0.795; each keeps roughly a third of its margin in reserve.
    assert float(np.median(lit_scores)) > 1.3 * cal.threshold
    assert min(dull) > 1.5 * cal.threshold, f"dull floor {min(dull):.3f}"
    # Measured 0.071 for the gate, every look's best.
    assert max(scores[last + 1:]) < 0.25 * cal.threshold, "the results screen"

    # The loop's gadget window (`loop.GADGET_LOOK_SECONDS`). Bull owns a gadget and the window
    # sees it, but not on the frame the gate opens on here: the disc flashes white just after it
    # comes up, reading 0.19-0.27, and replayed every frame the gate opens 0.17 s in, inside the
    # flash. The live gate at 12 Hz opens after it. That flash is why one look is not a verdict.
    assert not gadget[0] and any(gadget), gadget[:15]


@pytest.mark.vision
def test_gate_opens_on_the_real_app_where_no_hypercharge_is_drawn():
    """Design §5.2, 2026-11-04: the user's own account in the real app, a Starfruit Supernova
    match (BlueStacks 1440p, 30 fps). Its brawler, Shelly at power 3, owns no gadget and no
    hypercharge, so neither disc is drawn. The gate that sat on the hypercharge alone never
    opened: the live run waited through a whole match and pressed nothing.

    Replayed at the live tick rate, not every frame. The dips here are the player holding a Super
    drag, 2-3 ticks at 12 Hz, and a 30 fps replay would count each as 5-8 samples against
    `exit_samples` 4, a gate the live loop never runs.

    Measured through this path: opens at 15.73 s from a lobby peak of 0.208; in-match dips of 1,
    3, 1, 2 and 3 ticks, the longest one short of `exit_samples`; the hypercharge alone peaks at
    0.312. At the open, refine finds the uncharged Super and nothing else (0.762), and over the
    loop's 1 s gadget window the gadget's empty spot peaks at 0.132.
    """
    cv2 = pytest.importorskip("cv2")
    from brawl_deployment.config import load_deployment_config
    from brawl_deployment.loop import GADGET_LOOK_SECONDS
    cal = Calibration.load()
    cap = cv2.VideoCapture(REAL_APP_CLIP)
    if not cap.isOpened():
        pytest.skip(f"{REAL_APP_CLIP} not available")
    hz = load_deployment_config().loop_tick_hz
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ticks = {round(k * fps / hz) for k in range(int(n_frames * hz / fps) + 1)}
    ms, hyper_only = MatchState(cal), MatchState.on(cal, "hypercharge")
    verdicts, scores, hyper_scores, gadget = [], [], [], []
    opened_at = drawn = None
    i = -1
    while cap.grab():
        i += 1
        if i not in ticks:
            continue
        frame = to_viewport(cap.retrieve()[1], cal.viewport)
        was_in = ms.in_match
        verdicts.append(ms.update(frame))
        hyper_only.update(frame)
        if verdicts[-1] and not was_in:
            ms.refine(frame)                      # as loop._begin_match does
            opened_at, drawn = i / fps, ms.drawn
        if ms.in_match and len(gadget) < round(GADGET_LOOK_SECONDS * hz):
            gadget.append(ms.button_drawn(frame, "gadget"))
        scores.append(ms.last_score)
        hyper_scores.append(hyper_only.last_score)
    cap.release()

    transitions = sum(1 for a, b in zip(verdicts, verdicts[1:]) if a != b)
    assert transitions == 1 and verdicts[-1], "the gate must open once and hold to the clip's end"
    assert 14.0 < opened_at < 17.0, f"opened at {opened_at:.2f} s"
    assert drawn == ("super (uncharged)",), drawn

    first = verdicts.index(True)
    # The lobby: every tick before the run that opened the gate.
    assert max(scores[:first - cal.enter_samples + 1]) < 0.70 * cal.threshold
    dips, run = [], 0
    for s in scores[first:]:
        if s < cal.threshold:
            run += 1
        elif run:
            dips.append(run)
            run = 0
    assert dips and max(dips) < cal.exit_samples, f"in-match dips {dips}"
    # The old gate, on the same ticks: it never opens.
    assert max(hyper_scores) < cal.threshold, f"the hypercharge alone peaked {max(hyper_scores):.3f}"

    # The loop's gadget window: Shelly at power 3 draws no gadget on any tick of it.
    assert len(gadget) == round(GADGET_LOOK_SECONDS * hz)
    assert not any(d for d, _ in gadget), gadget
    assert max(s for _, s in gadget) < 0.5 * cal.threshold


# ---- adb discovery ----------------------------------------------------------------------------

CONF = """bst.feature.rooting="0"
bst.instance.Pie64.adb_port="5555"
bst.instance.Pie64.status.adb_port="5555"
bst.instance.Rvc64_2.status.adb_port="5585"
bst.instance.Pie64.status.session_id="6"
"""


def test_the_adb_port_is_read_from_bluestacks_own_config(tmp_path):
    """BlueStacks publishes the port it listens on, so nobody has to read it off a settings screen
    and copy it into a yaml file -- which is what BRAWL_DEPLOYMENT_DESIGN.md 10.2 asked for until
    this existed. Note the two keys per instance: `adb_port` is the CONFIGURED one and
    `status.adb_port` the one it is actually serving. Only the latter is authoritative."""
    conf = tmp_path / "bluestacks.conf"
    conf.write_text(CONF)
    assert find_adb_serial(conf) == "127.0.0.1:5555"
    assert find_adb_serial(conf, "Rvc64_2") == "127.0.0.1:5585"


def test_a_second_instance_does_not_get_the_first_ones_port(tmp_path):
    """The reason this is read rather than defaulted to 5555. A second instance is assigned 5585,
    and connecting to the wrong one either fails as a timeout that looks like the emulator being
    closed, or -- worse -- succeeds against the wrong emulator."""
    conf = tmp_path / "bluestacks.conf"
    conf.write_text('bst.instance.Rvc64_2.status.adb_port="5585"\n')
    assert find_adb_serial(conf) == "127.0.0.1:5585"

    with pytest.raises(KeyError, match="Pie64"):
        find_adb_serial(conf, "Pie64")


def test_a_missing_or_portless_config_raises_rather_than_guessing(tmp_path):
    """Same fail-closed reasoning as everywhere else in this package: a fabricated serial produces
    a connect timeout indistinguishable from a closed emulator, which is a much worse thing to
    debug than a named error at startup."""
    with pytest.raises(FileNotFoundError):
        find_adb_serial(tmp_path / "nope.conf")

    empty = tmp_path / "empty.conf"
    empty.write_text('bst.feature.rooting="0"\n')
    with pytest.raises(ValueError, match="adb_port"):
        find_adb_serial(empty)


def test_an_auto_aimed_attack_is_a_bare_tap_on_the_attack_button():
    """Attack value 4 (`action.auto_aim`, 2026-09-26): the game aims a bare tap at the nearest
    target by itself, which is exactly what the sim's value 4 models, so this is the second press
    with no drag. Down on the attack point, the same origin as value 1, up on the next settle,
    never a move, whatever bearing is passed; and no aim point, like the gadget."""
    b = NullBackend()
    btn = Buttons(b, attack=(1000.0, 500.0), super_=(3.0, 4.0), gadget=(1559.9, 901.1))
    assert btn.origin(ATTACK_AUTO) == btn.origin(ATTACK_FIRE) == (1000.0, 500.0)
    assert btn.press(ATTACK_AUTO, math.pi / 2)
    assert b.log == [("down", SLOT_TAP, 1000.0, 500.0)]
    assert btn.is_held
    btn.settle()
    assert b.log[1:] == [("up", SLOT_TAP)]
    assert not btn.is_held and SLOT_TAP not in b.contacts
    btn.settle()
    assert len(b.log) == 2
    with pytest.raises(ValueError, match="no aim point"):
        btn.aim_point(ATTACK_AUTO, 0.0)


def test_a_super_with_no_bearing_is_a_bare_tap_on_the_super_button():
    """An idle super (2026-10-06): the sim aims it like the game's tap-to-fire, so the shadow
    hands over no bearing and the press is down on the super button, up on the next settle, never
    a move. With a bearing it is still the drag. An attack with no bearing is a bug upstream, so
    it raises rather than tapping, which the game would auto-aim."""
    b = NullBackend()
    btn = Buttons(b, attack=(1000.0, 500.0), super_=(1462.5, 900.0), gadget=(1559.9, 901.1))
    assert btn.press(ATTACK_SUPER, None)
    assert b.log == [("down", SLOT_TAP, 1462.5, 900.0)]
    btn.settle()
    assert b.log[1:] == [("up", SLOT_TAP)]
    assert not btn.is_held and SLOT_TAP not in b.contacts

    assert btn.press(ATTACK_SUPER, 0.0)
    btn.settle()
    assert b.log[-1][0] == "move"
    btn.settle()

    with pytest.raises(ValueError, match="needs a bearing"):
        btn.press(ATTACK_FIRE, None)


@pytest.mark.parametrize("bearing", [k * math.pi / 8 for k in range(16)] + [0.3, 2.0, -1.1])
def test_a_full_super_drags_to_the_screen_edge_along_its_bearing(bearing):
    """Bull's charge (BULL_HERO_PLAN.md Q6): the drag sets its length and the shadow models it at
    full length, so `full` drags the super as far as the screen allows. The point must stay on
    the bearing EXACTLY, because the backend would clamp an off-screen point one axis at a time
    and bend it. So: on the ray, on the screen, 2 px inside the nearer exit, and never shorter
    than the plain radius. The super is the real one, 179 px above the bottom edge."""
    screen = (1920, 1080)
    btn = Buttons(NullBackend(), attack=(1690.0, 594.0), super_=(1462.5, 900.0),
                  gadget=(1559.9, 901.1), screen=screen)
    ox, oy = btn.super_
    x, y = btn.aim_point(ATTACK_SUPER, bearing, full=True)
    r = math.hypot(x - ox, y - oy)
    assert math.atan2(y - oy, x - ox) == pytest.approx(math.atan2(math.sin(bearing),
                                                                  math.cos(bearing)), abs=1e-9)
    assert 0 <= x <= screen[0] - 1 and 0 <= y <= screen[1] - 1
    assert r > btn.aim_radius_px
    one_more = (ox + (r + 2.5) * math.cos(bearing), oy + (r + 2.5) * math.sin(bearing))
    assert not (0 <= one_more[0] <= screen[0] - 1 and 0 <= one_more[1] <= screen[1] - 1)


def test_full_reaches_only_the_super_and_needs_the_screen():
    """`press(..., full=True)` is the loop passing `charge_super` on every press, so it must
    leave an attack's drag at `aim_radius_px` and a tap a tap. A full drag with no `screen` has
    nowhere to stop, so it raises rather than guessing one."""
    b = NullBackend()
    btn = Buttons(b, attack=(1000.0, 500.0), super_=(1462.5, 900.0), gadget=(1559.9, 901.1),
                  screen=(1920, 1080))
    btn.press(ATTACK_FIRE, 0.0, full=True)
    btn.settle()
    assert b.log[-1] == ("move", SLOT_TAP, 1000.0 + btn.aim_radius_px, 500.0)
    btn.settle()
    btn.press(ATTACK_SUPER, 0.0, full=True)
    btn.settle()
    assert b.log[-1] == ("move", SLOT_TAP, 1917.0, 900.0)
    btn.settle()
    btn.press(ATTACK_SUPER, None, full=True)
    btn.settle()
    assert b.log[-2:] == [("down", SLOT_TAP, 1462.5, 900.0), ("up", SLOT_TAP)]

    blind = Buttons(NullBackend(), attack=(1000.0, 500.0), super_=(1462.5, 900.0),
                    gadget=(1559.9, 901.1))
    with pytest.raises(ValueError, match="needs `screen`"):
        blind.press(ATTACK_SUPER, 0.0, full=True)
