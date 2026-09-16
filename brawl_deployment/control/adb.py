"""Touch injection over a PERSISTENT `adb shell`: the device-side shell's builtin `print` writes
packed `input_event` structs into the emulator's multitouch device. See BRAWL_DEPLOYMENT_DESIGN.md
4.1 and 4.6.

MEASURED ON THIS SETUP, and every design choice below follows from one of these numbers:

    persistent adb shell, bare round trip                 0.51 ms   (p95 0.65)
    four chained shell builtins                           0.48 ms   (p95 0.61)
    13 events in ONE builtin write to the device          0.58 ms   (max 0.61)   <- an aimed press
    four `sendevent` process spawns, emulator idle        5.77 ms   (p95 5.94)   <- 2026-09-08
    ONE `sendevent` spawn, game loaded                   ~45 ms                  <- 2026-09-16
    adb exec-out screencap                              ~290 ms     (why capture goes through mss, not here)

**Why a persistent shell and not `adb shell <cmd>` per event.** Each `adb shell` invocation is a
fresh TCP conversation, a fresh device-side shell and a process teardown. One long-lived shell
turns every touch step into a single pipe write.

**Why the shell writes the bytes itself, and not `sendevent` (REVISED 2026-09-16).** `sendevent`
is one process spawn per event, and a spawn inside the VM costs whatever the game leaves it: 1.4 ms
on an idle emulator, ~45 ms with Nulls Brawl loaded. An aimed press is 13 events (down 6, drag 4,
lift 3) and a joystick turn 4, so with the game running one attack was 620 ms of serial device
time against a 250 ms decision -- the pipe backed up, touches landed seconds late, and the GAME
stuttered under the fork storm. That was the "insane lag whenever the agent attacks". The fix is to
spawn nothing: `print -n '<escaped bytes>' >&3` is an mksh BUILTIN writing to a file descriptor the
shell opened on the device at startup (`exec 3>/dev/input/event4`), so a whole batch is one
`write(2)` of packed structs, and its cost does not depend on the event count or on the game.

**Why the bytes travel as TEXT.** BlueStacks ships **adb 1.0.36**, whose Windows stdin forwarding
fails on binary payloads (`OSError 22` on the first flush, device-side shell still alive). So the
pipe carries octal escapes (`\\0ooo`, five ASCII characters a byte; ~1.6 KB for a press) and the
device-side `print` turns them into bytes. Verified on this image: `print` is a builtin, both
`\\0ooo` and `\\xhh` escapes produce NUL bytes correctly, and the shell can open the device.

**Why the event size is read off the device, not assumed.** The kernel takes `struct input_event`
in the WRITER's ABI -- 24 bytes from a 64-bit process (two 8-byte time fields), 16 from a 32-bit
one -- and rejects any other length with EINVAL rather than injecting garbage, which is what makes
the startup handshake a real check. The timestamps are ignored on the write path (`evdev_write`
-> `input_inject_event` takes only type/code/value), so they go out as zero.

**Why no root.** `/dev/input/event4` is writable by uid 2000(shell), which is in group
1004(input). Checked, not assumed: `[ -w /dev/input/event4 ]` returns true over plain adb.

**Coordinates are NOT pixels.** The device advertises `ABS_MT_POSITION_X/Y` over 0..32767 against a
1920x1080 screen, so every pixel coordinate is scaled on the way in. Getting this wrong puts every
touch in the top-left 6% of the screen, which looks like "injection does not work" rather than like
a units bug.
"""
import re
import shutil
import struct
import subprocess
import threading
from pathlib import Path

from .backend import SLOT_MOVE, SLOT_TAP

# Linux input event codes (uapi/linux/input-event-codes.h). Spelled out rather than imported
# because there is no dependency that provides them and three of these are easy to typo.
EV_SYN, SYN_REPORT = 0, 0
EV_KEY, BTN_TOUCH = 1, 330
EV_ABS = 3
ABS_MT_SLOT = 47
ABS_MT_TOUCH_MAJOR = 48
ABS_MT_POSITION_X = 53
ABS_MT_POSITION_Y = 54
ABS_MT_TRACKING_ID = 57

# MEASURED on this emulator via `getevent -pl`: event4 is the only device exposing ABS_MT_SLOT
# (max 15, so 16 concurrent contacts) and ABS_MT_POSITION_X/Y (max 32767). Not hardcoded blindly --
# `find_touch_device` re-derives it and this is the fallback.
DEFAULT_DEVICE = "/dev/input/event4"
DEFAULT_ABS_MAX = 32767

# `struct input_event` as the device-side shell writes it: {tv_sec, tv_usec} in the shell's own
# `long`, then u16 type, u16 code, s32 value. Keyed by the size the shell's ELF class implies.
EVENT_STRUCTS = {24: struct.Struct("<qqHHi"), 16: struct.Struct("<iiHHi")}
# The descriptor the device-side shell holds open on the touch device for the life of the session.
DEVICE_FD = 3
# How long the device-side shell gets to answer the startup handshake before the session is
# declared dead. Generous: the first command after `adb shell` waits on the shell itself starting.
HANDSHAKE_TIMEOUT_S = 15.0

_BLUESTACKS_ADB = Path(r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe")
_BLUESTACKS_CONF = Path(r"C:\ProgramData\BlueStacks_nxt\bluestacks.conf")
_ADB_PORT_LINE = re.compile(r'\s*bst\.instance\.([^.]+)\.status\.adb_port\s*=\s*"?(\d+)"?\s*$')


def find_adb_serial(conf=None, instance: str | None = None) -> str:
    """`127.0.0.1:<port>` for a BlueStacks instance, read from BlueStacks' own config.

    The emulator publishes the port it listens on as
    `bst.instance.<name>.status.adb_port="5555"` in `bluestacks.conf`, so there is no reason for a
    human to read it off a settings screen and copy it into a yaml file -- which is what
    BRAWL_DEPLOYMENT_DESIGN.md 10.2 was asking for until this existed.

    The port is **per instance and not always 5555** (a second instance gets 5585, and BlueStacks
    reassigns it on some upgrades), which is why it is read rather than defaulted. With no
    `instance`, the first one the file names wins; MEASURED 2026-09-08 on this machine that is
    `Pie64` at 5555, matching `AdbTouchBackend`'s default. Raises rather than guessing: a wrong
    serial fails as a connect timeout, which looks exactly like the emulator being closed.
    """
    conf = Path(conf) if conf is not None else _BLUESTACKS_CONF
    if not conf.exists():
        raise FileNotFoundError(
            f"no BlueStacks config at {conf}. Pass an explicit serial, or check the install path."
        )
    ports = {}
    for line in conf.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _ADB_PORT_LINE.match(line)
        if m:
            ports[m.group(1)] = int(m.group(2))
    if not ports:
        raise ValueError(f"{conf} names no `bst.instance.<name>.status.adb_port`")
    if instance is None:
        instance = next(iter(ports))
    elif instance not in ports:
        raise KeyError(f"{conf} has no instance {instance!r}; it knows {sorted(ports)}")
    return f"127.0.0.1:{ports[instance]}"


def find_adb() -> str:
    """The adb binary to drive. Prefers one on PATH (likely newer) over the BlueStacks-bundled
    1.0.36, but either works -- this module deliberately does not need a modern adb."""
    on_path = shutil.which("adb")
    if on_path:
        return on_path
    if _BLUESTACKS_ADB.exists():
        return str(_BLUESTACKS_ADB)
    raise FileNotFoundError(
        "no adb found. Install Android platform-tools, or check that BlueStacks is installed at "
        f"{_BLUESTACKS_ADB.parent}."
    )


def find_touch_device(adb: str, serial: str) -> tuple[str, int]:
    """`(device_path, abs_max)` for the multitouch device, by asking the emulator rather than
    trusting `DEFAULT_DEVICE`.

    The touch device's node number is not stable across BlueStacks versions or instance types, and
    a wrong node fails silently -- the writes succeed, they just go somewhere that is not a
    touchscreen. Identified by the two properties only a Type-B multitouch device has:
    `ABS_MT_SLOT` and `ABS_MT_POSITION_X`.
    """
    out = subprocess.run([adb, "-s", serial, "shell", "getevent", "-pl"],
                         capture_output=True, text=True, timeout=30).stdout
    device, abs_max = None, DEFAULT_ABS_MAX
    current = None
    found_slot = False
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("add device"):
            # A new device block: the previous one only counts if it showed BOTH markers.
            current = stripped.rsplit(" ", 1)[-1]
            found_slot = False
        elif "ABS_MT_SLOT" in stripped:
            found_slot = True
        elif "ABS_MT_POSITION_X" in stripped and found_slot and current:
            device = current
            if "max " in stripped:
                try:
                    abs_max = int(stripped.split("max ")[1].split(",")[0])
                except (IndexError, ValueError):
                    pass
            break
    return device or DEFAULT_DEVICE, abs_max


class AdbTouchBackend:
    """A held-contact touch injector over one long-lived `adb shell`.

    Not thread-safe and deliberately so: there is one loop and one agent, and a lock here would be
    machinery for a concurrency this package does not have (see the package docstring).
    """

    def __init__(self, serial: str = "127.0.0.1:5555", screen: tuple[int, int] = (1920, 1080),
                 adb: str | None = None, device: str | None = None, abs_max: int | None = None,
                 connect: bool = True):
        self.adb = adb or find_adb()
        self.serial = serial
        self.screen_w, self.screen_h = screen

        if connect:
            subprocess.run([self.adb, "connect", serial], capture_output=True, timeout=30)

        if device is None or abs_max is None:
            found_device, found_max = find_touch_device(self.adb, serial)
            device = device or found_device
            abs_max = abs_max or found_max
        self.device, self.abs_max = device, abs_max

        self._proc = subprocess.Popen(
            [self.adb, "-s", serial, "shell"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._contacts: dict[int, int] = {}     # slot -> tracking id
        self._next_tracking_id = 1
        self.event_struct = EVENT_STRUCTS[24]
        # Opens the device on the shell side and proves the write path -- see `_handshake`. Also
        # what used to be a 0.4 s sleep: the handshake returns when the shell is actually ready.
        self._handshake()

    def _handshake(self) -> None:
        """One exchange that leaves the session ready or raises with the reason.

        The shell reports its own ELF class (byte 4 of its executable: 02 is 64-bit), opens the
        device on `DEVICE_FD`, and writes ONE `SYN_REPORT` of the size that class implies. A
        `SYN_REPORT` with no preceding event changes nothing -- the kernel drops it as a duplicate
        sync -- and a struct of the wrong size is refused with EINVAL, so `OK` here means the whole
        path works and the size is right. Stdout is read only here, then never again: every later
        command writes to the device, not to the pipe.
        """
        proc = self._proc
        script = (f"od -An -tx1 -j4 -N1 /proc/$$/exe; "
                  f"[ -w {self.device} ] && exec {DEVICE_FD}>{self.device} && echo OPEN "
                  f"|| echo NOOPEN")
        proc.stdin.write((script + "\n").encode())
        proc.stdin.flush()
        lines = self._read_lines(2)
        if lines is None:
            self._kill()
            raise RuntimeError(f"adb shell on {self.serial} did not answer within "
                               f"{HANDSHAKE_TIMEOUT_S:.0f}s; is the emulator running?")
        elf_class, opened = lines
        if opened != "OPEN":
            self._kill()
            raise RuntimeError(f"the shell on {self.serial} cannot open {self.device} for "
                               f"writing; `getevent -pl` should list it and uid 2000 must be in "
                               f"group input")
        size = 24 if elf_class == "02" else 16
        self.event_struct = EVENT_STRUCTS[size]
        self._send([(EV_SYN, SYN_REPORT, 0)], probe=True)
        verdict = self._read_lines(1)
        if verdict != ["OK"]:
            self._kill()
            raise RuntimeError(f"a {size}-byte input_event write to {self.device} was refused "
                               f"(shell ELF class {elf_class!r}): the struct size does not match "
                               f"the shell's ABI, or the device is not an evdev node")

    def _read_lines(self, n: int) -> list[str] | None:
        """`n` stripped lines from the shell's stdout, or None on `HANDSHAKE_TIMEOUT_S` or EOF. A
        thread, because a pipe read has no timeout on Windows -- startup only, never in the loop."""
        out: list[str] = []

        def reader():
            for _ in range(n):
                line = self._proc.stdout.readline()
                if not line:
                    return
                out.append(line.decode(errors="replace").strip())

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(HANDSHAKE_TIMEOUT_S)
        return out if len(out) == n else None

    # -- coordinates --

    def _to_raw(self, x: float, y: float) -> tuple[int, int]:
        """Screen pixels -> the device's 0..abs_max axis units.

        `screen_w - 1` rather than `screen_w` in the denominator so the rightmost pixel maps to
        `abs_max` exactly rather than one unit past it. Clamped because a joystick anchor near an
        edge plus a saturation radius can legitimately compute a point just off-screen, and a
        clamped touch is a better failure than a rejected one.
        """
        rx = round(x * self.abs_max / max(self.screen_w - 1, 1))
        ry = round(y * self.abs_max / max(self.screen_h - 1, 1))
        return max(0, min(self.abs_max, rx)), max(0, min(self.abs_max, ry))

    # -- transport --

    def _send(self, events: list[tuple[int, int, int]], probe: bool = False) -> None:
        """One touch step's events as ONE write on the device: a `print` builtin, no spawns.

        The line is `print -n '<octal escapes>' >&3` -- ASCII on the pipe (adb 1.0.36 cannot
        forward binary), packed structs on the device. `probe` appends a verdict for the
        handshake to read; the hot path never produces output.
        """
        if not self.is_alive:
            return
        line = self.encode(events)
        if probe:
            line += " && echo OK || echo FAIL"
        try:
            self._proc.stdin.write(line.encode("ascii") + b"\n")
            self._proc.stdin.flush()
        except (OSError, ValueError, AttributeError):
            # The shell died under us. Do not raise: every caller here is either the hot loop or a
            # fail-closed handler, and both want `is_alive` to go False rather than an exception.
            self._kill()

    def encode(self, events: list[tuple[int, int, int]]) -> str:
        """The shell command for `events`: every byte of every struct as a `\\0ooo` escape, so the
        line is plain ASCII whatever the values are. Public so a test can decode it back."""
        raw = b"".join(self.event_struct.pack(0, 0, t, c, v) for t, c, v in events)
        payload = "".join(f"\\0{b:03o}" for b in raw)
        return f"print -n '{payload}' >&{DEVICE_FD}"

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        self._contacts.clear()
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    # -- InputBackend --

    def down(self, slot: int, x: float, y: float) -> None:
        rx, ry = self._to_raw(x, y)
        tracking_id = self._next_tracking_id
        self._next_tracking_id += 1
        first_contact = not self._contacts
        events = [
            (EV_ABS, ABS_MT_SLOT, slot),
            (EV_ABS, ABS_MT_TRACKING_ID, tracking_id),
            (EV_ABS, ABS_MT_POSITION_X, rx),
            (EV_ABS, ABS_MT_POSITION_Y, ry),
            (EV_ABS, ABS_MT_TOUCH_MAJOR, 8),
        ]
        # BTN_TOUCH is a whole-device "is anything touching", NOT a per-slot flag. Emitting it on
        # every down would report a fresh press each time a second finger lands, which is how a
        # held joystick contact ends up being re-acquired by the game every time we shoot.
        if first_contact:
            events.append((EV_KEY, BTN_TOUCH, 1))
        events.append((EV_SYN, SYN_REPORT, 0))
        self._contacts[slot] = tracking_id
        self._send(events)

    def move(self, slot: int, x: float, y: float) -> None:
        if slot not in self._contacts:
            return
        rx, ry = self._to_raw(x, y)
        self._send([
            (EV_ABS, ABS_MT_SLOT, slot),
            (EV_ABS, ABS_MT_POSITION_X, rx),
            (EV_ABS, ABS_MT_POSITION_Y, ry),
            (EV_SYN, SYN_REPORT, 0),
        ])

    def up(self, slot: int) -> None:
        if slot not in self._contacts:
            return
        del self._contacts[slot]
        events = [
            (EV_ABS, ABS_MT_SLOT, slot),
            (EV_ABS, ABS_MT_TRACKING_ID, -1),       # -1 is the Type-B "this slot is now empty"
        ]
        if not self._contacts:
            events.append((EV_KEY, BTN_TOUCH, 0))
        events.append((EV_SYN, SYN_REPORT, 0))
        self._send(events)

    def release_all(self) -> None:
        """The fail-closed primitive (design 8). Must not raise -- `_send` already swallows a dead
        transport, and iterating a copy keeps `up`'s mutation of `_contacts` safe."""
        for slot in list(self._contacts):
            self.up(slot)

    def close(self) -> None:
        self.release_all()
        self._kill()

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def injects(self) -> bool:
        """This one really does reach the device. See `InputBackend.injects`."""
        return True

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False
