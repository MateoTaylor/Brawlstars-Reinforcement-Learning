"""Locate the emulator window, resolve which monitor to capture, and notice when it moves.

See BRAWL_DEPLOYMENT_DESIGN.md 8 for where this sits in the interlock.

**This module's job is to make `mss` and the calibration agree about what is being captured.**
`DeployCapture` grabs a whole *monitor* and crops to whatever `detect_content_box` finds; it has
no idea which window it is looking at. Everything downstream -- the homography, the HUD mask, the
tap anchors -- is calibrated against a BlueStacks client area filling that monitor. So somebody has
to check that the window is where it was when the calibration was taken, and that somebody is here.

**Identify the emulator by its PROCESS IMAGE, not by window title or class.** The title is the
instance name and the user can rename it; the class name is Qt's (`Qt5152QWindowIcon` and friends)
and changes with the BlueStacks build. The executable is stable, and -- more useful -- matching it
against `control/adb.py`'s install directory ties this window to *the same installation* the ADB
serial was read from. Two BlueStacks installs, or a stale window from a previous version, then
fail loudly rather than being driven by an adb connection to the other one.

**DPI awareness is set before any geometry call, and this is load-bearing.** Without it Windows
lies to a process about window coordinates, scaling them by the display's DPI factor, while `mss`
(which makes itself DPI-aware) reports physical pixels. The two then disagree by exactly the
scale factor, silently, and the window "is" somewhere it is not. This is the same class of bug as
the silent viewport rescale in `capture.py`: geometry that looks self-consistent and is wrong.

**Focus is NOT required and is deliberately not checked.** Touch goes in over ADB, to the
emulator, not through `SendInput` to the foreground window -- so the agent keeps playing with the
window unfocused, which is the whole point of the ADB backend. What *does* matter is narrower and
is what `WindowGuard` checks:

- the window is **still there and not minimized** -- otherwise the capture is of a desktop;
- its client area **has not moved or resized** -- the content box is computed once and cached
  (`capture.py`), so a window that moves invalidates every calibrated coordinate with nothing
  downstream noticing;
- optionally, that **nothing is drawn on top of it** -- which `find_occluders` answers directly,
  and which the cached content box cannot: an occluder appearing mid-session corrupts pixels
  inside an unchanged crop. That is the mid-session half of the hazard whose startup half
  `DeployCapture.MIN_HEIGHT_COVERAGE` closes.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

# The BlueStacks 5 ("nxt") player process. The launcher and the multi-instance manager are
# separate executables with their own windows; only this one renders a running Android instance.
PLAYER_EXE = "HD-Player.exe"

# Same directory `control/adb.py` finds `HD-Adb.exe` in. Kept as a *hint*, not a requirement: a
# non-default install location is a supported thing to have, and refusing to run in one would be
# inventing a constraint. It is reported so a mismatch is visible.
INSTALL_DIR = Path(r"C:\Program Files\BlueStacks_nxt")

# (left, top, right, bottom) in virtual-screen pixels, right/bottom EXCLUSIVE -- Win32's own
# convention for RECT, kept rather than converted so these can be compared to GetWindowRect output
# without a mental translation step at every boundary.
Rect = tuple[int, int, int, int]

GA_ROOT = 2
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
MONITOR_DEFAULTTONULL = 0
MONITORINFOF_PRIMARY = 1

_dpi_aware = False


def set_dpi_aware() -> None:
    """Opt this process into physical pixels. Idempotent; safe to call from anywhere.

    Must happen before the first `GetWindowRect` -- and, in practice, before `mss` is used, since
    `mss` sets it too and whichever runs first wins for the whole process. Windows offers no way
    to *lower* awareness once raised, so there is no ordering hazard beyond "do it early".

    `SetProcessDpiAwarenessContext` (Win10 1703+) is preferred because per-monitor-v2 stays
    correct when a window is dragged between monitors of different DPI; `SetProcessDPIAware` is
    the system-wide fallback and is enough for the single-monitor setup this project targets.
    """
    global _dpi_aware
    if _dpi_aware:
        return
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    try:
        # -4 == DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        user32.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        try:
            user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass   # a DPI-unaware process still works at 100% scaling, which is this setup
    _dpi_aware = True


def _user32() -> ctypes.WinDLL:
    """user32 with the signatures that matter declared.

    **The restypes are not decoration.** ctypes defaults an undeclared return to `c_int`, which
    truncates a 64-bit `HWND` to 32 bits. Handles are usually small enough to survive that, which
    is precisely why it is dangerous: it works for months and then one process gets a high handle
    and every window comparison silently fails.
    """
    u = ctypes.WinDLL("user32", use_last_error=True)
    u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    u.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    u.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    u.GetForegroundWindow.restype = wintypes.HWND
    u.WindowFromPoint.restype = wintypes.HWND
    u.WindowFromPoint.argtypes = [wintypes.POINT]
    u.GetAncestor.restype = wintypes.HWND
    u.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    u.MonitorFromWindow.restype = wintypes.HMONITOR
    u.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    return u


class _MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]


def _rect(r: wintypes.RECT) -> Rect:
    return (int(r.left), int(r.top), int(r.right), int(r.bottom))


def _window_text(u: ctypes.WinDLL, hwnd: int) -> str:
    n = u.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    u.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def _class_name(u: ctypes.WinDLL, hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    u.GetClassNameW(hwnd, buf, 256)
    return buf.value


def _exe_path(hwnd: int, u: ctypes.WinDLL) -> str | None:
    """Full path of the process owning `hwnd`, or `None` if it cannot be read.

    `None` is a real outcome, not just an error path: a process running at a higher integrity
    level refuses `OpenProcess` even for `QUERY_LIMITED_INFORMATION`. Callers fall back to name
    matching rather than treating it as "not the emulator".
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    pid = wintypes.DWORD()
    u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return None
    try:
        size = wintypes.DWORD(32768)
        buf = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return None
        return buf.value
    finally:
        kernel32.CloseHandle(handle)


@dataclass(frozen=True)
class WindowInfo:
    """One top-level window, as Win32 sees it. Plain data -- everything that decides anything
    takes one of these rather than an `hwnd`, so the deciding is testable without a desktop."""
    hwnd: int
    title: str
    class_name: str
    exe: str | None
    rect: Rect            # includes the frame and title bar
    client: Rect          # the drawn area, in SCREEN coordinates -- what the game fills
    minimized: bool

    @property
    def client_size(self) -> tuple[int, int]:
        return (self.client[2] - self.client[0], self.client[3] - self.client[1])

    @property
    def exe_name(self) -> str:
        return Path(self.exe).name if self.exe else ""


def enumerate_windows() -> list[WindowInfo]:
    """Every visible top-level window, with its geometry and owning executable."""
    set_dpi_aware()
    u = _user32()
    out: list[WindowInfo] = []

    def visit(hwnd, _lparam):
        if not u.IsWindowVisible(hwnd):
            return True
        wr, cr = wintypes.RECT(), wintypes.RECT()
        u.GetWindowRect(hwnd, ctypes.byref(wr))
        u.GetClientRect(hwnd, ctypes.byref(cr))
        origin = wintypes.POINT(0, 0)
        u.ClientToScreen(hwnd, ctypes.byref(origin))
        out.append(WindowInfo(
            hwnd=int(hwnd),
            title=_window_text(u, hwnd),
            class_name=_class_name(u, hwnd),
            exe=_exe_path(hwnd, u),
            rect=_rect(wr),
            client=(origin.x, origin.y, origin.x + int(cr.right), origin.y + int(cr.bottom)),
            minimized=bool(u.IsIconic(hwnd)),
        ))
        return True

    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(visit)
    u.EnumWindows(proc, 0)
    return out


def select_emulator(windows: list[WindowInfo], exe_name: str = PLAYER_EXE) -> WindowInfo:
    """Pick the emulator out of a window list. Pure -- `find_emulator_window` supplies the list.

    Raises rather than guessing when there is not exactly one candidate. **Ambiguity here is not a
    tie to break, it is a fact worth surfacing:** two player windows means two running instances,
    and `find_adb_serial` returns the port of *one* of them. Driving the window of one instance
    with the adb connection of another produces an agent that sees a game it is not playing --
    which is a bad afternoon to debug and a one-line message to prevent.
    """
    named = [w for w in windows if w.exe_name.lower() == exe_name.lower()]
    if not named:
        # Nothing matched by executable. Either it is not running, or every OpenProcess failed --
        # tell those apart, because the fixes are completely different.
        unreadable = sum(1 for w in windows if w.exe is None)
        raise LookupError(
            f"no visible window belongs to {exe_name}. Is BlueStacks running with an instance "
            f"started? ({len(windows)} visible top-level windows"
            + (f", {unreadable} whose process could not be read" if unreadable else "") + ")"
        )
    usable = [w for w in named if w.client_size[0] > 0 and w.client_size[1] > 0]
    if not usable:
        raise LookupError(
            f"found {len(named)} {exe_name} window(s), all with an empty client area. The "
            f"instance is probably still booting."
        )
    if len(usable) > 1:
        detail = ", ".join(f"{w.title!r} at {w.client}" for w in usable)
        raise LookupError(
            f"{len(usable)} {exe_name} windows are open, so it is ambiguous which one the adb "
            f"serial addresses: {detail}. Close all but the instance the agent should play."
        )
    return usable[0]


def find_emulator_window(exe_name: str = PLAYER_EXE) -> WindowInfo:
    """The emulator's window. Raises `LookupError` if it is not running or is ambiguous."""
    return select_emulator(enumerate_windows(), exe_name)


def monitor_index_for(rect: Rect, monitors: list[dict]) -> int:
    """The `mss` monitor index whose area contains the centre of `rect`.

    **Never index 0.** `mss.monitors[0]` is the union of every display -- capturing it on a
    multi-monitor desktop would hand `detect_content_box` a frame containing other screens, and
    the content box would come back describing something that is not the game at all. This
    function only ever returns `1..n`, and says so if nothing matches.

    Takes the monitor list rather than an `mss` instance so the mapping can be tested against
    fabricated layouts, including the multi-monitor case this machine does not have.
    """
    cx = (rect[0] + rect[2]) / 2.0
    cy = (rect[1] + rect[3]) / 2.0
    for i, m in enumerate(monitors):
        if i == 0:
            continue
        if (m["left"] <= cx < m["left"] + m["width"]
                and m["top"] <= cy < m["top"] + m["height"]):
            return i
    raise LookupError(
        f"the window centred at ({cx:.0f}, {cy:.0f}) is not on any monitor mss reports "
        f"({monitors[1:]}). A window dragged mostly off-screen does this."
    )


def monitor_rect(hwnd: int) -> Rect:
    """The screen rectangle of the monitor `hwnd` sits on, from Win32 rather than from mss.

    Used only to answer "is the client area filling this monitor". mss's list answers the capture
    question; this answers the geometry one, and they are cross-checked rather than assumed equal.
    """
    set_dpi_aware()
    u = _user32()
    handle = u.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONULL)
    if not handle:
        raise LookupError(f"window {hwnd} is not on any monitor")
    info = _MONITORINFO()
    info.cbSize = ctypes.sizeof(_MONITORINFO)
    if not u.GetMonitorInfoW(handle, ctypes.byref(info)):
        raise OSError(f"GetMonitorInfoW failed for window {hwnd}")
    return _rect(info.rcMonitor)


def sample_points(rect: Rect, inset: float = 0.25) -> list[tuple[int, int]]:
    """Five points inside `rect`: the centre and four inset quadrant centres.

    Five, not one: a window covering a corner of the emulator -- a notification toast, a chat
    overlay -- misses a centre-only probe entirely, and those are the overlaps that actually
    happen. Five, not fifty: this runs in the perception loop and each point costs a
    `WindowFromPoint`, and the failure being caught is "a window is sitting on the game", which is
    not a subtle few-pixel condition.
    """
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    dx, dy = int((x1 - x0) * inset), int((y1 - y0) * inset)
    return [(cx, cy), (cx - dx, cy - dy), (cx + dx, cy - dy),
            (cx - dx, cy + dy), (cx + dx, cy + dy)]


def find_occluders(window: WindowInfo, inset: float = 0.25) -> list[int]:
    """Root window handles drawn over the emulator's client area, at `sample_points`.

    Empty means the sampled points all belong to the emulator. This is the *mid-session* half of
    the occlusion hazard: the content box is cached at startup, so a window appearing later cannot
    move the crop, but it absolutely can put its own pixels where the game should be -- which
    reads downstream as bad detections and a bad gate score, with nothing raising.

    `GetAncestor(GA_ROOT)` because a hit inside the emulator lands on one of Qt's child windows,
    not on the top-level handle, and comparing the raw result would report the emulator as
    occluding itself.
    """
    set_dpi_aware()
    u = _user32()
    seen: list[int] = []
    for x, y in sample_points(window.client, inset):
        hit = u.WindowFromPoint(wintypes.POINT(x, y))
        if not hit:
            continue
        root = int(u.GetAncestor(hit, GA_ROOT) or hit)
        if root != window.hwnd and root not in seen:
            seen.append(root)
    return seen


@dataclass(frozen=True)
class WindowState:
    """What `WindowGuard` pinned at startup. Everything the calibration depends on, and nothing
    else -- the title is here for the error message, not for comparison, because BlueStacks
    rewrites it (FPS counters, instance names) while the geometry stays put."""
    hwnd: int
    title: str
    client: Rect
    monitor_index: int
    monitor: Rect

    @property
    def fills_monitor(self) -> bool:
        return self.client == self.monitor


class WindowFault(str):
    """Why the window check failed: the operator-facing sentence, tagged with a `kind`.

    A `str` subclass on purpose. `check()`'s contract is "return a reason, do not raise" -- the
    loop has to *release contacts* before it stops, and an exception mid-release is how a held
    contact gets left down -- so the reason is logged and formatted as a string in most places.
    But two callers must tell an occlusion from a geometry fault, because the right response
    differs: a covered window is recoverable (pause, resume when it clears), a moved or minimized
    one is not (every calibrated coordinate is now wrong).

    **This existed as a substring test and it was silently broken.** `loop.py` asked
    `"occlud" in reason`, and the message this class produces says "drawn over" -- so the pause
    branch was unreachable and every occlusion took the terminal-stop path instead. It was found
    by a live run, not by a test, because the tests asserted on the same prose the producer wrote.
    Branch on `kind`; never re-derive it from the words.
    """

    __slots__ = ("kind",)

    KINDS = ("gone", "minimized", "moved", "occluded")

    def __new__(cls, kind: str, message: str) -> "WindowFault":
        if kind not in cls.KINDS:
            raise ValueError(f"unknown window fault kind {kind!r}; expected one of {cls.KINDS}")
        self = super().__new__(cls, message)
        self.kind = kind
        return self

    @property
    def recoverable(self) -> bool:
        """Can the run continue if the operator fixes it? Only an occlusion can."""
        return self.kind == "occluded"


class WindowGuard:
    """Pins the emulator's geometry at startup and reports when it stops matching.

    Fail-closed hook for the §8 table. `check()` returns `None` when everything is as pinned, or a
    one-line reason when it is not; the loop turns a reason into "release all contacts and stop".
    It returns a string rather than raising because the loop has to *release* before it stops, and
    an exception in the middle of that is how a held contact gets left down.

    Occlusion is opt-in (`check_occlusion`) and separated from geometry for a reason: a moved
    window means every calibrated coordinate is wrong and the run is over, while a window on top
    means some frames are unreadable and the run could reasonably continue once it goes away. The
    loop is entitled to treat those differently; this class does not decide for it.
    """

    def __init__(self, window: WindowInfo, monitors: list[dict], *,
                 require_fullscreen: bool = True, check_occlusion: bool = True):
        self.require_fullscreen = require_fullscreen
        self.check_occlusion = check_occlusion
        self.state = WindowState(
            hwnd=window.hwnd,
            title=window.title,
            client=window.client,
            monitor_index=monitor_index_for(window.client, monitors),
            monitor=monitor_rect(window.hwnd),
        )
        if require_fullscreen and not self.state.fills_monitor:
            raise ValueError(
                f"the emulator's client area {self.state.client} does not fill its monitor "
                f"{self.state.monitor}. Every calibrated coordinate -- the homography, the HUD "
                f"mask, the tap anchors -- was measured against a fullscreen window, and a "
                f"windowed one is a different projection, not a smaller picture of the same one. "
                f"Put BlueStacks fullscreen, or pass require_fullscreen=False and expect to "
                f"recalibrate."
            )

    @property
    def monitor_index(self) -> int:
        """What to hand `DeployCapture(monitor=...)`."""
        return self.state.monitor_index

    def check(self) -> "WindowFault | None":
        """`None` if the window is still exactly where it was pinned, else why not.

        The return is a `WindowFault` -- a string, so every caller that logs or formats it is
        unchanged, carrying a `kind` for the callers that must BRANCH on it. See `WindowFault`
        for why that tag exists rather than the obvious substring test.
        """
        try:
            current = self._current()
        except LookupError as exc:
            return WindowFault("gone", f"the emulator window is gone: {exc}")
        if current.minimized:
            return WindowFault(
                "minimized",
                "the emulator window is minimized, so the capture is of the desktop behind it")
        if current.client != self.state.client:
            return WindowFault(
                "moved",
                f"the emulator window moved or resized: client area was "
                f"{self.state.client}, is now {current.client}. The content box was computed "
                f"once at startup and is now cropping the wrong pixels.")
        if self.check_occlusion:
            covering = find_occluders(current)
            if covering:
                return WindowFault(
                    "occluded",
                    f"{len(covering)} window(s) are drawn over the emulator "
                    f"(hwnd {covering}). The capture is showing them, not the game.")
        return None

    def _current(self) -> WindowInfo:
        """Re-read the pinned window by handle, so a *second* instance opening mid-run cannot
        silently redirect the guard to it -- `select_emulator` would raise on the ambiguity, which
        is right at startup and wrong here, where we already know which window we chose."""
        for w in enumerate_windows():
            if w.hwnd == self.state.hwnd:
                return w
        raise LookupError(f"no visible window with handle {self.state.hwnd} "
                          f"(was {self.state.title!r})")


def client_box_in_monitor(window: WindowInfo, monitor: dict) -> Rect:
    """The window's client area as an INCLUSIVE `(x0, x1, y0, y1)` box in the monitor's own pixels.

    This is `ScreenCapture`'s crop rectangle, taken from the window manager instead of inferred
    from pixels -- the shape `detect_content_box` returns, so it drops straight into the same
    slot. `mss` grabs a monitor, so the grab's pixel (0, 0) is the monitor's top-left and the
    conversion is one subtraction per axis.

    **Why supply it at all, when detection works.** Detection answers "which pixels are lit",
    which is only the same question as "where does the game render" when the app happens to be
    drawing to its edges. MEASURED on this setup: during gameplay `detect_content_box` returns
    essentially the whole client rect (2558x1439 of 2560x1440), but at the Nulls Brawl **lobby**
    it returns 2560x1418 -- the lobby's own artwork has a genuinely black 21 px band across the
    top. `ScreenCapture` computes the box once and caches it, so a loop started at the lobby
    would spend the whole match cropping 21 real rows off every gameplay frame and rescaling the
    rest by 1.5%, with every projected tile coordinate carrying the error and nothing raising.

    The window manager does not have that problem: the client rect is the same whatever is drawn
    in it. The guest here renders 1920x1080 into a 2560x1440 client area -- identical aspect, no
    letterbox -- so the client rect *is* the render rectangle, exactly.
    """
    x0 = window.client[0] - monitor["left"]
    y0 = window.client[1] - monitor["top"]
    x1 = window.client[2] - monitor["left"] - 1      # inclusive, as detect_content_box returns
    y1 = window.client[3] - monitor["top"] - 1
    if x0 < 0 or y0 < 0 or x1 >= monitor["width"] or y1 >= monitor["height"]:
        raise ValueError(
            f"the client area {window.client} is not fully inside the monitor "
            f"{(monitor['left'], monitor['top'], monitor['width'], monitor['height'])}. A crop "
            f"box outside the grab would silently index the wrong pixels."
        )
    return (x0, x1, y0, y1)
