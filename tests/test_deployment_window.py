"""`brawl_deployment.window` -- finding the emulator and noticing when it stops being findable.

No desktop, no emulator: every decision this module makes is a pure function of a window list, and
the Win32 calls that produce that list are monkeypatched. That is deliberate rather than merely
convenient -- the cases worth testing (two instances open, a window dragged off-screen, a
multi-monitor layout) are ones this machine cannot produce on demand.
"""
from dataclasses import replace

import pytest

from brawl_deployment import window as W

MON_1440 = {"left": 0, "top": 0, "width": 2560, "height": 1440, "is_primary": True}
MON_1080_RIGHT = {"left": 2560, "top": 0, "width": 1920, "height": 1080}
# mss.monitors[0] is the union of every display. Included in the fixtures precisely because
# skipping it is the bug `monitor_index_for` exists to prevent.
MONITORS = [{"left": 0, "top": 0, "width": 4480, "height": 1440}, MON_1440, MON_1080_RIGHT]


def _win(hwnd=1000, exe=r"C:\Program Files\BlueStacks_nxt\HD-Player.exe",
         client=(0, 0, 2560, 1440), title="Pie64", minimized=False, cls="Qt5152QWindowIcon",
         owner=0, ex=0):
    return W.WindowInfo(hwnd=hwnd, title=title, class_name=cls, exe=exe,
                        rect=client, client=client, minimized=minimized, owner=owner, ex_style=ex)


# The pair listed live on 2026-11-04, the first run on the real Brawl Stars app: handles, titles,
# classes, client rects, owner and extended style (WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW |
# WS_EX_LAYERED) exactly as Win32 reported them. Both belong to the same HD-Player.exe process.
APP_PLAYER = _win(hwnd=0xA806A2, title="BlueStacks App Player", cls="Qt672QWindowIcon")
KEYMAP_OVERLAY = _win(hwnd=0x708CE, title="BlueStacks Keymap Overlay",
                      cls="Qt672QWindowToolSaveBits", client=(0, 0, 2560, 1439),
                      owner=0xA806A2, ex=0x000800A0)


# ---------------------------------------------------------------------------------------------
# Picking the emulator out of a window list.
# ---------------------------------------------------------------------------------------------

def test_the_emulator_is_matched_by_executable_not_by_title_or_class():
    """Titles are instance names the user can rename and BlueStacks itself rewrites; the Qt class
    name changes with the build. Neither is a stable identity, so neither is consulted."""
    windows = [
        _win(hwnd=1, exe=r"C:\Windows\explorer.exe", title="BlueStacks 5", cls="Qt5152QWindowIcon"),
        _win(hwnd=2, exe=r"C:\Program Files\BlueStacks_nxt\HD-Player.exe",
             title="", cls="SomethingElse"),
    ]
    assert W.select_emulator(windows).hwnd == 2


def test_two_open_instances_raise_rather_than_picking_one():
    """The whole point. `find_adb_serial` returns the port of ONE instance; if two player windows
    are open, silently choosing either gives an agent that watches one game and plays another.
    A tie here is information, not a problem to solve."""
    windows = [_win(hwnd=1, title="Pie64"), _win(hwnd=2, title="Nougat64", client=(0, 0, 1280, 720))]
    with pytest.raises(LookupError, match="ambiguous"):
        W.select_emulator(windows)


def test_a_booting_instance_with_no_client_area_is_not_a_candidate():
    """A zero-area client is a window that exists but is not rendering yet. Reported as "still
    booting" rather than as "not found", because the operator's next move differs: wait, or start
    the thing."""
    with pytest.raises(LookupError, match="booting"):
        W.select_emulator([_win(client=(100, 100, 100, 100))])


def test_not_running_says_so_and_counts_unreadable_processes():
    """`OpenProcess` can be refused for a higher-integrity process, which looks identical to "not
    running" unless the count is reported. The two have completely different fixes."""
    windows = [_win(hwnd=1, exe=None), _win(hwnd=2, exe=r"C:\Windows\explorer.exe")]
    with pytest.raises(LookupError, match="1 whose process could not be read"):
        W.select_emulator(windows)


def test_no_unreadable_processes_leaves_that_clause_out():
    with pytest.raises(LookupError) as exc:
        W.select_emulator([_win(exe=r"C:\Windows\explorer.exe")])
    assert "could not be read" not in str(exc.value)


def test_the_keymap_overlay_is_part_of_its_instance_not_a_second_one():
    """The run of 2026-11-04 refused to start. 'BlueStacks Keymap Overlay' belongs to
    HD-Player.exe too, so the two windows read as two instances. The overlay is owned by the
    player window, and that ownership is what decides it, whichever order Win32 lists them in.
    It listed the overlay first, since an owned window sits above its owner."""
    for windows in ([KEYMAP_OVERLAY, APP_PLAYER], [APP_PLAYER, KEYMAP_OVERLAY]):
        assert W.select_emulator(windows) == APP_PLAYER


def test_two_instances_still_raise_when_each_has_its_overlay():
    """Ownership decides which of an instance's windows to drive. It must not hide the ambiguity
    between two instances, which is what the error exists for."""
    other = _win(hwnd=2, title="BlueStacks App Player 1")
    windows = [KEYMAP_OVERLAY, APP_PLAYER, replace(KEYMAP_OVERLAY, hwnd=3, owner=2), other]
    with pytest.raises(LookupError, match="2 HD-Player.exe windows are open") as exc:
        W.select_emulator(windows)
    assert "Keymap Overlay" not in str(exc.value)


def test_the_overlays_to_hide_are_the_click_through_windows_the_emulator_owns():
    """Each property alone describes something else (`overlays_of`). An owned window that takes
    clicks is a dialog, which `find_occluders` sees and the loop pauses for. A click-through
    window nobody owns belongs to another program."""
    dialog = _win(hwnd=5, title="Game controls", owner=APP_PLAYER.hwnd)
    foreign = _win(hwnd=6, exe=r"C:\Program Files\Other\overlay.exe", ex=W.CLICK_THROUGH)
    windows = [KEYMAP_OVERLAY, dialog, foreign, APP_PLAYER]
    assert W.overlays_of(APP_PLAYER, windows) == [KEYMAP_OVERLAY]
    assert KEYMAP_OVERLAY.click_through and not APP_PLAYER.click_through
    assert not _win(ex=W.WS_EX_LAYERED).click_through      # layered alone still takes clicks


# ---------------------------------------------------------------------------------------------
# Which monitor to capture.
# ---------------------------------------------------------------------------------------------

def test_the_monitor_index_is_the_display_the_window_centre_is_on():
    assert W.monitor_index_for((0, 0, 2560, 1440), MONITORS) == 1
    assert W.monitor_index_for((2560, 0, 4480, 1080), MONITORS) == 2


def test_index_zero_is_never_returned_even_though_it_contains_everything():
    """`monitors[0]` spans both displays here and would match any window. Returning it would hand
    `detect_content_box` a frame containing a second monitor, and the box would then describe
    something that is not the game."""
    for rect in [(0, 0, 2560, 1440), (2560, 0, 4480, 1080), (1000, 200, 3000, 900)]:
        assert W.monitor_index_for(rect, MONITORS) != 0


def test_a_window_dragged_off_screen_raises():
    with pytest.raises(LookupError, match="not on any monitor"):
        W.monitor_index_for((-4000, -4000, -3000, -3000), MONITORS)


def test_a_window_straddling_two_monitors_follows_its_centre():
    """Not a coin flip -- whichever display holds more than half the window is the one whose
    capture contains more than half the game. There is no better answer available, and
    `require_fullscreen` rejects this geometry anyway."""
    assert W.monitor_index_for((2000, 0, 3000, 1080), MONITORS) == 1   # centre 2500, still left
    assert W.monitor_index_for((2300, 0, 3300, 1080), MONITORS) == 2   # centre 2800, now right


# ---------------------------------------------------------------------------------------------
# Occlusion sampling.
# ---------------------------------------------------------------------------------------------

def test_sample_points_reach_into_the_corners_not_just_the_centre():
    """A toast or a chat overlay lands in a corner of the emulator, which a centre-only probe
    misses entirely -- and a corner is where they land *because* it is out of the way."""
    pts = W.sample_points((0, 0, 2560, 1440))
    assert pts[0] == (1280, 720)
    assert len(set(pts)) == 5
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    assert min(xs) == 640 and max(xs) == 1920
    assert min(ys) == 360 and max(ys) == 1080


def test_every_sample_point_is_strictly_inside_the_rect():
    for rect in [(0, 0, 2560, 1440), (100, 50, 300, 150), (2560, 0, 4480, 1080)]:
        for x, y in W.sample_points(rect):
            assert rect[0] <= x < rect[2] and rect[1] <= y < rect[3]


# ---------------------------------------------------------------------------------------------
# The guard.
# ---------------------------------------------------------------------------------------------

@pytest.fixture
def patched(monkeypatch):
    """Drives `WindowGuard` off a mutable list instead of the desktop.

    Hiding a window takes it out of the list and showing it puts it back, unless its handle is
    in `stuck`, which models a window that ignores `ShowWindowAsync` and stays listed.
    """
    state = {"windows": [_win()], "monitor": (0, 0, 2560, 1440), "occluders": [],
             "stuck": set(), "hid": [], "offscreen": {}}

    def hide(hwnds, timeout_s=0.5):
        state["hid"].append(list(hwnds))
        for w in [w for w in state["windows"] if w.hwnd in hwnds and w.hwnd not in state["stuck"]]:
            state["windows"].remove(w)
            state["offscreen"][w.hwnd] = w
        return [h for h in hwnds if h in state["stuck"]]

    def show(hwnds):
        for h in hwnds:
            if h in state["offscreen"]:
                state["windows"].append(state["offscreen"].pop(h))
        return list(hwnds)

    monkeypatch.setattr(W, "enumerate_windows", lambda: list(state["windows"]))
    monkeypatch.setattr(W, "monitor_rect", lambda hwnd: state["monitor"])
    monkeypatch.setattr(W, "find_occluders", lambda w, inset=0.25: list(state["occluders"]))
    monkeypatch.setattr(W, "hide_windows", hide)
    monkeypatch.setattr(W, "show_windows", show)
    return state


def test_a_fullscreen_window_pins_cleanly(patched):
    guard = W.WindowGuard(_win(), MONITORS)
    assert guard.monitor_index == 1
    assert guard.state.fills_monitor
    assert guard.check() is None


def test_a_windowed_emulator_is_refused_at_construction(patched):
    """Not a smaller picture of the same scene -- a different projection. The homography, the HUD
    mask and the tap anchors were all measured against a fullscreen client area."""
    small = _win(client=(200, 100, 2000, 1112))
    with pytest.raises(ValueError, match="does not fill its monitor"):
        W.WindowGuard(small, MONITORS)
    W.WindowGuard(small, MONITORS, require_fullscreen=False)   # opt out, and own the consequences


def test_a_moved_window_is_reported_with_both_rects(patched):
    """The message has to carry where it was AND where it is: "the window moved" without the
    numbers cannot be told from a monitor layout change, and the fixes differ."""
    guard = W.WindowGuard(_win(), MONITORS)
    patched["windows"] = [_win(client=(10, 10, 2570, 1450))]
    reason = guard.check()
    assert "moved or resized" in reason
    assert reason.kind == "moved" and not reason.recoverable
    assert "(0, 0, 2560, 1440)" in reason and "(10, 10, 2570, 1450)" in reason


def test_a_minimized_window_is_caught_before_the_geometry_test(patched):
    """A minimized window keeps its last client rect, so the geometry test passes while the
    capture is of the desktop behind it. Order matters, and the message has to be the true one."""
    guard = W.WindowGuard(_win(), MONITORS)
    patched["windows"] = [_win(minimized=True)]
    assert "minimized" in guard.check()
    assert guard.check().kind == "minimized"


def test_a_closed_window_is_a_reason_not_an_exception(patched):
    """`check()` never raises. The loop must release its held contacts before it stops, and an
    exception thrown into the middle of that is how a joystick contact gets left down on a game
    nobody is playing any more."""
    guard = W.WindowGuard(_win(), MONITORS)
    patched["windows"] = []
    assert "gone" in guard.check()
    assert guard.check().kind == "gone"


def test_a_second_instance_opening_mid_run_does_not_redirect_the_guard(patched):
    """`select_emulator` raises on ambiguity, which is right at startup and wrong here: the guard
    already knows which handle it chose, so it re-reads that one by hwnd and ignores the other."""
    guard = W.WindowGuard(_win(hwnd=1000), MONITORS)
    patched["windows"] = [_win(hwnd=1000), _win(hwnd=2000, title="Nougat64")]
    assert guard.check() is None


def test_occlusion_is_reported_and_can_be_switched_off(patched):
    """Separated from geometry because the two deserve different responses: a moved window ends
    the run, a window on top makes some frames unreadable and can clear on its own."""
    patched["occluders"] = [4242]
    fault = W.WindowGuard(_win(), MONITORS).check()
    assert "drawn over the emulator" in fault
    assert W.WindowGuard(_win(), MONITORS, check_occlusion=False).check() is None


def test_only_an_occlusion_is_recoverable_and_the_tag_is_what_says_so(patched):
    """The bug of 2026-09-09: `loop.py` decided pause-vs-stop with `"occlud" in reason`, and this
    module's occlusion message says "drawn over" -- so the pause branch never ran and alt-tabbing
    away ended the run for good. The tag exists so that decision cannot be made from prose again.

    Both halves are asserted here on purpose: that the fault is tagged, AND that the word the old
    code looked for genuinely is not in the message it looks at."""
    patched["occluders"] = [4242]
    fault = W.WindowGuard(_win(), MONITORS).check()
    assert fault.kind == "occluded" and fault.recoverable
    assert "occlud" not in fault.lower()

    patched["occluders"] = []
    patched["windows"] = [_win(minimized=True)]
    assert not W.WindowGuard(_win(), MONITORS).check().recoverable


def test_a_fault_cannot_be_built_with_a_kind_nobody_handles(patched):
    """`recoverable` is a closed question over `KINDS`. A typo'd kind that silently answered
    "not recoverable" would reintroduce the same class of bug quietly."""
    with pytest.raises(ValueError, match="unknown window fault kind"):
        W.WindowFault("obscured", "close, but not a kind anything branches on")


def test_geometry_is_checked_before_occlusion(patched):
    """Both wrong at once should report the one that ends the run, not the one that might pass."""
    guard = W.WindowGuard(_win(), MONITORS)
    patched["windows"] = [_win(client=(10, 10, 2570, 1450))]
    patched["occluders"] = [4242]
    assert guard.check().kind == "moved"


def test_the_guard_hides_the_keymap_overlay_for_the_run_and_shows_it_again(patched):
    """The overlay puts BlueStacks' key hints on the HUD buttons, and the grab includes them, so
    it is hidden for the run. It is shown again after, since the operator may still want it."""
    patched["windows"] = [KEYMAP_OVERLAY, APP_PLAYER]
    guard = W.WindowGuard(APP_PLAYER, MONITORS)
    assert guard.suppress_overlays() == [KEYMAP_OVERLAY]
    assert patched["windows"] == [APP_PLAYER]
    assert guard.check() is None
    assert guard.restore_overlays() == [KEYMAP_OVERLAY]
    assert KEYMAP_OVERLAY in patched["windows"]
    assert guard.restore_overlays() == []                  # once, however often it is called


def test_an_overlay_shown_again_mid_run_is_hidden_at_the_next_check(patched):
    """Whether BlueStacks shows it again by itself is untested (it stayed hidden while
    unfocused), so `check()` does not assume it won't."""
    patched["windows"] = [KEYMAP_OVERLAY, APP_PLAYER]
    guard = W.WindowGuard(APP_PLAYER, MONITORS)
    guard.suppress_overlays()
    patched["windows"] = [KEYMAP_OVERLAY, APP_PLAYER]      # BlueStacks shows it again
    assert guard.check() is None
    assert patched["windows"] == [APP_PLAYER]
    assert patched["hid"] == [[KEYMAP_OVERLAY.hwnd]] * 2


def test_an_overlay_that_will_not_hide_pauses_the_run(patched):
    """It is still drawn over the game, so it gets the same response as any window on top: a
    recoverable fault. Here the guard is the only thing that can report it, because the
    occlusion probe sees straight through it."""
    patched["windows"] = [KEYMAP_OVERLAY, APP_PLAYER]
    patched["stuck"] = {KEYMAP_OVERLAY.hwnd}
    guard = W.WindowGuard(APP_PLAYER, MONITORS)
    assert guard.suppress_overlays() == []
    fault = guard.check()
    assert fault.kind == "occluded" and fault.recoverable
    assert "BlueStacks Keymap Overlay" in fault
    # It was tried, so it is shown again at the end too: a hide that lands late gets undone.
    assert guard.restore_overlays() == [KEYMAP_OVERLAY]


def test_hiding_overlays_can_be_switched_off(patched):
    """`window.hide_overlays: false`. Nothing else catches the overlay then, since it is
    click-through: that is the cost of turning the switch off."""
    patched["windows"] = [KEYMAP_OVERLAY, APP_PLAYER]
    guard = W.WindowGuard(APP_PLAYER, MONITORS, hide_overlays=False)
    assert guard.suppress_overlays() == []
    assert guard.check() is None
    assert patched["hid"] == [] and guard.restore_overlays() == []


# ---------------------------------------------------------------------------------------------
# The crop box, taken from the window instead of inferred from pixels.
# ---------------------------------------------------------------------------------------------

def test_the_client_box_is_the_client_rect_in_the_monitors_own_pixels():
    """`mss` grabs a monitor, so the grab's (0, 0) is the monitor's top-left. Inclusive bounds,
    matching what `detect_content_box` returns, so this drops into the same slot."""
    assert W.client_box_in_monitor(_win(client=(0, 0, 2560, 1440)), MON_1440) == (0, 2559, 0, 1439)


def test_the_client_box_subtracts_a_secondary_monitors_origin():
    """The case this machine cannot produce. A window on the right-hand display has screen
    coordinates starting at 2560, but its grab starts at 0 -- forget the subtraction and every
    crop is off by a full screen width, which is not subtle but is also not testable here without
    fabricating the layout."""
    w = _win(client=(2560, 0, 4480, 1080))
    assert W.client_box_in_monitor(w, MON_1080_RIGHT) == (0, 1919, 0, 1079)


@pytest.mark.parametrize("client", [(-10, 0, 2550, 1440), (0, 0, 2570, 1440), (0, 100, 2560, 1540)])
def test_a_client_area_outside_the_monitor_raises(client):
    """A crop box outside the grab does not raise in numpy -- it silently clips, or wraps, and
    returns the wrong pixels at a plausible size."""
    with pytest.raises(ValueError, match="not fully inside the monitor"):
        W.client_box_in_monitor(_win(client=client), MON_1440)


def test_the_lobby_is_why_the_box_comes_from_the_window(patched):
    """MEASURED 2026-09-08 and the reason this function exists. `detect_content_box` returned
    2560x1418 at the Nulls Brawl lobby and ~2560x1440 during gameplay, because the lobby's own
    artwork has a black 21 px band across the top. The box is cached, so a loop started at the
    lobby crops 21 real rows off every gameplay frame for the whole match. The window manager has
    no such dependence on what is being drawn -- pinned here so the two numbers stay in the repo.
    """
    em = _win(client=(0, 0, 2560, 1440))
    box = W.client_box_in_monitor(em, MON_1440)
    assert box[3] - box[2] + 1 == 1440          # the client rect, whatever the app draws
    lobby_inferred_height = 1418
    assert 1440 / lobby_inferred_height == pytest.approx(1.0155, abs=1e-4)


def test_the_title_is_kept_for_messages_but_never_compared(patched):
    """BlueStacks rewrites its own title -- FPS counters, instance renames -- while the geometry
    stays put. Comparing it would stop the agent for a cosmetic change."""
    guard = W.WindowGuard(_win(title="Pie64"), MONITORS)
    patched["windows"] = [_win(title="Pie64 - 60 FPS")]
    assert guard.check() is None
    assert guard.state.title == "Pie64"
