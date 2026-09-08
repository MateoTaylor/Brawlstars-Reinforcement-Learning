"""brawl_deployment's control layer and in-match gate.

Everything here runs WITHOUT BlueStacks attached -- the touch backend is exercised through
`NullBackend`, and only the tests that read real footage carry `@pytest.mark.vision`. That is
deliberate: a test suite that needs an emulator running is a test suite nobody runs.
"""
import json
import math

import numpy as np
import pytest

from brawl_deployment.control import (ATTACK_FIRE, ATTACK_NONE, ATTACK_SUPER, Buttons, Joystick,
                                      NullBackend, SLOT_MOVE, SLOT_TAP)
from brawl_deployment.control.adb import AdbTouchBackend, find_adb_serial
from brawl_deployment.capture import to_viewport
from brawl_deployment.match_state import (CALIBRATION_PATH, Calibration, MatchState,
                                          refine_radius)
from brawl_vision import gameplay as _G

CLIP = "tests/fixtures/vision/bluestacks-example-new.mp4"


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

def test_tap_targets_the_right_button():
    b = NullBackend()
    btn = Buttons(b, attack=(1676.5, 999.5), super_=(1462.5, 1000.5))
    assert btn.tap(ATTACK_FIRE)
    assert b.contacts[SLOT_TAP] == (1676.5, 999.5)
    btn.settle()
    assert btn.tap(ATTACK_SUPER)
    assert b.contacts[SLOT_TAP] == (1462.5, 1000.5)


def test_none_presses_nothing():
    b = NullBackend()
    btn = Buttons(b, attack=(1.0, 2.0), super_=(3.0, 4.0))
    assert not btn.tap(ATTACK_NONE)
    assert b.log == []


def test_repeat_tap_releases_first():
    """Two taps in a row must not swallow the second -- a down on an already-held slot is not a
    fresh press."""
    b = NullBackend()
    btn = Buttons(b, attack=(1.0, 2.0), super_=(3.0, 4.0))
    btn.tap(ATTACK_FIRE)
    btn.tap(ATTACK_FIRE)
    kinds = [e[0] for e in b.log]
    assert kinds == ["down", "up", "down"]


def test_tap_does_not_disturb_the_movement_contact():
    """The whole point of multitouch here: firing must not lift the held joystick contact."""
    b = NullBackend()
    j = Joystick(b, anchor=(300.0, 700.0), radius_px=140.0)
    btn = Buttons(b, attack=(1676.5, 999.5), super_=(1462.5, 1000.5))
    j.apply(7)
    held = b.contacts[SLOT_MOVE]
    btn.tap(ATTACK_FIRE)
    btn.settle()
    assert b.contacts[SLOT_MOVE] == held
    assert j.is_down


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


# ---------------------------------------------------------------- calibration

def test_calibration_loads_and_is_self_consistent():
    cal = Calibration.load()
    assert cal.screen == (1920, 1080)
    assert cal.gate_anchor in cal.buttons
    for name in ("attack", "super", "gadget"):
        x, y = cal.button(name)
        assert 0 < x < cal.screen[0] and 0 < y < cal.screen[1]


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


# ---------------------------------------------------------------- the in-match gate

def test_match_state_starts_out_of_match():
    """The safe initial state: a loop started on a menu must stay quiet."""
    ms = MatchState(Calibration.load())
    assert not ms.in_match


def test_match_state_needs_a_sustained_run_to_enter(monkeypatch):
    cal = Calibration.load()
    ms = MatchState(cal)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    monkeypatch.setattr(ms, "_anchor", (100.0, 100.0, 30.0))
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
    assert ms._anchor[2] == pytest.approx(cal.viewport_button(cal.gate_anchor)[2], abs=6.0)


@pytest.mark.vision
def test_gate_separates_gameplay_from_menus_on_real_footage():
    """The end-to-end claim, against the real BlueStacks recording: the gate is out of a match at
    the start, in one through the middle, and out again after "Defeated".

    Measured margins on this clip at the 2002x1126 viewport (gadget anchor, radius refined):
    menus 0.000-0.077, gameplay 0.709-0.823, median 0.802 -- a 10.5x margin.
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
    assert min(hi) > 0.7, f"gameplay floor {min(hi):.3f} too close to the threshold"
    assert max(lo) < 0.2, f"menu ceiling {max(lo):.3f} too close to the threshold"
    # The gap is what makes the 0.45 threshold safe to leave alone across HUD-neutral changes.
    assert min(hi) / max(lo) > 5.0


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
