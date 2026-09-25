"""`scripts/deploy_calibrate.py`'s job verdicts, driven off fake readings instead of an emulator.

The jobs themselves need a live BlueStacks. What they DECIDE does not: a job is a reading loop
plus a verdict, and the verdict is the part that has been wrong. On 2026-09-22 `job_super` printed
FAIL for a Super the operator watched fire correctly at the commanded bearing, because it took one
reading at a flat 0.8 s against a deployment observable lag of roughly a second. Nothing caught it
because nothing tested it. These do.

`SuperReading` is the real dataclass, not a stand-in, so a change to its contract fails here
rather than passing against a double that agrees with itself.
"""
import time as real_time

import numpy as np
import pytest

from brawl_vision.object_detection.hp_detection.hero_bars import SuperReading
from scripts import deploy_calibrate as dc


def _reading(ready: bool) -> SuperReading:
    """A charged or spent super track, with the plausible fields filled in."""
    return SuperReading(charge=1.0 if ready else 0.0, ready=ready,
                        state="ready" if ready else "empty", track_px=96, row=712)


class _Clock:
    """A fake `time` for the module under test: `sleep` advances it, `perf_counter` reads it.

    Patched in wholesale so the poll runs at its real sample count in no wall-clock time, and so
    the reported latency is an exact number this file can assert rather than a flaky window.

    Only the two timing calls are faked; everything else falls through to the real module. `_log`
    calls `time.strftime` on every line, and a double that answered only the calls I had in mind
    would make every job in this file die on a log statement.
    """

    def __init__(self):
        self.t = 0.0

    def sleep(self, seconds):
        self.t += seconds

    def perf_counter(self):
        return self.t

    def __getattr__(self, name):
        return getattr(real_time, name)


class _Rig:
    """Serves a scripted sequence of super readings, then HOLDS the last one forever.

    Holding matters: "the bar never drains" has to be expressible, and a list that runs out would
    raise instead of exercising the timeout path.
    """

    def __init__(self, readings):
        self._readings, self.calls = list(readings), 0
        self.controls = type("_C", (), {"buttons": type("_B", (), {"super_": (1559.9, 901.1)})()})()

    def super_charge(self):
        reading = self._readings[min(self.calls, len(self._readings) - 1)]
        self.calls += 1
        return reading


@pytest.fixture
def presses(monkeypatch):
    """Record presses instead of emitting them, and put the module on a fake clock."""
    fired = []
    monkeypatch.setattr(dc.cal_lib, "press_and_lift",
                        lambda buttons, action, bearing: fired.append((action, bearing)))
    monkeypatch.setattr(dc, "time", _Clock())
    return fired


def test_job_super_passes_on_a_drain_that_lands_after_the_old_fixed_window(presses):
    """The 2026-09-22 regression, stated as a number. The old code slept 0.8 s, took ONE reading
    and scored it; this drain arrives at 1.2 s, which that code reports as FAIL and this one
    reports as PASS at the latency it actually saw. The 0.8 s assert is the point of the test --
    a future "simplification" back to a fixed sleep has to beat the measured lag to pass."""
    rig = _Rig([_reading(True)] * 12 + [_reading(False)])
    out = dc.job_super(rig, None)

    assert out["ok"] is True
    assert out["after"].state == "empty"
    latency = out["samples"][-1][0]
    assert latency == pytest.approx(1.2)
    assert latency > 0.8, "the drain lands after the window the old single-sample check used"
    assert presses == [(dc.ATTACK_SUPER, dc.AIM_BEARING)]


def test_job_super_skips_frames_where_the_hero_cannot_be_found(presses):
    """Mortis's super is a DASH, so the hero detector is least reliable exactly when this job
    looks, and `super_charge` returns None for both "no player detection" and "track unreadable".
    A None decides nothing: it is counted and the poll keeps going. The old code scored whatever
    single frame it happened to land on, so one lost detection read as a failed super."""
    rig = _Rig([_reading(True), None, None, _reading(False)])
    out = dc.job_super(rig, None)

    assert out["ok"] is True
    assert out["unreadable"] == 2
    assert [s[1] for s in out["samples"]] == ["empty"], "unreadable frames are not samples"


def test_job_super_fails_on_a_track_that_never_drains_and_prints_the_trajectory(presses):
    """A genuine miss still has to fail, and has to leave enough behind to read itself. The
    trajectory is returned, not just the verdict, so the next such failure says whether the bar
    held steady (the tap missed) or crept (the window was short)."""
    rig = _Rig([_reading(True)])
    out = dc.job_super(rig, None)

    assert out["ok"] is False and out["after"] is None
    assert len(out["samples"]) == pytest.approx(dc.SUPER_DRAIN_WINDOW_S / dc.SUPER_SAMPLE_S, abs=1)
    assert out["samples"][-1][0] >= dc.SUPER_DRAIN_WINDOW_S - dc.SUPER_SAMPLE_S


def test_job_super_reports_an_uncharged_super_as_skipped_and_presses_nothing(presses):
    """Best-effort by design -- but this is also the hole the 2026-09-09 button-label rotation
    hid in for two weeks (design 5.1). A wrong super point and an uncharged super produced
    byte-identical output, so no run ever discriminated them. Pinning "skipped, and no press"
    keeps that a deliberate property rather than an accident, and keeps `ok` absent: a skip is
    never a pass."""
    out = dc.job_super(_Rig([_reading(False)]), None)

    assert out == {"skipped": "empty"}
    assert "ok" not in out
    assert presses == [], "nothing is pressed when there is nothing to spend"


def test_the_roi_digest_moves_on_a_pixel_the_disc_colour_would_miss():
    """The digest exists to break a specific ambiguity (2026-09-22): one distinct SCORE across a
    whole trace reads either as a pixel-for-pixel static disc or as a stale ROI, and a score cannot
    separate them. So the digest has to be stable on identical pixels and move on a single changed
    one -- and specifically on a pixel between `0.8 * r` and `r`, the band `_disc_colour` averages
    away and the band `ring_score_at` actually reads. Bounded, too: a pixel outside the ROI box
    must not move it, or the digest stops being about this anchor."""
    image = np.full((200, 200, 3), 40, dtype=np.uint8)
    at = dc._roi_digest(image, 100.0, 100.0, 20.0)
    assert dc._roi_digest(image.copy(), 100.0, 100.0, 20.0) == at, "same pixels, same digest"

    on_the_ring = image.copy()
    on_the_ring[100, 119] = 41  # r=19 from centre: outside 0.8*r=16, inside the ring
    assert dc._roi_digest(on_the_ring, 100.0, 100.0, 20.0) != at
    assert dc._disc_colour(on_the_ring, 100.0, 100.0, 20.0) == dc._disc_colour(image, 100, 100, 20)

    far_away = image.copy()
    far_away[100, 160] = 41  # well outside the padded box
    assert dc._roi_digest(far_away, 100.0, 100.0, 20.0) == at, "the digest is anchor-local"
