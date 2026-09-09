"""The ammo-timing measurement. See scripts/measure_reload.py.

**Everything here is a synthetic pip series**, which is the whole reason the script is split the
way it is: reading the bar needs a detector and a GPU, and turning a sequence of pip counts into
`reload_seconds` does not. Every rule below is one that a wrong answer would slip through
silently, because there is no ground truth to compare a reload measurement against -- if this
script says 2.5 s and the truth is 5.0 s because one transition was missed, nothing downstream
will ever notice.

The series are written as `(t, ammo)` at a nominal 30 Hz, matching the clip sample rate the script
defaults to.
"""
import pytest

from scripts.measure_reload import (
    HOLD_SECONDS, describe, histogram, intervals, series_from_telemetry, transitions,
)


def hold(level, t0, seconds, dt=1 / 30):
    """`seconds` of a steady reading, starting at `t0`."""
    n = int(round(seconds / dt))
    return [(t0 + i * dt, level) for i in range(n)]


def series(*spans, dt=1 / 30):
    """`(level, seconds)` pairs laid end to end. `None` is an unreadable stretch."""
    out, t = [], 0.0
    for level, seconds in spans:
        out += hold(level, t, seconds, dt)
        t += seconds
    return out


def steps_of(*spans, hold_s=HOLD_SECONDS):
    return transitions(series(*spans), hold_s)


# ---------------------------------------------------------------------------
# transitions: what counts as a level change
# ---------------------------------------------------------------------------

def test_a_steady_reading_is_one_transition_not_one_per_frame():
    assert steps_of((3, 2.0)) == [(0.0, 3)]


def test_a_single_frame_blip_is_not_a_transition():
    """The reader misreads a few percent of frames. A blip would otherwise manufacture a spend and
    a gain 33 ms apart, and 33 ms is a reload time this script would then report."""
    s = series((3, 1.0), (1, 1 / 30), (3, 1.0))
    assert [level for _, level in transitions(s, HOLD_SECONDS)] == [3]


def test_a_change_is_timestamped_at_its_FIRST_frame_not_the_end_of_the_hold():
    """The widget snaps rather than ramps, so the first frame showing the new count IS the
    transition. Timing it at the end of the hold would add the hold to both endpoints -- which
    cancels inside one run and quietly does not across two runs measured at different holds."""
    steps = steps_of((3, 1.0), (2, 1.0))
    assert steps[1][1] == 2
    assert steps[1][0] == pytest.approx(1.0, abs=1 / 30)


def test_an_unreadable_stretch_is_skipped_rather_than_ending_the_level():
    """A gap is missing information, not a reading of zero. If it ended the level, every gap would
    be followed by a fabricated transition back to the same number."""
    s = series((3, 1.0), (None, 1.0), (3, 1.0))
    assert [level for _, level in transitions(s, HOLD_SECONDS)] == [3]


# ---------------------------------------------------------------------------
# intervals: which pairs of transitions are a measurement
# ---------------------------------------------------------------------------

def readable_ts(*spans):
    return [t for t, v in series(*spans) if v is not None]


def measure(*spans, max_ammo=3, max_gap=0.4):
    return intervals(transitions(series(*spans), HOLD_SECONDS), readable_ts(*spans),
                     max_ammo, max_gap)


def test_two_gains_with_no_shot_between_them_is_one_reload():
    """Family A. Ammo climbing 1 -> 2 -> 3 untouched times exactly one reload cycle, whatever the
    sensor's lag is, because a constant lag subtracts out of a difference."""
    got = measure((1, 1.0), (2, 2.5), (3, 1.0))
    assert got["reload"] == pytest.approx([2.5], abs=0.05)


def test_a_shot_between_two_gains_disqualifies_the_interval():
    """`brawl_sim/core/hero.tick_timers` pauses the reload for `attack_cooldown` when you fire, so
    a gain -> shot -> gain span is `reload + cooldown` and belongs in neither family. Counting it
    as a reload is how the pause would hide inside the number it is supposed to be measured from.
    """
    got = measure((1, 1.0), (2, 1.0), (1, 1.0), (2, 1.0))
    assert got["reload"] == []


def test_only_a_shot_from_a_FULL_magazine_starts_the_clock_at_a_known_instant():
    """Family B. Below full the reload timer is already running when the shot lands, so the
    interval measures whatever was left of it -- a number with no name."""
    from_full = measure((3, 1.0), (2, 2.6), (3, 1.0))
    assert from_full["shot_to_gain"] == pytest.approx([2.6], abs=0.05)

    from_partial = measure((3, 1.0), (2, 0.5), (1, 2.6), (2, 1.0))
    assert from_partial["shot_to_gain"] == []


def test_a_two_pip_jump_drops_the_intervals_on_BOTH_sides():
    """A jump means a level we never saw. The span ending at it is two reloads reported as one,
    and the span starting from it is anchored on a time that is not a transition."""
    got = measure((1, 1.0), (3, 2.5), (2, 1.0), (3, 2.5))
    assert got["dropped_multi"] == 1
    assert got["reload"] == []


def test_an_interval_containing_an_unreadable_stretch_is_dropped_and_counted():
    """This is the failure that matters most and it is invisible in the answer: a missed
    transition does not corrupt an interval a little, it DOUBLES it. So the guard is on the gap,
    never on how far the result sits from the median -- that would be fitting the filter to the
    conclusion."""
    got = measure((1, 1.0), (2, 1.0), (None, 1.0), (3, 1.0))
    assert got["reload"] == []
    assert got["dropped_gap"] == 1


def test_a_gap_shorter_than_the_tolerance_keeps_the_interval():
    got = measure((1, 1.0), (2, 1.0), (None, 0.2), (3, 1.0), max_gap=0.4)
    assert len(got["reload"]) == 1


def test_shot_to_shot_is_collected_as_a_floor_and_not_as_either_family():
    got = measure((3, 1.0), (2, 0.8), (1, 1.0))
    assert got["shot_to_shot"] == pytest.approx([0.8], abs=0.05)
    assert got["reload"] == []
    assert got["shot_to_gain"] == []


def test_the_two_families_are_collected_from_the_same_run_without_contaminating_each_other():
    """B - A is `attack_cooldown`, so the one thing that must never happen is a sample landing in
    both. Full magazine, one shot, a reload, two more shots, then two clean reloads: the first
    gain is a B and only the last gain is an A."""
    got = measure((3, 1.0), (2, 2.6), (3, 0.4), (2, 0.4), (1, 0.4), (2, 2.5), (3, 2.5))
    assert got["shot_to_gain"] == pytest.approx([2.6], abs=0.06)
    assert got["reload"] == pytest.approx([2.5], abs=0.06)


def test_two_shots_closer_together_than_the_debounce_collapse_into_a_DROPPED_jump():
    """The debounce is 0.25 s and Mortis can fire faster than that, so this case is real. It has
    to fail in the safe direction, and it does: the pair collapses into a 3 -> 1 jump, which the
    multi-pip rule drops. A measurement is lost; a wrong one is not produced."""
    got = measure((3, 1.0), (2, 0.1), (1, 1.0), (2, 2.5), (3, 2.5))
    assert got["dropped_multi"] == 1
    assert got["reload"] == pytest.approx([2.5], abs=0.06)


# ---------------------------------------------------------------------------
# telemetry, summary, histogram
# ---------------------------------------------------------------------------

def test_the_telemetry_sentinel_is_a_gap_and_never_an_empty_magazine(tmp_path):
    """`-1.0` in `ammo_cv` means no decision this tick, no hero box, or the reader returned None.
    Reading it as 0 would invent a full magazine dump followed by a three-pip reload, out of a
    tick where nothing happened at all."""
    csv_path = tmp_path / "t.csv"
    csv_path.write_text("t,ammo_cv,ammo_shadow\n0.0,-1.0,-1.0\n0.25,2.97,3.0\n0.5,-1.0,-1.0\n",
                        encoding="utf-8")
    assert series_from_telemetry(csv_path) == [(0.0, None), (0.25, 3), (0.5, None)]


def test_telemetry_without_the_column_fails_loudly_rather_than_measuring_nothing(tmp_path):
    csv_path = tmp_path / "old.csv"
    csv_path.write_text("t,phase\n0.0,waiting\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="ammo_cv"):
        series_from_telemetry(csv_path)


def test_describe_reports_the_count_alongside_the_median():
    """Two samples and two hundred get the same median and mean very different things. `n` is
    reported everywhere the median is, so a thin measurement cannot be quoted as a thick one."""
    assert describe([]) is None
    got = describe([2.4, 2.5, 2.6])
    assert got["n"] == 3
    assert got["median"] == pytest.approx(2.5)


def test_the_histogram_shows_the_gap_between_two_modes():
    """A second mode at twice the first is a missed transition, and no summary statistic shows it.
    That is why the empty buckets in between are drawn rather than skipped."""
    lines = histogram([2.5, 2.5, 5.0]).splitlines()
    assert len(lines) > 3
    assert lines[0].strip().endswith("##")
    assert lines[-1].strip().endswith("#")
    assert any(line.strip().endswith(".") for line in lines[1:-1])


def test_a_long_tail_is_summed_into_an_overflow_row_rather_than_dropped():
    """A discarded outlier is a hidden one, and here an outlier is the evidence of a defect."""
    out = histogram([2.5, 40.0], rows=6)
    assert out.splitlines()[-1].strip().startswith(">=")
    assert out.splitlines()[-1].strip().endswith("#")
