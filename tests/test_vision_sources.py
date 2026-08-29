"""The FrameSource seam: one interface over live capture and recorded clips. See
Terrain_Perception_Build_Plan.md Phase A.

The point of these tests is *interchangeability*. It is easy to write a source API that both
classes technically satisfy and that still forces every caller to branch on which one it got, so
what is checked here is the properties a caller actually relies on -- same lifecycle, same frame
type, same timing name, and a finite/unbounded answer that is impossible to ignore.
"""
import inspect
from pathlib import Path

import numpy as np
import pytest

from brawl_vision.capture import Frame, ScreenCapture
from brawl_vision.clips import ClipReader
from brawl_vision.sources import FrameSource, is_live, open_source

CLIPS = Path(__file__).resolve().parent / "fixtures" / "vision"
CLIP = CLIPS / "standstill.mp4"

_PROTOCOL_MEMBERS = ("__enter__", "__exit__", "__iter__", "n_frames", "read_seconds")


@pytest.mark.parametrize("cls", [ScreenCapture, ClipReader])
def test_both_implementations_satisfy_the_protocol(cls):
    """`FrameSource` is not runtime_checkable on purpose -- `isinstance` against a Protocol only
    checks that names exist, which anything with an `__iter__` would pass. This checks the real
    classes carry every member instead, which is the assertion that would actually have caught
    `ClipReader` missing `__enter__`."""
    for name in _PROTOCOL_MEMBERS:
        assert hasattr(cls, name), f"{cls.__name__} is missing FrameSource.{name}"


def test_protocol_lists_exactly_what_the_tests_check():
    """Keeps the two lists from drifting: a member added to the Protocol without being added here
    would be unenforced, which is the same as not being in the Protocol at all."""
    declared = {n for n in vars(FrameSource) if not n.startswith("_")}
    declared |= {n for n in ("__enter__", "__exit__", "__iter__") if n in vars(FrameSource)}
    assert declared == set(_PROTOCOL_MEMBERS)


# ---------------------------------------------------------------------------
# open_source dispatch
# ---------------------------------------------------------------------------

def test_a_path_opens_a_clip():
    assert isinstance(open_source(CLIP), ClipReader)
    assert isinstance(open_source(str(CLIP)), ClipReader)


@pytest.mark.parametrize("spec", [None, "screen"])
def test_screen_specs_open_live_capture(spec):
    assert isinstance(open_source(spec), ScreenCapture)


def test_a_monitor_can_be_named_in_the_spec():
    assert open_source("screen:2").monitor == 2


def test_a_bare_number_is_not_treated_as_a_monitor():
    """"1" is ambiguous with a filename, and silently grabbing a screen when the caller meant a
    clip surfaces only as garbage frames much later. It must fail as a missing path instead."""
    with pytest.raises(FileNotFoundError):
        open_source("1")


def test_a_missing_clip_says_how_to_ask_for_the_screen():
    with pytest.raises(FileNotFoundError, match="screen"):
        open_source(CLIPS / "does_not_exist.mp4")


def test_keyword_arguments_reach_the_underlying_source():
    assert open_source(CLIP, trim_tail=False).trim_tail is False
    assert open_source("screen", monitor=3).monitor == 3


# ---------------------------------------------------------------------------
# the properties callers rely on
# ---------------------------------------------------------------------------

def test_a_clip_is_finite_and_the_screen_is_not():
    """`None` rather than 0 or a large number: a bounded loop or a progress bar over live capture
    is a bug, and `None` makes it fail where it is written instead of running forever.

    The clip's count is checked as *a finite positive int*, not as a literal. It used to assert
    409 and broke when Phase A's tail detector was fixed — a change that had nothing to do with
    this file's subject, which is that one source is bounded and the other is not. The exact
    frame count belongs to `test_vision_clips.py`, which owns that measurement.
    """
    n = open_source(CLIP).n_frames
    assert isinstance(n, int) and n > 0
    assert open_source("screen").n_frames is None
    assert is_live(open_source("screen")) and not is_live(open_source(CLIP))


def test_read_seconds_exists_on_both_and_starts_at_zero():
    assert open_source(CLIP).read_seconds == 0.0
    assert open_source("screen").read_seconds == 0.0


def test_read_seconds_aliases_the_concrete_name():
    """The specific names stay because they say WHICH cost it is; the shared name exists so a
    stage can report read time without knowing what it is reading."""
    clip = open_source(CLIP)
    clip.decode_seconds = 1.25
    assert clip.read_seconds == 1.25
    screen = open_source("screen")
    screen.grab_seconds = 2.5
    assert screen.read_seconds == 2.5


def test_a_clip_is_usable_as_a_context_manager_without_holding_anything():
    """`ClipReader` opens its VideoCapture per `__iter__`, so entering is a no-op -- but callers
    must be able to write one `with` block that works for either source."""
    with open_source(CLIP) as source:
        assert isinstance(source.n_frames, int) and source.n_frames > 0
    with open_source(CLIP) as source:      # re-enterable, unlike a handle-holding source
        assert sum(1 for _ in zip(source, range(3))) == 3


def test_exit_does_not_swallow_exceptions():
    """A `__exit__` returning anything truthy silences errors raised inside the `with` -- the
    kind of bug that turns a crash in a capture loop into a silent early return."""
    for spec in (CLIP, "screen"):
        assert open_source(spec).__exit__(None, None, None) is False


@pytest.mark.vision
def test_a_clip_yields_normalized_frames_of_the_common_type():
    with open_source(CLIP) as source:
        frame = next(iter(source))
    assert isinstance(frame, Frame)
    assert frame.image.shape == (1126, 2002, 3) and frame.image.dtype == np.uint8
    assert frame.index == 0


@pytest.mark.vision
@pytest.mark.parametrize("name", ["showdown_alternate_map", "showdown_alternate_map2",
                                  "showdown_has_gadget", "training_gadget"])
def test_every_clip_normalizes_to_the_calibrated_viewport(name):
    """The property that lets ONE homography serve every recording. `showdown_alternate_map2`'s
    detected content box is 2 px wider than its siblings' and still lands here, which is
    `normalize_viewport`'s width slack doing real work -- box detection is per-frame thresholding
    on a compressed recording, so demanding an exact match would reject a good clip."""
    with open_source(CLIPS / f"{name}.mp4") as source:
        frame = next(iter(source))
    assert frame.image.shape == (1126, 2002, 3)


@pytest.mark.vision
def test_read_seconds_accumulates_while_reading():
    with open_source(CLIP) as source:
        for _ in zip(source, range(5)):
            pass
        assert source.read_seconds > 0.0


def test_screen_capture_still_requires_being_entered():
    """Interchangeability must not have quietly relaxed this: `mss` holds an OS handle, and a
    capture loop leaking one per restart is exactly what Phase A's acceptance guards."""
    with pytest.raises(RuntimeError, match="context manager"):
        open_source("screen").grab()


def test_open_source_is_documented_with_its_spec_forms():
    """This docstring is the only place the accepted spellings are written down, and the error
    message for a missing clip points at it."""
    doc = inspect.getdoc(open_source)
    for form in ("screen", "screen:2", ".mp4"):
        assert form in doc
