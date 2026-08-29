"""One interface over every way frames reach the pipeline: the live screen, or a recorded clip.

**Why this exists as its own module.** Every stage downstream of Phase A only ever wants "the next
normalized frame". Which end it came from is a deployment question, not a pipeline question, and
the moment a stage takes a `ClipReader` specifically, replaying it against live capture means
editing that stage. `FrameSource` is the seam: stages take a `FrameSource`, and choosing live or
recorded is one call at the top of a script.

**The two implementations were already interchangeable by accident; this makes it on purpose.**
`capture.ScreenCapture` and `clips.ClipReader` both yield `capture.Frame` with the same fields and
the same normalization applied. What they did *not* share was lifecycle (`ScreenCapture` holds an
OS handle and must be a context manager; `ClipReader` did not implement the protocol at all) or a
common name for how much time reading has cost. Both gaps are closed here rather than left for a
caller to special-case.

**A source is finite or it is not, and callers must handle both.** `n_frames` is `None` for live
capture -- not zero, not a sentinel large number -- because a progress bar or a bounded loop over
live capture is a bug, and `None` makes it one at the point it is written.
"""
from pathlib import Path
from typing import Iterator, Protocol

from .capture import Frame
from .config import VisionConfig

# Accepted spellings for "the live screen" in `open_source`. A bare monitor index is deliberately
# NOT accepted: "1" is ambiguous with a filename, and silently grabbing a screen when the caller
# meant a clip is the kind of mistake that only shows up as garbage frames.
_SCREEN_PREFIX = "screen"


class FrameSource(Protocol):
    """What every stage may assume about where frames come from.

    Not `runtime_checkable`: `isinstance` against a Protocol only checks that attribute *names*
    exist on the class, which would pass for anything with an `__iter__` and tell you nothing.
    `tests/test_vision_sources.py` checks conformance against the real classes instead.
    """

    normalize: bool

    def __enter__(self) -> "FrameSource": ...

    def __exit__(self, *exc) -> bool: ...

    def __iter__(self) -> Iterator[Frame]: ...

    @property
    def n_frames(self) -> int | None:
        """Frames this source will yield, or `None` if it is unbounded (live capture)."""

    @property
    def read_seconds(self) -> float:
        """Cumulative seconds spent getting frames in -- decode for a clip, grab for a screen.

        Reported separately from any stage's own timing so that "the pipeline is slow" can be
        told apart from "reading 60 fps 2436x1126 h264 is slow", which are different problems
        with different fixes.
        """


def open_source(spec: str | Path | None = None, cfg: VisionConfig | None = None,
                **kwargs) -> FrameSource:
    """Open a live or recorded frame source from one spec.

        open_source()                       # the configured monitor
        open_source("screen")               # the same
        open_source("screen:2")             # monitor 2
        open_source("clips/match.mp4")      # a recording
        open_source(Path("match.mp4"))      # ditto

    Extra keyword arguments pass through to the underlying `ScreenCapture` / `ClipReader`, which
    is where the options that only make sense for one of them live (`trim_tail`, `monitor`).

    Both are context managers, so the calling shape is the same either way::

        with open_source(args.source) as src:
            for frame in src:
                ...
    """
    if spec is None:
        spec = _SCREEN_PREFIX
    text = str(spec)
    if text == _SCREEN_PREFIX or text.startswith(_SCREEN_PREFIX + ":"):
        from .capture import ScreenCapture      # lazy: needs the [vision] extra's `mss`
        _, _, monitor = text.partition(":")
        if monitor:
            kwargs.setdefault("monitor", int(monitor))
        return ScreenCapture(cfg, **kwargs)

    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(
            f"no such clip: {path}. For live capture pass 'screen', 'screen:<monitor>', or "
            f"nothing at all."
        )
    from .clips import ClipReader
    return ClipReader(path, cfg, **kwargs)


def is_live(source: FrameSource) -> bool:
    """True when the source never ends. Worth asking before anything that buffers or seeks."""
    return source.n_frames is None
