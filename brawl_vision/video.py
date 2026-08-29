"""Writing numpy frames to a video file. See Terrain_Perception_Build_Plan.md Phase L.

**Why this exists next to `TerrainOverlay.save` rather than inside it.** That method writes a
*matplotlib figure* per frame, which Phase E measured at ~0.49 s/frame -- 39 minutes for the
4802-frame fixture. Phase L renders a numpy array instead and needs a sink that takes one. The two
cannot share a writer, but they must share how the encoder is *found* and what the user is told
when it is missing, which is the part that was worth factoring rather than copying.
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np

from brawl_sim.render.viewer import locate_ffmpeg

_NO_FFMPEG = (
    "no ffmpeg binary found for .mp4 export. Either install ffmpeg and put it on PATH, or run "
    "`pip install imageio-ffmpeg` (bundles one, no system install needed) -- or save as .gif "
    "instead, which needs neither."
)


def require_ffmpeg() -> str:
    """`locate_ffmpeg()`, or raise with the instruction for fixing it.

    `locate_ffmpeg` returns None on failure because some callers have a .gif fallback. Every caller
    that does not needs the same sentence, and three copies of it drift.
    """
    found = locate_ffmpeg()
    if found is None:
        raise RuntimeError(_NO_FFMPEG)
    return found


class VideoSink:
    """Stream RGB uint8 frames of a fixed size to `path`. Use as a context manager.

    `.mp4` pipes raw frames straight into ffmpeg, so memory is one frame regardless of clip length.
    `.gif` cannot do that -- the format needs the whole animation before it writes -- so it buffers,
    and is refused past `max_gif_frames` rather than quietly exhausting memory on a 4802-frame clip.

    **Dimensions are forced even.** `yuv420p` subsamples chroma by two, so an odd width or height is
    rejected by the encoder with a message that says nothing about the caller's tile scale. Padding
    by one pixel here is invisible; failing at frame zero after a minute of pass-one is not.
    """

    def __init__(self, path, size: tuple[int, int], fps: float = 20.0,
                 crf: int = 18, max_gif_frames: int = 600):
        self.path = Path(path)
        w, h = size
        self.size = (w + w % 2, h + h % 2)
        self.fps = float(fps)
        self.crf = int(crf)
        self.max_gif_frames = int(max_gif_frames)
        self.n_frames = 0
        self._proc = None
        self._gif: list[np.ndarray] = []
        suffix = self.path.suffix.lower()
        if suffix not in (".mp4", ".gif"):
            raise ValueError(f"unsupported video extension {suffix!r} -- use .mp4 or .gif")
        self._suffix = suffix

    def __enter__(self) -> "VideoSink":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._suffix == ".mp4":
            w, h = self.size
            self._proc = subprocess.Popen(
                [require_ffmpeg(), "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                 "-r", f"{self.fps:g}", "-i", "-",
                 "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                 "-crf", str(self.crf), str(self.path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        return self

    def write(self, frame: np.ndarray) -> None:
        """`frame` is (h, w, 3) uint8 RGB. Padded to the sink's even size if it is one pixel short
        -- which it will be exactly when the caller's own size was odd."""
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected (h, w, 3) uint8 RGB, got {frame.shape} {frame.dtype}")
        w, h = self.size
        if frame.shape[:2] != (h, w):
            padded = np.zeros((h, w, 3), np.uint8)
            padded[:frame.shape[0], :frame.shape[1]] = frame[:h, :w]
            frame = padded
        if self._proc is not None:
            self._proc.stdin.write(frame.tobytes())
        else:
            if len(self._gif) >= self.max_gif_frames:
                raise RuntimeError(
                    f"{self.path.name} would exceed {self.max_gif_frames} frames. GIF has to hold "
                    f"the whole animation in memory before it writes -- use .mp4 for a clip this "
                    f"long, or raise max_gif_frames if you know what it will cost."
                )
            self._gif.append(frame.copy())
        self.n_frames += 1

    def __exit__(self, *exc) -> bool:
        if self._proc is not None:
            self._proc.stdin.close()
            err = self._proc.stderr.read().decode("utf-8", "replace").strip()
            code = self._proc.wait()
            self._proc = None
            if code != 0 and exc[0] is None:
                raise RuntimeError(f"ffmpeg exited {code} writing {self.path}: {err or '(silent)'}")
        elif self._gif and exc[0] is None:
            from PIL import Image
            frames = [Image.fromarray(f) for f in self._gif]
            frames[0].save(self.path, save_all=True, append_images=frames[1:],
                           duration=int(round(1000 / max(self.fps, 1))), loop=0)
        self._gif = []
        return False


def ffmpeg_available() -> bool:
    """For tests and for `--help` text; never for control flow that has a real fallback."""
    return locate_ffmpeg() is not None or shutil.which("ffmpeg") is not None
