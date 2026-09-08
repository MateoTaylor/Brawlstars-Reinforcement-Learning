"""Where OUR projectile weights live and how a caller is told when they are not there yet.

The sibling `object_detection/weights.py` fetches somebody else's file from a URL and checks its
digest. **This one downloads nothing**, and the difference is the whole point: that model is a
fixed third-party artifact with one correct answer, this one is produced by `train.py` on the
machine running it and there is no canonical copy to fetch. So the job here is not "get the file",
it is "resolve which of your own runs to use, and say something useful when there are none".

**The default path is a stable name, not a run directory.** `runs/` increments -- `projectiles`,
`projectiles2`, `projectiles3` -- so a config pointing into it would name whichever run happened
to be first and silently keep using it after ten better ones. `weights/projectiles.onnx` is the
one you promoted; copying a `best.onnx` there is the act of choosing it. `require` prints that
copy command with the actual runs it can see, newest first.

Nothing here is committed: `.gitignore` covers this directory for the same reason it covers the
dataset and the `.pt` checkpoints.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEIGHTS_DIR = HERE / "weights"
RUNS_DIR = HERE / "runs"

# The promoted model. A plain name under our own package, resolved relative to this file rather
# than to the working directory, so a script run from anywhere finds the same one.
DEFAULT_PATH = WEIGHTS_DIR / "projectiles.onnx"


def weights_path(configured: str | Path | None = None) -> Path:
    """The file `configured` names, or `DEFAULT_PATH` when it is empty.

    An empty string counts as unset, not as a path to "". YAML has no way to spell "use the
    default" other than omitting the key or leaving it blank, and both should mean the same thing.
    """
    if configured is None or str(configured).strip() == "":
        return DEFAULT_PATH
    return Path(configured).expanduser()


def trained_runs() -> list[Path]:
    """Every `runs/*/weights/best.onnx` that exists, newest first.

    `best.onnx` and not `best.pt`: a `.pt` is a torch checkpoint this package cannot run at all
    (nothing at inference time imports ultralytics), so listing one would be offering something
    that does not work. `train.py` exports the ONNX at the end of every run.
    """
    if not RUNS_DIR.is_dir():
        return []
    found = [p for p in RUNS_DIR.glob("*/weights/best.onnx") if p.is_file()]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def require(configured: str | Path | None = None) -> Path:
    """The resolved path, or raise with the command that fixes it. Never downloads.

    Same contract as `object_detection.weights.require` -- a detector constructed inside a render
    loop must fail with an instruction rather than reach for the network or guess. The instruction
    is different because the fix is: there, fetch a known file; here, either train a model or say
    which of your runs you meant.
    """
    path = weights_path(configured)
    if path.exists():
        return path

    runs = trained_runs()
    if runs:
        lines = "\n".join(f"    {p}" for p in runs)
        hint = (f"\nTrained runs that DO have an exported model, newest first:\n{lines}\n\n"
                f"Promote one (this is how you choose which run is current):\n"
                f"    copy \"{runs[0]}\" \"{path}\"\n"
                f"or point at a run directly, without promoting it:\n"
                f"    --projectile-model \"{runs[0]}\"")
    else:
        hint = ("\nThere are no exported runs under\n"
                f"    {RUNS_DIR}\n"
                "Train one first -- it writes best.onnx beside best.pt when it finishes:\n"
                "    python brawl_vision/object_detection/projectile_detection/train.py")

    raise FileNotFoundError(f"no projectile model at {path}{hint}")
