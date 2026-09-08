"""Where the third-party detector weights live and how they get here.

**They are fetched, not committed.** Two independent reasons, either of which alone would be
enough. They are 10.6 MB against a repo whose largest tracked file is a 753 KB checkpoint; and
they are somebody else's AGPL-licensed artifact, which this repo is not in a position to
redistribute. `weights/NOTICE.md` is tracked and records the provenance, the SHA-256, and the
licence -- the file that is missing is only the tensor blob, and one command puts it back.

This mirrors how `data/` is treated: the folder structure and the README are tracked, the heavy
opaque payload is not.
"""
import hashlib
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# Upstream is a directory literally named ".onnx" inside one named "entity detection", so the URL
# needs both a %20 and a path segment starting with a dot. Spelled out here rather than built from
# the model name, because that naming is upstream's business and will not generalize.
_BASE = ("https://raw.githubusercontent.com/AngelFireLA/BrawlStarsBotMaking/main/"
         "models/entity%20detection/.onnx/")

# name -> (filename, url, sha256). The digest is checked on every fetch: these come from a `main`
# branch with no tags and no releases, so "the file at that URL" is not a fixed thing, and a
# silently different model would show up as mysteriously worse boxes rather than as an error.
MODELS = {
    "entity_v2": (
        "PylaEntityDetectorV2.onnx",
        _BASE + "PylaEntityDetectorV2.onnx",
        "e9e2394ab3b65c92b88334a4ecc6675c054ee717c5fea9f35ad8f111ba13a9a4",
    ),
}

DEFAULT_MODEL = "entity_v2"


def weights_path(model: str = DEFAULT_MODEL) -> Path:
    """Where `model`'s file belongs, whether or not it is there yet."""
    if model not in MODELS:
        raise KeyError(f"unknown detector {model!r} -- known: {', '.join(sorted(MODELS))}")
    return WEIGHTS_DIR / MODELS[model][0]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(model: str = DEFAULT_MODEL, force: bool = False) -> Path:
    """Download `model` if it is not already here, verify its digest, return its path.

    Returns early on an existing file WITHOUT hashing it -- an 11 MB digest on every detector
    construction is a cost paid to catch a case (someone edited the blob in place) that does not
    happen. `force=True` re-downloads, and the fetch path always verifies.
    """
    path = weights_path(model)
    if path.exists() and not force:
        return path
    _, url, digest = MODELS[model]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    got = sha256(tmp)
    if got != digest:
        tmp.unlink()
        raise RuntimeError(
            f"{url} hashed {got}, expected {digest}. Upstream tracks `main` with no releases, so "
            f"the file there may simply have been replaced -- compare against "
            f"brawl_vision/object_detection/weights/NOTICE.md before updating MODELS."
        )
    tmp.replace(path)
    return path


def require(model: str = DEFAULT_MODEL) -> Path:
    """`model`'s path, or raise with the one command that fixes it. Never downloads.

    Separate from `fetch` because a detector constructed deep inside a render loop must not
    quietly reach for the network. Scripts call `fetch` up front, by an explicit flag or by
    default; the detector itself calls this.
    """
    path = weights_path(model)
    if not path.exists():
        raise FileNotFoundError(
            f"detector weights are missing: {path}\n"
            f"They are fetched rather than committed (third-party, AGPL, 10.6 MB -- see "
            f"weights/NOTICE.md). Get them with:\n"
            f"    python scripts/vision_fetch_detector.py"
        )
    return path
