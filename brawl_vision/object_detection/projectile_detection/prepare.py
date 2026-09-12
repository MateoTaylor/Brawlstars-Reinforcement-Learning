"""Turn a CVAT "Ultralytics YOLO Detection" export into a training set Ultralytics reads correctly.

`train.py` calls `prepare` and prints `audit`. They live here, not in the script, so an export can
be checked -- and this module tested -- without importing ultralytics or torch
(`train.py --audit-only`).

Each step exists because the export, used as-is, trains the wrong model and raises no error:

  **Class ids come from the export.** CVAT numbers labels in the order they were created in the
  task, and the three-class export moved `Projectile` from id 0 to id 2. A `names:` list written
  by hand here would silently call every crate a projectile. So `names` is read from the export's
  own `data.yaml` and written through unchanged, after checking every name against
  `classes.KNOWN`. A label renamed in CVAT would otherwise drop out of every consumer that
  matches on the string.

  **Every label row is checked before anything is unpacked.** Ultralytics treats an image as
  "corrupt" if its label names a class id >= nc or has a coordinate outside [0, 1]. It skips that
  image with one warning line in a scan of two thousand, and the image is never trained on. A
  class list shorter than the export's is exactly how that happens. Here it is an error that names
  the file.

  **One directory per archive, unpacked again when the archive changes.** The archive goes to
  `dataset/<zip stem>/`, stamped with its size and mtime. A re-export under the same name must not
  reuse the old tree: Ultralytics finds a frame's labels by path, so a label file left over from a
  frame that is now a background would still be read. Deleting the tree is the only way to be sure
  it is gone, and only a tree carrying this module's stamp is ever deleted.

  **Validation holds out whole recordings.** Frames from one recording are seconds apart: the
  same crates, the same brawlers, the same bush. A random frame split scores the model on
  near-copies of its training images, and mAP climbs toward 1.0 whatever the model has learnt.
  Recordings are chosen so each class has about `val_frac` of its boxes held out. Each capture
  source (the phone's 2002x1126, the emulator's 1920x1080) lands in both halves whenever it has
  more than one recording.

The image lists are written from the archive's member list, not from a walk of the disk, so a
stray file under `dataset/` can never join the training set.
"""
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Sequence

import cv2
import numpy as np
import yaml

from .classes import KNOWN, PROJECTILE

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
DEFAULT_VAL_FRAC = 0.15

_STAMP = ".prepared.json"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_FRAME_INDEX = re.compile(r"_f(\d+)$")


@dataclass
class Frame:
    member: str                  # the image's path inside the archive
    labels: list                 # [(class_id, cx, cy, w, h)], normalised; empty for a background
    index: int | None            # the number in `..._f000123.jpg`, for the continuity check

    def count(self, class_id: int) -> int:
        return sum(1 for c, *_ in self.labels if c == class_id)


@dataclass
class Recording:
    name: str
    size: tuple[int, int]        # (width, height) of its frames
    frames: list[Frame] = field(default_factory=list)

    def boxes(self, nc: int) -> np.ndarray:
        ids = np.array([c for f in self.frames for c, *_ in f.labels], np.int64)
        return np.bincount(ids, minlength=nc)


@dataclass
class Prepared:
    archive: Path
    root: Path                   # where the archive was unpacked
    data_yaml: Path              # what to hand `YOLO.train(data=...)`
    names: dict[int, str]
    recordings: dict[str, Recording]
    val: tuple[str, ...]         # held-out recordings; empty means validating on the training set


# ---------------------------------------------------------------------------
# finding and reading the export
# ---------------------------------------------------------------------------

def find_archive(archive: str | Path | None = None, here: Path = HERE) -> Path:
    """The export to train on: `archive` as given, or else the newest `*.zip` in `here`.

    A bare file name is looked up in `here` too, so `--archive x.zip` works from any directory.
    """
    if archive is None:
        found = sorted(here.glob("*.zip"), key=lambda p: (p.stat().st_mtime, p.name))
        if not found:
            raise FileNotFoundError(f"no CVAT export (*.zip) in {here} -- drop one there, or "
                                    f"pass --archive")
        return found[-1].resolve()
    path = Path(archive).expanduser()
    if not path.exists() and not path.is_absolute():
        path = here / path
    if not path.exists():
        raise FileNotFoundError(f"no archive at {archive} (looked in {here} too)")
    return path.resolve()


def label_member(image: str) -> str:
    """The label file Ultralytics will look for: the LAST `images` directory swapped for
    `labels`, extension swapped for `.txt` (`ultralytics.data.utils.img2label_paths`)."""
    parts = list(PurePosixPath(image).parts)
    dirs = parts[:-1]
    if "images" not in dirs:
        raise ValueError(f"{image} is not under an images/ directory -- is this a CVAT "
                         f"'Ultralytics YOLO Detection' export?")
    parts[len(dirs) - 1 - dirs[::-1].index("images")] = "labels"
    return str(PurePosixPath(*parts).with_suffix(".txt"))


def _read_names(zf: zipfile.ZipFile) -> dict[int, str]:
    try:
        meta = yaml.safe_load(zf.read("data.yaml")) or {}
    except KeyError:
        raise ValueError("the archive has no data.yaml at its root -- export from CVAT as "
                         "'Ultralytics YOLO Detection'") from None
    raw = meta.get("names")
    if isinstance(raw, list):
        raw = dict(enumerate(raw))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"data.yaml has no class names (names: {raw!r})")
    names = {int(k): str(v) for k, v in sorted(raw.items(), key=lambda kv: int(kv[0]))}
    if list(names) != list(range(len(names))):
        raise ValueError(f"data.yaml class ids must run 0..{len(names) - 1}, got {list(names)}")
    if len(set(names.values())) != len(names):
        raise ValueError(f"data.yaml names a class twice: {names}")
    unknown = sorted(set(names.values()) - KNOWN)
    if unknown:
        raise ValueError(
            f"data.yaml has labels that classes.py does not know: {unknown}. Known: "
            f"{sorted(KNOWN)}. Detections keep the CVAT label verbatim and everything downstream "
            f"matches on it, so fix the label in CVAT -- or, for a genuinely new class, add it "
            f"to classes.KNOWN.")
    if PROJECTILE not in names.values():
        raise ValueError(f"data.yaml has no {PROJECTILE!r} class ({names}), and this is the "
                         f"projectile detector's training set")
    return names


def _parse_labels(text: str, member: str, nc: int, errors: list[str]) -> list:
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        where = f"{member}:{n}"
        if len(fields) != 5:
            errors.append(f"{where}: {len(fields)} fields, expected 'class cx cy w h'")
            continue
        try:
            c = float(fields[0])
            box = tuple(float(v) for v in fields[1:])
        except ValueError:
            errors.append(f"{where}: not numbers: {line.strip()!r}")
            continue
        if c != int(c) or not 0 <= c < nc:
            errors.append(f"{where}: class id {fields[0]}, but data.yaml names {nc} classes "
                          f"(0..{nc - 1})")
            continue
        if not all(0.0 <= v <= 1.0 for v in box):
            errors.append(f"{where}: coordinates outside [0, 1]: {line.strip()!r}")
            continue
        rows.append((int(c), *box))
    return rows


def _image_size(zf: zipfile.ZipFile, member: str) -> tuple[int, int]:
    img = cv2.imdecode(np.frombuffer(zf.read(member), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"{member} is not a readable image")
    return img.shape[1], img.shape[0]


def read_export(archive: Path) -> tuple[dict[int, str], dict[str, Recording]]:
    """Class names and every image, grouped by recording, read from the archive without
    unpacking it. Raises on anything Ultralytics would silently drop.

    A recording is an image's parent directory, which is how CVAT lays out a task with one video
    per job. An image with no label file is a background, the same as Ultralytics reads it.
    """
    with zipfile.ZipFile(archive) as zf:
        names = _read_names(zf)
        members = zf.namelist()
        present = set(members)
        images = sorted(m for m in members if PurePosixPath(m).suffix.lower() in _IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"{archive.name} contains no images")
        errors: list[str] = []
        recordings: dict[str, Recording] = {}
        for member in images:
            path = PurePosixPath(member)
            label = label_member(member)
            rows = (_parse_labels(zf.read(label).decode("utf-8"), label, len(names), errors)
                    if label in present else [])
            rec = recordings.get(path.parent.name)
            if rec is None:
                rec = recordings[path.parent.name] = Recording(path.parent.name,
                                                               _image_size(zf, member))
            match = _FRAME_INDEX.search(path.stem)
            rec.frames.append(Frame(member, rows, int(match.group(1)) if match else None))
    if errors:
        shown = "\n  ".join(errors[:12])
        more = f"\n  ... and {len(errors) - 12} more" if len(errors) > 12 else ""
        raise ValueError(f"{len(errors)} label rows in {archive.name} that Ultralytics would "
                         f"reject, dropping the whole image with one warning line:"
                         f"\n  {shown}{more}")
    return names, recordings


# ---------------------------------------------------------------------------
# unpacking
# ---------------------------------------------------------------------------

def extract(archive: Path, dataset_dir: Path = DATASET) -> Path:
    """Unpack `archive` to `dataset_dir/<stem>/`, reusing a finished unpack of the same file.

    "The same file" means the same size and mtime. Anything else -- a re-export under the same
    name, or an unpack that was interrupted -- deletes the tree and starts again (see the module
    docstring for why). A directory without this module's stamp was not made here and is refused
    rather than deleted.
    """
    root = dataset_dir / archive.stem
    stat = archive.stat()
    stamp = {"archive": archive.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    marker = root / _STAMP
    if root.exists():
        if not marker.is_file():
            raise FileExistsError(f"{root} exists but prepare.py did not unpack it (there is no "
                                  f"{_STAMP}). Move it aside, or delete it if it is yours to lose.")
        if json.loads(marker.read_text(encoding="utf-8")) == {**stamp, "complete": True}:
            return root
        print(f"{archive.name} changed since it was unpacked, or the unpack never finished -- "
              f"unpacking it again")
        shutil.rmtree(root)
    print(f"unpacking {archive.name} ({stat.st_size / 1e6:.0f} MB) -> {root} ...")
    root.mkdir(parents=True)
    marker.write_text(json.dumps({**stamp, "complete": False}), encoding="utf-8")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    marker.write_text(json.dumps({**stamp, "complete": True}), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# the split
# ---------------------------------------------------------------------------

def choose_val(recordings: dict[str, Recording], nc: int,
               frac: float = DEFAULT_VAL_FRAC) -> tuple[str, ...]:
    """Whole recordings to hold out, so each class has about `frac` of its boxes in validation.

    Greedy and deterministic. It first takes one recording from every frame size that has more
    than one, so each capture source is scored and still trained on. Then it keeps adding
    whichever recording brings the per-class held-out shares closest to `frac`, and stops when
    no addition helps. Every class counts equally, which is what keeps the rare
    `Power Cube Dropped` from ending up with nothing to be scored on.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {frac} -- use no_val to validate on "
                         f"the training set")
    if len(recordings) < 2:
        raise ValueError(f"the export has {len(recordings)} recording(s), so there is nothing to "
                         f"hold out -- pass --no-val")
    boxes = {name: rec.boxes(nc) for name, rec in recordings.items()}
    total = np.maximum(sum(boxes.values()), 1)

    def miss(chosen: list[str]) -> float:
        held = sum((boxes[r] for r in chosen), np.zeros(nc))
        return float(((held / total - frac) ** 2).sum())

    def best(chosen: list[str], pool) -> str:
        return min(sorted(pool), key=lambda r: miss([*chosen, r]))

    chosen: list[str] = []
    by_size = defaultdict(list)
    for name, rec in recordings.items():
        by_size[rec.size].append(name)
    for size in sorted(by_size):
        if len(by_size[size]) > 1:
            chosen.append(best(chosen, by_size[size]))
    while pool := [r for r in recordings if r not in chosen]:
        pick = best(chosen, pool)
        if miss([*chosen, pick]) >= miss(chosen):
            break
        chosen.append(pick)
    return tuple(sorted(chosen))


def resolve_val(spec: str | Sequence[str], recordings: dict[str, Recording]) -> tuple[str, ...]:
    """Recording names from `--val`: comma-separated, each an exact name or a substring that
    matches exactly one (`08-31` rather than `ScreenRecording_08-31-2026_10-16-29_1`)."""
    wanted = [s.strip() for s in (spec.split(",") if isinstance(spec, str) else spec)]
    chosen = set()
    for want in filter(None, wanted):
        hits = [want] if want in recordings else [r for r in recordings if want in r]
        if len(hits) != 1:
            what = "matches no recording" if not hits else f"matches {len(hits)}: {sorted(hits)}"
            listing = "\n  ".join(sorted(recordings))
            raise ValueError(f"--val {want!r} {what}. Recordings in this export:\n  {listing}")
        chosen.add(hits[0])
    if not chosen:
        raise ValueError("--val names no recordings")
    if chosen == set(recordings):
        raise ValueError("--val holds out every recording, which leaves nothing to train on")
    return tuple(sorted(chosen))


def write_split(root: Path, names: dict[int, str], recordings: dict[str, Recording],
                val: tuple[str, ...]) -> Path:
    """`yolo_train.txt`, `yolo_val.txt` and the `yolo_data.yaml` pointing at them, in `root`.

    CVAT's own data.yaml points at a train.txt whose `data/images/...` paths do not resolve from
    here, so these are written instead, with absolute paths. An empty `val` validates on the
    training list.
    """
    def listing(recs) -> str:
        return "".join(f"{(root / f.member).as_posix()}\n"
                       for r in sorted(recs) for f in recordings[r].frames)

    (root / "yolo_train.txt").write_text(listing(r for r in recordings if r not in val),
                                         encoding="utf-8")
    val_list = root / "yolo_val.txt"
    if val:
        val_list.write_text(listing(val), encoding="utf-8")
    else:
        val_list.unlink(missing_ok=True)       # our own output from an earlier split; now unused
    data = {"path": root.as_posix(), "train": "yolo_train.txt",
            "val": "yolo_val.txt" if val else "yolo_train.txt", "names": dict(names)}
    data_yaml = root / "yolo_data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                         encoding="utf-8")
    return data_yaml


def prepare(archive: Path, *, val: str | Sequence[str] | None = None,
            val_frac: float = DEFAULT_VAL_FRAC, no_val: bool = False,
            dataset_dir: Path = DATASET) -> Prepared:
    """Read, check, unpack and split `archive`. The export is checked BEFORE it is unpacked, so a
    bad one fails in seconds rather than after a gigabyte of extraction."""
    if no_val and val:
        raise ValueError("pass --val or --no-val, not both")
    names, recordings = read_export(archive)
    if no_val:
        held: tuple[str, ...] = ()
    elif val:
        held = resolve_val(val, recordings)
    else:
        held = choose_val(recordings, len(names), val_frac)
    root = extract(archive, dataset_dir)
    return Prepared(archive, root, write_split(root, names, recordings, held), names,
                    recordings, held)


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------

def _spread(values) -> str:
    return " / ".join(f"{x:.0f}" for x in np.percentile(values, [0, 10, 50, 90, 100]))


def _cut(centre: float, extent: float, margin: float = 0.02) -> bool:
    """Whether a box runs into the frame edge along one axis (normalised centre and extent).

    2% of the frame, not a pixel or two: labellers stop a box on a half-visible crate 10-20 px
    short of the edge rather than snapping to it.
    """
    return centre - extent / 2 <= margin or centre + extent / 2 >= 1.0 - margin


def _most_unusual(shape: list) -> np.ndarray:
    """Indices into `shape`, most unusual first.

    A whole box scores its aspect ratio or its size against the class's medians, whichever is
    further off. A box cut off by the frame edge has had one dimension cropped by the camera
    rather than by the labeller, so it scores only the dimension it still has: a crate sliced
    to 117x51 by the top of the screen has a normal crate's width and is not news. Medians come
    from the whole boxes alone.
    """
    w, h = (np.log([s[k] for s in shape]) for k in (0, 1))
    cut_x, cut_y = (np.array([s[k] for s in shape]) for k in (5, 6))
    whole = ~cut_x & ~cut_y
    ref = whole if whole.any() else np.ones_like(whole)
    med_w, med_h = np.median(w[ref]), np.median(h[ref])
    med_aspect, med_size = np.median((w - h)[ref]), np.median(((w + h) / 2)[ref])
    score = np.where(whole, np.maximum(abs(w - h - med_aspect), abs((w + h) / 2 - med_size)), 0.0)
    score = np.where(cut_y & ~cut_x, abs(w - med_w), score)
    score = np.where(cut_x & ~cut_y, abs(h - med_h), score)
    return np.argsort(-score, kind="stable")


def audit(p: Prepared, imgsz: int = 960, *, unusual: int = 5) -> str:
    """What the model is about to be trained and scored on, as text.

    Per-recording counts and the split, box sizes at the training resolution, the few boxes
    whose shape or size is furthest from their class (where a mislabel usually hides), and
    frames where a persistent object vanishes between two neighbours that both have it.
    """
    nc = len(p.names)
    short = [p.names[i].split()[-1] for i in range(nc)]
    if len(set(short)) < nc:
        short = [f"c{i}" for i in range(nc)]
    col = max(7, *(len(s) + 1 for s in short))
    frames = [f for rec in p.recordings.values() for f in rec.frames]
    labelled = sum(1 for f in frames if f.labels)
    out = [f"archive   {p.archive.name}: {len(p.recordings)} recordings, {len(frames)} images, "
           f"{labelled} labelled, {len(frames) - labelled} background",
           f"unpacked  {p.root}",
           f"data      {p.data_yaml}",
           "classes   " + ", ".join(f"{i} {n}" for i, n in p.names.items()),
           "",
           f"{'recording':40s} {'size':>9s} {'images':>7s} {'bkgd':>5s}"
           + "".join(f"{s:>{col}s}" for s in short) + "  split"]

    totals = {"train": np.zeros(nc, np.int64), "val": np.zeros(nc, np.int64)}
    images = Counter()
    for name in sorted(p.recordings):
        rec = p.recordings[name]
        split = "val" if name in p.val else "train"
        b = rec.boxes(nc)
        totals[split] += b
        images[split] += len(rec.frames)
        bg = sum(1 for f in rec.frames if not f.labels)
        out.append(f"{name[:40]:40s} {rec.size[0]:>4d}x{rec.size[1]:<4d} {len(rec.frames):7d} "
                   f"{bg:5d}" + "".join(f"{v:{col}d}" for v in b) + f"  {split}")
    for split in ("train", "val"):
        if images[split]:
            out.append(f"{'= ' + split:40s} {'':9s} {images[split]:7d} {'':5s}"
                       + "".join(f"{v:{col}d}" for v in totals[split]))
    if p.val:
        share = totals["val"] / np.maximum(totals["train"] + totals["val"], 1)
        out.append(f"{'= val share of boxes':40s} {'':9s} {'':7s} {'':5s}"
                   + "".join(f"{s:>{col}.0%}" for s in share))
    else:
        out.append("no held-out recordings (--no-val): val is the training set, so mAP will climb "
                   "toward 1.0 and says nothing about new footage")

    # Sizes, and the boxes furthest from their class's typical shape.
    out.append(f"\nbox size, sqrt(w*h) px, min / p10 / median / p90 / max -- in the source frame,"
               f" then letterboxed to {imgsz}")
    odd = []
    for c in range(nc):
        src, fit, shape = [], [], []
        for rec in p.recordings.values():
            W, H = rec.size
            for f in rec.frames:
                for cc, cx, cy, w, h in f.labels:
                    if cc != c:
                        continue
                    s = float(np.sqrt(w * W * h * H))
                    src.append(s)
                    fit.append(s * imgsz / max(W, H))
                    shape.append((w * W, h * H, (cx - w / 2) * W, (cy - h / 2) * H, f.member,
                                  _cut(cx, w), _cut(cy, h)))
        if not src:
            out.append(f"  {p.names[c]:20s} no boxes")
            continue
        out.append(f"  {p.names[c]:20s} {_spread(src):24s}   at {imgsz}: {_spread(fit)}")
        for i in _most_unusual(shape)[:unusual]:
            w, h, x0, y0, member, cut_x, cut_y = shape[i]
            cut = " (cut off by the frame edge)" if cut_x or cut_y else ""
            odd.append(f"  {p.names[c]:20s} {w:4.0f}x{h:<4.0f} px at ({x0:4.0f}, {y0:4.0f})  "
                       f"{PurePosixPath(member).name}{cut}")
    out.append(f"\nthe {unusual} most unusual boxes per class -- shape or size furthest from the "
               f"class median. Where a mislabel usually hides; worth a look in CVAT:")
    out.extend(odd)

    # A crate or a dropped cube does not blink out for one frame. Projectiles do, so they are
    # left out: a gap there is the object leaving, not a missed label.
    holes = []
    for rec in p.recordings.values():
        seq = sorted((f for f in rec.frames if f.index is not None), key=lambda f: f.index)
        for a, b, d in zip(seq, seq[1:], seq[2:]):
            for c in range(nc):
                if p.names[c] == PROJECTILE:
                    continue
                if a.count(c) >= 2 and d.count(c) >= 2 and b.count(c) == 0:
                    holes.append(f"  {p.names[c]:20s} {a.count(c)} before, none here, "
                                 f"{d.count(c)} after  {PurePosixPath(b.member).name}")
    out.append(f"\nframes where a crate or cube class is missing between two neighbours that each "
               f"have 2+ of it -- a missed label, or a real break/pickup: {len(holes)}")
    out.extend(holes[:15])
    if len(holes) > 15:
        out.append(f"  ... and {len(holes) - 15} more")
    return "\n".join(out)
