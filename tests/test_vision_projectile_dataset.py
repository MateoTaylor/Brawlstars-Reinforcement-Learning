"""`projectile_detection/prepare.py`: from a CVAT export to the lists Ultralytics trains on.

Every archive here is synthetic and built by the test, so this runs on a fresh clone. The real
export is a gigabyte of gitignored game footage.

**The class order in the fixtures is deliberately NOT the order in `classes.py`.** The bug these
tests exist for is a class list written from a constant instead of read from the export. The
three-class export already moved `Projectile` from id 0 to id 2, and a fixture built in the
constant's order would pass against exactly that bug.
"""
import json
import os
import zipfile

import cv2
import numpy as np
import pytest
import yaml

from brawl_vision.object_detection.projectile_detection import prepare as prep
from brawl_vision.object_detection.projectile_detection.classes import (
    CUBE_BOX, CUBE_DROPPED, KNOWN, PROJECTILE,
)

NAMES = {0: CUBE_DROPPED, 1: PROJECTILE, 2: CUBE_BOX}     # not classes.py's order, on purpose
DROP, PROJ, BOX = 0, 1, 2

ROW = {c: f"{c} 0.5 0.5 0.1 0.1" for c in NAMES}


def _jpeg(size=(64, 36)) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((size[1], size[0], 3), 90, np.uint8))
    assert ok
    return buf.tobytes()


def make_export(path, recordings, *, names=NAMES, sizes=None):
    """`recordings` is {name: {frame_index: rows}}. `rows` is a list of label lines, or None for
    a background frame, which (as in CVAT's export) gets no label file at all."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data.yaml", yaml.safe_dump({"names": names, "path": ".",
                                                 "train": "train.txt"}))
        for rec, frames in recordings.items():
            jpeg = _jpeg((sizes or {}).get(rec, (64, 36)))
            for i, rows in frames.items():
                stem = f"{rec}_f{i:06d}"
                zf.writestr(f"images/train/frames/{rec}/{stem}.jpg", jpeg)
                if rows is not None:
                    zf.writestr(f"labels/train/frames/{rec}/{stem}.txt", "\n".join(rows) + "\n")
    return path


def four_recordings():
    """Each class present in at least two recordings, so a split can give it to both halves."""
    return {
        "rec_a": {0: [ROW[BOX], ROW[BOX]], 30: [ROW[PROJ]], 60: None},
        "rec_b": {0: [ROW[BOX], ROW[DROP]], 30: [ROW[PROJ], ROW[PROJ]], 60: None},
        "rec_c": {0: [ROW[DROP]], 30: [ROW[BOX], ROW[PROJ]]},
        "rec_d": {0: [ROW[BOX]], 30: [ROW[PROJ], ROW[DROP]], 60: None},
    }


def listed(path) -> list[str]:
    return path.read_text(encoding="utf-8").split() if path.exists() else []


def recording_of(image_path: str) -> str:
    return image_path.split("/")[-2]


# ---------------------------------------------------------------------------
# class names
# ---------------------------------------------------------------------------

def test_class_ids_come_from_the_export_not_from_classes_py(tmp_path):
    archive = make_export(tmp_path / "export.zip", four_recordings())
    ready = prep.prepare(archive, dataset_dir=tmp_path / "dataset")
    written = yaml.safe_load(ready.data_yaml.read_text(encoding="utf-8"))
    assert written["names"] == NAMES
    assert ready.names == NAMES


def test_the_label_strings_are_the_cvat_labels_verbatim():
    """They reach `Detection.label` unchanged and the colour table keys on them, so a respelling
    here is a silent miss everywhere downstream."""
    assert KNOWN == {"Projectile", "Power Cube Box", "Power Cube Dropped"}


@pytest.mark.parametrize("names, message", [
    ({0: "Power cube box", 1: PROJECTILE}, "does not know"),      # a CVAT typo
    ({0: CUBE_BOX, 1: CUBE_DROPPED}, "no 'Projectile' class"),
    ({0: PROJECTILE, 2: CUBE_BOX}, "must run 0..1"),
    ({0: PROJECTILE, 1: PROJECTILE}, "names a class twice"),
])
def test_names_the_rest_of_the_pipeline_cannot_match_are_refused(tmp_path, names, message):
    archive = make_export(tmp_path / "export.zip", {"rec_a": {0: [ROW[0]]}}, names=names)
    with pytest.raises(ValueError, match=message):
        prep.prepare(archive, dataset_dir=tmp_path / "dataset", no_val=True)


def test_a_names_list_is_read_in_its_own_order(tmp_path):
    archive = make_export(tmp_path / "export.zip", {"rec_a": {0: [ROW[0]]}},
                          names=[CUBE_BOX, PROJECTILE])
    assert prep.read_export(archive)[0] == {0: CUBE_BOX, 1: PROJECTILE}


# ---------------------------------------------------------------------------
# label rows Ultralytics would drop the whole image for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row, message", [
    ("3 0.5 0.5 0.1 0.1", "class id 3, but data.yaml names 3 classes"),
    ("1 0.5 0.5 1.2 0.1", r"outside \[0, 1\]"),
    ("1 0.5 0.5 0.1", "4 fields"),
    ("box 0.5 0.5 0.1 0.1", "not numbers"),
])
def test_a_row_ultralytics_would_reject_fails_naming_the_file(tmp_path, row, message):
    archive = make_export(tmp_path / "export.zip",
                          {"rec_a": {0: [ROW[PROJ]], 30: [ROW[BOX], row]}})
    with pytest.raises(ValueError, match=message) as err:
        prep.prepare(archive, dataset_dir=tmp_path / "dataset", no_val=True)
    assert "rec_a_f000030.txt:2" in str(err.value)


def test_a_bad_export_fails_before_anything_is_unpacked(tmp_path):
    archive = make_export(tmp_path / "export.zip", {"rec_a": {0: ["7 0.5 0.5 0.1 0.1"]}})
    with pytest.raises(ValueError):
        prep.prepare(archive, dataset_dir=tmp_path / "dataset", no_val=True)
    assert not (tmp_path / "dataset").exists()


# ---------------------------------------------------------------------------
# the split
# ---------------------------------------------------------------------------

def test_no_recording_is_in_both_halves_and_every_image_is_in_one(tmp_path):
    recs = four_recordings()
    archive = make_export(tmp_path / "export.zip", recs)
    ready = prep.prepare(archive, dataset_dir=tmp_path / "dataset", val_frac=0.3)
    train = listed(ready.root / "yolo_train.txt")
    val = listed(ready.root / "yolo_val.txt")
    assert val, "the default holds something out"
    assert not {recording_of(p) for p in train} & {recording_of(p) for p in val}
    assert len(train) + len(val) == len(set(train + val)) == sum(map(len, recs.values()))
    assert {recording_of(p) for p in val} == set(ready.val)


def test_every_class_is_scored_even_the_rare_one(tmp_path):
    """Hold out 'the last two recordings' and Dropped -- present in two of eight -- is never
    scored. Weighing every class's share equally is what prevents that."""
    recs = {f"rec_{i}": {0: [ROW[BOX]] * 3, 30: [ROW[PROJ]] * 2} for i in range(8)}
    recs["rec_0"][60] = [ROW[DROP]] * 2
    recs["rec_1"][60] = [ROW[DROP]] * 2
    archive = make_export(tmp_path / "export.zip", recs)
    ready = prep.prepare(archive, dataset_dir=tmp_path / "dataset", val_frac=0.25)
    for c in NAMES:
        held = sum(int(ready.recordings[r].boxes(3)[c]) for r in ready.val)
        trained = sum(int(ready.recordings[r].boxes(3)[c])
                      for r in ready.recordings if r not in ready.val)
        assert held > 0 and trained > 0, f"{NAMES[c]}: {held} held out, {trained} trained on"


def test_each_capture_source_is_in_both_halves(tmp_path):
    """The emulator's 1920x1080 and the phone's 2002x1126 are different captures of the same
    game. A val set drawn from one of them says nothing about the other."""
    recs = {f"phone_{i}": {0: [ROW[BOX], ROW[PROJ], ROW[DROP]] * 3} for i in range(5)}
    recs.update({f"emu_{i}": {0: [ROW[BOX], ROW[PROJ]]} for i in range(2)})
    sizes = {r: (60, 34) for r in recs if r.startswith("emu")}
    archive = make_export(tmp_path / "export.zip", recs, sizes=sizes)
    ready = prep.prepare(archive, dataset_dir=tmp_path / "dataset")
    assert any(r.startswith("emu") for r in ready.val)
    assert any(r.startswith("phone") for r in ready.val)
    assert any(r.startswith("emu") for r in ready.recordings if r not in ready.val)


def test_the_automatic_split_is_the_same_every_time(tmp_path):
    archive = make_export(tmp_path / "export.zip", four_recordings())
    first = prep.prepare(archive, dataset_dir=tmp_path / "dataset", val_frac=0.3).val
    assert prep.prepare(archive, dataset_dir=tmp_path / "dataset", val_frac=0.3).val == first


def test_val_can_be_named_by_a_unique_substring(tmp_path):
    archive = make_export(tmp_path / "export.zip", four_recordings())
    ready = prep.prepare(archive, dataset_dir=tmp_path / "dataset", val="_c, rec_d")
    assert ready.val == ("rec_c", "rec_d")


@pytest.mark.parametrize("spec, message", [
    ("rec_z", "matches no recording"),
    ("rec", "matches 4"),
    ("rec_a,rec_b,rec_c,rec_d", "nothing to train on"),
])
def test_a_val_that_cannot_be_honoured_lists_what_can(tmp_path, spec, message):
    archive = make_export(tmp_path / "export.zip", four_recordings())
    with pytest.raises(ValueError, match=message):
        prep.prepare(archive, dataset_dir=tmp_path / "dataset", val=spec)


def test_no_val_validates_on_the_training_list(tmp_path):
    archive = make_export(tmp_path / "export.zip", four_recordings())
    dataset = tmp_path / "dataset"
    prep.prepare(archive, dataset_dir=dataset)                  # leaves a yolo_val.txt behind
    ready = prep.prepare(archive, dataset_dir=dataset, no_val=True)
    written = yaml.safe_load(ready.data_yaml.read_text(encoding="utf-8"))
    assert written["val"] == written["train"] == "yolo_train.txt"
    assert ready.val == ()
    assert not (ready.root / "yolo_val.txt").exists()


def test_one_recording_cannot_be_split(tmp_path):
    archive = make_export(tmp_path / "export.zip", {"rec_a": {0: [ROW[PROJ]], 30: None}})
    with pytest.raises(ValueError, match="--no-val"):
        prep.prepare(archive, dataset_dir=tmp_path / "dataset")


def test_backgrounds_are_trained_on(tmp_path):
    """A frame with no label file is an explicit 'nothing here', not a gap to skip."""
    archive = make_export(tmp_path / "export.zip", four_recordings())
    ready = prep.prepare(archive, dataset_dir=tmp_path / "dataset", no_val=True)
    assert any(p.endswith("rec_a_f000060.jpg") for p in listed(ready.root / "yolo_train.txt"))


# ---------------------------------------------------------------------------
# unpacking
# ---------------------------------------------------------------------------

def test_each_archive_gets_its_own_tree(tmp_path):
    dataset = tmp_path / "dataset"
    a = prep.prepare(make_export(tmp_path / "old.zip", four_recordings()), dataset_dir=dataset)
    b = prep.prepare(make_export(tmp_path / "new.zip", four_recordings()), dataset_dir=dataset)
    assert a.root != b.root
    assert a.root.parent == b.root.parent == dataset
    assert all(os.path.exists(p) for p in listed(a.root / "yolo_train.txt"))


def test_an_unchanged_archive_is_not_unpacked_twice(tmp_path):
    archive = make_export(tmp_path / "export.zip", four_recordings())
    root = prep.prepare(archive, dataset_dir=tmp_path / "dataset").root
    (root / "sentinel").write_text("still here")
    prep.prepare(archive, dataset_dir=tmp_path / "dataset")
    assert (root / "sentinel").exists()


def test_a_re_export_under_the_same_name_leaves_no_stale_label_behind(tmp_path):
    """The case that forces deleting the tree. A frame labelled in the first export and a
    background in the second has no label file in the second archive -- but extracting over the
    old tree would leave the first one's in place, and Ultralytics would read it."""
    archive = tmp_path / "export.zip"
    recs = four_recordings()
    root = prep.prepare(make_export(archive, recs), dataset_dir=tmp_path / "dataset").root
    stale = root / "labels/train/frames/rec_a/rec_a_f000000.txt"
    assert stale.exists()

    recs["rec_a"][0] = None
    make_export(archive, recs)
    os.utime(archive, ns=(archive.stat().st_atime_ns, archive.stat().st_mtime_ns + 10**9))
    prep.prepare(archive, dataset_dir=tmp_path / "dataset")
    assert not stale.exists()


def test_an_interrupted_unpack_is_redone(tmp_path):
    archive = make_export(tmp_path / "export.zip", four_recordings())
    root = prep.prepare(archive, dataset_dir=tmp_path / "dataset").root
    stamp = json.loads((root / ".prepared.json").read_text(encoding="utf-8"))
    (root / ".prepared.json").write_text(json.dumps({**stamp, "complete": False}))
    (root / "sentinel").write_text("from the interrupted run")
    prep.prepare(archive, dataset_dir=tmp_path / "dataset")
    assert not (root / "sentinel").exists()


def test_a_directory_it_did_not_unpack_is_never_deleted(tmp_path):
    archive = make_export(tmp_path / "export.zip", four_recordings())
    theirs = tmp_path / "dataset" / "export"
    theirs.mkdir(parents=True)
    (theirs / "keep.txt").write_text("somebody's")
    with pytest.raises(FileExistsError, match="did not unpack it"):
        prep.prepare(archive, dataset_dir=tmp_path / "dataset")
    assert (theirs / "keep.txt").exists()


def test_the_newest_archive_is_the_default(tmp_path):
    old = make_export(tmp_path / "zzz_old.zip", four_recordings())
    new = make_export(tmp_path / "aaa_new.zip", four_recordings())
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    assert prep.find_archive(here=tmp_path) == new.resolve()
    assert prep.find_archive("zzz_old.zip", here=tmp_path) == old.resolve()
    with pytest.raises(FileNotFoundError, match="--archive"):
        prep.find_archive(here=tmp_path / "empty_dir_that_has_no_zips")


# ---------------------------------------------------------------------------
# the contract with Ultralytics itself
# ---------------------------------------------------------------------------

def test_ultralytics_finds_every_label_where_prepare_put_it(tmp_path):
    """Checked against Ultralytics' own path function rather than a re-derivation of it: a test
    that mapped image to label with the same rule `prepare` uses could not fail. This is how
    their dataset reads our list (`BaseDataset.get_img_files`), then where it looks for labels."""
    utils = pytest.importorskip("ultralytics.data.utils")
    archive = make_export(tmp_path / "export.zip", four_recordings())
    ready = prep.prepare(archive, dataset_dir=tmp_path / "dataset", val_frac=0.3)
    written = yaml.safe_load(ready.data_yaml.read_text(encoding="utf-8"))
    for key in ("train", "val"):
        images = [p.replace("/", os.sep)
                  for p in listed(prep.Path(written["path"]) / written[key])]
        assert images
        for image, label in zip(images, utils.img2label_paths(images)):
            labelled = not image.endswith("_f000060.jpg")         # the backgrounds above
            assert os.path.exists(image)
            assert os.path.exists(label) == labelled, (image, label)


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------

def test_the_audit_accounts_for_every_recording_and_its_split(tmp_path):
    archive = make_export(tmp_path / "export.zip", four_recordings())
    ready = prep.prepare(archive, dataset_dir=tmp_path / "dataset", val="rec_b")
    text = prep.audit(ready)
    rows = {line.split()[0]: line.split()[-1] for line in text.splitlines()
            if line.startswith("rec_")}
    assert rows == {"rec_a": "train", "rec_b": "val", "rec_c": "train", "rec_d": "train"}
    assert "4 recordings, 11 images, 8 labelled, 3 background" in text


def test_a_box_cut_by_the_frame_edge_is_judged_on_the_side_it_still_has(tmp_path):
    """The shape of the real export's mislabels. Crates sliced flat by the top of the screen are
    common and correct; a Power Cube Box twice a crate's height is not, even at the frame edge
    (in the real export, a pink orb on the left edge)."""
    recs = {"rec_a": {i: [f"{BOX} 0.5 0.5 0.06 0.14"] * 3 for i in range(3)}}
    recs["rec_a"][30] = [f"{BOX} 0.5 0.02 0.06 0.04"]      # a crate's width, top edge cut
    recs["rec_a"][60] = [f"{BOX} 0.03 0.5 0.06 0.30"]      # left edge, but twice as tall
    archive = make_export(tmp_path / "export.zip", recs)
    text = prep.audit(prep.prepare(archive, dataset_dir=tmp_path / "dataset", no_val=True))
    unusual = text.split("most unusual")[1]
    first = next(line for line in unusual.splitlines() if CUBE_BOX in line)
    assert "rec_a_f000060.jpg" in first


def test_the_audit_flags_a_crate_that_vanishes_for_one_frame(tmp_path):
    recs = {"rec_a": {0: [ROW[BOX]] * 2, 30: [ROW[PROJ]], 60: [ROW[BOX]] * 2},
            "rec_b": {0: [ROW[PROJ]] * 2, 30: None, 60: [ROW[PROJ]] * 2}}
    archive = make_export(tmp_path / "export.zip", recs)
    text = prep.audit(prep.prepare(archive, dataset_dir=tmp_path / "dataset", no_val=True))
    # Only the hole lines: rec_a_f000030 also holds a projectile box, which the "most unusual
    # boxes" section may list by file name -- matching the whole text would pass for that alone.
    holes = [line for line in text.splitlines() if "none here" in line]
    assert len(holes) == 1 and "rec_a_f000030.jpg" in holes[0] and CUBE_BOX in holes[0]
    assert not any("rec_b" in line for line in holes), "a projectile leaving is not a missed label"
