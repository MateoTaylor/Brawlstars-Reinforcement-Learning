"""`scripts/vision_score_projectiles.py`: the matching and AP arithmetic behind the promotion check.

Synthetic boxes only. The script's value is that its numbers can be trusted when deciding whether
a model ships, so what is pinned here is the arithmetic: a box matches only its own class, each
true box is matched once, and AP agrees with Ultralytics' so the two can be compared.

Labels are spelled literally rather than imported from `classes.py`, for the reason
`test_deployment_projectiles.py` gives.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from brawl_vision.object_detection.projectile_detection import prepare as prep
from scripts.vision_score_projectiles import (
    average_precision, counts_at, holds_out, iou, label_path, main, match,
)
from tests.test_vision_projectile_dataset import four_recordings, make_export

A = (0.0, 0.0, 10.0, 10.0)
B = (100.0, 100.0, 110.0, 110.0)


def test_iou_of_a_box_with_itself_is_one_and_with_a_stranger_zero():
    assert iou(A, A) == pytest.approx(1.0)
    assert iou(A, B) == 0.0
    assert iou(A, (5.0, 0.0, 15.0, 10.0)) == pytest.approx(50 / 150)


def test_a_box_only_ever_matches_its_own_class():
    """A crate called a projectile is not a correct projectile. It is a false positive, and it
    is named as a confusion, because that failure differs from a projectile that is not there."""
    records, confused = match([("Projectile", 0.9, A)], [("Power Cube Box", A)])
    assert records == [("Projectile", 0.9, False)]
    assert confused == [("Projectile", "Power Cube Box", 0.9)]


def test_each_true_box_is_matched_once_and_the_more_confident_box_takes_it():
    records, _ = match([("Projectile", 0.4, A), ("Projectile", 0.8, A)], [("Projectile", A)])
    assert records == [("Projectile", 0.8, True), ("Projectile", 0.4, False)]


def test_a_box_below_the_iou_threshold_is_a_miss_not_a_near_hit():
    shifted = (6.0, 0.0, 16.0, 10.0)                       # IoU 4/16 = 0.25
    assert match([("Projectile", 0.9, shifted)], [("Projectile", A)])[0][0][2] is False
    assert match([("Projectile", 0.9, shifted)], [("Projectile", A)], iou_thr=0.2)[0][0][2]


def test_raising_the_threshold_only_removes_matches_it_never_reassigns_them():
    """The property that lets one low-threshold pass stand in for a pass at every threshold."""
    preds = [("Projectile", 0.9, A), ("Projectile", 0.3, A), ("Projectile", 0.5, B)]
    truth = [("Projectile", A), ("Projectile", B)]
    records, _ = match(preds, truth)
    assert counts_at(records, 2, 0.2) == (2, 1, 0)
    assert counts_at(records, 2, 0.6) == (1, 0, 1)
    assert counts_at(records, 2, 0.95) == (0, 0, 2)


def test_ap_is_as_high_as_it_goes_for_a_perfect_ranking_and_nan_with_nothing_to_find():
    """0.995 rather than 1.0 is Ultralytics' own ceiling, not an off-by-one here: the 101st
    interpolation point lands on the sentinel where precision has already dropped to zero."""
    assert average_precision([("Projectile", 0.9, True)], 1) == pytest.approx(0.995)
    assert average_precision([], 3) == 0.0
    assert math.isnan(average_precision([("Projectile", 0.9, False)], 0))


def test_ap_agrees_with_ultralytics_on_the_same_ranking():
    """The comparison the script's docstring leans on: our AP50 next to the training log's. If
    the two formulas differed, a gap between them would mean nothing. They did differ once:
    Ultralytics 8.4 added a sentinel that stops crediting recall a class never reaches, and this
    test is what caught the copy here still using the old formula.

    Some true boxes are never found, on purpose. That case is exactly where the two formulas
    part company."""
    metrics = pytest.importorskip("ultralytics.utils.metrics")
    rng = np.random.default_rng(0)
    oks = rng.random(60) < 0.6
    confs = np.sort(rng.random(60))[::-1]
    n_true = int(oks.sum()) + 7                            # some true boxes never found
    records = [("Projectile", float(c), bool(ok)) for c, ok in zip(confs, oks)]
    tp = np.cumsum(oks)
    recall, precision = tp / n_true, tp / np.arange(1, len(oks) + 1)
    want, _, _ = metrics.compute_ap(recall, precision)
    assert average_precision(records, n_true) == pytest.approx(want, abs=1e-9)


def test_a_no_val_dataset_is_refused_as_a_held_out_split(tmp_path, capsys):
    """`train.py --no-val` rewrites the shared data file so its val entry names the training list.
    Run afterwards, the promotion check would score training frames and call them held out, and
    they score far better than unseen footage ever does. The data files come from `prepare`
    itself, not written by hand here, so this breaks if prepare changes how --no-val writes them."""
    archive = make_export(tmp_path / "export.zip", four_recordings())
    split = prep.prepare(archive, dataset_dir=tmp_path / "dataset", val="rec_a").data_yaml
    assert holds_out(split)
    fitted = prep.prepare(archive, dataset_dir=tmp_path / "dataset", no_val=True).data_yaml
    assert not holds_out(fitted)
    with pytest.raises(SystemExit):
        main(["--data", str(fitted), "--model", "unused.onnx"])
    assert "--no-val" in capsys.readouterr().err


def test_labels_are_found_where_ultralytics_looks_for_them():
    """The LAST `images` directory, so a dataset that itself lives under a folder called images
    still resolves -- which is Ultralytics' own `img2label_paths` rule."""
    image = Path("C:/data/images/set/images/train/frames/rec/rec_f000090.jpg")
    assert label_path(image) == Path("C:/data/images/set/labels/train/frames/rec/rec_f000090.txt")
