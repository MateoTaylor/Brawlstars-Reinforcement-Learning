"""Score projectile-model ONNX files per class on held-out frames. The check before promotion.

    python scripts/vision_score_projectiles.py
    python scripts/vision_score_projectiles.py --model path/to/best.onnx --model other.onnx
    python scripts/vision_score_projectiles.py --recordings 9-10_new3

With no `--model` it scores the newest run's `best.onnx` and the promoted model side by side, on
the held-out frames of the newest prepared dataset.

**It scores the ONNX that ships, through the class that runs it.** Ultralytics' validation, the
numbers in `results.csv`, scores `best.pt` with its own letterbox and its own decode. The live
loop runs `best.onnx` through `ProjectileDetector`, which does both of those itself. So this goes
through `ProjectileDetector.predict`, and it reports AP50 with the same 101-point interpolation
Ultralytics uses. If the two AP50s disagree by more than noise, the export or our decode has
drifted from what was trained. That is worth knowing before promotion rather than after.

**Per class, because the average hides the question.** Promotion asks "did projectiles get
worse?" and "are crates and cubes good enough to use?" One mAP over all three answers neither.

**Per recording, because a model scored on frames it trained on has not been scored.** The
held-out split is by recording, but only for the run that made it. An earlier model may well have
trained on some of those same recordings. The model promoted on 2026-09-06 saw two of the three
held out from the three-class export, so only rows it never trained on are a fair comparison.
This prints every recording's row and leaves the judgement to the reader.

Matching is the standard one: in confidence order, each box takes the unmatched ground-truth box
of its own class with the highest IoU, if that IoU is at least `--iou`. Anything else is a false
positive. Because the order is by confidence, the matches at a high threshold are exactly a
prefix of the matches at a low one. So each frame runs once at the lowest threshold, and every
threshold is read off the same matches.
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from brawl_vision.config import load_vision_config
from brawl_vision.object_detection.projectile_detection import ProjectileDetector
from brawl_vision.object_detection.projectile_detection import weights as _weights
from brawl_vision.object_detection.projectile_detection.prepare import DATASET

DEFAULT_CONFS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)
# Low enough to trace the whole precision-recall curve for AP. Ultralytics validates at 0.001; at
# 0.01 the padding rows (~0.003 on this NMS-free head) are already gone, and nothing a deployed
# threshold would ever keep is lost.
FLOOR_CONF = 0.01


# ---------------------------------------------------------------- the arithmetic

def iou(a, b) -> float:
    """IoU of two `(x0, y0, x1, y1)` boxes."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, truth, iou_thr: float = 0.5):
    """One frame. `preds` are `(label, conf, xyxy)`, `truth` are `(label, xyxy)`.

    Returns `(records, confused)`. `records` is `(label, conf, is_tp)` per prediction, in
    confidence order. `confused` is `(pred_label, true_label, conf)` for each false positive that
    sits on another class's box, which is the one kind of false positive worth naming: a crate
    called a projectile is a different failure from a projectile that is not there.
    """
    taken = [False] * len(truth)
    records, confused = [], []
    for label, conf, box in sorted(preds, key=lambda p: -p[1]):
        best, best_i = 0.0, -1
        for i, (t_label, t_box) in enumerate(truth):
            if taken[i] or t_label != label:
                continue
            o = iou(box, t_box)
            if o > best:
                best, best_i = o, i
        if best >= iou_thr:
            taken[best_i] = True
            records.append((label, conf, True))
            continue
        records.append((label, conf, False))
        other = max(((iou(box, t_box), t_label) for t_label, t_box in truth if t_label != label),
                    default=(0.0, None))
        if other[0] >= iou_thr:
            confused.append((label, other[1], conf))
    return records, confused


def counts_at(records, n_true: int, conf: float) -> tuple[int, int, int]:
    """`(tp, fp, fn)` for one class's records at one threshold."""
    tp = sum(1 for _, c, ok in records if c >= conf and ok)
    fp = sum(1 for _, c, ok in records if c >= conf and not ok)
    return tp, fp, n_true - tp


def average_precision(records, n_true: int) -> float:
    """AP at the matching IoU, with Ultralytics' 101-point interpolation (`metrics.compute_ap`),
    so the number is comparable with the training log's mAP50.

    Written out rather than imported, because nothing on the inference side imports ultralytics.
    Its formula has changed before, and a test pins this copy to the installed version. As of
    8.4 there is an extra sentinel at `(recall[-1], 0)`. Without it, precision ramps linearly from
    the last recall reached out to recall 1, which credits a class for recall it never achieved.
    That overstated Projectile's AP most, since it has the lowest recall ceiling.

    A perfect ranking scores 0.995, not 1.0, in Ultralytics too. The 101st sample sits exactly
    on the recall-1 sentinel, where the curve has already dropped to zero.
    """
    if n_true == 0:
        return float("nan")
    if not records:
        return 0.0
    ordered = sorted(records, key=lambda r: -r[1])
    tp = np.cumsum([1.0 if ok else 0.0 for _, _, ok in ordered])
    fp = np.cumsum([0.0 if ok else 1.0 for _, _, ok in ordered])
    recall = tp / n_true
    precision = tp / (tp + fp)
    mrec = np.concatenate(([0.0], recall, [recall[-1]], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


# ---------------------------------------------------------------- the data

def newest_data_yaml() -> Path:
    found = sorted(DATASET.glob("*/yolo_data.yaml"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no prepared dataset under {DATASET}; run train.py --audit-only")
    return found[-1]


def holds_out(data_yaml: Path) -> bool:
    """False after a `train.py --no-val` run: its data file still has a `val` entry, but that
    entry names the training list. Scoring it would score training frames under a held-out name."""
    spec = yaml.safe_load(data_yaml.read_text())
    return spec["val"] != spec["train"]


def label_path(image: Path) -> Path:
    """Ultralytics' own rule: the last `images` directory becomes `labels`, the suffix `.txt`."""
    parts = list(image.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_split(data_yaml: Path, split: str = "val"):
    """`(names, [(image_path, recording, [(label, xyxy_normalised)])])` for one split."""
    spec = yaml.safe_load(data_yaml.read_text())
    names = {int(k): v for k, v in spec["names"].items()}
    root = Path(spec.get("path") or data_yaml.parent)
    listing = root / spec[split]
    frames = []
    for line in listing.read_text().splitlines():
        if not line.strip():
            continue
        image = Path(line.strip())
        image = image if image.is_absolute() else root / image
        lbl = label_path(image)
        truth = []
        if lbl.exists():
            for row in lbl.read_text().splitlines():
                if row.strip():
                    c, cx, cy, w, h = row.split()[:5]
                    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
                    truth.append((names[int(c)], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        frames.append((image, image.parent.name, truth))
    return names, frames


# ---------------------------------------------------------------- scoring one model

def score(detector, frames, iou_thr: float):
    """Run the detector over every frame once and match. Returns per-class and per-recording
    records, true-box counts, cross-class confusions and the mean predict time in ms."""
    records = defaultdict(list)                         # label -> [(label, conf, ok)]
    n_true = defaultdict(int)
    rec_records = defaultdict(lambda: defaultdict(list))
    rec_true = defaultdict(lambda: defaultdict(int))
    confused = []
    spent = 0.0
    for image_path, recording, truth_norm in frames:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        h, w = image.shape[:2]
        truth = [(lab, (x0 * w, y0 * h, x1 * w, y1 * h)) for lab, (x0, y0, x1, y1) in truth_norm]
        t0 = time.perf_counter()
        dets = detector.predict(image, conf=FLOOR_CONF)
        spent += time.perf_counter() - t0
        recs, conf_pairs = match([(d.label, d.confidence, d.xyxy) for d in dets], truth, iou_thr)
        confused += conf_pairs
        for lab, _ in truth:
            n_true[lab] += 1
            rec_true[recording][lab] += 1
        for r in recs:
            records[r[0]].append(r)
            rec_records[recording][r[0]].append(r)
    return records, n_true, rec_records, rec_true, confused, 1e3 * spent / max(len(frames), 1)


# ---------------------------------------------------------------- reporting

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if tp else 0.0
    return p, r, f


def report(name, detector, result, classes, confs, at, say):
    records, n_true, rec_records, rec_true, confused, ms = result
    model_classes = set(detector.names.values())
    say(f"\n=== {name}")
    say(f"    {sorted(model_classes)} on {detector.provider}, {ms:.1f} ms/frame")
    summary = {}
    for label in classes:
        if label not in model_classes:
            say(f"\n  {label}: not a class of this model")
            continue
        recs, n = records[label], n_true[label]
        ap = average_precision(recs, n)
        say(f"\n  {label}  ({n} true boxes)  AP50 {ap:.3f}")
        say("     conf     P      R     F1     TP    FP    FN")
        best = (0.0, None)
        for c in confs:
            tp, fp, fn = counts_at(recs, n, c)
            p, r, f = _prf(tp, fp, fn)
            best = max(best, (f, c), key=lambda x: x[0])
            mark = " <- config" if abs(c - at) < 1e-9 else ""
            say(f"     {c:4.2f}  {p:5.2f}  {r:5.2f}  {f:5.2f}  {tp:5d} {fp:5d} {fn:5d}{mark}")
        if best[1] is not None:
            say(f"     best F1 {best[0]:.2f} at conf {best[1]:.2f}")
        summary[label] = ap
    if confused:
        pairs = defaultdict(int)
        for pred, true, c in confused:
            if c >= at:
                pairs[(pred, true)] += 1
        if pairs:
            say(f"\n  at conf {at:.2f}, false positives sitting on another class's box:")
            for (pred, true), k in sorted(pairs.items(), key=lambda kv: -kv[1]):
                say(f"     called {pred!r}, was {true!r}: {k}")
    say(f"\n  at conf {at:.2f}, per recording (recall TP/true, then false positives):")
    shown = [c for c in classes if c in model_classes]
    say("     " + f"{'recording':34s}" + "".join(f"{c[:18]:>24s}" for c in shown))
    for recording in sorted(rec_true.keys() | rec_records.keys()):
        cells = []
        for label in shown:
            n = rec_true[recording][label]
            tp, fp, _ = counts_at(rec_records[recording][label], n, at)
            cells.append(f"{tp:3d}/{n:<3d} FP {fp:<3d}" if n else f"  -     FP {fp:<3d}")
        say("     " + f"{recording[:34]:34s}" + "".join(f"{c:>24s}" for c in cells))
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", default=None,
                   help="an .onnx to score; repeat to compare (default: the newest run's "
                        "best.onnx and the promoted weights/projectiles.onnx)")
    p.add_argument("--data", default=None,
                   help="a prepared yolo_data.yaml (default: the newest under dataset/)")
    p.add_argument("--split", default="val", choices=("val", "train"))
    p.add_argument("--recordings", default=None,
                   help="comma-separated substrings; score only recordings matching one")
    p.add_argument("--iou", type=float, default=0.5, help="IoU to count a match (default 0.5)")
    p.add_argument("--confs", default=",".join(f"{c:g}" for c in DEFAULT_CONFS),
                   help="thresholds to tabulate, comma-separated")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = p.parse_args(argv)

    cfg = load_vision_config()
    # The configured threshold is always a row, so "what the loop gets today" is never missing.
    confs = {float(c) for c in args.confs.split(",") if c.strip()}
    confs = tuple(sorted(confs | {cfg.projectile_conf}))
    data_yaml = Path(args.data) if args.data else newest_data_yaml()
    if args.split == "val" and not holds_out(data_yaml):
        p.error(f"{data_yaml} holds nothing out: it was last prepared with --no-val, so its val "
                f"list is the training list. Re-create the held-out split without training "
                f"(train.py --audit-only --val <recordings>), or score chosen recordings "
                f"knowingly with --split train --recordings <names>.")
    names, frames = load_split(data_yaml, args.split)
    if args.recordings:
        wanted = [w.strip() for w in args.recordings.split(",") if w.strip()]
        frames = [f for f in frames if any(w in f[1] for w in wanted)]
        if not frames:
            p.error(f"no {args.split} recording matches {args.recordings!r}")

    models = args.model
    if not models:
        runs = _weights.trained_runs()
        models = ([str(runs[0])] if runs else []) + [str(_weights.DEFAULT_PATH)]
    say = lambda *a: print(*a, flush=True)                       # noqa: E731
    say(f"data  {data_yaml}  ({args.split}: {len(frames)} frames, "
        f"{len({f[1] for f in frames})} recordings), match at IoU >= {args.iou:g}")

    classes = [names[i] for i in sorted(names)]
    for path in models:
        try:
            det = ProjectileDetector.from_config(cfg, path=path, conf=FLOOR_CONF,
                                                 device=args.device)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"\n=== {path}\n    {exc}", file=sys.stderr)
            continue
        report(path, det, score(det, frames, args.iou), classes, confs, cfg.projectile_conf, say)
    return 0


if __name__ == "__main__":
    sys.exit(main())
