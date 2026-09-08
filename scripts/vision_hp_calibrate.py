"""Regenerate the HP digit templates from real footage. Two passes, with your eyes in between.

    # 1. harvest glyphs and cluster them; writes a contact sheet to look at
    python scripts/vision_hp_calibrate.py harvest tests/fixtures/vision/day10_gameplay.mp4

    # 2. read the digits off clusters.png, left to right, and hand them back
    python scripts/vision_hp_calibrate.py build --labels 0,1,0,6,0,7,9,4,8,6,8,4,9,1,3,2,5,8,2,6,8,8,0,0

**Why a human is in the loop at all.** Nothing in this repo knows what a `7` looks like, and
nothing should: the alternative to labelling is shipping a font file or a hand-drawn glyph set,
and both are guesses about what the game renders rather than measurements of it. Clustering is
what keeps the labelling cheap -- 1184 harvested glyphs collapse to 62 clusters, of which the
largest 24 cover 89% of the bank, so the manual step is reading two dozen digits off one image.

**This is the same contract as the homography.** The output is a MEASUREMENT living in
`hp_detection/data/digits.npz`, regenerated rather than edited, and meaningless to hand-tweak.
Re-run it if the game's HUD font or the capture resolution changes -- both would show up first as
`read.py` returning `no-digits` on footage that used to work.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brawl_vision.config import load_vision_config
from brawl_vision.object_detection import ObjectDetector
from brawl_vision.object_detection.hp_detection import glyphs as glyph_mod
from brawl_vision.object_detection.hp_detection import locate
from brawl_vision.sources import open_source

# Cluster merge threshold. 0.82 was picked by looking at the result: lower and distinct digits
# merge (6 and 8 correlate 0.80 as class means), higher and the bank shatters into hundreds of
# near-duplicates that are tedious to label without being any more informative.
CLUSTER_NCC = 0.82
# Glyphs below this against their assigned cluster are left out of the average -- they are the
# fragments and popup edges that survived the size filter.
ASSIGN_FLOOR = 0.80


def harvest(clips, cfg, every: int, limit: int, quiet: bool):
    """Every digit-sized white glyph inside a detection box, as canonical canvases."""
    detector = ObjectDetector.from_config(cfg, ignore=())
    bank, sources = [], []
    for clip in clips:
        with open_source(clip, cfg) as source:
            for frame in source:
                if frame.index % every:
                    continue
                if limit and frame.index > limit:
                    break
                h, w = frame.image.shape[:2]
                for det in detector.predict(frame.image):
                    if det.label not in ("player", "enemy"):
                        continue
                    x0, y0, x1, y1, box_cx = locate.crop_for(
                        det, frame.image.shape, cfg.hp_crop_height_frac, cfg.hp_crop_pad_frac)
                    if x1 - x0 < 16 or y1 - y0 < cfg.hp_glyph_max_height:
                        continue
                    hsv = cv2.cvtColor(frame.image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
                    row = locate.find_digit_row(
                        hsv, box_cx, max_sat=cfg.hp_white_max_sat,
                        min_val=cfg.hp_white_min_val, min_h=cfg.hp_glyph_min_height,
                        max_h=cfg.hp_glyph_max_height)
                    # >= 3 glyphs: a short row is more likely a fragment than a number, and the
                    # bank only needs clean examples, not every example.
                    if row is None or row.popup_overlap or len(row.blobs) < 3:
                        continue
                    for blob in row.blobs:
                        bank.append(glyph_mod.normalize(blob.mask))
        sources.append(Path(clip).name)
        if not quiet:
            print(f"   {clip}: bank now {len(bank)}", flush=True)
    return np.array(bank, np.float32), sources


def cluster(X: np.ndarray, threshold: float):
    """Greedy single-pass NCC clustering. Returns cluster means, largest first, and their sizes.

    Greedy rather than k-means: the number of clusters is not known (it is not 10 -- each digit
    has several distinct renderings depending on what it sits over), and greedy assignment at a
    fixed correlation is both cheaper and easier to reason about than a fitted partition.
    """
    flat = glyph_mod._whiten(X)
    centres, members = [], []
    for i, row in enumerate(flat):
        if centres:
            sims = np.array([row @ c for c in centres])
            j = int(sims.argmax())
            if sims[j] > threshold:
                members[j].append(i)
                continue
        centres.append(row)
        members.append([i])
    order = sorted(range(len(members)), key=lambda k: -len(members[k]))
    means = np.array([X[members[k]].mean(0) for k in order], np.float32)
    return means, [len(members[k]) for k in order]


def contact_sheet(means: np.ndarray, path: Path, zoom: int = 7, per_row: int = 12) -> Path:
    gh, gw = means.shape[1:]
    rows = (len(means) + per_row - 1) // per_row
    pad, foot = 10, 16
    sheet = np.zeros((rows * (gh * zoom + pad + foot) + pad,
                      per_row * (gw * zoom + pad) + pad), np.uint8)
    for i, mean in enumerate(means):
        r, c = divmod(i, per_row)
        big = cv2.resize(mean, (gw * zoom, gh * zoom), interpolation=cv2.INTER_NEAREST)
        y, x = pad + r * (gh * zoom + pad + foot), pad + c * (gw * zoom + pad)
        sheet[y:y + big.shape[0], x:x + big.shape[1]] = (big * 255).astype(np.uint8)
        cv2.putText(sheet, str(i), (x + 10, y + gh * zoom + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    cv2.imwrite(str(path), sheet)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="collect glyphs from footage and cluster them")
    h.add_argument("clips", nargs="+", help="gameplay .mp4 files")
    h.add_argument("--every", type=int, default=25, help="sample every Nth frame (default: 25)")
    h.add_argument("--limit", type=int, default=9000, help="last frame index to read")
    h.add_argument("--work", default="hp_calibration", help="scratch directory for the bank")
    h.add_argument("--quiet", action="store_true")

    b = sub.add_parser("build", help="label the clusters and write the template file")
    b.add_argument("--labels", required=True,
                   help="comma-separated digit for each cluster in clusters.png, in order")
    b.add_argument("--work", default="hp_calibration")
    b.add_argument("-o", "--out", default=None,
                   help=f"output .npz (default: {glyph_mod.TEMPLATES_PATH})")
    args = p.parse_args(argv)

    work = Path(args.work)
    cfg = load_vision_config()

    if args.cmd == "harvest":
        work.mkdir(parents=True, exist_ok=True)
        X, sources = harvest(args.clips, cfg, args.every, args.limit, args.quiet)
        if not len(X):
            print("no glyphs found -- is the detector finding boxes at all? "
                  "try scripts/vision_detect.py first", file=sys.stderr)
            return 2
        means, sizes = cluster(X, CLUSTER_NCC)
        np.save(work / "glyphs.npy", X)
        np.save(work / "cluster_means.npy", means)
        (work / "sources.txt").write_text("\n".join(sources))
        sheet = contact_sheet(means[:24], work / "clusters.png")
        print(f"\n{len(X)} glyphs -> {len(means)} clusters "
              f"(top 24 cover {sum(sizes[:24]) / len(X):.0%} of the bank)")
        print(f"sizes: {sizes[:24]}")
        print(f"\nlook at {sheet}, read the digits left to right, then:\n"
              f"    python scripts/vision_hp_calibrate.py build --labels "
              f"{','.join('?' * min(24, len(means)))}".replace("?,", "?,"))
        return 0

    labels = [int(v) for v in args.labels.split(",") if v.strip() != ""]
    if any(d not in range(10) for d in labels):
        print(f"labels must be digits 0-9, got {labels}", file=sys.stderr)
        return 2
    X = np.load(work / "glyphs.npy")
    means = np.load(work / "cluster_means.npy")[:len(labels)]
    if len(labels) != len(means):
        print(f"{len(labels)} labels for {len(means)} clusters", file=sys.stderr)
        return 2
    missing = sorted(set(range(10)) - set(labels))
    if missing:
        print(f"no cluster was labelled {missing} -- every digit needs at least one example. "
              f"Harvest more footage, or label more clusters.", file=sys.stderr)
        return 2

    sims = glyph_mod._whiten(X) @ glyph_mod._whiten(means).T
    assigned = np.array(labels)[sims.argmax(1)]
    keep = sims.max(1) > ASSIGN_FLOOR
    templates = np.zeros((10, glyph_mod.GLYPH_H, glyph_mod.GLYPH_W), np.float32)
    counts = []
    for digit in range(10):
        sel = keep & (assigned == digit)
        counts.append(int(sel.sum()))
        templates[digit] = X[sel].mean(0)
    sources = (work / "sources.txt").read_text().splitlines() if \
        (work / "sources.txt").exists() else []
    out = glyph_mod.save(templates, args.out, counts=counts, n_glyphs=len(X),
                         sources=sources or [""])

    print(f"wrote {out}")
    print(f"   {int(keep.sum())}/{len(X)} glyphs used; per-digit counts {counts}")
    whitened = glyph_mod._whiten(templates)
    confusion = whitened @ whitened.T
    np.fill_diagonal(confusion, 0.0)
    a, bx = np.unravel_index(confusion.argmax(), confusion.shape)
    print(f"   worst confusable pair: {a} vs {bx} at NCC {confusion[a, bx]:.3f}")
    contact_sheet(templates, work / "templates.png", per_row=10)
    print(f"   {work / 'templates.png'} should read 0123456789 -- check it before committing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
