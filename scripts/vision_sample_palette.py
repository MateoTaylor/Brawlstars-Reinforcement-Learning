"""Measure what colour each terrain class actually is in a clip, for retargeting the map palette.

    python scripts/vision_sample_palette.py tests/fixtures/vision/9_5_brawlstars_eval.mp4

`brawl_vision/terrain/palette.py` paints the reconstructed map, and its whole job is to echo the
footage the map is looked at NEXT TO. Brawl Stars reskins its environments, so that table is
correct for one skin and merely arbitrary for the others, and retargeting it is a normal thing to
do rather than a repair. This script is what makes that a measurement instead of a pipette.

**Sampled by classifier label, not by hand.** Point-sampling a screenshot means choosing which
pixel counts, and the choice is invisible afterwards. Here every cell the terrain classifier is
confident about contributes all of its pixels, so a class's colour is a median over thousands of
cells and the count is printed beside it -- a class with four cells is visibly not a sample.

**Percentiles, not just a mean, because of walls.** A WALL cell contains a lit top face and a
shadowed side. Their average is a colour that appears nowhere on screen, and it is also the wrong
one to draw: what a viewer reads as "wall" is the top. So p50/p75/p90 by luminance are all
reported and the choice between them is left to a human with a reason. `palette.py` records which
one each entry took.

Output is one row per class; paste the hex values into `palette.py` and re-run
`pytest tests/test_vision_evaluate.py -k palette or separable` to check the two constraints the
table has to satisfy (60 RGB between every pair, 0.3 from UNKNOWN in the brightest channel).
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
from brawl_vision.config import load_vision_config
from brawl_vision.sources import open_source
from brawl_vision.terrain.classifier import TerrainClassifier
from brawl_vision.terrain.labeling import CLASSES

DEFAULT_TERRAIN = Path(__file__).resolve().parent.parent / "brawl_vision" / "data" / "terrain.pt"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", help="a gameplay .mp4 whose skin the palette should echo")
    p.add_argument("--step", type=int, default=40,
                   help="sample every Nth frame. The map barely changes between neighbours, so "
                        "this trades nothing for speed (default: 40)")
    p.add_argument("--min-conf", type=float, default=0.85,
                   help="skip cells the classifier is less sure of than this. A hedged cell is "
                        "usually a boundary, and a boundary is a MIX of two classes -- exactly "
                        "the pixels that would drag both medians toward each other (default: .85)")
    p.add_argument("--terrain", default=str(DEFAULT_TERRAIN))
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    cfg = load_vision_config()
    plan = build_rectify_plan(load_camera_model(), load_hud_mask())
    clf = TerrainClassifier.load(args.terrain, device=args.device)
    ppt = plan.pixels_per_tile
    cols, rows = plan.size_tiles

    by_class: dict[int, list] = defaultdict(list)
    with open_source(args.clip, cfg, step=args.step) as src:
        for n, frame in enumerate(src):
            rect = plan.rectify(frame.image)
            cells, conf = clf.predict(rect, plan)
            for r in range(rows):
                for c in range(cols):
                    if conf[r, c] < args.min_conf:
                        continue
                    sl = (slice(r * ppt, (r + 1) * ppt), slice(c * ppt, (c + 1) * ppt))
                    # Fully inside the rectified quad. A partly-valid cell has black padding in
                    # it, which is not a colour the game ever drew.
                    if plan.valid[sl].mean() < 0.99:
                        continue
                    by_class[int(cells[r, c])].append(rect[sl].reshape(-1, 3))
            if n % 20 == 0:
                print(f"   {n:4d} sampled frames", flush=True)

    print(f"\n{'class':7s} {'cells':>6s} {'pixels':>10s}   "
          + "   ".join(f"p{q}".ljust(17) for q in (50, 75, 90)))
    for index, tile in enumerate(CLASSES):
        chunks = by_class.get(index)
        if not chunks:
            print(f"{tile.name:7s} {0:6d}          -   (absent from this clip)")
            continue
        px = np.concatenate(chunks).astype(np.float32)
        # Ordered by LUMA, so "p90" means the brightest tenth of the class's pixels -- the lit
        # face -- rather than the 90th percentile of each channel independently, which would
        # compose a colour out of three unrelated pixels.
        order = np.argsort(px @ np.float32([0.114, 0.587, 0.299]))
        row = [f"{tile.name:7s} {len(chunks):6d} {len(px):10d}"]
        for q in (50, 75, 90):
            band = px[order[int(len(px) * (q - 5) / 100):int(len(px) * (q + 5) / 100)]]
            b, g, r_ = (int(round(v)) for v in np.median(band, axis=0))
            row.append(f"   #{r_:02x}{g:02x}{b:02x} ({r_:3d},{g:3d},{b:3d})")
        print("".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
