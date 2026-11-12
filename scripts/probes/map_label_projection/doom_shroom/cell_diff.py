"""Where classifiers' held-out maps disagree with the label, tile by tile: texture_score.py's build
(vision_score_map.py's protocol), then every map tile any model gets wrong in any frame, with each
model's answer per frame that scores it (`.#b~f` = floor, wall, bush, water, fence).

    python cell_diff.py --labels DIR --textures CSV --model A.pt --model B.pt

Models are named m0, m1, ... in the order given. It found which tiles v3 misses that v2 didn't
(MAP_ONBOARDING.md, Doom Shroom, "Classifier v3").
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import texture_score as ts                                                  # noqa: E402
import vision_score_map as vsm                                              # noqa: E402
from brawl_vision.config import load_vision_config                          # noqa: E402
from brawl_vision.terrain.classifier import TerrainClassifier               # noqa: E402
from brawl_vision.terrain.labeling import hud_plans, load_label_dir, observed_cells  # noqa: E402

CH = ".#b~f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--textures", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    tex = np.array([line.rstrip("\n").split(",") for line in open(a.textures) if line.strip("\n")])
    cfg = load_vision_config()
    plans = hud_plans()
    grids = load_label_dir(a.labels, plans["phone"])
    names = [f"m{k}" for k in range(len(a.model))]
    models = {n: TerrainClassifier.load(p, a.device) for n, p in zip(names, a.model)}
    for n, p in zip(names, a.model):
        print(f"{n} = {p}")
    best = ts.build(grids, models, plans, cfg, vsm._occluders(cfg))
    # per map tile: its label, its texture, and per model the answers of the frames that score it
    tiles = defaultdict(lambda: {"label": None, "tex": "", "ans": defaultdict(list), "frames": []})
    for g in grids:
        truth = g.as_class_index()
        A, S = ts.registration(g.notes)
        for n in names:
            b = best[(n, g.frame)]
            s = observed_cells(plans[g.hud]) & (truth >= 0) & (b >= 0)
            jj, ii = np.nonzero(s)
            x = np.floor(A[0] + (ii + 0.5) / S[0]).astype(int)
            y = np.floor(A[1] + (jj + 0.5) / S[1]).astype(int)
            for j, i, tx, ty in zip(jj, ii, x, y):
                t = tiles[(int(tx), int(ty))]
                t["label"] = CH[truth[j, i]]
                t["tex"] = tex[ty, tx] if 0 <= tx < 60 and 0 <= ty < 60 else "off-map"
                t["ans"][n].append(CH[b[j, i]])
                if n == names[0]:
                    t["frames"].append(g.frame)
    print(f"\n{len(tiles)} tiles scored; tiles where any model misses in any frame:")
    print("  (x, y)   label texture       " + "  ".join(f"{n:>14s}" for n in names) + "   frames")
    wrong = {n: 0 for n in names}
    for (x, y), t in sorted(tiles.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if not any(c != t["label"] for n in names for c in t["ans"][n]):
            continue
        cols = []
        for n in names:
            ans = t["ans"][n]
            w = sum(c != t["label"] for c in ans)
            wrong[n] += w
            cols.append(f"{''.join(ans):>10s} {w:>2d}/{len(ans):<2d}")
        print(f"  ({x:2d},{y:2d})  {t['label']}     {t['tex'] or 'plain':12s}  " + "  ".join(cols)
              + "   " + " ".join(str(f) for f in t["frames"]))
    print("cells wrong: " + ", ".join(f"{n} {w}" for n, w in wrong.items()))


if __name__ == "__main__":
    main()
