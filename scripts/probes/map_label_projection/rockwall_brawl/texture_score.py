"""vision_score_map.py's protocol (each labelled frame's map built from the views around it, the deploy
loop's gates), broken down by TEXTURE in every class and with the full 5-class confusion, for one
held-out folder of projected labels from one map. A copy of Doom Shroom's (../doom_shroom/), which
breaks down every class because that floor has textures of its own (speed strips, pads). Rockwall
Brawl's textures are all walls (cube, cone), so here the breakdown names only those two.

    python texture_score.py --labels DIR --textures CSV --model A.pt [--model B.pt ...]

Each labelled cell's map tile is its centre through the registration in the label's notes
(`A + (i + 0.5) / S`, label_projection.labelled_tiles); its texture is that tile's in CSV.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import vision_score_map as vsm                                              # noqa: E402
from brawl_vision.config import load_vision_config                          # noqa: E402
from brawl_vision.terrain.classifier import TerrainClassifier               # noqa: E402
from brawl_vision.terrain.labeling import CLASSES, hud_plans, load_label_dir, observed_cells  # noqa: E402
from brawl_vision.terrain.occupancy import OccupancyMap                     # noqa: E402
from brawl_vision.terrain.zone import detect_zone                           # noqa: E402

NAMES = [t.name for t in CLASSES]


def registration(notes):
    S = re.search(r"scale \(([-\d.]+), ([-\d.]+)\)", notes)
    A = re.search(r"A=\(([-\d.]+), ([-\d.]+)\)", notes)
    return (float(A.group(1)), float(A.group(2))), (float(S.group(1)), float(S.group(2)))


def build(grids, models, plans, cfg, occluders):
    """{(model, frame): best (rows, cols)} exactly as vsm.score builds them."""
    out = {}
    by_clip = defaultdict(list)
    for g in grids:
        by_clip[g.clip].append(g)
    for clip, gs in sorted(by_clip.items()):
        maps = {(n, g.frame): OccupancyMap.from_config(cfg) for n in models for g in gs}
        for image, takers in vsm.clip_views(clip, gs, plans, cfg):
            covered = occluders(image)
            for g, d in takers:
                plan = plans[g.hud].registered(d)
                rect = plan.rectify(image)
                zone = detect_zone(rect, plan, cfg).at_least(0.05)
                occluded = covered(plan)
                odo = SimpleNamespace(position_tiles=tuple(d), status="ok", segment=0)
                for name, model in models.items():
                    cells, _ = model.predict(rect, plan)
                    maps[(name, g.frame)].update(cells, odo, plan, zone=zone, occluded=occluded, cfg=cfg)
        for g in gs:
            for name in models:
                m = maps[(name, g.frame)]
                m.confidence_floor = cfg.occupancy_confidence_floor
                out[(name, g.frame)] = vsm.label_crop(m, plans[g.hud])
        print(f"  {clip}: {len(gs)} labels built", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--textures", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    tex = [line.rstrip("\n").split(",") for line in open(a.textures) if line.strip("\n")]
    tex = np.array(tex)
    cfg = load_vision_config()
    plans = hud_plans()
    grids = load_label_dir(a.labels, plans["phone"])
    models = {Path(p).name if Path(p).name != "terrain.pt" else str(Path(p).parent.name) + "/terrain.pt":
              TerrainClassifier.load(p, a.device) for p in a.model}
    best = build(grids, models, plans, cfg, vsm._occluders(cfg))
    wall, water, bush = NAMES.index("WALL"), NAMES.index("WATER"), NAMES.index("BUSH")
    blocking = vsm.BLOCKING
    for name in models:
        conf = np.zeros((len(NAMES), len(NAMES)), int)
        per_tex = defaultdict(lambda: [0, 0, 0])     # n, read as its label class, read as blocking
        for g in grids:
            truth = g.as_class_index()
            b = best[(name, g.frame)]
            s = observed_cells(plans[g.hud]) & (truth >= 0) & (b >= 0)
            np.add.at(conf, (truth[s], b[s]), 1)
            A, S = registration(g.notes)
            jj, ii = np.nonzero(s)
            x = np.floor(A[0] + (ii + 0.5) / S[0]).astype(int)
            y = np.floor(A[1] + (jj + 0.5) / S[1]).astype(int)
            for j, i, tx, ty in zip(jj, ii, x, y):
                t = tex[ty, tx] if 0 <= tx < 60 and 0 <= ty < 60 else "off-map"
                k = (NAMES[truth[j, i]], t or "plain")
                per_tex[k][0] += 1
                per_tex[k][1] += int(b[j, i] == truth[j, i])
                per_tex[k][2] += int(blocking[b[j, i]])
        print(f"\n== {name}: {len(grids)} frames, {conf.sum()} cells scored")
        print("   confusion, rows = label, cols = map's answer:")
        print("   " + " " * 7 + "".join(f"{n[:6]:>8}" for n in NAMES) + "   recall")
        for k, n in enumerate(NAMES):
            if conf[k].sum():
                print(f"   {n[:6]:>7}" + "".join(f"{v:>8d}" for v in conf[k]) + f"   {conf[k, k] / conf[k].sum():.3f}")
        print(f"   bush read as water: {conf[bush, water]} of {conf[bush].sum()};  "
              f"water read as bush: {conf[water, bush]} of {conf[water].sum()}")
        print("   cells by label class and texture: n, read as their class, read as blocking")
        for (cls_name, t), (n, w, bl) in sorted(per_tex.items(), key=lambda kv: (kv[0][0], -kv[1][0])):
            print(f"     {cls_name:6s} {t:12s} {n:5d}   {w / n:.3f}   {bl / n:.3f}")


if __name__ == "__main__":
    main()
