"""Train the per-cell terrain classifier on labelled frames. See
Terrain_Perception_Build_Plan.md Phase H.

    python scripts/vision_train_terrain.py                          # train on every label file
    python scripts/vision_train_terrain.py --hold-out showdown_alternate_map
    python scripts/vision_train_terrain.py --epochs 120 --out brawl_vision/data/terrain.pt

**Validation is split BY MAP, not by frame.** Holding out random cells from maps the model also
trained on measures memorisation, which this pipeline explicitly does not want: the goal is to
survive a reskin, so the only honest question is how it does on a map it has never seen.
`--hold-out` takes a clip name and keeps it entirely out of training.

With one map's labels there is nothing to hold out and the script says so rather than reporting a
same-map number that would look like a result.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
from brawl_vision.config import load_vision_config
from brawl_vision.sources import open_source
from brawl_vision.terrain.classifier import build_examples, evaluate, train
from brawl_vision.terrain.labeling import CLASSES, load_label_dir
from brawl_vision.terrain.zone import detect_zone

REPO = Path(__file__).resolve().parent.parent
CLIPS = REPO / "tests" / "fixtures" / "vision"
LABELS = CLIPS / "labels"


def _rects_for(grids, plan, cfg):
    """Decode each labelled frame once. Sequential, because a label is addressed by frame index and
    seeking returns a frame near it rather than that one."""
    wanted = {}
    for g in grids:
        wanted.setdefault(g.clip, set()).add(g.frame)
    out = {}
    for clip, frames in wanted.items():
        with open_source(CLIPS / f"{clip}.mp4", cfg) as src:
            for frame in src:
                if frame.index in frames:
                    out[(clip, frame.index)] = plan.rectify(frame.image)
                if frame.index > max(frames):
                    break
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default=str(LABELS), help="directory of label .json files")
    p.add_argument("--hold-out", default=None, help="clip name to keep out of training entirely")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--no-augment", action="store_true", help="disable colour/flip augmentation")
    p.add_argument("--out", default=str(REPO / "brawl_vision" / "data" / "terrain.pt"))
    args = p.parse_args(argv)

    cfg = load_vision_config()
    plan = build_rectify_plan(load_camera_model(), load_hud_mask())
    grids = load_label_dir(args.labels, plan)
    if not grids:
        print(f"no label files in {args.labels}. Run scripts/vision_label.py first.",
              file=sys.stderr)
        return 1

    rects = _rects_for(grids, plan, cfg)
    exclude = {(g.clip, g.frame): detect_zone(rects[(g.clip, g.frame)], plan, cfg).at_least(0.05)
               for g in grids}
    examples = build_examples([(g, rects[(g.clip, g.frame)]) for g in grids], plan, exclude)

    clips = sorted({e.clip for e in examples})
    labelled = sum(int((e.target >= 0).sum()) for e in examples)
    print(f"{len(examples)} frames, {labelled} labelled cells, {len(clips)} clips: {clips}")

    if args.hold_out:
        train_set = [e for e in examples if e.clip != args.hold_out]
        val_set = [e for e in examples if e.clip == args.hold_out]
        if not val_set:
            print(f"--hold-out {args.hold_out!r} matched no labelled clip", file=sys.stderr)
            return 1
    else:
        train_set, val_set = examples, []
        if len(clips) > 1:
            print("note: no --hold-out given, so nothing measures cross-map generalization.")

    model = train(train_set, epochs=args.epochs, seed=args.seed, device=args.device,
                  augment_data=not args.no_augment, cfg=cfg, log_every=max(args.epochs // 6, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"wrote {args.out}")

    for name, subset in (("train", train_set), (f"held-out {args.hold_out}", val_set)):
        if not subset:
            continue
        m = evaluate(model, subset)
        print(f"\n{name}: overall {m['overall']:.3f}   WALL-vs-FENCE {m['wall_vs_fence']:.3f}")
        for t in CLASSES:
            print(f"    {t.name:6s} recall {m['recall'][t.name]:.3f}  "
                  f"(n={m['support'][t.name]})")
    if len(clips) < 2:
        print("\nOnly one map is labelled. Cross-map accuracy -- the number this phase is "
              "actually judged on -- cannot be measured until a second map has labels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
