"""Train the per-cell terrain classifier on labelled frames. See
Terrain_Perception_Build_Plan.md Phase H.

    python scripts/vision_train_terrain.py                          # train on every label file
    python scripts/vision_train_terrain.py --hold-out showdown_alternate_map
    python scripts/vision_train_terrain.py --device cuda --out brawl_vision/data/terrain.pt

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
from brawl_vision.terrain.classifier import build_examples, evaluate, train
from brawl_vision.terrain.labeling import CLASSES, load_label_dir, read_label_frames
from brawl_vision.terrain.zone import detect_zone

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "tests" / "fixtures" / "vision" / "labels"


def _rects_for(grids, plan):
    """Decode each labelled frame once, one sequential pass per clip, exactly as the labeller did
    (`labeling.read_label_frames`): same lookup, same frame counting, same resize."""
    wanted = {}
    for g in grids:
        wanted.setdefault(g.clip, set()).add(g.frame)
    out = {}
    for clip, frames in wanted.items():
        try:
            images = read_label_frames(clip, frames, plan.viewport)
        except FileNotFoundError as e:
            raise SystemExit(f"a label points at a missing recording: {e}")
        missing = sorted(frames - set(images))
        if missing:
            raise SystemExit(f"{clip}.mp4 has no frame(s) {missing}, but labels point at them")
        for index, image in images.items():
            out[(clip, index)] = plan.rectify(image)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default=str(LABELS), help="directory of label .json files")
    p.add_argument("--hold-out", default=None, help="clip name to keep out of training entirely")
    # 60 underfit: in-sample blocking recall 0.87, map F1 0.874 against 0.930 at 200 (2026-09-14).
    p.add_argument("--epochs", type=int, default=200)
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

    rects = _rects_for(grids, plan)
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
