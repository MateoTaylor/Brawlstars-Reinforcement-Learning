"""Score terrain classifiers the way the deployed map uses them: build the occupancy map around
each labelled frame from the views either side of it, and compare that map with the labels.

    python scripts/vision_score_map.py                          # the shipped weights
    python scripts/vision_score_map.py --model a.pt --model b.pt
    python scripts/vision_score_map.py --hold-out-each bluestacks-example-new bluestacks-example3 \\
        --save-dir runs/terrain_holdout
    python scripts/vision_score_map.py --held-out-model bluestacks-example3=runs/x/held_out_bluestacks-example3.pt
    python scripts/vision_score_map.py --one-mask phone         # the protocol before 2026-09-15

Per-frame accuracy on the labelled frames says little about the map. The labelled frame is the
one view the classifier was trained on, and the map is built from every other view.

**The protocol** (2026-09-14; `occupancy.py`'s 0.942 was measured on it):

- The truth is each labelled frame. Its map is built only from other views: every frame up to
  `REACH_S` either side, at `RATE_HZ`, leaving out the `SKIP_S` nearest the label.
- One `Odometry` per label window, from the window's first frame. A view deposits at its position
  relative to the labelled frame, through the plan registered there (`RectifyPlan.registered`), so
  the map's cells are the label's cells. Views in another odometry segment, or not "ok", are left out.
- The deploy loop's gates: gas (`detect_zone(...).at_least(0.05)`), crate and brawler boxes
  (`loot.crate_occlusion | loot.box_occlusion`) and the footprint (`OccupancyMap`). Nothing freezes.
- Scored over labelled cells inside the label mask's footprint that the map saw. Blocking is WALL,
  WATER or FENCE: recall, precision and F1. "Edge" keeps only the walls touching a free cell, "inner"
  only the walls inside a run; free cells count in both. Then 5-class accuracy.
- Each label is scored under its own HUD mask (`LabelGrid.hud`): odometry, gates and footprint all
  use it. `--one-mask phone` puts every label under the phone's, as the 0.942 measurement did.

**Held-out recordings** (`--hold-out-each`). Per named recording, one classifier trained on every
label except that recording's (the shipped recipe, `build_examples` + `train`), scored on that
recording's windows only. `--save-dir` keeps the checkpoints, and `--held-out-model` scores a kept
one again. That is how a round of new labels is judged. Keep today's checkpoints, label, then run
`--hold-out-each` again alongside `--held-out-model` for the old ones. Both are then scored on the
held-out recording's full label set, new labels included, and differ only in what they trained on.
"""
import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brawl_sim.constants import Tile                                            # noqa: E402
from brawl_vision.clips import load_bounds                                      # noqa: E402
from brawl_vision.config import TERRAIN_WEIGHTS_PATH, load_vision_config        # noqa: E402
from brawl_vision.gameplay import walk_frames                                   # noqa: E402
from brawl_vision.terrain.labeling import (CLASSES, find_clip, hud_plans,       # noqa: E402
                                           load_label_dir, observed_cells)
from brawl_vision.terrain.occupancy import OccupancyMap                         # noqa: E402
from brawl_vision.terrain.odometry import Odometry                              # noqa: E402
from brawl_vision.terrain.zone import detect_zone                               # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "tests" / "fixtures" / "vision" / "labels"

RATE_HZ = 12.0      # the deploy loop's perception rate
REACH_S = 10.0      # views this far either side of a label build its map
SKIP_S = 1.0        # ...but none this close, so none is nearly the labelled view
BLOCKING = np.array([t in (Tile.WALL, Tile.WATER, Tile.FENCE) for t in CLASSES])


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

@dataclass
class Tally:
    """Blocking-vs-free confusion counts."""
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0

    def add(self, truth: np.ndarray, pred: np.ndarray) -> None:
        self.tp += int((truth & pred).sum())
        self.fn += int((truth & ~pred).sum())
        self.fp += int((~truth & pred).sum())
        self.tn += int((~truth & ~pred).sum())

    def __add__(self, other: "Tally") -> "Tally":
        return Tally(self.tp + other.tp, self.fn + other.fn, self.fp + other.fp,
                     self.tn + other.tn)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def f1(self) -> float:
        r, p = self.recall, self.precision
        return 2 * r * p / max(r + p, 1e-9)


@dataclass
class MapScore:
    """One map, or a sum of them, against the labels."""
    blocking: Tally = field(default_factory=Tally)
    edge: Tally = field(default_factory=Tally)
    inner: Tally = field(default_factory=Tally)
    correct: int = 0
    cells: int = 0
    windows: int = 0

    def add(self, truth: np.ndarray, best: np.ndarray, scored: np.ndarray) -> None:
        """`truth` and `best` are (rows, cols) class indices, -1 where unlabelled or never seen;
        `scored` is the cells in play (the label mask's footprint)."""
        s = scored & (truth >= 0) & (best >= 0)
        tb = BLOCKING[np.clip(truth, 0, None)]
        pb = BLOCKING[np.clip(best, 0, None)]
        edges = wall_edges(truth)
        self.blocking.add(tb[s], pb[s])
        self.edge.add(tb[s & (edges | ~tb)], pb[s & (edges | ~tb)])
        self.inner.add(tb[s & ~edges], pb[s & ~edges])
        self.correct += int(((best == truth) & s).sum())
        self.cells += int(s.sum())
        self.windows += 1

    def __add__(self, other: "MapScore") -> "MapScore":
        return MapScore(self.blocking + other.blocking, self.edge + other.edge,
                        self.inner + other.inner, self.correct + other.correct,
                        self.cells + other.cells, self.windows + other.windows)

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.cells, 1)


def wall_edges(truth: np.ndarray) -> np.ndarray:
    """Labelled blocking cells with a labelled free cell beside them (4-neighbour): the walls a
    policy steers around, as opposed to the inside of a run."""
    known = truth >= 0
    blocking = known & BLOCKING[np.clip(truth, 0, None)]
    free = known & ~blocking
    beside = np.zeros_like(free)
    beside[1:] |= free[:-1]
    beside[:-1] |= free[1:]
    beside[:, 1:] |= free[:, :-1]
    beside[:, :-1] |= free[:, 1:]
    return blocking & beside


def label_crop(occupancy: OccupancyMap, plan) -> np.ndarray:
    """The map's cells under the labelled frame, whose camera sits at the map's position 0."""
    cols, rows = plan.size_tiles
    ox, oy = occupancy.origin
    r0, c0 = int(plan.origin_tile[1]) - oy, int(plan.origin_tile[0]) - ox
    return occupancy.best()[r0:r0 + rows, c0:c0 + cols]


# ---------------------------------------------------------------------------
# the views around each label
# ---------------------------------------------------------------------------

def window_frames(label: int, fps: float, last: int, rate_hz: float = RATE_HZ,
                  reach_s: float = REACH_S) -> list[int]:
    """Frame indices of `label`'s window: every `fps / rate_hz`-th frame within `reach_s`, on the
    label's own phase, clamped to [0, last]."""
    step = max(1, int(round(fps / rate_hz)))
    lo, hi = max(label - int(reach_s * fps), 0), min(label + int(reach_s * fps), last)
    return sorted({i for i in range(lo, hi + 1) if (i - label) % step == 0} | {label})


def _frames(path, bounds, wanted, viewport):
    for index, image in walk_frames(path, tuple(bounds["content_box"]), wanted):
        if image.shape[1::-1] != tuple(viewport):
            image = cv2.resize(image, tuple(viewport), interpolation=cv2.INTER_AREA)
        yield index, image


def clip_views(clip, grids, plans, cfg, skip_s: float = SKIP_S):
    """Yield `(image, [(grid, d), ...])` per frame of `clip` that some label's map takes, `d` being
    the view's position in tiles relative to that labelled frame.

    Two decodes: odometry first, so a view before its label knows where the label is. Holding those
    frames until the label arrives instead would cost ~800 MB a window.
    """
    path = find_clip(clip)
    bounds = load_bounds(path)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.release()
    last = bounds["usable"][1]
    windows = {g.frame: set(window_frames(g.frame, fps, last)) for g in grids}
    wanted = sorted(set().union(*windows.values()))
    viewport = next(iter(plans.values())).viewport

    odos, track = {}, defaultdict(dict)
    for index, image in _frames(path, bounds, wanted, viewport):
        for g in grids:
            if index in windows[g.frame]:
                plan = plans[g.hud]
                odo = odos.setdefault(g.frame, Odometry(plan, cfg))
                track[g.frame][index] = odo.update(plan.rectify(image))

    for index, image in _frames(path, bounds, wanted, viewport):
        takers = []
        for g in grids:
            r = track[g.frame].get(index)
            ref = track[g.frame].get(g.frame)
            if r is None or ref is None or abs(index - g.frame) / fps < skip_s:
                continue
            if r.segment != ref.segment or r.status != "ok":
                continue
            takers.append((g, np.subtract(r.position_tiles, ref.position_tiles)))
        if takers:
            yield image, takers


def score(grids, models, plans, cfg, occluders, log=print):
    """`{model name: {clip: MapScore}}`. `models` is `{name: (TerrainClassifier, clips or None)}`,
    a model being scored only on the windows of `clips` when given. `occluders(image)` returns a
    `plan -> (rows, cols) bool` of the cells boxes cover."""
    by_clip = defaultdict(list)
    for g in grids:
        by_clip[g.clip].append(g)
    out = defaultdict(lambda: defaultdict(MapScore))
    t0 = time.time()
    views = 0
    for clip, gs in sorted(by_clip.items()):
        users = {n: m for n, (m, only) in models.items() if only is None or clip in only}
        if not users:
            continue
        maps = {(n, g.frame): OccupancyMap.from_config(cfg) for n in users for g in gs}
        for image, takers in clip_views(clip, gs, plans, cfg):
            covered = occluders(image)
            for g, d in takers:
                plan = plans[g.hud].registered(d)
                rect = plan.rectify(image)
                zone = detect_zone(rect, plan, cfg).at_least(0.05)
                occluded = covered(plan)
                odo = SimpleNamespace(position_tiles=tuple(d), status="ok", segment=0)
                for name, model in users.items():
                    cells, _ = model.predict(rect, plan)
                    maps[(name, g.frame)].update(cells, odo, plan, zone=zone, occluded=occluded,
                                                 cfg=cfg)
                views += 1
        for g in gs:
            scored = observed_cells(plans[g.hud])
            for name in users:
                best = label_crop(maps[(name, g.frame)], plans[g.hud])
                out[name][clip].add(g.as_class_index(), best, scored)
        log(f"  {clip}: {len(gs)} labels, {views} deposits so far, {time.time() - t0:.0f}s")
    return out


# ---------------------------------------------------------------------------
# the script
# ---------------------------------------------------------------------------

def _line(s: MapScore) -> str:
    b = s.blocking
    return (f"F1 {b.f1:.3f}  R {b.recall:.3f}  P {b.precision:.3f}  (n={b.tp + b.fn:5d})   "
            f"edge {s.edge.f1:.3f}  inner {s.inner.f1:.3f}   5-class {s.accuracy:.3f}")


def _report(results, grids, plans_used, pools):
    """`pools` is `{title: [model names]}`, each group of single-recording models summed into one
    row: the before and after of a labelling round."""
    emulator = {g.clip for g in grids if g.hud == "emulator"}
    print(f"\nblocking = WALL|WATER|FENCE; HUD masks: {plans_used}")
    for name, by_clip in results.items():
        clips = sorted(by_clip)
        print(f"\n== {name}")
        total = sum((by_clip[c] for c in clips), MapScore())
        print(f"   {'all windows':34s} ({total.windows:2d})  {_line(total)}")
        emu = [c for c in clips if c in emulator]
        if emu and len(emu) < len(clips):
            s = sum((by_clip[c] for c in emu), MapScore())
            print(f"   {'emulator windows':34s} ({s.windows:2d})  {_line(s)}")
        if 1 < len(clips) <= 8:
            for c in clips:
                print(f"   {c[:34]:34s} ({by_clip[c].windows:2d})  {_line(by_clip[c])}")
    for title, names in pools.items():
        if len(names) > 1:
            pooled = sum((s for n in names for s in results[n].values()), MapScore())
            print(f"\n== {title}, each recording by the model that never saw it")
            print(f"   {'pooled':34s} ({pooled.windows:2d})  {_line(pooled)}")


def _occluders(cfg):
    from brawl_deployment.perception.loot import box_occlusion, crate_occlusion
    from brawl_vision.object_detection.detector import ObjectDetector
    from brawl_vision.object_detection.projectile_detection.detect import ProjectileDetector
    entities, projectiles = ObjectDetector.from_config(cfg), ProjectileDetector.from_config(cfg)

    def occluders(image):
        crates, brawlers = projectiles.predict(image), entities.predict(image)
        return lambda plan: crate_occlusion(crates, plan) | box_occlusion(brawlers, plan)
    return occluders


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default=str(LABELS), help="directory of label .json files")
    p.add_argument("--model", action="append", default=[],
                   help="checkpoint to score on every window (default: the shipped weights)")
    p.add_argument("--held-out-model", action="append", default=[], metavar="CLIP=PATH",
                   help="checkpoint to score on CLIP's windows only")
    p.add_argument("--hold-out-each", nargs="+", default=[], metavar="CLIP",
                   help="train one model per CLIP without its labels, scored on CLIP only")
    p.add_argument("--save-dir", default=None, help="where --hold-out-each keeps its checkpoints")
    p.add_argument("--clips", nargs="+", default=None, help="score only these recordings")
    p.add_argument("--one-mask", default=None, choices=["phone", "emulator"],
                   help="score every label under this HUD mask instead of its own")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args(argv)

    from brawl_vision.terrain.classifier import TerrainClassifier, train

    cfg = load_vision_config()
    plans = hud_plans()
    if args.one_mask:
        plans = {name: plans[args.one_mask] for name in plans}
    grids = load_label_dir(args.labels, plans["phone"])
    if not grids:
        print(f"no label files in {args.labels}", file=sys.stderr)
        return 1
    known = {g.clip for g in grids}
    for clip in args.hold_out_each + [s.split("=", 1)[0] for s in args.held_out_model]:
        if clip not in known:
            print(f"{clip!r} has no labels in {args.labels}", file=sys.stderr)
            return 1

    models = {}
    if args.model or not (args.hold_out_each or args.held_out_model):
        for path in args.model or [TERRAIN_WEIGHTS_PATH]:
            models[Path(path).name] = (TerrainClassifier.load(path, args.device), None)
    for spec in args.held_out_model:
        clip, path = spec.split("=", 1)
        models[f"{Path(path).name} on {clip}"] = (TerrainClassifier.load(path, args.device), {clip})
    if args.hold_out_each:
        from scripts.vision_train_terrain import examples_for
        examples = examples_for(grids, plans, cfg)
        for clip in args.hold_out_each:
            t0 = time.time()
            model = train([e for e in examples if e.clip != clip], epochs=args.epochs,
                          seed=args.seed, device=args.device, cfg=cfg)
            if args.save_dir:
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                model.save(Path(args.save_dir) / f"held_out_{clip}.pt")
            models[f"held out: {clip}"] = (model, {clip})
            print(f"trained without {clip} in {time.time() - t0:.0f}s", flush=True)

    scored = [g for g in grids if args.clips is None or g.clip in args.clips]
    results = score(scored, models, plans, cfg, _occluders(cfg))
    pools = {"kept checkpoints": [n for n, (_, only) in models.items()
                                  if only and not n.startswith("held out: ")],
             "trained now": [n for n in models if n.startswith("held out: ")]}
    _report(results, scored, "each label's own" if not args.one_mask else f"{args.one_mask} only",
            {title: [n for n in names if n in results] for title, names in pools.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
