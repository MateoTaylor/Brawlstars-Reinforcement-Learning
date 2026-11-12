"""Terrain-classifier labels for one recording, projected from its map's label (MAP_ONBOARDING.md, M3b).

    python scripts/map_project_labels.py MAP CLIP              # register and fit only, to pick frames
    python scripts/map_project_labels.py MAP CLIP --train F [F ...] --held-out F [F ...]
        [--leave C,R ...] [--leave-row R:C0-C1,C2-C3 ...] [--leave-why TEXT] [--broken C,R@F ...]

MAP names `brawl_deployment/data/maps/<MAP>.{png,csv,json}`: the top-down image, the hand-checked
label, and the lattice `map_label.py` fitted to the image. CLIP is a `bluestacks-` recording in a
`CLIP_DIRS` folder (`brawl_vision/data/training_videos/` is gitignored). The method, the label rule
and the refusals are in `brawl_vision/terrain/label_projection.py`'s docstring. Run it on the CPU:
`CUDA_VISIBLE_DEVICES=-1`, and `--device cpu` (the default) for the detectors.

Stages, each cached in --work (default `runs/map_checks/<MAP>/projection/<CLIP>/`, gitignored):

| stage | writes | Starfruit, 2221 frames at 1440p |
|---|---|---|
| 1. odometry on every frame; scale-1 NCC of every --step-th against the map image | `reg.npz` | ~80 s |
| 2. the scale fit, then NCC at that scale of every sample in the matched odometry segments | `reg_at_scale.npz` | ~1 min |
| 3. one `LabelGrid` per frame, and review overlays | the JSONs; `review/` | ~3 s a frame |

Stages 1 and 2 rerun only when their inputs change (clip, map image and lattice, --step, the fitted
scale) or with --redo, so a second set of frames costs stage 3 alone.

**Frames.** --train labels go to --train-dir, the classifier's training folder. --held-out labels go
to --held-out-dir, a folder of their own: `--hold-out` separates by clip name, and both sets come
from one clip. Keep the held-out frames on ground the training frames never saw. The last lines
printed count the map tiles each set labels and the tiles both share, which should be none. A new
held-out folder goes in `HELD_OUT_LABELS` in tests/test_vision_classifier.py, which checks every
label against its HUD mask.

Each frame needs the frames `NEIGHBOUR` (8) either side in its own odometry segment, for the
transient check, and that segment has to be one stage 2 registered. Run with no frames first, and
it prints the frames that qualify. A full-screen flash (a super) fails most of a view on the
transient check, so move such a frame rather than loosen the check.

**Scale.** Stage 2's S (rectified cells per map tile) comes from regressing the scale-1 NCC positions
on odometry, so it is only as good as the odometry. On Truth (2026-11-05) the "SHOWDOWN!" banner
froze odometry for 3 s while the camera moved about 6 tiles, and the fit read (0.939, 0.935). The
`block stretch` line after stage 2 measures the scale from the image instead, out of the 3 x 3
block residuals (`label_projection.block_stretch`). It reads near 0 when S is right. When the frame
edge comes out more than DELTA off, it prints the S the blocks point to. `--scale SX SY` then sets
S for stage 2 and the labels; rerun until the line reads near 0. Truth used `--scale 1.022 1.004`,
the Starfruit fit: same emulator, same camera.

**Cells to leave.** --leave and --leave-row name map cells that never label, whatever the map label
says: Starfruit's meeple figures, `#` in its label but wall or fence unsettled. --leave-why says why,
in every label's notes.

**Broken cells.** The map label is the ground as the match starts, so a wall or bush that a super or
a shot destroys would label its floor `#` or `b` in every frame after. `--broken C,R@F` names a map
cell broken by frame F. It labels from the map before F and is left `?` from F on, and each label's
notes list the cells it left so. Only labelled frames count, so F can be any frame after the last
labelled view of the cell standing, up to the first labelled view of it broken. The first run
labels from the map as it stands, its review sheets show what fell, and a second run names those
cells. Rockwall Brawl (2026-11-11) lost 8 destructible cones, 4 rock cubes and 4 bushes in its
clip: scripts/probes/map_label_projection/rockwall_brawl/ says how they were found.

**Review.** `review/<set>/review_f<frame>.jpg` tints each labelled cell with its class and letter,
dims `?`, and draws a diagonal on refused cells: white under a detection or a transient, yellow
under gas. `review/sheet_<set>.jpg` puts a set on one page. A wrong cell is fixed in the map label
and regenerated, never edited in a frame: every frame that sees that ground would stay wrong.

Starfruit Supernova's committed labels come out byte-identical. This writes them to a scratch
folder; scripts/probes/map_label_projection/README.md compares them with the committed ones:

    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/map_project_labels.py starfruit_supernova \\
        bluestacks-starfruit-supernova-west-upper \\
        --train 460 550 640 690 820 910 1000 1090 1180 1340 \\
        --held-out 1840 1900 1960 2020 2080 2140 2200 \\
        --train-dir runs/map_checks/starfruit_supernova/repro/labels \\
        --held-out-dir runs/map_checks/starfruit_supernova/repro/labels_starfruit_test \\
        --leave-row 26:0-3,8-14,18-21,25-31,35-38,43-44,50-52 33:3-7,13-18,22-24,28-35,39-45,51-54,58-59 \\
        --leave-why "meeple cells (map rows 26/33, wall-or-fence unsettled)" --date 2026-11-04
"""
import argparse
import dataclasses
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from brawl_deployment.perception.known_map import MAPS_DIR, read_chars          # noqa: E402
from brawl_vision.clips import load_bounds                                        # noqa: E402
from brawl_vision.config import load_vision_config                                # noqa: E402
from brawl_vision.gameplay import walk_frames                                     # noqa: E402
from brawl_vision.terrain import label_projection as lp                           # noqa: E402
from brawl_vision.terrain.labeling import (LabelGrid, default_hud, find_clip,     # noqa: E402
                                           hud_plans, read_label_frames)
from brawl_vision.terrain.odometry import Odometry                                # noqa: E402
from brawl_vision.terrain.zone import detect_zone                                 # noqa: E402

LABELS = REPO / "tests" / "fixtures" / "vision"
HUD = "emulator"     # TEXT_BOX and KILL_BOX are measured on the emulator's HUD


# ---------------------------------------------------------------------------
# stages 1 and 2: registration, cached
# ---------------------------------------------------------------------------

def _load_cached(path, meta, redo):
    """The cached arrays at `path` if they were made from `meta`, else None."""
    if redo or not path.exists():
        return None
    d = np.load(path)
    if json.loads(str(d["meta"])) != meta:
        print(f"{path.name}: made from other inputs, redoing it")
        return None
    return {k: d[k] for k in d.files if k != "meta"}


def _viewport(image, plan):
    if image.shape[1::-1] != tuple(plan.viewport):
        image = cv2.resize(image, plan.viewport, interpolation=cv2.INTER_AREA)
    return image


def register_clip(path, bounds, plan, cfg, matcher, step):
    """Stage 1: odometry on every usable frame, scale-1 NCC on every step-th."""
    first, last = bounds["usable"]
    n = last + 1
    pos = np.full((n, 2), np.nan)
    status = np.full(n, "", "<U9")
    seg = np.full(n, -1)
    inl = np.zeros(n)
    idx, P, score, coarse, runner_up = [], [], [], [], []
    odo = Odometry(plan, cfg)
    t0 = time.time()
    for i, im in walk_frames(path, tuple(bounds["content_box"]), range(first, n)):
        rect = plan.rectify(_viewport(im, plan))
        o = odo.update(rect)
        pos[i], status[i], seg[i], inl[i] = o.position_tiles, o.status, o.segment, o.inlier_ratio
        if i % step == 0:
            p, s, sc, s2 = matcher.coarse(rect)
            for column, value in ((idx, i), (P, p), (score, s), (coarse, sc), (runner_up, s2)):
                column.append(value)
        if i % 300 == 0:
            print(f"  {i}/{n}  {time.time() - t0:.0f} s  odometry {o.status}, segment {o.segment}",
                  flush=True)
    return dict(pos=pos, status=status, seg=seg, inl=inl, idx=np.array(idx), P=np.array(P),
                score=np.array(score), coarse=np.array(coarse), runner_up=np.array(runner_up))


def register_at_scale(path, bounds, plan, matcher, reg, idx, S):
    """Stage 2: every sample in `idx` again at scale S, near its scale-1 position."""
    where = {int(i): k for k, i in enumerate(reg["idx"])}
    A, score, blocks = [], [], []
    for i, im in walk_frames(path, tuple(bounds["content_box"]), idx):
        rect = plan.rectify(_viewport(im, plan))
        a, s, b = matcher.at_scale(rect, S, reg["P"][where[i]])
        for column, value in ((A, a), (score, s), (blocks, b)):
            column.append(value)
    if len(A) != len(idx):
        raise RuntimeError(f"the clip ended after {len(A)} of {len(idx)} samples")
    return dict(idx=np.asarray(idx), A=np.array(A), score=np.array(score), blocks=np.array(blocks))


def registration(args, plan, cfg, path, bounds, grid_shape):
    """Stages 1 and 2, from the cache where it is current. Returns (Track, ScaleFit, reg)."""
    image_path, lattice_path = MAPS_DIR / f"{args.map}.png", MAPS_DIR / f"{args.map}.json"
    lattice = json.loads(lattice_path.read_text())
    key = hashlib.sha1(image_path.read_bytes()
                       + json.dumps([lattice["pitch_px"], lattice["origin_px"]]).encode())
    meta = {"clip": args.clip, "map": args.map, "map_key": key.hexdigest()[:16],
            "step": args.step, "usable": list(bounds["usable"])}
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    matcher = None

    def get_matcher():
        nonlocal matcher
        if matcher is None:
            matcher = lp.MapMatcher(image, lattice["pitch_px"], lattice["origin_px"],
                                    grid_shape[::-1], plan)
        return matcher

    reg = _load_cached(args.work / "reg.npz", meta, args.redo)
    if reg is None:
        print(f"stage 1: odometry and scale-1 NCC over frames {bounds['usable'][0]}-"
              f"{bounds['usable'][1]}")
        t0 = time.time()
        reg = register_clip(path, bounds, plan, cfg, get_matcher(), args.step)
        np.savez(args.work / "reg.npz", meta=json.dumps(meta), **reg)
        print(f"stage 1: {time.time() - t0:.0f} s")
    fit = lp.fit_scale(reg["pos"], reg["seg"], reg["idx"], reg["P"], reg["score"])
    print("scale fit:\n" + fit.describe())
    if args.scale:
        fit = dataclasses.replace(fit, S=tuple(round(s, lp.SCALE_DECIMALS) for s in args.scale))
        print(f"--scale: S = ({fit.S[0]:.3f}, {fit.S[1]:.3f}) instead")
    segments = lp.segments_to_register(reg["seg"], reg["idx"], reg["score"])
    idx = [int(i) for i in reg["idx"] if reg["seg"][i] in segments]
    meta2 = meta | {"S": list(fit.S), "idx": idx}
    reg2 = _load_cached(args.work / "reg_at_scale.npz", meta2, args.redo)
    if reg2 is None:
        print(f"stage 2: {len(idx)} samples at S = ({fit.S[0]:.3f}, {fit.S[1]:.3f})")
        t0 = time.time()
        reg2 = register_at_scale(path, bounds, plan, get_matcher(), reg, idx, fit.S)
        np.savez(args.work / "reg_at_scale.npz", meta=json.dumps(meta2), **reg2)
        print(f"stage 2: {time.time() - t0:.0f} s")
    track = lp.Track(reg["pos"], reg["seg"], reg2["idx"], reg2["A"], fit.S)
    b = reg2["blocks"]
    print(f"registered: odometry segment(s) "
          + ", ".join(f"{s} (frames {track.span(s)[0]}-{track.span(s)[1]})" for s in segments)
          + f"; NCC at S median {np.median(reg2['score']):.3f}, min {reg2['score'].min():.3f}; "
          f"block residual median |dx| {np.nanmedian(np.abs(b[..., 0])):.2f}, "
          f"|dy| {np.nanmedian(np.abs(b[..., 1])):.2f} tile")
    kx, ky = lp.block_stretch(b, fit.S, plan.size_tiles)
    edge = max(abs(kx) * plan.size_tiles[0], abs(ky) * plan.size_tiles[1]) / 2
    print(f"block stretch x {kx:+.4f}, y {ky:+.4f}: the frame edge {edge:.2f} tile off"
          + (f"  <- over DELTA ({lp.DELTA}): S looks wrong; the blocks point to "
             f"--scale {fit.S[0] * (1 - kx):.3f} {fit.S[1] * (1 - ky):.3f}" if edge > lp.DELTA else ""))
    return track, fit, reg


# ---------------------------------------------------------------------------
# stage 3: the labels
# ---------------------------------------------------------------------------

COLOURS = {".": (200, 200, 200), "#": (40, 40, 255), "b": (40, 230, 40), "~": (255, 170, 0),
           "f": (0, 230, 255)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_review(rect, labels, ppt):
    """Labelled cells tinted with their class and letter, `?` dimmed, refused cells crossed."""
    out = rect.copy()
    dim = (rect * 0.35).astype(np.uint8)
    rows, cols = labels.chars.shape
    for j in range(rows):
        for i in range(cols):
            c = labels.chars[j, i]
            x0, y0 = i * ppt, j * ppt
            if c == "?":
                out[y0:y0 + ppt, x0:x0 + ppt] = dim[y0:y0 + ppt, x0:x0 + ppt]
                continue
            colour = COLOURS[c]
            tile = out[y0:y0 + ppt, x0:x0 + ppt]
            if c != ".":
                tile[:] = (0.72 * tile + 0.28 * np.array(colour)).astype(np.uint8)
            cv2.rectangle(out, (x0 + 1, y0 + 1), (x0 + ppt - 2, y0 + ppt - 2), colour, 1)
            if c != ".":
                cv2.putText(out, c, (x0 + 15, y0 + 32), FONT, 0.9, (0, 0, 0), 4)
                cv2.putText(out, c, (x0 + 15, y0 + 32), FONT, 0.9, colour, 2)
    r = labels.refused
    for mask, colour in ((r["occluded"] | r["transient"], (255, 255, 255)),
                         (r["gas"], (0, 255, 255))):
        for j, i in zip(*np.nonzero(mask)):
            cv2.line(out, (i * ppt + 4, j * ppt + 4), (i * ppt + ppt - 4, j * ppt + ppt - 4), colour, 1)
    return out


def contact_sheet(named, columns=3, scale=0.5):
    """`named` (caption, image) pairs on one page, `columns` across, captioned bottom-left."""
    small = [(name, cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
             for name, im in named]
    h, w = small[0][1].shape[:2]
    sheet = np.zeros((-(-len(small) // columns) * h, columns * w, 3), np.uint8)
    for k, (name, im) in enumerate(small):
        r, c = divmod(k, columns)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
        for colour, thick in (((0, 0, 0), 5), ((255, 255, 255), 2)):
            cv2.putText(sheet, name, (c * w + 8, (r + 1) * h - 10), FONT, 0.8, colour, thick)
    return sheet


def detectors(cfg, device):
    from brawl_vision.object_detection.detector import ObjectDetector
    from brawl_vision.object_detection.projectile_detection.detect import ProjectileDetector
    return (ObjectDetector.from_config(cfg, device=device),
            ProjectileDetector.from_config(cfg, device=device))


def detect_boxes(dets, image):
    """(kind, Detection) for every box either detector finds: entity boxes are all "brawler",
    projectile-detector boxes go by label (a `RECT_PAD` key, else "other")."""
    ents, projs = dets
    boxes = [("brawler", d) for d in ents.predict(image)]
    boxes += [(d.label if d.label in lp.RECT_PAD else "other", d) for d in projs.predict(image)]
    return boxes


def label_set(name, frames, outdir, ctx):
    """Stage 3 for one set: write each frame's LabelGrid and review overlay. Returns
    {frame: (A, labels)}."""
    plan, args = ctx["plan"], ctx["args"]
    ppt = plan.pixels_per_tile
    cols, rows = plan.size_tiles
    n = lp.NEIGHBOUR
    images = read_label_frames(args.clip, {x for f in frames for x in (f - n, f, f + n)},
                               plan.viewport)
    outdir.mkdir(parents=True, exist_ok=True)
    review = args.work / "review" / name
    review.mkdir(parents=True, exist_ok=True)
    print(f"\n{name}: {len(frames)} frames -> {outdir}")
    print(f"{'frame':>6} {'A':>17} {'proj':>5} {'unobs':>5} {'gas':>4} {'occ':>4} {'text':>4} "
          f"{'trans':>5} {'kept':>5}  boxes")
    out, sheet = {}, []
    for f in sorted(frames):
        A = ctx["track"](f)
        ground = lp.grid_at(ctx["grid"], ctx["broken"], f)
        gone = [cell for cell, at in ctx["broken"] if at <= f]
        rect = plan.rectify(images[f])
        gassed = detect_zone(rect, plan, ctx["cfg"]).at_least(lp.GAS_FRACTION)
        boxes = detect_boxes(ctx["detectors"], images[f])
        pos = ctx["pos"]
        moving = lp.transient_cells(rect, [(plan.rectify(images[x]), pos[x] - pos[f])
                                           for x in (f - n, f + n)], plan)
        labels = lp.label_cells(ground, A, ctx["S"], plan, gassed,
                                [(k, d.xyxy) for k, d in boxes], moving)
        grid = LabelGrid(clip=args.clip, frame=int(f),
                         origin_tile=tuple(int(round(v)) for v in plan.origin_tile),
                         size_tiles=(cols, rows), pixels_per_tile=ppt, chars=labels.chars,
                         notes=lp.note(ctx["map_csv"], args.date, ctx["S"], A, args.leave_why, gone),
                         hud=HUD)
        grid.save(outdir / f"{args.clip}_f{f}.json")
        img = draw_review(rect, labels, ppt)
        cv2.imwrite(str(review / f"review_f{f}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        sheet.append((f"review_f{f}", img))
        s = labels.stats()
        kinds = sorted({k for k, _ in boxes})
        print(f"{f:>6} ({A[0]:7.3f},{A[1]:7.3f}) {s['projected']:>5} {s['unobserved']:>5} "
              f"{s['gas']:>4} {s['occluded']:>4} {s['text']:>4} {s['transient']:>5} "
              f"{s['labelled']:>5}  {len(boxes)} {' '.join(kinds)}"
              + ("" if ctx["status"][f] == "ok" else f"  [odometry {ctx['status'][f]}]"))
        out[f] = (A, labels)
    cv2.imwrite(str(args.work / "review" / f"sheet_{name}.jpg"), contact_sheet(sheet),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    totals = {c: sum(int((lab.chars == c).sum()) for _, lab in out.values()) for c in lp.TERRAIN}
    names = {".": "floor", "#": "wall", "b": "bush", "~": "water", "f": "fence"}
    print(f"{name}: {sum(totals.values())} cells: "
          + ", ".join(f"{names[c]} {v}" for c, v in totals.items() if v))
    return out


# ---------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", help="name in brawl_deployment/data/maps/")
    ap.add_argument("clip", help="a bluestacks- recording name, without .mp4")
    ap.add_argument("--train", type=int, nargs="+", default=[], metavar="F",
                    help="frames to label for training")
    ap.add_argument("--held-out", type=int, nargs="+", default=[], metavar="F",
                    help="frames to label for scoring only")
    ap.add_argument("--train-dir", type=Path, default=LABELS / "labels")
    ap.add_argument("--held-out-dir", type=Path,
                    help="default tests/fixtures/vision/labels_<map>_test")
    ap.add_argument("--leave", nargs="+", default=[], metavar="C,R",
                    help="map cells that never label")
    ap.add_argument("--leave-row", nargs="+", default=[], metavar="R:C0-C1,...",
                    help="runs of map cells on one row that never label, columns inclusive")
    ap.add_argument("--leave-why", default="",
                    help="why those cells never label, for the notes")
    ap.add_argument("--broken", nargs="+", default=[], metavar="C,R@F",
                    help="map cells broken by frame F (see Broken cells): they label before F, "
                         "never from F on")
    ap.add_argument("--step", type=int, default=10, help="NCC every step-th frame (default 10)")
    ap.add_argument("--scale", type=float, nargs=2, metavar=("SX", "SY"),
                    help="S for stage 2 and the labels, instead of the odometry fit (see Scale)")
    ap.add_argument("--work", type=Path, help="default runs/map_checks/<map>/projection/<clip>")
    ap.add_argument("--device", default="cpu", choices=("cpu", "auto", "cuda"),
                    help="for the detectors (default cpu)")
    ap.add_argument("--date", default=datetime.date.today().isoformat(),
                    help="the date the notes give (default today)")
    ap.add_argument("--redo", action="store_true", help="ignore the cached registration")
    args = ap.parse_args(argv)
    args.held_out_dir = args.held_out_dir or LABELS / f"labels_{args.map}_test"
    args.work = args.work or REPO / "runs" / "map_checks" / args.map / "projection" / args.clip
    return args


def check_frames(args, track, usable):
    """Refuse a frame list the labels can't be made from, before the slow part."""
    both = set(args.train) & set(args.held_out)
    if both:
        raise SystemExit(f"frames {sorted(both)} are in both --train and --held-out")
    if args.held_out and args.train_dir.resolve() == args.held_out_dir.resolve():
        raise SystemExit("--held-out-dir is --train-dir: training would read the held-out labels")
    n = lp.NEIGHBOUR
    for f in args.train + args.held_out:
        if not (usable[0] <= f - n and f + n <= usable[1]):
            raise SystemExit(f"frame {f}: the transient check needs frames {f - n} and {f + n}, "
                             f"and the clip's usable frames are {usable[0]}-{usable[1]}")
        if not track.covers(f):
            spans = ", ".join(f"{a}-{b}" for a, b in map(track.span, track.segments))
            raise SystemExit(f"frame {f} is outside the registered odometry segments "
                             f"(frames {spans})")
        if not track.seg[f - n] == track.seg[f] == track.seg[f + n]:
            a, b = track.span(int(track.seg[f]))
            raise SystemExit(f"frame {f}: the transient check compares it with frames {f - n} and "
                             f"{f + n}, which have to be in its odometry segment (frames {a}-{b})")


def main(argv=None):
    args = parse_args(argv)
    if default_hud(args.clip) != HUD:
        raise SystemExit(f"{args.clip} is not an emulator recording (a bluestacks- name): the text "
                         f"and kill-card boxes are measured on the emulator HUD")
    leave = lp.parse_leave(args.leave, args.leave_row)
    broken = lp.parse_broken(args.broken)
    chars = read_chars(MAPS_DIR / f"{args.map}.csv")
    grid = lp.class_grid(chars, leave)
    lp.grid_at(grid, broken, 0)          # a broken cell off the map fails here, not after stage 2
    plan = hud_plans()[HUD]
    cfg = load_vision_config()
    path = find_clip(args.clip)
    bounds = load_bounds(path)
    args.work.mkdir(parents=True, exist_ok=True)
    track, fit, reg = registration(args, plan, cfg, path, bounds, grid.shape)
    if not (args.train or args.held_out):
        n = lp.NEIGHBOUR
        spans = ", ".join(f"{a + n}-{b - n}" for a, b in map(track.span, track.segments))
        print(f"\nno frames asked for: pick --train and --held-out frames in {spans} (each needs "
              f"the frames {n} either side in its own odometry segment)")
        return
    check_frames(args, track, bounds["usable"])
    if leave:
        print(f"leaving {len(leave)} map cells '?': {args.leave_why or '(no --leave-why)'}")
    if broken:
        print(f"{len(broken)} broken cells, left '?' from their frame on (map class in brackets): "
              + " ".join(f"{c},{r}[{chars[r, c]}]@{at}" for (c, r), at in broken))
    ctx = dict(args=args, plan=plan, cfg=cfg, grid=grid, broken=broken, track=track, S=fit.S,
               pos=reg["pos"], status=reg["status"], detectors=detectors(cfg, args.device),
               map_csv=(MAPS_DIR / f"{args.map}.csv").relative_to(REPO).as_posix())
    sets = {}
    for name, frames, outdir in (("train", args.train, args.train_dir),
                                 ("held_out", args.held_out, args.held_out_dir)):
        if frames:
            sets[name] = label_set(name, frames, outdir, ctx)
    tiles = {name: set().union(*(lp.labelled_tiles(lab.chars, A, fit.S) for A, lab in out.values()))
             for name, out in sets.items()}
    print("\nmap tiles labelled: " + ", ".join(f"{k} {len(v)}" for k, v in tiles.items()))
    if len(tiles) == 2:
        shared = tiles["train"] & tiles["held_out"]
        print(f"shared by both sets: {len(shared)}"
              + (f"  <- the held-out set is not unseen ground: {sorted(shared)[:12]}"
                 + (" ..." if len(shared) > 12 else "") if shared else ""))
    print(f"review sheets: {args.work / 'review'}")


if __name__ == "__main__":
    main()
