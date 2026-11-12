"""Every odometry window's own phase-correlation shift, frame pair by frame pair, for one clip.

    python scripts/probes/odometry_overlay/dump_shifts.py CLIP --hud emulator --rate 12 --out F.npz

`--rate` picks frames the way the deployed loop sees them: clip frame round(k * fps / rate). Leave
it out for every frame. Writes `idx` (the later frame of each pair), `prev` (the earlier one),
`shift` (pairs x windows x 2, image px, as `cv2.phaseCorrelate` returns it), `resp` (pairs x
windows), `spots` and `ppt`. Nothing is filtered: `vote.py` applies the response floor itself, so
one dump serves any estimator scored against it.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from brawl_vision.clips import load_bounds  # noqa: E402
from brawl_vision.config import load_vision_config  # noqa: E402
from brawl_vision.gameplay import walk_frames  # noqa: E402
from brawl_vision.sources import open_source  # noqa: E402
from brawl_vision.terrain.labeling import find_clip, hud_plans  # noqa: E402
from brawl_vision.terrain.odometry import Odometry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--hud", choices=("emulator", "phone"), required=True)
    ap.add_argument("--rate", type=float, default=None, help="decision-loop rate in Hz; default every frame")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plan = hud_plans()[args.hud]
    odo = Odometry(plan, load_vision_config())
    win, spots = odo._win, odo.spots
    path = find_clip(args.clip)
    cap = cv2.VideoCapture(str(path))
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if Path(str(path) + ".bounds.json").exists():
        bounds = load_bounds(path)
        lo, hi = bounds["usable"]
        source = lambda frames: walk_frames(path, tuple(bounds["content_box"]), frames)  # noqa: E731
    else:   # the phone fixtures carry no bounds; read them the way tests/test_vision_odometry.py does
        lo, hi = 0, count - 1

        def source(frames):
            want = set(frames)
            with open_source(path) as src:
                for frame in src:
                    if frame.index in want:
                        yield frame.index, frame.image
    if args.rate:
        frames = sorted({int(round(k * fps / args.rate)) for k in range(int((hi - lo) * args.rate / fps) + 1)})
        frames = [lo + f for f in frames if lo + f <= hi]
    else:
        frames = list(range(lo, hi + 1))

    prev_g, prev_i = None, None
    idx, prv, shift, resp = [], [], [], []
    for i, im in source(frames):
        if im.shape[1::-1] != tuple(plan.viewport):
            im = cv2.resize(im, plan.viewport, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(plan.rectify(im), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_g is not None:
            row_s, row_r = [], []
            for x, y in spots:
                (dx, dy), r = cv2.phaseCorrelate(np.ascontiguousarray(prev_g[y:y + win, x:x + win]),
                                                 np.ascontiguousarray(g[y:y + win, x:x + win]), odo._hann)
                row_s.append((dx, dy))
                row_r.append(r)
            idx.append(i); prv.append(prev_i); shift.append(row_s); resp.append(row_r)
        prev_g, prev_i = g, i
    np.savez_compressed(args.out, idx=np.array(idx), prev=np.array(prv), shift=np.array(shift, np.float32),
                        resp=np.array(resp, np.float32), spots=np.array(spots), ppt=plan.pixels_per_tile,
                        fps=fps, rate=args.rate or 0.0)
    print(f"{args.clip}: {len(idx)} pairs x {len(spots)} windows -> {args.out}")


if __name__ == "__main__":
    main()
