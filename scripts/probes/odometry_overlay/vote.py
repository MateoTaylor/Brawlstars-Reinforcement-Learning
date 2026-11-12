"""The odometry's overlay rule on real frames, through the shipped `Odometry`.

    python scripts/probes/odometry_overlay/vote.py DUMP.npz CLIP --hud emulator

Reads a `dump_shifts.py` dump and hands each frame pair's windows (those over the response floor)
to `Odometry._split`, the shipped trigger. It puts a pair in question in two cases:

  - the median does not sit with a coherent moving group. On Truth the SHOWDOWN banner held 5 of
    the 8 emulator windows still while Bull walked, and the camera stopped;
  - it does, but the still windows alone make the frame `uncertain`. This is the hero and its name
    tag on a brisk 12 Hz walk.

For each pair in question it re-reads both frames and runs them through the shipped `update`
twice: once with the rule on (configs/vision.yaml) and once off (`overlay_share: 1.0`). It prints
the pixel vote, the status and the camera motion each way, then the vote's spread and the status
changes. Nothing here re-implements the rule, so a change to it shows up in this output unedited.
"""
import argparse
import sys
from collections import Counter
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


def in_question(z, odo):
    """(pair index, n still, n windows, |c1| tiles, |median - c1| tiles) for each pair `_split`
    puts in question."""
    cfg, ppt = odo.cfg, odo.plan.pixels_per_tile
    out = []
    for k in range(len(z["idx"])):
        s = z["shift"][k][z["resp"][k] >= cfg.odometry_min_response].astype(np.float64)
        if len(s) < cfg.odometry_min_windows:
            continue
        median = np.median(s, axis=0)
        split = odo._split(s, median)
        if split is not None:
            still, _, c1 = split
            out.append((k, int(still.sum()), len(s), np.linalg.norm(c1) / ppt,
                        np.linalg.norm(median - c1) / ppt))
    return out


def frames(path, plan, want):
    """(index, rectified BGR frame) for each wanted clip frame, in order."""
    if Path(str(path) + ".bounds.json").exists():
        source = walk_frames(path, tuple(load_bounds(path)["content_box"]), sorted(want))
    else:   # the phone fixtures carry no bounds; read them the way tests/test_vision_odometry.py does
        def gen():
            with open_source(path) as src:
                for f in src:
                    if f.index in want:
                        yield f.index, f.image
        source = gen()
    for i, im in source:
        if im.shape[1::-1] != tuple(plan.viewport):
            im = cv2.resize(im, plan.viewport, interpolation=cv2.INTER_AREA)
        yield i, plan.rectify(im)


def pair(odo, a, b):
    odo.reset()
    odo.update(a)
    return odo.update(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("clip")
    ap.add_argument("--hud", choices=("emulator", "phone"), required=True)
    args = ap.parse_args()

    z = np.load(args.dump)
    plan = hud_plans()[args.hud]
    on = Odometry(plan, load_vision_config())
    off = Odometry(plan, load_vision_config(overrides={"odometry": {"overlay_share": 1.0}}))
    share = on.cfg.odometry_overlay_share
    pairs = in_question(z, on)
    print(f"{args.clip}: {len(pairs)} of {len(z['idx'])} pairs in question (overlay_share {share})")

    by_frame = {}
    for row in pairs:
        k = row[0]
        by_frame.setdefault(int(z["idx"][k]), []).append(row)
    want = {int(z["prev"][k]) for k, *_ in pairs} | set(by_frame)
    cache, votes, moves = {}, [], Counter()
    for i, img in frames(find_clip(args.clip), plan, want):
        cache[i] = img
        for k, n_still, n, c1, gap in by_frame.get(i, []):
            a = cache.get(int(z["prev"][k]))
            if a is None:
                print(f"{i:6d}  frame {int(z['prev'][k])} did not decode: skipped")
                continue
            r_on, r_off = pair(on, a, img), pair(off, a, img)
            v = r_on.overlay_vote
            votes.append(np.nan if v is None else v)
            moves[(r_off.status, r_on.status)] += 1
            shown = "  n/a" if v is None else f"{v:5.2f}"
            print(f"{i:6d}  still {n_still}/{n}  |c1| {c1:.3f}  median off c1 {gap:.3f}  vote {shown}  "
                  f"off {r_off.status:9s} ({r_off.delta_tiles[0]:+.3f},{r_off.delta_tiles[1]:+.3f})  "
                  f"on {r_on.status:9s} ({r_on.delta_tiles[0]:+.3f},{r_on.delta_tiles[1]:+.3f})")
        cache = {j: im for j, im in cache.items() if j >= i}   # pairs run forward: older frames are done
    if votes:
        v = np.array(votes)
        known = v[~np.isnan(v)]
        quart = np.percentile(known, [0, 25, 50, 75, 100]).round(2) if len(known) else "-"
        print(f"summary: {len(v)} pairs, {int(np.isnan(v).sum())} too flat to vote, "
              f"{int((known > share).sum())} won by the moving group; vote quartiles {quart}")
        print("status off -> on: " + ", ".join(f"{a} -> {b} x{c}" for (a, b), c in sorted(moves.items())))


if __name__ == "__main__":
    main()
