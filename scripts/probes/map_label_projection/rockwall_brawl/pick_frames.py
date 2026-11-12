"""Training and held-out frames for Rockwall Brawl's projected labels, split by ground (region R).

R is `in_r` below (MAP_ONBOARDING.md, Rockwall Brawl, says why there). Held-out frames are those
whose view holds the most R tiles, weighted toward cones and bushes, at least GAP frames apart;
training frames are every STEP frames over the match that are at least NEAR frames from a held-out
one. Held-out frames come only from the first pass over R, to R_CLEAN_LAST. Both sets skip frames
whose track and own registration differ by more than 0.06 tile (dA), frames near a tint, and frames
that show the SHOWDOWN! banner (`skip`), and keep to FIRST-LAST, after the gate and before the
clip's end. region_score.py imports `in_r`, so the replay's region and the labels' can't disagree.

    python pick_frames.py          prints both lists and the --leave-row strings for each set
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

SCR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
TEX = np.array([l.rstrip("\n").split(",") for l in open(SCR / "textures.csv") if l.strip("\n")])

FIRST, LAST = 470, 3870          # after the gate (~15.2 s, frame ~458) to the clip's end (3885)
STEP, GAP, NEAR, N_HELD = 40, 30, 12, 12
# R is seen cleanly only in the first pass, to frame 1320 (`in_r`). Frame 2760 sees as many R tiles,
# but under the gas, and labelled nothing (2026-11-11), so held-out frames come from that pass alone.
R_CLEAN_LAST = 1340
W_TEX = {"cone": 3, "cube": 1}   # per R tile in view: cones are 18 tiles on the whole map
W_BUSH = 0.5                     # bushes are a class with no texture, and the palette's other risk
# Four short full-screen magenta tints (frames 747-753, 1617-1638, 1680-1686, 3789-3798), cause not
# known: neither set takes a frame within 10 of one.
TINTS = ((747, 753), (1617, 1638), (1680, 1686), (3789, 3798))
# The "SHOWDOWN!" banner (two brawlers left) across the upper middle of the view, frames 3319-3469:
# its letters cover ground the labels would name. Neither set takes a frame that shows it, as
# Truth's pickers skipped Truth's banner (../truth/pick_heldout.py).
BANNER = (3319, 3469)


def skip(f):
    return any(a - 10 <= f <= b + 10 for a, b in TINTS) or BANNER[0] <= f <= BANNER[1]


def in_r(x, y):
    """Region R, the north strip: map rows 0-9, the whole width.

    The clip sees it cleanly once, 29-44 s (frames 880-1320), before the gas reaches it; when the
    hero comes back, at 88-97 s, the gas covers it. It holds the top bush border with the clumps
    at (26-28, 7), (32-34, 7) and (51-53, 9), both north spawn pockets' cubes, and the three top
    cones (29-31, 6). No held-out label covers those cones: until they break, by frame 1110, they
    sit under the brawlers-left text and kill card or past the view's edge in every held-out
    frame. Cones therefore train (21 cells on 7 tiles) but are not scored here (the held-out
    README, The cones).
    """
    return y <= 9


def leave_rows(keep_r: bool):
    """--leave-row specs that leave R (keep_r False: training) or everything but R (held-out)."""
    out = []
    for y in range(60):
        cols = [x for x in range(60) if in_r(x, y) != keep_r]
        runs, start = [], None
        for x in range(61):
            on = x < 60 and x in cols
            if on and start is None:
                start = x
            if not on and start is not None:
                runs.append((start, x - 1))
                start = None
        out += [f"{y}:{a}-{b}" for a, b in runs]
    return out


def main():
    import sys
    sys.path.insert(0, str(REPO))
    from brawl_deployment.perception.known_map import MAPS_DIR, read_chars      # noqa: E402
    from clip_paths import COVERAGE                                             # noqa: E402
    chars = read_chars(MAPS_DIR / "rockwall_brawl.csv")
    cov = {int(k): v for k, v in json.load(open(COVERAGE)).items()}
    ok = [f for f in sorted(cov) if FIRST <= f <= LAST and cov[f]["dA"] <= 0.06 and not skip(f)]
    scored = []
    for f in ok:
        t = [(x, y) for x, y in cov[f]["tiles"] if in_r(x, y)]
        w = sum(W_TEX.get(TEX[y, x], 0) + (W_BUSH if chars[y, x] == "b" else 0) for x, y in t)
        scored.append((len(t) + 4 * w, f, len(t)))
    scored.sort(key=lambda r: -r[0])
    held = []
    for s, f, n in scored:
        if n >= 120 and f <= R_CLEAN_LAST and all(abs(f - g) >= GAP for g in held):
            held.append(f)
        if len(held) == N_HELD:
            break
    held.sort()
    train = []
    for f in ok:
        if all(abs(f - g) >= NEAR for g in held) and (not train or f - train[-1] >= STEP):
            train.append(f)

    for name, frames, keep in (("held-out", held, True), ("training", train, False)):
        cells, tiles = Counter(), {}
        for f in frames:
            for x, y in cov[f]["tiles"]:
                if in_r(x, y) == keep:
                    key = TEX[y, x] or {"#": "wall", "b": "bush", "S": "floor"}.get(chars[y, x], "floor")
                    cells[key] += 1
                    tiles.setdefault(key, set()).add((x, y))
        print(f"{name}: {len(frames)} frames: {' '.join(map(str, frames))}")
        print("   tile-views / distinct tiles in its ground: "
              + ", ".join(f"{k} {cells[k]}/{len(tiles[k])}" for k in sorted(cells)))
    print("\nheld-out --leave-row:", " ".join(leave_rows(True)))
    print("\ntraining --leave-row:", " ".join(leave_rows(False)))


if __name__ == "__main__":
    main()
