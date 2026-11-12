"""The Starfruit Supernova map label as a class grid, plus a registration-check overlay.

class_grid() reads the map CSV (spawns count as floor) and gives the meeple cells, '#' in the CSV
but wall-or-fence unsettled, a class of their own, so no classifier label is ever drawn from them.
footprints() draws the map's non-floor tiles where a registration puts their GROUND footprint on a
rectified frame: the quickest way to judge a registration by eye (a wall's top face shows ~0.9
tile north of its footprint, in the floor cell above it).
"""
import cv2
import numpy as np

from common import MAP, MAP_DIR  # first: puts the repo on sys.path
from brawl_deployment.perception.known_map import read_chars  # noqa: E402

NAMES = [".", "#", "b", "~", "f", "M", "X"]          # floor wall bush water fence meeple off-map
IDX = {c: k for k, c in enumerate(NAMES)}

# The 66 meeple-sprite cells, as {map row: inclusive column runs}.
MEEPLE_RUNS = {
    26: [(0, 3), (8, 14), (18, 21), (25, 31), (35, 38), (43, 44), (50, 52)],
    33: [(3, 7), (13, 18), (22, 24), (28, 35), (39, 45), (51, 54), (58, 59)],
}
MEEPLES = [(r, c) for r, runs in MEEPLE_RUNS.items() for a, b in runs for c in range(a, b + 1)]


def class_grid(meeples=MEEPLES):
    """(60, 60) int8 grid of NAMES indices."""
    chars = read_chars(MAP_DIR / f"{MAP}.csv")
    g = np.vectorize(lambda c: IDX["."] if c == "S" else IDX[c])(chars).astype(np.int8)
    for r, c in meeples:
        g[r, c] = IDX["M"]
    return g


FILL = {1: (40, 40, 255), 2: (40, 255, 40), 3: (255, 160, 0), 4: (0, 255, 255), 5: (255, 0, 255)}


def footprints(rect, g, A, s, ppt=48, alpha=0.35):
    """Map-label tiles drawn where the registration (A, s) puts their ground footprint."""
    out = rect.copy()
    fill = rect.copy()
    h, w = rect.shape[:2]
    for ty in range(max(0, int(np.floor(A[1]))), min(60, int(A[1] + h / (ppt * s[1])) + 2)):
        for tx in range(max(0, int(np.floor(A[0]))), min(60, int(A[0] + w / (ppt * s[0])) + 2)):
            k = int(g[ty, tx])
            if k not in FILL:
                continue
            x0 = int(round((tx - A[0]) * s[0] * ppt))
            x1 = int(round((tx + 1 - A[0]) * s[0] * ppt))
            y0 = int(round((ty - A[1]) * s[1] * ppt))
            y1 = int(round((ty + 1 - A[1]) * s[1] * ppt))
            cv2.rectangle(fill, (x0, y0), (x1, y1), FILL[k], -1)
            cv2.rectangle(out, (x0, y0), (x1, y1), FILL[k], 2)
    return cv2.addWeighted(fill, alpha, out, 1 - alpha, 0)
