"""The localizer's offset against image registration, one column per replay CSV:
    python loc_vs_image.py CSV [CSV ...]

loc_off - img_off, with img_off = A + hero_offset/S - 30 - hero_odo, is one constant per odometry
segment while the fix is right (the hero's anchor in the view), and the same constant in every
replay of the clip, so it prints absolute and two runs compare line by line. A change in it is the
fix moving off (or onto) the truth.

hero_odo and hero_offset are interpolated to the registered frame's time between the decision rows
either side of it (both within 0.3 s, same offset). Until 2026-11-05 this took the nearest row, up
to 1/12 s away: on a 2.7 tiles/s walk that is ~0.3 tile, and it put match 2's walk drift at 0.6
tile where it is 0.3 (MAP_ONBOARDING.md, Truth).
"""
import csv, json, math, sys
from pathlib import Path
from clip_paths import CLIP, COVERAGE, FIRST_CLIP, scale
S = scale()
cov = {int(k): v for k, v in json.load(open(COVERAGE)).items()}
SEGS = [(476, 2383)] if CLIP == FIRST_CLIP else [(min(cov), max(cov))]


def g(r, k):
    try: return float(r[k])
    except ValueError: return math.nan


def decision_rows(path):
    rows = [r for r in csv.DictReader(open(path))
            if not math.isnan(g(r, "hero_odo_x")) and r["map_state"] in ("fixed", "confirm")]
    return sorted(rows, key=lambda r: g(r, "t"))


def loc_minus_img(rows, f):
    t = f / 30
    a = next((r for r in reversed(rows) if g(r, "t") <= t), None)
    b = next((r for r in rows if g(r, "t") > t), None)
    if a is None or b is None or t - g(a, "t") > 0.3 or g(b, "t") - t > 0.3:
        return None
    if (a["map_dx"], a["map_dy"]) != (b["map_dx"], b["map_dy"]):
        return None
    w = (t - g(a, "t")) / (g(b, "t") - g(a, "t"))
    lerp = lambda k: g(a, k) + w * (g(b, k) - g(a, k))  # noqa: E731
    A = cov[f]["A"]
    ix = A[0] + lerp("hero_offset_x") / S[0] - 30 - lerp("hero_odo_x")
    iy = A[1] + lerp("hero_offset_y") / S[1] - 30 - lerp("hero_odo_y")
    return g(a, "map_dx") - ix, g(a, "map_dy") - iy


runs = [decision_rows(p) for p in sys.argv[1:]]
print("frame      t  " + "  ".join(f"{Path(p).stem[:15]:>15s}" for p in sys.argv[1:]))
for lo, hi in SEGS:
    for f in sorted(cov):
        if lo <= f <= hi:
            vals = [loc_minus_img(rows, f) for rows in runs]
            if any(v is not None for v in vals):
                print(f"f{f:5d} {f / 30:6.1f}  " + "  ".join(
                    f"{'-':>15s}" if v is None else f"({v[0]:+6.2f},{v[1]:+6.2f})" for v in vals))
    print()
