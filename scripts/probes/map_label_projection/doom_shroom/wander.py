"""How far the fix wanders from the zero, and whether the crate spread follows it.

    .venv/Scripts/python.exe wander.py LOC_VS_IMAGE.txt COLUMN ZERO_X ZERO_Y [REPLAY.csv]

LOC_VS_IMAGE.txt is `loc_vs_image.py`'s output, COLUMN its 0-based model column, and the zero is
`crop_split.py`'s (where the fix is exact at the view's centre): (+13.93, +10.08) for the first
recording, (+13.93, +10.00) for the second. A row's value minus the zero is the fix's error at the
view's centre. Rows more than 1.5 tile off are listed and left out of the statistics: on the second
recording the one such row (91.3 s) interpolates toward a row where the hero has died and its
offset reads (0, 0).

With a replay CSV it also pairs each crate residual (`map_crate_resid`) with the fix's error at the
nearest registered frame (within 0.2 s). A residual that moves against the error, slope about -1,
is the fix's wander, and the spread with the error added back is what is left of it.
"""
import csv
import re
import sys

import numpy as np

path, col, zx, zy = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
rows = []
for line in open(path):
    m = re.match(r"f\s*(\d+)\s+([\d.]+)\s+(.*)", line)
    if not m:
        continue
    toks = re.findall(r"\([^)]*\)|(?<=\s)-(?=\s|$)", " " + m.group(3))
    if len(toks) > col and toks[col].startswith("("):
        x, y = map(float, toks[col].strip("()").split(","))
        rows.append((int(m.group(1)), float(m.group(2)), x - zx, y - zy))
far = [r for r in rows if max(abs(r[2]), abs(r[3])) > 1.5]
lv = np.array([r for r in rows if r not in far])
err = lv[:, 2:]
print(f"{len(rows)} registered frames; left out, more than 1.5 tile off: "
      f"{[(f, t, round(dx, 2), round(dy, 2)) for f, t, dx, dy in far]}")
print(f"error median ({np.median(err[:, 0]):+.2f}, {np.median(err[:, 1]):+.2f}); "
      f"max |x| {np.abs(err[:, 0]).max():.2f}, max |y| {np.abs(err[:, 1]).max():.2f}; "
      f"an axis past 0.25 in {int((np.abs(err) > 0.25).any(1).sum())}, past 0.5 in "
      f"{int((np.abs(err) > 0.5).any(1).sum())}:")
for f, t, dx, dy in lv[(np.abs(err) > 0.5).any(1)]:
    print(f"   f {int(f)} {t:.1f} s ({dx:+.2f}, {dy:+.2f})")

if len(sys.argv) > 5:
    res, at = [], []
    for r in csv.DictReader(open(sys.argv[5])):
        if not r["map_crate_resid"]:
            continue
        t = float(r["t"])
        i = int(np.argmin(np.abs(lv[:, 1] - t)))
        if abs(lv[i, 1] - t) > 0.2:
            continue
        for pair in r["map_crate_resid"].split(";"):
            res.append(tuple(map(float, pair.split())))
            at.append(err[i])
    res, at = np.array(res), np.array(at)

    def spread(a):
        d = a - a.mean(0)
        return float(np.sqrt((d ** 2).sum(1).mean()))

    print(f"\n{len(res)} crate residuals with a registered frame within 0.2 s")
    for k, ax in enumerate("xy"):
        print(f"  {ax}: correlation with the fix's error {np.corrcoef(res[:, k], at[:, k])[0, 1]:+.2f}, "
              f"slope {np.polyfit(at[:, k], res[:, k], 1)[0]:+.2f}")
    print(f"  spread about the mean {spread(res):.3f}; with the fix's error added back {spread(res + at):.3f}")
