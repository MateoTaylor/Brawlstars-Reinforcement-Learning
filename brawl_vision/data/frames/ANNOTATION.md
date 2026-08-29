# Phase C calibration — what to annotate

The PNGs in this directory are gitignored (Supercell IP); this file and your CSVs are tracked.

## The frame

**`calib_showdown_f060.png`** — 2002×1126, Solo Showdown, already normalized (Section 3), so it
is exactly what the pipeline sees. Open it in any editor that shows cursor pixel coordinates.

Two helpers, both regenerable, neither needing annotation:

- `calib_GUIDE_ruler.png` — the same frame with a 200 px grid, to sanity-check your coordinates
- `calib_GUIDE_zoom.png` — a zoomed wall run showing which features to click and which to avoid

## Why wall-block corners, and why height matters

Brawl Stars' wall blocks are **exactly one tile each**, and their top faces form a crisp,
unambiguous grid — by far the most precise thing to click in the frame. The training-cave floor
was considered and rejected: it is completely featureless, with no tile seams at all.

But a homography maps **one plane**, and wall tops sit *above* the ground. Fitting only to tops
gives a valid homography for the wall-top plane, which is offset from the ground by the wall
height — roughly half a tile of apparent shift under this camera's tilt. Ground features
(floor, bush, water) would then land about half a tile off from walls, which would smear the map.

So: **Set A fixes the grid** (precise, from top corners) and **Set B measures the height offset**
(3 short vertical pairs) so the fit can be shifted onto the ground plane. Set B is small; skipping
it would bake in a half-tile error that is very hard to spot later.

---

## Set A — grid points → `calib_points.csv`

**At least 6, ideally 8–10.** Click points where **four block tops meet** (or an outer corner of
a block cluster). These are the crisp light-purple corners in the zoom guide.

**Spread them across the whole frame** — top-left, top-right, bottom-left, bottom-right, middle.
Points clustered in one corner give a fit that is accurate there and drifts badly elsewhere; this
matters more than adding extra points.

**Assigning tile coordinates.** Pick any one corner as your origin and call it `0,0`. Then count
block edges along the grid — **the map grid is rotated relative to the screen, so follow the block
edges, not screen horizontal/vertical.** One block edge = 1 tile.

- `+x` = along the block rows, toward screen-right
- `+y` = along the block columns, toward screen-down

(That matches the simulator: x = column, y = row, y increasing downward.) Coordinates may be
negative, and half-integers are fine if you use a mid-edge point. They do **not** need to relate
to any real map position — only to each other.

```csv
px,py,tx,ty
812,344,0,0
889,341,1,0
966,338,2,0
815,421,0,1
1290,690,6,4
430,880,-5,7
```

## Set B — height pairs → `calib_height.csv`

**Exactly 3 pairs (6 rows).** For each, pick a spot where you can see a block's **vertical front
edge**: click the top-front corner, then the point on the floor **directly below it** at the base
of that same edge. The base is the soft shadow line — it does not need to be precise to the pixel,
±5 px is fine, because three pairs are averaged and this only measures an offset.

Use pairs from **different parts of the frame** (e.g. one near the top, one middle, one lower).

```csv
pair,kind,px,py
1,top,812,344
1,base,818,398
2,top,1290,690
2,base,1297,747
3,top,430,880
3,base,436,935
```

## Set C — independent repeat → `calib_points_b.csv`

**At least 6**, same format as Set A. Choose **different corners** and re-pick your origin from
scratch rather than nudging Set A's numbers — the point is to check the *process* is repeatable,
and copying Set A would make the check vacuous. Phase C's acceptance is that solving from Set C
agrees with Set A within a small pixel tolerance.

---

## Practical notes

- **Precision needed: ±3–5 px.** At ~77 px per tile that is under 0.07 tiles, and the
  least-squares fit over 6+ points averages independent errors down further.
- **Avoid** the soft shadow where a wall meets the floor (for Set A), bush edges, anything under
  a player, the loot box, and the masked HUD regions (joystick bottom-left, attack button, chat
  bubble, "Brawlers left").
- **Do not resize or crop the PNG** before reading coordinates — the frame is already in the
  normalized viewport the homography is defined against.
- Comment rows starting with `#` are ignored, so feel free to annotate your own working notes.

Drop the CSVs in this directory. Once `calib_points.csv` and `calib_height.csv` are here I can
solve, validate, and report; `calib_points_b.csv` is what closes Phase C's acceptance.
