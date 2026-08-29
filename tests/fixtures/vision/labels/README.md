# Terrain labels — what to label, and how much

Tracked text, in the map-CSV legend you already use. One file per labelled frame:
`<clip>_f<frame>.json`.

## The tool

```
python scripts/vision_label.py <clip> <frame>          # e.g. zone_grows_from_east 700
python scripts/vision_label.py <clip> <frame> --review # look without editing
```

| key | does |
|---|---|
| `1` `2` `3` `4` `5` | hold FLOOR `.` / WALL `#` / BUSH `b` / WATER `~` / FENCE `f` |
| `0` | hold CLEAR — un-label a cell |
| click / drag | paint the held class |
| **`c`** | **assign the held class to every cell that looks like the one under the cursor** |
| `g` | toggle the grouping overlay, to see what `c` would select |
| `s` / `u` / `q` | save / undo / save and quit |

**`c` is the one that matters.** A frame has ~300 labellable cells and about six distinct
materials. Naming the six and fixing the stragglers is a different job from clicking 300 times.
The grouping is k-means on cell appearance — it knows nothing about terrain, so look at what it
selected (press `g`) before trusting it.

Re-running on a frame you have already labelled resumes it.

## How much to do

**Start with ~5 frames per map, and prefer breadth over depth.** Per the plan: 20 labelled cells
across 6 maps beats 100 across 2. What is being trained is reskin-robustness, and a map the model
has never seen is the only thing that measures it.

Frames worth picking:

- Ones where the camera is somewhere **new** — labelling two frames 5 apart is labelling the same
  cells twice.
- Ones containing **water** and **fence**, which are rare. FLOOR and WALL will over-supply
  themselves without any effort; the confusion matrix will be starved on the other three.
- **WALL next to FENCE**, if you can find it. Both block movement and only WALL blocks shots, so
  that pair is the single most tactically loaded call the classifier makes and is reported on its
  own.

## What the tool refuses

**Gassed cells are hatched and cannot be painted.** Gas tints the terrain underneath, so a gassed
cell is not a clean example of anything. Same reasoning as Phase I refusing to vote on them.

Cells mostly outside the viewport are likewise not labellable — the rectified patch is the
trapezoid the camera actually sees, padded to a rectangle, and the padding is not world.

Cells hidden by a **loot box or a brawler** should be left unlabelled. The tool cannot detect
those yet (that is the entity chunk), so it is on you to skip them; a box sitting on floor is not
an example of floor.

## Then

```
python scripts/vision_train_terrain.py --hold-out <a clip you labelled>
```

It reports per-class recall, WALL-vs-FENCE on its own, and refuses to print a cross-map number
when only one map is labelled — a same-map score would look like a result and measure
memorisation.

## Format

```json
{ "clip": "...", "frame": 700,
  "origin_tile": [-8, -8], "size_tiles": [30, 19], "pixels_per_tile": 48,
  "legend": {".": "FLOOR", "#": "WALL", "b": "BUSH", "~": "WATER", "f": "FENCE", "?": "UNLABELLED"},
  "grid": ["??????????...", "...."] }
```

One string per grid row, so a changed cell is one character in a diff. `?` is unlabelled — it does
not distinguish "skipped" from "not reached", because nothing downstream needs it to.

The geometry travels with the labels and is checked on load. If the camera is ever recalibrated
and the rectified window moves, every stored label would point at different world with nothing
looking wrong, so that mismatch is a hard error rather than a warning.
