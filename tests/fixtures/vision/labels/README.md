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

`<clip>` is a recording name without `.mp4`, looked up in `tests/fixtures/vision/` and then
`brawl_vision/data/training_videos/`; quote names with spaces. Recordings at 1080p (the BlueStacks
and emulator captures) are resized to the calibrated 2002x1126 viewport first, as the deployed
capture does. Where the same recording exists under two names (`day10_gameplay` is
`ScreenRecording_08-29-2026 19-39-53_1`), label it under one: `--hold-out` goes by name, and a
copy under the other name would leak the held-out map into training.

## The current round: `2maps_dataset`

`TERRAIN_RELABEL_2MAPS.md` at the repo root holds a picked, reviewed list of 100 frames from
`2maps_dataset.mp4` — one BlueStacks session, 13 back-to-back matches on 13 distinct maps, six
graveyard and seven desert. It is meant to be the whole training set on its own, with four matches
held out for test; the older labels here are not mixed in. That file also carries the map table,
the per-frame notes and the train/test split, because `--hold-out` cannot split a set whose labels
all share one clip name.

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

**Cells under the recording's buttons are refused**, by the HUD mask the file records (`"hud"`).
There are two (`brawl_vision/data/`): `hud_mask.json`, the phone's Layout A, and
`hud_mask_emulator.json`, BlueStacks + Nulls Brawl. A new file on an emulator recording
(`bluestacks-*`, `9-10_new*`, `edited_day14_broll`, `2maps_dataset`:
`labeling.EMULATOR_CLIP_PREFIXES`) takes the
emulator one and anything else the phone one; `--hud emulator` or `--hud phone` overrides that, and
on an existing file it moves the labels across, clearing any the new mask covers. A new emulator
recording needs its name added there: a test checks the rule against every recording's size.

**Some on-screen buttons still show through.** The phone mask misses the green gadget button in
every recording, and its emote bubble. The recordings with the smaller buttons further right
(`day12_recording*`, `ScreenRecording_08-31-2026 10-16-29_1`, all the `09-04` recordings) are a
third layout with no mask of their own, so their attack and super buttons are offered while world
nobody is covering is refused. On the emulator recordings a human's joystick can sit outside the
two spots the emulator mask covers. Leave cells under any button `?`, and check them after a `c`
fill, which will assign a button to whatever cluster it resembles.

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
  "origin_tile": [-8, -8], "size_tiles": [30, 19], "pixels_per_tile": 48, "hud": "emulator",
  "legend": {".": "FLOOR", "#": "WALL", "b": "BUSH", "~": "WATER", "f": "FENCE", "?": "UNLABELLED"},
  "grid": ["??????????...", "...."] }
```

One string per grid row, so a changed cell is one character in a diff. `?` is unlabelled — it does
not distinguish "skipped" from "not reached", because nothing downstream needs it to.

The geometry travels with the labels and is checked on load. If the camera is ever recalibrated
and the rectified window moves, every stored label would point at different world with nothing
looking wrong, so that mismatch is a hard error rather than a warning.

`hud` names the mask the cells were judged against; a file without it predates the key and was
drawn under the phone mask. The 19 emulator labels were moved to the emulator mask on 2026-09-15,
which cleared 634 labelled cells under its buttons and opened 35 cells per frame that the phone
mask had refused. Those 35 are unlabelled until someone reopens the frame. The same day, 43 cells in
eleven phone labels were cleared: they sat under the phone's chat-bubble rect, widened after they
were drawn.
