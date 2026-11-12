# Rockwall Brawl: held-out classifier labels

This folder holds eleven `LabelGrid`s from `bluestacks-rockwall-brawl-center-east`, a BlueStacks
recording at 2560x1440 with the emulator HUD. They're used for scoring only.

| frame | 880 | 910 | 960 | 1000 | 1040 | 1070 | 1110 | 1190 | 1270 | 1300 | 1330 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| labelled cells | 104 | 169 | 189 | 128 | 170 | 111 | 110 | 72 | 123 | 98 | 119 | **1393** |

By class: FLOOR 850, WALL 191, BUSH 352, WATER 0, FENCE 0. Every wall cell is a rock cube, on 52
cube tiles. The set holds no cone (**The cones**, below).

## Why they're in their own folder

As with `../labels_starfruit_test/`, `../labels_truth_test/` and `../labels_doom_shroom_test/`:

- **Training can't see them.** `load_label_dir` only reads the top level of a folder.
- **`--hold-out` can't separate them.** It holds out by clip name, and the training labels come
  from the same clip.

## The split is by ground, not by time

The recording is one Shelly match, which the hero wins. The hero spawns at (37,29) and crosses most
of the map. It sees the north strip cleanly only once, walking west below it at frames 880-1330,
before the gas covers it. When the hero comes back, at frames 2640-2910, the gas covers the strip.
The split is drawn on the map:

- **This set labels only the held-out region R:** map rows 0-9, the whole width.
- **The 76 training frames in `../labels/` (`bluestacks-rockwall-brawl-center-east_f*.json`) label
  everywhere except R.** They leave R `?`.
- **No frame is in both sets**, and no map tile is labelled in both. By cell centre, training
  covers 1834 tiles and this set covers 369, with none shared.

R holds both north spawn pockets' cubes and the top bush border, with its clumps at (26-28, 7),
(32-34, 7) and (51-53, 9). `scripts/probes/map_label_projection/rockwall_brawl/textures.csv` names
each wall's texture, and MAP_ONBOARDING.md (Rockwall Brawl) says what each looks like in the game.

Both sets come from one map, with the same palette and floor art. A score here therefore measures
**unseen ground on a seen map**, not an unseen map.

### The cones

R also holds three of the map's 18 destructible cones, at (29-31, 6), but none of them labels here.
Before they break, they sit at the top left of every held-out view. There they're under the
brawlers-left text or the kill card, or past the view's edge, and the projection refuses those
cells. Frame 1110 already shows them broken. So this set scores no cone. The training frames hold
21 cone cells, on 7 tiles.

## How they were made

These labels were made by machine, not drawn by hand. Each one is the map label
(`brawl_deployment/data/maps/rockwall_brawl.csv`) projected through a registration of the frame
against the map image. Cells the rules can't vouch for are left `?`. From the user: the
destructible cones are walls, and the map has no water, only bushes and walls.

**Ground that breaks is left `?` from then on.** The map label is the ground as the match starts.
A cone, cube or bush that a shot or a super breaks shows floor from then on, and a projection would
still label it from the map. Sixteen cells broke in this clip: 8 cones, 4 cubes and 4 bushes.
`--broken C,R@F` names each one with a frame that shows it broken. Each labels from the map before
F and is left `?` from F on. Of the sixteen, only the three top cones lie in R. MAP_ONBOARDING.md
(Rockwall Brawl) lists all sixteen, and
`scripts/probes/map_label_projection/rockwall_brawl/broken_cells.py` found them.

**The scale was set from the blocks.** The odometry fit read (1.024, 1.007), which left the frame
edge 0.12 tile off. Both sets use `--scale 1.026 1.020`, where the block residuals pointed, and
that leaves the edge 0.02 tile off. `scripts/map_project_labels.py`'s docstring (**Scale.**) has
the check.

**The frames.** `scripts/probes/map_label_projection/rockwall_brawl/pick_frames.py` picked both
lists and prints both `--leave-row` lists.

- Held-out frames come only from the first pass over R, to frame 1340. Frame 2760 sees as many R
  tiles, but under the gas, and it labelled nothing.
- Neither set takes a frame within 10 of the clip's four short magenta tints.
- Neither takes one that shows the "SHOWDOWN!" banner (two brawlers left, frames 3319-3469), whose
  letters cover ground the labels would name.

Rebuild both sets from Git Bash. The recording is gitignored, in
`brawl_vision/data/training_videos/`. Both sets take the same broken cells:

```bash
BROKEN="29,6@1110 30,6@1110 31,6@1110 41,33@2100 41,25@2640 19,25@3010 19,33@3300 41,29@3510 26,35@3340 26,32@3420 25,32@3510 26,36@3630 23,34@3340 23,35@3340 29,33@3420 29,34@3420"
```

The held-out set:

```bash
LEAVE=$(for r in $(seq 10 59); do printf "%s:0-59 " $r; done)
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/map_project_labels.py rockwall_brawl bluestacks-rockwall-brawl-center-east --scale 1.026 1.020 --held-out 880 910 960 1000 1040 1070 1110 1190 1270 1300 1330 --held-out-dir tests/fixtures/vision/labels_rockwall_brawl_test --leave-row $LEAVE --leave-why "held-out split: only region R (map rows 0-9) labels in this set, and the training labels leave it '?'" --broken $BROKEN --date 2026-11-11
```

The training set:

```bash
LEAVE=$(for r in $(seq 0 9); do printf "%s:0-59 " $r; done)
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/map_project_labels.py rockwall_brawl bluestacks-rockwall-brawl-center-east --scale 1.026 1.020 --train 470 510 550 590 630 670 710 770 810 850 940 980 1020 1090 1130 1170 1210 1250 1350 1390 1430 1470 1510 1550 1590 1650 1700 1740 1780 1820 1860 1900 1940 1980 2020 2060 2100 2140 2180 2220 2260 2300 2340 2400 2440 2480 2520 2560 2600 2640 2680 2720 2760 2800 2840 2890 2930 2970 3010 3050 3090 3130 3170 3210 3250 3290 3470 3510 3550 3590 3630 3670 3710 3750 3810 3850 --leave-row $LEAVE --leave-why "held-out split: region R (map rows 0-9) is the held-out set's (labels_rockwall_brawl_test) and never labels here" --broken $BROKEN --date 2026-11-11
```

## Scoring

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/vision_score_map.py --labels tests/fixtures/vision/labels_rockwall_brawl_test --model brawl_vision/data/terrain.pt --device cpu
```

Add another `--model` argument for each extra model.

## Results (2026-11-11)

Scored with `vision_score_map.py`, as above. "Blocking" is wall, water or fence.

| classifier | blocking F1 (recall, precision) | 5-class |
|---|---|---|
| default (`terrain.pt`) | 0.041 (0.064, 0.031) | 0.556 |
| `terrain_dark_passage.pt` | 0.038 (0.058, 0.028) | 0.562 |
| `truth` | 0.032 (0.047, 0.025) | 0.571 |
| `doom_shroom` | 0.034 (0.052, 0.025) | 0.554 |
| **`rockwall_brawl`** | **0.987** (1.000, 0.974) | **0.996** |

- **The older classifiers read this map's bush and wall swapped.** Each reads every bush cell as
  wall (the default reads one as water), and 93-95% of the cube cells as bush. Their floor reads
  stay at 0.90-0.92.
  `scripts/probes/map_label_projection/rockwall_brawl/texture_score.py` gives the confusions.
- **`rockwall_brawl` reads every wall and bush cell it answers right:** all 191 wall cells and 348
  of the 352 bush cells. Its map answers 1385 of the 1393 cells (the scorer's `cov`). The scorer
  skips the 8 it leaves unanswered, 4 bush and 4 floor, as it skips every model's (the others
  answer 1365-1380). Its six misses are all floor:
  - five read as wall: (41,2) in all three frames that score it, (39,2) and (18,3);
  - one read as bush: (54,9).

  Each sits beside a block or a bush, or just north of one. In the game view a block's top covers
  about a tile north of its base, and the label reads the ground at the base. `cell_diff.py` (the
  same folder) lists every model's answer on each tile.
- **The other maps' held-out sets lose nothing**, in blocking F1 (recall, precision); 5-class.
  The `doom_shroom` column is that classifier's (v3), from MAP_ONBOARDING.md's Doom Shroom section,
  scored the same way:

  | held-out set | `doom_shroom` | **`rockwall_brawl`** |
  |---|---|---|
  | Doom Shroom west, 12 frames | 0.982 (0.971, 0.994); 0.991 | 0.997 (0.994, 1.000); 0.998 |
  | Doom Shroom east, 12 frames | 0.959 (0.995, 0.926); 0.980 | 0.971 (0.954, 0.988); 0.988 |
  | Truth, 10 frames | 0.987 (1.000, 0.975); 0.996 | 0.986 (0.990, 0.981); 0.995 |
  | Starfruit, 7 frames | 0.942 (0.995, 0.895); 0.982 | 0.952 (0.962, 0.942); 0.985 |

- **No cone is scored here** (**The cones**, above). MAP_ONBOARDING.md's Rockwall Brawl section
  counts them over the replay.
