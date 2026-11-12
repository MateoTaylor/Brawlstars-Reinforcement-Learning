# Doom Shroom: held-out classifier labels

This folder holds twelve `LabelGrid`s from `bluestacks-doom-shroom-south-center`, a BlueStacks
recording at 2560x1440 with the emulator HUD. They're used for scoring only.

| frame | 710 | 740 | 770 | 1470 | 1500 | 1550 | 1580 | 1630 | 1660 | 1710 | 1740 | 1770 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| labelled cells | 91 | 117 | 118 | 103 | 111 | 106 | 100 | 103 | 92 | 50 | 67 | 65 | **1123** |

By class: FLOOR 861, WALL 134, BUSH 89, WATER 39, FENCE 0. These are the rebuilt labels, with the
fences as walls (**How they were made**). Before the rebuild the set had 1153 cells, FLOOR 902 and
WALL 123.

## Why they're in their own folder

As with `../labels_starfruit_test/` and `../labels_truth_test/`:

- **Training can't see them.** `load_label_dir` only reads the top level of a folder.
- **`--hold-out` can't separate them.** It holds out by clip name, and the training labels come
  from the same clip.

## The split is by ground, not by time

The recording is one Shelly match. The hero spawns at (28,38) and stays in the west half. It walks
the map's west edge twice, at frames 700-770 and 1370-1770, so a cut in time would put the same
ground in both sets. The split is drawn on the map instead:

- **This set labels only the held-out region R:** map tiles x 0-13 without the bounce pad (7-8,
  15-16), plus the planter patch (21-23, 33-35).
- **The 40 training frames in `../labels/` (`bluestacks-doom-shroom-south-center_f*.json`) label
  everywhere except R.** They leave R `?`.
- **No frame is in both sets**, and no map tile is labelled in both. By cell centre, training
  covers 1221 tiles and this set covers 355, with none shared.

R was chosen so that every texture the match shows is in both sets, except the one bounce pad,
which trains. Cells / tiles by texture in this set:

| pink_cross | pink_plain | barrel | planter | mailbox | gray | fence | speed (floor) |
|---|---|---|---|---|---|---|---|
| 71 / 29 | 23 / 7 | 16 / 3 | 9 / 3 | 3 / 2 | 1 / 1 | 11 / 7 | 57 / 18 |

`scripts/probes/map_label_projection/doom_shroom/textures.csv` names the textures, and
MAP_ONBOARDING.md (Doom Shroom) says what each looks like in the game.

Both sets come from one map, with the same palette and floor art. A score here therefore measures
**unseen ground on a seen map**, not an unseen map.

## How they were made

These labels were made by machine, not drawn by hand. Each one is the map label
(`brawl_deployment/data/maps/doom_shroom.csv`) projected through a registration of the frame
against the map image. Cells the rules can't vouch for are left `?`. From the user: teleporters,
bounce pads, heal pads and speed tiles are floor, and gray walls, pink walls, barrels, planter
boxes, the orange barriers and the fences are walls.

**The fences became walls later on 2026-11-10,** and both sets were rebuilt then. Before that they
were floor, and two kinds of cell were left `?` on purpose (`--leave`):

- **The 11 orange barriers**, at (47-51,24), (51,25) and (51,29-33), whose class the user had not
  named yet. They're destructible in the game, and since the sim has no destructible walls, the
  user made them plain walls. They lie outside R, so this set could never label them, and the
  training frames never label one either: the two they see sit at the view's edge.
- **Three fence posts standing in grass**, at (20,48), (19,50) and (17,58), which might have been
  bush.

Neither `--leave` is in the rebuild. With fences as walls, 11 fence cells here went from floor to
wall, and 30 cells along the fence lines went from floor to `?`. A cell labels only when one class
holds 0.60 of it under every 0.12-tile shift, and a cell straddling a one-tile fence no longer
does. The training frames hold 171 fence cells as wall: 165 were floor, 3 lie on the posts in
grass, and 3 on the fence line along the pink wall at (27-29, 45), where a cell that straddled
floor and wall is now all wall.

**The scale was set from the blocks.** The odometry fit read (1.010, 1.010), which left the frame
edge 0.16 tile off. Both sets use `--scale 1.021 1.011`, where the block residuals pointed, and
that leaves the edge 0.01 tile off. `scripts/map_project_labels.py`'s docstring (**Scale.**) has
the check.

Rebuild both sets from Git Bash (the recording is gitignored, in
`brawl_vision/data/training_videos/`). The held-out set:

```bash
LEAVE=$(for r in $(seq 0 59); do case $r in 15|16) printf "%s:7-8 %s:14-59 " $r $r;; 33|34|35) printf "%s:14-20 %s:24-59 " $r $r;; *) printf "%s:14-59 " $r;; esac; done)
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/map_project_labels.py doom_shroom bluestacks-doom-shroom-south-center --scale 1.021 1.011 --held-out 710 740 770 1470 1500 1550 1580 1630 1660 1710 1740 1770 --held-out-dir tests/fixtures/vision/labels_doom_shroom_test --leave-row $LEAVE --leave-why "held-out split: only region R (map x 0-13 without the bounce pad 7-8,15-16, plus the planter patch 21-23,33-35) labels in this set, and the training labels leave it '?'" --date 2026-11-10
```

The training set:

```bash
LEAVE=$(for r in $(seq 0 59); do case $r in 15|16) printf "%s:0-6 %s:9-13 " $r $r;; 33|34|35) printf "%s:0-13 %s:21-23 " $r $r;; *) printf "%s:0-13 " $r;; esac; done)
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/map_project_labels.py doom_shroom bluestacks-doom-shroom-south-center --scale 1.021 1.011 --train 530 570 610 650 690 790 830 870 910 950 1000 1040 1080 1120 1160 1210 1250 1290 1330 1370 1410 1450 1520 1600 1680 1790 1830 1870 1910 1950 1990 2030 2070 2110 2150 2190 2230 2270 2310 2350 --leave-row $LEAVE --leave-why "held-out split: region R (map x 0-13 without the bounce pad 7-8,15-16, plus the planter patch 21-23,33-35) is the held-out set's (labels_doom_shroom_test) and never labels here" --date 2026-11-10
```

`scripts/probes/map_label_projection/doom_shroom/pick_frames.py` picked both frame lists and
prints both `--leave-row` lists.

## Scoring

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/vision_score_map.py --labels tests/fixtures/vision/labels_doom_shroom_test --model brawl_vision/data/terrain.pt --device cpu
```

Add another `--model` argument for each extra model.

## Results (2026-11-11)

Scored against the rebuilt labels, with the fences as walls. **`doom_shroom` v3 is shipped** as
`brawl_vision/data/terrain_doom_shroom.pt`. It was trained on every label, 217 frames: the 40
frames in `../labels/` from this recording, the second recording's 45, and every older label, with
the defaults (200 epochs, seed 0, CPU). v1 and v2 were trained while the fences were still floor,
and both were deleted when v3 shipped. v2 was scored here before its deletion. MAP_ONBOARDING.md
(Doom Shroom, **Classifier v2** and **Classifier v3**) has both.

| model | blocking F1 | recall | precision | edge | inner | 5-class | cells |
|---|---|---|---|---|---|---|---|
| `terrain.pt` (default) | 0.809 | 0.859 | 0.765 | 0.654 | 0.765 | 0.889 | 1067 |
| `terrain_dark_passage.pt` | 0.679 | 0.661 | 0.698 | 0.484 | 0.670 | 0.830 | 1059 |
| `terrain_truth.pt` | 0.878 | 0.878 | 0.878 | 0.803 | 0.846 | 0.931 | 1092 |
| `doom_shroom` v2 (deleted) | 0.967 | 0.936 | **1.000** | **0.979** | 0.958 | 0.986 | 1095 |
| **`doom_shroom` v3, shipped** | **0.982** | **0.971** | 0.994 | 0.972 | **0.985** | **0.991** | 1094 |

Recall per class, from `scripts/probes/map_label_projection/doom_shroom/texture_score.py`:

| model | floor | wall | bush | water | bush read as water | water read as bush |
|---|---|---|---|---|---|---|
| `terrain.pt` | 0.896 | 0.827 | 0.938 | 0.867 | 0 of 80 | 1 of 30 |
| `terrain_dark_passage.pt` | 0.850 | 0.773 | 0.976 | 0.250 | 0 of 83 | 27 of 36 |
| `terrain_truth.pt` | 0.935 | 0.842 | 1.000 | 1.000 | 0 of 83 | 0 of 39 |
| `doom_shroom` v2 | **0.995** | 0.917 | 1.000 | 1.000 | 0 of 83 | 0 of 39 |
| **`doom_shroom` v3, shipped** | 0.994 | **0.962** | 1.000 | 1.000 | 0 of 83 | 0 of 39 |

Cells by texture, read as their label class (cells scored in brackets):

| model | speed (floor) | fence | pink_cross | pink_plain | barrel | planter | mailbox | gray |
|---|---|---|---|---|---|---|---|---|
| `terrain.pt` | 0.095 (42) | 0.000 (11) | 1.000 (71) | 0.955 (22) | 0.875 (16) | 0.000 (9) | 1.000 (3) | 1.000 (1) |
| `terrain_dark_passage.pt` | 0.038 (53) | 0.545 (11) | 1.000 (70) | 0.955 (22) | 0.250 (16) | 0.000 (9) | 0.333 (3) | 0.000 (1) |
| `terrain_truth.pt` | 0.167 (54) | 0.182 (11) | 1.000 (71) | 1.000 (22) | 0.875 (16) | 0.000 (9) | 0.667 (3) | 1.000 (1) |
| `doom_shroom` v2 | 0.930 (57) | 0.000 (11) | 1.000 (71) | 1.000 (22) | **1.000** (16) | 1.000 (9) | 1.000 (3) | 1.000 (1) |
| **`doom_shroom` v3, shipped** | 0.930 (57) | **1.000** (11) | 1.000 (71) | 1.000 (22) | 0.667 (15) | 1.000 (9) | 1.000 (3) | 1.000 (1) |

- **v2 reads every fence cell as floor,** the class it was trained on. v3 reads all 11 as wall.
  Of the older classifiers, `dark_passage` reads the most as wall, 6 of 11.
- **v3's misses are barrels.** `cell_diff.py`, in the same probe folder, puts them on two tiles:
  (21,33), read as bush in 3 of the 4 frames that score it, and (7,37), read as floor in 2 of 5.
  The replays read barrels better than this set does (below).
- **The speed strips and the planters are what the older classifiers miss.** The default reads 4 of
  its 42 speed cells as floor, 15 as blocking and 23 as bush. Neither it nor `truth` reads a
  planter cell as a wall.
- **Mailbox and gray are too few here to rank anything,** at 3 cells and 1. The region check below
  counts them in tiles over a whole replay.

### Region R in the replay

`scripts/probes/map_label_projection/truth/replay_seen.py` replays the recording with each model and
keeps the map the session built. `region_score.py`, in the Doom Shroom probe folder, scores that map
against `doom_shroom.csv`, inside R and outside it. Inside R, no training label exists. The default
never fixes on this map, so it builds no map to score. v2 and v3 replayed against the new label, and
`truth` against the one with fences as floor. On the new label v2 loses its fix for 6.7 s.

| model | R: 5-class | R: blocking F1 | R: fences | R: speed read as floor | R: planters | R: barrels | elsewhere: blocking F1 |
|---|---|---|---|---|---|---|---|
| `terrain_truth.pt` | 0.930 | 0.814 | 1 of 18 | 6 of 22 | 0 of 3 | 6 of 6 | 0.702 |
| `doom_shroom` v2 | **0.955** | 0.877 | 0 of 18 | **25 of 27** | 3 of 3 | **7 of 7** | 0.852 |
| **`doom_shroom` v3, shipped** | 0.953 | **0.916** | **13 of 16** | 23 of 27 | 3 of 3 | 6 of 7 | **0.954** |

- **v3's three fence misses in R** are on the line at x 0-4, rows 46-48: (4,46), (1,47) and (2,47).
- **Outside R, v3 reads 42 of the 45 fence tiles it sees as wall.** Two of the three misses are the
  posts standing in grass, (20,48) and (19,50), read as bush.
- **v3's barrel miss in R is (3,41),** read as floor.

v1, the first classifier, missed four barrels in this replay, because its crops sat 0.25-0.5 tile
off the tiles. MAP_ONBOARDING.md, Doom Shroom, has the trace.

### The other maps

The same models on the other held-out sets, blocking F1:

| held-out set | `terrain.pt` | `terrain_truth.pt` | `doom_shroom` v2 | `doom_shroom` v3 |
|---|---|---|---|---|
| `../labels_truth_test/`, 10 frames | 0.908 | **0.990** | **0.990** | 0.987 |
| `../labels_starfruit_test/`, 7 frames | 0.946 | 0.945 | **0.952** | 0.942 |

v3's Starfruit score is the lowest of the four, in precision. No session runs `doom_shroom` on
Starfruit.

The Truth scores here include the 2026-11-05 odometry overlay rule. `../labels_truth_test/`'s own
README was scored before it, so it shows 0.907 and 0.982.
