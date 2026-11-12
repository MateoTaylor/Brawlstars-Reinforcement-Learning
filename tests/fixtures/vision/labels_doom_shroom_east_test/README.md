# Doom Shroom, east: held-out classifier labels

This folder holds twelve `LabelGrid`s from `bluestacks-doom-shroom-north-center`, the second Doom
Shroom recording: BlueStacks at 2560x1440, emulator HUD, one Shelly match from the north-centre
spawn (33,2).
They're used for scoring only. The first recording's held-out set, `../labels_doom_shroom_test/`,
covers the west strip. This one covers the east strip, which the first recording never reached.

| frame | 800 | 840 | 890 | 920 | 960 | 990 | 1020 | 1050 | 1090 | 1160 | 1250 | 1290 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| labelled cells | 65 | 91 | 93 | 90 | 81 | 89 | 84 | 82 | 83 | 67 | 76 | 64 | **965** |

By class: FLOOR 564, WALL 154, BUSH 190, WATER 57, FENCE 0.

## Why they're in their own folder

As with `../labels_starfruit_test/`, `../labels_truth_test/` and `../labels_doom_shroom_test/`:

- **Training can't see them.** `load_label_dir` only reads the top level of a folder.
- **`--hold-out` can't separate them.** It holds out by clip name, and the training labels come
  from the same clip.

## The split is by ground

- **This set labels only region R2, map x 52-59.**
- **Neither recording's training labels touch R2.**
  - The 45 frames from this recording in `../labels/`
    (`bluestacks-doom-shroom-north-center_f*.json`) leave R2 `?`. They also leave R `?`, the
    first held-out set's region: map x 0-13 without the bounce pad (7-8, 15-16), plus the planter
    patch (21-23, 33-35).
  - The first recording's 40 training frames never reach x 52.
- **No frame is in both sets, and no map tile is labelled in both.** Every training frame is at
  least 20 frames (0.67 s) from a held-out one. By cell centre, the 85 Doom Shroom training frames
  cover 1885 tiles and this set covers 315, with none shared. This set also shares none with
  `../labels_doom_shroom_test/`.

**Why R2 starts at x 52.** The textures the first recording never labelled, the orange barriers
and the heal pad, all sit at x 47-51. With R2 from x 50, this set would have taken 8 of the 11
barriers and half the heal pad, leaving training only 3 barriers. From x 52, they're all on
training ground.

- **9 barriers label in training**, 35 cells.
- **The other two, (51,29) and (51,30), border the heal pad.** The gas refusal below grows by one
  cell onto them.

So this set scores no orange tile. The replay check in MAP_ONBOARDING.md (Doom Shroom) reports how
the barriers read instead, on training ground.

Cells / tiles by texture in this set:

| pink_cross | pink_plain | barrel | planter |
|---|---|---|---|
| 75 / 41 | 56 / 21 | 6 / 2 | 17 / 3 |

Every texture here also trains. `scripts/probes/map_label_projection/doom_shroom/textures.csv`
names the textures, and MAP_ONBOARDING.md (Doom Shroom) says what each looks like in the game. A
score here measures **unseen ground on a seen map**, as the first set's does.

## How they were made

These labels were made by machine, as `../labels_doom_shroom_test/`'s were. Each is
`brawl_deployment/data/maps/doom_shroom.csv` projected through a registration of the frame against
the map image, and cells the rules can't vouch for are left `?`. The user's class rules are the
same: teleporters, bounce pads, heal pads and speed tiles are floor, and gray walls, pink walls,
barrels and planter boxes are walls. **The orange barriers are walls too** (user, 2026-11-10).
They're destructible in the game, but the sim has no destructible walls. The map label already had
them as `#`, so these labels needed no `--leave` for them.

**So are the fences** (user, later on 2026-11-10). They were floor when this set was first built,
and the three fence posts standing in grass, at (20,48), (19,50) and (17,58), were left `?`. R2
holds no fence tile, so the rebuild with fences as walls changed no cell here, only each label's
note.

**The heal pad (49-50, 29-30) never labels, in either set.** The projection refuses any cell
that's at least 5% gas-coloured (`label_projection.GAS_FRACTION`), and the pad's green cross fills
0.2-0.4 of each of its tiles. The live loop refuses the same cells, at the same 5%, before the
classifier's reading of them reaches the occupancy map or the localizer. A label there would train
an answer the loop discards. What the policy's gas map latches is a different test (60% of a cell,
`zone.min_cell_fraction`), and the pad never comes close: its highest share, over every tenth frame
of this recording, is 0.41. MAP_ONBOARDING.md (Doom Shroom) has the check.

**Scale.** The block stretch for this recording reads S = (1.021, 1.019). The x scale matches
the first recording's. The y scale is 0.008 above it, so both sets of this recording use
`--scale 1.021 1.019`.

**Frames.** `scripts/probes/map_label_projection/doom_shroom/pick_frames_east.py` (its defaults:
R2 from x 52, frames 570-2700) picked the first lists. Three of its held-out frames, 1880, 1910 and
1940, came after the gas reached R2 and kept 2-3 cells each. Frames 960, 990 and 1020 replaced
them, chosen for the most R2 cells among frames 780-1380. The two training frames that then sat
within 12 frames of a held-out one, 980 and 1020, were dropped. So the lists below are the
script's, with those changes.

Rebuild both sets from Git Bash. The recording is gitignored, in
`brawl_vision/data/training_videos/`. The held-out set:

```bash
LEAVE=$(for r in $(seq 0 59); do printf "%s:0-51 " $r; done)
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/map_project_labels.py doom_shroom bluestacks-doom-shroom-north-center --scale 1.021 1.019 --held-out 800 840 890 920 960 990 1020 1050 1090 1160 1250 1290 --held-out-dir tests/fixtures/vision/labels_doom_shroom_east_test --leave-row $LEAVE --leave-why "held-out split: only region R2 (map x 52-59) labels in this set, and the training labels from both Doom Shroom clips leave it '?'" --date 2026-11-10
```

The training set:

```bash
LEAVE=$(for r in $(seq 0 59); do case $r in 15|16) printf "%s:0-6 %s:9-13 %s:52-59 " $r $r $r;; 33|34|35) printf "%s:0-13 %s:21-23 %s:52-59 " $r $r $r;; *) printf "%s:0-13 %s:52-59 " $r $r;; esac; done)
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/map_project_labels.py doom_shroom bluestacks-doom-shroom-north-center --scale 1.021 1.019 --train 590 630 670 710 750 860 940 1070 1110 1180 1220 1270 1310 1350 1390 1430 1470 1530 1570 1610 1650 1690 1740 1780 1820 1860 1960 2000 2040 2080 2120 2160 2200 2240 2280 2320 2360 2400 2440 2480 2520 2560 2600 2640 2680 --leave-row $LEAVE --leave-why "held-out split: region R (map x 0-13 without the bounce pad 7-8,15-16, plus the planter patch 21-23,33-35) is labels_doom_shroom_test's and region R2 (map x 52-59) is labels_doom_shroom_east_test's, so neither labels here" --date 2026-11-10
```

## Scoring

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/vision_score_map.py --labels tests/fixtures/vision/labels_doom_shroom_east_test --model brawl_vision/data/terrain_doom_shroom.pt --device cpu
```

Add another `--model` argument for each extra model.

Scored 2026-11-10, v3 on 2026-11-11 (blocking = wall, water or fence; each model's map covers
slightly different cells). The rebuild with fences as walls changed no cell here, so the older
rows stand:

| model | blocking F1 | recall | precision | 5-class | cells scored | water read as bush |
|---|---|---|---|---|---|---|
| default (`terrain.pt`) | 0.870 | 0.872 | 0.867 | 0.941 | 913 | 13 of 42 |
| `dark_passage` | 0.789 | 0.756 | 0.826 | 0.909 | 916 | 31 of 42 |
| `truth` | 0.904 | 0.891 | 0.916 | 0.955 | 927 | 13 of 50 |
| `doom_shroom` v1 (deleted) | 0.968 | 0.951 | 0.985 | 0.984 | 952 | 7 of 56 |
| `doom_shroom` v2 (deleted) | **0.971** | 0.954 | **0.989** | **0.987** | 944 | 8 of 52 |
| **`doom_shroom` v3, shipped** | 0.959 | **0.995** | 0.926 | 0.980 | 938 | **1 of 49** |

- **v1 trained on no frame of this recording.** v2 and v3 trained on its 45 training frames, none
  in R2.
- **v3's lower precision is 15 floor cells read as wall** (v2: 1).
  `scripts/probes/map_label_projection/doom_shroom/cell_diff.py` places them:
  - 11 sit on the tile above or beside a tall sprite: the pink walls at (53-54,18) and (53,45), the
    barrels at (52-53,33) and the orange barrier at (51,24).
  - 3 sit above the Power Cube Box at (58,21) in frame 990. No label holds a box.
  - (53,11) is the last. v2 misses it too.
- **Barrels:** v2 reads 6 of 6 cells as wall, v3 5 of 5, v1 2 of 5.
- **Planters:** the older classifiers read 0-56% of them as wall, every `doom_shroom` version all.
- **Water is the weak class for every model but v3,** read as bush. The other way round is rare:
  one bush cell reads as water, in `truth`'s map.

`scripts/probes/map_label_projection/doom_shroom/texture_score.py` gives the per-texture tables
(`runs/terrain_doom_shroom3/texture_east.txt` for v3, `runs/terrain_doom_shroom2/texture_east*.txt`
for v2).
