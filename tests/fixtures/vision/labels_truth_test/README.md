# Truth: held-out classifier labels

This folder holds ten `LabelGrid`s from `bluestacks-truth-west-upper`, a BlueStacks recording at
2560x1440 with the emulator HUD. They're used for scoring only.

| frame | 1830 | 1870 | 1910 | 1940 | 1980 | 2010 | 2090 | 2220 | 2260 | 2300 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| labelled cells | 117 | 174 | 180 | 208 | 199 | 189 | 194 | 239 | 193 | 217 | **1910** |

By class: FLOOR 1457, WALL 225, BUSH 139, WATER 89, FENCE 0.

## Why they're in their own folder

As with `../labels_starfruit_test/`:

- **Training can't see them.** `load_label_dir` only reads the top level of a folder.
- **`--hold-out` can't separate them.** It holds out by clip name, and the training labels come
  from the same clip.

## The split is by ground, not by time

The recording is one clip with two matches:

- **Match 1** spawns at (12,26) and stays in the north-west; the hero dies at 27 s.
- **Match 2** spawns at (47,33) and crosses the south into the south-west quadrant (frames
  1830-2300). It then goes north through the west and the centre.

A cut in time would put the same ground in both sets: match 2's last minute re-crosses the north-west
that match 1 covered. So the split is drawn on the map:

- **This set labels only the held-out region R:** map tiles x 0-28, y 29-59, the south-west
  quadrant.
- **The 30 training frames in `../labels/` label everywhere except R.** They leave R `?`.
- **No frame is in both sets**, and no map tile is labelled in both. By cell centre, training
  covers 1418 tiles and this set covers 639, with none shared.

R holds two of the map's eight crate clusters: the inner one beside the red teleporter pads, and
the outer one by the corner pool. Wall cells in this set, by texture (cells / tiles):

| stone | crate | mailbox | barrel | teleporter |
|---|---|---|---|---|
| 209 / 71 | 8 / 5 | 4 / 3 | 2 / 2 | 2 / 1 |

**The teleporter pads are mostly `?` here, by design of the tool.** The pads animate, so the
transient check (cells that change against the frames 8 either side) refuses them in most views.
Only 1 of R's 8 pad tiles carries a label. The region check below covers the rest.

Both sets come from one map, with the same palette and floor art. A score here therefore measures
**unseen ground on a seen map**, not an unseen map.

## How they were made

These labels were made by machine, not drawn by hand. Each one is the map label
(`brawl_deployment/data/maps/truth.csv`) projected through a registration of the frame against
the map image. Teleporter pads are `#` in the map label: they're walls for the bot, the user's
call. Cells the rules can't vouch for are left `?`.

**The scale was set, not fitted.** The odometry fit read (0.939, 0.935): the "SHOWDOWN!" banner at
89-93 s froze odometry while the camera moved about 7 tiles. The block residuals put the true
scale at (1.022, ~1.01), and both sets use `--scale 1.022 1.004`, Starfruit's fit from the same
emulator. `scripts/map_project_labels.py`'s docstring (**Scale.**) has the check. The freeze was
the odometry's, and the overlay rule fixed it on 2026-11-05 (MAP_ONBOARDING.md, Truth). These
labels keep the set scale.

Rebuild both sets from Git Bash (the recording is gitignored, in
`brawl_vision/data/training_videos/`):

```bash
LEAVE=$(for r in $(seq 0 28); do printf "%s:0-59 " $r; done; for r in $(seq 29 59); do printf "%s:29-59 " $r; done)
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/map_project_labels.py truth bluestacks-truth-west-upper --scale 1.022 1.004 --held-out 1830 1870 1910 1940 1980 2010 2090 2220 2260 2300 --leave-row $LEAVE --leave-why "held-out split: only the south-west, map tiles x 0-28 and y 29-59, labels in this set, and the training labels leave it '?'" --date 2026-11-05
```

```bash
LEAVE=$(for r in $(seq 29 59); do printf "%s:0-28 " $r; done)
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/map_project_labels.py truth bluestacks-truth-west-upper --scale 1.022 1.004 --train 530 570 610 650 690 730 1460 1510 1560 1600 1640 1680 1720 1760 1800 2490 2580 2850 2880 2910 2950 3000 3030 3060 3100 3140 3220 3250 3280 3320 --leave-row $LEAVE --leave-why "held-out split: the south-west, map tiles x 0-28 and y 29-59, is the held-out set's (labels_truth_test) and never labels here" --date 2026-11-05
```

Frame 770 was dropped from training. A damage flash left 13 of its cells standing.

## Scoring

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/vision_score_map.py --labels tests/fixtures/vision/labels_truth_test --model brawl_vision/data/terrain.pt --device cpu
```

Add another `--model` argument for each extra model.

## Results (2026-11-05)

Four models were scored here. **`truth` shipped** as `brawl_vision/data/terrain_truth.pt`: it was
trained on the 30 frames in `../labels/` plus every older label. Candidate 2 was trained on 17 more
Truth frames (47 in all). Both trainings used the defaults (200 epochs, seed 0, CPU).

| model | blocking F1 | recall | precision | edge | inner | 5-class | cells |
|---|---|---|---|---|---|---|---|
| `terrain.pt` (default) | 0.907 | 0.986 | 0.839 | 0.805 | 0.855 | 0.954 | 1872 |
| `terrain_dark_passage.pt` | 0.711 | 0.721 | 0.702 | 0.612 | 0.623 | 0.891 | 1859 |
| **`terrain_truth.pt`** | **0.982** | 0.984 | **0.980** | **0.968** | **0.975** | **0.992** | 1896 |
| candidate 2 (47 frames) | 0.981 | 0.990 | 0.972 | 0.961 | 0.971 | 0.989 | 1899 |

**Default's misses are floor read as blocking.** Its precision of 0.839 comes from 48 floor cells
read as wall and 26 read as bush. `truth` reads 4 and 1. Recall per class, from
`scripts/probes/map_label_projection/truth/texture_score.py`:

| model | floor | wall | bush | water | bush read as water | water read as bush |
|---|---|---|---|---|---|---|
| `terrain.pt` | 0.949 | 0.991 | 0.941 | 0.958 | 0 of 135 | 0 of 72 |
| `terrain_truth.pt` | 0.997 | 0.982 | 0.956 | 0.988 | 1 of 137 | 0 of 82 |
| candidate 2 | 0.994 | 0.987 | 0.941 | 1.000 | 0 of 135 | 0 of 87 |

**Bush and water barely meet**, in either direction or in any model. Their misses go to floor or
wall.

Wall cells by texture, read as wall by `terrain_truth.pt` (`terrain.pt` in brackets):

| stone | crate | mailbox | barrel | teleporter |
|---|---|---|---|---|
| 0.986 of 208 (0.995) | 1.000 of 7 (1.000) | 1.000 of 4 (1.000) | 0.500 of 2 (0.500) | 1.000 of 2 (1.000) |

The rare textures are too few here to rank anything. The region check below counts them in tiles
over a whole replay.

### Region R in the replay

`scripts/probes/map_label_projection/truth/replay_seen.py` replays the recording with each model
and keeps the map the session built. `region_score.py` scores that map against `truth.csv`, inside
R and outside it. Inside R, no training label exists. Outside, the training frames labelled the
ground, but under other views than most of the replay's.

| model | R: 5-class | R: blocking F1 | R: water | R: bush | elsewhere: blocking F1 |
|---|---|---|---|---|---|
| `terrain.pt` | 0.960 | 0.917 | 27 of 31 | 86 of 87 | 0.863 |
| **`terrain_truth.pt`** | **0.982** | **0.966** | **36 of 36** | 84 of 86 | 0.906 |
| candidate 2 | 0.972 | 0.949 | 37 of 37 | 82 of 86 | 0.932 |

Candidate 2 ties `truth` here and on Starfruit's held-out labels (blocking F1 0.946 against 0.945,
`../labels_starfruit_test/`). It failed the regression replays on Dark Passage, though
(MAP_ONBOARDING.md, Truth), and was not shipped.
