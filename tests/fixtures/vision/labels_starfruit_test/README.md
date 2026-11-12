# Starfruit Supernova: held-out classifier labels

This folder holds seven `LabelGrid`s from `bluestacks-starfruit-supernova-west-upper`, a
BlueStacks recording at 2560x1440 with the emulator HUD. They're used for scoring only.

| frame | 1840 | 1900 | 1960 | 2020 | 2080 | 2140 | 2200 | total |
|---|---|---|---|---|---|---|---|---|
| labelled cells | 170 | 218 | 227 | 194 | 168 | 184 | 185 | **1346** |

By class: FLOOR 1093, WALL 178, BUSH 58, WATER 17, FENCE 0.

## Why they're in their own folder

- **Training can't see them.** `load_label_dir` only reads the top level of a folder, so
  `vision_train_terrain.py --labels tests/fixtures/vision/labels` never reads this one.
- **`--hold-out` can't separate them.** It holds out by clip name. Every Starfruit label, training
  and held-out, comes from the same clip, so the held-out labels can't share a folder with the
  training ones.

## The split

The ten training frames (460–1340, in `../labels/`) cover the north of the map and the
west-upper spawn, where the hero spends the first half of the match. These seven cover the
south half, after the hero crosses the meeple row at map row 26.

No world tile is labelled in both sets. By cell centre, training covers 955 tiles and this set
covers 502, with none shared. Frames 1380–1800 (the map centre) are labelled in neither, because
they would straddle the two.

Both sets come from one map, with the same palette and floor art. A score here therefore measures
**unseen ground on a seen map**, not an unseen map.

## How they were made

These labels were made by machine, not drawn by hand. Each one is the map label
(`brawl_deployment/data/maps/starfruit_supernova.csv`) projected through a registration of the
frame against the map image. Cells the rules can't vouch for are left `?`.

`scripts/probes/map_label_projection/README.md` has the method and the scripts. Each JSON's
`notes` field records its own registration.

## Scoring

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/vision_score_map.py --labels tests/fixtures/vision/labels_starfruit_test --model brawl_vision/data/terrain.pt --device cpu
```

Add another `--model` argument for each extra model. The scorer keys results by file name.

## Results (2026-11-04)

Three models were scored here; the CPU run took 1 min 40 s:

- **`terrain_dark_passage.pt`**, the model shipped as `brawl_vision/data/terrain.pt` until
  2026-11-04.
- **`control.pt`**, trained on `../labels/` without the ten Starfruit frames: 92 frames, 25,205
  cells.
- **`candidate.pt`**, trained on `../labels/` with them: 102 frames, 27,013 cells.

Both new models used `scripts/vision_train_terrain.py`'s recipe and are kept in
`runs/terrain_starfruit/` (not tracked). The candidate was shipped as
`brawl_vision/data/terrain.pt` on 2026-11-04. The old model is kept as
`brawl_vision/data/terrain_dark_passage.pt` (`MAP_ONBOARDING.md`, Starfruit Supernova).

| model | blocking F1 | recall | precision | edge F1 | inner F1 | 5-class | cells scored |
|---|---|---|---|---|---|---|---|
| `terrain_dark_passage.pt` (old) | 0.502 | 0.589 | 0.438 | 0.333 | 0.415 | 0.831 | 1272 |
| `control.pt` | 0.548 | 0.526 | 0.572 | 0.429 | 0.474 | 0.866 | 1274 |
| `candidate.pt` | **0.946** | 0.963 | 0.929 | 0.857 | 0.945 | **0.984** | 1325 |

Blocking means WALL, WATER and FENCE. "Edge" scores only walls that touch a free cell, and "inner"
only walls inside a run. "Cells scored" is out of 1346.

- **The Starfruit labels are what moved the score.** The control and the candidate share a recipe
  and differ only in those ten frames. The control is no better than the old model.
- **Each model was trained once.** The control and the old model differ by 0.046 F1, which
  gives an idea of the run-to-run spread. The candidate's gain is about ten times that.
- **Both sets were projected from one map label.** This score measures agreement with that label,
  so an error in the label would be learned and then scored as correct. The label check after the
  candidate's replay looked at the 37 cells where the classifier disagreed with the label. None was
  a label error (`MAP_ONBOARDING.md`, Starfruit section).
