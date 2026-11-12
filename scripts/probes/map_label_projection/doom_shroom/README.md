# Doom Shroom probes, 2026-11-10

These probes produced the numbers in MAP_ONBOARDING.md's Doom Shroom section and in the two
held-out READMEs, `tests/fixtures/vision/labels_doom_shroom_test/README.md` and
`tests/fixtures/vision/labels_doom_shroom_east_test/README.md`. The labels themselves came from
`scripts/map_project_labels.py`, with no edits. These scripts picked the frames, scored the
classifiers and checked the localizer. Most are copies of Truth's (`../truth/`), changed where the
table says.

Run from the repo root on the CPU (`CUDA_VISIBLE_DEVICES=-1`), in Git Bash. They read
`runs/map_checks/doom_shroom/` (gitignored), and need:

- **`projection/<CLIP>/reg.npz` and `reg_at_scale.npz`.** These come from the
  `map_project_labels.py` commands in each clip's held-out README.
- **The clip's coverage file.** `coverage.py` writes it (see **Which clip** below).

**`textures.csv`** is a 60 x 60 grid aligned with `brawl_deployment/data/maps/doom_shroom.csv`.
Every wall cell carries its texture (pink_cross, pink_plain, gray, barrel, planter, mailbox,
orange, fence), and so does every floor special (speed, bounce, heal, teleporter). Every other cell
is empty. The fences were a floor special until later on 2026-11-10, when the user made them walls.
Only scoring reads it. The deploy never does. The names describe the map image;
MAP_ONBOARDING.md's materials table says what each looks like in the game.

| script | does | used for |
|---|---|---|
| `coverage.py` | Takes every registered frame and writes its offset A, how far that sits from the frame's own registration (dA), the NCC, and the map tiles it sees with their classes and textures, to `coverage.json` | everything below |
| `pick_frames.py` | Holds region R and picks both frame sets: held-out frames by the R tiles they see, training frames every 40 frames away from them. Prints both lists and both `--leave-row` lists. | the split |
| `label_textures.py DIR...` | Counts cells per class and texture in label folders, floor specials included, and the tiles they share. Truth's counts wall textures only. | "every texture is in both sets", and no tile in both |
| `texture_score.py --labels DIR --textures CSV --model PT...` | Runs `vision_score_map.py`'s protocol and adds the full 5-class confusion and a breakdown by class and texture. Truth's breaks down walls only. | the held-out tables |
| `cell_diff.py --labels DIR --textures CSV --model PT...` | New: `texture_score.py`'s build, then every map tile any model gets wrong in any frame, with each model's answer in each frame that scores it | which tiles v3 misses that v2 didn't (MAP_ONBOARDING.md, Doom Shroom, **Classifier v3**) |
| `loc_vs_image.py CSV...` | Truth's, with this clip's S and coverage: the localizer's offset minus the image registration's at each registered frame. It is a constant while the fix is right, and Truth's shipped replay gives the same constant. | that the fix is right where `map_localize_report.py` calls it a tile off |
| `regress.sh TAG=TERRAIN...` | Replays the three Dark Passage tuning clips, Starfruit and Truth with each model, five at a time, through `../truth/replay_seen.py`. A name outside TERRAIN_MODELS comes from `$EXTRA_TERRAIN`. | the regression table |
| `region_score.py [--r2] NAME=seen.npz...` | Scores replay seen maps (`../truth/replay_seen.py`'s npz) against `doom_shroom.csv`, inside R and outside it: per class, per texture with the misses listed (floor specials included), and bush/water confusion. Spawn tiles count as floor. `--r2` also scores R2 (map x 52-59), and the training ground becomes what is in neither. Truth's scores wall textures only. | the region R and R2 tables |
| `crop_offset.py TRACE.txt` | Truth's, with this clip's S and coverage: for each vote `../truth/trace_deposits.py` listed, how far its crop sat off the tile, by the image registration. | the barrel misses in R |
| `crop_split.py TRACE.txt CSV` | New: splits each vote's offset into the fix's error at the view's centre and the scale term. It also prints the zero of `loc_vs_image.py`'s output, (+13.93, +10.08). | what put the barrel crops off, and how far off the fix sits |
| `clip_paths.py` | New: which recording the probes read (`$CLIP`), where its coverage goes, and its S from `reg_at_scale.npz`. Imported, never run. | every probe that reads `$CLIP` |
| `pick_frames_east.py` | New: `pick_frames.py` for the second recording. Holds region R2 (map x 52-59) and picks both frame sets the same way, weighting the rare textures. Its training `--leave-row` list leaves R and R2. | the east split |
| `heal_gas.py [STEP]` | New: runs the loop's `detect_zone` on every STEP-th registered frame and maps each cell to its tile. Prints the heal pad's gas share per tile, what the 5% refusal and the 60% `GasMap` latch would take, and the first frame each textured tile latches. | the heal-pad check: why the pad never labels, and that the policy's gas map never latches it |
| `wander.py LOC_VS_IMAGE.txt COLUMN ZX ZY [CSV]` | New: the fix's error at the view's centre, `loc_vs_image.py`'s output minus `crop_split.py`'s zero, and with a replay CSV, how the crate residuals follow it. | that the second recording's crate spread is the fix's sub-tile wander |

**Which clip.** `coverage.py`, `loc_vs_image.py`, `crop_offset.py`, `crop_split.py` and
`heal_gas.py` read `$CLIP`, the first clip (`bluestacks-doom-shroom-south-center`) when it's unset. `clip_paths.py`
says where each clip's coverage goes: the first clip's stays at `coverage.json`, and any other's
goes to `coverage_<CLIP>.json`. S is the scale the clip's own registration used, from its
`reg_at_scale.npz`. The first clip's is `(1.021, 1.011)`, the block stretch's scale, which
`--scale` set (MAP_ONBOARDING.md, Doom Shroom). Before `$CLIP` existed this S was hard-coded, and
the three probes give the same output now as they did then. The match runs from the gate at frame 518
(17.27 s) to "Defeated" at about 80 s. The scripts that walk frames keep to 530–2350.

Each `doom_shroom` version shipped unchanged as `brawl_vision/data/terrain_doom_shroom.pt`
(`--terrain doom_shroom`): v1 first, then v2 the same day, then v3 on 2026-11-11, trained after the
user made the fences walls. v1's and v2's weights were deleted when v3 shipped. Each version's
scores and training log stay in `runs/terrain_doom_shroom/`, `runs/terrain_doom_shroom2/` and
`runs/terrain_doom_shroom3/`, and v3's weights are also `runs/terrain_doom_shroom3/candidate.pt`.
The replays and reports are in `runs/map_checks/doom_shroom/`:

- `m3_v2/`, v2 on the label with fences as floor;
- `m3_v2new/`, v2 on the new label;
- `m3_v3/`;
- `regression/*-cand2.*` and `regression/*-cand3.*`.
