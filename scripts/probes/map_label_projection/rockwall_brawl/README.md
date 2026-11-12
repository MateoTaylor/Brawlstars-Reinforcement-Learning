# Rockwall Brawl probes, 2026-11-11

These probes produced the numbers in MAP_ONBOARDING.md's Rockwall Brawl section and in
`tests/fixtures/vision/labels_rockwall_brawl_test/README.md`. The labels themselves came from
`scripts/map_project_labels.py`, with no edits: its `--broken` leaves the 16 cells that broke
mid-match `?` from the frame they break. These scripts picked the frames, found the broken cells,
scored the classifiers and checked the localizer. Most are copies of Doom Shroom's (`../doom_shroom/`), which
are mostly copies of Truth's (`../truth/`), changed where the table says.

Run from the repo root on the CPU (`CUDA_VISIBLE_DEVICES=-1`), in Git Bash. They read
`runs/map_checks/rockwall_brawl/` (gitignored), and need:

- **`projection/<CLIP>/reg.npz` and `reg_at_scale.npz`.** These come from the
  `map_project_labels.py` commands in the held-out README.
- **The clip's coverage file.** `coverage.py` writes it (see **Which clip** below).

**`textures.csv`** is a 60 x 60 grid aligned with `brawl_deployment/data/maps/rockwall_brawl.csv`.
Every wall cell carries its texture: `cube` (the 536 rock blocks) or `cone` (the 18 destructible
cones, walls by the user's rule). Every other cell is empty: the map has no floor specials, no water
and no fences. Only scoring reads it. The deploy never does. MAP_ONBOARDING.md's materials table
says what each looks like in the game.

| script | does | used for |
|---|---|---|
| `coverage.py` | Takes every registered frame and writes its offset A, how far that sits from the frame's own registration (dA), the NCC, and the map tiles it sees with their classes and textures, to `coverage.json` | everything below |
| `pick_frames.py` | Holds region R (`in_r`) and picks both frame sets: held-out frames by the R tiles they see, from the first pass over R only (to frame 1340), and training frames every 40 frames away from them. Neither set takes a frame near one of the four magenta tints or one showing the SHOWDOWN! banner. Prints both lists and both `--leave-row` lists. | the split |
| `broken_cells.py [--sheet PNG] [DIR...]` | Scores every map tile the labels call wall or bush, in every label frame: its floor share, plus a wall's bright share or a bush's colour. Prints each tile that leaves the standing range, with every view in frame order, and says whether it stays out to its last view (`stays from F`) or comes back. `--sheet` writes crops of each. Run it on a projection made without `--broken` to find what broke, and on the final labels to check that nothing stays. | the `--broken` list |
| `label_textures.py DIR...` | Counts cells per class and texture in label folders, and the tiles they share | "every texture is in both sets", and no tile in both |
| `texture_score.py --labels DIR --textures CSV --model PT...` | Runs `vision_score_map.py`'s protocol and adds the full 5-class confusion and a breakdown by class and texture | the held-out tables |
| `cell_diff.py --labels DIR --textures CSV --model PT...` | `texture_score.py`'s build, then every map tile any model gets wrong in any frame, with each model's answer in each frame that scores it | which tiles a model misses |
| `loc_vs_image.py CSV...` | The localizer's offset minus the image registration's at each registered frame. It is a constant while the fix is right. | that the fix is right where `map_localize_report.py` calls it a tile off |
| `regress.sh TAG=TERRAIN...` | Replays the three Dark Passage tuning clips, Starfruit, Truth and both Doom Shroom recordings with each model, seven at a time, through `../truth/replay_seen.py`. A name outside TERRAIN_MODELS comes from `$EXTRA_TERRAIN`. | the regression table |
| `region_score.py NAME=seen.npz... [--broken C,R@F...]` | Scores replay seen maps (`../truth/replay_seen.py`'s npz) against `rockwall_brawl.csv`, inside R and outside it: per class, per texture with the misses listed, and bush/wall confusion. Spawn tiles count as floor. R is `pick_frames.in_r`, so the two can't disagree. A seen map sums the whole replay, so a cell that broke reads floor against the map's wall or bush: `--broken`, the labels' list, leaves those cells out (only the cells; the frames are ignored). Give the maps first. | the region R table |
| `clip_paths.py` | Which recording the probes read (`$CLIP`), where its coverage goes, and its S from `reg_at_scale.npz`. Imported, never run. | every probe that reads `$CLIP` |

**Which clip.** `coverage.py`, `loc_vs_image.py` and `broken_cells.py` read `$CLIP`, the first clip
(`bluestacks-rockwall-brawl-center-east`) when it's unset. `clip_paths.py` says where each clip's
coverage goes: the first clip's stays at `coverage.json`, and any other's goes to
`coverage_<CLIP>.json`. S is the scale the clip's own registration used, from its
`reg_at_scale.npz`.

The classifier shipped unchanged as `brawl_vision/data/terrain_rockwall_brawl.pt`
(`--terrain rockwall_brawl`). Its training log, held-out scores and weights (`candidate.pt`) stay
in `runs/terrain_rockwall_brawl/`, with `texture_score.py`'s output as `texture_rockwall.txt` and
`cell_diff.py`'s as `cell_diff.txt`. The replays and reports are in
`runs/map_checks/rockwall_brawl/`:

- `m3/`, the default, `truth` and `doom_shroom`, none of which fixes;
- `m3_cand/`, the candidate loaded by path, with `map_localize_report.py`'s `report.txt`,
  `loc_vs_image.txt` and `region.txt`;
- `shipped/`, the same replay under the shipped name, identical but for the timing columns;
- `regression/*-cand.*`, with `report-cand.txt`.
