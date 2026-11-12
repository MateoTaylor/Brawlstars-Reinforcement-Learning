# Truth probes, 2026-11-05

These probes produced the numbers in MAP_ONBOARDING.md's Truth section and in
`tests/fixtures/vision/labels_truth_test/README.md`. The labels themselves came from
`scripts/map_project_labels.py`, with no edits. These scripts picked the frames, scored the
candidates, and checked the localizer.

Run from the repo root on the CPU (`CUDA_VISIBLE_DEVICES=-1`), in Git Bash. They read
`runs/map_checks/truth/` (gitignored), and need:

- **`projection/bluestacks-truth-west-upper/reg.npz` and `reg_at_scale.npz`.** These come from
  the `map_project_labels.py` commands in the held-out README.
- **`coverage.json`.** `coverage.py` writes it.

**`textures.csv`** is a 60 x 60 grid aligned with `brawl_deployment/data/maps/truth.csv`. Each `#`
carries its wall texture (stone, crate, mailbox, barrel, teleporter), and every other cell is
empty. Only scoring reads it. The deploy never does.

| script | does | used for |
|---|---|---|
| `coverage.py` | Takes every 10th registered frame and writes its offset A, the map tiles it sees, and their classes and textures to `coverage.json` | everything below |
| `pick_heldout.py` | Ranks frames by how many textured tiles of R (map x 0–28, y 29–59) they show | the held-out set |
| `pick_extra.py` | Ranks frames by textured tiles outside R, mailbox first, away from the bad windows and the frames already chosen | candidate 2's 17 extra frames |
| `label_textures.py DIR...` | Counts cells per class and per wall texture in label folders, and the tiles they share | "every texture is in both sets" |
| `label_shift.py --labels DIR --model PT` | Scores one model's single-frame reads against each label shifted by a cell or two | that the labels sit at (0, 0) |
| `texture_score.py --labels DIR --textures CSV --model PT...` | Runs `vision_score_map.py`'s protocol and adds the full 5-class confusion and a breakdown by texture | the held-out tables |
| `replay_seen.py CLIP ...` | Runs `replay_clip.py` unchanged, saves the label check's seen map to `$SEEN_OUT`, and registers `$EXTRA_TERRAIN="name=path;..."` as terrain names | candidate replays without a config edit |
| `region_score.py NAME=NPZ...` | Scores seen maps against `truth.csv` inside R and outside it, by class and texture | the region-R table |
| `wall_misses.py NAME=NPZ...` | Lists every wall tile a seen map reads as passable, with texture and confidence | the edge-of-view finding |
| `probe_cells.py NAME=PT... -- X,Y...` | Gives a model's per-frame class at tiles, at their image-registered position, with the view row and column | sorting the disputed tiles |
| `loc_vs_image.py CSV...` | Compares the localizer's offset with image registration at each registered frame, one column per replay. The value is constant while the fix is right, and the same constant in every replay of the clip. Since 2026-11-05 it interpolates the decision rows to the frame's time. The nearest row it used to take inflated a walk's drift by up to 0.3 tile. | the drift, slips and banner; the overlay rule's before and after |
| `crops_disputed.py LOG OUT` | Makes contact sheets of the label check's disputed tiles: map image beside a clip frame | `runs/map_checks/truth/shipped/disputed_*.jpg` |
| `trace_deposits.py X,Y... -- CLIP ...` | Runs `replay_clip.py` unchanged and lists every vote the label check's map takes on the given tiles: clip time, class, weight, the view cell it came from, and the frame's re-aim f. `$OVERLAY=off` replays with the overlay rule off. | sorting the nine tiles disputed after the overlay rule |
| `crop_offset.py TRACE.txt` | Reads `trace_deposits.py`'s output and gives each vote's crop offset d: how far the crop sat off its tile, from the view cell, f and the registration. A vote in a bad window gets none. | the same nine: which neighbour each crop carried |
| `regress.sh TAG=TERRAIN...` | Replays the three Dark Passage clips and Starfruit with each model, 4 at a time | the regression table |

The bad windows are frames 0–520, 2390–2480, 2630–2840 and 3340 on: before the first gate, the
odometry y jump, the SHOWDOWN banner, and the end. The scripts that walk frames skip them. The y
jump and the banner were the odometry's. The overlay rule fixed both on 2026-11-05
(MAP_ONBOARDING.md, Truth). `coverage.json` predates the fix and still leaves them out.
`S = (1.022, 1.004)` is hard-coded where the registration needs it.

The two candidate weights are in `runs/terrain_truth/`: `candidate.pt` (shipped as
`brawl_vision/data/terrain_truth.pt`) and `candidate_plus.pt`. Their scores and training logs are
there too.
